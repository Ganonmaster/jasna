# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Quantize RF-DETR to INT8 with NNCF for OpenVINO (Intel Arc) — step 2 of the
INT8 detection effort.

Consumes the calibration set produced by `python -m jasna.calibration_frames`
(PNG frames + manifest.json) and writes an OpenVINO IR next to the source
model, named so the existing plumbing picks it up as a detection-model choice:

    python -m jasna.quantize_rfdetr --calibration calibration_frames \\
        [--model model_weights/rfdetr-v5.onnx]
    # -> model_weights/rfdetr-v5-int8.xml (+ .bin)
    # then: jasna ... --detection-model rfdetr-v5-int8 --device xpu:0

Details that matter:
- Calibration batches replicate RfDetrMosaicDetectionModel._preprocess exactly
  (/255, bilinear 768x768 align_corners=False, ImageNet mean/std) — quantizer
  ranges are only valid for the distribution the model sees at inference.
- The shipped rfdetr-v5.onnx has batch=4 baked into an internal Expand node,
  so calibration samples are batches of 4 frames, like production.
- model_type=TRANSFORMER: DETR attention is quantization-sensitive; this makes
  NNCF keep the known-fragile patterns (softmax/layernorm chains) accurate.
  If the detection-quality A/B (jasna.detection_ab) still shows misses, rerun
  with --ignored-scope-json to keep named layers in fp precision.

NNCF is a one-time build dependency, not a runtime one: `pip install nncf`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RESOLUTION = 768
BATCH = 4
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_frames(frames_uint8_hwc: list[np.ndarray]) -> np.ndarray:
    """uint8 (H, W, C) RGB frames -> float32 (B, 3, 768, 768), matching
    RfDetrMosaicDetectionModel._preprocess bit-for-bit on cpu/fp32. Each frame
    is resized ONCE from its native size (frames in a batch may come from
    videos of different resolutions), exactly like production batches."""
    resized = []
    for frame in frames_uint8_hwc:
        x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
        resized.append(
            F.interpolate(x, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)
        )
    x = torch.cat(resized)
    mean = x.new_tensor(_IMAGENET_MEAN)[:, None, None]
    std = x.new_tensor(_IMAGENET_STD)[:, None, None]
    return ((x - mean) / std).numpy()


def load_manifest_batches(calibration_dir: Path, batch: int = BATCH) -> list[list[Path]]:
    """Group the extractor's frames into inference-shaped batches. Frames in a
    batch may come from different videos/resolutions — each is resized to
    768x768 independently, exactly as production batches mix crops."""
    manifest = json.loads((calibration_dir / "manifest.json").read_text(encoding="utf-8"))
    files = [calibration_dir / entry["file"] for entry in manifest["frames"]]
    missing = [f for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frames from manifest.json are missing, e.g. {missing[0]}"
        )
    if len(files) < batch:
        raise ValueError(f"Need at least {batch} calibration frames, found {len(files)}")
    groups = [files[i : i + batch] for i in range(0, len(files) - batch + 1, batch)]
    return groups


def _load_batch(paths: list[Path]) -> np.ndarray:
    from PIL import Image

    frames = []
    for p in paths:
        with Image.open(p) as img:
            frames.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
    return preprocess_frames(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantize RF-DETR to INT8 (OpenVINO IR) using NNCF."
    )
    parser.add_argument("--calibration", required=True,
                        help="Output directory of jasna.calibration_frames")
    parser.add_argument("--model", default="model_weights/rfdetr-v5.onnx")
    parser.add_argument("--out", default=None,
                        help="Output IR path (default: <model dir>/<stem>-int8.xml)")
    parser.add_argument("--subset-size", type=int, default=0,
                        help="Calibration batches to use (0 = all)")
    parser.add_argument("--ignored-scope-json", default=None,
                        help="JSON file with an NNCF IgnoredScope dict (e.g. "
                             '{"names": [...]} or {"patterns": [...]}) to keep '
                             "listed layers in fp precision")
    parser.add_argument("--model-type", choices=["transformer", "plain"],
                        default="transformer",
                        help="'transformer' protects DETR's quantization-"
                             "sensitive attention (best quality). 'plain' "
                             "quantizes maximally aggressively — use it to "
                             "measure the INT8 speed CEILING, ignoring quality")
    args = parser.parse_args(argv)

    try:
        import nncf
        import openvino as ov
    except ImportError as exc:
        print(f"Missing dependency ({exc.name}). Quantization needs:\n"
              f"    pip install nncf\n"
              f"(one-time build tool; not required at jasna runtime)")
        return 1

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        print(f"Model not found: {model_path}")
        return 1
    out_path = (
        Path(args.out)
        if args.out
        else model_path.parent / f"{model_path.stem}-int8.xml"
    )
    calibration_dir = Path(args.calibration)
    batches = load_manifest_batches(calibration_dir)
    if args.subset_size:
        batches = batches[: args.subset_size]
    print(f"Calibration: {len(batches)} batches of {BATCH} frames from {calibration_dir}")

    core = ov.Core()
    model = core.read_model(str(model_path))
    model.reshape({model.inputs[0].any_name: ov.PartialShape([BATCH, 3, RESOLUTION, RESOLUTION])})
    input_name = model.inputs[0].any_name

    ignored_scope = None
    if args.ignored_scope_json:
        scope_kwargs = json.loads(Path(args.ignored_scope_json).read_text(encoding="utf-8"))
        ignored_scope = nncf.IgnoredScope(**scope_kwargs)
        print(f"Ignored scope: {scope_kwargs}")

    dataset = nncf.Dataset(batches, lambda paths: {input_name: _load_batch(paths)})
    model_type = (
        nncf.ModelType.TRANSFORMER if args.model_type == "transformer" else None
    )
    print(f"Quantizing (model_type={args.model_type}; this takes a few minutes)...")
    start = time.perf_counter()
    quantized = nncf.quantize(
        model,
        dataset,
        model_type=model_type,
        subset_size=len(batches),
        ignored_scope=ignored_scope,
    )
    elapsed = time.perf_counter() - start
    ov.save_model(quantized, str(out_path))
    bin_size = out_path.with_suffix(".bin").stat().st_size / 1e6
    print(f"Quantized in {elapsed:.0f}s -> {out_path} ({bin_size:.0f} MB weights)")
    print("\nNext steps:")
    name = out_path.stem
    print(f"  speed:   jasna --benchmark --benchmark-filter rfdetr "
          f"--detection-model {name} --device xpu:0")
    print(f"  quality: python -m jasna.detection_ab --frames <eval set> "
          f"--model-b {name} --device xpu:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
