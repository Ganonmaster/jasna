# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Detection-quality A/B: does model B find the mosaics model A finds?

The quality gate for INT8 RF-DETR (step 3 of the INT8 effort) — a detection B
misses is a mosaic left visible in the output video. Compares two detection
models frame-by-frame on an evaluation set produced by
`python -m jasna.calibration_frames` (run it with a DIFFERENT --seed than the
calibration set, so the quantized model is not evaluated on the frames that
tuned it):

    python -m jasna.calibration_frames --library DIR --out eval_frames \\
        --count 400 --seed 999 --device xpu:0
    python -m jasna.detection_ab --frames eval_frames \\
        --model-a rfdetr-v5 --model-b rfdetr-v5-int8 --device xpu:0

Reports, per stratum (the manifest's YOLO-based high-conf / borderline /
no-mosaic labels — borderline detections are the ones INT8 rounding is most
likely to flip):
- frame agreement: frames where A detects >=1 mosaic but B detects none (the
  worst failure: an entire mosaic missed);
- detection recall: A-detections matched by B at IoU >= 0.5, and vice versa
  (extras in B = new false positives);
- mean box IoU and mean mask IoU of matched pairs (localization drift).

Writes ab_report.json next to the eval frames for later inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BATCH = 4
MATCH_IOU = 0.5


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], a[:, 2, None], a[:, 3, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(iou: np.ndarray, threshold: float = MATCH_IOU) -> list[tuple[int, int, float]]:
    """Highest-IoU-first one-to-one matching; returns (idx_a, idx_b, iou)."""
    matches: list[tuple[int, int, float]] = []
    if iou.size == 0:
        return matches
    iou = iou.copy()
    while True:
        idx = np.unravel_index(np.argmax(iou), iou.shape)
        best = iou[idx]
        if best < threshold:
            return matches
        matches.append((int(idx[0]), int(idx[1]), float(best)))
        iou[idx[0], :] = -1.0
        iou[:, idx[1]] = -1.0


def _resample_bool_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    import torch.nn.functional as F

    return (
        F.interpolate(
            mask[None, None].float(), size=shape,
            mode="bilinear", align_corners=False,
        )[0, 0]
        > 0.5
    )


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    if mask_a.shape != mask_b.shape:
        # Model families use different mask grids (RF-DETR: a square head
        # resolution over the aspect-stretched frame; YOLO: aspect-scaled to a
        # max side). Both grids cover the full frame extent, so resampling one
        # onto the other's grid aligns them in frame space — upsample the
        # coarser mask for fidelity.
        if mask_a.numel() < mask_b.numel():
            mask_a = _resample_bool_mask(mask_a, tuple(mask_b.shape))
        else:
            mask_b = _resample_bool_mask(mask_b, tuple(mask_a.shape))
    inter = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return inter / union if union else 1.0


class StratumStats:
    def __init__(self) -> None:
        self.frames = 0
        self.frames_a_detected = 0
        self.frames_b_blind = 0  # A saw a mosaic, B saw nothing at all
        self.dets_a = 0
        self.dets_b = 0
        self.matched = 0
        self.box_ious: list[float] = []
        self.mask_ious: list[float] = []

    def as_dict(self) -> dict:
        missed = self.dets_a - self.matched
        extra = self.dets_b - self.matched
        return {
            "frames": self.frames,
            "frames_a_detected": self.frames_a_detected,
            "frames_b_blind": self.frames_b_blind,
            "detections_a": self.dets_a,
            "detections_b": self.dets_b,
            "matched": self.matched,
            "missed_by_b": missed,
            "miss_rate": missed / self.dets_a if self.dets_a else 0.0,
            "extra_in_b": extra,
            "mean_box_iou": float(np.mean(self.box_ious)) if self.box_ious else None,
            "mean_mask_iou": float(np.mean(self.mask_ious)) if self.mask_ious else None,
        }


def compare_frame(
    stats: StratumStats,
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
) -> None:
    stats.frames += 1
    stats.dets_a += len(boxes_a)
    stats.dets_b += len(boxes_b)
    if len(boxes_a):
        stats.frames_a_detected += 1
        if not len(boxes_b):
            stats.frames_b_blind += 1
    matches = greedy_match(box_iou_matrix(boxes_a, boxes_b))
    stats.matched += len(matches)
    for ia, ib, iou in matches:
        stats.box_ious.append(iou)
        stats.mask_ious.append(mask_iou(masks_a[ia], masks_b[ib]))


def _load_eval_set(frames_dir: Path) -> list[dict]:
    manifest = json.loads((frames_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["frames"]
    for e in entries:
        path = frames_dir / e["file"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest frame missing: {path}")
    return entries


def _batches_by_resolution(frames_dir: Path, entries: list[dict]):
    """Yield (uint8 BCHW tensor, entries) batches of BATCH frames; frames are
    grouped by resolution so they stack, and the final partial group is padded
    by repeating its last frame (padding results are discarded)."""
    from PIL import Image

    groups: dict[tuple[int, int], list[tuple[dict, np.ndarray]]] = defaultdict(list)
    for e in entries:
        with Image.open(frames_dir / e["file"]) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        groups[arr.shape[:2]].append((e, arr))
        # Flush any full batches of this resolution to bound memory.
        bucket = groups[arr.shape[:2]]
        while len(bucket) >= BATCH:
            chunk, groups[arr.shape[:2]] = bucket[:BATCH], bucket[BATCH:]
            bucket = groups[arr.shape[:2]]
            yield _stack_chunk(chunk), [e for e, _ in chunk]
    for bucket in groups.values():
        if bucket:
            chunk = bucket + [bucket[-1]] * (BATCH - len(bucket))
            yield _stack_chunk(chunk), [e for e, _ in bucket]


def _stack_chunk(chunk) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([arr for _, arr in chunk])
    ).permute(0, 3, 1, 2).contiguous()


def run_ab(args: argparse.Namespace) -> int:
    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        require_detection_model_weights,
    )

    frames_dir = Path(args.frames)
    entries = _load_eval_set(frames_dir)
    device = torch.device(args.device)
    print(f"A/B on {len(entries)} frames from {frames_dir} ({device})")

    models = {}
    for label, name in (("A", args.model_a), ("B", args.model_b)):
        coerced = coerce_detection_model_name(name)
        models[label] = build_detection_model(
            coerced,
            require_detection_model_weights(coerced),
            batch_size=BATCH,
            device=device,
            score_threshold=float(args.score_threshold),
            fp16=True,
        )
        print(f"  model {label}: {coerced}")

    per_stratum: dict[str, StratumStats] = defaultdict(StratumStats)
    overall = StratumStats()
    done = 0
    with torch.inference_mode():
        for batch, batch_entries in _batches_by_resolution(frames_dir, entries):
            target_hw = (int(batch.shape[2]), int(batch.shape[3]))
            det_a = models["A"](batch.clone(), target_hw=target_hw)
            det_b = models["B"](batch.clone(), target_hw=target_hw)
            for i, entry in enumerate(batch_entries):
                stratum = entry.get("stratum") or "unlabeled"
                for stats in (per_stratum[stratum], overall):
                    compare_frame(
                        stats,
                        det_a.boxes_xyxy[i], det_b.boxes_xyxy[i],
                        det_a.masks[i], det_b.masks[i],
                    )
            done += len(batch_entries)
            if done % 100 < BATCH:
                print(f"  [{done}/{len(entries)}]")
    for m in models.values():
        m.close()

    report = {
        "config": {k: v for k, v in vars(args).items()},
        "overall": overall.as_dict(),
        "per_stratum": {k: v.as_dict() for k, v in sorted(per_stratum.items())},
    }
    out_file = frames_dir / "ab_report.json"
    out_file.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{'stratum':12s} {'frames':>6s} {'A-dets':>7s} {'missed':>7s} "
          f"{'miss%':>6s} {'blind':>6s} {'extra':>6s} {'boxIoU':>7s} {'maskIoU':>8s}")
    rows = list(sorted(per_stratum.items())) + [("OVERALL", overall)]
    for name, stats in rows:
        d = stats.as_dict()
        print(f"{name:12s} {d['frames']:6d} {d['detections_a']:7d} "
              f"{d['missed_by_b']:7d} {d['miss_rate'] * 100:5.1f}% "
              f"{d['frames_b_blind']:6d} {d['extra_in_b']:6d} "
              f"{d['mean_box_iou'] or 0:7.3f} {d['mean_mask_iou'] or 0:8.3f}")

    d = overall.as_dict()
    if d["detections_a"] == 0:
        # Every extractor eval set contains high-conf-stratum frames, so a
        # baseline that detects NOTHING is always pathological (wrong weights,
        # broken engine, absurd threshold) — a "0.0% missed" verdict here would
        # green-light the INT8 model on zero evidence.
        print(f"\nFAILED: baseline model A produced no detections on any of the "
              f"{d['frames']} frames — the A/B is meaningless. Check --model-a, "
              f"its weights, and --score-threshold. Report: {out_file}")
        return 1
    print(f"\nVerdict inputs: B misses {d['miss_rate'] * 100:.1f}% of A's detections "
          f"({d['frames_b_blind']} frames where B is fully blind to a mosaic A sees). "
          f"Report: {out_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two detection models frame-by-frame on an eval set."
    )
    parser.add_argument("--frames", required=True,
                        help="Eval set dir from jasna.calibration_frames "
                             "(use a different --seed than the calibration set)")
    parser.add_argument("--model-a", default="rfdetr-v5")
    parser.add_argument("--model-b", default="rfdetr-v5-int8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)
    return run_ab(args)


if __name__ == "__main__":
    sys.exit(main())
