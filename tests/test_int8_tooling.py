"""Tests for the INT8 quantization tooling (step 2): the quantize script's
preprocessing parity with production, calibration batching, the A/B harness's
matching math, and the rfdetr-v5-int8 registry/guard wiring."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from jasna.detection_ab import (
    StratumStats,
    box_iou_matrix,
    compare_frame,
    greedy_match,
    mask_iou,
)
from jasna.mosaic.detection_registry import (
    detection_model_spec,
    detection_model_weights_path,
    is_rfdetr_model,
)
from jasna.quantize_rfdetr import (
    BATCH,
    RESOLUTION,
    load_manifest_batches,
    preprocess_frames,
)


# ---------------------------------------------------------------- quantize ---

def test_preprocess_matches_production_rfdetr_preprocess():
    """The calibration transform must equal RfDetrMosaicDetectionModel's
    _preprocess bit-for-bit — quantizer ranges are only valid for the exact
    inference-time distribution."""
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

    ref = object.__new__(RfDetrMosaicDetectionModel)
    ref.device = torch.device("cpu")
    ref.input_dtype = torch.float32
    ref.resolution = RESOLUTION

    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (270, 480, 3), dtype=np.uint8) for _ in range(BATCH)]
    ours = preprocess_frames(frames)

    frames_bchw = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    theirs = ref._preprocess(frames_bchw).numpy()

    assert ours.shape == (BATCH, 3, RESOLUTION, RESOLUTION)
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-5)


def test_preprocess_handles_mixed_resolutions():
    frames = [
        np.zeros((270, 480, 3), dtype=np.uint8),
        np.zeros((540, 960, 3), dtype=np.uint8),
    ]
    out = preprocess_frames(frames)
    assert out.shape == (2, 3, RESOLUTION, RESOLUTION)


def test_load_manifest_batches(tmp_path):
    files = []
    for i in range(10):
        f = tmp_path / f"calib_{i:04d}.png"
        f.write_bytes(b"png")  # existence check only
        files.append({"file": f.name})
    (tmp_path / "manifest.json").write_text(json.dumps({"frames": files}))
    batches = load_manifest_batches(tmp_path)
    assert len(batches) == 2  # 10 frames -> 2 full batches of 4, remainder dropped
    assert all(len(b) == BATCH for b in batches)


def test_load_manifest_batches_missing_frame(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"frames": [{"file": "gone.png"}] * 4})
    )
    with pytest.raises(FileNotFoundError):
        load_manifest_batches(tmp_path)


# --------------------------------------------------------------- A/B match ---

def test_box_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float64)
    iou = box_iou_matrix(a, b)
    assert iou.shape == (1, 3)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(25 / 175)
    assert iou[0, 2] == pytest.approx(0.0)


def test_greedy_match_is_one_to_one_and_ordered():
    iou = np.array([
        [0.9, 0.6, 0.0],
        [0.8, 0.7, 0.0],
    ])
    matches = greedy_match(iou, threshold=0.5)
    assert (0, 0, 0.9) == matches[0]
    assert len(matches) == 2
    assert matches[1][:2] == (1, 1)  # 0.8 col taken -> falls to 0.7


def test_greedy_match_empty_and_below_threshold():
    assert greedy_match(np.zeros((0, 0))) == []
    assert greedy_match(np.array([[0.3]]), threshold=0.5) == []


def test_mask_iou():
    a = torch.zeros(8, 8, dtype=torch.bool)
    b = torch.zeros(8, 8, dtype=torch.bool)
    a[:4], b[2:6] = True, True
    assert mask_iou(a, b) == pytest.approx(16 / 48)
    assert mask_iou(torch.zeros(4, 4, dtype=torch.bool),
                    torch.zeros(4, 4, dtype=torch.bool)) == 1.0


def test_mask_iou_across_different_grids():
    """Cross-family comparison: RF-DETR masks are square (e.g. 192x192 over the
    stretched frame) while YOLO's are aspect-scaled (e.g. 144x256) — same frame
    extent, different grids. IoU must align them instead of crashing."""
    # Upper half of the frame, expressed on three different grids: coarse
    # square, finer square, and a wide aspect grid — all must agree exactly.
    a = torch.zeros(8, 8, dtype=torch.bool)
    a[:4] = True
    b = torch.zeros(16, 16, dtype=torch.bool)
    b[:8] = True
    assert mask_iou(a, b) == pytest.approx(1.0)
    w = torch.zeros(8, 16, dtype=torch.bool)
    w[:4] = True
    assert mask_iou(a, w) == pytest.approx(1.0)
    # Odd grid whose row boundary does not align: still close, never a crash.
    odd = torch.zeros(9, 16, dtype=torch.bool)
    odd[:5] = True
    assert mask_iou(a, odd) >= 0.75
    # Disjoint halves stay at zero regardless of grid mismatch.
    c = torch.zeros(16, 16, dtype=torch.bool)
    c[8:] = True
    assert mask_iou(a, c) == pytest.approx(0.0)
    # Same-grid fast path is untouched (no resample when shapes match).
    assert mask_iou(a, a.clone()) == 1.0


def test_compare_frame_counts_misses_and_blindness():
    stats = StratumStats()
    boxes_a = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float64)
    boxes_b = np.array([[1, 1, 10, 10]], dtype=np.float64)  # matches the first only
    masks_a = torch.ones(2, 8, 8, dtype=torch.bool)
    masks_b = torch.ones(1, 8, 8, dtype=torch.bool)
    compare_frame(stats, boxes_a, boxes_b, masks_a, masks_b)
    # And a frame where B is completely blind:
    compare_frame(stats, boxes_a[:1], boxes_a[:0], masks_a[:1], masks_a[:0])
    d = stats.as_dict()
    assert d["frames"] == 2
    assert d["detections_a"] == 3
    assert d["matched"] == 1
    assert d["missed_by_b"] == 2
    assert d["frames_b_blind"] == 1
    assert d["mean_mask_iou"] == pytest.approx(1.0)


# ---------------------------------------------------------------- registry ---

def test_discovery_lists_int8_ir_variants(tmp_path):
    (tmp_path / "rfdetr-v5.onnx").write_bytes(b"onnx")
    (tmp_path / "rfdetr-v5-int8.xml").write_bytes(b"ir")
    (tmp_path / "rfdetr-v5-int8.bin").write_bytes(b"weights")  # must NOT list
    (tmp_path / "other-int8.xml").write_bytes(b"ir")  # not an rfdetr name
    from jasna.mosaic.detection_registry import discover_available_detection_models

    names = discover_available_detection_models(tmp_path)
    assert "rfdetr-v5" in names
    assert "rfdetr-v5-int8" in names
    assert "other-int8" not in names


def _stub_eval_dir(tmp_path, n_frames=4):
    from PIL import Image

    entries = []
    for i in range(n_frames):
        name = f"calib_{i:04d}.png"
        Image.new("RGB", (64, 48), (i * 40, 90, 120)).save(tmp_path / name)
        entries.append({"file": name, "stratum": "high-conf"})
    (tmp_path / "manifest.json").write_text(json.dumps({"frames": entries}))
    return tmp_path


class _StubModel:
    """Duck-typed detection model; returns fixed boxes per frame."""

    def __init__(self, boxes_per_frame: np.ndarray):
        self._boxes = boxes_per_frame

    def __call__(self, frames, *, target_hw):
        from jasna.mosaic.detections import Detections

        b = frames.shape[0]
        n = len(self._boxes)
        return Detections(
            boxes_xyxy=[self._boxes.copy() for _ in range(b)],
            masks=[torch.ones(n, 8, 8, dtype=torch.bool) for _ in range(b)],
        )

    def close(self):
        pass


def _run_ab_with_stubs(tmp_path, monkeypatch, boxes_a, boxes_b) -> int:
    from argparse import Namespace

    import jasna.mosaic.detection_registry as registry
    from jasna.detection_ab import run_ab

    stubs = iter([_StubModel(boxes_a), _StubModel(boxes_b)])
    monkeypatch.setattr(registry, "build_detection_model",
                        lambda *a, **k: next(stubs))
    monkeypatch.setattr(registry, "require_detection_model_weights",
                        lambda n: tmp_path / f"{n}.fake")
    monkeypatch.setattr(registry, "coerce_detection_model_name", lambda n: n)
    return run_ab(Namespace(
        frames=str(_stub_eval_dir(tmp_path)),
        model_a="a", model_b="b", device="cpu", score_threshold=0.25,
        max_annotated=80,
    ))


def test_ab_fails_loudly_when_baseline_detects_nothing(tmp_path, monkeypatch):
    rc = _run_ab_with_stubs(
        tmp_path, monkeypatch,
        boxes_a=np.zeros((0, 4)), boxes_b=np.zeros((0, 4)),
    )
    assert rc == 1, "dead baseline must fail, not report a perfect 0.0% miss"


def test_ab_passes_and_writes_report_with_live_baseline(tmp_path, monkeypatch):
    boxes = np.array([[4.0, 4.0, 20.0, 20.0]])
    rc = _run_ab_with_stubs(tmp_path, monkeypatch, boxes_a=boxes, boxes_b=boxes)
    assert rc == 0
    report = json.loads((tmp_path / "ab_report.json").read_text(encoding="utf-8"))
    assert report["overall"]["miss_rate"] == 0.0
    assert report["overall"]["detections_a"] == 4
    assert report["per_stratum"]["high-conf"]["mean_box_iou"] == pytest.approx(1.0)
    # Perfect agreement: no disagreement records, no annotated copies.
    assert report["disagreements"] == []
    assert list((tmp_path / "ab_disagreements").iterdir()) == []


def test_ab_annotates_disagreement_frames(tmp_path, monkeypatch):
    """B blind on every frame: each frame is recorded as a disagreement and an
    annotated copy with A's box drawn (green) lands in ab_disagreements/."""
    from PIL import Image

    boxes_a = np.array([[4.0, 4.0, 20.0, 20.0]])
    rc = _run_ab_with_stubs(
        tmp_path, monkeypatch, boxes_a=boxes_a, boxes_b=np.zeros((0, 4)),
    )
    assert rc == 0
    report = json.loads((tmp_path / "ab_report.json").read_text(encoding="utf-8"))
    assert len(report["disagreements"]) == 4
    assert all(d["b_blind"] and d["missed_by_b"] == 1 for d in report["disagreements"])

    copies = sorted((tmp_path / "ab_disagreements").iterdir())
    assert len(copies) == 4
    assert all(c.name.startswith("blind_") for c in copies)
    with Image.open(copies[0]) as img:
        # A's box outline (green, width 3) passes through (4, 10).
        assert img.getpixel((4, 10)) == (64, 255, 64)
        # Far outside any box, the original solid color is untouched.
        assert img.getpixel((40, 40)) != (64, 255, 64)


def test_int8_name_resolves_to_openvino_ir():
    assert is_rfdetr_model("rfdetr-v5-int8")
    spec = detection_model_spec("rfdetr-v5-int8")
    assert spec.backend == "rfdetr"
    assert spec.filename == "rfdetr-v5-int8.xml"
    assert detection_model_weights_path("rfdetr-v5-int8").name == "rfdetr-v5-int8.xml"
    # The fp16 model keeps resolving to ONNX.
    assert detection_model_spec("rfdetr-v5").filename == "rfdetr-v5.onnx"


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_non_intel_rejects_ir_model_with_clear_error(tmp_path, monkeypatch, vendor):
    import jasna.mosaic.rfdetr as module

    monkeypatch.setattr(module, "is_amd_device", lambda _d: vendor == "amd")
    monkeypatch.setattr(module, "is_intel_device", lambda _d: False)
    monkeypatch.setattr(module, "is_nvidia_device", lambda _d: vendor == "nvidia")
    with pytest.raises(RuntimeError, match="OpenVINO IR"):
        module.RfDetrMosaicDetectionModel(
            onnx_path=tmp_path / "rfdetr-v5-int8.xml",
            batch_size=4,
            device=torch.device("cuda:0"),
        )
    with pytest.raises(RuntimeError, match="OpenVINO IR"):
        module.compile_rfdetr_engine(
            tmp_path / "rfdetr-v5-int8.xml", torch.device("cuda:0"),
            batch_size=4, fp16=True,
        )


def test_intel_accepts_ir_suffix(monkeypatch, tmp_path):
    """The guard itself must not block the intended Intel path."""
    from jasna.mosaic.rfdetr import _require_onnx_on_non_intel
    import jasna.mosaic.rfdetr as module

    monkeypatch.setattr(module, "is_intel_device", lambda _d: True)
    _require_onnx_on_non_intel(tmp_path / "rfdetr-v5-int8.xml", torch.device("xpu:0"))
