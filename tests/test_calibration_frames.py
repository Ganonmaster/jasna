"""Tests for the calibration-frame extractor.

Selection logic runs on synthetic candidates (no video, no models); the
end-to-end test encodes a tiny real mp4 with PyAV and fakes only the detector,
so seek/decode/score/select/extract plumbing is exercised for real on cpu.
"""
from __future__ import annotations

import json
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from jasna.calibration_frames import (
    HIGH_CONF,
    LOW_CONF,
    MIN_GAP_S,
    Candidate,
    _frame_features,
    _iter_videos,
    _probe_timestamps,
    main,
    select_calibration_set,
)


def _candidate(video="v.mp4", ts=0.0, score=0.9, **kw) -> Candidate:
    base = dict(mask_area=0.05, luma_mean=0.5, luma_std=0.2, colorfulness=0.3)
    base.update(kw)
    return Candidate(video=video, ts=ts, score=score, **base)


def _pool() -> list[Candidate]:
    rng = np.random.default_rng(7)
    pool = []
    for v in range(12):
        for k in range(30):
            pool.append(_candidate(
                video=f"video_{v}.mp4",
                ts=float(k * 60),
                score=float(rng.uniform(0.0, 1.0)),
                luma_mean=float(rng.uniform(0.05, 0.95)),
                luma_std=float(rng.uniform(0.05, 0.4)),
                colorfulness=float(rng.uniform(0.0, 0.6)),
                mask_area=float(rng.uniform(0.0, 0.3)),
            ))
    return pool


def test_strata_shares_and_count():
    selected = select_calibration_set(
        _pool(), 60, negatives_frac=0.15, low_conf_frac=0.25,
        per_video_cap=10, seed=1,
    )
    assert len(selected) == 60
    strata = {s: sum(1 for c in selected if c.stratum == s) for s in
              ("high-conf", "borderline", "no-mosaic")}
    assert strata["no-mosaic"] == 9      # 15% of 60
    assert strata["borderline"] == 15    # 25% of 60
    assert strata["high-conf"] == 36
    for c in selected:
        if c.stratum == "high-conf":
            assert c.score >= HIGH_CONF
        elif c.stratum == "borderline":
            assert LOW_CONF <= c.score < HIGH_CONF
        else:
            assert c.score < LOW_CONF


def test_per_video_cap_and_time_gap():
    selected = select_calibration_set(
        _pool(), 60, negatives_frac=0.15, low_conf_frac=0.25,
        per_video_cap=4, seed=1,
    )
    per_video: dict[str, list[float]] = {}
    for c in selected:
        per_video.setdefault(c.video, []).append(c.ts)
    for video, stamps in per_video.items():
        assert len(stamps) <= 4, f"{video} exceeded the cap"
        stamps.sort()
        assert all(b - a >= MIN_GAP_S for a, b in zip(stamps, stamps[1:]))


def test_selection_is_deterministic():
    a = select_calibration_set(_pool(), 40, negatives_frac=0.15,
                               low_conf_frac=0.25, per_video_cap=6, seed=42)
    b = select_calibration_set(_pool(), 40, negatives_frac=0.15,
                               low_conf_frac=0.25, per_video_cap=6, seed=42)
    assert [(c.video, c.ts) for c in a] == [(c.video, c.ts) for c in b]


def test_shortfall_refills_from_positive_pool():
    # No negatives available at all: the set must still reach the target.
    pool = [c for c in _pool() if c.score >= LOW_CONF]
    selected = select_calibration_set(
        pool, 40, negatives_frac=0.25, low_conf_frac=0.25,
        per_video_cap=10, seed=3,
    )
    assert len(selected) == 40
    assert all(c.stratum != "no-mosaic" for c in selected)


def test_probe_timestamps_interior_and_bounded():
    ts = _probe_timestamps(3600.0, cap=48)
    assert len(ts) == 48
    assert 0 < ts[0] < ts[-1] < 3600.0
    assert len(_probe_timestamps(120.0, cap=48)) == 12  # floor


def test_frame_features_ranges():
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    bright = np.full((64, 64, 3), 250, dtype=np.uint8)
    l1, s1, c1 = _frame_features(dark)
    l2, s2, c2 = _frame_features(bright)
    assert l1 < 0.05 < 0.9 < l2
    assert s1 == pytest.approx(0.0, abs=1e-4)
    assert c1 == pytest.approx(0.0, abs=1e-4) and c2 < 0.1  # gray-ish


def _write_test_video(path, seconds=180, fps=4, size=64, tint=(0, 0, 0)):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        for i in range(seconds * fps):
            # Vary brightness over time so features differ between probes; the
            # per-video tint keeps the two clips visually distinct so the
            # diversity selection has a reason to draw from both.
            level = int(20 + 180 * (i / (seconds * fps)))
            img = np.full((size, size, 3), level, dtype=np.uint8)
            img = np.clip(img.astype(np.int16) + np.array(tint, dtype=np.int16), 0, 255)
            frame = av.VideoFrame.from_ndarray(img.astype(np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class _FakeDetector:
    """Scores by frame brightness: bright frames 'contain mosaics'."""

    def __init__(self, *_, **__):
        pass

    def scan_scores_masks(self, frames_uint8_bchw, *, mask_hw):
        b = frames_uint8_bchw.shape[0]
        luma = frames_uint8_bchw.float().mean(dim=(1, 2, 3)) / 255.0
        scores = (luma * 1.2).clamp(0, 1)
        masks = torch.zeros((b, *mask_hw), dtype=torch.bool)
        masks[:, : mask_hw[0] // 4] = (scores > LOW_CONF).view(b, 1, 1)
        return scores, masks

    def close(self):
        pass


def test_end_to_end_on_synthetic_video(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    # 360s: probes land ~28s apart, clearing MIN_GAP_S like real-length videos
    # (with 180s they'd be ~14s apart and the gap constraint starves the count).
    _write_test_video(lib / "clip_a.mp4", seconds=360)
    _write_test_video(lib / "clip_b.mp4", seconds=360, tint=(-15, 0, 40))
    (lib / "notes.txt").write_text("not a video")

    import jasna.mosaic.detection_registry as registry
    monkeypatch.setattr(registry, "build_detection_model",
                        lambda *a, **k: _FakeDetector())
    monkeypatch.setattr(registry, "require_detection_model_weights",
                        lambda _n: tmp_path / "fake.pt")

    out = tmp_path / "calib"
    rc = main([
        "--library", str(lib), "--out", str(out), "--count", "12",
        "--device", "cpu", "--seed", "5", "--probes-per-video", "16",
    ])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    assert len(frames) == 12
    videos_used = {f["video"] for f in frames}
    assert len(videos_used) == 2, "both videos must contribute"
    for f in frames:
        png = out / f["file"]
        assert png.exists() and png.stat().st_size > 0
    assert manifest["stats"]["videos_probed"] == 2
    assert (out / "probe_cache.jsonl").exists()

    # Resume: a second run must reuse the probe cache. The detector is still
    # constructed (some videos might be uncached), so the bomb detonates on
    # SCAN — which a full cache hit must never trigger.
    class _Bomb:
        def scan_scores_masks(self, *a, **k):
            raise AssertionError("probe cache was not used")

        def close(self):
            pass

    monkeypatch.setattr(registry, "build_detection_model", lambda *a, **k: _Bomb())
    rc = main([
        "--library", str(lib), "--out", str(out), "--count", "12",
        "--device", "cpu", "--seed", "5", "--probes-per-video", "16",
    ])
    assert rc == 0
