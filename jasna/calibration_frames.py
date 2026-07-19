# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Hands-off calibration-frame extraction for INT8 detector quantization.

Point it at a media library and it builds a representative, diverse frame set:

    python -m jasna.calibration_frames --library /path/to/library \\
        --out calibration_frames --count 800 --device xpu:0

Three passes:

1. PROBE — every video is sparsely sampled (seek + decode single frames,
   roughly one probe per minute, capped). Each probe frame is scored by the
   fast YOLO mosaic detector (`scan_scores_masks`, ~135 fps on the B50) and
   reduced to a small feature record (detector score, mosaic-mask area, luma
   mean/std, colorfulness); pixels are discarded, so RAM stays flat across
   hundreds of videos. Probes near hits get one densification round to find
   more distinct mosaic scenes. Per-video results are cached in the output
   directory (keyed by path+size+mtime), so an interrupted run resumes.
   Using YOLO rather than RF-DETR to select frames also avoids self-selection
   bias when the set later calibrates RF-DETR.

2. SELECT — candidates are split into strata: high-confidence mosaics,
   BORDERLINE mosaics (the detections quantization is most likely to flip),
   and no-mosaic frames (the detector sees plenty of those in production).
   Within each stratum a greedy farthest-point pass over normalized features
   spreads the picks across visual looks, under a per-video cap and a minimum
   time gap so no title or scene dominates.

3. EXTRACT — only the winners are re-decoded and written as lossless PNGs
   plus a manifest.json describing every frame (source, timestamp, features,
   stratum) and the run configuration.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
import torch

logger = logging.getLogger("jasna.calibration_frames")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".wmv", ".mov", ".m2ts", ".ts", ".mpg", ".mpeg", ".m4v"}
SCAN_MASK_HW = (72, 128)
PROBE_BATCH = 16
HIGH_CONF = 0.55
LOW_CONF = 0.25
DENSIFY_OFFSETS_S = (-90.0, 90.0)
MIN_GAP_S = 20.0


@dataclass
class Candidate:
    video: str
    ts: float
    score: float
    mask_area: float  # fraction of frame covered by detected mosaic
    luma_mean: float
    luma_std: float
    colorfulness: float
    stratum: str = ""

    def features(self) -> list[float]:
        return [self.luma_mean, self.luma_std, self.colorfulness, self.mask_area, self.score]


def _iter_videos(roots: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(root)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(p)
    return videos


def _video_duration_s(container) -> float | None:
    if container.duration is not None:
        return container.duration / av.time_base
    stream = container.streams.video[0]
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return None


def _probe_timestamps(duration_s: float, cap: int) -> list[float]:
    # ~1 probe/minute, at least 12, capped; placed mid-interval so the first
    # probe is not the title card and the last is not the credits fade.
    count = int(max(12, min(cap, duration_s / 60.0)))
    step = duration_s / (count + 1)
    return [step * (i + 1) for i in range(count)]


def _decode_frame_at(container, stream, ts_s: float) -> np.ndarray | None:
    """Seek near ts_s and decode the first frame at/after it (RGB HWC uint8)."""
    try:
        container.seek(int(ts_s * av.time_base), backward=True)
    except av.FFmpegError:
        return None
    try:
        for frame in container.decode(stream):
            if frame.time is None or frame.time >= ts_s - 0.5:
                return frame.to_ndarray(format="rgb24")
    except av.FFmpegError:
        return None
    return None


def _frame_features(rgb: np.ndarray) -> tuple[float, float, float]:
    small = rgb[::4, ::4].astype(np.float32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    # Hasler & Süsstrunk colorfulness, cheap and rank-stable.
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(abs(rg.mean()), abs(yb.mean()))
    )
    return float(luma.mean() / 255.0), float(luma.std() / 255.0), colorfulness / 255.0


def _score_batch(detector, frames: list[np.ndarray]) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    areas: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(frames), PROBE_BATCH):
            chunk = frames[i : i + PROBE_BATCH]
            batch = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2).contiguous()
            s, masks = detector.scan_scores_masks(batch, mask_hw=SCAN_MASK_HW)
            scores.extend(float(v) for v in s.cpu())
            mask_px = float(SCAN_MASK_HW[0] * SCAN_MASK_HW[1])
            areas.extend(float(m.sum()) / mask_px for m in masks.cpu())
    return scores, areas


def _cache_key(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"


def _load_probe_cache(cache_file: Path) -> dict[str, list[dict]]:
    cache: dict[str, list[dict]] = {}
    if not cache_file.exists():
        return cache
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            cache[entry["key"]] = entry["candidates"]
        except (json.JSONDecodeError, KeyError):
            continue
    return cache


def _probe_video(video: Path, detector, probes_cap: int) -> list[Candidate]:
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = _video_duration_s(container)
        if not duration or duration < 60.0:
            return []

        def probe_at(timestamps: list[float]) -> list[Candidate]:
            # Decode + score one batch at a time so peak RAM per video is one
            # PROBE_BATCH of frames (a 4K batch is ~400 MB; 48 held at once
            # would be >1 GB).
            out: list[Candidate] = []
            for i in range(0, len(timestamps), PROBE_BATCH):
                frames, kept_ts = [], []
                for ts in timestamps[i : i + PROBE_BATCH]:
                    # Round BEFORE decoding: the extraction pass re-seeks at the
                    # manifest's (rounded) ts, and probing at the unrounded value
                    # shifts the seek/acceptance window enough to save an
                    # adjacent frame instead of the scored one (~10% at 30fps).
                    ts = round(ts, 2)
                    rgb = _decode_frame_at(container, stream, ts)
                    if rgb is not None:
                        frames.append(rgb)
                        kept_ts.append(ts)
                if not frames:
                    continue
                scores, areas = _score_batch(detector, frames)
                for ts, rgb, score, area in zip(kept_ts, frames, scores, areas):
                    luma_mean, luma_std, colorfulness = _frame_features(rgb)
                    out.append(Candidate(
                        video=str(video), ts=ts, score=round(score, 4),
                        mask_area=round(area, 5), luma_mean=round(luma_mean, 4),
                        luma_std=round(luma_std, 4), colorfulness=round(colorfulness, 4),
                    ))
            return out

        candidates = probe_at(_probe_timestamps(duration, probes_cap))
        # Densify around hits: mosaic scenes cluster, and one probe per scene
        # is enough — look a couple of minutes away for *other* scenes.
        seen = {c.ts for c in candidates}
        extra_ts = []
        for c in candidates:
            if c.score >= LOW_CONF:
                for off in DENSIFY_OFFSETS_S:
                    ts = c.ts + off
                    if 0 < ts < duration and all(abs(ts - s) > MIN_GAP_S for s in seen):
                        extra_ts.append(ts)
                        seen.add(ts)
        if extra_ts:
            candidates.extend(probe_at(sorted(extra_ts)[: probes_cap]))
    return candidates


def _normalize(matrix: np.ndarray) -> np.ndarray:
    lo = matrix.min(axis=0)
    span = matrix.max(axis=0) - lo
    span[span == 0] = 1.0
    return (matrix - lo) / span


def _farthest_point_select(
    pool: list[Candidate],
    count: int,
    rng: random.Random,
    per_video_cap: int,
    already: list[Candidate],
) -> list[Candidate]:
    """Greedy k-center over normalized features, honoring the per-video cap
    and a minimum time gap against everything selected so far."""
    if not pool or count <= 0:
        return []
    feats = _normalize(np.array([c.features() for c in pool], dtype=np.float64))
    selected_idx: list[int] = []
    video_counts: dict[str, int] = {}
    picked_ts: dict[str, list[float]] = {}
    for c in already:
        video_counts[c.video] = video_counts.get(c.video, 0) + 1
        picked_ts.setdefault(c.video, []).append(c.ts)

    def admissible(c: Candidate) -> bool:
        if video_counts.get(c.video, 0) >= per_video_cap:
            return False
        return all(abs(c.ts - t) >= MIN_GAP_S for t in picked_ts.get(c.video, ()))

    min_dist = np.full(len(pool), np.inf)
    start = rng.randrange(len(pool))
    order = [start] + [i for i in range(len(pool)) if i != start]
    for idx in order:
        if admissible(pool[idx]):
            selected_idx.append(idx)
            break
    if not selected_idx:
        return []
    c0 = pool[selected_idx[0]]
    video_counts[c0.video] = video_counts.get(c0.video, 0) + 1
    picked_ts.setdefault(c0.video, []).append(c0.ts)
    min_dist = np.minimum(min_dist, np.linalg.norm(feats - feats[selected_idx[0]], axis=1))

    while len(selected_idx) < count:
        ranked = np.argsort(-min_dist)
        chosen = next(
            (int(i) for i in ranked if int(i) not in selected_idx and admissible(pool[int(i)])),
            None,
        )
        if chosen is None:
            break
        selected_idx.append(chosen)
        c = pool[chosen]
        video_counts[c.video] = video_counts.get(c.video, 0) + 1
        picked_ts.setdefault(c.video, []).append(c.ts)
        min_dist = np.minimum(min_dist, np.linalg.norm(feats - feats[chosen], axis=1))
    return [pool[i] for i in selected_idx]


def select_calibration_set(
    candidates: list[Candidate],
    count: int,
    *,
    negatives_frac: float,
    low_conf_frac: float,
    per_video_cap: int,
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    high = [c for c in candidates if c.score >= HIGH_CONF]
    low = [c for c in candidates if LOW_CONF <= c.score < HIGH_CONF]
    neg = [c for c in candidates if c.score < LOW_CONF]
    for c, name in ((high, "high-conf"), (low, "borderline"), (neg, "no-mosaic")):
        for x in c:
            x.stratum = name

    n_neg = int(round(count * negatives_frac))
    n_low = int(round(count * low_conf_frac))
    n_high = count - n_neg - n_low

    selected: list[Candidate] = []
    # High first (largest share), then borderline, then negatives; shortfalls
    # in one stratum are refilled from high-conf, the safest pool.
    selected += _farthest_point_select(high, n_high, rng, per_video_cap, selected)
    selected += _farthest_point_select(low, n_low, rng, per_video_cap, selected)
    selected += _farthest_point_select(neg, n_neg, rng, per_video_cap, selected)
    if len(selected) < count:
        pool = [c for c in high + low if c not in selected]
        selected += _farthest_point_select(pool, count - len(selected), rng, per_video_cap, selected)
    return selected


def _extract_frames(selected: list[Candidate], out_dir: Path) -> list[dict]:
    from PIL import Image

    by_video: dict[str, list[Candidate]] = {}
    for c in selected:
        by_video.setdefault(c.video, []).append(c)

    entries: list[dict] = []
    idx = 0
    for video, group in by_video.items():
        try:
            with av.open(video) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for c in sorted(group, key=lambda x: x.ts):
                    rgb = _decode_frame_at(container, stream, c.ts)
                    if rgb is None:
                        logger.warning("could not re-decode %s @ %.1fs; skipping", video, c.ts)
                        continue
                    name = f"calib_{idx:04d}_{Path(video).stem[:40]}_{int(c.ts)}s.png"
                    Image.fromarray(rgb).save(out_dir / name)
                    entries.append({"file": name, **asdict(c)})
                    idx += 1
        except (av.FFmpegError, OSError) as exc:
            logger.warning("extraction failed for %s: %s", video, exc)
    return entries


def run_extraction(args: argparse.Namespace) -> int:
    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        require_detection_model_weights,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    videos = _iter_videos([Path(p) for p in args.library])
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        print("No videos found under the given library path(s).")
        return 1
    print(f"Probing {len(videos)} videos with {args.detection_model} on {device} ...")

    det_name = coerce_detection_model_name(args.detection_model)
    detector = build_detection_model(
        det_name,
        require_detection_model_weights(det_name),
        batch_size=PROBE_BATCH,
        device=device,
        score_threshold=0.2,
        fp16=True,
    )

    cache_file = out_dir / "probe_cache.jsonl"
    cache = {} if args.no_resume else _load_probe_cache(cache_file)
    candidates: list[Candidate] = []
    probed = failed = 0
    start = time.perf_counter()
    with cache_file.open("a", encoding="utf-8") as cache_out:
        for n, video in enumerate(videos, 1):
            try:
                key = _cache_key(video)
                if key in cache:
                    video_cands = [Candidate(**d) for d in cache[key]]
                else:
                    video_cands = _probe_video(video, detector, args.probes_per_video)
                    cache_out.write(json.dumps(
                        {"key": key, "candidates": [asdict(c) for c in video_cands]}
                    ) + "\n")
                    cache_out.flush()
                candidates.extend(video_cands)
                probed += 1
            except (av.FFmpegError, OSError, ValueError, IndexError, RuntimeError) as exc:
                # IndexError: no video stream (audio-only/odd containers).
                # RuntimeError: torch/detector failures on degenerate frames.
                # One bad file must never kill a hands-off library run.
                failed += 1
                logger.warning("skipping unreadable video %s: %s", video, exc)
            if n % 10 == 0 or n == len(videos):
                rate = n / max(time.perf_counter() - start, 1e-6)
                print(f"  [{n}/{len(videos)}] {len(candidates)} candidates "
                      f"({rate * 60:.0f} videos/min, {failed} unreadable)")
    detector.close()

    hits = sum(1 for c in candidates if c.score >= LOW_CONF)
    print(f"Probe done: {len(candidates)} candidates, {hits} with mosaics "
          f"({probed} videos probed, {failed} unreadable).")
    if hits == 0:
        print("No mosaic-positive frames found — check the library path/model.")
        return 1

    per_video_cap = max(2, math.ceil(3 * args.count / max(len(videos), 1)))
    selected = select_calibration_set(
        candidates, args.count,
        negatives_frac=args.negatives_frac, low_conf_frac=args.low_conf_frac,
        per_video_cap=per_video_cap, seed=args.seed,
    )
    strata = {s: sum(1 for c in selected if c.stratum == s) for s in
              ("high-conf", "borderline", "no-mosaic")}
    n_videos_used = len({c.video for c in selected})
    print(f"Selected {len(selected)} frames from {n_videos_used} videos "
          f"(per-video cap {per_video_cap}): {strata}")

    print("Extracting frames ...")
    entries = _extract_frames(selected, out_dir)
    manifest = {
        "config": {k: v for k, v in vars(args).items()},
        "stats": {
            "videos_found": len(videos), "videos_probed": probed,
            "videos_unreadable": failed, "candidates": len(candidates),
            "mosaic_positive_candidates": hits, "selected": len(entries),
            "videos_in_selection": n_videos_used, "strata": strata,
        },
        "frames": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(f"Wrote {len(entries)} frames + manifest.json to {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a diverse mosaic calibration frame set from a media library."
    )
    parser.add_argument("--library", action="append", required=True,
                        help="Library directory (repeatable) or a single video file")
    parser.add_argument("--out", default="calibration_frames")
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detection-model", default="lada-yolo-v4",
                        help="Fast prober; intentionally not RF-DETR to avoid "
                             "self-selection bias in the calibration set")
    parser.add_argument("--negatives-frac", type=float, default=0.15)
    parser.add_argument("--low-conf-frac", type=float, default=0.25)
    parser.add_argument("--probes-per-video", type=int, default=48)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore the probe cache and re-probe everything")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return run_extraction(args)


if __name__ == "__main__":
    sys.exit(main())
