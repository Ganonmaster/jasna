from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from jasna.accelerator import is_nvidia_device, synchronize

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer

# basicvrspp_tenorrt_compilation / basicvsrpp_sub_engines pull in jasna.trt ->
# `import tensorrt`, which no AMD/Intel build ships. They are only needed on the
# NVIDIA TensorRT split path, so both are imported lazily below.

CLIP_LENGTH = 60
SIZE = 256
WARMUP = 3
RUNS = 100


def _timed(label: str, fn, *args, **kwargs):
    synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    synchronize()
    dt = time.perf_counter() - t0
    print(f"  {label:30s} {dt*1000:8.1f} ms")
    return result


def _profile_split_forward(split: "BasicVSRPlusPlusNetSplit", device: torch.device, dtype: torch.dtype) -> None:
    """Per-stage timings mirroring BasicVSRPlusPlusNetSplit.forward() with the
    CURRENT engine structure: the fused preprocess engine (feat_extract +
    bicubic downsample + bidirectional SPyNet), per-direction flow precompute,
    the four propagate passes (loop-body engines), and the upsample engine.
    Stages call the split's real methods rather than re-implementing them, so
    a future engine refactor breaks this loudly instead of drifting."""
    T = CLIP_LENGTH
    lqs = torch.randn(1, T, 3, SIZE, SIZE, device=device, dtype=dtype)

    for _ in range(WARMUP):
        split(lqs)
    synchronize()

    print(f"\n=== Profiling BasicVSRPlusPlusNetSplit (T={T}) ===")
    n, t, c, h, w = lqs.size()

    # Stage 1: fused preprocess engine. T=60 >= the engine's min batch, so
    # forward()'s short-clip padding branch never triggers here.
    lqs_flat = lqs.view(-1, c, h, w)
    feats_, flows_fwd, flows_bwd = _timed(
        "preprocess (TRT, fused)", split._preprocess_engine, lqs_flat
    )
    h_f, w_f = feats_.shape[2:]

    feats_ = feats_.view(n, t, -1, h_f, w_f)
    feats: dict[str, list[torch.Tensor]] = {
        "spatial": [feats_[:, i, :, :, :] for i in range(t)]
    }
    flows_forward = flows_fwd.view(n, t - 1, 2, h // 4, w // 4)
    flows_backward = flows_bwd.view(n, t - 1, 2, h // 4, w // 4)

    grid = split._make_identity_grid(h_f, w_f, lqs.device, lqs.dtype)

    # Stage 2: per-direction flow precompute (accumulated flows + grids).
    flow_data: dict[str, tuple] = {}
    synchronize()
    tp0 = time.perf_counter()
    for direction in ["backward", "forward"]:
        flows = flows_backward if direction == "backward" else flows_forward
        flow_data[direction] = split._precompute_flow_data(flows, direction, grid)
    synchronize()
    print(f"  {'flow precompute':30s} {(time.perf_counter() - tp0)*1000:8.1f} ms")

    # Stage 3: the four propagation passes (loop-body TRT engines + the
    # first-frame eager backbone call inside each).
    total_propagate = 0.0
    for iter_ in [1, 2]:
        for direction in ["backward", "forward"]:
            module = f"{direction}_{iter_}"
            feats[module] = []
            flows = flows_backward if direction == "backward" else flows_forward
            fi, fli, af, fg, ag = flow_data[direction]
            synchronize()
            t0 = time.perf_counter()
            feats = split.propagate(feats, flows, module, grid, fi, fli, af, fg, ag)
            synchronize()
            dt = time.perf_counter() - t0
            total_propagate += dt
            print(f"  {f'propagate {module} (TRT)':30s} {dt*1000:8.1f} ms")
    print(f"  {'propagate total':30s} {total_propagate*1000:8.1f} ms")

    # Stage 4: upsample engine (consumes the feats dict like forward() does).
    _timed("upsample (TRT)", split.upsample, lqs, feats)

    durations: list[float] = []
    for _ in range(RUNS):
        synchronize()
        t0 = time.perf_counter()
        split(lqs)
        synchronize()
        durations.append(time.perf_counter() - t0)

    med = statistics.median(durations)
    print(f"\n  {'FULL FORWARD median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")


def benchmark_basicvsrpp_restoration(
    *,
    device: torch.device,
    fp16: bool,
    restoration_model_path: Path | None,
    compile_basicvsrpp: bool,
    **_: object,
) -> None:
    if restoration_model_path is None or not restoration_model_path.resolve().exists():
        return
    path = restoration_model_path.resolve()

    if is_nvidia_device(device):
        from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy

        use_tensorrt = basicvsrpp_startup_policy(
            restoration_model_path=str(path),
            device=device,
            fp16=fp16,
            compile_basicvsrpp=compile_basicvsrpp,
            max_clip_size=CLIP_LENGTH,
        )
    else:
        # AMD/Intel run BasicVSR++ eager; TensorRT is never used off NVIDIA.
        use_tensorrt = False
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(path),
        device=device,
        max_clip_size=CLIP_LENGTH,
        use_tensorrt=use_tensorrt,
        fp16=fp16,
    )

    dtype = torch.float16 if fp16 else torch.float32

    if restorer._split_forward is not None:
        with torch.inference_mode():
            _profile_split_forward(restorer._split_forward, device, dtype)
    else:
        print("\nNo split forward available (engines missing?), skipping detailed profiling.")

        durations: list[float] = []
        # raw_process expects (C, H, W) RGB tensors (matching the pipeline's
        # resized_crops); an (H, W, 3) HWC layout feeds the 3 channels in as width
        # and blows up in the model's downsample (W: 3 -> 0).
        video = [
            torch.randint(0, 256, (3, SIZE, SIZE), dtype=torch.uint8, device=device)
            for _ in range(CLIP_LENGTH)
        ]
        with torch.inference_mode():
            for _ in range(RUNS):
                start = time.perf_counter()
                restorer.raw_process(video)
                synchronize()
                durations.append(time.perf_counter() - start)

        med = statistics.median(durations)
        print(f"\n  {'raw_process median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")
    print()
