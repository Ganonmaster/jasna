# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Profile deform_conv2d's share of BasicVSR++ restoration.

Answers "is a fused (Triton/SYCL) deform kernel worth building for xpu?" with
four measurements:

1. Instrumented clip pass — wraps the real deform dispatch inside raw_process
   and times every call, giving deform's share of a 60-frame clip per backend
   (torchvision native where available, the grid_sample composition always).
2. Micro-benchmark on the dominant recorded shape — native vs composition on
   identical inputs, plus a stage breakdown of the composition (grid build /
   grid_sample / mask multiply / 1x1 conv).
3. Bandwidth floor — measured device copy bandwidth + itemized traffic
   estimates give the theoretical minimum time for a perfectly fused kernel,
   i.e. the ceiling on what one could recover.
4. torch.compile probe — can inductor fuse the composition for free? (Runs
   last; a compiler crash cannot eat the results above. JASNA_PROFILE_COMPILE=0
   skips it.)

Opt-in via `jasna --benchmark --benchmark-filter deform`; never part of the
default benchmark run.
"""
from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchvision

from jasna.accelerator import DEFORM_BACKEND_ENV, device_name, synchronize
from jasna.models.basicvsrpp.deformconv import (
    _grid_constants,
    modulated_deform_conv2d_grid_sample,
)
from torch.nn.modules.utils import _pair

CLIP_LENGTH = 60
SIZE = 256
CLEAN_RUNS = 7
INSTRUMENTED_RUNS = 3
MICRO_WARMUP = 10
MICRO_ITERS = 50


def _median_ms(fn, device: torch.device, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000.0


class _CallRecord:
    __slots__ = ("count", "total_s", "args")

    def __init__(self) -> None:
        self.count = 0
        self.total_s = 0.0
        self.args: tuple | None = None


def _signature(x, offset, weight, stride, padding, dilation, mask, groups):
    return (
        tuple(x.shape),
        tuple(offset.shape),
        tuple(weight.shape),
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        tuple(mask.shape),
        int(groups),
    )


def _run_instrumented_pass(restorer, video, device) -> dict[tuple, _CallRecord]:
    """One raw_process with every deform call sync-timed. ~240 syncs add only
    tens of ms of overhead; totals are reported against the clean pass."""
    import jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net as net_mod

    records: dict[tuple, _CallRecord] = defaultdict(_CallRecord)
    real = net_mod.deform_conv2d_dispatch

    def timed(x, offset, weight, bias, stride, padding, dilation, mask, groups=1):
        synchronize(device)
        start = time.perf_counter()
        out = real(x, offset, weight, bias, stride, padding, dilation, mask, groups)
        synchronize(device)
        rec = records[_signature(x, offset, weight, stride, padding, dilation, mask, groups)]
        rec.count += 1
        rec.total_s += time.perf_counter() - start
        if rec.args is None:
            rec.args = (_pair(stride), _pair(padding), _pair(dilation), int(groups))
        return out

    net_mod.deform_conv2d_dispatch = timed
    try:
        restorer.raw_process(video)
    finally:
        net_mod.deform_conv2d_dispatch = real
    return records


def _profile_backend(backend: str, restorer, video, device) -> dict:
    prev = os.environ.get(DEFORM_BACKEND_ENV)
    os.environ[DEFORM_BACKEND_ENV] = backend
    try:
        clean = _median_ms(
            lambda: restorer.raw_process(video), device,
            warmup=2, iters=CLEAN_RUNS,
        )
        deform_totals = []
        records: dict[tuple, _CallRecord] = {}
        for _ in range(INSTRUMENTED_RUNS):
            records = _run_instrumented_pass(restorer, video, device)
            deform_totals.append(sum(r.total_s for r in records.values()) * 1000.0)
        deform_ms = statistics.median(deform_totals)
    finally:
        if prev is None:
            os.environ.pop(DEFORM_BACKEND_ENV, None)
        else:
            os.environ[DEFORM_BACKEND_ENV] = prev
    return {"clean_ms": clean, "deform_ms": deform_ms, "records": records}


def _dominant(records: dict[tuple, _CallRecord]) -> tuple | None:
    if not records:
        return None
    return max(records.items(), key=lambda kv: kv[1].total_s)[0]


def _make_micro_inputs(sig: tuple, device: torch.device, dtype: torch.dtype):
    x_shape, offset_shape, weight_shape, stride, padding, dilation, mask_shape, groups = sig
    x = torch.randn(x_shape, device=device, dtype=dtype)
    # Realistic offsets stay within a few pixels; huge random offsets would
    # sample mostly zeros and flatter the composition's cache behavior.
    offset = torch.randn(offset_shape, device=device, dtype=dtype) * 2.0
    mask = torch.rand(mask_shape, device=device, dtype=dtype)
    weight = torch.randn(weight_shape, device=device, dtype=dtype) * 0.05
    bias = torch.randn(weight_shape[0], device=device, dtype=dtype) * 0.05
    return x, offset, mask, weight, bias, stride, padding, dilation, groups


def _micro_composition(inputs) -> float:
    x, offset, mask, weight, bias, stride, padding, dilation, groups = inputs
    return lambda: modulated_deform_conv2d_grid_sample(
        x, offset, mask, weight, bias, stride, padding, dilation, groups
    )


def _micro_native(inputs):
    x, offset, mask, weight, bias, stride, padding, dilation, groups = inputs
    return lambda: torchvision.ops.deform_conv2d(
        x, offset, weight, bias, stride, padding, dilation, mask
    )


def _stage_breakdown(inputs, device: torch.device) -> dict[str, float]:
    """Per-stage timings of the composition. Mirrors
    modulated_deform_conv2d_grid_sample's implementation; if that changes,
    update this breakdown alongside it (numbers-only tool, not on any
    production path)."""
    import torch.nn.functional as F

    x, offset, mask, weight, bias, stride, padding, dilation, groups = inputs
    B, Cin, H, W = x.shape
    Cout, Cin_per_group, kh, kw = weight.shape
    K = kh * kw
    G = offset.shape[1] // (2 * K)
    sy, sx = _pair(stride)
    py, px = _pair(padding)
    dy, dx = _pair(dilation)
    Ho = (H + 2 * py - dy * (kh - 1) - 1) // sy + 1
    Wo = (W + 2 * px - dx * (kw - 1) - 1) // sx + 1

    def build_grid():
        # Constants are cached across calls in production (~240 calls/clip on
        # one geometry), so this measures the steady-state cached-path cost:
        # dict lookup + offset scale/add + stack/cast.
        const_y, const_x = _grid_constants(
            H, W, Ho, Wo, kh, kw, sy, sx, py, px, dy, dx, device
        )
        off = offset.view(B, G, K, 2, Ho, Wo).float()
        gy = off[:, :, :, 0].mul(2.0 / H).add_(const_y)
        gx = off[:, :, :, 1].mul(2.0 / W).add_(const_x)
        return torch.stack((gx, gy), dim=-1).view(B * G, K * Ho, Wo, 2).to(x.dtype)

    grid = build_grid()
    x_g = x.view(B, G, Cin // G, H, W).reshape(B * G, Cin // G, H, W)

    def sample():
        return F.grid_sample(
            x_g, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    sampled = sample().view(B, G, Cin // G, K, Ho, Wo)
    mask_view = mask.view(B, G, 1, K, Ho, Wo).to(sampled.dtype)

    def mask_mul():
        return sampled * mask_view

    cols = mask_mul().reshape(B, Cin * K, Ho, Wo)
    w_flat = weight.reshape(Cout, Cin_per_group * K, 1, 1)

    def conv():
        return F.conv2d(cols, w_flat, bias, groups=groups)

    return {
        "grid build": _median_ms(build_grid, device, warmup=MICRO_WARMUP, iters=MICRO_ITERS),
        "grid_sample": _median_ms(sample, device, warmup=MICRO_WARMUP, iters=MICRO_ITERS),
        "mask multiply": _median_ms(mask_mul, device, warmup=MICRO_WARMUP, iters=MICRO_ITERS),
        "1x1 conv": _median_ms(conv, device, warmup=MICRO_WARMUP, iters=MICRO_ITERS),
    }


def _measure_copy_bandwidth(device: torch.device) -> float:
    """Achieved device copy bandwidth in GB/s (read + write)."""
    n = 64 * 1024 * 1024  # 256 MiB of fp32
    src = torch.empty(n, device=device, dtype=torch.float32).normal_()
    dst = torch.empty_like(src)
    ms = _median_ms(lambda: dst.copy_(src), device, warmup=3, iters=10)
    return (2 * n * 4) / (ms / 1000.0) / 1e9


def _traffic_estimate(sig: tuple, dtype: torch.dtype, bw_gbs: float) -> dict:
    """Itemized memory-traffic estimate: composition today vs a perfectly
    fused kernel. Approximate by construction — treats caches as perfect for
    single-pass reads and counts each materialized intermediate once per
    producer/consumer."""
    x_shape, offset_shape, weight_shape, stride, padding, dilation, mask_shape, groups = sig
    e = torch.tensor([], dtype=dtype).element_size()
    B, Cin, H, W = x_shape
    Cout, Cin_per_group, kh, kw = weight_shape
    K = kh * kw
    G = offset_shape[1] // (2 * K)
    py, px = _pair(padding)
    dy, dx = _pair(dilation)
    sy, sx = _pair(stride)
    Ho = (H + 2 * py - dy * (kh - 1) - 1) // sy + 1
    Wo = (W + 2 * px - dx * (kw - 1) - 1) // sx + 1

    x_b = B * Cin * H * W * e
    off_b = B * G * K * 2 * Ho * Wo * e
    mask_b = B * G * K * Ho * Wo * e
    grid_b = B * G * K * Ho * Wo * 2 * e
    cols_b = B * Cin * K * Ho * Wo * e
    out_b = B * Cout * Ho * Wo * e

    # Composition: grid built (off read, grid write) -> grid_sample (x + grid
    # read, cols write) -> mask mul (cols + mask read, cols write) -> conv
    # (cols read, out write).
    current = off_b + grid_b * 2 + x_b + cols_b * 4 + mask_b + out_b
    # Fused: gather x (imperfect cache -> ~1.5x), read offsets + mask, write out.
    fused = x_b * 1.5 + off_b + mask_b + out_b
    return {
        "current_mb": current / 1e6,
        "fused_mb": fused / 1e6,
        "current_floor_ms": current / 1e9 / bw_gbs * 1000.0,
        "fused_floor_ms": fused / 1e9 / bw_gbs * 1000.0,
        "cols_mb": cols_b / 1e6,
    }


def _compile_probe(inputs, device: torch.device, eager_ms: float) -> None:
    if os.environ.get("JASNA_PROFILE_COMPILE", "1") == "0":
        print("\n[4] torch.compile probe skipped (JASNA_PROFILE_COMPILE=0)")
        return
    print("\n[4] torch.compile probe (inductor; first call compiles, may take minutes)...")
    try:
        compiled = torch.compile(modulated_deform_conv2d_grid_sample)
        x, offset, mask, weight, bias, stride, padding, dilation, groups = inputs
        fn = lambda: compiled(x, offset, mask, weight, bias, stride, padding, dilation, groups)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        compile_s = time.perf_counter() - t0
        ms = _median_ms(fn, device, warmup=MICRO_WARMUP, iters=MICRO_ITERS)
        print(f"  compiled composition   {ms:8.3f} ms/call  (eager {eager_ms:.3f} ms; "
              f"{eager_ms / ms:.2f}x; compile took {compile_s:.0f}s)")
    except Exception as exc:  # noqa: BLE001 - report-and-continue diagnostics tool
        print(f"  torch.compile failed on this backend: {exc}")


def benchmark_deform_profile(
    *,
    device: torch.device,
    fp16: bool,
    restoration_model_path: Path | None,
    **_: object,
) -> None:
    from jasna.restorer.basicvsrpp_mosaic_restorer import (
        TORCH_COMPILE_ENV,
        BasicvsrppMosaicRestorer,
    )

    print(f"\n=== deform_conv2d profile on {device} ({device_name(device)}), "
          f"fp16={fp16} ===")
    if restoration_model_path is None or not restoration_model_path.resolve().exists():
        print("Restoration model weights not found; cannot run the clip pass.")
        return

    # Eager everywhere: TensorRT would bypass the deform dispatch on NVIDIA,
    # and eager is the path comparable to xpu. The production
    # JASNA_TORCH_COMPILE wiring must also stay off — a dynamo-compiled
    # generator would graph-break/recompile on the monkeypatched deform
    # dispatch and corrupt the instrumented timings.
    prev_flag = os.environ.pop(TORCH_COMPILE_ENV, None)
    if prev_flag is not None:
        print(f"  (note: {TORCH_COMPILE_ENV} was set; ignored inside this profile)")
    try:
        restorer = BasicvsrppMosaicRestorer(
            checkpoint_path=str(restoration_model_path.resolve()),
            device=device,
            max_clip_size=CLIP_LENGTH,
            use_tensorrt=False,
            fp16=fp16,
        )
    finally:
        if prev_flag is not None:
            os.environ[TORCH_COMPILE_ENV] = prev_flag
    video = [
        torch.randint(0, 256, (3, SIZE, SIZE), dtype=torch.uint8, device=device)
        for _ in range(CLIP_LENGTH)
    ]
    dtype = torch.float16 if fp16 else torch.float32

    # torchvision's kernel is native on cuda (incl. ROCm); on xpu it would
    # silently round-trip through the CPU, which is not a useful comparison.
    backends = ["grid_sample"] if device.type == "xpu" else ["torchvision", "grid_sample"]

    print(f"\n[1] Instrumented clip pass ({CLIP_LENGTH} frames @ {SIZE}x{SIZE}, "
          f"median of {CLEAN_RUNS} clean / {INSTRUMENTED_RUNS} instrumented runs)")
    results: dict[str, dict] = {}
    with torch.inference_mode():
        for backend in backends:
            r = _profile_backend(backend, restorer, video, device)
            results[backend] = r
            calls = sum(rec.count for rec in r["records"].values())
            share = r["deform_ms"] / r["clean_ms"] * 100.0
            print(f"  {backend:12s} clip={r['clean_ms']:8.1f} ms   "
                  f"deform={r['deform_ms']:7.1f} ms ({share:4.1f}% of clip, {calls} calls)")

    ref = results[backends[-1]]  # grid_sample records exist on every device
    sig = _dominant(ref["records"])
    if sig is None:
        print("No deform calls were recorded — nothing to profile further.")
        return
    x_shape, offset_shape, weight_shape, *_rest = sig
    n_calls = ref["records"][sig].count
    print(f"\n  dominant call: x={list(x_shape)} offset={list(offset_shape)} "
          f"weight={list(weight_shape)}  ({n_calls} calls/clip)")

    print(f"\n[2] Micro-benchmark on the dominant shape "
          f"(median of {MICRO_ITERS} iters)")
    with torch.inference_mode():
        inputs = _make_micro_inputs(sig, device, dtype)
        comp_ms = _median_ms(
            _micro_composition(inputs), device, warmup=MICRO_WARMUP, iters=MICRO_ITERS
        )
        print(f"  grid_sample composition  {comp_ms:8.3f} ms/call")
        native_ms = None
        if device.type != "xpu":
            native_ms = _median_ms(
                _micro_native(inputs), device, warmup=MICRO_WARMUP, iters=MICRO_ITERS
            )
            print(f"  torchvision native       {native_ms:8.3f} ms/call  "
                  f"(composition is {comp_ms / native_ms:.2f}x slower)")
        stages = _stage_breakdown(inputs, device)
        for name, ms in stages.items():
            print(f"    {name:22s} {ms:8.3f} ms")

    print("\n[3] Bandwidth floor (fused-kernel ceiling)")
    with torch.inference_mode():
        bw = _measure_copy_bandwidth(device)
    est = _traffic_estimate(sig, dtype, bw)
    print(f"  measured copy bandwidth  {bw:8.1f} GB/s")
    print(f"  composition traffic     ~{est['current_mb']:7.1f} MB/call "
          f"(cols intermediate {est['cols_mb']:.1f} MB) -> floor {est['current_floor_ms']:.3f} ms")
    print(f"  fused-kernel traffic    ~{est['fused_mb']:7.1f} MB/call "
          f"-> floor {est['fused_floor_ms']:.3f} ms")
    recoverable = max(0.0, comp_ms - est["fused_floor_ms"]) * n_calls
    print(f"  per-clip ceiling: ({comp_ms:.3f} - {est['fused_floor_ms']:.3f}) ms "
          f"x {n_calls} calls = {recoverable:.0f} ms recoverable "
          f"of {ref['clean_ms']:.0f} ms clip ({recoverable / ref['clean_ms'] * 100.0:.1f}%)")

    with torch.inference_mode():
        _compile_probe(inputs, device, comp_ms)

    print("\n=== summary for estimation ===")
    for backend in backends:
        r = results[backend]
        print(f"  {backend:12s} clip {r['clean_ms']:8.1f} ms | deform "
              f"{r['deform_ms']:7.1f} ms | share {r['deform_ms'] / r['clean_ms'] * 100.0:.1f}%")
    if native_ms is not None:
        print(f"  native-vs-composition per-call: {native_ms:.3f} ms vs {comp_ms:.3f} ms "
              f"({comp_ms / native_ms:.2f}x)")
    print(f"  fused ceiling recoverable/clip: ~{recoverable:.0f} ms "
          f"({recoverable / ref['clean_ms'] * 100.0:.1f}% of clip)")
