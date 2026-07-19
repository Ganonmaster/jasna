# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Whole-model torch.compile probe for BasicVSR++ restoration.

On NVIDIA, restoration runs TensorRT-compiled (3.09x over eager on a 4090:
148.7 vs 459.5 ms per 60-frame clip); on Intel/AMD it runs eager. This probe
measures how much of that compiled-vs-eager gap torch.compile (inductor)
recovers on the current device: eager raw_process baseline, then the model's
inference generator swapped for a torch.compile'd wrapper, same clip re-timed.

Distinct from the deform profile's op-level compile probe — whole-graph
compilation fuses across ALL ops (flow warp, conv stacks, upsampling), which
is where eager mode loses to TensorRT.

Opt-in via `jasna --benchmark --benchmark-filter compile`. On xpu, inductor
JIT-compiles through triton-xpu, which needs the Level Zero dev headers
(`apt install libze-dev`) at runtime. JASNA_COMPILE_MODE selects a
torch.compile mode (e.g. max-autotune); default is torch.compile's default.
"""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch

from jasna.accelerator import device_name, synchronize

CLIP_LENGTH = 60
SIZE = 256
RUNS = 7


def _raw_process_median_ms(restorer, video, device, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        restorer.raw_process(video)
    synchronize(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        restorer.raw_process(video)
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000.0


def benchmark_compile_profile(
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

    print(f"\n=== whole-model torch.compile profile on {device} "
          f"({device_name(device)}), fp16={fp16} ===")
    if restoration_model_path is None or not restoration_model_path.resolve().exists():
        print("Restoration model weights not found; skipping.")
        return

    # The production JASNA_TORCH_COMPILE wiring must stay OFF inside this probe:
    # with it active, the "eager" baseline would silently run the production-
    # compiled generator (60 frames == max_clip_size) and the probe's own swap
    # would be overwritten every call — both numbers ~1.0x and wrong.
    prev_flag = os.environ.pop(TORCH_COMPILE_ENV, None)
    if prev_flag is not None:
        print(f"  (note: {TORCH_COMPILE_ENV} was set; ignored inside this probe)")
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

    with torch.inference_mode():
        eager_ms = _raw_process_median_ms(restorer, video, device, warmup=2, iters=RUNS)
    print(f"  eager raw_process        {eager_ms:8.1f} ms/clip "
          f"({CLIP_LENGTH} frames @ {SIZE}x{SIZE}, median of {RUNS})")

    # forward_tensor runs generator_ema at inference when the EMA copy exists;
    # compile whichever module it will actually call.
    model = restorer.model
    attr = "generator_ema" if getattr(model, "generator_ema", None) is not None else "generator"
    generator = getattr(model, attr)

    mode = os.environ.get("JASNA_COMPILE_MODE") or None
    print(f"  compiling model.{attr} with torch.compile"
          f"{f'(mode={mode!r})' if mode else ''} — first clip compiles, may take minutes...")
    try:
        setattr(model, attr, torch.compile(generator, mode=mode))
        with torch.inference_mode():
            t0 = time.perf_counter()
            restorer.raw_process(video)
            synchronize(device)
            compile_s = time.perf_counter() - t0
            compiled_ms = _raw_process_median_ms(
                restorer, video, device, warmup=1, iters=RUNS
            )
        print(f"  compiled raw_process     {compiled_ms:8.1f} ms/clip "
              f"(compile took {compile_s:.0f}s)")
        print(f"\n=== summary: eager {eager_ms:.1f} ms vs compiled {compiled_ms:.1f} ms "
              f"-> {eager_ms / compiled_ms:.2f}x ===")
        print("  (4090 reference: TensorRT achieves 3.09x over eager — 459.5 -> 148.7 ms.)")
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        print(f"  inductor cache: {cache_dir or 'default (/tmp/torchinductor_<user> — cleared on reboot)'}")
        print("  Compiled kernels persist in that cache: RE-RUN THIS COMMAND to measure")
        print("  the warm-start compile time — that, not the first-run time, is the")
        print("  recurring cost of a production wiring. Set TORCHINDUCTOR_CACHE_DIR to a")
        print("  persistent path to keep the cache across reboots.")
        print("  Caveat: production clips vary in length; variable T triggers dynamo")
        print("  recompiles unless clips are padded to a fixed length (or T is exported")
        print("  as a dynamic dim via AOTInductor, mirroring TRT's dynamic batch).")
    except Exception as exc:  # noqa: BLE001 - report-and-continue diagnostics tool
        print(f"  torch.compile failed on this backend: {exc}")
    finally:
        setattr(model, attr, generator)
