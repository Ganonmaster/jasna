"""Smoke tests for the deform profiling benchmark (tiny shapes, cpu).

The full profile needs model weights + a GPU; these cover the measurement
helpers so the tool cannot silently rot: micro inputs match the recorded
signature, composition/native micro-benchmarks agree numerically, the stage
breakdown mirrors the composition, and the traffic estimate is self-consistent.
"""
from __future__ import annotations

import torch
import torchvision

from jasna.benchmark import BENCHMARKS, PROFILES
from jasna.benchmark.deform_profile import (
    _make_micro_inputs,
    _median_ms,
    _micro_composition,
    _micro_native,
    _signature,
    _stage_breakdown,
    _traffic_estimate,
)

# B=1, Cin=8, 12x12, K=9, G=2, Cout=8 — small enough for instant cpu runs.
_SIG = _signature(
    torch.empty(1, 8, 12, 12),
    torch.empty(1, 2 * 9 * 2, 12, 12),
    torch.empty(8, 8, 3, 3),
    1,  # stride
    1,  # padding
    1,  # dilation
    torch.empty(1, 2 * 9, 12, 12),
    1,  # groups
)
_DEVICE = torch.device("cpu")


def test_micro_inputs_match_signature():
    x, offset, mask, weight, bias, stride, padding, dilation, groups = (
        _make_micro_inputs(_SIG, _DEVICE, torch.float32)
    )
    assert _signature(x, offset, weight, stride, padding, dilation, mask, groups) == _SIG


def test_micro_composition_matches_native_numerically():
    torch.manual_seed(0)
    inputs = _make_micro_inputs(_SIG, _DEVICE, torch.float32)
    out_comp = _micro_composition(inputs)()
    out_native = _micro_native(inputs)()
    assert out_comp.shape == out_native.shape
    assert torch.allclose(out_comp, out_native, atol=1e-4, rtol=1e-4)


def test_median_ms_and_stage_breakdown_run():
    inputs = _make_micro_inputs(_SIG, _DEVICE, torch.float32)
    ms = _median_ms(_micro_composition(inputs), _DEVICE, warmup=1, iters=3)
    assert ms > 0
    stages = _stage_breakdown(inputs, _DEVICE)
    assert set(stages) == {"grid build", "grid_sample", "mask multiply", "1x1 conv"}
    assert all(v > 0 for v in stages.values())


def test_traffic_estimate_fused_below_current():
    est = _traffic_estimate(_SIG, torch.float32, bw_gbs=100.0)
    assert 0 < est["fused_mb"] < est["current_mb"]
    assert 0 < est["fused_floor_ms"] < est["current_floor_ms"]
    assert est["cols_mb"] > 0


def test_profile_is_not_in_the_default_benchmark_run():
    names = [fn.__name__ for fn in BENCHMARKS]
    assert "benchmark_deform_profile" not in names
    assert any(fn.__name__ == "benchmark_deform_profile" for fn in PROFILES)
