"""Parity of the grid_sample deform-conv composition vs torchvision's kernel.

The grid_sample path is what BasicVSR++ runs on xpu; torchvision's native
kernel is the reference. They must agree to float tolerance on CPU for the
xpu path to be trustworthy.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchvision

import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from jasna.models.basicvsrpp.deformconv import (
    _GRID_CONST_CACHE,
    deform_conv2d_dispatch,
    modulated_deform_conv2d_grid_sample,
)


def _random_case(
    *,
    B=2,
    Cin=8,
    Cout=6,
    H=10,
    W=12,
    kh=3,
    kw=3,
    G=2,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
    offset_scale=2.5,
    seed=0,
):
    torch.manual_seed(seed)
    sy, sx = (stride, stride) if isinstance(stride, int) else stride
    py, px = (padding, padding) if isinstance(padding, int) else padding
    dy, dx = (dilation, dilation) if isinstance(dilation, int) else dilation
    Ho = (H + 2 * py - dy * (kh - 1) - 1) // sy + 1
    Wo = (W + 2 * px - dx * (kw - 1) - 1) // sx + 1
    K = kh * kw

    x = torch.randn(B, Cin, H, W)
    offset = torch.randn(B, 2 * G * K, Ho, Wo) * offset_scale
    mask = torch.rand(B, G * K, Ho, Wo)
    weight = torch.randn(Cout, Cin // groups, kh, kw) * 0.2
    b = torch.randn(Cout) * 0.1 if bias else None
    return x, offset, mask, weight, b, stride, padding, dilation, groups


CASES = [
    dict(),  # generic
    dict(B=1, Cin=64, Cout=64, H=16, W=16, G=16, seed=1),  # BasicVSR++ shape (small spatial)
    dict(stride=2, seed=2),
    dict(dilation=2, padding=2, seed=3),
    dict(padding=0, seed=4),
    dict(groups=2, Cin=8, Cout=8, seed=5),
    dict(bias=False, seed=6),
    dict(offset_scale=25.0, seed=7),  # many samples land out of bounds
]


@pytest.mark.parametrize("case", CASES)
def test_parity_with_torchvision(case):
    x, offset, mask, weight, bias, stride, padding, dilation, groups = _random_case(**case)
    ref = torchvision.ops.deform_conv2d(
        x, offset, weight, bias, stride, padding, dilation, mask
    )
    got = modulated_deform_conv2d_grid_sample(
        x, offset, mask, weight, bias, stride, padding, dilation, groups
    )
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def _uncached_reference(x, offset, mask, weight, bias, stride, padding, dilation, groups):
    """The pre-cache composition, verbatim: grid built from scratch every call
    with the un-folded normalization. Proves the cached/folded constants in
    modulated_deform_conv2d_grid_sample reproduce the original math."""
    B, Cin, H, W = x.shape
    Cout, Cin_per_group, kh, kw = weight.shape
    K = kh * kw
    G = offset.shape[1] // (2 * K)
    sy, sx = _pair(stride)
    py, px = _pair(padding)
    dy, dx = _pair(dilation)
    Ho = (H + 2 * py - dy * (kh - 1) - 1) // sy + 1
    Wo = (W + 2 * px - dx * (kw - 1) - 1) // sx + 1

    device = x.device
    base_y = torch.arange(Ho, device=device, dtype=torch.float32).mul_(sy).sub_(py)
    base_x = torch.arange(Wo, device=device, dtype=torch.float32).mul_(sx).sub_(px)
    tap_y = torch.arange(kh, device=device, dtype=torch.float32).mul_(dy)
    tap_x = torch.arange(kw, device=device, dtype=torch.float32).mul_(dx)
    tap_y = tap_y.view(kh, 1).expand(kh, kw).reshape(K)
    tap_x = tap_x.view(1, kw).expand(kh, kw).reshape(K)

    off = offset.view(B, G, K, 2, Ho, Wo).float()
    pos_y = off[:, :, :, 0] + base_y.view(1, 1, 1, Ho, 1) + tap_y.view(1, 1, K, 1, 1)
    pos_x = off[:, :, :, 1] + base_x.view(1, 1, 1, 1, Wo) + tap_x.view(1, 1, K, 1, 1)
    gx = pos_x.mul_(2.0).add_(1.0).div_(W).sub_(1.0)
    gy = pos_y.mul_(2.0).add_(1.0).div_(H).sub_(1.0)
    grid = torch.stack((gx, gy), dim=-1).view(B * G, K * Ho, Wo, 2).to(x.dtype)

    x_g = x.view(B, G, Cin // G, H, W).reshape(B * G, Cin // G, H, W)
    sampled = F.grid_sample(x_g, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    sampled = sampled.view(B, G, Cin // G, K, Ho, Wo)
    sampled = sampled * mask.view(B, G, 1, K, Ho, Wo).to(sampled.dtype)
    cols = sampled.reshape(B, Cin * K, Ho, Wo)
    return F.conv2d(cols, weight.reshape(Cout, Cin_per_group * K, 1, 1), bias, groups=groups)


@pytest.mark.parametrize("case", CASES)
def test_cached_constants_match_uncached_math(case):
    """Folding base + tap + normalization into cached constants only
    reassociates float adds; agreement must be far tighter than the
    torchvision parity tolerance."""
    args = _random_case(**case)
    got = modulated_deform_conv2d_grid_sample(*args)
    ref = _uncached_reference(*args)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_grid_constant_cache_hit_and_miss():
    _GRID_CONST_CACHE.clear()
    args = _random_case(seed=10)
    out1 = modulated_deform_conv2d_grid_sample(*args)
    assert len(_GRID_CONST_CACHE) == 1
    const_y, const_x = next(iter(_GRID_CONST_CACHE.values()))
    snap_y, snap_x = const_y.clone(), const_x.clone()

    out2 = modulated_deform_conv2d_grid_sample(*args)  # identical geometry: hit
    assert len(_GRID_CONST_CACHE) == 1
    hit_y, hit_x = next(iter(_GRID_CONST_CACHE.values()))
    assert hit_y is const_y and hit_x is const_x
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
    # Cached constants must never be mutated by the per-call math.
    torch.testing.assert_close(const_y, snap_y, atol=0.0, rtol=0.0)
    torch.testing.assert_close(const_x, snap_x, atol=0.0, rtol=0.0)

    modulated_deform_conv2d_grid_sample(*_random_case(stride=2, seed=11))  # miss
    assert len(_GRID_CONST_CACHE) == 2


def test_dispatch_env_override(monkeypatch):
    x, offset, mask, weight, bias, stride, padding, dilation, groups = _random_case(seed=8)
    monkeypatch.setenv("JASNA_DEFORM_BACKEND", "grid_sample")
    got = deform_conv2d_dispatch(x, offset, weight, bias, stride, padding, dilation, mask, groups)
    monkeypatch.setenv("JASNA_DEFORM_BACKEND", "torchvision")
    ref = deform_conv2d_dispatch(x, offset, weight, bias, stride, padding, dilation, mask, groups)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_backend_selection():
    from jasna.accelerator import deform_conv2d_backend

    assert deform_conv2d_backend(torch.device("cpu")) == "torchvision"
    assert deform_conv2d_backend(torch.device("cuda:0")) == "torchvision"
    assert deform_conv2d_backend(torch.device("xpu:0")) == "grid_sample"


_WEIGHTS = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")


@pytest.mark.skipif(not _WEIGHTS.exists(), reason="restoration weights not present")
def test_basicvsrpp_end_to_end_parity(monkeypatch):
    """Whole-net A/B on CPU: grid_sample vs torchvision deform backends."""
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer

    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(_WEIGHTS),
        device=torch.device("cpu"),
        max_clip_size=3,
        use_tensorrt=False,
        fp16=False,
    )
    torch.manual_seed(0)
    video = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8) for _ in range(3)]

    monkeypatch.setenv("JASNA_DEFORM_BACKEND", "torchvision")
    ref = restorer.raw_process([f.clone() for f in video])
    monkeypatch.setenv("JASNA_DEFORM_BACKEND", "grid_sample")
    got = restorer.raw_process([f.clone() for f in video])

    # Outputs in [0, 1]; agreement well below visible quantization (1/255).
    max_delta = (got - ref).abs().max().item()
    assert max_delta < 2e-3, f"max output delta {max_delta}"
