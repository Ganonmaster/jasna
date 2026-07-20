"""Unit tests for NV12/P010 -> planar RGB uint8 conversion (BT.709/BT.601, limited/full range)."""
import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media import yuv_to_rgb
from jasna.media.rgb_to_p010 import (
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_limited,
)
from jasna.media.yuv_to_rgb import YuvToRgbConverter

CPU = torch.device("cpu")


def _converter(h=4, w=4, space=AvColorspace.ITU709, full_range=False, is_10bit=False):
    return YuvToRgbConverter(h, w, space, full_range, is_10bit, CPU)


def _uniform_planes(y_val, u_val, v_val, h=4, w=4, dtype=torch.uint8):
    y = torch.full((h, w), y_val, dtype=dtype)
    uv = torch.empty((h // 2, w // 2, 2), dtype=dtype)
    uv[:, :, 0] = u_val
    uv[:, :, 1] = v_val
    return y, uv


def test_limited_white_and_black_8bit():
    conv = _converter()
    y, uv = _uniform_planes(235, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.full((3, 4, 4), 255, dtype=torch.uint8))
    y, uv = _uniform_planes(16, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.zeros(3, 4, 4, dtype=torch.uint8))


def test_full_range_white_and_black_8bit():
    conv = _converter(full_range=True)
    y, uv = _uniform_planes(255, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.full((3, 4, 4), 255, dtype=torch.uint8))
    y, uv = _uniform_planes(0, 128, 128)
    assert torch.equal(conv.convert(y, uv), torch.zeros(3, 4, 4, dtype=torch.uint8))


def test_limited_white_10bit_dithered():
    conv = _converter(is_10bit=True)
    y, uv = _uniform_planes(940 << 6, 512 << 6, 512 << 6, dtype=torch.int32)
    out = conv.convert(y.to(torch.float32), uv.to(torch.float32))
    assert torch.equal(out, torch.full((3, 4, 4), 255, dtype=torch.uint8))


def test_bt601_and_bt709_differ_for_colored_input():
    y, uv = _uniform_planes(120, 90, 200)
    out709 = _converter().convert(y, uv)
    out601 = _converter(space=AvColorspace.ITU601).convert(y, uv)
    assert not torch.equal(out709, out601)


def _p010_to_planes(p010: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    as_u16 = p010.to(torch.int32) & 0xFFFF
    y = as_u16[:h].to(torch.float32)
    uv = as_u16[h:].reshape(h // 2, w // 2, 2).to(torch.float32)
    return y, uv


def test_round_trip_uniform_colors_bt709():
    for rgb in [(200, 30, 60), (10, 250, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]:
        img = torch.empty(3, 8, 8, dtype=torch.uint8)
        for c, val in enumerate(rgb):
            img[c] = val
        y, uv = _p010_to_planes(chw_rgb_to_p010_bt709_limited(img), 8, 8)
        out = YuvToRgbConverter(8, 8, AvColorspace.ITU709, False, True, CPU).convert(y, uv)
        assert (out.to(torch.int16) - img.to(torch.int16)).abs().max() <= 2


def test_round_trip_uniform_colors_bt601():
    img = torch.empty(3, 8, 8, dtype=torch.uint8)
    for c, val in enumerate((60, 180, 240)):
        img[c] = val
    y, uv = _p010_to_planes(chw_rgb_to_p010_bt601_limited(img), 8, 8)
    out = YuvToRgbConverter(8, 8, AvColorspace.ITU601, False, True, CPU).convert(y, uv)
    assert (out.to(torch.int16) - img.to(torch.int16)).abs().max() <= 2


def test_dither_is_deterministic():
    conv = _converter(h=8, w=8, is_10bit=True)
    y = torch.randint(64 << 6, 940 << 6, (8, 8), dtype=torch.int32).to(torch.float32)
    uv = torch.randint(64 << 6, 960 << 6, (4, 4, 2), dtype=torch.int32).to(torch.float32)
    assert torch.equal(conv.convert(y, uv), conv.convert(y, uv))


def test_matches_swscale_reference_bt709_limited():
    import av

    rng = np.random.default_rng(7)
    h, w = 16, 16
    # random luma, uniform chroma: chroma upsampling filter differences vanish
    y_plane = rng.integers(16, 236, (h, w), dtype=np.uint8)
    uv_plane = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    uv_plane[:, :, 0] = 90
    uv_plane[:, :, 1] = 190

    nv12 = np.vstack([y_plane, uv_plane.reshape(h // 2, w)])
    frame = av.VideoFrame.from_ndarray(nv12, format="nv12")
    frame.colorspace = 1  # bt709
    frame.color_range = 1  # mpeg/limited
    reference = frame.to_ndarray(format="rgb24").astype(np.int16)  # (H, W, 3)

    out = _converter(h=h, w=w).convert(
        torch.from_numpy(y_plane), torch.from_numpy(uv_plane)
    )
    ours = out.permute(1, 2, 0).numpy().astype(np.int16)
    assert np.abs(ours - reference).max() <= 3


class _FakeCudaTensor(torch.Tensor):
    """CPU tensor that reports itself CUDA-resident (stands in for ROCm planes)."""

    @property
    def is_cuda(self):
        return True


class _FakeDeviceTensor(torch.Tensor):
    """CPU tensor that claims to live on a device the converter was not built for."""

    @property
    def device(self):
        return torch.device("cuda", 0)


def test_gpu_planes_without_kernel_route_to_eager(monkeypatch):
    # AMD/Intel scenario: device planes but no fused kernel must dispatch to
    # the eager path, not raise (regression for the old is_cuda guard).
    monkeypatch.setattr(yuv_to_rgb, "is_nvidia_device", lambda device: False)
    conv = _converter()
    assert conv._cuda_kernel is None
    calls = []
    eager = conv._convert_eager
    monkeypatch.setattr(
        conv, "_convert_eager", lambda y, uv, out: (calls.append(1), eager(y, uv, out))
    )
    y, uv = _uniform_planes(120, 90, 200)
    out = torch.empty((3, 4, 4), dtype=torch.uint8)
    conv.convert_into(y.as_subclass(_FakeCudaTensor), uv.as_subclass(_FakeCudaTensor), out)
    assert calls == [1]


def test_gpu_planes_without_kernel_match_eager_reference(monkeypatch):
    monkeypatch.setattr(yuv_to_rgb, "is_nvidia_device", lambda device: False)
    conv = _converter(h=8, w=8)
    torch.manual_seed(3)
    y = torch.randint(16, 236, (8, 8), dtype=torch.uint8)
    uv = torch.randint(16, 241, (4, 4, 2), dtype=torch.uint8)
    reference = _converter(h=8, w=8).convert(y, uv)
    out = torch.zeros((3, 8, 8), dtype=torch.uint8)
    conv.convert_into(y.as_subclass(_FakeCudaTensor), uv.as_subclass(_FakeCudaTensor), out)
    assert torch.equal(out, reference)


def test_gpu_planes_without_kernel_match_eager_reference_10bit(monkeypatch):
    monkeypatch.setattr(yuv_to_rgb, "is_nvidia_device", lambda device: False)
    conv = _converter(h=8, w=8, is_10bit=True)
    torch.manual_seed(5)
    y = torch.randint(64 << 6, 940 << 6, (8, 8), dtype=torch.int32).to(torch.float32)
    uv = torch.randint(64 << 6, 960 << 6, (4, 4, 2), dtype=torch.int32).to(torch.float32)
    reference = _converter(h=8, w=8, is_10bit=True).convert(y, uv)
    out = torch.zeros((3, 8, 8), dtype=torch.uint8)
    conv.convert_into(y.as_subclass(_FakeCudaTensor), uv.as_subclass(_FakeCudaTensor), out)
    assert torch.equal(out, reference)


def test_eager_converter_rejects_planes_on_other_device():
    conv = _converter()
    y, uv = _uniform_planes(120, 90, 200)
    out = torch.empty((3, 4, 4), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="cannot process planes on"):
        conv.convert_into(y.as_subclass(_FakeDeviceTensor), uv, out)


def test_cuda_kernel_converter_rejects_cpu_planes(monkeypatch):
    # Kernel construction is lazy (no driver call until launch), so forcing the
    # NVIDIA branch is safe on machines without CUDA.
    monkeypatch.setattr(yuv_to_rgb, "is_nvidia_device", lambda device: True)
    conv = _converter()
    assert conv._cuda_kernel is not None
    y, uv = _uniform_planes(120, 90, 200)
    with pytest.raises(RuntimeError, match="cannot process CPU planes"):
        conv.convert_into(y, uv, torch.empty((3, 4, 4), dtype=torch.uint8))
