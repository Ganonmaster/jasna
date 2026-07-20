# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init as init
from torch.nn.modules.utils import _pair, _single
import math

# Offset-independent grid constants per geometry (same pattern as
# jasna.tracking.blending._KERNEL_CACHE). Values are read-only: the per-call
# math must never apply in-place ops to them.
_GRID_CONST_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _grid_constants(
    H: int, W: int, Ho: int, Wo: int, kh: int, kw: int,
    sy: int, sx: int, py: int, px: int, dy: int, dx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached offset-independent part of the sampling grid, in fp32.

    The absolute position p = out * stride - pad + tap * dilation + offset is
    mapped to grid_sample's align_corners=False coords g = (2p + 1)/dim - 1.
    Everything but the offset term is geometry-only, so the normalization is
    folded into the cached constant:

        g = offset * (2/dim) + [(2 * (out*stride - pad + tap*dilation) + 1)/dim - 1]

    Returned shapes broadcast directly against (B, G, K, Ho, Wo) offsets:
    const_y is (1, 1, K, Ho, 1), const_x is (1, 1, K, 1, Wo).
    """
    key = (H, W, Ho, Wo, kh, kw, sy, sx, py, px, dy, dx, str(device), torch.float32)
    cached = _GRID_CONST_CACHE.get(key)
    if cached is None:
        K = kh * kw
        base_y = torch.arange(Ho, device=device, dtype=torch.float32).mul_(sy).sub_(py)
        base_x = torch.arange(Wo, device=device, dtype=torch.float32).mul_(sx).sub_(px)
        tap_y = torch.arange(kh, device=device, dtype=torch.float32).mul_(dy)
        tap_x = torch.arange(kw, device=device, dtype=torch.float32).mul_(dx)
        tap_y = tap_y.view(kh, 1).expand(kh, kw).reshape(K)
        tap_x = tap_x.view(1, kw).expand(kh, kw).reshape(K)
        const_y = (base_y.view(1, Ho) + tap_y.view(K, 1)).mul_(2.0).add_(1.0).div_(H).sub_(1.0)
        const_x = (base_x.view(1, Wo) + tap_x.view(K, 1)).mul_(2.0).add_(1.0).div_(W).sub_(1.0)
        cached = (const_y.view(1, 1, K, Ho, 1), const_x.view(1, 1, K, 1, Wo))
        _GRID_CONST_CACHE[key] = cached
    return cached


def modulated_deform_conv2d_grid_sample(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Modulated deformable conv v2 built from grid_sample + a 1x1 conv.

    Numerically equivalent to ``torchvision.ops.deform_conv2d`` (same offset
    and mask layout, bilinear sampling with zero padding), but composed only
    of ops with native XPU kernels — torchvision ships no XPU deform kernel
    (pytorch/vision RFC #8679), so calling it on xpu tensors silently round
    trips through the CPU.

    All K taps are folded into one grid_sample call (taps stacked along the
    output height) and the kernel reduction becomes a 1x1 convolution over
    the (C_in * K) sampled columns.
    """
    B, Cin, H, W = x.shape
    Cout, Cin_per_group, kh, kw = weight.shape
    K = kh * kw
    G = offset.shape[1] // (2 * K)  # deformable groups
    sy, sx = _pair(stride)
    py, px = _pair(padding)
    dy, dx = _pair(dilation)
    Ho = (H + 2 * py - dy * (kh - 1) - 1) // sy + 1
    Wo = (W + 2 * px - dx * (kw - 1) - 1) // sx + 1

    # Sampling grid in fp32: normalized position = offset * (2/dim) + cached
    # constant (base + tap + align_corners=False mapping folded in, see
    # _grid_constants). Offset channels are group-major, tap-major, (dy, dx)
    # interleaved — torchvision's layout.
    const_y, const_x = _grid_constants(
        H, W, Ho, Wo, kh, kw, sy, sx, py, px, dy, dx, x.device
    )
    off = offset.view(B, G, K, 2, Ho, Wo).float()
    # .mul() allocates fresh tensors; add_ writes into those, never into the
    # cached constants.
    gy = off[:, :, :, 0].mul(2.0 / H).add_(const_y)
    gx = off[:, :, :, 1].mul(2.0 / W).add_(const_x)
    grid = torch.stack((gx, gy), dim=-1).view(B * G, K * Ho, Wo, 2).to(x.dtype)

    x_g = x.view(B, G, Cin // G, H, W).reshape(B * G, Cin // G, H, W)
    sampled = F.grid_sample(x_g, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    sampled = sampled.view(B, G, Cin // G, K, Ho, Wo)
    sampled = sampled * mask.view(B, G, 1, K, Ho, Wo).to(sampled.dtype)

    # (B, G, C/G, K, Ho, Wo) -> (B, Cin*K, Ho, Wo): channel-major, tap-minor —
    # matching weight.view(Cout, Cin_per_group * K), so the kernel reduction
    # is a plain 1x1 conv (conv groups split contiguously along Cin).
    cols = sampled.reshape(B, Cin * K, Ho, Wo)
    return F.conv2d(cols, weight.reshape(Cout, Cin_per_group * K, 1, 1), bias, groups=groups)


def deform_conv2d_dispatch(
    x: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    mask: torch.Tensor,
    groups: int = 1,
) -> torch.Tensor:
    """torchvision's kernel where it is native (cuda/rocm/cpu), grid_sample on xpu."""
    from jasna.accelerator import deform_conv2d_backend

    if deform_conv2d_backend(x.device) == "grid_sample":
        return modulated_deform_conv2d_grid_sample(
            x, offset, mask, weight, bias, stride, padding, dilation, groups
        )
    return torchvision.ops.deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)


class ModulatedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, offset, mask):
        pass