"""
GPU equirectangular <-> equidistant fisheye remap for VR stereo frames.

Operates on the in-pipeline decoded frame tensors produced by
NvidiaVideoReader.frames(): (N, 3, H, W) uint8, RGB planar, on CUDA.

The remap geometry is static, so a normalized [-1, 1] sampling grid is built once
and reused with torch.nn.functional.grid_sample (same pattern as the GPU color-LUT
applier in jasna/media/lut.py). Bilinear sampling matches ffmpeg v360 interp=line.

Three knobs cover the formats we see (all of them half-equirect VR180 content):

  layout    sbs | tb | mono   -- the stereo split axis. SBS splits the frame on X,
            TB on Y; mono is a single eye.
  eye_hfov  the horizontal degrees the *eye* spans. 180 when the 180deg content fills
            the eye (e.g. VRKM, a square eye); 360 when the 180deg content is centred
            in a 360deg-wide canvas with black side pillars (e.g. 3DSVR/VOVS). This is
            the only thing that differs between "tight" and "padded" equirect: it sets
            the longitude->source-x scale.
  fov_deg   the fisheye lens FOV for the equidistant mapping (180 here).

The fisheye disc is inscribed by the eye HEIGHT and centred, so a 2:1 (padded) eye
gets a height-diameter disc with black side pillars, while a 1:1 (tight) eye gets a
disc inscribed in the square (black corners) -- the original VRKM behaviour.

Grid math verified against ffmpeg v360
(input=hequirect:output=fisheye:in_stereo=sbs:out_stereo=sbs:ih_fov=180:iv_fov=180
for the SBS-tight case). The grid is kept in float32 -- fp16 grid coordinates would
round to ~2 px error at 8K -- so frames are sampled in float32 and returned as uint8.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _eye_split(oy, ox, layout: str):
    """Map full-frame normalized coords to (first_eye_mask, eye-local u, eye-local v,
    eye width fraction, eye height fraction)."""
    if layout == "tb":
        first = oy < 0.5
        eu = ox
        ev = torch.where(first, oy * 2.0, (oy - 0.5) * 2.0)
        return first, eu, ev, 1.0, 0.5
    if layout == "sbs":
        first = ox < 0.5
        eu = torch.where(first, ox * 2.0, (ox - 0.5) * 2.0)
        ev = oy
        return first, eu, ev, 0.5, 1.0
    # mono
    first = torch.ones_like(ox, dtype=torch.bool)
    return first, ox, oy, 1.0, 1.0


def _eye_recombine(first, su, sv, layout: str):
    """Map eye-local source coords + eye mask back to full-frame source coords [0,1]."""
    if layout == "tb":
        sx = su
        sy = torch.where(first, sv * 0.5, 0.5 + sv * 0.5)
    elif layout == "sbs":
        sx = torch.where(first, su * 0.5, 0.5 + su * 0.5)
        sy = sv
    else:  # mono
        sx, sy = su, sv
    return sx, sy


def _mesh(W: int, H: int):
    ys = (torch.arange(H, dtype=torch.float64) + 0.5) / H
    xs = (torch.arange(W, dtype=torch.float64) + 0.5) / W
    return torch.meshgrid(ys, xs, indexing="ij")  # oy, ox  (H, W)


class FisheyeRemapper:
    """Equirectangular (half-equirect, tight or padded) -> equidistant fisheye."""

    def __init__(self, width: int, height: int, device: torch.device,
                 layout: str = "sbs", eye_hfov: float = 180.0,
                 fov_deg: float = 180.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self.layout = layout
        self._grid = self._build_grid(self.width, self.height, layout, eye_hfov, fov_deg).to(device)

    @staticmethod
    def _build_grid(W: int, H: int, layout: str, eye_hfov: float, fov_deg: float) -> torch.Tensor:
        half_fov = math.radians(fov_deg) * 0.5      # 90 deg at the disc edge (180 lens)
        eye_hfov_rad = math.radians(eye_hfov)        # eye horizontal span (pi tight / 2pi padded)
        oy, ox = _mesh(W, H)

        first, eu, ev, wfrac, hfrac = _eye_split(oy, ox, layout)
        aspect = (W * wfrac) / (H * hfrac)           # eye W/H; disc radius 1 == eye-height/2

        fx = (eu - 0.5) * 2.0 * aspect               # centred fisheye coords (disc r<=1)
        fy = (ev - 0.5) * 2.0
        r = torch.sqrt(fx * fx + fy * fy)

        theta = r * half_fov                         # equidistant: angle ~ radius
        phi = torch.atan2(fy, fx)
        sinT = torch.sin(theta)
        X = sinT * torch.cos(phi)                    # right
        Y = sinT * torch.sin(phi)                    # down
        Z = torch.cos(theta)                         # forward

        lon = torch.atan2(X, Z)                      # [-pi/2, pi/2] over the forward hemisphere
        lat = torch.asin(torch.clamp(Y, -1.0, 1.0))
        su = lon / eye_hfov_rad + 0.5                # eye-local source x [0,1]
        sv = lat / math.pi + 0.5                     # eye-local source y [0,1]

        sx, sy = _eye_recombine(first, su, sv, layout)
        gx = sx * 2.0 - 1.0
        gy = sy * 2.0 - 1.0

        # outside the fisheye disc: sample out of range -> padding_mode='zeros' -> black
        outside = r > 1.0
        gx = torch.where(outside, torch.full_like(gx, 2.0), gx)
        gy = torch.where(outside, torch.full_like(gy, 2.0), gy)

        return torch.stack((gx, gy), dim=-1).unsqueeze(0).to(torch.float32)  # (1,H,W,2)

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (3,H,W) or (N,3,H,W) uint8 CUDA RGB -> same shape/dtype, remapped.

        Sampled one frame at a time: at 8K an fp32 grid_sample transient is ~0.8 GB
        per frame, so a whole batch at once would spike VRAM on a tool that runs
        restoration concurrently. The per-frame loop caps the transient; grid_sample
        is fast enough that the extra launches are negligible.
        """
        single = frames.ndim == 3
        if single:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected (3,H,W) or (N,3,H,W), got {tuple(frames.shape)}")
        if frames.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"frame size {tuple(frames.shape[-2:])} != grid {(self.height, self.width)}")
        out = torch.empty_like(frames)
        for i in range(frames.shape[0]):
            sampled = F.grid_sample(frames[i:i + 1].float(), self._grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            out[i] = sampled[0].round_().clamp_(0, 255).to(torch.uint8)
        return out[0] if single else out


class InverseFisheyeRemapper:
    """Equidistant fisheye -> equirectangular (half-equirect, tight or padded): the
    inverse of FisheyeRemapper. Used to reproject the restored frame back into the
    source projection so the exported file matches the original VR180 ("reproject to
    source").

    This is a full-frame round trip (every pixel resampled once more). Measured fidelity
    on 8K SBS-tight is high: SSIM ~0.996 / PSNR ~42 dB overall, ~0.998-0.999 in the
    central viewing area; loss is concentrated at the periphery where fisheye and equirect
    sampling densities differ most.

    NOTE (option 2, not implemented): a marginally higher-quality variant would keep the
    untouched background pixel-perfect by inverse-remapping only the restored regions (via
    the restoration mask) and compositing them onto the original equirect frame, instead of
    round-tripping the whole frame. The gain is small (background SSIM ~0.996 -> ~1.0) and
    was deemed not worth the extra plumbing for now. If pursued, do it at the crop level in
    blend_buffer rather than here.

    Bilinear sampling, matching the forward remap.
    """

    def __init__(self, width: int, height: int, device: torch.device,
                 layout: str = "sbs", eye_hfov: float = 180.0,
                 fov_deg: float = 180.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self.layout = layout
        self._grid = self._build_grid(self.width, self.height, layout, eye_hfov, fov_deg).to(device)

    @staticmethod
    def _build_grid(W: int, H: int, layout: str, eye_hfov: float, fov_deg: float) -> torch.Tensor:
        half_fov = math.radians(fov_deg) * 0.5
        eye_hfov_rad = math.radians(eye_hfov)
        oy, ox = _mesh(W, H)

        first, eu_out, ev_out, wfrac, hfrac = _eye_split(oy, ox, layout)
        aspect = (W * wfrac) / (H * hfrac)
        # eu_out/ev_out are the eye-local *equirect* output coords here.
        lon = (eu_out - 0.5) * eye_hfov_rad          # [-eye_hfov/2, eye_hfov/2]
        lat = (ev_out - 0.5) * math.pi
        X = torch.cos(lat) * torch.sin(lon)
        Y = torch.sin(lat)
        Z = torch.cos(lat) * torch.cos(lon)

        theta = torch.acos(torch.clamp(Z, -1.0, 1.0)) # angle from forward
        phi = torch.atan2(Y, X)
        r = theta / half_fov                          # equidistant: radius ~ angle
        fx = r * torch.cos(phi)
        fy = r * torch.sin(phi)
        eu = fx / (2.0 * aspect) + 0.5                # eye-local fisheye source [0,1]
        ev = fy / 2.0 + 0.5

        sx, sy = _eye_recombine(first, eu, ev, layout)
        gx = sx * 2.0 - 1.0
        gy = sy * 2.0 - 1.0

        # beyond the captured hemisphere (e.g. the padded pillars, lon>90deg) -> black
        outside = r > 1.0
        gx = torch.where(outside, torch.full_like(gx, 2.0), gx)
        gy = torch.where(outside, torch.full_like(gy, 2.0), gy)

        return torch.stack((gx, gy), dim=-1).unsqueeze(0).to(torch.float32)  # (1,H,W,2)

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Accepts (3,H,W) or (N,3,H,W) uint8 CUDA RGB; returns the same shape."""
        single = frames.ndim == 3
        if single:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected (3,H,W) or (N,3,H,W), got {tuple(frames.shape)}")
        if frames.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"frame size {tuple(frames.shape[-2:])} != grid {(self.height, self.width)}")
        out = torch.empty_like(frames)
        for i in range(frames.shape[0]):
            sampled = F.grid_sample(frames[i:i + 1].float(), self._grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            out[i] = sampled[0].round_().clamp_(0, 255).to(torch.uint8)
        return out[0] if single else out
