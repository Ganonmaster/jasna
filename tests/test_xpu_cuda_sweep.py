"""Guards against raw torch.cuda.* creeping back onto the xpu runtime path.

Intel Arc is a genuine third device type (device.type=='xpu',
torch.cuda.is_available()==False), so ANY torch.cuda.* call reached on the xpu
pipeline crashes with "Torch not compiled with CUDA enabled". The runtime hot
path and benchmark tooling must therefore route every accelerator op through
jasna.accelerator's vendor-neutral free functions. These tests fail if a future
edit reintroduces a raw torch.cuda.* on those modules, and lock in the
VramOffloader offload-device fix (which would otherwise silently disable spilling
on xpu and OOM the 16 GB B50).
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import torch

import jasna.vram_offloader as vram_offloader_mod
from jasna.vram_offloader import VramOffloader

# Modules that execute on the xpu batch/streaming/benchmark path. Every raw
# torch.cuda.* here was verified reachable (or reachable once the dominating
# crash is removed) by the xpu-cuda audit.
_CUDA_FREE_MODULES = [
    "jasna/pipeline.py",
    "jasna/pipeline_threads.py",
    "jasna/vram_offloader.py",
    "jasna/streaming_pipeline.py",
    "jasna/benchmark/__init__.py",
    "jasna/benchmark/basicvsrpp_restoration.py",
    "jasna/benchmark/harness.py",
    "jasna/benchmark/lada_yolo_detection_speed.py",
    "jasna/benchmark/rfdetr_detection_speed.py",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_runtime_modules_have_no_raw_torch_cuda():
    offenders = []
    for rel in _CUDA_FREE_MODULES:
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), 1):
            if "torch.cuda." in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "raw torch.cuda.* on the xpu runtime path (use jasna.accelerator instead):\n"
        + "\n".join(offenders)
    )


def _make_offloader(device: torch.device) -> VramOffloader:
    return VramOffloader(
        device=device,
        blend_buffer=MagicMock(),
        crop_buffers={},
        crop_lock=threading.Lock(),
    )


def test_offload_device_type_tracks_device_not_hardcoded_cuda(monkeypatch):
    # mem_get_info is the accelerator seam; return a fixed total so __init__ does
    # not touch real hardware. The bug this guards: _offload_device_type was
    # hardcoded "cuda", so on xpu the device.type check in _offload never matched
    # and nothing ever spilled.
    monkeypatch.setattr(
        vram_offloader_mod, "mem_get_info", lambda _device: (0, 16 * 1024**3)
    )
    for spec, expected_type in [("xpu:0", "xpu"), ("cuda:0", "cuda"), ("cpu", "cpu")]:
        offloader = _make_offloader(torch.device(spec))
        assert offloader._offload_device_type == expected_type


def test_threshold_uses_accelerator_mem_get_info(monkeypatch):
    total = 16 * 1024**3
    safetynet = 750 * 1024 * 1024
    monkeypatch.setattr(vram_offloader_mod, "mem_get_info", lambda _device: (0, total))
    offloader = _make_offloader(torch.device("xpu:0"))
    assert offloader._threshold == total - safetynet


def test_offload_matches_frames_on_the_offloader_device(monkeypatch):
    # Drive _offload with a CPU-resident result and a device whose type matches,
    # proving the spill path keys off _offload_device_type (== device.type), not a
    # literal "cuda". Using device.type=="cpu" lets us build real tensors here.
    monkeypatch.setattr(vram_offloader_mod, "mem_get_info", lambda _device: (0, 1 << 30))
    offloader = _make_offloader(torch.device("cpu"))

    frame = torch.ones(4, 4)
    result = MagicMock()
    result.start_frame = 0
    result.restored_frames = [frame]
    result.masks = []
    offloader._blend_buffer.offloadable_results.return_value = [result]

    freed = offloader._offload(bytes_to_free=1)
    assert freed == frame.nelement() * frame.element_size()
