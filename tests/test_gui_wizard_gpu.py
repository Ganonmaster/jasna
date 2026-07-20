from __future__ import annotations

import sys
import types

import pytest

# Bind jasna.accelerator to the real torch before any test installs a fake
# sys.modules["torch"]: check_supported_gpu re-imports it lazily, and its
# module-level torch reference must not capture the SimpleNamespace fake.
import jasna.accelerator  # noqa: F401
from jasna.gui.wizard import FirstRunWizard


class _FakeTorchDevice:
    """Minimal stand-in for torch.device: exposes .index parsed from the spec."""

    def __init__(self, spec: str) -> None:
        spec = str(spec)
        self.index = int(spec.split(":", 1)[1]) if ":" in spec else None


def _make_fake_torch(
    is_available: bool,
    get_device_capability: tuple[int, int] | None = None,
    get_device_name: str = "Fake GPU",
):
    def _get_device_capability(device: int = 0) -> tuple[int, int]:
        return get_device_capability if get_device_capability is not None else (7, 5)

    def _get_device_name(device: int = 0) -> str:
        return get_device_name

    return types.SimpleNamespace(
        device=_FakeTorchDevice,
        cuda=types.SimpleNamespace(
            is_available=lambda: is_available,
            device_count=lambda: 1,
            get_device_capability=_get_device_capability,
            get_device_name=_get_device_name,
        ),
    )


def _call_check_gpu(monkeypatch, fake_torch) -> tuple[bool, str]:
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    stub = types.SimpleNamespace()
    return FirstRunWizard._check_gpu(stub)


def test_check_gpu_passes_when_cuda_available_and_compute_75(monkeypatch):
    fake_torch = _make_fake_torch(True, get_device_capability=(7, 5))
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is True
    assert msg == "Fake GPU"


def test_check_gpu_passes_when_compute_80(monkeypatch):
    fake_torch = _make_fake_torch(True, get_device_capability=(8, 0))
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is True


def test_check_gpu_passes_when_compute_121(monkeypatch):
    fake_torch = _make_fake_torch(True, get_device_capability=(12, 1))
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is True


def test_check_gpu_passes_when_compute_121a(monkeypatch):
    """compute_121a (SM 12.1) is reported by PyTorch as (12, 1); should pass."""
    fake_torch = _make_fake_torch(True, get_device_capability=(12, 1))
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is True


def test_check_gpu_fails_when_compute_70(monkeypatch):
    fake_torch = _make_fake_torch(True, get_device_capability=(7, 0))
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is False
    assert "7.5" in msg
    assert "7.0" in msg


def test_check_gpu_fails_when_compute_61(monkeypatch):
    fake_torch = _make_fake_torch(True, get_device_capability=(6, 1))
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is False
    assert "7.5" in msg
    assert "6.1" in msg


def test_check_gpu_fails_when_no_cuda(monkeypatch):
    fake_torch = _make_fake_torch(False)
    passed, msg = _call_check_gpu(monkeypatch, fake_torch)
    assert passed is False
    assert "compatible GPU" in msg
