from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import importlib.util
import logging
import os
import subprocess
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Override for the BasicVSR++ deformable-conv implementation (A/B testing).
DEFORM_BACKEND_ENV = "JASNA_DEFORM_BACKEND"

# NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
# spatial dimensions but a variable temporal clip length (and therefore variable
# effective convolution batches), so FAST avoids repeated runtime profiling while
# still using MIOpen's system/user performance databases. Users can override this.
if getattr(torch.version, "hip", None):
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    CPU = "cpu"


@dataclass(frozen=True)
class AcceleratorCapabilities:
    vendor: AcceleratorVendor
    pytorch_device_type: str
    tensorrt: bool
    migraphx: bool
    nvcodec: bool
    amf: bool
    xpu: bool


def vendor_for_device(device: torch.device | str | None = None) -> AcceleratorVendor:
    resolved = torch.device(device) if device is not None else None
    if resolved is not None and resolved.type == "cpu":
        return AcceleratorVendor.CPU
    if resolved is not None and resolved.type == "xpu":
        return AcceleratorVendor.INTEL
    if getattr(torch.version, "hip", None):
        return AcceleratorVendor.AMD
    if resolved is None:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return AcceleratorVendor.INTEL
    if getattr(torch.version, "cuda", None):
        return AcceleratorVendor.NVIDIA
    return AcceleratorVendor.CPU


def capabilities_for_device(
    device: torch.device | str | None = None,
) -> AcceleratorCapabilities:
    vendor = vendor_for_device(device)
    device_type = (
        torch.device(device).type
        if device is not None
        else "xpu" if vendor is AcceleratorVendor.INTEL else "cuda" if vendor in {
            AcceleratorVendor.NVIDIA,
            AcceleratorVendor.AMD,
        } else "cpu"
    )
    return AcceleratorCapabilities(
        vendor=vendor,
        pytorch_device_type=device_type,
        tensorrt=vendor is AcceleratorVendor.NVIDIA,
        migraphx=vendor is AcceleratorVendor.AMD,
        nvcodec=vendor is AcceleratorVendor.NVIDIA,
        amf=vendor is AcceleratorVendor.AMD,
        xpu=vendor is AcceleratorVendor.INTEL,
    )


def is_nvidia_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.NVIDIA


def is_amd_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.AMD


def is_intel_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.INTEL


def is_gpu(device: torch.device | str) -> bool:
    return torch.device(device).type != "cpu"


def device_module(device: torch.device | str):
    return torch.get_device_module(torch.device(device))


def device_context(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return nullcontext()
    return device_module(resolved).device(resolved)


def stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    try:
        return device_module(stream.device).stream(stream)
    except (TypeError, ValueError):
        # Also supports lightweight stream doubles used by callers/tests.
        return torch.cuda.stream(stream)


def new_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).Stream(resolved)


def current_stream(device: torch.device | str):
    resolved = torch.device(device)
    return device_module(resolved).current_stream(resolved)


def new_event(device: torch.device | str):
    return device_module(torch.device(device)).Event()


def set_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).set_device(resolved)


def synchronize(device: torch.device | str | None = None) -> None:
    if device is None:
        torch.accelerator.synchronize()
        return
    resolved = torch.device(device)
    if resolved.type != "cpu":
        device_module(resolved).synchronize(resolved)


def empty_cache(device: torch.device | str | None = None) -> None:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.empty_cache()
        return
    if device is not None:
        module = device_module(torch.device(device))
        if hasattr(module, "empty_cache"):
            module.empty_cache()


def ipc_collect(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


def reset_peak_memory_stats(device: torch.device | str) -> None:
    module = device_module(torch.device(device))
    if hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats(torch.device(device))


def mem_get_info(device: torch.device | str) -> tuple[int, int]:
    module = device_module(torch.device(device))
    return module.mem_get_info(torch.device(device))


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    return str(device_module(resolved).get_device_name(resolved))


def autocast(device: torch.device | str, dtype: torch.dtype | None = None, enabled: bool = True):
    """Vendor-neutral autocast. fp16 on cuda (incl. ROCm), bf16 on xpu (fp16 is
    unreliable on Arc); nullcontext on cpu or when disabled."""
    resolved = torch.device(device)
    if not enabled or resolved.type == "cpu":
        return nullcontext()
    if dtype is None:
        dtype = torch.float16 if resolved.type == "cuda" else torch.bfloat16
    return torch.autocast(resolved.type, dtype=dtype)


def host_buffer(shape: tuple[int, ...], dtype: torch.dtype, want_pinned: bool) -> torch.Tensor:
    """Host tensor for GPU staging; falls back to pageable memory when pinning fails."""
    if want_pinned:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError as e:
            logger.warning("Pinned host memory unavailable (%s); using pageable buffers", e)
    return torch.empty(shape, dtype=dtype)


def deform_conv2d_backend(device: torch.device | str) -> str:
    """"torchvision" (native kernel) or "grid_sample" (pure-torch composition).

    torchvision ships no XPU deform_conv2d kernel (pytorch/vision RFC #8679); its
    dispatcher silently falls back to the fp32 CPU kernel, dragging the alignment
    features through host memory every clip. The grid_sample composition runs
    natively on xpu. cuda/cpu keep the torchvision kernel (bit-identical). AMD
    (device.type=="cuda") also keeps torchvision — torchvision-ROCm has the kernel.
    JASNA_DEFORM_BACKEND overrides for A/B testing.
    """
    override = os.environ.get(DEFORM_BACKEND_ENV)
    if override in ("torchvision", "grid_sample"):
        return override
    if torch.device(device).type == "xpu":
        return "grid_sample"
    return "torchvision"


def _module_available(name: str) -> bool:
    """Package presence without importing it (importing tensorrt/nvvfx has
    DLL-ordering side effects). Already-loaded / test-injected modules count."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def supports_tensorrt(device: torch.device | str | None = None) -> bool:
    return is_nvidia_device(device) and _module_available("tensorrt")


def supports_nvvfx(device: torch.device | str | None = None) -> bool:
    return is_nvidia_device(device) and _module_available("nvvfx")


# .format(index=...) template; {{count}} survives as an f-string brace pair.
_XPU_PROBE_CODE = """\
import sys
import torch
ok = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
if not ok:
    sys.exit(1)
count = torch.xpu.device_count()
if {index} >= count:
    print(f"xpu device index {index} out of range (found {{count}})", file=sys.stderr)
    sys.exit(1)
print(torch.xpu.get_device_name({index}))
"""


@lru_cache(maxsize=8)
def probe_xpu_subprocess(index: int = 0) -> tuple[bool, str]:
    """Probe torch.xpu availability in a subprocess → (ok, name_or_reason).

    A subprocess is not paranoia: on broken driver stacks torch.xpu.is_available()
    can hard-crash the process (Lada #292), and on Battlemage the default
    Level-Zero backend can SIGABRT (intel/compute-runtime#922). On failure, retry
    via the OpenCL adapter; if that works, export ONEAPI_DEVICE_SELECTOR for this
    process so all later xpu use inherits the working backend. Results are
    cached per device index.
    """
    from jasna._frozen import is_frozen

    if is_frozen():
        # Frozen builds cannot re-invoke a Python interpreter (sys.executable is
        # the app binary); fall back to a guarded in-process check. Everything
        # stays inside the try: get_device_name itself can raise on Level-Zero
        # enumeration failures (the compute-runtime#922 class).
        try:
            ok = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
            if not ok:
                return False, "torch.xpu unavailable"
            count = torch.xpu.device_count()
            if index >= count:
                return False, f"xpu device index {index} out of range (found {count})"
            return True, torch.xpu.get_device_name(index)
        except Exception as exc:
            return False, f"torch.xpu check failed: {exc}"

    def _run(extra_env: dict[str, str]) -> tuple[bool, str]:
        env = {**os.environ, **extra_env}
        try:
            result = subprocess.run(
                [sys.executable, "-c", _XPU_PROBE_CODE.format(index=index)],
                capture_output=True, text=True, timeout=120, env=env,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"xpu probe failed to run: {exc}"
        if result.returncode == 0:
            return True, result.stdout.strip()
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, detail[-1] if detail else f"exit code {result.returncode}"

    ok, info = _run({})
    if ok:
        return True, info
    logger.info("torch.xpu probe failed via default backend (%s); retrying via OpenCL", info)
    ok_ocl, info_ocl = _run({"ONEAPI_DEVICE_SELECTOR": "opencl:gpu"})
    if ok_ocl:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:gpu"
        logger.warning(
            "torch.xpu works only via the OpenCL backend on this driver stack "
            "(see intel/compute-runtime#922); exported ONEAPI_DEVICE_SELECTOR=opencl:gpu"
        )
        return True, info_ocl
    return False, info
