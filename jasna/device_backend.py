"""Device backend abstraction: one place that knows cuda vs xpu vs cpu.

torch is imported lazily inside functions because jasna.main runs its
preflight checks (and prints friendly errors) before importing torch.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import nullcontext
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

GPU_DEVICE_TYPES = ("cuda", "xpu")

# Environment override for the BasicVSR++ deformable-conv implementation,
# useful for A/B testing once the grid_sample backend lands.
DEFORM_BACKEND_ENV = "JASNA_DEFORM_BACKEND"


def _torch():
    import torch

    return torch


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def is_valid_device_string(spec: str) -> bool:
    if spec == "auto" or spec == "cpu":
        return True
    for prefix in GPU_DEVICE_TYPES:
        if spec == prefix:
            return True
        if spec.startswith(prefix + ":"):
            index = spec[len(prefix) + 1 :]
            return index.isdigit()
    return False


def _xpu_usable() -> bool:
    """torch.xpu availability, without risking an in-process hard crash.

    On broken Intel driver stacks torch.xpu.is_available() can abort the
    whole process (Lada #292 / intel/compute-runtime#922), so first-touch
    happens in a subprocess. Frozen builds ship Nvidia-only and have no
    re-invokable interpreter, so they use the in-process check.
    """
    torch = _torch()
    if getattr(torch, "xpu", None) is None:
        return False
    # torch.xpu the *module* exists in every build; only +xpu wheels carry the
    # runtime. Skip the multi-second subprocess probe on cuda/cpu builds.
    if getattr(torch.version, "xpu", None) is None:
        return False
    from jasna._frozen import is_frozen

    if is_frozen():
        return torch.xpu.is_available()
    ok, _ = probe_xpu_subprocess()
    return ok and torch.xpu.is_available()


def resolve_auto_spec() -> str:
    """Pick the device for --device auto.

    An install ships exactly one GPU torch runtime (+cu130 or +xpu wheels),
    so the torch build decides which probe to run: this keeps CUDA startup
    free of the xpu subprocess probe, and Intel installs never see CUDA.
    """
    torch = _torch()
    if torch.version.cuda is not None and torch.cuda.is_available():
        return "cuda:0"
    if _xpu_usable():
        return "xpu:0"
    return "cpu"


def resolve_device(spec: str) -> "torch.device":
    torch = _torch()
    if spec == "auto":
        return torch.device(resolve_auto_spec())
    if not is_valid_device_string(spec):
        raise ValueError(f"Invalid device: {spec!r} (expected auto, cpu, cuda[:N] or xpu[:N])")
    return torch.device(spec)


def list_devices() -> list[tuple[str, str]]:
    """[(device string, human name)] for every usable device."""
    torch = _torch()
    devices: list[tuple[str, str]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append((f"cuda:{i}", torch.cuda.get_device_name(i)))
    if _xpu_usable():
        for i in range(torch.xpu.device_count()):
            devices.append((f"xpu:{i}", torch.xpu.get_device_name(i)))
    devices.append(("cpu", "CPU"))
    return devices


def is_gpu(device: "torch.device | str") -> bool:
    torch = _torch()
    return torch.device(device).type in GPU_DEVICE_TYPES


# ---------------------------------------------------------------------------
# Backend adapters: the subset of the torch.cuda module surface Jasna uses,
# uniform across cuda / xpu / cpu.
# ---------------------------------------------------------------------------

class _TorchGpuBackend:
    """Delegates to torch.cuda or torch.xpu; papers over the small gaps."""

    def __init__(self, mod: Any) -> None:
        self._mod = mod

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mod, name)

    def ipc_collect(self) -> None:
        # CUDA-only concept; torch.xpu has no equivalent.
        fn = getattr(self._mod, "ipc_collect", None)
        if fn is not None:
            fn()


class _CpuNoopStream:
    def __enter__(self) -> "_CpuNoopStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def wait_event(self, event: object) -> None:
        return None

    def record_event(self, event: object = None) -> object:
        return event if event is not None else _CpuNoopEvent()


class _CpuNoopEvent:
    def record(self, stream: object = None) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def wait(self, stream: object = None) -> None:
        return None


class _CpuBackend:
    """No-op stand-in so `--device cpu` exercises the exact GPU code paths."""

    def is_available(self) -> bool:
        return True

    def set_device(self, device: object) -> None:
        return None

    def synchronize(self, device: object = None) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def ipc_collect(self) -> None:
        return None

    def reset_peak_memory_stats(self, device: object = None) -> None:
        return None

    def mem_get_info(self, device: object = None) -> tuple[int, int]:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)

    def memory_allocated(self, device: object = None) -> int:
        return 0

    def memory_reserved(self, device: object = None) -> int:
        return 0

    def get_device_properties(self, device: object = None) -> Any:
        import psutil

        class _Props:
            name = "CPU"
            total_memory = int(psutil.virtual_memory().total)

        return _Props()

    def get_device_name(self, device: object = None) -> str:
        return "CPU"

    def current_stream(self, device: object = None) -> _CpuNoopStream:
        return _CpuNoopStream()

    def Stream(self, device: object = None, **kwargs: object) -> _CpuNoopStream:
        return _CpuNoopStream()

    def Event(self, **kwargs: object) -> _CpuNoopEvent:
        return _CpuNoopEvent()

    def stream(self, stream: object = None):
        return nullcontext()

    def device(self, device: object = None):
        return nullcontext()


_CPU_BACKEND = _CpuBackend()
_GPU_BACKENDS: dict[str, Any] = {}


def gpu_mod(device: "torch.device | str") -> Any:
    """torch.cuda / torch.xpu / no-op equivalent for the given device.

    Backends are cached per device type: this gets called on per-frame paths.
    """
    torch = _torch()
    dev_type = torch.device(device).type
    if dev_type not in GPU_DEVICE_TYPES:
        return _CPU_BACKEND
    backend = _GPU_BACKENDS.get(dev_type)
    if backend is None:
        backend = _TorchGpuBackend(torch.cuda if dev_type == "cuda" else torch.xpu)
        _GPU_BACKENDS[dev_type] = backend
    return backend


def device_ctx(device: "torch.device | str"):
    """`with torch.cuda.device(...)` generalized; nullcontext on cpu."""
    torch = _torch()
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.device(dev)
    if dev.type == "xpu":
        return torch.xpu.device(dev)
    return nullcontext()


def autocast_ctx(device: "torch.device | str", dtype: "torch.dtype | None" = None, enabled: bool = True):
    torch = _torch()
    dev = torch.device(device)
    if not enabled or dev.type not in GPU_DEVICE_TYPES:
        return nullcontext()
    if dtype is None:
        # fp16 is proven on CUDA; bf16 is the stable reduced precision on Arc.
        dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16
    return torch.autocast(dev.type, dtype=dtype)


# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------

def resolve_fp16(explicit: bool | None, device: "torch.device | str") -> bool:
    """Resolve an optional --fp16 flag: explicit value wins, else device default."""
    if explicit is not None:
        return bool(explicit)
    return default_fp16(device)


def host_buffer(shape: tuple[int, ...], dtype: "torch.dtype", want_pinned: bool) -> "torch.Tensor":
    """Host tensor for GPU staging; falls back to pageable memory when pinning fails."""
    torch = _torch()
    if want_pinned:
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError as e:
            logger.warning("Pinned host memory unavailable (%s); using pageable buffers", e)
    return torch.empty(shape, dtype=dtype)


def default_fp16(device: "torch.device | str") -> bool:
    """Whether fp16 inference is the sane default on this device.

    xpu is False until BasicVSR++'s deform_conv2d runs natively on xpu:
    the implicit XPU->CPU dispatcher fallback uses the CPU kernel, which
    has no half support.
    """
    torch = _torch()
    return torch.device(device).type == "cuda"


def _module_available(name: str) -> bool:
    """Package presence without importing it (imports here have DLL-ordering
    side effects — see rtx_superres_secondary_restorer._preload_tensorrt_runtime).

    An already-loaded module (including test-injected mocks) counts as
    available; find_spec raises ValueError on mocks with __spec__ = None.
    """
    if name in sys.modules:
        return True
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def supports_tensorrt(device: "torch.device | str") -> bool:
    torch = _torch()
    if torch.device(device).type != "cuda":
        return False
    return _module_available("tensorrt")


def supports_nvvfx(device: "torch.device | str") -> bool:
    torch = _torch()
    if torch.device(device).type != "cuda":
        return False
    return _module_available("nvvfx")


def hw_media(device: "torch.device | str") -> str | None:
    """Hardware video decode/encode family for this compute device."""
    torch = _torch()
    dev_type = torch.device(device).type
    if dev_type == "cuda":
        return "nvdec_nvenc"
    if dev_type == "xpu":
        return "qsv"
    return None


def deform_conv2d_backend(device: "torch.device | str") -> str:
    """"torchvision" (bit-identical current behavior) or "grid_sample".

    The grid_sample implementation is the planned XPU-native path; until it
    lands, xpu also returns "torchvision" and relies on PyTorch's implicit
    XPU->CPU fallback for the op (matching what upstream Lada ships).
    """
    override = os.environ.get(DEFORM_BACKEND_ENV)
    if override in ("torchvision", "grid_sample"):
        return override
    return "torchvision"


# ---------------------------------------------------------------------------
# XPU preflight probe
# ---------------------------------------------------------------------------

_XPU_PROBE_CODE = (
    "import torch, sys; "
    "ok = getattr(torch, 'xpu', None) is not None and torch.xpu.is_available(); "
    "print(torch.xpu.get_device_name(0) if ok else ''); "
    "sys.exit(0 if ok else 1)"
)


@lru_cache(maxsize=1)
def probe_xpu_subprocess() -> tuple[bool, str]:
    """Probe torch.xpu availability in a subprocess.

    A subprocess is not paranoia: on broken driver stacks
    torch.xpu.is_available() can hard-crash the process (Lada #292), and on
    Battlemage the default Level-Zero backend can SIGABRT
    (intel/compute-runtime#922). On failure, retry via the OpenCL adapter;
    if that works we export ONEAPI_DEVICE_SELECTOR for this process so all
    later xpu use inherits the working backend.
    """

    from jasna._frozen import is_frozen

    if is_frozen():
        # Frozen builds cannot re-invoke a Python interpreter (sys.executable
        # is the app binary); fall back to a guarded in-process check.
        torch = _torch()
        try:
            ok = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"torch.xpu check failed: {exc}"
        return (True, torch.xpu.get_device_name(0)) if ok else (False, "torch.xpu unavailable")

    def _run(extra_env: dict[str, str]) -> tuple[bool, str]:
        env = {**os.environ, **extra_env}
        try:
            result = subprocess.run(
                [sys.executable, "-c", _XPU_PROBE_CODE],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
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
