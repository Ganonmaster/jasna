"""Compile TensorRT engines in a subprocess to guarantee full VRAM release."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

from jasna._frozen import is_frozen

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30 * 60


@dataclass
class EngineCompilationRequest:
    device: str
    fp16: bool

    basicvsrpp: bool = False
    basicvsrpp_model_path: str = ""
    basicvsrpp_max_clip_size: int = 60

    detection: bool = False
    detection_model_name: str = ""
    detection_model_path: str = ""
    detection_batch_size: int = 4

    unet4x: bool = False

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> EngineCompilationRequest:
        return EngineCompilationRequest(**json.loads(s))


@dataclass
class EngineCompilationResult:
    use_basicvsrpp_tensorrt: bool = False


def _basicvsrpp_engines_exist(model_path: str, fp16: bool, max_clip_size: int) -> bool:
    from jasna.engine_paths import all_basicvsrpp_sub_engines_exist
    return all_basicvsrpp_sub_engines_exist(model_path, fp16, max_clip_size)


def _ov_warmup_marker_path(detection_model_path: str, batch_size: int) -> Path:
    """Marker recording a completed OV warm-up for this model+config.

    OpenVINO keys its blob cache internally, so blob presence can't be checked
    per model; the marker makes the "already warmed" decision exact and a
    model/batch change re-triggers the warm-up (with its progress feedback)
    instead of a silent mid-pipeline compile. OV GPU precision is fixed (f16
    hint), so it is not part of the key.
    """
    from jasna.engine_paths import ov_cache_dir

    stem = Path(detection_model_path).stem
    return ov_cache_dir() / f".warmed.{stem}.bs{int(batch_size)}"


def _detection_engine_exists(
    detection_model_name: str,
    detection_model_path: str,
    batch_size: int,
    fp16: bool,
    device_type: str = "cuda",
) -> bool:
    from jasna.engine_paths import get_onnx_tensorrt_engine_path, get_yolo_tensorrt_engine_path
    from jasna.mosaic.detection_registry import is_rfdetr_model, is_yolo_model

    if device_type == "xpu":
        if is_rfdetr_model(detection_model_name):
            return _ov_warmup_marker_path(detection_model_path, batch_size).exists()
        return True  # YOLO runs eagerly on xpu
    if is_rfdetr_model(detection_model_name):
        return get_onnx_tensorrt_engine_path(detection_model_path, batch_size=batch_size, fp16=fp16).exists()
    if is_yolo_model(detection_model_name):
        return get_yolo_tensorrt_engine_path(detection_model_path, fp16=fp16).exists()
    return True


def _unet4x_engine_exists(fp16: bool) -> bool:
    from jasna.engine_paths import (
        expected_unet4x_engine_path,
        get_unet4x_encrypted_engine_path,
        unet4x_plaintext_available,
    )

    if unet4x_plaintext_available():
        return expected_unet4x_engine_path(fp16=fp16).exists()

    engine_path = get_unet4x_encrypted_engine_path(fp16=fp16)
    if not engine_path.exists():
        return False

    from jasna.protection import ProtectionError, protected_model
    try:
        protected_model.decrypt_engine_bytes("unet-4x", engine_path.read_bytes())
    except ProtectionError:
        return False
    return True


def ensure_engines_compiled(
    req: EngineCompilationRequest,
    log_callback: typing.Callable[[str], None] | None = None,
) -> EngineCompilationResult:
    result = EngineCompilationResult()

    # TensorRT engines only exist on CUDA. On xpu the detection model gets an
    # OpenVINO cache warm-up through the same subprocess flow; cpu needs nothing.
    device_type = str(req.device).split(":")[0]
    tensorrt_capable = device_type == "cuda"

    need_basicvsrpp = tensorrt_capable and req.basicvsrpp and req.fp16 and not _basicvsrpp_engines_exist(
        req.basicvsrpp_model_path, req.fp16, req.basicvsrpp_max_clip_size
    )
    need_detection = device_type in ("cuda", "xpu") and req.detection and not _detection_engine_exists(
        req.detection_model_name, req.detection_model_path, req.detection_batch_size, req.fp16, device_type
    )
    need_unet4x = tensorrt_capable and req.unet4x and not _unet4x_engine_exists(req.fp16)

    if need_unet4x:
        from jasna.engine_paths import unet4x_plaintext_available
        from jasna.protection import license_store
        if not unet4x_plaintext_available() and not license_store.is_licensed():
            raise RuntimeError("unet-4x is a supporter feature. Enter your license to enable it.")

    if req.basicvsrpp:
        result.use_basicvsrpp_tensorrt = tensorrt_capable and req.fp16 and not need_basicvsrpp

    if not (need_basicvsrpp or need_detection or need_unet4x):
        return result

    logger.info("Spawning engine compilation subprocess...")
    if tensorrt_capable:
        start_msg = "Compiling TensorRT engines (this may take several minutes)..."
    else:
        start_msg = "Compiling the OpenVINO engine cache (first run may take a few minutes)..."
    # The frozen GUI drops its console (FreeConsole), leaving stdout invalid — an
    # unconditional print() there raises WinError 6. Print only on the CLI (no callback).
    if log_callback:
        log_callback(start_msg)
    else:
        print(start_msg)

    if is_frozen():
        cmd = [sys.executable, "--compile-engines", req.to_json()]
    else:
        cmd = [sys.executable, "-m", "jasna.engine_compiler", req.to_json()]

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,  # don't inherit the GUI's detached (invalid) stdin
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **kwargs)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        if line:
            if log_callback:
                log_callback(line)
                logger.debug("[compiler] %s", line)
            else:
                print(line)
                logger.info("[compiler] %s", line)
    returncode = proc.wait(timeout=_TIMEOUT_SECONDS)

    if returncode != 0:
        raise RuntimeError(f"Engine compilation subprocess failed (exit code {returncode})")

    if req.basicvsrpp:
        # Same gates as the pre-compile decision: stale TRT engines on disk
        # (e.g. a weights dir shared with an Nvidia box) must not flip this
        # on for a device that cannot run them.
        result.use_basicvsrpp_tensorrt = tensorrt_capable and req.fp16 and _basicvsrpp_engines_exist(
            req.basicvsrpp_model_path, req.fp16, req.basicvsrpp_max_clip_size
        )

    return result


def _subprocess_compile(req: EngineCompilationRequest) -> None:
    import logging as _logging
    import warnings
    warnings.filterwarnings("ignore")
    _logging.disable(_logging.WARNING)

    from jasna._suppress_noise import install as _install_noise_filters
    _install_noise_filters()
    import torch

    # The compile subprocess imports torch_tensorrt (-> torch._inductor) directly, without
    # going through jasna.pipeline, so the source-introspection shims aren't installed yet.
    # In the compiled (Nuitka) binary that introspection raises; patch before any such import.
    from jasna._frozen import patch_frozen_torch
    patch_frozen_torch()

    device = torch.device(req.device)

    if device.type == "cuda" and req.basicvsrpp and req.fp16 and not _basicvsrpp_engines_exist(
        req.basicvsrpp_model_path, req.fp16, req.basicvsrpp_max_clip_size
    ):
        from jasna.restorer.basicvrspp_tenorrt_compilation import compile_mosaic_restoration_model
        print(f"Compiling BasicVSR++ sub-engines (max_clip_size={req.basicvsrpp_max_clip_size})...")
        compile_mosaic_restoration_model(
            mosaic_restoration_model_path=req.basicvsrpp_model_path,
            device=device,
            fp16=req.fp16,
            max_clip_size=req.basicvsrpp_max_clip_size,
        )
        print("BasicVSR++ sub-engines compiled.")

    if req.detection and device.type in ("cuda", "xpu") and not _detection_engine_exists(
        req.detection_model_name, req.detection_model_path, req.detection_batch_size, req.fp16, device.type
    ):
        from jasna.mosaic.detection_registry import precompile_detection_engine
        print(f"Compiling detection engine ({req.detection_model_name})...")
        precompile_detection_engine(
            detection_model_name=req.detection_model_name,
            detection_model_path=Path(req.detection_model_path),
            batch_size=req.detection_batch_size,
            device=device,
            fp16=req.fp16,
        )
        if device.type == "xpu":
            marker = _ov_warmup_marker_path(req.detection_model_path, req.detection_batch_size)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        print("Detection engine compiled.")

    if req.unet4x and device.type == "cuda" and not _unet4x_engine_exists(req.fp16):
        from jasna.restorer.unet4x_secondary_restorer import compile_unet4x_engine
        print("Compiling Unet4x engine...")
        compile_unet4x_engine(device, fp16=req.fp16)
        print("Unet4x engine compiled.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m jasna.engine_compiler <json_request>", file=sys.stderr)
        sys.exit(1)
    req = EngineCompilationRequest.from_json(sys.argv[1])
    _subprocess_compile(req)
