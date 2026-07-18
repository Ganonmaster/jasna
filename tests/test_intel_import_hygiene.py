"""Intel installs ship no tensorrt/torch-tensorrt/nvvfx.

Everything on the non-CUDA code path must import without them; TRT imports
must stay behind lazy cuda-only branches.
"""
from __future__ import annotations

import subprocess
import sys

_PROBE = """
import importlib.abc
import importlib.machinery
import sys

# Block at loader level, not find_spec: torch._dynamo probes optional modules
# with importlib.util.find_spec and handles absence, but a raise from a
# meta-path finder there would crash torch itself. Only real imports must fail.
class _BlockedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ImportError(f"blocked Nvidia-only module: {spec.name}")

    def exec_module(self, module):
        raise ImportError(f"blocked Nvidia-only module: {module.__name__}")

class _BlockNvidiaOnly:
    BLOCKED = ("tensorrt", "torch_tensorrt", "nvvfx", "tensorrt_libs")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            return importlib.machinery.ModuleSpec(name, _BlockedLoader())
        return None

sys.meta_path.insert(0, _BlockNvidiaOnly())

import jasna.main
import jasna.device_backend
from jasna.pipeline import Pipeline
import jasna.streaming_pipeline
import jasna.streaming_encoder
from jasna.media.video_decoder import create_video_reader, SoftwareVideoReader
from jasna.media.video_encoder import VideoEncoder, ENCODER_SPECS_SOFTWARE
from jasna.media.qsv import QsvVideoReader
from jasna.mosaic.detection_registry import build_detection_model
import jasna.mosaic.rfdetr
import jasna.mosaic.yolo
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.restoration_pipeline import RestorationPipeline
import jasna.engine_compiler
import jasna.vram_offloader

# Module imports alone don't cover call-time imports (a bare `import
# tensorrt` inside a function body); exercise the startup-path functions
# every Intel run goes through.
from jasna._suppress_noise import install
install()
from jasna.device_backend import default_fp16, hw_media, resolve_fp16, supports_tensorrt
import torch
assert supports_tensorrt(torch.device("cpu")) is False
assert hw_media(torch.device("cpu")) is None
assert resolve_fp16(None, torch.device("cpu")) is False
from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
result = ensure_engines_compiled(EngineCompilationRequest(device="cpu", fp16=False, basicvsrpp=False))
assert result.use_basicvsrpp_tensorrt is False
print("INTEL_IMPORT_OK")
"""


def test_intel_path_imports_without_nvidia_packages():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "INTEL_IMPORT_OK" in result.stdout
