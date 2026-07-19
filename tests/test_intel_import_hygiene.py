"""AMD/Intel installs ship no tensorrt/torch-tensorrt/nvvfx.

Everything on the non-CUDA code path must import without them; TRT imports
must stay behind lazy nvidia-only branches. This is the regression net for the
lazy-import guards (rfdetr TrtRunner, _suppress_noise.install, conftest).
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
import jasna.accelerator
from jasna.pipeline import Pipeline
import jasna.streaming_pipeline
import jasna.streaming_encoder
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.media.video_encoder import NvidiaVideoEncoder, QSV_ENCODER_SPECS
import jasna.media.qsv
from jasna.mosaic.detection_registry import build_detection_model
import jasna.mosaic.rfdetr
import jasna.mosaic.yolo
from jasna.ov.ov_runner import OvRunner
from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import deform_conv2d_dispatch
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.restoration_pipeline import RestorationPipeline
import jasna.engine_compiler
import jasna.vram_offloader

# The `jasna --benchmark --device xpu` CLI path: benchmark/__init__ eagerly
# imports the per-benchmark modules, which must not drag in jasna.trt (tensorrt).
import jasna.benchmark
from jasna.benchmark import run_benchmark_cli
from jasna.benchmark.basicvsrpp_restoration import benchmark_basicvsrpp_restoration

# Module imports alone don't cover call-time imports (a bare `import tensorrt`
# inside a function body); exercise the startup-path functions every non-CUDA
# run goes through.
import torch
from jasna._suppress_noise import install
install()
from jasna.accelerator import deform_conv2d_backend, is_intel_device, supports_tensorrt
assert supports_tensorrt(torch.device("cpu")) is False
assert deform_conv2d_backend(torch.device("xpu:0")) == "grid_sample"
assert is_intel_device(torch.device("cpu")) is False
from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
result = ensure_engines_compiled(
    EngineCompilationRequest(device="cpu", fp16=False, basicvsrpp=False, detection=False)
)
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
