import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.accelerator import empty_cache, is_nvidia_device
from jasna.models.basicvsrpp.inference import load_model

INFERENCE_SIZE = 256

# Opt-in torch.compile for the eager restoration path (JASNA_TORCH_COMPILE=1).
# Measured on the Arc Pro B50: 1.3x on a full 60-frame clip; first compile per
# cache ~27 min, warm cache ~1 min (kernels persist in TORCHINDUCTOR_CACHE_DIR).
# On xpu, inductor JIT-compiles through triton-xpu, which needs the Level Zero
# dev headers (`apt install libze-dev`) at runtime.
TORCH_COMPILE_ENV = "JASNA_TORCH_COMPILE"
COMPILE_MODE_ENV = "JASNA_COMPILE_MODE"


class BasicvsrppMosaicRestorer:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        max_clip_size: int,
        use_tensorrt: bool,
        fp16: bool,
        config: str | dict | None = None,
    ):
        self.device = torch.device(device)
        self.max_clip_size = int(max_clip_size)
        self.use_tensorrt = bool(use_tensorrt)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.input_dtype = self.dtype

        self._split_forward = None
        self.model = None
        self._gen_attr: str | None = None
        self._eager_generator = None
        self._compiled_generator = None

        if self.use_tensorrt and is_nvidia_device(self.device):
            from jasna.restorer.basicvsrpp_sub_engines import create_split_forward

            pytorch_model = load_model(config, checkpoint_path, self.device, fp16)
            self._split_forward = create_split_forward(
                model=pytorch_model,
                model_weights_path=checkpoint_path,
                device=self.device,
                fp16=fp16,
                max_clip_size=self.max_clip_size,
            )
            if self._split_forward is not None:
                logger.info("BasicVSR++ using TRT sub-engines (fp16=%s)", fp16)
            else:
                self.model = pytorch_model
                logger.info("BasicVSR++ sub-engines not found, using PyTorch model (fp16=%s)", fp16)
        else:
            self.use_tensorrt = False
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

        if (
            os.environ.get(TORCH_COMPILE_ENV) == "1"
            and self.model is not None
            and self._split_forward is None
            and self.device.type != "cpu"
        ):
            self._setup_torch_compile(checkpoint_path)

    def _setup_torch_compile(self, checkpoint_path: str) -> None:
        """Compile the inference generator; warm up so the compile cost lands
        at startup (like the TensorRT engine build), not mid-pipeline where it
        would trip the encode-stall watchdog. Any failure falls back to eager."""
        # Persist compiled kernels next to the model weights so the cache
        # survives reboots (the /tmp default does not). Must be set before the
        # first inductor use in this process; an explicit user setting wins.
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path(checkpoint_path).resolve().parent / "torchinductor_cache"),
        )
        self._gen_attr = (
            "generator_ema"
            if getattr(self.model, "generator_ema", None) is not None
            else "generator"
        )
        self._eager_generator = getattr(self.model, self._gen_attr)
        mode = os.environ.get(COMPILE_MODE_ENV) or None
        logger.info(
            "%s=1: compiling BasicVSR++ restoration graph (first run per cache "
            "may take ~30 min; warm cache ~1 min; cache: %s)...",
            TORCH_COMPILE_ENV,
            os.environ["TORCHINDUCTOR_CACHE_DIR"],
        )
        try:
            self._compiled_generator = torch.compile(self._eager_generator, mode=mode)
            start = time.perf_counter()
            self._warmup_compiled()
            # The warmup routes through _run_model, whose mid-run safety net
            # disables the compiled path (with its own warning) on failure.
            if self._compiled_generator is not None:
                logger.info(
                    "BasicVSR++ torch.compile ready in %.0fs (full %d-frame clips "
                    "run compiled; shorter clips run eager)",
                    time.perf_counter() - start,
                    self.max_clip_size,
                )
        except Exception as exc:  # noqa: BLE001 - opt-in accel, never fail the run
            logger.warning(
                "torch.compile unavailable for BasicVSR++ (%s); continuing eager",
                exc,
            )
            self._compiled_generator = None

    def _warmup_compiled(self) -> None:
        dummy = [
            torch.randint(
                0, 256, (3, INFERENCE_SIZE, INFERENCE_SIZE),
                dtype=torch.uint8, device=self.device,
            )
            for _ in range(self.max_clip_size)
        ]
        self.raw_process(dummy)

    def close(self) -> None:
        if self._split_forward is not None:
            self._split_forward.close()
            self._split_forward = None
        self.model = None

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (C, H, W) tensors in RGB format, [0, 255]
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        with torch.inference_mode():
            stacked = torch.stack(video).to(device=self.device, dtype=self.input_dtype, memory_format=torch.contiguous_format).div_(255.0)

            if self._split_forward is not None:
                result = self._split_forward(stacked.unsqueeze(0))
            else:
                result = self._run_model(stacked.unsqueeze(0), clip_len=len(video))
            return result.squeeze(0)

    def _run_model(self, inputs: torch.Tensor, *, clip_len: int) -> torch.Tensor:
        # Route only FULL-length clips through the compiled generator: every
        # distinct clip length is a fresh dynamo compile (~minutes each), so
        # variable-length tail clips stay eager — no recompiles, no padding
        # waste, and the dominant full clips get the speedup.
        use_compiled = (
            self._compiled_generator is not None and clip_len == self.max_clip_size
        )
        if not use_compiled:
            return self.model(inputs=inputs)
        try:
            setattr(self.model, self._gen_attr, self._compiled_generator)
            return self.model(inputs=inputs)
        except Exception:
            # A mid-run compiled failure must not kill a multi-hour job:
            # disable the compiled path and redo this clip eager. The retry
            # happens OUTSIDE this handler: while it is active the exception's
            # traceback pins the failed forward's activation tensors, and if
            # the failure was an OOM, retrying under that pinned memory would
            # just OOM again.
            logger.warning(
                "compiled BasicVSR++ generator failed mid-run; falling back to "
                "eager for the rest of this run",
                exc_info=True,
            )
            self._compiled_generator = None
        finally:
            setattr(self.model, self._gen_attr, self._eager_generator)
        # Reached only on compiled failure, with the traceback released.
        empty_cache(self.device)
        return self.model(inputs=inputs)

    def restore(self, video: list[Tensor]) -> list[Tensor]:
        """
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            list of (256, 256, C) uint8 tensors in RGB format
        """
        result = self.raw_process([frame.permute(2, 0, 1) for frame in video])
        result = result.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1)
        return list(torch.unbind(result, 0))
