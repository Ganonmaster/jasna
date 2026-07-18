"""OpenVINO analog of jasna.trt.trt_runner.TrtRunner.

Same contract: static input shapes fixed at construction, persistent output
tensors on the compute device, ``infer(dict) -> dict``. Model call sites can
hold either runner without caring which backend is underneath.

Unlike the CUDA path there is no zero-copy between torch.xpu tensors
(Level-Zero USM) and the OpenVINO GPU plugin (OpenCL), so inputs stage
through the host. For the static detection models this is a few MB per
batch — negligible next to inference itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _ov():
    import openvino as ov

    return ov


_OV_TO_TORCH_DTYPE = {
    "f32": torch.float32,
    "f16": torch.float16,
    "i64": torch.int64,
    "i32": torch.int32,
    "u8": torch.uint8,
    "boolean": torch.bool,
}


def _torch_dtype(ov_type) -> torch.dtype:
    name = ov_type.get_type_name()
    try:
        return _OV_TO_TORCH_DTYPE[name]
    except KeyError as exc:
        raise TypeError(f"Unsupported OpenVINO element type: {name}") from exc


class OvRunner:
    def __init__(
        self,
        model: Path | str,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        *,
        ov_device: str = "GPU",
        cache_dir: Path | None = None,
        fp16: bool = True,
    ) -> None:
        self.model_path = Path(model)
        self._setup(self.model_path, input_shapes, device, ov_device, cache_dir, fp16, str(model))

    @classmethod
    def from_model_bytes(
        cls,
        model_bytes: bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        *,
        ov_device: str = "GPU",
        fp16: bool = True,
        source: str = "<memory>",
    ) -> "OvRunner":
        """Build from in-memory model bytes (encrypted-model path).

        No cache_dir on purpose: OpenVINO's disk cache would persist a
        plaintext compiled blob of a model that never touches disk decrypted.
        """
        self = cls.__new__(cls)
        self.model_path = None
        self._setup(model_bytes, input_shapes, device, ov_device, None, fp16, source)
        return self

    def _setup(
        self,
        model: Path | bytes,
        input_shapes: dict[str, tuple[int, ...]] | list[tuple[int, ...]],
        device: torch.device,
        ov_device: str,
        cache_dir: Path | None,
        fp16: bool,
        source: str,
    ) -> None:
        ov = _ov()
        self.device = torch.device(device)
        self.ov_device = ov_device

        core = ov.Core()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})

        if isinstance(model, bytes):
            ov_model = core.read_model(model=model)
        else:
            ov_model = core.read_model(model=str(model))

        if isinstance(input_shapes, list):
            names = [port.any_name for port in ov_model.inputs]
            input_shapes = dict(zip(names, input_shapes))
        try:
            ov_model.reshape({name: ov.PartialShape(list(shape)) for name, shape in input_shapes.items()})
        except RuntimeError as exc:
            raise RuntimeError(
                f"OpenVINO could not reshape {source} to {input_shapes}. Some exported "
                "models bake a fixed batch size into the graph (rfdetr-v5.onnx only "
                "supports batch 4); try the matching --batch-size."
            ) from exc

        config: dict[str, str] = {}
        if ov_device.startswith("GPU"):
            config["INFERENCE_PRECISION_HINT"] = "f16" if fp16 else "f32"
        try:
            self.compiled = core.compile_model(ov_model, ov_device, config)
        except Exception as exc:
            raise RuntimeError(f"Failed to compile OpenVINO model {source} for {ov_device}: {exc}") from exc
        # Two infer requests for double-buffered async: batch N+1 can run on the
        # GPU while batch N-1's result is fetched and post-processed on the CPU.
        self._requests = [self.compiled.create_infer_request() for _ in range(2)]
        self.request = self._requests[0]  # sync infer() uses request 0
        self._feeds: list[dict | None] = [None, None]  # keep numpy alive per slot (share_inputs)
        self._next_slot = 0

        self.input_names = [port.any_name for port in self.compiled.inputs]
        self.output_names = [port.any_name for port in self.compiled.outputs]
        self.input_dtypes = {
            port.any_name: _torch_dtype(port.get_element_type()) for port in self.compiled.inputs
        }

        self.outputs: dict[str, torch.Tensor] = {}
        for port in self.compiled.outputs:
            shape = tuple(int(d) for d in port.get_shape())
            self.outputs[port.any_name] = torch.empty(
                shape, dtype=_torch_dtype(port.get_element_type()), device=self.device
            )
        logger.info("OpenVINO model compiled: %s on %s (cache=%s)", source, ov_device, cache_dir)

    def close(self) -> None:
        self.outputs.clear()
        self._requests = []
        self.request = None
        self.compiled = None

    @staticmethod
    def _to_feed(inputs: dict[str, torch.Tensor]) -> dict:
        return {
            name: tensor.detach().contiguous().cpu().numpy()
            for name, tensor in inputs.items()
        }

    def _store_outputs(self, results) -> dict[str, torch.Tensor]:
        for port, array in results.items():
            # copy_ fuses dtype conversion and host->device in one hop.
            self.outputs[port.any_name].copy_(torch.from_numpy(array))
        return self.outputs

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # share_inputs: the staged numpy arrays are contiguous and outlive the
        # call, so OpenVINO can wrap them instead of memcpying a second time.
        results = self.request.infer(self._to_feed(inputs), share_inputs=True)
        return self._store_outputs(results)

    def submit(self, inputs: dict[str, torch.Tensor]) -> int:
        """Start an async inference; returns a handle to fetch() later.

        Uses two ping-ponged infer requests, so a submit() may be issued
        while a prior request is still running (double buffering). The feed
        numpy is retained per slot because share_inputs wraps it by reference.
        """
        slot = self._next_slot
        self._next_slot ^= 1
        feed = self._to_feed(inputs)
        self._feeds[slot] = feed
        self._requests[slot].start_async(feed, share_inputs=True)
        return slot

    def fetch(self, handle: int) -> dict[str, torch.Tensor]:
        """Wait for a submit()ed inference and copy its outputs to device.

        The returned tensors are the shared persistent output buffers, so the
        caller must consume them (post-process) before the next fetch().
        """
        req = self._requests[handle]
        req.wait()
        self._feeds[handle] = None
        return self._store_outputs(req.results)
