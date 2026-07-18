from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
from torch.nn import functional as F

from jasna.accelerator import is_amd_device, is_intel_device, is_nvidia_device
from jasna.engine_paths import get_onnx_tensorrt_engine_path
from jasna.mosaic.detections import Detections


def TrtRunner(*args, **kwargs):
    from jasna.trt.trt_runner import TrtRunner as Runner

    return Runner(*args, **kwargs)


def compile_rfdetr_engine(
    onnx_path: Path,
    device: torch.device,
    batch_size: int,
    fp16: bool = True,
) -> Path:
    if is_amd_device(device):
        from jasna.mosaic.migraphx_runner import MigraphxRunner

        runner = MigraphxRunner(
            onnx_path,
            input_shapes=[(int(batch_size), 3, 768, 768)],
            device=device,
            fp16=bool(fp16),
        )
        cache_path = runner.cache_dir
        runner.close()
        return cache_path
    if is_intel_device(device):
        from jasna.ov.ov_runner import OvRunner, ov_cache_dir

        cache_path = ov_cache_dir(onnx_path, device, fp16=bool(fp16))
        # Constructing the runner compiles the model, populating the OV blob cache.
        runner = OvRunner(
            onnx_path,
            input_shapes=[(int(batch_size), 3, 768, 768)],
            device=device,
            ov_device="GPU",
            cache_dir=cache_path,
            fp16=bool(fp16),
        )
        runner.close()
        return cache_path
    if not is_nvidia_device(device):
        raise RuntimeError(
            f"RF-DETR is not supported on device backend {device.type!r}"
        )

    from jasna.trt import compile_onnx_to_tensorrt_engine
    return compile_onnx_to_tensorrt_engine(
        onnx_path,
        device,
        batch_size=int(batch_size),
        fp16=bool(fp16),
        workspace_gb=20,
    )


class RfDetrMosaicDetectionModel:
    DEFAULT_RESOLUTION = 768
    DEFAULT_SCORE_THRESHOLD = 0.25
    DEFAULT_MAX_SELECT = 16

    def __init__(
        self,
        *,
        onnx_path: Path,
        batch_size: int,
        device: torch.device,
        resolution: int = DEFAULT_RESOLUTION,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_select: int = DEFAULT_MAX_SELECT,
        fp16: bool = True,
    ) -> None:
        self.onnx_path = onnx_path
        self.batch_size = int(batch_size)
        self.device = device
        self.resolution = int(resolution)
        self.score_threshold = float(score_threshold)
        self.max_select = int(max_select)

        if is_amd_device(self.device):
            from jasna.mosaic.migraphx_runner import MigraphxRunner

            self.runner = MigraphxRunner(
                self.onnx_path,
                input_shapes=[
                    (self.batch_size, 3, self.resolution, self.resolution)
                ],
                device=self.device,
                fp16=bool(fp16),
            )
            self.engine_path = self.runner.cache_dir
        elif is_intel_device(self.device):
            from jasna.ov.ov_runner import OvRunner, ov_cache_dir

            cache_dir = ov_cache_dir(self.onnx_path, self.device, fp16=bool(fp16))
            self.runner = OvRunner(
                self.onnx_path,
                input_shapes=[(self.batch_size, 3, self.resolution, self.resolution)],
                device=self.device,
                ov_device="GPU",
                cache_dir=cache_dir,
                fp16=bool(fp16),
            )
            self.engine_path = cache_dir
        elif is_nvidia_device(self.device):
            self.engine_path = get_onnx_tensorrt_engine_path(
                self.onnx_path, batch_size=self.batch_size, fp16=bool(fp16),
            )
            if not self.engine_path.exists():
                raise FileNotFoundError(
                    f"RF-DETR engine not found: {self.engine_path}. "
                    "Run engine compilation first via ensure_engines_compiled()."
                )
            self.runner = TrtRunner(
                self.engine_path,
                input_shapes=[
                    (self.batch_size, 3, self.resolution, self.resolution)
                ],
                device=self.device,
            )
        else:
            raise RuntimeError(
                f"RF-DETR is not supported on device backend {self.device.type!r}"
            )
        self._input_name = self.runner.input_names[0]
        self.input_dtype = self.runner.input_dtypes[self._input_name]

        self.boxes_out = next(
            k for k in self.runner.output_names if self.runner.outputs[k].ndim == 3 and self.runner.outputs[k].shape[-1] == 4
        )
        self.masks_out = next(k for k in self.runner.output_names if self.runner.outputs[k].ndim == 4)
        self.logits_out = next(k for k in self.runner.output_names if k not in {self.boxes_out, self.masks_out})
        logger.info("RF-DETR detection model loaded: %s (batch_size=%d)", self.engine_path, self.batch_size)

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None

    def _preprocess(self, frames_uint8_bchw: torch.Tensor) -> torch.Tensor:
        x = frames_uint8_bchw.to(device=self.device, dtype=self.input_dtype).div_(255.0)
        x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std

    @staticmethod
    def _postprocess(
        *,
        pred_boxes: torch.Tensor,  # (B, Q, 4) cxcywh normalized
        pred_logits: torch.Tensor,  # (B, Q, C)
        pred_masks: torch.Tensor,  # (B, Q, Hm, Wm)
        target_hw: tuple[int, int],
        score_threshold: float,
        max_select: int,
    ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
        b, q, c = pred_logits.shape
        prob = pred_logits.sigmoid()
        k = min(max_select, q)
        topk_values, topk_indexes = torch.topk(prob.view(b, -1), k, dim=1)
        topk_boxes = topk_indexes // c

        x_c, y_c, w, h = pred_boxes.unbind(-1)
        boxes = torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)
        boxes = boxes.gather(1, topk_boxes.unsqueeze(-1).expand(b, k, 4))

        th, tw = target_hw
        boxes = boxes * boxes.new_tensor((tw, th, tw, th))

        hm, wm = pred_masks.shape[-2], pred_masks.shape[-1]
        masks = pred_masks.gather(1, topk_boxes[:, :, None, None].expand(b, k, hm, wm)) > 0.0

        valid_mask = topk_values > score_threshold  # (B, K)
        boxes_cpu = boxes.to(device='cpu', dtype=torch.float32).numpy()  # (B, K, 4)
        valid_mask_cpu = valid_mask.cpu().numpy()  # (B, K)

        boxes_list: list[np.ndarray] = []
        masks_list: list[torch.Tensor] = []
        for i in range(b):
            valid_i = valid_mask_cpu[i]
            boxes_list.append(boxes_cpu[i][valid_i])  # (N_i, 4) CPU
            # Select masks by integer index from the already-synced CPU mask,
            # not by boolean-indexing the device tensor (which runs nonzero() +
            # a D2H sync every frame — stalls the xpu/rocm pipeline). Bit-
            # identical: nonzero yields ascending indices == boolean-index order.
            idx = np.nonzero(valid_i)[0]
            if idx.size:
                index = torch.as_tensor(idx, dtype=torch.long, device=masks.device)
                masks_list.append(masks[i].index_select(0, index))
            else:
                masks_list.append(masks[i][:0])

        return boxes_list, masks_list

    def scan_scores_masks(
        self, frames_uint8_bchw: torch.Tensor, *, mask_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU-only fast path for whole-video scanning: per-frame best score
        (B,) float32 and merged low-res mask (B, mask_h, mask_w) bool, with no
        host synchronization."""

        x = self._preprocess(frames_uint8_bchw)
        outs = self.runner.infer({self._input_name: x})
        per_query = outs[self.logits_out].sigmoid().amax(dim=-1)  # (B, Q)
        scores = per_query.amax(dim=-1).float()  # (B,)
        pred_masks = outs[self.masks_out]  # (B, Q, Hm, Wm)
        active = (pred_masks > 0.0) & (per_query > self.score_threshold)[:, :, None, None]
        merged = active.any(dim=1, keepdim=True).float()
        merged = F.interpolate(merged, size=mask_hw, mode="area") > 0.0
        return scores, merged[:, 0]

    def __call__(self, frames_uint8_bchw: torch.Tensor, *, target_hw: tuple[int, int]) -> Detections:
        x = self._preprocess(frames_uint8_bchw)
        outs = self.runner.infer({self._input_name: x})
        boxes_list, masks_list = self._postprocess(
            pred_boxes=outs[self.boxes_out],
            pred_logits=outs[self.logits_out],
            pred_masks=outs[self.masks_out],
            target_hw=target_hw,
            score_threshold=self.score_threshold,
            max_select=self.max_select,
        )
        return Detections(
            boxes_xyxy=boxes_list,
            masks=masks_list,
        )
