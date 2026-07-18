"""Intel Quick Sync Video (QSV) decode/encode helpers for Arc GPUs.

The QSV decoder runs in copy-back mode (``gpu_copy=on``): the media engine
decodes on the GPU and the driver hands back system-memory NV12/P010 frames,
which the shared software path then reformats and uploads to xpu. There is no
zero-copy path from a QSV surface into a torch.xpu tensor, so host staging is
the intended design (the same shape as the AMD/AMF and NVDEC-software paths).
"""

from __future__ import annotations

import logging
import os

import av

log = logging.getLogger(__name__)

# FFmpeg QSV decoder names by container codec name.
_QSV_DECODERS = {
    "h264": "h264_qsv",
    "hevc": "hevc_qsv",
    "av1": "av1_qsv",
    "vp9": "vp9_qsv",
    "mpeg2video": "mpeg2_qsv",
    "vc1": "vc1_qsv",
}


def qsv_disabled() -> bool:
    """Escape hatch: JASNA_DISABLE_QSV=1 forces software decode and encode."""
    return os.environ.get("JASNA_DISABLE_QSV", "").lower() in ("1", "true", "yes")


def qsv_decoder_name(codec_name: str) -> str | None:
    return _QSV_DECODERS.get(str(codec_name).lower())


def qsv_decoder_available(codec_name: str) -> bool:
    if qsv_disabled():
        return False
    name = qsv_decoder_name(codec_name)
    if name is None:
        return False
    try:
        av.Codec(name, "r")
    except Exception:
        return False
    # Codec presence only proves the FFmpeg build; opening an encoder session
    # proves the driver/runtime (both share the oneVPL session).
    from jasna.os_utils import check_qsv_available

    ok, _ = check_qsv_available()
    return ok
