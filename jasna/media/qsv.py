"""Intel Quick Sync Video (QSV) decode support for Arc GPUs.

The QSV decoders run in copy-back mode (``gpu_copy=on``): the media engine
decodes on the GPU and the driver hands back system-memory NV12/P010 frames.
There is no zero-copy path from a VA/QSV surface into a torch.xpu tensor, so
host staging is the intended design, not a shortcut — the SoftwareVideoReader
conversion/upload pipeline this class inherits is already exactly that.
"""

from __future__ import annotations

import logging

import av

from jasna.media.video_decoder import SoftwareVideoReader, VideoDecodeError
from av.video.reformatter import ColorRange as AvColorRange

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


def qsv_decoder_name(codec_name: str) -> str | None:
    return _QSV_DECODERS.get(str(codec_name).lower())


def qsv_decoder_available(codec_name: str) -> bool:
    name = qsv_decoder_name(codec_name)
    if name is None:
        return False
    try:
        av.Codec(name, "r")
    except Exception:
        return False
    # Codec presence only proves the FFmpeg build; opening a session proves
    # the driver/runtime. The encoder probe shares the same VPL runtime.
    from jasna.os_utils import check_qsv_available

    ok, _ = check_qsv_available()
    return ok


class QsvVideoReader(SoftwareVideoReader):
    """GPU (QSV) decode with copy-back host frames, then the shared upload path."""

    def __enter__(self):
        try:
            self.container = av.open(self.file)
            self.video_stream = self.container.streams.video[0]
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to open {self.file}: {e}") from e

        decoder = qsv_decoder_name(self.metadata.codec_name)
        if decoder is None:
            raise VideoDecodeError(
                f"No QSV decoder for codec {self.metadata.codec_name!r} ({self.file})"
            )
        stream_ctx = self.video_stream.codec_context
        try:
            ctx = av.Codec(decoder, "r").create()
            # gpu_copy=on = copy-back mode: decoded surfaces are transferred to
            # system memory by the driver and frames arrive as nv12/p010le.
            ctx.options = {"gpu_copy": "on"}
            if stream_ctx.extradata:
                ctx.extradata = stream_ctx.extradata
            ctx.open()
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to open QSV decoder {decoder} for {self.file}: {e}") from e
        self._qsv_ctx = ctx

        self.width = stream_ctx.width
        self.height = stream_ctx.height
        self._full_range = (
            stream_ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._qsv_ctx = None
        self.container.close()

    def _decode_frames_from_packet(self, packet) -> list:
        # Demux emits one empty flush packet per stream at EOF; our private
        # codec context must be flushed explicitly.
        if packet.size == 0 and packet.dts is None and packet.pts is None:
            return self._qsv_ctx.decode(None)
        return self._qsv_ctx.decode(packet)
