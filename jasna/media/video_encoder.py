from __future__ import annotations

import heapq
import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import av
import numpy as np
import torch
from av.video.frame import CudaContext
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.device_backend import gpu_mod, host_buffer, hw_media, is_gpu
from jasna.media import (
    NVENC_ENCODER_SETTINGS_BY_CODEC,
    QSV_ENCODER_SETTINGS_BY_CODEC,
    SOFTWARE_ENCODER_SETTINGS_BY_CODEC,
    VideoMetadata,
    validate_encoder_settings,
)
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.lut import GpuLutApplier, parse_cube_file
from jasna.media.rgb_to_nv12 import (
    chw_rgb_to_nv12_bt2020_full,
    chw_rgb_to_nv12_bt2020_limited,
    chw_rgb_to_nv12_bt601_full,
    chw_rgb_to_nv12_bt601_limited,
    chw_rgb_to_nv12_bt709_full,
    chw_rgb_to_nv12_bt709_limited,
)
from jasna.media.rgb_to_p010 import (
    chw_rgb_to_p010_bt2020_full,
    chw_rgb_to_p010_bt2020_limited,
    chw_rgb_to_p010_bt601_full,
    chw_rgb_to_p010_bt601_limited,
    chw_rgb_to_p010_bt709_full,
    chw_rgb_to_p010_bt709_limited,
)

av.logging.set_level(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "main10",
    "rc": "vbr",
    "cq": "25",
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}

# lookahead_level breaks avcodec_open2 on h264_nvenc with this lookahead/AQ
# combination (ENOSYS on RTX 5090), so H.264 deliberately omits it.
DEFAULT_H264_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "profile": "high",
    "rc": "vbr",
    # CQ 24 matched HEVC CQ 25 in representative VMAF comparisons.
    "cq": "24",
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}

# AV1 target quality uses a 0..63 scale rather than H.264/HEVC's 0..51.
# CQ 32 matched HEVC CQ 25 in representative VMAF/SSIM comparisons. AV1 QP limits use a separate
# 0..255 scale, so the HEVC qmin/qmax/init_qp values must not be copied here.
# No profile: P010 input makes av1_nvenc emit AV1 Main 10-bit on its own.
# av1_nvenc only consumes the hyphenated spatial-aq spelling.
DEFAULT_AV1_ENCODER_OPTIONS: dict[str, str] = {
    "preset": "p5",
    "tune": "hq",
    "rc": "vbr",
    "cq": "32",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial-aq": "1",
    "aq-strength": "8",
    "bf": "4",
    "b_ref_mode": "middle",
}


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    encoder_name: str
    frame_format: str  # packed tensor layout produced by the RGB converters: "nv12" or "p010le"
    default_options: Mapping[str, str]
    ten_bit: bool
    # The codec context pix_fmt: "cuda" for NVENC hardware frames; for other
    # backends the format the encoder consumes (PyAV reformats automatically
    # when it differs from frame_format, e.g. p010le -> yuv420p10le for x265).
    encoder_pix_fmt: str = "cuda"
    # encoder_pix_fmt when the spec is downgraded to 8-bit NV12 frames
    # (match_input_bit_depth with an 8-bit source), declared explicitly per
    # backend instead of inferred from string patterns.
    encoder_pix_fmt_8bit: str = "cuda"
    supported_settings: frozenset[str] = field(default_factory=frozenset)


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        name="hevc",
        encoder_name="hevc_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_ENCODER_OPTIONS),
        ten_bit=True,
        encoder_pix_fmt="cuda",
        supported_settings=NVENC_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="h264_nvenc",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_H264_ENCODER_OPTIONS),
        ten_bit=False,
        encoder_pix_fmt="cuda",
        supported_settings=NVENC_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="av1_nvenc",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_AV1_ENCODER_OPTIONS),
        ten_bit=True,
        encoder_pix_fmt="cuda",
        supported_settings=NVENC_ENCODER_SETTINGS_BY_CODEC["av1"],
    ),
}

# QSV (Intel Arc) option sets are first-cut placeholders pending VMAF tuning
# against the NVENC defaults above. ICQ rate control via global_quality.
DEFAULT_QSV_HEVC_OPTIONS: dict[str, str] = {
    "preset": "slow",
    "global_quality": "22",
    "profile": "main10",
    "g": "250",
    "bf": "4",
}
DEFAULT_QSV_H264_OPTIONS: dict[str, str] = {
    "preset": "slow",
    "global_quality": "23",
    "g": "250",
    "bf": "4",
}
DEFAULT_QSV_AV1_OPTIONS: dict[str, str] = {
    "preset": "slow",
    "global_quality": "30",
    "g": "250",
}

ENCODER_SPECS_QSV: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        name="hevc",
        encoder_name="hevc_qsv",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_QSV_HEVC_OPTIONS),
        ten_bit=True,
        encoder_pix_fmt="p010le",
        encoder_pix_fmt_8bit="nv12",
        supported_settings=QSV_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="h264_qsv",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_QSV_H264_OPTIONS),
        ten_bit=False,
        encoder_pix_fmt="nv12",
        encoder_pix_fmt_8bit="nv12",
        supported_settings=QSV_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="av1_qsv",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_QSV_AV1_OPTIONS),
        ten_bit=True,
        encoder_pix_fmt="p010le",
        encoder_pix_fmt_8bit="nv12",
        supported_settings=QSV_ENCODER_SETTINGS_BY_CODEC["av1"],
    ),
}

# Software encoders keep cpu-device runs and QSV-less setups working; quality
# placeholders, not tuned. x265/svtav1 read yuv420p10le, PyAV reformats P010.
DEFAULT_SOFTWARE_HEVC_OPTIONS: dict[str, str] = {"preset": "medium", "crf": "21", "g": "250"}
DEFAULT_SOFTWARE_H264_OPTIONS: dict[str, str] = {"preset": "medium", "crf": "20", "g": "250"}
DEFAULT_SOFTWARE_AV1_OPTIONS: dict[str, str] = {"preset": "6", "crf": "30", "g": "250"}

ENCODER_SPECS_SOFTWARE: dict[str, EncoderSpec] = {
    "hevc": EncoderSpec(
        name="hevc",
        encoder_name="libx265",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_SOFTWARE_HEVC_OPTIONS),
        ten_bit=True,
        encoder_pix_fmt="yuv420p10le",
        encoder_pix_fmt_8bit="yuv420p",
        supported_settings=SOFTWARE_ENCODER_SETTINGS_BY_CODEC["hevc"],
    ),
    "h264": EncoderSpec(
        name="h264",
        encoder_name="libx264",
        frame_format="nv12",
        default_options=MappingProxyType(DEFAULT_SOFTWARE_H264_OPTIONS),
        ten_bit=False,
        encoder_pix_fmt="nv12",
        encoder_pix_fmt_8bit="nv12",
        supported_settings=SOFTWARE_ENCODER_SETTINGS_BY_CODEC["h264"],
    ),
    "av1": EncoderSpec(
        name="av1",
        encoder_name="libsvtav1",
        frame_format="p010le",
        default_options=MappingProxyType(DEFAULT_SOFTWARE_AV1_OPTIONS),
        ten_bit=True,
        encoder_pix_fmt="yuv420p10le",
        encoder_pix_fmt_8bit="yuv420p",
        supported_settings=SOFTWARE_ENCODER_SETTINGS_BY_CODEC["av1"],
    ),
}

_CODEC_MAP = {spec.name: spec.encoder_name for spec in ENCODER_SPECS.values()}


@lru_cache(maxsize=1)
def _qsv_encoders_usable() -> bool:
    from jasna.os_utils import check_qsv_available

    ok, info = check_qsv_available()
    if not ok:
        logger.warning("QSV encoders unavailable (%s); falling back to software encoding", info)
    return ok


def encoder_specs_for_device(device: torch.device) -> dict[str, EncoderSpec]:
    media = hw_media(device)
    if media == "nvdec_nvenc":
        return ENCODER_SPECS
    if media == "qsv" and _qsv_encoders_usable():
        return ENCODER_SPECS_QSV
    return ENCODER_SPECS_SOFTWARE


def _downgraded_pix_fmt(spec: EncoderSpec) -> str:
    """encoder_pix_fmt for a spec forced from 10-bit down to 8-bit NV12."""
    if spec.encoder_pix_fmt == "cuda":
        return "cuda"
    if spec.encoder_pix_fmt == spec.frame_format:  # direct-consuming (QSV)
        return "nv12"
    return "yuv420p"  # software 8-bit hevc/av1

# ITU-T H.273 matrix, primaries, and transfer-characteristic code points.
_COLOR_TAGS = {
    AvColorspace.ITU709: (1, 1, 1),
    AvColorspace.ITU601: (6, 6, 6),
    AvColorspace.BT2020: (9, 9, 14),  # bt2020nc, bt2020 primaries, bt2020-10 transfer
}
_COLOR_PRIMARIES = {
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "bt2020": 9,
}
_COLOR_TRANSFERS = {
    "bt709": 1,
    "smpte170m": 6,
    "bt2020-10": 14,
    "smpte2084": 16,
    "arib-std-b67": 18,
}
_COLOR_CONVERTERS = {
    (AvColorspace.ITU709, AvColorRange.MPEG): chw_rgb_to_p010_bt709_limited,
    (AvColorspace.ITU709, AvColorRange.JPEG): chw_rgb_to_p010_bt709_full,
    (AvColorspace.ITU601, AvColorRange.MPEG): chw_rgb_to_p010_bt601_limited,
    (AvColorspace.ITU601, AvColorRange.JPEG): chw_rgb_to_p010_bt601_full,
    (AvColorspace.BT2020, AvColorRange.MPEG): chw_rgb_to_p010_bt2020_limited,
    (AvColorspace.BT2020, AvColorRange.JPEG): chw_rgb_to_p010_bt2020_full,
}
_COLOR_CONVERTERS_NV12 = {
    (AvColorspace.ITU709, AvColorRange.MPEG): chw_rgb_to_nv12_bt709_limited,
    (AvColorspace.ITU709, AvColorRange.JPEG): chw_rgb_to_nv12_bt709_full,
    (AvColorspace.ITU601, AvColorRange.MPEG): chw_rgb_to_nv12_bt601_limited,
    (AvColorspace.ITU601, AvColorRange.JPEG): chw_rgb_to_nv12_bt601_full,
    (AvColorspace.BT2020, AvColorRange.MPEG): chw_rgb_to_nv12_bt2020_limited,
    (AvColorspace.BT2020, AvColorRange.JPEG): chw_rgb_to_nv12_bt2020_full,
}


def _option_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class VideoEncoder:
    def __init__(
        self,
        file: str,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        codec: str,
        encoder_settings: dict[str, object],
        lut_path: str | Path | None = None,
        output_fps: Fraction | None = None,
        mux_audio: bool = True,
        pts_origin: int = 0,
        match_input_bit_depth: bool = False,
        smart_fragment: bool = False,
    ):
        specs = encoder_specs_for_device(device)
        if codec not in specs:
            raise ValueError(f"Unsupported codec: {codec}")
        spec = specs[codec]
        if match_input_bit_depth and codec in {"hevc", "av1"} and not metadata.is_10bit:
            options = dict(spec.default_options)
            if codec == "hevc":
                options["profile"] = "main"
            spec = EncoderSpec(
                name=spec.name,
                encoder_name=spec.encoder_name,
                frame_format="nv12",
                default_options=MappingProxyType(options),
                ten_bit=False,
                encoder_pix_fmt=spec.encoder_pix_fmt_8bit,
                encoder_pix_fmt_8bit=spec.encoder_pix_fmt_8bit,
                supported_settings=spec.supported_settings,
            )
        converter_map = _COLOR_CONVERTERS if spec.frame_format == "p010le" else _COLOR_CONVERTERS_NV12
        converter = converter_map.get((metadata.color_space, metadata.color_range))
        if converter is None:
            raise ValueError(f"Unsupported color space or color range: {metadata.color_space} {metadata.color_range}")
        if encoder_settings:
            # Strict per-backend check: the CLI already validated against the
            # cross-backend union, this rejects e.g. nvenc-only options on QSV
            # up front instead of mid-encode with a partial output file.
            validate_encoder_settings(
                encoder_settings,
                codec=codec,
                supported=spec.supported_settings,
                scope_name=spec.encoder_name,
            )

        self.metadata = metadata
        self.device = device
        self.file = file
        self.output_path = Path(file)
        self.codec = codec
        self.spec = spec
        self.encoder_name = spec.encoder_name
        self.mux_audio = bool(mux_audio)
        self.pts_origin = int(pts_origin)
        self.smart_fragment = bool(smart_fragment)
        self.output_fps = Fraction(
            metadata.video_fps_exact if output_fps is None else output_fps
        )

        self._lut_applier: GpuLutApplier | None = None
        if lut_path:
            lut = parse_cube_file(lut_path)
            self._lut_applier = GpuLutApplier(lut, device)

        self._to_yuv = converter

        self.encoder_options = dict(spec.default_options)
        if encoder_settings:
            overrides = {k: _option_value(v) for k, v in encoder_settings.items()}
            # FFmpeg accepts both spellings for HEVC/H.264, but their defaults
            # use the underscore key. Normalize the alias so a user override
            # replaces that default instead of passing two conflicting options.
            if "spatial-aq" in overrides and "spatial_aq" in self.encoder_options:
                overrides["spatial_aq"] = overrides.pop("spatial-aq")
            self.encoder_options.update(overrides)
        if self.smart_fragment:
            self.encoder_options["forced-idr"] = "1"

        self.BUFFER_MAX_SIZE = 8
        self._lut_flags: deque[bool] = deque()

    def __enter__(self):
        try:
            av.Codec(self.encoder_name, "w")
        except ValueError as exc:  # av.codec.codec.UnknownCodecError
            raise RuntimeError(
                f"Encoder {self.encoder_name} (codec {self.codec}) is not available in the "
                f"bundled FFmpeg libraries: {exc}"
            ) from exc
        self._src = av.open(self.metadata.video_file)

        container_options = {}
        if self.output_path.suffix.lower() in {".mp4", ".mov"}:
            container_options["movflags"] = "+faststart"
        self.dst = av.open(str(self.output_path), "w", container_options=container_options)
        self.dst.metadata.update(self._src.metadata)

        out_v = self.dst.add_stream(
            self.encoder_name,
            rate=self.output_fps,
            options=dict(self.encoder_options),
        )
        out_v.width = self.metadata.video_width
        out_v.height = self.metadata.video_height
        out_v.time_base = self.metadata.time_base
        ctx = out_v.codec_context
        ctx.time_base = self.metadata.time_base
        ctx.framerate = self.output_fps
        # Never leave PyAV's yuv420p default in place: NVENC needs the "cuda"
        # hardware format and QSV rejects yuv420p outright (wants nv12/p010le).
        ctx.pix_fmt = self.spec.encoder_pix_fmt
        if self.smart_fragment:
            from av.codec.context import Flags

            ctx.flags |= Flags.closed_gop
        if self.metadata.sample_aspect_ratio != 1:
            ctx.sample_aspect_ratio = self.metadata.sample_aspect_ratio
        matrix, primaries, transfer = _COLOR_TAGS[self.metadata.color_space]
        primaries = _COLOR_PRIMARIES.get(self.metadata.color_primaries.lower(), primaries)
        transfer = _COLOR_TRANSFERS.get(self.metadata.color_transfer.lower(), transfer)
        ctx.color_range = int(self.metadata.color_range)
        ctx.colorspace = matrix
        ctx.color_primaries = primaries
        ctx.color_trc = transfer
        self.out_stream = out_v

        self._setup_audio()

        # Wrap torch's already-current primary context.  FFmpeg's primary_ctx
        # mode tries to change its scheduling flags and fails once torch has
        # initialized it; current_ctx leaves the context and its flags alone.
        # Keeping conversion and NVENC in one context also avoids a ~500 MiB
        # secondary CUDA context and cross-context scheduling overhead.
        self._cuda_ctx = None
        if self.device.type == "cuda":
            self._cuda_ctx = CudaContext(
                device_id=self.device.index or 0,
                primary_ctx=False,
                current_ctx=True,
            )
        self.stream = gpu_mod(self.device).Stream(self.device)
        self._host_packed: torch.Tensor | None = None
        self.pts_heap: list[int] = []
        self.frame_buffer: deque = deque()
        self._lut_flags.clear()
        self.pts_set: set[int] = set()
        self._video_started = False
        self._options_validated = False
        self._worker_error: Exception | None = None

        self._stop_sentinel = object()
        self._encode_queue: queue.Queue = queue.Queue(maxsize=self.BUFFER_MAX_SIZE)
        self._encode_thread = threading.Thread(target=self._encode_worker, name="VideoEncoderWorker", daemon=True)
        self._encode_thread.start()
        return self

    def _setup_audio(self):
        self._audio_pipes: dict[int, tuple[str, object, object]] = {}
        self._audio_backlog: deque = deque()
        self._audio_iter = None
        if not self.mux_audio:
            return
        audio_streams = list(self._src.streams.audio)
        if not audio_streams:
            return
        for in_a in audio_streams:
            if needs_audio_reencode(in_a.codec_context.name, self.output_path.suffix):
                logger.info("re-encoding audio %s -> aac for %s", in_a.codec_context.name, self.output_path.suffix)
                out_a = self.dst.add_stream("aac", rate=in_a.codec_context.sample_rate)
                out_a.codec_context.layout = in_a.codec_context.layout
                out_a.bit_rate = 256_000
                resampler = av.AudioResampler(
                    format="fltp",
                    layout=in_a.codec_context.layout,
                    rate=in_a.codec_context.sample_rate,
                )
                self._audio_pipes[in_a.index] = ("transcode", out_a, resampler)
            else:
                out_a = self.dst.add_stream_from_template(in_a)
                self._audio_pipes[in_a.index] = ("copy", out_a, None)
            # add_stream_from_template copies neither of these
            out_a.metadata.update(in_a.metadata)
            out_a.disposition = in_a.disposition
        self._audio_iter = self._src.demux(audio_streams)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                while self.frame_buffer:
                    self._process_buffer(flush_all=True)
            self._encode_queue.join()
            self._encode_queue.put(self._stop_sentinel)
            self._encode_thread.join()

            if exc_type is None and self._worker_error is None and self.out_stream.codec_context.is_open:
                for packet in self.out_stream.encode(None):
                    self._mux_video(packet)
                if self.mux_audio:
                    self._drain_audio()
        finally:
            self.dst.close()
            self._src.close()
        if exc_type is None and self._worker_error is not None:
            raise self._worker_error

    def _encode_worker(self):
        gpu_mod(self.device).set_device(self.device)

        while True:
            item = self._encode_queue.get()
            try:
                if item is self._stop_sentinel:
                    return
                if self._worker_error is None:
                    if len(item) == 3:
                        frame, pts, ready_event = item
                        apply_lut = True
                    else:
                        frame, pts, apply_lut, ready_event = item
                    self._handle_encode_item(frame, pts, apply_lut, ready_event)
            except Exception as exc:
                self._worker_error = exc
                logger.exception("[encoder-worker] crashed")
            finally:
                self._encode_queue.task_done()

    def _build_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
        apply_lut: bool = True,
    ):
        gpu = gpu_mod(self.device)
        producer_stream = gpu.current_stream(self.device)
        ready_event = gpu.Event()
        producer_stream.record_event(ready_event)
        return frame, pts, bool(apply_lut), ready_event

    def _handle_encode_item(
        self,
        frame: torch.Tensor,
        pts: int,
        apply_lut: bool,
        ready_event,
    ) -> None:
        self.stream.wait_event(ready_event)
        if is_gpu(self.device):
            frame.record_stream(self.stream)
        self._encode_frame(frame, pts, apply_lut=apply_lut)

    def _validate_encoder_options(self):
        leftover = dict(self.out_stream.codec_context.options)
        if leftover:
            raise ValueError(f"{self.encoder_name} did not accept encoder option(s): {sorted(leftover)}")
        self._options_validated = True

    def _mux_video(self, packet: av.Packet):
        threshold = (
            float(packet.dts * packet.time_base)
            if packet.dts is not None and packet.time_base is not None
            else None
        )
        try:
            self.dst.mux(packet)
        except av.FFmpegError as exc:
            raise RuntimeError(
                f"Failed to mux {self.codec} video into '{self.output_path.suffix}' output: {exc}"
            ) from exc
        if not self._video_started:
            self._video_started = True
        if not self._options_validated:
            self._validate_encoder_options()
        if threshold is not None:
            self._pump_audio(threshold)

    def _produce_audio_packets(self, in_packet) -> list:
        kind, out_a, resampler = self._audio_pipes[in_packet.stream.index]
        if kind == "copy":
            if in_packet.dts is None and in_packet.pts is None:
                return []
            in_packet.stream = out_a
            return [in_packet]
        out_packets = []
        for aframe in in_packet.decode():
            for rframe in resampler.resample(aframe):
                out_packets.extend(out_a.encode(rframe))
        return out_packets

    def _pump_audio(self, upto_seconds: float | None):
        if self._audio_iter is None:
            return
        while True:
            if self._audio_backlog:
                packet = self._audio_backlog[0]
                ts = packet.dts if packet.dts is not None else packet.pts
                if (
                    upto_seconds is not None
                    and ts is not None
                    and float(ts * packet.time_base) > upto_seconds
                ):
                    return
                self._audio_backlog.popleft()
                self.dst.mux(packet)
                continue
            in_packet = next(self._audio_iter, None)
            if in_packet is None:
                self._audio_iter = None
                return
            self._audio_backlog.extend(self._produce_audio_packets(in_packet))

    def _drain_audio(self):
        self._pump_audio(None)
        for kind, out_a, resampler in self._audio_pipes.values():
            if kind != "transcode":
                continue
            packets = []
            for rframe in resampler.resample(None):
                packets.extend(out_a.encode(rframe))
            packets.extend(out_a.encode(None))
            for packet in packets:
                self.dst.mux(packet)

    def _process_buffer(self, flush_all=False):
        if len(self.frame_buffer) > (self.BUFFER_MAX_SIZE // 2) or (flush_all and self.frame_buffer):
            frame_to_encode = self.frame_buffer.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)
            apply_lut = self._lut_flags.popleft() if self._lut_flags else True
            if apply_lut:
                item = self._build_encode_item(frame_to_encode, pts_to_assign)
            else:
                item = self._build_encode_item(frame_to_encode, pts_to_assign, False)
            self._encode_queue.put(item)

    def _encoder_open_error(self, exc: Exception) -> RuntimeError:
        try:
            gpu = gpu_mod(self.device).get_device_name(self.device)
        except Exception:
            gpu = str(self.device)
        message = (
            f"Failed to open {self.codec} encoder ({self.encoder_name}) for "
            f"'{self.output_path.suffix}' output on {gpu}: {exc}"
        )
        if self.codec == "av1":
            message += ". AV1 NVENC encoding requires a GPU/driver generation that provides it."
        return RuntimeError(message)

    def _encode_frame(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        gpu = gpu_mod(self.device)
        with gpu.stream(self.stream):
            if apply_lut and self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            packed = self._to_yuv(frame)
            if self.device.type != "cuda" and packed.device.type != "cpu":
                # Reusable pinned buffer: a fresh pageable tensor per frame
                # would force a synchronous D2H copy and churn the allocator.
                if (
                    self._host_packed is None
                    or self._host_packed.shape != packed.shape
                    or self._host_packed.dtype != packed.dtype
                ):
                    self._host_packed = host_buffer(tuple(packed.shape), packed.dtype, want_pinned=True)
                self._host_packed.copy_(packed, non_blocking=True)
                packed = self._host_packed

        height = self.metadata.video_height
        if self.device.type == "cuda":
            # NVENC consumes these pointers asynchronously, so finish the conversion
            # before constructing the hardware frame on the shared CUDA context.
            self.stream.synchronize()
            if self.spec.frame_format == "p010le":
                planes = [packed[:height].view(torch.uint16), packed[height:].view(torch.uint16)]
            else:
                planes = [packed[:height], packed[height:]]
            out_frame = av.VideoFrame.from_dlpack(
                planes,
                format=self.spec.frame_format,
                cuda_context=self._cuda_ctx,
            )
        else:
            # QSV and software encoders consume system-memory frames; the
            # packed tensor was staged to the host on the encode stream above.
            self.stream.synchronize()
            out_frame = self._build_host_frame(packed, height)
        out_frame.pts = pts
        out_frame.time_base = self.metadata.time_base
        try:
            packets = self.out_stream.encode(out_frame)
        except av.FFmpegError as exc:
            if not self._video_started:
                raise self._encoder_open_error(exc) from exc
            raise
        for packet in packets:
            self._mux_video(packet)

    def _build_host_frame(self, packed: torch.Tensor, height: int) -> av.VideoFrame:
        """Packed host NV12/P010 rows -> an av.VideoFrame in system memory.

        The packed layout is (H + H/2, W) uint8 for NV12 and (H + H/2, W)
        int16 for P010. Rows are copied as raw bytes because the frame's
        line_size may include padding beyond the visible width.
        """
        vf = av.VideoFrame(self.metadata.video_width, height, self.spec.frame_format)
        packed_bytes = packed if packed.dtype == torch.uint8 else packed.view(torch.uint8)
        y_src = packed_bytes[:height].numpy()
        uv_src = packed_bytes[height:].numpy()
        row_bytes = y_src.shape[1]
        y_plane, uv_plane = vf.planes
        np.frombuffer(y_plane, np.uint8).reshape(-1, y_plane.line_size)[:, :row_bytes] = y_src
        np.frombuffer(uv_plane, np.uint8).reshape(-1, uv_plane.line_size)[:, :row_bytes] = uv_src
        return vf

    def encode(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True):
        if self._worker_error is not None:
            raise self._worker_error
        pts = int(pts) - self.pts_origin
        while pts in self.pts_set:
            pts += 1
        heapq.heappush(self.pts_heap, pts)
        self.frame_buffer.append(frame)
        self._lut_flags.append(bool(apply_lut))
        self.pts_set.add(pts)
        self._process_buffer()


# Historical name, kept for existing imports and test patch targets; the class
# has been device-generic (NVENC / QSV / software) since the Intel Arc port.
NvidiaVideoEncoder = VideoEncoder
