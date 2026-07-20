import ctypes
import logging
import sys
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange, VideoReformatter

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    host_buffer,
    new_stream,
    stream_context,
    vendor_for_device,
)
from jasna.media import VideoMetadata, codec_open_lock, resolve_video_start_pts
from jasna.media.yuv_to_rgb import YuvToRgbConverter

log = logging.getLogger(__name__)

CORRUPT_PACKET_TOLERANCE = 10
_libcuda: ctypes.CDLL | None = None


class VideoDecodeError(RuntimeError):
    pass


class _HardwareDecodeInitError(Exception):
    """Dedicated *_qsv/*_amf context failed before producing any frame.

    qsvdec/AMF defer real session init until header packets arrive, so an
    unsupported profile (H.264 High 4:4:4, 12-bit HEVC, ...) passes open()
    and only fails per-packet. Zero frames decoded so far marks that
    deferred-init failure — distinct from mid-stream corruption, which stays
    subject to CORRUPT_PACKET_TOLERANCE — so the reader can retry in software.
    """

    def __init__(self, error: av.FFmpegError):
        super().__init__(str(error))
        self.error = error


def _cuda_driver() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        lib = loader("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuStreamCreate.restype = ctypes.c_int
        lib.cuStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cuStreamDestroy.restype = ctypes.c_int
        _libcuda = lib
    return _libcuda


def _create_blocking_cuda_stream(device: torch.device) -> tuple[int, torch.cuda.ExternalStream]:
    handle = ctypes.c_void_p()
    result = _cuda_driver().cuStreamCreate(ctypes.byref(handle), 0)
    if result != 0 or handle.value is None:
        raise RuntimeError(f"cuStreamCreate failed (CUDA error {result})")
    return handle.value, torch.cuda.ExternalStream(handle.value, device=device)


class NvidiaVideoReader:
    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        frame_stride: int = 1,
    ):
        frame_stride = int(frame_stride)
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        self.device = device
        self.file = file
        self.batch_size = batch_size
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.vendor = vendor_for_device(device)
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._hw_decoder = False
        self._frames_decoded = False

    def __enter__(self):
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._hw_decoder = False
        self._frames_decoded = False
        current_stream(self.device)
        # Serialize hardware codec init across pipeline threads (see codec_open_lock):
        # concurrent QSV/oneVPL avcodec_open2 deadlocks.
        with codec_open_lock:
            try:
                if self.vendor is AcceleratorVendor.NVIDIA:
                    hwaccel = HWAccel(
                        "cuda",
                        device=str(self.device.index or 0),
                        allow_software_fallback=True,
                        is_hw_owned=True,
                    )
                    # Reuse torch's current primary context without changing its
                    # scheduling flags.
                    hwaccel.options["primary_ctx"] = "0"
                    hwaccel.options["current_ctx"] = "1"
                    self.container = av.open(self.file, hwaccel=hwaccel)
                else:
                    self.container = av.open(self.file)
                self.video_stream = self.container.streams.video[0]
            except av.FFmpegError as e:
                raise VideoDecodeError(f"Failed to open {self.file}: {e}") from e

            ctx = self.video_stream.codec_context
            if self.vendor is AcceleratorVendor.AMD:
                self._setup_amf_decoder(ctx)
            elif self.vendor is AcceleratorVendor.INTEL:
                self._setup_qsv_decoder(ctx)
            elif not ctx.is_hwaccel:
                # Definite software decode: let FFmpeg pick frame/slice threading.
                # CUDA contexts must keep their default threading configuration.
                ctx.thread_type = "AUTO"
        self.width = ctx.width
        self.height = ctx.height
        self._full_range = (
            ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )
        self._raw_stream: int | None = None
        return self

    def _setup_amf_decoder(self, source_ctx) -> None:
        decoder_name = {
            "h264": "h264_amf",
            "hevc": "hevc_amf",
            "av1": "av1_amf",
        }.get(str(source_ctx.name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        try:
            hwaccel = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                is_hw_owned=False,
            )
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            decoder.time_base = source_ctx.time_base
            decoder.framerate = source_ctx.framerate
            decoder.sample_aspect_ratio = source_ctx.sample_aspect_ratio
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            self._amd_hardware_decode = True
            self._hw_decoder = True
            log.info("Using AMF hardware decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "AMF cannot decode %s (codec %s): %s; using FFmpeg software "
                "decoding and uploading frames to ROCm",
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def _setup_qsv_decoder(self, source_ctx) -> None:
        from jasna.media.qsv import qsv_decoder_available, qsv_decoder_name

        decoder_name = qsv_decoder_name(str(source_ctx.name))
        if decoder_name is None or not qsv_decoder_available(self.metadata.codec_name):
            source_ctx.thread_type = "AUTO"
            return
        try:
            # Copy-back mode: the *_qsv decoder + gpu_copy=on decodes on the GPU
            # and hands back system-memory NV12/P010 frames (no HWAccel object),
            # which _frames_software then reformats and uploads to xpu.
            decoder = av.CodecContext.create(decoder_name, "r")
            decoder.options = {"gpu_copy": "on"}
            if source_ctx.extradata:
                decoder.extradata = source_ctx.extradata
            # Unlike the AMF decoder (created WITH a hwaccel), this copy-back
            # context is hwaccel-less, and PyAV rejects setting stream props like
            # time_base on it ("Cannot access 'time_base' as a decoder"). The
            # decoder recovers width/height/timing from the bitstream + extradata,
            # so only options + extradata are needed before open().
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            self._hw_decoder = True
            log.info("Using QSV hardware decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "QSV cannot decode %s (codec %s): %s; using FFmpeg software "
                "decoding and uploading frames to xpu",
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()
        self._decoder_ctx = None
        if self._raw_stream is None:
            return
        result = _cuda_driver().cuStreamDestroy(ctypes.c_void_p(self._raw_stream))
        if result != 0 and exc_type is None:
            raise RuntimeError(f"cuStreamDestroy failed (CUDA error {result})")

    def _deferred_hw_init_failure(self) -> bool:
        return (
            getattr(self, "_decoder_ctx", None) is not None
            and self._hw_decoder
            and not self._frames_decoded
        )

    def _decode_packet(self, packet, consecutive_errors: int) -> tuple[list, int]:
        try:
            frames = (
                self._decoder_ctx.decode(packet)
                if getattr(self, "_decoder_ctx", None) is not None
                else packet.decode()
            )
        except av.error.InvalidDataError as e:
            if self._deferred_hw_init_failure():
                raise _HardwareDecodeInitError(e) from e
            consecutive_errors += 1
            if consecutive_errors > CORRUPT_PACKET_TOLERANCE:
                raise VideoDecodeError(
                    f"Failed to decode {self.file}: too many consecutive corrupt packets "
                    f"({consecutive_errors}): {e}"
                ) from e
            log.warning("Recovered video corruption in %s: %s", self.file, e)
            return [], consecutive_errors
        except av.FFmpegError as e:
            if self._deferred_hw_init_failure():
                raise _HardwareDecodeInitError(e) from e
            raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
        if frames:
            consecutive_errors = 0
            self._frames_decoded = True
        return frames, consecutive_errors

    def _fall_back_to_software_decoder(self, error: Exception) -> None:
        hw_name = getattr(self._decoder_ctx, "name", None) or "hardware decoder"
        # Drop the hw context first (PyAV has no explicit close; freed on GC) so
        # a failure opening the software context cannot leave it half-active.
        self._decoder_ctx = None
        self._hw_decoder = False
        self._amd_hardware_decode = False
        log.warning(
            "%s produced no frames for %s (codec %s): %s; falling back to FFmpeg "
            "software decoding",
            hw_name,
            self.file,
            self.metadata.codec_name,
            error,
        )
        source_ctx = self.video_stream.codec_context
        try:
            with codec_open_lock:
                # Same constraint as the QSV copy-back context: hwaccel-less, so
                # only options + extradata may be set before open().
                decoder = av.CodecContext.create(str(source_ctx.name), "r")
                if source_ctx.extradata:
                    decoder.extradata = source_ctx.extradata
                decoder.thread_type = "AUTO"
                decoder.open(strict=False)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            raise VideoDecodeError(
                f"Failed to open software decoder for {self.file} after "
                f"{hw_name} failure: {exc}"
            ) from exc
        self._decoder_ctx = decoder

    def _position_at(self, seek_ts: float | None) -> int | None:
        """Seek to seek_ts (file start when None) and flush the active decoder."""
        start = resolve_video_start_pts(
            self.video_stream.start_time,
            self.metadata.start_pts,
        )
        target_pts = None
        if seek_ts is not None:
            target_pts = start + round(seek_ts / self.video_stream.time_base)
        self.container.seek(
            start if target_pts is None else target_pts,
            stream=self.video_stream,
            backward=True,
        )
        if self._decoder_ctx is not None:
            self._decoder_ctx.flush_buffers()
        return target_pts

    def _decoded_frames(self, seek_ts: float | None):
        target_pts = None
        if seek_ts is not None:
            target_pts = self._position_at(seek_ts)

        consecutive_errors = 0
        # ffmpeg's native decoders drop the frames of AV_PKT_FLAG_DISCARD
        # packets (mp4 edit-list pre-roll, demuxed with negative pts), but the
        # dedicated qsv/amf contexts have no discard handling, and a leaked
        # negative-pts frame corrupts QSV encoding downstream (oneVPL
        # timestamps are unsigned). Track flagged packets by pts and drop the
        # matching decoded frames ourselves. The pts join is best-effort:
        # qsvdec round-trips pts through the unsigned mfx timestamp, where a
        # negative value wraps to ~2**64 and loses its low bits to double
        # precision inside the runtime, so pre-roll frames can come back with
        # a pts (e.g. -22528, a 2048-multiple) matching no packet — while
        # flagged packets are pending, any frame still before the presentation
        # start is such mangled pre-roll.
        discard_pts: set[int] = set()
        preroll_dropped = 0
        preroll_done = False
        start_pts = resolve_video_start_pts(
            self.video_stream.start_time,
            self.metadata.start_pts,
        )
        packets = self.container.demux(self.video_stream)
        while True:
            packet = next(packets, None)
            if packet is None:
                return
            if (
                self._hw_decoder
                and packet.pts is not None
                and getattr(packet, "is_discard", False)
            ):
                discard_pts.add(packet.pts)
            try:
                frames, consecutive_errors = self._decode_packet(packet, consecutive_errors)
            except _HardwareDecodeInitError as e:
                # Deferred hw init failure: retry the whole stream in software.
                self._fall_back_to_software_decoder(e.error)
                target_pts = self._position_at(seek_ts)
                consecutive_errors = 0
                discard_pts.clear()
                preroll_dropped = 0
                preroll_done = False
                packets = self.container.demux(self.video_stream)
                continue
            for frame in frames:
                if frame.pts is not None and discard_pts:
                    if frame.pts in discard_pts:
                        discard_pts.discard(frame.pts)
                        preroll_dropped += 1
                        continue
                    if not preroll_done and frame.pts < start_pts:
                        preroll_dropped += 1
                        continue
                if not preroll_done:
                    preroll_done = True
                    if preroll_dropped:
                        log.info(
                            "Dropped %d edit-list pre-roll frame(s) from %s",
                            preroll_dropped,
                            self.file,
                        )
                if target_pts is not None and frame.pts is not None and frame.pts < target_pts:
                    continue
                target_pts = None
                yield frame

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group

    def _selected_frames(self, decoded):
        if self.frame_stride == 1:
            yield from decoded
            return
        for frame_index, frame in enumerate(decoded):
            if frame_index % self.frame_stride == 0:
                yield frame

    def frames(
        self,
        seek_ts: float | None = None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        if seek_ts is not None and self.frame_stride != 1:
            raise ValueError(
                "frame_stride > 1 is not supported with seek_ts because frame selection "
                "must stay anchored to the start of the file"
            )
        # The first decoded frame's format is the final backend decision: a codec
        # can advertise a CUDA config and still fall back to software when
        # hardware initialization rejects a profile or pixel format. Dispatch
        # once here so neither per-frame loop carries a backend branch.
        decoded = self._selected_frames(self._decoded_frames(seek_ts))
        group = self._read_group(decoded)
        if not group:
            return
        vendor = getattr(self, "vendor", AcceleratorVendor.NVIDIA)
        if (
            vendor is AcceleratorVendor.NVIDIA
            and group[0].format.name == "cuda"
        ):
            backend = self._frames_hardware(decoded, group)
        else:
            if vendor is AcceleratorVendor.NVIDIA:
                log.warning(
                    "CUDA/NVDEC cannot decode %s (codec %s, %s); using FFmpeg "
                    "software decoding and uploading frames to CUDA",
                    self.file,
                    self.metadata.codec_name,
                    group[0].format.name,
                )
            backend = self._frames_software(decoded, group)

        # The backend generator now owns the first group. Drop this outer
        # reference before yielding: retaining four 4K P010 NVDEC surfaces here
        # for the reader's lifetime costs about 96 MiB of avoidable VRAM.
        del group
        yield from backend

    def _frames_hardware(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # FFmpeg 8 maps NVDEC output on CUDA stream 0. Conversion runs in a
        # blocking stream in that same context, so legacy-default-stream ordering
        # makes the decoded writes visible before this kernel without a race.
        # Decode one group ahead while conversion runs; keep both groups' frame
        # references alive until conversion is synchronized so their mapped
        # surfaces cannot be recycled underneath queued work.
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            self.metadata.is_10bit,
            self.device,
        )
        if self._raw_stream is None:
            self._raw_stream, self.stream = _create_blocking_cuda_stream(self.device)
        while group:
            batch = torch.empty(
                (len(group), 3, self.height, self.width), device=self.device, dtype=torch.uint8
            )
            pts = [frame.pts for frame in group]
            with torch.cuda.stream(self.stream):
                converter.convert_frames_into(group, batch, self.stream.cuda_stream)

            next_group = self._read_group(decoded)
            self.stream.synchronize()
            group = next_group
            yield batch, pts

    def _frames_software(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # Normalize CPU frames to the two layouts the CUDA conversion kernel
        # accepts (NV12 for <=8-bit sources, P010 above), keeping the resolved
        # matrix/range identical on both reformat sides so swscale changes only
        # layout/subsampling/depth. The one authoritative YUV->RGB conversion
        # stays in the CUDA kernel.
        depth = max(
            (component.bits for component in group[0].format.components if component.bits),
            default=10 if self.metadata.is_10bit else 8,
        )
        ten_bit = depth > 8
        if depth > 10:
            log.warning(
                "Reducing %d-bit source %s to 10-bit P010 before CUDA upload", depth, self.file
            )
        target_format = "p010le" if ten_bit else "nv12"
        dtype = torch.uint16 if ten_bit else torch.uint8
        bytes_per_sample = 2 if ten_bit else 1

        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            ten_bit,
            self.device,
        )
        reformatter = VideoReformatter()
        color_range = AvColorRange.JPEG if self._full_range else AvColorRange.MPEG
        H, W = self.height, self.width

        # One packed pinned host batch and one packed device staging frame bound
        # the fallback's extra memory: H2D copies and conversion kernels are
        # ordered on the same stream, so the next H2D overwrite of the staging
        # frame starts only after the prior conversion kernel consumed it.
        pinned = host_buffer((self.batch_size, H + H // 2, W), dtype, want_pinned=True)
        staging = torch.empty((H + H // 2, W), dtype=dtype, device=self.device)
        stream = new_stream(self.device)

        while group:
            batch = torch.empty((len(group), 3, H, W), device=self.device, dtype=torch.uint8)
            pts = [frame.pts for frame in group]
            for i, frame in enumerate(group):
                try:
                    normalized = reformatter.reformat(
                        frame,
                        width=W,
                        height=H,
                        format=target_format,
                        src_colorspace=self.metadata.color_space,
                        dst_colorspace=self.metadata.color_space,
                        src_color_range=color_range,
                        dst_color_range=color_range,
                    )
                except av.FFmpegError as e:
                    raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
                y_plane, uv_plane = normalized.planes
                y = torch.frombuffer(y_plane, dtype=dtype).reshape(
                    H, y_plane.line_size // bytes_per_sample
                )[:, :W]
                uv = torch.frombuffer(uv_plane, dtype=dtype).reshape(
                    H // 2, uv_plane.line_size // bytes_per_sample
                )[:, :W]
                pinned[i, :H].copy_(y)
                pinned[i, H:].copy_(uv)

            with stream_context(stream):
                for i in range(len(group)):
                    staging.copy_(pinned[i], non_blocking=True)
                    converter.convert_into(
                        staging[:H], staging[H:].view(H // 2, W // 2, 2), batch[i]
                    )

            next_group = self._read_group(decoded)
            stream.synchronize()
            group = next_group
            yield batch, pts
