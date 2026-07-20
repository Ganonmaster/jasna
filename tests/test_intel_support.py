from __future__ import annotations

import errno
import logging
from contextlib import nullcontext
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import (
    AcceleratorVendor,
    capabilities_for_device,
    deform_conv2d_backend,
    is_intel_device,
    vendor_for_device,
)
from jasna.media import VideoMetadata, validate_encoder_settings


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mp4",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=False,
    )


def test_xpu_is_a_distinct_device_type_reported_as_intel() -> None:
    # Unlike ROCm (a cuda masquerade), xpu is a genuine third device type and
    # must be recognized off device.type alone — no torch.version shim needed.
    assert vendor_for_device("xpu:0") is AcceleratorVendor.INTEL
    assert is_intel_device(torch.device("xpu:0")) is True
    assert is_intel_device(torch.device("cuda:0")) is False

    capabilities = capabilities_for_device("xpu:0")
    assert capabilities.xpu is True
    assert capabilities.tensorrt is False
    assert capabilities.migraphx is False
    assert capabilities.amf is False
    assert capabilities.nvcodec is False


def test_intel_basicvsrpp_skips_tensorrt_compilation(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda _device: False)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device: False)
    monkeypatch.setattr(accelerator, "is_intel_device", lambda _device: True)
    monkeypatch.setattr(
        compiler,
        "_basicvsrpp_engines_exist",
        MagicMock(side_effect=AssertionError("TensorRT probe on Intel")),
    )
    result = compiler.ensure_engines_compiled(
        compiler.EngineCompilationRequest(
            device="xpu:0",
            fp16=True,
            basicvsrpp=True,
            basicvsrpp_model_path="model.pth",
        )
    )
    assert result.use_basicvsrpp_tensorrt is False


def test_deform_conv2d_backend_uses_grid_sample_on_xpu() -> None:
    # torchvision ships no xpu deform_conv2d kernel; the grid_sample composition
    # is the reason Intel needs a backend switch that cuda/AMD never hit.
    assert deform_conv2d_backend(torch.device("xpu:0")) == "grid_sample"
    assert deform_conv2d_backend(torch.device("cuda:0")) == "torchvision"
    assert deform_conv2d_backend(torch.device("cpu")) == "torchvision"


def test_qsv_encoder_settings_are_vendor_specific() -> None:
    assert validate_encoder_settings(
        {"global_quality": 22, "low_power": 1},
        codec="h264",
        vendor=AcceleratorVendor.INTEL,
    ) == {"global_quality": 22, "low_power": 1}
    # temporal-aq is an NVENC/AMF knob; QSV has no such option.
    with pytest.raises(ValueError, match="temporal-aq"):
        validate_encoder_settings(
            {"temporal-aq": 1},
            codec="h264",
            vendor=AcceleratorVendor.INTEL,
        )


def test_video_encoder_selects_qsv(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.INTEL,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("xpu:0"),
        _metadata(),
        codec="h264",
        encoder_settings={"global_quality": 21},
    )
    assert encoder.encoder_name == "h264_qsv"
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["global_quality"] == "21"


def test_smart_render_is_rejected_on_intel(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.INTEL,
    )
    with pytest.raises(ValueError, match="only with NVENC"):
        module.NvidiaVideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("xpu:0"),
            _metadata(),
            codec="h264",
            encoder_settings={},
            smart_fragment=True,
        )


def test_streaming_encoder_selects_qsv(monkeypatch, tmp_path) -> None:
    import jasna.streaming_encoder as module

    monkeypatch.setattr(module, "find_executable", lambda _name: "/ffmpeg")
    popen = MagicMock()
    popen.stderr = []
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(return_value=popen))
    encoder = module.StreamingEncoder(
        tmp_path,
        4.0,
        _metadata(),
        "missing.mp4",
        torch.device("xpu:0"),
    )
    encoder._vendor = AcceleratorVendor.INTEL
    encoder._launch_ffmpeg(0)
    cmd = module.subprocess.Popen.call_args.args[0]
    assert cmd[cmd.index("-c:v") + 1] == "h264_qsv"
    assert "-global_quality" in cmd
    assert "h264_nvenc" not in cmd
    assert "h264_amf" not in cmd


class _FakeCopyBackDecoder:
    """A copy-back (hwaccel-less) decoder context. PyAV rejects setting stream
    properties like time_base on such a context ("Cannot access 'time_base' as a
    decoder"), so this fake raises on those to catch a regression that reintroduces
    the setters — the decoder must be configured with options + extradata only."""

    _DECODER_ONLY_PROPS = frozenset(
        {"time_base", "width", "height", "framerate", "sample_aspect_ratio"}
    )

    def __init__(self) -> None:
        object.__setattr__(self, "options", None)
        object.__setattr__(self, "extradata", None)
        object.__setattr__(self, "opened_strict", None)

    def __setattr__(self, name, value):
        if name in self._DECODER_ONLY_PROPS:
            raise ValueError(f"Cannot access '{name}' as a decoder")
        object.__setattr__(self, name, value)

    def open(self, strict):
        object.__setattr__(self, "opened_strict", strict)


def test_qsv_encoder_uses_no_hwaccel_device() -> None:
    # QSV must NOT get an explicit HWAccel device context: av_hwdevice_ctx_create
    # for qsv fails on the bundled FFmpeg ("No supported child device type is
    # enabled"). h264_qsv builds its own internal oneVPL session from the packed
    # system-memory frame instead. Only AMF takes a HWAccel.
    import inspect
    import re

    import jasna.media.video_encoder as module

    source = inspect.getsource(module)
    hwaccel_kinds = set(re.findall(r'HWAccel\(\s*["\'](\w+)["\']', source))
    assert "qsv" not in hwaccel_kinds, (
        f"QSV must not use an explicit HWAccel device context; found {hwaccel_kinds}"
    )


def test_qsv_decoder_context_is_created(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    decoder = _FakeCopyBackDecoder()
    create = MagicMock(return_value=decoder)
    monkeypatch.setattr(
        module.av, "CodecContext", SimpleNamespace(create=create)
    )
    monkeypatch.setattr("jasna.media.qsv.qsv_decoder_name", lambda _codec: "h264_qsv")
    monkeypatch.setattr("jasna.media.qsv.qsv_decoder_available", lambda _codec: True)
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("xpu:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_qsv_decoder(source)
    # If a stream-prop setter is reintroduced, _FakeCopyBackDecoder raises, the
    # method's except swallows it, and _decoder_ctx stays None — failing here.
    assert create.call_args.args[:2] == ("h264_qsv", "r")
    assert decoder.options == {"gpu_copy": "on"}
    assert decoder.extradata == b"header"
    assert decoder.opened_strict is False
    assert reader._decoder_ctx is decoder


class _FakeContainer:
    def __init__(self, packets):
        self.packets = packets
        self.seeks: list[int] = []

    def demux(self, _stream):
        return iter(self.packets)

    def seek(self, pts, *, stream=None, backward=False):
        self.seeks.append(pts)


def _reader_with_hw_decoder(module, hw_decoder):
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("xpu:0"),
        _metadata(),
    )
    reader.container = _FakeContainer(
        [SimpleNamespace(pts=0), SimpleNamespace(pts=1)]
    )
    reader.video_stream = SimpleNamespace(
        start_time=0,
        time_base=Fraction(1, 30),
        codec_context=SimpleNamespace(name="h264", extradata=b"header"),
    )
    reader._decoder_ctx = hw_decoder
    reader._hw_decoder = True
    return reader


def test_qsv_decoder_falls_back_to_software_when_hw_yields_no_frames(
    monkeypatch, caplog
) -> None:
    # qsvdec defers real MFX init until header packets arrive: an unsupported
    # profile passes open(strict=False) and then fails on EVERY packet. With
    # zero frames produced the reader must swap in a software decoder, rewind,
    # and re-decode transparently instead of aborting the job.
    import jasna.media.video_decoder as module

    hw = MagicMock()
    hw.name = "h264_qsv"
    hw.decode.side_effect = av.error.InvalidDataError(errno.EINVAL, "mfx init failed")
    sw = MagicMock()
    sw.decode.side_effect = lambda packet: [SimpleNamespace(pts=packet.pts)]
    create = MagicMock(return_value=sw)
    monkeypatch.setattr(module.av, "CodecContext", SimpleNamespace(create=create))

    reader = _reader_with_hw_decoder(module, hw)
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        frames = list(reader._decoded_frames(None))

    assert [frame.pts for frame in frames] == [0, 1]
    assert hw.decode.call_count == 1
    # Software context built for the stream codec and rewound to the start.
    assert create.call_args.args[:2] == ("h264", "r")
    assert sw.extradata == b"header"
    sw.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is sw
    assert reader._hw_decoder is False
    assert reader.container.seeks == [0]
    assert "falling back to FFmpeg software decoding" in caplog.text


def test_qsv_decoder_error_after_frames_stays_fatal(monkeypatch) -> None:
    # Once the hw context has produced frames, a hard decode error is real
    # mid-stream corruption: no software retry, the job must abort.
    import jasna.media.video_decoder as module

    hw = MagicMock()
    hw.name = "h264_qsv"
    hw.decode.side_effect = [
        [SimpleNamespace(pts=0)],
        av.FFmpegError(errno.EINVAL, "hard failure"),
    ]
    create = MagicMock()
    monkeypatch.setattr(module.av, "CodecContext", SimpleNamespace(create=create))

    reader = _reader_with_hw_decoder(module, hw)
    with pytest.raises(module.VideoDecodeError, match="Failed to decode"):
        list(reader._decoded_frames(None))
    create.assert_not_called()
    assert reader._decoder_ctx is hw


def test_qsv_decoder_drops_edit_list_preroll_frames() -> None:
    # The mov demuxer flags edit-list pre-roll packets AV_PKT_FLAG_DISCARD and
    # demuxes them with negative pts. Native decoders drop the decoded frames
    # (decode.c), but qsvdec has no discard handling, so the reader must drop
    # frames matching flagged packets itself — a leaked negative-pts frame
    # breaks QSV encoding (oneVPL timestamps are unsigned).
    import jasna.media.video_decoder as module

    hw = MagicMock()
    hw.name = "h264_qsv"
    # B-frame reorder: pre-roll frames surface one packet late.
    hw.decode.side_effect = [
        [],
        [SimpleNamespace(pts=-2)],
        [SimpleNamespace(pts=-1), SimpleNamespace(pts=0)],
        [SimpleNamespace(pts=1)],
    ]

    reader = _reader_with_hw_decoder(module, hw)
    reader.container = _FakeContainer(
        [
            SimpleNamespace(pts=-2, is_discard=True),
            SimpleNamespace(pts=-1, is_discard=True),
            SimpleNamespace(pts=0, is_discard=False),
            SimpleNamespace(pts=1, is_discard=False),
        ]
    )

    frames = list(reader._decoded_frames(None))
    assert [frame.pts for frame in frames] == [0, 1]


def test_discard_tracking_only_applies_to_dedicated_hw_contexts() -> None:
    # Native/software decoders already honor AV_PKT_FLAG_DISCARD in decode.c;
    # the reader must not second-guess them.
    import jasna.media.video_decoder as module

    sw = MagicMock()
    sw.decode.side_effect = lambda packet: [SimpleNamespace(pts=packet.pts)]

    reader = _reader_with_hw_decoder(module, sw)
    reader._hw_decoder = False
    reader.container = _FakeContainer(
        [
            SimpleNamespace(pts=0, is_discard=True),
            SimpleNamespace(pts=1, is_discard=False),
        ]
    )

    frames = list(reader._decoded_frames(None))
    assert [frame.pts for frame in frames] == [0, 1]


class _FakePlane(bytearray):
    line_size: int


def _fake_software_frame(pts: int, height: int, width: int) -> SimpleNamespace:
    y = _FakePlane(height * width)
    y.line_size = width
    uv = _FakePlane((height // 2) * width)
    uv.line_size = width
    return SimpleNamespace(
        pts=pts,
        planes=(y, uv),
        format=SimpleNamespace(name="yuv420p", components=[SimpleNamespace(bits=8)]),
    )


def test_software_decode_survives_pinned_memory_failure(monkeypatch, caplog) -> None:
    # xpu/ROCm route decode through _frames_software; when pinning fails (the
    # reason accelerator.host_buffer exists) decode must continue on pageable
    # buffers instead of crashing on a bare pin_memory=True allocation.
    import jasna.media.video_decoder as module

    real_empty = torch.empty
    pin_attempts = []

    def fake_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            pin_attempts.append(args)
            raise RuntimeError("pinned memory unavailable")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock())
    monkeypatch.setattr(
        module, "VideoReformatter", lambda: SimpleNamespace(reformat=lambda frame, **_: frame)
    )
    monkeypatch.setattr(module, "new_stream", lambda _device: MagicMock())
    monkeypatch.setattr(module, "stream_context", lambda _stream: nullcontext())
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.INTEL)

    reader = module.NvidiaVideoReader(
        "input.mp4",
        2,
        torch.device("cpu"),
        _metadata(),
    )
    reader.height = 16
    reader.width = 16
    reader._full_range = False

    group = [_fake_software_frame(0, 16, 16), _fake_software_frame(1, 16, 16)]
    with caplog.at_level(logging.WARNING, logger="jasna.accelerator"):
        batches = list(reader._frames_software(iter(()), group))

    assert pin_attempts, "expected a pinned allocation attempt"
    assert "Pinned host memory unavailable" in caplog.text
    assert len(batches) == 1
    batch, pts = batches[0]
    assert batch.shape == (2, 3, 16, 16)
    assert pts == [0, 1]
