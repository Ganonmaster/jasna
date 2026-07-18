from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_qsv_decoder_context_is_created(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    decoder = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    monkeypatch.setattr(
        "jasna.media.qsv.qsv_decoder_name",
        lambda _codec: "h264_qsv",
    )
    monkeypatch.setattr(
        "jasna.media.qsv.qsv_decoder_available",
        lambda _codec: True,
    )
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
    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("h264_qsv", "r")
    assert decoder.options == {"gpu_copy": "on"}
    decoder.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is decoder
