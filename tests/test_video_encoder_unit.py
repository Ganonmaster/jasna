"""Unit tests for NvidiaVideoEncoder internals (options, color guard, buffer, worker, audio pump)."""
from __future__ import annotations

import queue
import threading
from contextlib import nullcontext
from collections import deque
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.video_encoder as video_encoder_module
from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata, validate_encoder_settings
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
from jasna.media.video_encoder import (
    AMF_ENCODER_SPECS,
    DEFAULT_AV1_ENCODER_OPTIONS,
    DEFAULT_ENCODER_OPTIONS,
    DEFAULT_H264_ENCODER_OPTIONS,
    ENCODER_SPECS,
    _CODEC_MAP,
    _align_yuv_pitch,
    _eight_bit_spec,
    NvidiaVideoEncoder,
)


def _fake_metadata(**overrides) -> VideoMetadata:
    defaults = dict(
        video_file="fake_input.mkv",
        num_frames=100,
        video_fps=24.0,
        average_fps=24.0,
        video_fps_exact=Fraction(24, 1),
        codec_name="hevc",
        duration=100.0 / 24.0,
        video_width=1920,
        video_height=1080,
        time_base=Fraction(1, 24),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=True,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


def _make_encoder(tmp_path, encoder_settings=None, codec="hevc", **meta_overrides) -> NvidiaVideoEncoder:
    return NvidiaVideoEncoder(
        file=str(tmp_path / "result.mkv"),
        device=torch.device("cuda:0"),
        metadata=_fake_metadata(**meta_overrides),
        codec=codec,
        encoder_settings=encoder_settings or {},
    )


def _make_qsv_encoder(tmp_path, encoder_settings=None, codec="hevc", **meta_overrides) -> NvidiaVideoEncoder:
    # torch.device("xpu:0") resolves to AcceleratorVendor.INTEL without needing
    # XPU hardware; only the constructor (and mocked internals) run in tests.
    return NvidiaVideoEncoder(
        file=str(tmp_path / "result.mkv"),
        device=torch.device("xpu:0"),
        metadata=_fake_metadata(**meta_overrides),
        codec=codec,
        encoder_settings=encoder_settings or {},
    )


# Pre-PR hevc_nvenc configuration; the HEVC path must never drift from it.
_HEVC_OPTIONS_SNAPSHOT = {
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


class TestCodecSpecs:
    def test_public_to_ffmpeg_codec_mapping(self):
        assert _CODEC_MAP == {"hevc": "hevc_nvenc", "h264": "h264_nvenc", "av1": "av1_nvenc"}

    def test_hevc_defaults_snapshot_unchanged(self):
        assert DEFAULT_ENCODER_OPTIONS == _HEVC_OPTIONS_SNAPSHOT
        assert dict(ENCODER_SPECS["hevc"].default_options) == _HEVC_OPTIONS_SNAPSHOT

    def test_h264_defaults_snapshot(self):
        expected = dict(_HEVC_OPTIONS_SNAPSHOT)
        expected["profile"] = "high"
        expected["cq"] = "24"
        del expected["lookahead_level"]
        assert DEFAULT_H264_ENCODER_OPTIONS == expected

    def test_av1_defaults_snapshot(self):
        expected = dict(_HEVC_OPTIONS_SNAPSHOT)
        del expected["profile"]
        expected["cq"] = "32"
        del expected["qmin"]
        del expected["qmax"]
        del expected["spatial_aq"]
        del expected["init_qpI"]
        del expected["init_qpP"]
        del expected["init_qpB"]
        expected["spatial-aq"] = "1"
        assert DEFAULT_AV1_ENCODER_OPTIONS == expected

    def test_av1_does_not_reuse_hevc_qp_scale(self):
        assert DEFAULT_AV1_ENCODER_OPTIONS["cq"] == "32"
        assert not {
            "qmin",
            "qmax",
            "init_qpI",
            "init_qpP",
            "init_qpB",
        } & DEFAULT_AV1_ENCODER_OPTIONS.keys()

    def test_frame_formats_and_bit_depth(self):
        assert ENCODER_SPECS["hevc"].frame_format == "p010le"
        assert ENCODER_SPECS["hevc"].ten_bit is True
        assert ENCODER_SPECS["h264"].frame_format == "nv12"
        assert ENCODER_SPECS["h264"].ten_bit is False
        assert ENCODER_SPECS["av1"].frame_format == "p010le"
        assert ENCODER_SPECS["av1"].ten_bit is True


class TestEncoderOptions:
    def test_smart_fragment_preserves_normal_closed_gop_settings(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            smart_fragment=True,
            mux_audio=False,
        )
        assert enc.encoder_options["g"] == DEFAULT_ENCODER_OPTIONS["g"]
        assert enc.encoder_options["bf"] == DEFAULT_ENCODER_OPTIONS["bf"]
        assert enc.encoder_options["forced-idr"] == "1"
        assert enc.encoder_options["b_ref_mode"] == DEFAULT_ENCODER_OPTIONS["b_ref_mode"]
        assert enc.mux_audio is False

    def test_smart_fragment_preserves_custom_gop_size(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={"g": "180"},
            smart_fragment=True,
            mux_audio=False,
        )

        assert enc.encoder_options["g"] == "180"

    @pytest.mark.parametrize("codec", ["hevc", "av1"])
    def test_smart_fragment_can_match_eight_bit_source(self, tmp_path, codec):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(is_10bit=False),
            codec=codec,
            encoder_settings={},
            match_input_bit_depth=True,
        )
        assert enc.spec.frame_format == "nv12"
        assert enc.spec.ten_bit is False
        if codec == "hevc":
            assert enc.encoder_options["profile"] == "main"

    def test_output_fps_defaults_to_source_rate(self, tmp_path):
        enc = _make_encoder(tmp_path)
        assert enc.output_fps == Fraction(24, 1)

    def test_output_fps_can_override_source_rate(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "result.mkv"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(video_fps_exact=Fraction(60_000, 1_001)),
            codec="hevc",
            encoder_settings={},
            output_fps=Fraction(30_000, 1_001),
        )
        assert enc.output_fps == Fraction(30_000, 1_001)

    def test_defaults_used_when_no_settings(self, tmp_path):
        enc = _make_encoder(tmp_path)
        assert enc.encoder_options == DEFAULT_ENCODER_OPTIONS

    @pytest.mark.parametrize(
        ("codec", "defaults"),
        [
            ("hevc", DEFAULT_ENCODER_OPTIONS),
            ("h264", DEFAULT_H264_ENCODER_OPTIONS),
            ("av1", DEFAULT_AV1_ENCODER_OPTIONS),
        ],
    )
    def test_defaults_per_codec(self, tmp_path, codec, defaults):
        enc = _make_encoder(tmp_path, codec=codec)
        assert enc.encoder_options == defaults
        assert enc.encoder_name == _CODEC_MAP[codec]

    @pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
    def test_settings_override_and_stringify(self, tmp_path, codec):
        enc = _make_encoder(tmp_path, codec=codec, encoder_settings={"cq": 22, "temporal-aq": False, "maxrate": "10M"})
        assert enc.encoder_options["cq"] == "22"
        assert enc.encoder_options["temporal-aq"] == "0"
        assert enc.encoder_options["maxrate"] == "10M"
        assert enc.encoder_options["preset"] == "p5"

    def test_unsupported_codec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported codec"):
            NvidiaVideoEncoder(
                file=str(tmp_path / "o.mkv"),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="vp9",
                encoder_settings={},
            )

    def test_codec_specific_settings_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="for codec av1.*profile"):
            _make_encoder(tmp_path, codec="av1", encoder_settings={"profile": "main"})
        with pytest.raises(ValueError, match="for codec av1.*spatial_aq"):
            _make_encoder(tmp_path, codec="av1", encoder_settings={"spatial_aq": 1})
        with pytest.raises(ValueError, match="for codec h264.*tier"):
            _make_encoder(tmp_path, codec="h264", encoder_settings={"tier": "high"})

    def test_h264_user_can_opt_into_lookahead_level(self, tmp_path):
        enc = _make_encoder(tmp_path, codec="h264", encoder_settings={"lookahead_level": 1})
        assert enc.encoder_options["lookahead_level"] == "1"

    @pytest.mark.parametrize("codec", ["hevc", "h264"])
    def test_hyphenated_spatial_aq_replaces_underscore_default(self, tmp_path, codec):
        enc = _make_encoder(tmp_path, codec=codec, encoder_settings={"spatial-aq": 0})
        assert enc.encoder_options["spatial_aq"] == "0"
        assert "spatial-aq" not in enc.encoder_options

    def test_leftover_options_raise(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.options = {"bogus": "1"}
        with pytest.raises(ValueError, match="did not accept encoder option.*bogus"):
            enc._validate_encoder_options()

    def test_no_leftover_options_pass(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.options = {}
        enc._options_validated = False
        enc._validate_encoder_options()
        assert enc._options_validated


class TestColorHandling:
    def test_unsupported_color_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported color space or color range"):
            _make_encoder(tmp_path, color_range=AvColorRange.UNSPECIFIED)

    @pytest.mark.parametrize(
        ("color_space", "color_range", "expected"),
        [
            (AvColorspace.ITU709, AvColorRange.MPEG, chw_rgb_to_p010_bt709_limited),
            (AvColorspace.ITU709, AvColorRange.JPEG, chw_rgb_to_p010_bt709_full),
            (AvColorspace.ITU601, AvColorRange.MPEG, chw_rgb_to_p010_bt601_limited),
            (AvColorspace.ITU601, AvColorRange.JPEG, chw_rgb_to_p010_bt601_full),
            (AvColorspace.BT2020, AvColorRange.MPEG, chw_rgb_to_p010_bt2020_limited),
            (AvColorspace.BT2020, AvColorRange.JPEG, chw_rgb_to_p010_bt2020_full),
        ],
    )
    @pytest.mark.parametrize("codec", ["hevc", "av1"])
    def test_selects_p010_converter_for_hevc_and_av1(self, tmp_path, codec, color_space, color_range, expected):
        enc = _make_encoder(tmp_path, codec=codec, color_space=color_space, color_range=color_range)
        assert enc._to_yuv is expected

    @pytest.mark.parametrize(
        ("color_space", "color_range", "expected"),
        [
            (AvColorspace.ITU709, AvColorRange.MPEG, chw_rgb_to_nv12_bt709_limited),
            (AvColorspace.ITU709, AvColorRange.JPEG, chw_rgb_to_nv12_bt709_full),
            (AvColorspace.ITU601, AvColorRange.MPEG, chw_rgb_to_nv12_bt601_limited),
            (AvColorspace.ITU601, AvColorRange.JPEG, chw_rgb_to_nv12_bt601_full),
            (AvColorspace.BT2020, AvColorRange.MPEG, chw_rgb_to_nv12_bt2020_limited),
            (AvColorspace.BT2020, AvColorRange.JPEG, chw_rgb_to_nv12_bt2020_full),
        ],
    )
    def test_selects_nv12_converter_for_h264(self, tmp_path, color_space, color_range, expected):
        enc = _make_encoder(tmp_path, codec="h264", color_space=color_space, color_range=color_range)
        assert enc._to_yuv is expected

    @pytest.mark.parametrize("codec", ["h264", "av1"])
    def test_unsupported_color_range_raises_for_new_codecs(self, tmp_path, codec):
        with pytest.raises(ValueError, match="Unsupported color space or color range"):
            _make_encoder(tmp_path, codec=codec, color_range=AvColorRange.UNSPECIFIED)


class TestQsvEncoderSettings:
    @pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
    def test_validate_accepts_portable_cq(self, codec):
        settings = {"cq": 23}
        assert validate_encoder_settings(
            settings, codec=codec, vendor=AcceleratorVendor.INTEL
        ) is settings

    @pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
    def test_cq_translates_to_global_quality(self, tmp_path, codec):
        enc = _make_qsv_encoder(tmp_path, codec=codec, encoder_settings={"cq": 23})
        assert enc.encoder_options["global_quality"] == "23"
        assert "cq" not in enc.encoder_options

    def test_cq_conflicts_with_global_quality(self, tmp_path):
        with pytest.raises(ValueError, match="cq and global_quality are aliases"):
            _make_qsv_encoder(
                tmp_path,
                encoder_settings={"cq": 23, "global_quality": 22},
            )


class TestEightBitSpecRebuild:
    @pytest.mark.parametrize("codec", ["hevc", "av1"])
    def test_amf_rebuild_drops_bitdepth(self, codec):
        assert AMF_ENCODER_SPECS[codec].default_options["bitdepth"] == "10"
        rebuilt = _eight_bit_spec(AMF_ENCODER_SPECS[codec])
        assert "bitdepth" not in rebuilt.default_options
        assert rebuilt.frame_format == "nv12"
        assert rebuilt.ten_bit is False

    def test_amf_hevc_rebuild_uses_main_profile(self):
        assert _eight_bit_spec(AMF_ENCODER_SPECS["hevc"]).default_options["profile"] == "main"

    def test_nvenc_hevc_rebuild_only_touches_profile(self):
        rebuilt = _eight_bit_spec(ENCODER_SPECS["hevc"])
        expected = dict(DEFAULT_ENCODER_OPTIONS)
        expected["profile"] = "main"
        assert dict(rebuilt.default_options) == expected


class _LockSpy:
    """Context-manager double for codec_open_lock recording whether it is held."""

    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True

    def __exit__(self, *exc_info):
        self.held = False


class _FakeCodecContext:
    def __init__(self, lock_spy):
        self._lock_spy = lock_spy
        self.is_open = False
        self.open_strict = None
        self.opened_under_lock = None
        self.options = {}

    def open(self, strict=True):
        self.open_strict = strict
        self.opened_under_lock = self._lock_spy.held
        self.is_open = True


class TestCodecOpenLockCoverage:
    def test_enter_opens_qsv_codec_context_under_lock(self, tmp_path, monkeypatch):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "result.mkv"),
            device=torch.device("xpu:0"),
            metadata=_fake_metadata(video_width=64, video_height=32, is_10bit=False),
            codec="h264",
            encoder_settings={},
            mux_audio=False,
        )
        spy = _LockSpy()
        ctx = _FakeCodecContext(spy)
        out_v = SimpleNamespace(codec_context=ctx)
        dst = MagicMock()
        dst.add_stream.return_value = out_v
        monkeypatch.setattr(video_encoder_module, "codec_open_lock", spy)
        monkeypatch.setattr(video_encoder_module.av, "Codec", MagicMock())
        monkeypatch.setattr(
            video_encoder_module.av, "open", MagicMock(side_effect=[MagicMock(), dst])
        )
        monkeypatch.setattr(video_encoder_module, "new_stream", MagicMock())
        monkeypatch.setattr(video_encoder_module, "set_device", MagicMock())
        try:
            enc.__enter__()
            # The real avcodec_open2 must run inside __enter__ (not at the
            # first worker-thread encode) and under codec_open_lock.
            assert ctx.is_open is True
            assert ctx.open_strict is False
            assert ctx.opened_under_lock is True
        finally:
            enc._encode_queue.put(enc._stop_sentinel)
            enc._encode_thread.join(timeout=5)

    def test_nvenc_lazy_open_first_encode_holds_lock(self, tmp_path, monkeypatch):
        enc = _make_encoder(tmp_path, codec="h264", video_width=2, video_height=2)
        enc.stream = MagicMock()
        enc._cuda_ctx = object()
        enc._lut_applier = None
        enc._to_yuv = lambda frame: torch.zeros((3, 2), dtype=torch.uint8)
        spy = _LockSpy()
        monkeypatch.setattr(video_encoder_module, "codec_open_lock", spy)
        held_during_encode = []
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.is_open = False

        def _encode(hw_frame):
            held_during_encode.append(spy.held)
            enc.out_stream.codec_context.is_open = True
            return []

        enc.out_stream.encode.side_effect = _encode
        hw_frame = SimpleNamespace(pts=None, time_base=None)
        monkeypatch.setattr(
            video_encoder_module.av,
            "VideoFrame",
            SimpleNamespace(from_dlpack=MagicMock(return_value=hw_frame)),
        )
        monkeypatch.setattr(video_encoder_module.torch.cuda, "stream", lambda stream: nullcontext())

        enc._encode_frame(torch.zeros((3, 2, 2), dtype=torch.uint8), 0)
        enc._encode_frame(torch.zeros((3, 2, 2), dtype=torch.uint8), 512)

        # NVENC's open happens inside the first encode(); it must be locked,
        # and steady-state encodes must not be.
        assert held_during_encode == [True, False]


class TestQsvFrameOwnership:
    def test_consecutive_frames_do_not_alias_staging_buffer(self, tmp_path, monkeypatch):
        enc = _make_qsv_encoder(
            tmp_path, codec="h264", video_width=4, video_height=2, is_10bit=False
        )
        enc.stream = MagicMock()
        enc._lut_applier = None
        yuv = [
            torch.arange(12, dtype=torch.uint8).reshape(3, 4),
            torch.arange(12, 24, dtype=torch.uint8).reshape(3, 4),
        ]
        staged_yuv = iter(yuv)
        enc._to_yuv = lambda frame: next(staged_yuv)
        enc._host_yuv = torch.zeros((3, 4), dtype=torch.uint8)
        enc.out_stream = MagicMock()
        enc.out_stream.encode.return_value = []
        hw_frame = SimpleNamespace(pts=None, time_base=None)
        from_dlpack = MagicMock(return_value=hw_frame)
        monkeypatch.setattr(
            video_encoder_module.av,
            "VideoFrame",
            SimpleNamespace(from_dlpack=from_dlpack),
        )
        monkeypatch.setattr(video_encoder_module.torch.cuda, "stream", lambda stream: nullcontext())

        enc._encode_frame(torch.zeros((3, 2, 4), dtype=torch.uint8), 0)
        enc._encode_frame(torch.zeros((3, 2, 4), dtype=torch.uint8), 512)

        (first_planes,), first_kwargs = from_dlpack.call_args_list[0]
        (second_planes,), second_kwargs = from_dlpack.call_args_list[1]
        assert first_kwargs == {"format": "nv12"}
        assert second_kwargs == {"format": "nv12"}
        # qsvenc encodes asynchronously and refs system-memory frames, so no
        # submitted frame may alias the reused staging buffer or another frame.
        staging_ptrs = {enc._host_yuv[:2].data_ptr(), enc._host_yuv[2:].data_ptr()}
        for planes in (first_planes, second_planes):
            assert {planes[0].data_ptr(), planes[1].data_ptr()}.isdisjoint(staging_ptrs)
        assert first_planes[0].data_ptr() != second_planes[0].data_ptr()
        # Each frame kept the pixels staged for it even after the staging
        # buffer was overwritten by the next frame.
        assert torch.equal(torch.cat([first_planes[0], first_planes[1]]), yuv[0])
        assert torch.equal(torch.cat([second_planes[0], second_planes[1]]), yuv[1])


def _buffered_encoder(tmp_path) -> NvidiaVideoEncoder:
    enc = _make_encoder(tmp_path)
    enc.pts_heap = []
    enc.frame_buffer = deque()
    enc.pts_set = set()
    enc._worker_error = None
    enc._encode_queue = MagicMock()
    enc._build_encode_item = MagicMock(side_effect=lambda frame, pts: (frame, pts, None))
    return enc


class TestEncodeBuffer:
    @pytest.mark.parametrize(
        ("dtype", "width", "expected_pitch"),
        [
            (torch.uint8, 852, 864),
            (torch.uint8, 854, 864),
            (torch.uint8, 856, 864),
            (torch.uint8, 860, 864),
            (torch.uint8, 864, 864),
            (torch.int16, 852, 856),
            (torch.int16, 854, 856),
            (torch.int16, 856, 856),
            (torch.int16, 860, 864),
            (torch.int16, 864, 864),
        ],
    )
    def test_yuv_pitch_is_aligned_without_changing_visible_data(
        self,
        dtype,
        width,
        expected_pitch,
    ):
        height = 4
        packed = torch.arange(height * 3 // 2 * width, dtype=dtype).reshape(
            height * 3 // 2,
            width,
        )

        aligned = _align_yuv_pitch(packed)

        assert aligned.shape == packed.shape
        assert torch.equal(aligned, packed)
        assert aligned.stride() == (expected_pitch, 1)
        assert aligned.stride(0) * aligned.element_size() % 16 == 0
        assert aligned[height:].data_ptr() - aligned.data_ptr() == (
            height * aligned.stride(0) * aligned.element_size()
        )
        if packed.stride(0) * packed.element_size() % 16 == 0:
            assert aligned.data_ptr() == packed.data_ptr()
        else:
            assert aligned.data_ptr() != packed.data_ptr()

    def test_from_dlpack_reuses_cuda_context_without_repeating_context_flags(self, tmp_path, monkeypatch):
        enc = _make_encoder(tmp_path, codec="h264", video_width=2, video_height=2)
        enc.stream = MagicMock()
        enc._cuda_ctx = object()
        enc._lut_applier = None
        enc._to_yuv = lambda frame: torch.zeros((3, 2), dtype=torch.uint8)
        enc.out_stream = MagicMock()
        enc.out_stream.encode.return_value = []
        hw_frame = SimpleNamespace(pts=None, time_base=None)
        from_dlpack = MagicMock(return_value=hw_frame)
        monkeypatch.setattr(
            video_encoder_module.av,
            "VideoFrame",
            SimpleNamespace(from_dlpack=from_dlpack),
        )
        monkeypatch.setattr(video_encoder_module.torch.cuda, "stream", lambda stream: nullcontext())

        enc._encode_frame(torch.zeros((3, 2, 2), dtype=torch.uint8), 7)

        _, kwargs = from_dlpack.call_args
        assert kwargs == {"format": "nv12", "cuda_context": enc._cuda_ctx}

    def test_pts_origin_is_removed_from_fragment_timestamps(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.pts_origin = 100
        enc.encode("frame", 110)
        assert enc.pts_heap == [10]

    def test_bridge_frame_records_lut_bypass(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("frame", 10, apply_lut=False)
        assert list(enc.frame_buffer) == ["frame"]
        assert list(enc._lut_flags) == [False]

    def test_encode_pushes_to_buffer_and_heap(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("frame0", 10)
        assert list(enc.frame_buffer) == ["frame0"]
        assert enc.pts_heap == [10]
        assert enc.pts_set == {10}

    def test_encode_dedup_pts(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("a", 5)
        enc.encode("b", 5)
        assert sorted(enc.pts_set) == [5, 6]

    def test_flush_starts_above_half_buffer(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        for i in range(enc.BUFFER_MAX_SIZE // 2):
            enc.encode(f"f{i}", i)
        enc._encode_queue.put.assert_not_called()
        enc.encode("one-more", 99)
        enc._encode_queue.put.assert_called_once_with(("f0", 0, None))

    def test_smallest_pts_pairs_with_oldest_frame(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        for i, pts in enumerate([30, 10, 20, 40]):
            enc.encode(f"f{i}", pts)
        enc._process_buffer(flush_all=True)
        enc._encode_queue.put.assert_called_once_with(("f0", 10, None))

    def test_encode_raises_pending_worker_error(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc._worker_error = RuntimeError("nvenc exploded")
        with pytest.raises(RuntimeError, match="nvenc exploded"):
            enc.encode("frame", 0)


class TestWorkerErrorChannel:
    def test_worker_records_error_and_keeps_consuming(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.device = torch.device("cpu")
        enc._worker_error = None
        enc._encode_queue = queue.Queue()
        enc._stop_sentinel = object()
        enc._handle_encode_item = MagicMock(side_effect=RuntimeError("mux failed"))

        worker = threading.Thread(target=enc._encode_worker, daemon=True)
        worker.start()
        enc._encode_queue.put(("frame", 0, None))
        enc._encode_queue.put(("frame", 1, None))
        enc._encode_queue.join()
        enc._encode_queue.put(enc._stop_sentinel)
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert isinstance(enc._worker_error, RuntimeError)
        assert enc._handle_encode_item.call_count == 1


def _packet(stream_index, dts, time_base=Fraction(1, 1000)):
    return SimpleNamespace(
        stream=SimpleNamespace(index=stream_index),
        dts=dts,
        pts=dts,
        time_base=time_base,
    )


class TestAudioPump:
    def _audio_encoder(self, tmp_path, packets):
        enc = _make_encoder(tmp_path)
        out_a = MagicMock()
        enc._audio_pipes = {1: ("copy", out_a, None)}
        enc._audio_backlog = deque()
        enc._audio_iter = iter(packets)
        enc.dst = MagicMock()
        return enc, out_a

    def test_copy_reassigns_stream_and_skips_flush_packet(self, tmp_path):
        enc, out_a = self._audio_encoder(tmp_path, [])
        pkt = _packet(1, dts=0)
        assert enc._produce_audio_packets(pkt) == [pkt]
        assert pkt.stream is out_a

        flush = SimpleNamespace(stream=SimpleNamespace(index=1), dts=None, pts=None, time_base=None)
        assert enc._produce_audio_packets(flush) == []

    def test_pump_respects_threshold(self, tmp_path):
        packets = [_packet(1, dts=0), _packet(1, dts=500), _packet(1, dts=1500)]
        enc, _ = self._audio_encoder(tmp_path, packets)

        enc._pump_audio(1.0)
        assert enc.dst.mux.call_count == 2
        assert len(enc._audio_backlog) == 1  # dts=1500 held back

        enc._pump_audio(None)
        assert enc.dst.mux.call_count == 3

    def test_pump_without_audio_is_noop(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc._audio_iter = None
        enc._pump_audio(1.0)
