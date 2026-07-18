import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_cli_creates_stream_on_chosen_device(tmp_path: Path) -> None:
    input_path = tmp_path / "in.mp4"
    input_path.touch()
    output_path = tmp_path / "out.mkv"
    model_weights = tmp_path / "model_weights"
    model_weights.mkdir()
    restoration_path = model_weights / "lada_mosaic_restoration_model_generic_v1.2.pth"
    restoration_path.touch()
    detection_path = model_weights / "rfdetr-v3.onnx"
    detection_path.touch()

    device_capture: list = []
    fake_stream = MagicMock()

    class NoOpDeviceContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def record_device(device):
        device_capture.append(device)
        return NoOpDeviceContext()

    pipeline_capture: dict = {}

    def capture_pipeline(**kwargs):
        pipeline_capture.update(kwargs)
        mock = MagicMock()
        return mock

    with (
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
        patch("jasna.main.check_required_executables"),        patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.pipeline.Pipeline", side_effect=capture_pipeline),
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        import torch

        with patch("torch.cuda.device", side_effect=record_device), patch(
            "torch.cuda.Stream", return_value=fake_stream
        ):
            with patch.object(
                sys,
                "argv",
                [
                    "jasna",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--device",
                    "cuda:1",
                    "--restoration-model-path",
                    str(restoration_path),
                    "--detection-model-path",
                    str(detection_path),
                ],
            ):
                from jasna.main import main

                main()

    assert any(d == torch.device("cuda:1") for d in device_capture)
    assert pipeline_capture["device"] == torch.device("cuda:1")


def _cli_pipeline_device(tmp_path: Path, device_arg: list[str], extra_patches: list) -> dict:
    """Run main() with a mocked stack; return the kwargs Pipeline received."""
    input_path = tmp_path / "in.mp4"
    input_path.touch()
    output_path = tmp_path / "out.mkv"
    restoration_path = tmp_path / "restore.pth"
    restoration_path.touch()
    detection_path = tmp_path / "rfdetr-v3.onnx"
    detection_path.touch()

    pipeline_capture: dict = {}

    def capture_pipeline(**kwargs):
        pipeline_capture.update(kwargs)
        return MagicMock()

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in [
            patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
            patch("jasna.main.check_required_executables"),
            patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
            patch("jasna.pipeline.Pipeline", side_effect=capture_pipeline),
            patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
            *extra_patches,
        ]:
            stack.enter_context(p)
        with patch.object(
            sys,
            "argv",
            [
                "jasna",
                "--input", str(input_path),
                "--output", str(output_path),
                *device_arg,
                "--restoration-model-path", str(restoration_path),
                "--detection-model-path", str(detection_path),
            ],
        ):
            from jasna.main import main

            main()
    return pipeline_capture


def test_cli_xpu_device_plumbs_through(tmp_path: Path) -> None:
    import torch

    captured = _cli_pipeline_device(
        tmp_path,
        ["--device", "xpu:0"],
        [
            patch("jasna.main.check_intel_gpu", return_value=(True, "Fake Arc")),
            # xpu tensors don't exist on this box; keep the device context inert.
            patch("jasna.device_backend.device_ctx", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.main.device_ctx", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
        ],
    )
    assert captured["device"] == torch.device("xpu:0")
    # fp16 defaults off on xpu (deform_conv2d CPU fallback has no half support)
    assert captured["fp16"] is False


def test_cli_auto_resolves_before_preflight(tmp_path: Path) -> None:
    import torch

    captured = _cli_pipeline_device(
        tmp_path,
        ["--device", "auto"],
        [
            patch("jasna.main.resolve_auto_spec", return_value="cuda:0", create=True),
            patch("jasna.device_backend.resolve_auto_spec", return_value="cuda:0"),
            patch("jasna.main.check_nvidia_gpu", return_value=(True, "Fake GPU")),
            patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
            patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        ],
    )
    assert captured["device"] == torch.device("cuda:0")
    assert captured["fp16"] is True


def test_cli_invalid_device_exits(tmp_path: Path) -> None:
    input_path = tmp_path / "in.mp4"
    input_path.touch()
    with (
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch("jasna.main.check_required_executables"),
    ):
        with patch.object(
            sys,
            "argv",
            ["jasna", "--input", str(input_path), "--output", str(tmp_path / "o.mkv"), "--device", "tpu:0"],
        ):
            from jasna.main import main

            with pytest.raises(SystemExit):
                main()


def test_cli_list_devices(capsys) -> None:
    with patch(
        "jasna.device_backend.list_devices",
        return_value=[("cuda:0", "Fake GPU"), ("xpu:0", "Fake Arc"), ("cpu", "CPU")],
    ):
        with patch.object(sys, "argv", ["jasna", "--list-devices"]):
            from jasna.main import main

            main()
    out = capsys.readouterr().out
    assert "cuda:0" in out and "xpu:0" in out and "cpu" in out
