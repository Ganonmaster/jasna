"""ov_cache_dir must key on model content (digest), mirroring migraphx_cache_dir,
so re-exporting a model at the same path invalidates ov_cache_is_ready."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

from jasna.ov.ov_runner import ov_cache_dir, ov_cache_is_ready

_KEY_PATTERN = re.compile(r"^[0-9a-f]{16}-fp(16|32)-(cpu|gpu)$")


def _write_model(path: Path, payload: bytes, *, mtime_ns: int) -> None:
    # Pin mtime explicitly: the digest cache is keyed on (path, size, mtime_ns)
    # and back-to-back writes can land on the same filesystem timestamp tick.
    path.write_bytes(payload)
    os.utime(path, ns=(mtime_ns, mtime_ns))


@pytest.fixture
def onnx_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    _write_model(path, b"original-model-bytes", mtime_ns=1_000_000_000)
    return path


def test_same_model_yields_stable_dir(onnx_path: Path) -> None:
    device = torch.device("cpu")
    first = ov_cache_dir(onnx_path, device, fp16=True)
    second = ov_cache_dir(onnx_path, device, fp16=True)
    assert first == second
    assert first.parent == onnx_path.parent / "model.openvino"
    assert _KEY_PATTERN.match(first.name)


def test_precision_separates_dirs_but_shares_digest(onnx_path: Path) -> None:
    device = torch.device("cpu")
    fp16_dir = ov_cache_dir(onnx_path, device, fp16=True)
    fp32_dir = ov_cache_dir(onnx_path, device, fp16=False)
    assert fp16_dir != fp32_dir
    assert fp16_dir.name.split("-")[0] == fp32_dir.name.split("-")[0]
    assert fp16_dir.name.endswith("-fp16-cpu")
    assert fp32_dir.name.endswith("-fp32-cpu")


def test_modified_model_bytes_change_dir(onnx_path: Path) -> None:
    device = torch.device("cpu")
    before = ov_cache_dir(onnx_path, device, fp16=True)
    _write_model(onnx_path, b"re-quantized-model-bytes", mtime_ns=2_000_000_000)
    after = ov_cache_dir(onnx_path, device, fp16=True)
    assert before != after
    assert before.parent == after.parent
    assert before.name.split("-", 1)[1] == after.name.split("-", 1)[1]


def test_ir_sidecar_bin_participates_in_digest(tmp_path: Path) -> None:
    device = torch.device("cpu")
    xml_path = tmp_path / "model.xml"
    bin_path = tmp_path / "model.bin"
    _write_model(xml_path, b"<net topology unchanged/>", mtime_ns=1_000_000_000)
    _write_model(bin_path, b"int8-weights-v1", mtime_ns=1_000_000_000)
    before = ov_cache_dir(xml_path, device, fp16=True)
    # Re-quantization can rewrite only the weights sidecar, not the .xml.
    _write_model(bin_path, b"int8-weights-v2", mtime_ns=2_000_000_000)
    after = ov_cache_dir(xml_path, device, fp16=True)
    assert before != after


def test_cache_is_ready_semantics(onnx_path: Path) -> None:
    device = torch.device("cpu")
    assert not ov_cache_is_ready(onnx_path, device, fp16=True)

    directory = ov_cache_dir(onnx_path, device, fp16=True)
    directory.mkdir(parents=True)
    assert not ov_cache_is_ready(onnx_path, device, fp16=True)  # empty dir

    nested = directory / "blobs"
    nested.mkdir()
    (nested / "model.blob").write_bytes(b"compiled")
    assert ov_cache_is_ready(onnx_path, device, fp16=True)


def test_reexported_model_is_not_ready(onnx_path: Path) -> None:
    device = torch.device("cpu")
    directory = ov_cache_dir(onnx_path, device, fp16=True)
    directory.mkdir(parents=True)
    (directory / "model.blob").write_bytes(b"compiled")
    assert ov_cache_is_ready(onnx_path, device, fp16=True)

    _write_model(onnx_path, b"re-quantized-model-bytes", mtime_ns=2_000_000_000)
    assert not ov_cache_is_ready(onnx_path, device, fp16=True)
