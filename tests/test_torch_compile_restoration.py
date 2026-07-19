"""Unit coverage for the opt-in torch.compile restoration path.

Covers the logic that cannot regress silently: full-length-clip routing (only
T == max_clip_size runs compiled — any other length is a fresh multi-minute
dynamo compile in production), the eager restore after every compiled call,
the mid-run failure fallback, and the env gating.

The fake generators are real nn.Modules assigned in _FakeModel.__init__, so
generator_ema is a REGISTERED submodule and the setattr swaps exercise
nn.Module.__setattr__'s registered-slot path — the exact semantics production
relies on (torch.compile's OptimizedModule is also an nn.Module; assigning a
non-Module to a registered slot raises TypeError).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from jasna.restorer.basicvsrpp_mosaic_restorer import (
    TORCH_COMPILE_ENV,
    BasicvsrppMosaicRestorer,
)


class _Generator(torch.nn.Module):
    def __init__(self, calls: dict, key: str, fail: bool = False):
        super().__init__()
        self._calls = calls
        self._key = key
        self._fail = fail

    def forward(self, inputs):
        self._calls[self._key] += 1
        if self._fail:
            raise RuntimeError("driver hiccup")
        return inputs * 2.0


class _FakeModel(torch.nn.Module):
    """Stands in for the mmagic model: model(inputs=x) runs the generator_ema
    submodule, mirroring forward_tensor's dispatch."""

    def __init__(self, eager: _Generator):
        super().__init__()
        self.generator_ema = eager  # registers in _modules

    def forward(self, inputs):
        return self.generator_ema(inputs)


def _make_restorer(max_clip_size: int = 4) -> BasicvsrppMosaicRestorer:
    r = object.__new__(BasicvsrppMosaicRestorer)
    r.device = torch.device("cpu")
    r.max_clip_size = max_clip_size
    r.use_tensorrt = False
    r.dtype = torch.float32
    r.input_dtype = torch.float32
    r._split_forward = None
    r._gen_attr = "generator_ema"

    calls = {"eager": 0, "compiled": 0}
    eager = _Generator(calls, "eager")
    r.model = _FakeModel(eager)
    r._eager_generator = eager
    r._compiled_generator = _Generator(calls, "compiled")
    r._calls = calls  # test-only bookkeeping
    return r


def _video(n: int) -> list[torch.Tensor]:
    return [torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8) for _ in range(n)]


def test_generator_is_a_registered_submodule():
    # The swap-semantics guarantee below is only meaningful if the fake slot is
    # actually registered, like the real generator_ema.
    r = _make_restorer()
    assert "generator_ema" in r.model._modules


def test_full_length_clip_runs_compiled_and_restores_eager():
    r = _make_restorer(max_clip_size=4)
    r.raw_process(_video(4))
    assert r._calls == {"eager": 0, "compiled": 1}
    # The eager generator must be back on the model after the call.
    assert r.model.generator_ema is r._eager_generator


def test_short_clip_runs_eager():
    r = _make_restorer(max_clip_size=4)
    r.raw_process(_video(3))
    assert r._calls == {"eager": 1, "compiled": 0}


def test_no_compiled_generator_runs_eager():
    r = _make_restorer(max_clip_size=4)
    r._compiled_generator = None
    r.raw_process(_video(4))
    assert r._calls == {"eager": 1, "compiled": 0}


def test_mid_run_compiled_failure_falls_back_and_disables():
    r = _make_restorer(max_clip_size=4)
    r._compiled_generator = _Generator(r._calls, "compiled", fail=True)

    out = r.raw_process(_video(4))
    # The failing clip was redone eager, and the compiled path is now off.
    assert r._calls == {"eager": 1, "compiled": 1}
    assert r._compiled_generator is None
    assert r.model.generator_ema is r._eager_generator
    assert out.shape[0] == 4

    # Subsequent full-length clips stay eager without re-attempting.
    r.raw_process(_video(4))
    assert r._calls == {"eager": 2, "compiled": 1}


def test_non_module_compiled_value_is_contained():
    """Assigning a non-Module to the registered slot raises TypeError from
    nn.Module.__setattr__; the setattr sits inside the try, so this too must
    land on eager instead of killing the run."""
    r = _make_restorer(max_clip_size=4)
    r._compiled_generator = lambda inputs: inputs  # not an nn.Module

    out = r.raw_process(_video(4))
    assert r._calls == {"eager": 1, "compiled": 0}
    assert r._compiled_generator is None
    assert r.model.generator_ema is r._eager_generator
    assert out.shape[0] == 4


def test_env_gating(monkeypatch):
    captured = SimpleNamespace(called=False)
    monkeypatch.setattr(
        BasicvsrppMosaicRestorer,
        "_setup_torch_compile",
        lambda self, _path: setattr(captured, "called", True),
    )
    monkeypatch.setattr(
        "jasna.restorer.basicvsrpp_mosaic_restorer.load_model",
        lambda *_a, **_k: _FakeModel(_Generator({"eager": 0}, "eager")),
    )

    def build(device: str) -> None:
        captured.called = False
        BasicvsrppMosaicRestorer(
            checkpoint_path="model.pth",
            device=torch.device(device),
            max_clip_size=4,
            use_tensorrt=False,
            fp16=False,
        )

    monkeypatch.delenv(TORCH_COMPILE_ENV, raising=False)
    build("cpu")
    assert captured.called is False, "flag unset must not compile"

    monkeypatch.setenv(TORCH_COMPILE_ENV, "1")
    build("cpu")
    assert captured.called is False, "cpu must not compile even when opted in"

    build("meta")
    assert captured.called is True, "non-cpu device with flag set must compile"


def test_setup_failure_is_contained(monkeypatch, tmp_path):
    """A torch.compile blow-up at setup leaves a working eager restorer."""
    monkeypatch.setenv(TORCH_COMPILE_ENV, "1")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        torch, "compile",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no triton")),
    )
    r = _make_restorer(max_clip_size=4)
    r._compiled_generator = None
    r._setup_torch_compile(str(tmp_path / "model.pth"))
    assert r._compiled_generator is None
    # Cache dir was pinned next to the weights before the failure.
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor_cache")
    r.raw_process(_video(4))
    assert r._calls["eager"] == 1
