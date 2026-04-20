"""Tests for BasePipeline.pipeline_cache_key and adapter_fingerprint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelSpec
from omnirt.core.types import AdapterRef


class _Runtime(BackendRuntime):
    name = "test"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        return SimpleNamespace()

    def _compile(self, module, tag):
        return module


class _NullPipeline(BasePipeline):
    def prepare_conditions(self, req):
        return {}

    def prepare_latents(self, req, conditions):
        return {}

    def denoise_loop(self, latents, conditions, config):
        return {}

    def decode(self, latents):
        return None

    def export(self, raw, req):
        return []


@pytest.fixture()
def weight_files(tmp_path: Path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    a.write_bytes(b"")
    b.write_bytes(b"")
    return a, b


def _pipeline(adapters=None):
    return _NullPipeline(
        runtime=_Runtime(),
        model_spec=ModelSpec(id="dummy", task="text2image", pipeline_cls=_NullPipeline),
        adapters=adapters or [],
    )


def test_adapter_fingerprint_empty_when_no_adapters() -> None:
    assert _pipeline().adapter_fingerprint() == ()


def test_adapter_fingerprint_stable_across_order(weight_files) -> None:
    a_path, b_path = weight_files
    a = AdapterRef(kind="lora", path=str(a_path), scale=0.8)
    b = AdapterRef(kind="lora", path=str(b_path), scale=0.5)
    assert _pipeline([a, b]).adapter_fingerprint() == _pipeline([b, a]).adapter_fingerprint()


def test_cache_key_varies_by_scheduler() -> None:
    pipeline = _pipeline()
    k1 = pipeline.pipeline_cache_key(source="/m", torch_dtype="fp16", scheduler_name="euler-discrete")
    k2 = pipeline.pipeline_cache_key(source="/m", torch_dtype="fp16", scheduler_name="ddim")
    assert k1 != k2


def test_cache_key_varies_by_adapter_scale(weight_files) -> None:
    a_path, _ = weight_files
    p1 = _pipeline([AdapterRef(kind="lora", path=str(a_path), scale=1.0)])
    p2 = _pipeline([AdapterRef(kind="lora", path=str(a_path), scale=0.5)])
    args = dict(source="/m", torch_dtype="fp16", scheduler_name="euler-discrete")
    assert p1.pipeline_cache_key(**args) != p2.pipeline_cache_key(**args)


def test_cache_key_stable_for_same_inputs(weight_files) -> None:
    a_path, _ = weight_files
    adapter = AdapterRef(kind="lora", path=str(a_path), scale=1.0)
    p = _pipeline([adapter])
    args = dict(source="/m", torch_dtype="fp16", scheduler_name="euler-discrete")
    assert p.pipeline_cache_key(**args) == p.pipeline_cache_key(**args)
