from __future__ import annotations

from pathlib import Path
import os

import pytest

from tests.integration.conftest import require_module
from omnirt.models.indextts.runtime import create_indextts_runtime_from_env


def test_indextts_ascend_runtime_env_smoke(monkeypatch) -> None:
    require_module("torch_npu", "torch_npu is unavailable")

    monkeypatch.setenv("OMNIRT_INDEXTTS_DEVICE", "ascend")
    runtime = create_indextts_runtime_from_env()
    status = runtime.status()

    assert runtime.device.startswith("npu:")
    assert runtime.use_fp16 is True
    assert runtime.use_cuda_kernel is False
    assert status["device"].startswith("npu:")
    assert status["use_fp16"] is True
    assert status["use_cuda_kernel"] is False


def test_indextts_ascend_runtime_loads_when_assets_exist(monkeypatch) -> None:
    require_module("torch_npu", "torch_npu is unavailable")
    pytest.importorskip("indextts.infer_v2", reason="IndexTTS runtime package is unavailable")

    model_dir = Path(os.getenv("OMNIRT_INDEXTTS_MODEL_DIR", "")).expanduser()
    cfg_path = Path(os.getenv("OMNIRT_INDEXTTS_CFG_PATH", "")).expanduser()
    prompt_audio = Path(os.getenv("OMNIRT_INDEXTTS_PROMPT_AUDIO", "")).expanduser()
    if not model_dir.is_dir() or not cfg_path.is_file() or not prompt_audio.is_file():
        pytest.skip("IndexTTS model dir, config, and prompt audio are required for Ascend load smoke")

    monkeypatch.setenv("OMNIRT_INDEXTTS_DEVICE", "ascend")
    runtime = create_indextts_runtime_from_env()

    runtime.warmup(text="", max_chunks=1)

    assert runtime.status()["ready"] is True
