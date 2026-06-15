from pathlib import Path
import os

import pytest

from tests.integration.conftest import require_module
from omnirt.api import generate


def test_sensevoice_audio2text_smoke(tmp_path) -> None:
    require_module("funasr", "funasr is unavailable")

    audio = os.getenv("OMNIRT_SENSEVOICE_AUDIO_PATH")
    if not audio or not Path(audio).expanduser().exists():
        pytest.skip("OMNIRT_SENSEVOICE_AUDIO_PATH does not point to a local audio file")

    result = generate(
        {
            "task": "audio2text",
            "model": "sensevoice-small",
            "backend": "auto",
            "inputs": {"audio": str(Path(audio).expanduser())},
            "config": {
                "output_dir": str(tmp_path),
                "model_path": os.getenv("OMNIRT_SENSEVOICE_MODEL_PATH", "iic/SenseVoiceSmall"),
                "language": os.getenv("OMNIRT_SENSEVOICE_LANGUAGE", "auto"),
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].kind == "text"
    assert Path(result.outputs[0].path).read_text(encoding="utf-8").strip()


def test_sensevoice_audio2text_ascend_smoke(tmp_path) -> None:
    require_module("torch_npu", "torch_npu is unavailable")
    require_module("funasr", "funasr is unavailable")

    audio = os.getenv("OMNIRT_SENSEVOICE_AUDIO_PATH")
    if not audio or not Path(audio).expanduser().exists():
        pytest.skip("OMNIRT_SENSEVOICE_AUDIO_PATH does not point to a local audio file")

    result = generate(
        {
            "task": "audio2text",
            "model": "sensevoice-small",
            "backend": "ascend",
            "inputs": {"audio": str(Path(audio).expanduser())},
            "config": {
                "output_dir": str(tmp_path),
                "model_path": os.getenv("OMNIRT_SENSEVOICE_MODEL_PATH", "iic/SenseVoiceSmall"),
                "language": os.getenv("OMNIRT_SENSEVOICE_LANGUAGE", "auto"),
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].kind == "text"
    assert result.metadata.backend == "ascend"
    assert result.metadata.config_resolved["device"] == "npu:0"
    assert Path(result.outputs[0].path).read_text(encoding="utf-8").strip()
