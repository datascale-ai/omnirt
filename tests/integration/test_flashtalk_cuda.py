from pathlib import Path
import os

import pytest

from tests.integration.conftest import require_local_model_dir
from omnirt.api import generate


def _require_local_path(env_var: str, default: Path) -> str:
    candidate = Path(os.getenv(env_var, str(default))).expanduser()
    if not candidate.exists():
        pytest.skip(f"{env_var} does not exist locally: {candidate}")
    return str(candidate)


def test_flashtalk_cuda_smoke(tmp_path) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch is unavailable")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    repo_path = require_local_model_dir("OMNIRT_FLASHTALK_REPO_PATH")
    repo_root = Path(repo_path)
    image_path = _require_local_path("OMNIRT_FLASHTALK_IMAGE_PATH", repo_root / "examples" / "woman2.jpg")
    audio_path = _require_local_path("OMNIRT_FLASHTALK_AUDIO_PATH", repo_root / "examples" / "cantonese_16k.wav")
    python_executable = _require_local_path(
        "OMNIRT_FLASHTALK_PYTHON_EXECUTABLE",
        Path.home() / "flashtalk-cuda-venv/bin/python",
    )

    result = generate(
        {
            "task": "audio2video",
            "model": "soulx-flashtalk-14b",
            "backend": "cuda",
            "inputs": {"image": image_path, "audio": audio_path},
            "config": {
                "repo_path": repo_path,
                "ckpt_dir": "models/SoulX-FlashTalk-14B",
                "wav2vec_dir": "models/chinese-wav2vec2-base",
                "output_dir": str(tmp_path),
                "python_executable": python_executable,
                "launcher": "python",
                "visible_devices": os.getenv("OMNIRT_FLASHTALK_VISIBLE_DEVICES", "0"),
                "audio_encode_mode": "once",
                "max_chunks": 1,
                "cpu_offload": True,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".mp4")
    assert Path(result.outputs[0].path).exists()
