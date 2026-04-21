from pathlib import Path
import os

import pytest

from tests.integration.conftest import require_local_model_dir, require_module
from omnirt.api import generate


def _require_local_path(env_var: str, default: Path) -> str:
    candidate = Path(os.getenv(env_var, str(default))).expanduser()
    if not candidate.exists():
        pytest.skip(f"{env_var} does not exist locally: {candidate}")
    return str(candidate)


def test_flashtalk_ascend_smoke(tmp_path) -> None:
    require_module("torch_npu", "torch_npu is unavailable")

    repo_path = require_local_model_dir("OMNIRT_FLASHTALK_REPO_PATH")
    repo_root = Path(repo_path)
    image_path = _require_local_path("OMNIRT_FLASHTALK_IMAGE_PATH", repo_root / "examples" / "woman2.jpg")
    audio_path = _require_local_path("OMNIRT_FLASHTALK_AUDIO_PATH", repo_root / "examples" / "cantonese_16k.wav")
    python_executable = _require_local_path(
        "OMNIRT_FLASHTALK_PYTHON_EXECUTABLE",
        Path.home() / "flashtalk-venv/bin/python",
    )
    ascend_env_script = _require_local_path(
        "OMNIRT_FLASHTALK_ASCEND_ENV_SCRIPT",
        Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
    )

    result = generate(
        {
            "task": "audio2video",
            "model": "soulx-flashtalk-14b",
            "backend": "ascend",
            "inputs": {"image": image_path, "audio": audio_path},
            "config": {
                "repo_path": repo_path,
                "ckpt_dir": "models/SoulX-FlashTalk-14B",
                "wav2vec_dir": "models/chinese-wav2vec2-base",
                "output_dir": str(tmp_path),
                "python_executable": python_executable,
                "ascend_env_script": ascend_env_script,
                "launcher": "torchrun",
                "nproc_per_node": 8,
                "visible_devices": "0,1,2,3,4,5,6,7",
                "audio_encode_mode": "once",
                "max_chunks": 1,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".mp4")
    assert Path(result.outputs[0].path).exists()
    assert result.metadata.execution_mode == "persistent_worker"
    assert result.metadata.timings["chunk_count"] >= 1.0
    assert "chunk_core_ms_avg" in result.metadata.timings
