"""Component metadata for SoulX-FlashTalk."""

from __future__ import annotations

from pathlib import Path


DEFAULT_FLASHTALK_REPO_PATH = "/home/<user>/SoulX-FlashTalk"
DEFAULT_FLASHTALK_CKPT_DIR = "models/SoulX-FlashTalk-14B"
DEFAULT_FLASHTALK_WAV2VEC_DIR = "models/chinese-wav2vec2-base"
DEFAULT_FLASHTALK_ASCEND_ENV_SCRIPT = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
DEFAULT_FLASHTALK_PYTHON = "/home/<user>/flashtalk-venv/bin/python"
DEFAULT_FLASHTALK_PROMPT = (
    "A person is talking. Only the foreground characters are moving, the background remains static."
)


def resolve_flashtalk_python() -> str:
    candidate = Path(DEFAULT_FLASHTALK_PYTHON)
    return str(candidate) if candidate.exists() else "python3"
