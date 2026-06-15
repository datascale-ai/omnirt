"""Deployment metadata for LongCat-Video-Avatar.

The LongCat integration is intentionally script-backed. OmniRT owns request
normalization, registry metadata, launch configuration, and artifact handling;
the external LongCat-Video checkout owns model internals, checkpoints, and
Ascend-specific operator patches.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_LONGCAT_AVATAR_PROMPT = "A person is speaking naturally with audio-driven facial motion."
ENV_PREFIX = "OMNIRT_LONGCAT_AVATAR_"

_ENV_KEYS = {
    "repo_path": "REPO_PATH",
    "ckpt_dir": "CKPT_DIR",
    "base_ckpt_dir": "BASE_CKPT_DIR",
    "worker_script": "WORKER_SCRIPT",
    "env_script": "ENV_SCRIPT",
    "ascend_env_script": "ASCEND_ENV_SCRIPT",
    "cuda_env_script": "CUDA_ENV_SCRIPT",
    "python_executable": "PYTHON",
    "visible_devices": "VISIBLE_DEVICES",
}

_PROJECT_CONFIG_RELATIVE = Path("configs") / "longcat_video_avatar.yaml"
_USER_CONFIG_RELATIVE = Path(".omnirt") / "longcat_video_avatar.yaml"


class LongCatAvatarConfigurationError(RuntimeError):
    """Raised when a required LongCat Avatar deployment setting is missing."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise LongCatAvatarConfigurationError(
            f"Reading {path} requires PyYAML. Install it or set {ENV_PREFIX}* env vars instead."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise LongCatAvatarConfigurationError(f"{path} must contain a YAML mapping at the top level.")
    return data


@lru_cache(maxsize=1)
def _yaml_settings() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(_read_yaml(_project_root() / _PROJECT_CONFIG_RELATIVE))
    merged.update(_read_yaml(Path.home() / _USER_CONFIG_RELATIVE))
    return merged


def reset_config_cache() -> None:
    _yaml_settings.cache_clear()


def longcat_avatar_setting(key: str, *, required: bool = False) -> Optional[str]:
    env_key = _ENV_KEYS.get(key)
    if env_key is None:
        raise KeyError(f"Unknown LongCat Avatar setting: {key!r}")
    env_value = os.environ.get(ENV_PREFIX + env_key)
    if env_value and env_value.strip():
        return env_value.strip()
    yaml_value = _yaml_settings().get(key)
    if isinstance(yaml_value, str) and yaml_value.strip():
        return yaml_value.strip()
    if required:
        raise LongCatAvatarConfigurationError(
            f"LongCat Avatar setting {key!r} is not configured. "
            f"Set the {ENV_PREFIX + env_key} environment variable or add "
            f"'{key}' to configs/longcat_video_avatar.yaml or ~/.omnirt/longcat_video_avatar.yaml."
        )
    return None
