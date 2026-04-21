"""Component metadata for SoulX-FlashTalk.

Deployment paths (repo checkout, weights, Ascend env script, python venv) are
not hardcoded — they are resolved from, in priority order:

1. ``OMNIRT_FLASHTALK_*`` environment variables
2. ``configs/flashtalk.yaml`` at the project root
3. ``~/.omnirt/flashtalk.yaml`` per-user override

If none of the above supplies a setting, a :class:`FlashTalkConfigurationError`
is raised when the pipeline resolves its runtime config. This is deliberate —
previous releases shipped with absolute paths baked into the source tree which
broke every deployment that wasn't the original developer's workstation.

The :data:`DEFAULT_FLASHTALK_PROMPT` constant remains because it is a functional
default for the model's behavior, not a deployment concern.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_FLASHTALK_PROMPT = (
    "A person is talking. Only the foreground characters are moving, the background remains static."
)


ENV_PREFIX = "OMNIRT_FLASHTALK_"

_ENV_KEYS = {
    "repo_path": "REPO_PATH",
    "ckpt_dir": "CKPT_DIR",
    "wav2vec_dir": "WAV2VEC_DIR",
    "ascend_env_script": "ASCEND_ENV_SCRIPT",
    "python_executable": "PYTHON",
}

_PROJECT_CONFIG_RELATIVE = Path("configs") / "flashtalk.yaml"
_USER_CONFIG_RELATIVE = Path(".omnirt") / "flashtalk.yaml"


class FlashTalkConfigurationError(RuntimeError):
    """Raised when a required FlashTalk deployment setting is missing."""


def _project_root() -> Path:
    # src/omnirt/models/flashtalk/components.py → project root is four parents up.
    return Path(__file__).resolve().parents[4]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when pyyaml missing.
        raise FlashTalkConfigurationError(
            f"Reading {path} requires PyYAML. Install it or set {ENV_PREFIX}* env vars instead."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise FlashTalkConfigurationError(f"{path} must contain a YAML mapping at the top level.")
    return data


@lru_cache(maxsize=1)
def _yaml_settings() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    project_path = _project_root() / _PROJECT_CONFIG_RELATIVE
    merged.update(_read_yaml(project_path))
    user_path = Path.home() / _USER_CONFIG_RELATIVE
    merged.update(_read_yaml(user_path))
    return merged


def reset_config_cache() -> None:
    """Invalidate the YAML cache. Tests use this after patching env/yaml."""
    _yaml_settings.cache_clear()


def flashtalk_setting(key: str, *, required: bool = False) -> Optional[str]:
    """Resolve a FlashTalk deployment setting.

    Lookup order: ``OMNIRT_FLASHTALK_<KEY>`` env var → ``configs/flashtalk.yaml``
    → ``~/.omnirt/flashtalk.yaml``. Returns ``None`` when the setting is absent
    and ``required`` is ``False``; raises :class:`FlashTalkConfigurationError`
    otherwise.
    """
    env_key = _ENV_KEYS.get(key)
    if env_key is None:
        raise KeyError(f"Unknown FlashTalk setting: {key!r}")
    env_value = os.environ.get(ENV_PREFIX + env_key)
    if env_value and env_value.strip():
        return env_value.strip()
    yaml_value = _yaml_settings().get(key)
    if isinstance(yaml_value, str) and yaml_value.strip():
        return yaml_value.strip()
    if required:
        raise FlashTalkConfigurationError(
            f"FlashTalk setting {key!r} is not configured. "
            f"Set the {ENV_PREFIX + env_key} environment variable or add "
            f"'{key}' to configs/flashtalk.yaml or ~/.omnirt/flashtalk.yaml."
        )
    return None
