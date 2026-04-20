"""Shared user-facing presets for common generation profiles."""

from __future__ import annotations

from typing import Any, Dict, Tuple


_PRESET_ORDER: Tuple[str, ...] = ("fast", "balanced", "quality", "low-vram")

_BASE_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "num_inference_steps": 20,
    },
    "balanced": {},
    "quality": {
        "num_inference_steps": 40,
    },
    "low-vram": {
        "num_inference_steps": 18,
        "dtype": "fp16",
    },
}

_TASK_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "text2image": {
        "fast": {"guidance_scale": 5.5},
        "balanced": {"guidance_scale": 7.5},
        "quality": {"guidance_scale": 8.0},
    },
    "text2video": {
        "fast": {"guidance_scale": 4.0},
        "balanced": {"guidance_scale": 5.0},
        "quality": {"guidance_scale": 6.0},
        "low-vram": {"guidance_scale": 4.0},
    },
    "image2video": {
        "fast": {"guidance_scale": 3.0},
        "balanced": {"guidance_scale": 3.5},
        "quality": {"guidance_scale": 4.0},
        "low-vram": {"guidance_scale": 3.0},
    },
}

_TASK_PRESET_ALIASES: Dict[str, str] = {
    "image2image": "text2image",
    "inpaint": "text2image",
    "edit": "text2image",
}

_MODEL_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "flux2.dev": {
        "fast": {"guidance_scale": 2.0, "max_sequence_length": 384},
        "balanced": {"guidance_scale": 2.5, "max_sequence_length": 512},
        "quality": {"guidance_scale": 3.0, "max_sequence_length": 512},
        "low-vram": {"guidance_scale": 2.0, "max_sequence_length": 256, "dtype": "fp16"},
    },
    "flux2-dev": {
        "fast": {"guidance_scale": 2.0, "max_sequence_length": 384},
        "balanced": {"guidance_scale": 2.5, "max_sequence_length": 512},
        "quality": {"guidance_scale": 3.0, "max_sequence_length": 512},
        "low-vram": {"guidance_scale": 2.0, "max_sequence_length": 256, "dtype": "fp16"},
    },
}


def available_presets() -> Tuple[str, ...]:
    return _PRESET_ORDER


def resolve_preset(*, task: str, model: str, preset: str) -> Dict[str, Any]:
    if preset not in _PRESET_ORDER:
        available = ", ".join(_PRESET_ORDER)
        raise ValueError(f"Unknown preset {preset!r}. Available presets: [{available}]")

    merged: Dict[str, Any] = {}
    task_key = _TASK_PRESET_ALIASES.get(task, task)
    merged.update(_BASE_PRESETS.get(preset, {}))
    merged.update(_TASK_PRESETS.get(task_key, {}).get(preset, {}))
    merged.update(_MODEL_PRESETS.get(model, {}).get(preset, {}))
    return merged
