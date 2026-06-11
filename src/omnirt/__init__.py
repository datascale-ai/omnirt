"""OmniRT public package interface."""

import importlib

from omnirt import core
from omnirt import models
from omnirt import requests
from omnirt.core.presets import available_presets
from omnirt.core.types import (
    AudioToVideoRequest,
    AudioToTextRequest,
    EditRequest,
    GenerateRequest,
    GenerateResult,
    ImageToImageRequest,
    ImageToVideoRequest,
    InpaintRequest,
    TextToAudioRequest,
    TextToImageRequest,
    TextToVideoRequest,
)

_LAZY_API_NAMES = {"describe_model", "generate", "list_available_models", "pipeline", "validate"}
_LAZY_SUBMODULE_NAMES = {"server"}


def __getattr__(name: str):
    if name in _LAZY_API_NAMES:
        from omnirt import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULE_NAMES:
        value = importlib.import_module(f"omnirt.{name}")
        globals()[name] = value
        return value
    raise AttributeError(f"module 'omnirt' has no attribute {name!r}")


__all__ = [
    "GenerateRequest",
    "GenerateResult",
    "TextToImageRequest",
    "TextToVideoRequest",
    "TextToAudioRequest",
    "AudioToTextRequest",
    "ImageToImageRequest",
    "InpaintRequest",
    "EditRequest",
    "ImageToVideoRequest",
    "AudioToVideoRequest",
    "generate",
    "validate",
    "list_available_models",
    "describe_model",
    "pipeline",
    "available_presets",
    "core",
    "models",
    "requests",
    "server",
]
