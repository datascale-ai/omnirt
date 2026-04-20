"""Stable Diffusion 3 family component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_SD3_MEDIUM_MODEL_SOURCE = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_SD35_LARGE_MODEL_SOURCE = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_SD35_LARGE_TURBO_MODEL_SOURCE = "stabilityai/stable-diffusion-3.5-large-turbo"


@dataclass
class SD3Components:
    text_encoder: Optional[object] = None
    text_encoder_2: Optional[object] = None
    text_encoder_3: Optional[object] = None
    transformer: Optional[object] = None
    vae: Optional[object] = None
