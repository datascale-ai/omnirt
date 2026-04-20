"""SDXL component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_SDXL_MODEL_SOURCE = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SDXL_REFINER_MODEL_SOURCE = "stabilityai/stable-diffusion-xl-refiner-1.0"
DEFAULT_SDXL_TURBO_MODEL_SOURCE = "stabilityai/sdxl-turbo"


@dataclass
class SDXLComponents:
    text_encoder: Optional[object] = None
    text_encoder_2: Optional[object] = None
    unet: Optional[object] = None
    vae: Optional[object] = None
