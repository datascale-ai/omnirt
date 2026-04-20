"""SD 1.5 component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_SD15_MODEL_SOURCE = "runwayml/stable-diffusion-v1-5"
DEFAULT_SD21_MODEL_SOURCE = "stabilityai/stable-diffusion-2-1"


@dataclass
class SD15Components:
    text_encoder: Optional[object] = None
    unet: Optional[object] = None
    vae: Optional[object] = None
