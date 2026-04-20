"""Wan 2.2 component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_WAN2_2_T2V_MODEL_SOURCE = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_WAN2_2_I2V_MODEL_SOURCE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DEFAULT_WAN2_1_T2V_MODEL_SOURCE = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEFAULT_WAN2_1_I2V_MODEL_SOURCE = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


@dataclass
class WanComponents:
    text_encoder: Optional[object] = None
    image_encoder: Optional[object] = None
    transformer: Optional[object] = None
    transformer_2: Optional[object] = None
    vae: Optional[object] = None
