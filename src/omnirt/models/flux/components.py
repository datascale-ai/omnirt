"""Flux 1 component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_FLUX_DEV_MODEL_SOURCE = "black-forest-labs/FLUX.1-dev"
DEFAULT_FLUX_SCHNELL_MODEL_SOURCE = "black-forest-labs/FLUX.1-schnell"


@dataclass
class FluxComponents:
    text_encoder: Optional[object] = None
    transformer: Optional[object] = None
    vae: Optional[object] = None
