"""Image and video media helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from omnirt.core.types import DependencyUnavailableError


def load_image(path: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise DependencyUnavailableError("Pillow is required to load image inputs.") from exc

    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(path: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise DependencyUnavailableError("Pillow is required to load mask inputs.") from exc

    with Image.open(path) as image:
        return image.convert("L")


def save_video_frames(path: Path, frames: Iterable[object], *, fps: int) -> None:
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
        import numpy as np
    except ImportError as exc:
        raise DependencyUnavailableError(
            "imageio, imageio-ffmpeg, and numpy are required to export video artifacts."
        ) from exc

    arrays: List[object] = [np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("Cannot export an empty frame sequence.")
    imageio.mimsave(path, arrays, fps=fps, codec="libx264", macro_block_size=1)
