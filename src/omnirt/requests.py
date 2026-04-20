"""Ergonomic request builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omnirt.core.types import AdapterRef, BackendName, ImageToVideoRequest, TextToImageRequest, TextToVideoRequest


def text2image(
    *,
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> TextToImageRequest:
    return TextToImageRequest(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )


def text2video(
    *,
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> TextToVideoRequest:
    return TextToVideoRequest(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        fps=fps,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )


def image2video(
    *,
    model: str,
    image: str,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> ImageToVideoRequest:
    return ImageToVideoRequest(
        model=model,
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        fps=fps,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )
