"""Ergonomic request builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from omnirt.core.types import (
    AdapterRef,
    AudioToVideoRequest,
    BackendName,
    EditRequest,
    ImageToImageRequest,
    ImageToVideoRequest,
    InpaintRequest,
    TextToAudioRequest,
    TextToImageRequest,
    TextToVideoRequest,
)


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


def text2audio(
    *,
    model: str,
    prompt: str,
    audio: str,
    reference_text: Optional[str] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> TextToAudioRequest:
    return TextToAudioRequest(
        model=model,
        prompt=prompt,
        audio=audio,
        reference_text=reference_text,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )


def image2image(
    *,
    model: str,
    image: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> ImageToImageRequest:
    return ImageToImageRequest(
        model=model,
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )


def inpaint(
    *,
    model: str,
    image: str,
    mask: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> InpaintRequest:
    return InpaintRequest(
        model=model,
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )


def edit(
    *,
    model: str,
    image: Union[str, Sequence[str]],
    prompt: str,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> EditRequest:
    return EditRequest(
        model=model,
        image=image,
        prompt=prompt,
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


def audio2video(
    *,
    model: str,
    image: str,
    audio: str,
    prompt: Optional[str] = None,
    backend: BackendName = "auto",
    adapters: Optional[List[AdapterRef]] = None,
    **config: Any,
) -> AudioToVideoRequest:
    return AudioToVideoRequest(
        model=model,
        image=image,
        audio=audio,
        prompt=prompt,
        backend=backend,
        config=dict(config),
        adapters=adapters,
    )
