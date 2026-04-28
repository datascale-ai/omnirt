"""Shared data types used across OmniRT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, TypeVar, Union

import yaml


TaskName = Literal[
    "text2image",
    "image2image",
    "inpaint",
    "edit",
    "text2video",
    "image2video",
    "audio2video",
    "text2audio",
]
BackendName = Literal["cuda", "ascend", "cpu-stub", "auto"]
ArtifactKind = Literal["image", "video", "audio"]
AdapterKind = Literal["lora"]

T = TypeVar("T")


class OmniRTError(RuntimeError):
    """Base error for OmniRT."""


class BackendUnavailableError(OmniRTError):
    """Raised when no supported backend is available."""


class ModelNotRegisteredError(OmniRTError):
    """Raised when a requested model id is missing."""


class WeightFormatError(OmniRTError):
    """Raised when a weight file does not match the supported format."""


class DependencyUnavailableError(OmniRTError):
    """Raised when an optional runtime dependency is missing."""


class InsufficientMemoryError(OmniRTError):
    """Raised when the runtime estimates insufficient device memory."""

    def __init__(self, *, model: str, estimated_gb: float, available_gb: float, hint: str) -> None:
        self.model = model
        self.estimated_gb = estimated_gb
        self.available_gb = available_gb
        self.hint = hint
        message = (
            f"InsufficientMemoryError(model={model!r}, estimated_gb={estimated_gb}, "
            f"available_gb={available_gb}, hint={hint!r})"
        )
        super().__init__(message)


@dataclass
class AdapterRef:
    kind: AdapterKind
    path: str
    scale: float = 1.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AdapterRef":
        return cls(
            kind=payload["kind"],
            path=payload["path"],
            scale=float(payload.get("scale", 1.0)),
        )


ArtifactTransport = Literal["path", "inline_bytes"]


class ArtifactTooLargeError(OmniRTError):
    """Raised when an artifact exceeds the configured inline transport budget."""

    def __init__(self, *, path: str, size_bytes: int, max_bytes: int) -> None:
        self.path = path
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"Artifact {path!r} is {size_bytes} bytes, exceeds inline transport limit "
            f"of {max_bytes} bytes. Set OMNIRT_ARTIFACT_INLINE_MAX_MB to raise the limit "
            f"or ensure client and worker share a filesystem and request transport='path'."
        )


@dataclass
class Artifact:
    kind: ArtifactKind
    path: str
    mime: str
    width: int
    height: int
    num_frames: Optional[int] = None
    transport: ArtifactTransport = "path"
    data_b64: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Artifact":
        # Tolerant of older payloads that lack transport/data_b64.
        accepted = {"kind", "path", "mime", "width", "height", "num_frames", "transport", "data_b64"}
        filtered = {key: value for key, value in payload.items() if key in accepted}
        filtered.setdefault("transport", "path")
        filtered.setdefault("data_b64", None)
        return cls(**filtered)


@dataclass
class Capabilities:
    device: str
    dtype_options: List[str]
    compile_available: bool
    device_count: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Capabilities":
        return cls(**payload)


@dataclass
class BackendAttempt:
    level: str
    ok: bool
    reason: Optional[str] = None
    selected: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BackendAttempt":
        return cls(
            level=payload["level"],
            ok=bool(payload["ok"]),
            reason=payload.get("reason"),
            selected=bool(payload.get("selected", False)),
        )


@dataclass
class BackendTimelineEntry:
    module: str
    attempts: List[BackendAttempt]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BackendTimelineEntry":
        return cls(
            module=payload["module"],
            attempts=[BackendAttempt.from_dict(item) for item in payload.get("attempts", [])],
        )


@dataclass
class StageEventRecord:
    event: str
    stage: str
    timestamp_ms: int
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StageEventRecord":
        return cls(
            event=str(payload["event"]),
            stage=str(payload["stage"]),
            timestamp_ms=int(payload["timestamp_ms"]),
            data=dict(payload.get("data", {})),
        )


@dataclass
class RunReport:
    run_id: str
    task: TaskName
    model: str
    backend: str
    job_id: Optional[str] = None
    trace_id: Optional[str] = None
    worker_id: Optional[str] = None
    enqueued_at_ms: Optional[int] = None
    queue_wait_ms: Optional[float] = None
    execution_mode: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, float] = field(default_factory=dict)
    backend_timeline: List[BackendTimelineEntry] = field(default_factory=list)
    config_resolved: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    error: Optional[str] = None
    latent_stats: Optional[Dict[str, float]] = None
    cache_hits: List[str] = field(default_factory=list)
    device_placement: Dict[str, str] = field(default_factory=dict)
    batch_size: int = 1
    batch_group_id: Optional[str] = None
    stream_events: List[StageEventRecord] = field(default_factory=list)
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunReport":
        return cls(
            run_id=payload["run_id"],
            task=payload["task"],
            model=payload["model"],
            backend=payload["backend"],
            job_id=payload.get("job_id"),
            trace_id=payload.get("trace_id"),
            worker_id=payload.get("worker_id"),
            enqueued_at_ms=payload.get("enqueued_at_ms"),
            queue_wait_ms=payload.get("queue_wait_ms"),
            execution_mode=payload.get("execution_mode"),
            timings=payload.get("timings", {}),
            memory=payload.get("memory", {}),
            backend_timeline=[BackendTimelineEntry.from_dict(item) for item in payload.get("backend_timeline", [])],
            config_resolved=payload.get("config_resolved", {}),
            artifacts=[Artifact.from_dict(item) for item in payload.get("artifacts", [])],
            error=payload.get("error"),
            latent_stats=payload.get("latent_stats"),
            cache_hits=[str(item) for item in payload.get("cache_hits", [])],
            device_placement={str(key): str(value) for key, value in payload.get("device_placement", {}).items()},
            batch_size=int(payload.get("batch_size", 1)),
            batch_group_id=payload.get("batch_group_id"),
            stream_events=[StageEventRecord.from_dict(item) for item in payload.get("stream_events", [])],
            schema_version=str(payload.get("schema_version", "0.0.0")),
        )


@dataclass
class GenerateResult:
    outputs: List[Artifact]
    metadata: RunReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outputs": [asdict(item) for item in self.outputs],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GenerateResult":
        return cls(
            outputs=[Artifact.from_dict(item) for item in payload.get("outputs", [])],
            metadata=RunReport.from_dict(payload["metadata"]),
        )


def is_generate_result_like(value: Any) -> bool:
    return bool(
        value is not None
        and hasattr(value, "outputs")
        and hasattr(value, "metadata")
        and callable(getattr(value, "to_dict", None))
    )


@dataclass
class GenerateRequest:
    task: TaskName
    model: str
    backend: BackendName = "auto"
    inputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    adapters: Optional[List[AdapterRef]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.adapters is None:
            payload["adapters"] = None
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GenerateRequest":
        adapters = payload.get("adapters")
        return cls(
            task=payload["task"],
            model=payload["model"],
            backend=payload.get("backend", "auto"),
            inputs=dict(payload.get("inputs", {})),
            config=dict(payload.get("config", {})),
            adapters=[AdapterRef.from_dict(item) for item in adapters] if adapters else None,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "GenerateRequest":
        file_path = Path(path)
        raw = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".json":
            payload = json.loads(raw)
        else:
            payload = yaml.safe_load(raw)
        return cls.from_dict(payload)


class TextToImageRequest(GenerateRequest):
    task: Literal["text2image"] = "text2image"

    def __init__(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs = {"prompt": prompt}
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        super().__init__(task="text2image", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class TextToVideoRequest(GenerateRequest):
    task: Literal["text2video"] = "text2video"

    def __init__(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs = {"prompt": prompt}
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        if num_frames is not None:
            inputs["num_frames"] = num_frames
        if fps is not None:
            inputs["fps"] = fps
        super().__init__(task="text2video", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class TextToAudioRequest(GenerateRequest):
    task: Literal["text2audio"] = "text2audio"

    def __init__(
        self,
        *,
        model: str,
        prompt: str,
        audio: str,
        reference_text: Optional[str] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs = {"prompt": prompt, "audio": audio}
        if reference_text:
            inputs["reference_text"] = reference_text
        super().__init__(task="text2audio", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class ImageToImageRequest(GenerateRequest):
    task: Literal["image2image"] = "image2image"

    def __init__(
        self,
        *,
        model: str,
        image: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs: Dict[str, Any] = {"image": image, "prompt": prompt}
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        super().__init__(task="image2image", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class InpaintRequest(GenerateRequest):
    task: Literal["inpaint"] = "inpaint"

    def __init__(
        self,
        *,
        model: str,
        image: str,
        mask: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs: Dict[str, Any] = {"image": image, "mask": mask, "prompt": prompt}
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        super().__init__(task="inpaint", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class EditRequest(GenerateRequest):
    task: Literal["edit"] = "edit"

    def __init__(
        self,
        *,
        model: str,
        image: Union[str, Sequence[str]],
        prompt: str,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        if isinstance(image, (list, tuple)):
            inputs: Dict[str, Any] = {"image": list(image), "prompt": prompt}
        else:
            inputs = {"image": image, "prompt": prompt}
        super().__init__(task="edit", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class ImageToVideoRequest(GenerateRequest):
    task: Literal["image2video"] = "image2video"

    def __init__(
        self,
        *,
        model: str,
        image: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs: Dict[str, Any] = {"image": image}
        if prompt:
            inputs["prompt"] = prompt
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        if num_frames is not None:
            inputs["num_frames"] = num_frames
        if fps is not None:
            inputs["fps"] = fps
        super().__init__(task="image2video", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


class AudioToVideoRequest(GenerateRequest):
    task: Literal["audio2video"] = "audio2video"

    def __init__(
        self,
        *,
        model: str,
        image: str,
        audio: str,
        prompt: Optional[str] = None,
        backend: BackendName = "auto",
        config: Optional[Dict[str, Any]] = None,
        adapters: Optional[List[AdapterRef]] = None,
    ) -> None:
        inputs: Dict[str, Any] = {"image": image, "audio": audio}
        if prompt:
            inputs["prompt"] = prompt
        super().__init__(task="audio2video", model=model, backend=backend, inputs=inputs, config=dict(config or {}), adapters=adapters)


def dataclass_to_dict(instance: Any) -> Dict[str, Any]:
    return asdict(instance)


def listify(items: Optional[Sequence[T]]) -> List[T]:
    return list(items) if items else []
