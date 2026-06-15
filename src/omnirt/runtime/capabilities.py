"""Model capability manifests for public runtime integration surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from omnirt.core.registry import ModelSpec, supported_config_for_spec


RESIDENT_EXECUTION_MODES = frozenset({"persistent_worker"})


@dataclass(frozen=True)
class ModelCapabilityManifest:
    """Stable, serializable declaration of what a model can do at runtime."""

    schema_version: str
    model: str
    task: str
    tier: str
    role: str
    maturity: str
    inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    config: tuple[str, ...] = ()
    default_config: Mapping[str, Any] = field(default_factory=dict)
    streaming: bool = False
    resident: bool = False
    service_adapter: str | None = None
    backends: Mapping[str, str] = field(default_factory=dict)
    execution_mode: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "task": self.task,
            "tier": self.tier,
            "role": self.role,
            "maturity": self.maturity,
            "inputs": list(self.inputs),
            "optional_inputs": list(self.optional_inputs),
            "outputs": list(self.outputs),
            "config": list(self.config),
            "default_config": dict(self.default_config),
            "streaming": self.streaming,
            "resident": self.resident,
            "service_adapter": self.service_adapter,
            "backends": dict(self.backends),
            "execution_mode": self.execution_mode,
            "summary": self.summary,
        }


def default_role_for_task(task: str) -> str:
    if task == "text2audio":
        return "voice-generation"
    if task == "audio2text":
        return "voice-understanding"
    if task == "audio2video":
        return "realtime-avatar"
    if task in {"text2image", "image2image", "inpaint", "edit"}:
        return "avatar-asset"
    if task in {"text2video", "image2video", "video2video"}:
        return "idle-video"
    return "compatibility"


def infer_backend_status(spec: ModelSpec) -> dict[str, str]:
    if spec.capabilities.backend_status:
        return dict(spec.capabilities.backend_status)
    if spec.default_backend in {"cuda", "ascend", "cpu-stub"}:
        return {spec.default_backend: "supported"}
    accelerator = str(spec.resource_hint.get("accelerator", "")).lower()
    statuses: dict[str, str] = {}
    if "nvidia" in accelerator or "cuda" in accelerator:
        statuses["cuda"] = "supported"
    if "ascend" in accelerator or "910" in accelerator or "npu" in accelerator:
        statuses["ascend"] = "supported"
    if spec.task == "audio2text" or "cpu" in accelerator:
        statuses["cpu-stub"] = "validation-only" if spec.task != "audio2text" else "supported"
    return statuses or {"auto": "unknown"}


def capability_manifest_for_spec(spec: ModelSpec) -> ModelCapabilityManifest:
    caps = spec.capabilities
    artifact = caps.artifact_kind or _artifact_for_task(spec.task)
    streaming = bool(caps.streaming or caps.realtime)
    resident = bool(caps.resident or spec.execution_mode in RESIDENT_EXECUTION_MODES)
    role = caps.chain_role or default_role_for_task(spec.task)
    return ModelCapabilityManifest(
        schema_version="1.0.0",
        model=spec.id,
        task=spec.task,
        tier=caps.tier,
        role=role,
        maturity=caps.maturity,
        inputs=tuple(caps.required_inputs),
        optional_inputs=tuple(caps.optional_inputs),
        outputs=(artifact,) if artifact else (),
        config=tuple(supported_config_for_spec(spec)),
        default_config=dict(caps.default_config),
        streaming=streaming,
        resident=resident,
        service_adapter=caps.service_adapter or _default_service_adapter(spec.task, streaming=streaming),
        backends=infer_backend_status(spec),
        execution_mode=spec.execution_mode,
        summary=caps.summary,
    )


def validate_capability_manifest(data: Mapping[str, Any]) -> ModelCapabilityManifest:
    required = ("schema_version", "model", "task", "tier", "role", "maturity")
    missing = [key for key in required if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"capability manifest missing required fields: {', '.join(missing)}")
    for key in ("inputs", "optional_inputs", "outputs", "config"):
        value = data.get(key, ())
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"capability manifest field {key!r} must be a list")
    backends = data.get("backends", {})
    if backends is None:
        backends = {}
    if not isinstance(backends, Mapping):
        raise ValueError("capability manifest field 'backends' must be a mapping")
    default_config = data.get("default_config", {})
    if default_config is None:
        default_config = {}
    if not isinstance(default_config, Mapping):
        raise ValueError("capability manifest field 'default_config' must be a mapping")
    return ModelCapabilityManifest(
        schema_version=str(data["schema_version"]),
        model=str(data["model"]),
        task=str(data["task"]),
        tier=str(data["tier"]),
        role=str(data["role"]),
        maturity=str(data["maturity"]),
        inputs=tuple(str(item) for item in data.get("inputs", ()) or ()),
        optional_inputs=tuple(str(item) for item in data.get("optional_inputs", ()) or ()),
        outputs=tuple(str(item) for item in data.get("outputs", ()) or ()),
        config=tuple(str(item) for item in data.get("config", ()) or ()),
        default_config=dict(default_config),
        streaming=bool(data.get("streaming", False)),
        resident=bool(data.get("resident", False)),
        service_adapter=str(data["service_adapter"]) if data.get("service_adapter") else None,
        backends={str(key): str(value) for key, value in backends.items()},
        execution_mode=str(data.get("execution_mode", "")),
        summary=str(data.get("summary", "")),
    )


def _artifact_for_task(task: str) -> str:
    if task == "text2audio":
        return "audio"
    if task == "audio2text":
        return "text"
    if task in {"audio2video", "text2video", "image2video", "video2video"}:
        return "video"
    if task in {"text2image", "image2image", "inpaint", "edit"}:
        return "image"
    return ""


def _default_service_adapter(task: str, *, streaming: bool) -> str | None:
    if task == "text2audio":
        return "text2audio.service.v1"
    if task == "audio2video" and streaming:
        return "realtime-avatar.ws.v1"
    return None
