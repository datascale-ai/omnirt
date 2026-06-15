"""Runtime profile schema for multi-model OmniRT deployments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_BACKENDS = {"auto", "cuda", "ascend", "cpu-stub"}


@dataclass(frozen=True)
class RuntimeProfileModel:
    id: str
    task: str
    backend: str = "auto"
    service: str = ""
    port: int | None = None
    resources: Mapping[str, Any] = field(default_factory=dict)
    warmup: Mapping[str, Any] = field(default_factory=dict)
    concurrency: int = 1
    degrade_to: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "task": self.task,
            "backend": self.backend,
            "service": self.service,
            "resources": dict(self.resources),
            "warmup": dict(self.warmup),
            "concurrency": self.concurrency,
            "config": dict(self.config),
        }
        if self.port is not None:
            payload["port"] = self.port
        if self.degrade_to:
            payload["degrade_to"] = self.degrade_to
        return payload


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    version: str = "1.0.0"
    description: str = ""
    models: tuple[RuntimeProfileModel, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "defaults": dict(self.defaults),
            "environment": dict(self.environment),
            "models": [model.to_dict() for model in self.models],
        }


def load_runtime_profile(path: str | Path) -> RuntimeProfile:
    profile_path = Path(path).expanduser()
    suffix = profile_path.suffix.lower()
    text = profile_path.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{profile_path} must contain a mapping")
    return validate_runtime_profile(raw)


def validate_runtime_profile(data: Mapping[str, Any]) -> RuntimeProfile:
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("runtime profile field 'name' is required")
    models_data = data.get("models")
    if not isinstance(models_data, list) or not models_data:
        raise ValueError("runtime profile field 'models' must be a non-empty list")
    defaults = data.get("defaults", {}) or {}
    environment = data.get("environment", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("runtime profile field 'defaults' must be a mapping")
    if not isinstance(environment, Mapping):
        raise ValueError("runtime profile field 'environment' must be a mapping")
    return RuntimeProfile(
        name=name,
        version=str(data.get("version") or "1.0.0"),
        description=str(data.get("description") or ""),
        defaults=dict(defaults),
        environment={str(key): str(value) for key, value in environment.items()},
        models=tuple(_validate_profile_model(item, index) for index, item in enumerate(models_data)),
    )


def _validate_profile_model(data: Any, index: int) -> RuntimeProfileModel:
    if not isinstance(data, Mapping):
        raise ValueError(f"runtime profile models[{index}] must be a mapping")
    model_id = str(data.get("id", "")).strip()
    task = str(data.get("task", "")).strip()
    if not model_id:
        raise ValueError(f"runtime profile models[{index}].id is required")
    if not task:
        raise ValueError(f"runtime profile models[{index}].task is required")
    backend = str(data.get("backend") or "auto")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"runtime profile models[{index}].backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    port = data.get("port")
    if port is not None:
        port = int(port)
        if port <= 0 or port > 65535:
            raise ValueError(f"runtime profile models[{index}].port must be a TCP port")
    concurrency = int(data.get("concurrency") or 1)
    if concurrency < 1:
        raise ValueError(f"runtime profile models[{index}].concurrency must be >= 1")
    for key in ("resources", "warmup", "config"):
        value = data.get(key, {}) or {}
        if not isinstance(value, Mapping):
            raise ValueError(f"runtime profile models[{index}].{key} must be a mapping")
    return RuntimeProfileModel(
        id=model_id,
        task=task,
        backend=backend,
        service=str(data.get("service") or ""),
        port=port,
        resources=dict(data.get("resources", {}) or {}),
        warmup=dict(data.get("warmup", {}) or {}),
        concurrency=concurrency,
        degrade_to=str(data["degrade_to"]) if data.get("degrade_to") else None,
        config=dict(data.get("config", {}) or {}),
    )
