"""User-facing request validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnirt.backends import resolve_backend
from omnirt.core.presets import resolve_preset
from omnirt.core.registry import ModelSpec, list_models
from omnirt.core.types import GenerateRequest, OmniRTError


@dataclass
class ValidationIssue:
    level: str
    message: str


@dataclass
class ValidationResult:
    request: GenerateRequest
    resolved_backend: Optional[str] = None
    resolved_inputs: Dict[str, Any] = field(default_factory=dict)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    model_spec: Optional[ModelSpec] = None
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    def add_error(self, message: str) -> None:
        self.issues.append(ValidationIssue(level="error", message=message))

    def add_warning(self, message: str) -> None:
        self.issues.append(ValidationIssue(level="warning", message=message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "request": self.request.to_dict(),
            "resolved_backend": self.resolved_backend,
            "resolved_inputs": dict(self.resolved_inputs),
            "resolved_config": dict(self.resolved_config),
            "model": self.model_spec.id if self.model_spec else None,
            "issues": [{"level": issue.level, "message": issue.message} for issue in self.issues],
        }

    def format_errors(self) -> str:
        return "\n".join(f"- {issue.message}" for issue in self.errors)


def validate_request(request: GenerateRequest, *, backend: Optional[str] = None) -> ValidationResult:
    result = ValidationResult(request=request)

    try:
        spec = list_models()[request.model]
    except KeyError:
        suggestions = get_close_matches(request.model, sorted(list_models()), n=3)
        hint = f" Nearby models: {', '.join(suggestions)}." if suggestions else ""
        result.add_error(f"Unknown model {request.model!r}.{hint}")
        return result

    result.model_spec = spec
    caps = spec.capabilities
    result.resolved_inputs = dict(request.inputs)
    user_config = dict(request.config)
    preset_name = user_config.pop("preset", None)

    result.resolved_config = dict(caps.default_config)
    if preset_name:
        try:
            result.resolved_config.update(resolve_preset(task=request.task, model=spec.id, preset=str(preset_name)))
        except ValueError as exc:
            result.add_error(str(exc))
        else:
            result.add_warning(f"Applied preset {preset_name!r}. Explicit config values still win over preset defaults.")
    result.resolved_config.update(user_config)

    if request.task != spec.task:
        example = caps.example or f"omnirt generate --task {spec.task} --model {spec.id}"
        result.add_error(
            f"Model {spec.id!r} only supports task {spec.task!r}, got {request.task!r}. Try: {example}"
        )

    allowed_inputs = set(caps.required_inputs) | set(caps.optional_inputs)
    unsupported_inputs = sorted(set(request.inputs) - allowed_inputs)
    if unsupported_inputs:
        supported = ", ".join(sorted(allowed_inputs)) if allowed_inputs else "<none>"
        result.add_error(f"Unsupported inputs for model {spec.id!r}: {unsupported_inputs}. Supported: [{supported}]")

    unsupported_config = sorted(set(user_config) - set(caps.supported_config))
    if unsupported_config:
        supported = ", ".join(sorted(caps.supported_config)) if caps.supported_config else "<none>"
        result.add_error(
            f"Unsupported config keys for model {spec.id!r}: {unsupported_config}. Supported: [{supported}]"
        )

    for key in caps.required_inputs:
        value = request.inputs.get(key)
        if value is None or value == "":
            result.add_error(f"Missing required input {key!r} for model {spec.id!r}.")

    image_path = request.inputs.get("image")
    if isinstance(image_path, str) and image_path:
        path = Path(image_path).expanduser()
        if not path.exists():
            result.add_error(f"Input image does not exist locally: {path}")

    scheduler_name = result.resolved_config.get("scheduler")
    if scheduler_name is not None and caps.supported_schedulers and scheduler_name not in caps.supported_schedulers:
        supported = ", ".join(caps.supported_schedulers)
        result.add_error(
            f"Unsupported scheduler {scheduler_name!r} for model {spec.id!r}. Supported: [{supported}]"
        )

    if request.adapters:
        if not caps.adapter_kinds:
            result.add_error(f"Model {spec.id!r} does not currently declare adapter support.")
        for adapter in request.adapters:
            if caps.adapter_kinds and adapter.kind not in caps.adapter_kinds:
                supported = ", ".join(caps.adapter_kinds)
                result.add_error(
                    f"Adapter kind {adapter.kind!r} is unsupported for model {spec.id!r}. Supported: [{supported}]"
                )

    selected_backend = backend if backend is not None else (request.backend or "auto")
    try:
        runtime = resolve_backend(selected_backend)
        result.resolved_backend = getattr(runtime, "name", None) or selected_backend
    except OmniRTError as exc:
        result.add_error(str(exc))
    else:
        if result.resolved_backend == "cpu-stub":
            result.add_warning(
                "Resolved backend is cpu-stub. Validation is fine, but full generation still needs CUDA or Ascend."
            )

    if caps.alias_of is not None:
        result.add_warning(f"Model {spec.id!r} is an alias of {caps.alias_of!r}.")

    return result
