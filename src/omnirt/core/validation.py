"""User-facing request validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnirt.backends import resolve_backend
from omnirt.core.presets import resolve_preset
from omnirt.core.registry import ModelSpec, get_model, list_model_variants, list_models
from omnirt.core.types import GenerateRequest, ModelNotRegisteredError, OmniRTError


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
        spec = get_model(request.model, task=request.task)
    except ModelNotRegisteredError:
        variants = list_model_variants(request.model)
        if variants:
            supported_tasks = ", ".join(variants)
            result.add_error(
                f"Model {request.model!r} supports tasks [{supported_tasks}], got {request.task!r}."
            )
            return result
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

    for media_key, label in (("image", "image"), ("audio", "audio"), ("mask", "mask")):
        media_path = request.inputs.get(media_key)
        if isinstance(media_path, str) and media_path:
            path = Path(media_path).expanduser()
            if not path.exists():
                result.add_error(f"Input {label} does not exist locally: {path}")

    strength = result.resolved_config.get("strength")
    if strength is not None:
        try:
            strength_value = float(strength)
        except (TypeError, ValueError):
            result.add_error(f"Invalid strength value {strength!r}; expected a float between 0 and 1.")
        else:
            if not 0.0 <= strength_value <= 1.0:
                result.add_error(f"Invalid strength value {strength!r}; expected a float between 0 and 1.")

    offload_flags = (
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_offload",
    )
    enabled_offload_flags = [key for key in offload_flags if result.resolved_config.get(key)]
    if len(enabled_offload_flags) > 1:
        result.add_error(
            "Offload config flags are mutually exclusive; choose only one of "
            f"{', '.join(offload_flags)}."
        )
    group_offload_type = result.resolved_config.get("group_offload_type")
    if group_offload_type is not None and group_offload_type not in {"block_level", "leaf_level"}:
        result.add_error("group_offload_type must be either 'block_level' or 'leaf_level'.")

    repo_root = result.resolved_config.get("repo_path")
    repo_root_path = Path(str(repo_root)).expanduser() if isinstance(repo_root, str) and repo_root else None
    for config_key, label in (
        ("repo_path", "repository checkout"),
        ("ckpt_dir", "checkpoint directory"),
        ("wav2vec_dir", "wav2vec directory"),
        ("ascend_env_script", "Ascend environment script"),
        ("python_executable", "Python executable"),
    ):
        config_path = result.resolved_config.get(config_key)
        if isinstance(config_path, str) and config_path:
            path = Path(config_path).expanduser()
            if config_key in {"ckpt_dir", "wav2vec_dir"} and not path.is_absolute() and repo_root_path is not None:
                path = repo_root_path / path
            if not path.exists():
                result.add_error(f"Configured {label} does not exist locally: {path}")

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
