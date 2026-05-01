"""Model registration and lookup for OmniRT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

from omnirt.core.types import ModelNotRegisteredError


@dataclass
class ModelCapabilities:
    required_inputs: Tuple[str, ...] = ()
    optional_inputs: Tuple[str, ...] = ()
    supported_config: Tuple[str, ...] = ()
    default_config: Dict[str, Any] = field(default_factory=dict)
    supported_schedulers: Tuple[str, ...] = ()
    adapter_kinds: Tuple[str, ...] = ()
    artifact_kind: str = ""
    maturity: str = "experimental"
    summary: str = ""
    example: str = ""
    alias_of: Optional[str] = None
    supports_batching: bool = True
    domain: str = "digital-human"
    chain_role: str = ""
    realtime: bool = False


@dataclass
class ModelSpec:
    id: str
    task: str
    pipeline_cls: Type[Any]
    default_backend: str = "auto"
    resource_hint: Dict[str, Any] = field(default_factory=dict)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    execution_mode: Literal["legacy_call", "modular", "subprocess", "persistent_worker"] = "legacy_call"
    modular_pretrained_id: Optional[str] = None


_MODEL_REGISTRY: Dict[Tuple[str, str], ModelSpec] = {}

_PRIMARY_TASK_PRIORITY: Tuple[str, ...] = (
    "text2image",
    "image2image",
    "inpaint",
    "edit",
    "text2video",
    "image2video",
    "audio2video",
    "text2audio",
)
_EXECUTION_MODE_RUNTIME_CONFIG: Dict[str, Tuple[str, ...]] = {
    "legacy_call": (
        "device_map",
        "devices",
        "quantization",
        "quantization_backend",
        "enable_layerwise_casting",
        "layerwise_casting_storage_dtype",
        "layerwise_casting_compute_dtype",
        "cache",
        "enable_tea_cache",
        "tea_cache_ratio",
        "tea_cache_interval",
    ),
    "modular": (
        "device_map",
        "devices",
        "quantization",
        "quantization_backend",
        "enable_layerwise_casting",
        "layerwise_casting_storage_dtype",
        "layerwise_casting_compute_dtype",
        "cache",
        "enable_tea_cache",
        "tea_cache_ratio",
        "tea_cache_interval",
    ),
}


def clear_registry() -> None:
    _MODEL_REGISTRY.clear()


def register_model(
    *,
    id: str,
    task: str,
    default_backend: str = "auto",
    resource_hint: Optional[Dict[str, Any]] = None,
    capabilities: Optional[ModelCapabilities] = None,
    execution_mode: Literal["legacy_call", "modular", "subprocess", "persistent_worker"] = "legacy_call",
    modular_pretrained_id: Optional[str] = None,
) -> Callable[[Type[Any]], Type[Any]]:
    def decorator(pipeline_cls: Type[Any]) -> Type[Any]:
        registrations = list(getattr(pipeline_cls, "_omnirt_model_registrations", []))
        registrations.append(
            {
                "id": id,
                "task": task,
                "default_backend": default_backend,
                "resource_hint": dict(resource_hint or {}),
                "capabilities": capabilities or ModelCapabilities(),
                "execution_mode": execution_mode,
                "modular_pretrained_id": modular_pretrained_id,
            }
        )
        pipeline_cls._omnirt_model_registrations = registrations
        _MODEL_REGISTRY[(id, task)] = ModelSpec(
            id=id,
            task=task,
            pipeline_cls=pipeline_cls,
            default_backend=default_backend,
            resource_hint=dict(resource_hint or {}),
            capabilities=capabilities or ModelCapabilities(),
            execution_mode=execution_mode,
            modular_pretrained_id=modular_pretrained_id,
        )
        return pipeline_cls

    return decorator


def _primary_task_key(spec: ModelSpec) -> tuple:
    try:
        priority = _PRIMARY_TASK_PRIORITY.index(spec.task)
    except ValueError:
        priority = len(_PRIMARY_TASK_PRIORITY)
    return (priority, spec.task)


def has_model_variant(model_id: str, task: str) -> bool:
    return (model_id, task) in _MODEL_REGISTRY


def list_model_variants(model_id: str) -> Dict[str, ModelSpec]:
    variants = {task: spec for (registered_model_id, task), spec in _MODEL_REGISTRY.items() if registered_model_id == model_id}
    return dict(sorted(variants.items(), key=lambda item: _primary_task_key(item[1])))


def get_model(model_id: str, task: Optional[str] = None) -> ModelSpec:
    if task is not None:
        try:
            return _MODEL_REGISTRY[(model_id, task)]
        except KeyError as exc:
            raise ModelNotRegisteredError(f"Model {model_id!r} with task {task!r} is not registered.") from exc

    variants = list_model_variants(model_id)
    if not variants:
        raise ModelNotRegisteredError(f"Model {model_id!r} is not registered.")
    return min(variants.values(), key=_primary_task_key)


def list_models() -> Dict[str, ModelSpec]:
    models: Dict[str, ModelSpec] = {}
    for model_id, _task in _MODEL_REGISTRY:
        models[model_id] = get_model(model_id)
    return dict(sorted(models.items()))


def list_model_specs() -> Dict[Tuple[str, str], ModelSpec]:
    return dict(_MODEL_REGISTRY)


def supported_config_for_spec(spec: ModelSpec) -> Tuple[str, ...]:
    combined = list(spec.capabilities.supported_config)
    combined.extend(_EXECUTION_MODE_RUNTIME_CONFIG.get(spec.execution_mode, ()))
    return tuple(dict.fromkeys(combined))
