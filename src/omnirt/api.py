"""Public OmniRT API entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omnirt.backends import resolve_backend
from omnirt.core.registry import ModelSpec, get_model, list_models
from omnirt.core.types import GenerateRequest, GenerateResult
from omnirt.core.validation import ValidationResult, validate_request
from omnirt.models import ensure_registered


RequestLike = Union[GenerateRequest, Dict[str, Any], str, Path]


def _coerce_request(request: RequestLike) -> GenerateRequest:
    if isinstance(request, GenerateRequest):
        return request
    if isinstance(request, (str, Path)):
        return GenerateRequest.from_file(request)
    if isinstance(request, dict):
        return GenerateRequest.from_dict(request)
    raise TypeError(f"Unsupported request type: {type(request)!r}")


def list_available_models(*, include_aliases: bool = True) -> List[ModelSpec]:
    ensure_registered()
    specs = list(list_models().values())
    if not include_aliases:
        specs = [spec for spec in specs if spec.capabilities.alias_of is None]
    return sorted(specs, key=lambda spec: spec.id)


def describe_model(model_id: str) -> ModelSpec:
    ensure_registered()
    return get_model(model_id)


def validate(request: RequestLike, *, backend: Optional[str] = None) -> ValidationResult:
    ensure_registered()
    req = _coerce_request(request)
    return validate_request(req, backend=backend)


class OmniModelPipeline:
    def __init__(self, *, model: str, backend: Optional[str] = None) -> None:
        self.model = model
        self.backend = backend
        self.spec = describe_model(model)

    def __call__(self, **kwargs: Any) -> GenerateResult:
        caps = self.spec.capabilities
        inputs = dict(kwargs.pop("inputs", {}) or {})
        config = dict(kwargs.pop("config", {}) or {})
        adapters = kwargs.pop("adapters", None)
        backend = kwargs.pop("backend", self.backend)

        known_input_keys = set(caps.required_inputs) | set(caps.optional_inputs)
        for key in list(kwargs):
            if key in known_input_keys:
                inputs[key] = kwargs.pop(key)
            elif key in caps.supported_config:
                config[key] = kwargs.pop(key)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown pipeline call arguments for model {self.model!r}: {unknown}")

        request = GenerateRequest(
            task=self.spec.task,
            model=self.spec.id,
            backend=backend or "auto",
            inputs=inputs,
            config=config,
            adapters=adapters,
        )
        return generate(request, backend=backend)

    def validate(self, **kwargs: Any) -> ValidationResult:
        caps = self.spec.capabilities
        inputs = dict(kwargs.pop("inputs", {}) or {})
        config = dict(kwargs.pop("config", {}) or {})
        adapters = kwargs.pop("adapters", None)
        backend = kwargs.pop("backend", self.backend)

        known_input_keys = set(caps.required_inputs) | set(caps.optional_inputs)
        for key in list(kwargs):
            if key in known_input_keys:
                inputs[key] = kwargs.pop(key)
            elif key in caps.supported_config:
                config[key] = kwargs.pop(key)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown pipeline validation arguments for model {self.model!r}: {unknown}")

        request = GenerateRequest(
            task=self.spec.task,
            model=self.spec.id,
            backend=backend or "auto",
            inputs=inputs,
            config=config,
            adapters=adapters,
        )
        return validate(request, backend=backend)


def pipeline(model: str, *, backend: Optional[str] = None) -> OmniModelPipeline:
    return OmniModelPipeline(model=model, backend=backend)


def generate(request: RequestLike, *, backend: Optional[str] = None) -> GenerateResult:
    """Run a generation request through a registered pipeline.

    Backend resolution priority: explicit ``backend`` argument > ``req.backend`` > ``"auto"``.
    """

    ensure_registered()
    req = _coerce_request(request)
    validation = validate(req, backend=backend)
    if not validation.ok:
        raise ValueError(f"Request validation failed:\n{validation.format_errors()}")

    normalized_request = GenerateRequest(
        task=req.task,
        model=req.model,
        backend=req.backend,
        inputs=dict(validation.resolved_inputs),
        config=dict(validation.resolved_config),
        adapters=req.adapters,
    )

    spec = get_model(normalized_request.model)
    selected = backend if backend is not None else (req.backend or "auto")
    runtime = resolve_backend(selected)
    pipeline = spec.pipeline_cls(runtime=runtime, model_spec=spec, adapters=normalized_request.adapters)
    return pipeline.run(normalized_request)
