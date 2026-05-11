"""Shared request normalization helpers for HTTP routes."""

from __future__ import annotations

from omnirt.core.types import GenerateRequest
from omnirt.server.model_aliases import resolve_model_alias


def allowed_model_tiers(app_state) -> tuple[str, ...]:
    return tuple(getattr(app_state, "allowed_model_tiers", ()) or ())


def model_tier_allowed(model_spec, app_state) -> bool:
    allowed = allowed_model_tiers(app_state)
    return not allowed or model_spec.capabilities.tier in allowed


def normalize_generate_request(raw_request: GenerateRequest, app_state) -> GenerateRequest:
    backend = raw_request.backend if raw_request.backend != "auto" else app_state.default_backend
    merged_config = dict(getattr(app_state, "default_request_config", {}) or {})
    merged_config.update(raw_request.config)
    return GenerateRequest(
        task=raw_request.task,
        model=resolve_model_alias(raw_request.model, app_state.model_aliases),
        backend=backend,
        inputs=dict(raw_request.inputs),
        config=merged_config,
        adapters=raw_request.adapters,
    )
