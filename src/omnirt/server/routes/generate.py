"""Primary generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from omnirt.api import validate
from omnirt.core.types import GenerateRequest, is_generate_result_like
from omnirt.server.model_aliases import resolve_model_alias
from omnirt.server.schemas import GenerateSubmission

router = APIRouter()


def _normalize_request(raw_request: GenerateRequest, request: Request) -> GenerateRequest:
    backend = raw_request.backend if raw_request.backend != "auto" else request.app.state.default_backend
    merged_config = dict(getattr(request.app.state, "default_request_config", {}) or {})
    merged_config.update(raw_request.config)
    return GenerateRequest(
        task=raw_request.task,
        model=resolve_model_alias(raw_request.model, request.app.state.model_aliases),
        backend=backend,
        inputs=dict(raw_request.inputs),
        config=merged_config,
        adapters=raw_request.adapters,
    )


@router.post("/v1/generate")
async def generate_endpoint(payload: GenerateSubmission, request: Request):
    normalized = _normalize_request(payload.to_request(), request)
    validation = validate(normalized, backend=normalized.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    if payload.async_run:
        job = request.app.state.engine.submit(normalized, model_spec=validation.model_spec)
        return job.to_dict()
    result = request.app.state.engine.run_sync(normalized, model_spec=validation.model_spec)
    if is_generate_result_like(result):
        return result.to_dict()
    return {"result": result}
