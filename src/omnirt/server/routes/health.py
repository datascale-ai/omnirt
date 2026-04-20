"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request):
    engine = request.app.state.engine
    return {
        "ok": bool(engine.is_ready()),
        "job_store_backend": getattr(request.app.state, "job_store_backend", "memory"),
        "remote_worker_count": len(getattr(request.app.state, "remote_workers", []) or []),
    }


@router.get("/metrics")
async def metrics(request: Request):
    return PlainTextResponse(request.app.state.metrics.render(), media_type="text/plain; version=0.0.4")
