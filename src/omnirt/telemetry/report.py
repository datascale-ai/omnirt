"""Run report construction helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from omnirt.core.types import Artifact, BackendTimelineEntry, GenerateRequest, RunReport, StageEventRecord


def build_run_report(
    *,
    run_id: str,
    request: GenerateRequest,
    backend_name: str,
    timings: Dict[str, float],
    memory: Dict[str, float],
    backend_timeline: Iterable[BackendTimelineEntry],
    config_resolved: Dict[str, object],
    artifacts: Iterable[Artifact],
    error: Optional[str],
    latent_stats: Optional[Dict[str, float]] = None,
    cache_hits: Optional[Iterable[str]] = None,
    job_id: Optional[str] = None,
    enqueued_at_ms: Optional[int] = None,
    queue_wait_ms: Optional[float] = None,
    execution_mode: Optional[str] = None,
    stream_events: Optional[Iterable[StageEventRecord]] = None,
) -> RunReport:
    return RunReport(
        run_id=run_id,
        task=request.task,
        model=request.model,
        backend=backend_name,
        job_id=job_id,
        enqueued_at_ms=enqueued_at_ms,
        queue_wait_ms=queue_wait_ms,
        execution_mode=execution_mode,
        timings=dict(timings),
        memory=dict(memory),
        backend_timeline=list(backend_timeline),
        config_resolved=dict(config_resolved),
        artifacts=list(artifacts),
        error=error,
        latent_stats=dict(latent_stats) if latent_stats is not None else None,
        cache_hits=[str(item) for item in (cache_hits or [])],
        stream_events=list(stream_events or []),
    )
