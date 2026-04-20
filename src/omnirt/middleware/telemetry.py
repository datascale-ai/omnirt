"""Helpers for enriching run metadata with engine-level telemetry."""

from __future__ import annotations

from typing import Iterable

from omnirt.core.types import GenerateResult, StageEventRecord


def attach_stream_events(result: GenerateResult, events: Iterable[StageEventRecord]) -> GenerateResult:
    result.metadata.stream_events = list(events)
    result.metadata.schema_version = "0.3.0"
    return result
