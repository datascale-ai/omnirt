from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import ValidationError

from omnirt.server.schemas import RealtimeAvatarEvent, Text2AudioSynthesizeRequest


def test_text2audio_request_schema_defaults_to_streaming_pcm() -> None:
    payload = Text2AudioSynthesizeRequest(text="hello")

    assert payload.model == "indextts"
    assert payload.audio_format == "pcm_s16le"
    assert payload.stream is True
    assert payload.config == {}


def test_realtime_avatar_event_schema_accepts_metrics_and_errors() -> None:
    metrics = RealtimeAvatarEvent(
        type="metrics",
        session_id="s1",
        chunk_index=1,
        metrics={"ttff_ms": 120.5, "first_video_chunk_ms": 180.0},
    )
    error = RealtimeAvatarEvent(type="error", code="runtime_error", message="worker crashed")

    assert metrics.metrics["ttff_ms"] == 120.5
    assert error.code == "runtime_error"


def test_realtime_avatar_event_schema_rejects_unknown_event_type() -> None:
    with pytest.raises(ValidationError):
        RealtimeAvatarEvent(type="unknown")
