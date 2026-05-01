from __future__ import annotations

import base64
import json
import struct

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from omnirt.server import create_app
from omnirt.server.realtime_avatar import (
    MAGIC_AUDIO,
    MAGIC_VIDEO,
    RealtimeAvatarError,
    decode_jpeg_sequence,
    encode_jpeg_sequence,
)


def _image_b64() -> str:
    return base64.b64encode(b"fake-image-bytes").decode("ascii")


def _audio_payload(chunk_samples: int) -> bytes:
    return MAGIC_AUDIO + (b"\0\0" * chunk_samples)


def test_video_jpeg_sequence_round_trip() -> None:
    payload = encode_jpeg_sequence([b"jpeg-1", b"jpeg-2"])

    assert payload[:4] == MAGIC_VIDEO
    assert decode_jpeg_sequence(payload) == [b"jpeg-1", b"jpeg-2"]


def test_video_jpeg_sequence_rejects_malformed_frame_length() -> None:
    payload = MAGIC_VIDEO + struct.pack("<I", 1) + struct.pack("<I", 99) + b"tiny"

    with pytest.raises(RealtimeAvatarError) as exc:
        decode_jpeg_sequence(payload)

    assert exc.value.code == "bad_video_chunk"


def test_flashtalk_compatible_ws_init_generate_and_close() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/avatar/flashtalk") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64(), "prompt": "talk", "seed": 1})
        init = ws.receive_json()
        assert init["type"] == "init_ok"
        assert init["fps"] == 25
        assert init["slice_len"] == 28

        ws.send_bytes(_audio_payload(init["slice_len"] * 16000 // init["fps"]))
        video = ws.receive_bytes()
        assert video[:4] == MAGIC_VIDEO
        assert len(decode_jpeg_sequence(video)) == 1

        ws.send_json({"type": "close"})
        assert ws.receive_json()["type"] == "close_ok"


def test_flashtalk_compatible_ws_root_alias_for_opentalking_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        assert ws.receive_json()["type"] == "init_ok"


def test_flashtalk_compatible_ws_errors() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/avatar/flashtalk") as ws:
        ws.send_json({"type": "init"})
        missing = ws.receive_json()
        assert missing["type"] == "error"
        assert missing["code"] == "missing_image"

        ws.send_json({"type": "init", "ref_image": "not-base64"})
        bad_b64 = ws.receive_json()
        assert bad_b64["type"] == "error"
        assert bad_b64["code"] == "bad_image_base64"

        ws.send_json({"type": "init", "ref_image": _image_b64()})
        init = ws.receive_json()
        assert init["type"] == "init_ok"

        ws.send_bytes(b"NOPE")
        bad_magic = ws.receive_json()
        assert bad_magic["type"] == "error"
        assert bad_magic["code"] == "bad_audio_magic"

        ws.send_bytes(MAGIC_AUDIO + b"\0")
        bad_chunk = ws.receive_json()
        assert bad_chunk["type"] == "error"
        assert bad_chunk["code"] == "bad_audio_chunk"


def test_native_realtime_avatar_ws_flow() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/avatar/realtime") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "session.create",
                    "model": "soulx-flashtalk-14b",
                    "backend": "cpu-stub",
                    "inputs": {"image_b64": _image_b64(), "prompt": "talk"},
                    "config": {"chunk_samples": 16, "width": 32, "height": 32},
                }
            )
        )
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session_id"].startswith("avt_")
        assert created["trace_id"].startswith("trace_")
        assert created["audio"]["chunk_samples"] == 16
        assert created["video"]["width"] == 32

        ws.send_bytes(_audio_payload(16))
        metrics = ws.receive_json()
        assert metrics["type"] == "metrics"
        assert metrics["chunk_index"] == 1
        video = ws.receive_bytes()
        assert video[:4] == MAGIC_VIDEO

        ws.send_json({"type": "session.cancel"})
        assert ws.receive_json()["type"] == "session.cancelled"

        ws.send_json({"type": "session.close"})
        assert ws.receive_json()["type"] == "session.closed"
