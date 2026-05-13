from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from omnirt.server import create_app  # noqa: E402
from omnirt.server.realtime_avatar import (  # noqa: E402
    AvatarVideoSpec,
    MAGIC_AUDIO,
    MAGIC_VIDEO,
    RealtimeAvatarService,
    RealtimeAvatarError,
    decode_jpeg_sequence,
    encode_jpeg_sequence,
    _scale_video_to_max_long_edge,
)  # noqa: E402


def _image_b64() -> str:
    return base64.b64encode(b"fake-image-bytes").decode("ascii")


def _png_bytes(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (128, 96, 64)).save(buf, format="PNG")
    return buf.getvalue()


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


def test_wav2lip_scaled_video_dimensions_are_h264_safe() -> None:
    video = AvatarVideoSpec(width=830, height=1108, fps=30, slice_len=28)

    scaled = _scale_video_to_max_long_edge(video, 832)

    assert scaled.width % 2 == 0
    assert scaled.height % 2 == 0
    assert scaled.width == 622
    assert scaled.height == 832


def test_flashtalk_compatible_ws_init_generate_and_close() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/flashtalk") as ws:
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


def test_flashtalk_compatible_ws_offloads_audio_push(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    calls = 0
    real_to_thread = avatar_routes.asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        nonlocal calls
        calls += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(avatar_routes.asyncio, "to_thread", tracking_to_thread)
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/avatar/flashtalk") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        init = ws.receive_json()
        ws.send_bytes(_audio_payload(init["slice_len"] * 16000 // init["fps"]))
        video = ws.receive_bytes()

    assert video[:4] == MAGIC_VIDEO
    assert calls == 1


def test_flashtalk_compatible_ws_root_alias_for_opentalking_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        assert ws.receive_json()["type"] == "init_ok"


def test_audio2video_models_reports_wav2lip_unavailable_by_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == []
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["connected"] is False
    assert statuses["flashtalk"]["reason"] == "fallback_runtime"
    assert statuses["wav2lip"]["connected"] is False
    assert statuses["quicktalk"]["connected"] is False
    assert statuses["quicktalk"]["reason"] == "runtime_not_enabled"


def test_audio2video_models_reports_resident_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_REALTIME_AVATAR_RUNTIME", "resident")
    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["connected"] is True
    assert statuses["flashtalk"]["reason"] == "resident_runtime"


def test_avatar_models_alias_reports_wav2lip_unavailable_by_default() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/avatar/models")

    assert response.status_code == 200
    assert response.json()["models"] == []


def test_audio2video_models_reports_proxy_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    async def fake_reachable(_url: str) -> bool:
        return True

    monkeypatch.setattr(avatar_routes, "_is_ws_url_reachable", fake_reachable)
    client = TestClient(create_app(default_backend="cpu-stub"))
    client.app.state.avatar_model_ws_urls = {
        "flashtalk": "ws://127.0.0.1:8765",
        "wav2lip": "ws://127.0.0.1:8767",
        "quicktalk": "ws://127.0.0.1:8768",
    }

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["flashtalk", "wav2lip", "quicktalk"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["reason"] == "proxy"
    assert statuses["wav2lip"]["connected"] is True
    assert statuses["quicktalk"]["connected"] is True
    assert statuses["quicktalk"]["reason"] == "proxy"


def test_audio2video_models_reads_proxy_targets_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    async def fake_reachable(_url: str) -> bool:
        return True

    monkeypatch.setenv("OMNIRT_AVATAR_FLASHTALK_WS_URL", "ws://127.0.0.1:8765")
    monkeypatch.setenv("OMNIRT_AVATAR_WAV2LIP_WS_URL", "ws://127.0.0.1:8767")
    monkeypatch.setenv("OMNIRT_AVATAR_QUICKTALK_WS_URL", "ws://127.0.0.1:8768")
    monkeypatch.setattr(avatar_routes, "_is_ws_url_reachable", fake_reachable)

    client = TestClient(create_app(default_backend="cpu-stub"))
    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["flashtalk", "wav2lip", "quicktalk"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["flashtalk"]["reason"] == "proxy"
    assert statuses["wav2lip"]["reason"] == "proxy"
    assert statuses["quicktalk"]["reason"] == "proxy"


def test_audio2video_models_reports_quicktalk_runtime() -> None:
    class FakeRouter:
        runtime_kind = "router"
        wav2lip = None
        quicktalk = object()

    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=FakeRouter())
    client = TestClient(app)

    response = client.get("/v1/audio2video/models")

    assert response.status_code == 200
    payload = response.json()
    assert "quicktalk" in payload["models"]
    statuses = {item["id"]: item for item in payload["statuses"]}
    assert statuses["quicktalk"]["connected"] is True
    assert statuses["quicktalk"]["reason"] == "quicktalk_runtime"


def test_flashtalk_compatible_ws_errors() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/flashtalk") as ws:
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


def test_flashtalk_compatible_ws_reports_runtime_errors() -> None:
    class FailingRuntime:
        def render_chunk(self, session, pcm_s16le):
            del session, pcm_s16le
            raise RuntimeError("model failed")

    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=FailingRuntime())
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/flashtalk") as ws:
        ws.send_json({"type": "init", "ref_image": _image_b64()})
        init = ws.receive_json()
        ws.send_bytes(_audio_payload(init["slice_len"] * 16000 // init["fps"]))
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "runtime_error"
    assert "model failed" in error["message"]


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


def test_wav2lip_init_accepts_postprocess_mode_and_metadata() -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    metadata = {
        "source_image_hash": "abc123",
        "animation": {
            "mouth_center": [0.5, 0.56],
            "mouth_rx": 0.06,
            "mouth_ry": 0.02,
            "outer_lip": [[0.45, 0.55], [0.50, 0.53], [0.55, 0.55], [0.50, 0.58]],
            "inner_mouth": [[0.47, 0.55], [0.53, 0.55], [0.50, 0.57]],
        },
    }

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "wav2lip_postprocess_mode": "opentalking_improved",
                "mouth_metadata": metadata,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "wav2lip"
    assert init["wav2lip_postprocess_mode"] == "opentalking_improved"


def test_wav2lip_init_accepts_frame_reference_dir(tmp_path: Path) -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    client.app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "reference_mode": "frames",
                "ref_frame_dir": str(frame_dir),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "wav2lip"
    assert init["reference_mode"] == "frames"
    assert "ref_frame_dir" not in init


def test_quicktalk_compatible_ws_accepts_template_video(tmp_path: Path) -> None:
    template = tmp_path / "template.mp4"
    template.write_bytes(b"video")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "template_mode": "video",
                "template_video": str(template),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"
    assert init["template_mode"] == "video"
    assert "template_video" not in init


def test_quicktalk_compatible_ws_accepts_template_frame_dir(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "template_mode": "frames",
                "template_frame_dir": str(frame_dir),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"
    assert init["template_mode"] == "frames"


def test_quicktalk_template_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    template = outside / "template.mp4"
    template.write_bytes(b"video")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[allowed])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "template_mode": "video",
                "template_video": str(template),
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_template_video"


def test_wav2lip_video_dimensions_respect_max_long_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_WAV2LIP_MAX_LONG_EDGE", "768")
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "width": 830,
                "height": 1108,
                "fps": 30,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["width"] == 574
    assert init["height"] == 768


def test_quicktalk_video_dimensions_default_to_900_long_edge() -> None:
    service = RealtimeAvatarService()
    session = service.create_session(
        model="quicktalk",
        image_bytes=_png_bytes((1600, 1200)),
        config={"width": 1600, "height": 1200},
    )

    assert session.video.width == 900
    assert session.video.height == 674
    assert session.video.fps == 25


def test_quicktalk_video_dimensions_respect_max_long_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIRT_QUICKTALK_MAX_LONG_EDGE", "512")
    client = TestClient(create_app(default_backend="cpu-stub"))

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "width": 830,
                "height": 1108,
                "fps": 30,
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["width"] == 384
    assert init["height"] == 512
    assert init["fps"] == 25


def test_quicktalk_init_accepts_asset_face_cache_path(tmp_path: Path) -> None:
    cache_path = tmp_path / "quicktalk" / "face_cache_v3_900.npz"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"cache")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "quicktalk_face_cache": str(cache_path),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert init["model"] == "quicktalk"


def test_quicktalk_face_cache_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    cache_path = outside / "face_cache.npz"
    cache_path.write_bytes(b"cache")
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[allowed])
    client = TestClient(app)

    with client.websocket_connect("/v1/audio2video/quicktalk") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "quicktalk_face_cache": str(cache_path),
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_quicktalk_face_cache"


def test_quicktalk_static_template_video_uses_session_dimensions(tmp_path: Path) -> None:
    from omnirt.models.quicktalk.runtime import QuickTalkRealtimeRuntime

    session = RealtimeAvatarService().create_session(
        model="quicktalk",
        image_bytes=_png_bytes((2048, 2048)),
        config={"width": 512, "height": 512},
    )
    runtime = QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )

    template = runtime._template_video_for(session)

    import cv2

    cap = cv2.VideoCapture(str(template))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 512
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512
    finally:
        cap.release()


def test_quicktalk_runtime_passes_asset_face_cache_to_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    captured: dict[str, object] = {}

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        quicktalk_runtime.QuickTalkRealtimeRuntime,
        "_worker_class",
        staticmethod(lambda: FakeWorker),
    )
    cache = tmp_path / "quicktalk" / "face_cache_v3_900.npz"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    session = RealtimeAvatarService(allowed_frame_roots=[tmp_path]).create_session(
        model="quicktalk",
        image_bytes=_png_bytes((64, 64)),
        config={"quicktalk_face_cache": str(cache)},
    )
    runtime = quicktalk_runtime.QuickTalkRealtimeRuntime(
        model_root=tmp_path / "model",
        checkpoint=tmp_path / "quicktalk.pth",
        template_cache_dir=tmp_path / "templates",
    )

    runtime._worker_for(session)

    assert captured["face_cache_file"] == tmp_path / "quicktalk" / "face_cache_v3_900.npz"


def test_wav2lip_init_accepts_frame_metadata_path(tmp_path: Path) -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    client.app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[tmp_path])

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "ref_frame_metadata_path": str(metadata_path),
            }
        )
        init = ws.receive_json()

    assert init["type"] == "init_ok"
    assert "ref_frame_metadata_path" not in init


def test_wav2lip_frame_reference_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    client = TestClient(create_app(default_backend="cpu-stub"))
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    client.app.state.realtime_avatar_service = RealtimeAvatarService(allowed_frame_roots=[allowed])

    with client.websocket_connect("/v1/audio2video/wav2lip") as ws:
        ws.send_json(
            {
                "type": "init",
                "ref_image": _image_b64(),
                "reference_mode": "frames",
                "ref_frame_dir": str(outside),
            }
        )
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_frame_dir"
    assert str(outside) not in error["message"]


def test_wav2lip_preload_endpoint_uses_runtime_cache(tmp_path: Path) -> None:
    class FakePreloadRuntime:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def preload_reference(self, session):
            self.calls.append(session)
            return {
                "type": "preload_result",
                "frames": 2,
                "elapsed_ms": 12.5,
                "cache_hit": len(self.calls) > 1,
            }

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    metadata_path = tmp_path / "mouth_metadata.json"
    metadata_path.write_text('{"frames": {}}', encoding="utf-8")
    runtime = FakePreloadRuntime()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=runtime, allowed_frame_roots=[tmp_path])
    client = TestClient(app)
    payload = {
        "ref_frame_dir": str(frame_dir),
        "ref_frame_metadata_path": str(metadata_path),
        "width": 24,
        "height": 24,
        "fps": 30,
        "preprocessed": True,
        "wav2lip_postprocess_mode": "opentalking_improved",
    }

    first = client.post("/v1/audio2video/wav2lip/preload", json=payload)
    second = client.post("/v1/audio2video/wav2lip/preload", json=payload)

    assert first.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert len(runtime.calls) == 2
    assert runtime.calls[0].reference_mode == "frames"
    assert runtime.calls[0].preprocessed is True


def test_wav2lip_preload_endpoint_offloads_runtime_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnirt.server.routes import avatar as avatar_routes

    class FakePreloadRuntime:
        def preload_reference(self, session):
            return {"type": "preload_result", "frames": 1, "elapsed_ms": 1.0, "cache_hit": False}

    calls = 0
    real_to_thread = avatar_routes.asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        nonlocal calls
        calls += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(avatar_routes.asyncio, "to_thread", tracking_to_thread)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FakePreloadRuntime(),
        allowed_frame_roots=[tmp_path],
    )
    client = TestClient(app)

    response = client.post(
        "/v1/audio2video/wav2lip/preload",
        json={"ref_frame_dir": str(frame_dir), "width": 24, "height": 24},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "preload_result"
    assert calls == 1


def test_wav2lip_preload_endpoint_reports_runtime_error(tmp_path: Path) -> None:
    class FailingPreloadRuntime:
        def preload_reference(self, session):
            del session
            raise RuntimeError("preload failed")

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    app = create_app(default_backend="cpu-stub")
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=FailingPreloadRuntime(),
        allowed_frame_roots=[tmp_path],
    )
    client = TestClient(app)

    response = client.post(
        "/v1/audio2video/wav2lip/preload",
        json={"ref_frame_dir": str(frame_dir), "width": 24, "height": 24},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "error"
    assert response.json()["code"] == "runtime_error"
    assert "preload failed" in response.json()["message"]
