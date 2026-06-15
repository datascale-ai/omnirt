from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from omnirt.server import create_app


def test_text2audio_indextts_streams_l16_chunks(monkeypatch) -> None:
    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        async def synthesize_pcm_stream(self, text, *, voice=None, config=None):
            assert text == "你好，IndexTTS。"
            assert voice == "voice-a"
            assert config["quick_streaming_tokens"] == 8
            assert config["num_beams"] == 1
            assert config["top_p"] == 0.75
            assert config["max_mel_tokens"] == 640
            yield b"\x00\x00\x01\x00"
            yield b"\x02\x00"

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.post(
        "/v1/text2audio/indextts",
        json={
            "text": "你好，IndexTTS。",
            "voice": "voice-a",
            "quick_streaming_tokens": 8,
            "num_beams": 1,
            "top_p": 0.75,
            "max_mel_tokens": 640,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/L16")
    assert response.headers["x-audio-sample-rate"] == "16000"
    assert response.content == b"\x00\x00\x01\x00\x02\x00"



def test_text2audio_indextts_passes_token_window_config(monkeypatch) -> None:
    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        async def synthesize_pcm_stream(self, text, *, voice=None, config=None):
            assert text == "你好。"
            assert config["streaming_mode"] == "token_window"
            assert config["token_window_size"] == 16
            assert config["token_window_hop"] == 8
            assert config["token_window_overlap_ms"] == 80
            assert config["token_window_context"] == 24
            yield b"\x00\x00"

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.post(
        "/v1/text2audio/indextts",
        json={
            "text": "你好。",
            "streaming_mode": "token_window",
            "token_window_size": 16,
            "token_window_hop": 8,
            "token_window_overlap_ms": 80,
            "token_window_context": 24,
        },
    )

    assert response.status_code == 200
    assert response.content == b"\x00\x00"


def test_text2audio_indextts_passes_segment_quality_config(monkeypatch) -> None:
    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        async def synthesize_pcm_stream(self, text, *, voice=None, config=None):
            assert text == "你好。"
            assert config["streaming_mode"] == "segment"
            assert config["max_text_tokens_per_segment"] == 80
            assert config["quick_streaming_tokens"] == 4
            yield b"\x00\x00"

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.post(
        "/v1/text2audio/indextts",
        json={
            "text": "你好。",
            "streaming_mode": "segment",
            "max_text_tokens_per_segment": 80,
            "quick_streaming_tokens": 4,
        },
    )

    assert response.status_code == 200
    assert response.content == b"\x00\x00"


def test_text2audio_indextts_passes_emotion_config(monkeypatch) -> None:
    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        async def synthesize_pcm_stream(self, text, *, voice=None, config=None):
            assert text == "你好。"
            assert config["emo_alpha"] == 0.6
            assert config["use_emo_text"] is True
            assert config["emo_text"] == "开心、自然"
            assert config["use_random"] is False
            assert config["emo_vector"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
            assert config["emo_audio_prompt"] == "/tmp/indextts-emotion.wav"
            yield b"\x00\x00"

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.post(
        "/v1/text2audio/indextts",
        json={
            "text": "你好。",
            "emo_alpha": 0.6,
            "use_emo_text": True,
            "emo_text": "开心、自然",
            "use_random": False,
            "emo_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            "emo_audio_prompt": "/tmp/indextts-emotion.wav",
        },
    )

    assert response.status_code == 200
    assert response.content == b"\x00\x00"

def test_text2audio_indextts_status_disabled(monkeypatch) -> None:
    monkeypatch.delenv("OMNIRT_INDEXTTS_RUNTIME", raising=False)

    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/text2audio/models")

    assert response.status_code == 200
    assert response.json()["statuses"][0]["id"] == "indextts"
    assert response.json()["statuses"][0]["connected"] is False


def test_text2audio_only_app_exposes_status(monkeypatch) -> None:
    from omnirt.server.text2audio_app import create_text2audio_app

    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.text2audio_app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_text2audio_app())

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    response = client.get("/v1/text2audio/models")
    assert response.status_code == 200
    assert response.json()["statuses"][0]["connected"] is True

def test_text2audio_only_app_starts_indextts_warmup(monkeypatch) -> None:
    from omnirt.server.text2audio_app import create_text2audio_app

    calls: list[tuple[str, int]] = []

    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        def warmup(self, *, text="", max_chunks=1):
            calls.append((text, max_chunks))

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setenv("OMNIRT_INDEXTTS_PRELOAD", "1")
    monkeypatch.setenv("OMNIRT_INDEXTTS_WARMUP_TEXT", "启动预热")
    monkeypatch.setattr("omnirt.server.text2audio_app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    with TestClient(create_text2audio_app()) as client:
        assert client.get("/healthz").status_code == 200

    assert calls == [("启动预热", 1)]

def test_text2audio_models_exposes_indextts_streaming_boundary(monkeypatch) -> None:
    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {
                "ready": True,
                "model": "IndexTeam/IndexTTS-2",
                "streaming": True,
                "streaming_granularity": "segment",
                "model_internal_streaming": False,
                "streaming_note": "IndexTTS2 stream_return yields complete s2mel/BigVGAN segments.",
            }

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))

    response = client.get("/v1/text2audio/models")

    status = response.json()["statuses"][0]
    assert status["streaming"] is True
    assert status["streaming_granularity"] == "segment"
    assert status["model_internal_streaming"] is False
    assert status["connected"] is True


def test_text2audio_generic_stream_route_uses_service_adapter_contract(monkeypatch) -> None:
    captured = {}

    class FakeRuntime:
        sample_rate = 24000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        async def synthesize_pcm_stream(self, text, *, voice=None, config=None):
            captured["text"] = text
            captured["voice"] = voice
            captured["config"] = config
            yield b"\x01\x00"

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))
    response = client.post(
        "/v1/text2audio/stream",
        json={
            "model": "indextts",
            "text": "你好。",
            "speaker_profile": "voice-a",
            "prompt_audio": "/tmp/reference.wav",
            "reference_text": "参考文本",
            "config": {"streaming_mode": "segment"},
        },
    )

    assert response.status_code == 200
    assert response.headers["x-omnirt-adapter-schema"] == "text2audio.service.v1"
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.content == b"\x01\x00"
    assert captured == {
        "text": "你好。",
        "voice": "voice-a",
        "config": {
            "streaming_mode": "segment",
            "reference_text": "参考文本",
            "prompt_audio": "/tmp/reference.wav",
        },
    }


def test_text2audio_generic_health_metrics_and_warmup(monkeypatch) -> None:
    calls = []

    class FakeRuntime:
        sample_rate = 16000

        def status(self):
            return {"ready": True, "model": "IndexTeam/IndexTTS-2"}

        def warmup(self, *, text="", max_chunks=1):
            calls.append((text, max_chunks))

    monkeypatch.setenv("OMNIRT_INDEXTTS_RUNTIME", "1")
    monkeypatch.setattr("omnirt.server.app.create_indextts_runtime_from_env", lambda: FakeRuntime())

    client = TestClient(create_app(default_backend="cpu-stub"))

    assert client.get("/health").json()["ok"] is True
    models = client.get("/models").json()
    assert models["adapter_schema"] == "text2audio.service.v1"
    metrics = client.get("/v1/text2audio/metrics").json()
    assert metrics["ready_count"] == 1

    warmup = client.post("/warmup", json={"model": "indextts", "text": "启动预热", "max_chunks": 2})
    assert warmup.status_code == 200
    assert warmup.json()["ok"] is True
    assert calls == [("启动预热", 2)]
