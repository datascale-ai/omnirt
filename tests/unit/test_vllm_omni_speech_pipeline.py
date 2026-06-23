from __future__ import annotations

import base64
from pathlib import Path

import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.vllm_omni_speech.pipeline import VLLMOmniSpeechPipeline


class FakeRuntime(BackendRuntime):
    name = "cpu-stub"
    device_name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {}

    def available_memory_gb(self):
        return None


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="vllm-omni-speech",
        task="text2audio",
        pipeline_cls=VLLMOmniSpeechPipeline,
        default_backend="auto",
        resource_hint={"accelerator": "external"},
    )


def test_vllm_omni_speech_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("vllm-omni-speech", task="text2audio")

    assert spec.task == "text2audio"
    assert spec.default_backend == "auto"
    assert spec.capabilities.artifact_kind == "audio"
    assert spec.capabilities.chain_role == "voice-generation"
    assert "server_url" in spec.capabilities.supported_config
    assert "upstream_model" in spec.capabilities.supported_config
    assert "initial_codec_chunk_frames" in spec.capabilities.supported_config


def test_vllm_omni_speech_pipeline_posts_openai_compatible_request(tmp_path, monkeypatch) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"fake wav")
    captured = {}

    class FakeResponse:
        content = b"\x00\x00\x01\x00"
        headers = {"content-type": "audio/L16; rate=24000; channels=1"}

        def raise_for_status(self):
            captured["raised"] = False

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["closed"] = True

        def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    class FakeHttpx:
        Client = FakeClient

    monkeypatch.setattr(VLLMOmniSpeechPipeline, "_httpx", staticmethod(lambda: FakeHttpx))

    request = GenerateRequest(
        task="text2audio",
        model="vllm-omni-speech",
        backend="cpu-stub",
        inputs={
            "prompt": "你好，这是 vLLM-Omni 语音适配测试。",
            "audio": str(reference_audio),
            "reference_text": "参考音色文本。",
        },
        config={
            "server_url": "http://127.0.0.1:8091",
            "output_dir": str(tmp_path / "outputs"),
            "request_id": "fixed-request",
            "timeout": 12,
            "upstream_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "voice": "vivian",
            "response_format": "pcm",
            "stream": True,
            "language": "Chinese",
            "task_type": "Base",
            "initial_codec_chunk_frames": 3,
            "non_streaming_mode": False,
            "api_key": "secret",
        },
    )
    pipeline = VLLMOmniSpeechPipeline(runtime=FakeRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert captured["url"] == "http://127.0.0.1:8091/v1/audio/speech"
    assert captured["timeout"] == 12.0
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["input"] == "你好，这是 vLLM-Omni 语音适配测试。"
    assert captured["json"]["model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert captured["json"]["voice"] == "vivian"
    assert captured["json"]["response_format"] == "pcm"
    assert captured["json"]["stream"] is True
    assert captured["json"]["language"] == "Chinese"
    assert captured["json"]["task_type"] == "Base"
    assert captured["json"]["initial_codec_chunk_frames"] == 3
    assert captured["json"]["non_streaming_mode"] is False
    assert captured["json"]["ref_text"] == "参考音色文本。"
    assert captured["json"]["ref_audio"].startswith("data:audio/wav;base64,")
    encoded = captured["json"]["ref_audio"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"fake wav"
    assert captured["closed"] is True
    assert Path(result.outputs[0].path).read_bytes() == b"\x00\x00\x01\x00"
    assert result.outputs[0].path.endswith(".pcm")
    assert result.outputs[0].mime == "audio/L16; rate=24000; channels=1"
    assert result.metadata.config_resolved["ref_audio"] == "<inline-data-url>"
    assert result.metadata.config_resolved["api_key_configured"] is True


def test_vllm_omni_speech_pipeline_rejects_empty_response(tmp_path, monkeypatch) -> None:
    class FakeResponse:
        content = b""
        headers = {}

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, timeout):
            del timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, url, json, headers):
            del url, json, headers
            return FakeResponse()

    class FakeHttpx:
        Client = FakeClient

    monkeypatch.setattr(VLLMOmniSpeechPipeline, "_httpx", staticmethod(lambda: FakeHttpx))

    request = GenerateRequest(
        task="text2audio",
        model="vllm-omni-speech",
        backend="cpu-stub",
        inputs={"prompt": "test"},
        config={"output_dir": str(tmp_path / "outputs")},
    )
    pipeline = VLLMOmniSpeechPipeline(runtime=FakeRuntime(), model_spec=build_model_spec())

    with pytest.raises(RuntimeError, match="empty audio response"):
        pipeline.run(request)
