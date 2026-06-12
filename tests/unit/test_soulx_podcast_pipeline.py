from pathlib import Path

import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.soulx_podcast.pipeline import SoulXPodcastPipeline


class FakeCudaRuntime(BackendRuntime):
    name = "cuda"
    device_name = "cuda"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {"peak_mb": 16.0}

    def available_memory_gb(self):
        return 24.0


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="soulx-podcast-1.7b",
        task="text2audio",
        pipeline_cls=SoulXPodcastPipeline,
        default_backend="cuda",
        resource_hint={"min_vram_gb": 16, "dtype": "bf16/fp16"},
    )


def test_soulx_podcast_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("soulx-podcast-1.7b", task="text2audio")

    assert spec.task == "text2audio"
    assert spec.default_backend == "cuda"
    assert spec.capabilities.artifact_kind == "audio"
    assert spec.capabilities.chain_role == "voice-generation"
    assert "server_url" in spec.capabilities.supported_config
    assert "prompt_audios" in spec.capabilities.supported_config


def test_soulx_podcast_pipeline_posts_single_speaker_request(tmp_path, monkeypatch) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"fake wav")
    captured = {}

    class FakeResponse:
        content = b"RIFFfake"

        def raise_for_status(self):
            captured["raised"] = False

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["closed"] = True

        def post(self, url, files):
            captured["url"] = url
            captured["files"] = []
            for name, payload in files:
                if name == "prompt_audio":
                    filename, handle, mime = payload
                    captured["files"].append((name, filename, handle.read(), mime))
                else:
                    captured["files"].append((name, payload[1]))
            return FakeResponse()

    class FakeHttpx:
        Client = FakeClient

    monkeypatch.setattr(SoulXPodcastPipeline, "_httpx", staticmethod(lambda: FakeHttpx))

    request = GenerateRequest(
        task="text2audio",
        model="soulx-podcast-1.7b",
        backend="cuda",
        inputs={
            "prompt": "欢迎收听 OmniRT 播客。",
            "audio": str(reference_audio),
            "reference_text": "参考音色文本。",
        },
        config={
            "server_url": "http://127.0.0.1:18080",
            "output_dir": str(tmp_path / "outputs"),
            "request_id": "fixed-request",
            "timeout": 12,
            "seed": 1234,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    )
    pipeline = SoulXPodcastPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert captured["url"] == "http://127.0.0.1:18080/generate"
    assert captured["timeout"] == 12.0
    assert ("dialogue_text", "欢迎收听 OmniRT 播客。") in captured["files"]
    assert ("prompt_texts", "参考音色文本。") in captured["files"]
    assert ("seed", "1234") in captured["files"]
    assert ("temperature", "0.7") in captured["files"]
    assert ("top_k", "40") in captured["files"]
    assert ("top_p", "0.9") in captured["files"]
    assert ("repetition_penalty", "1.1") in captured["files"]
    assert ("prompt_audio", "reference.wav", b"fake wav", "audio/wav") in captured["files"]
    assert captured["closed"] is True
    assert Path(result.outputs[0].path).read_bytes() == b"RIFFfake"
    assert result.outputs[0].kind == "audio"
    assert result.outputs[0].mime == "audio/wav"
    assert result.metadata.config_resolved["server_url"] == "http://127.0.0.1:18080"


def test_soulx_podcast_pipeline_posts_multi_speaker_request(tmp_path, monkeypatch) -> None:
    speaker_a = tmp_path / "a.wav"
    speaker_b = tmp_path / "b.mp3"
    speaker_a.write_bytes(b"a")
    speaker_b.write_bytes(b"b")
    captured = {}

    class FakeResponse:
        content = b"RIFFmulti"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, timeout):
            del timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, url, files):
            del url
            captured["names"] = [name for name, _payload in files]
            captured["texts"] = [payload[1] for name, payload in files if name == "prompt_texts"]
            captured["audio"] = [(payload[0], payload[2]) for name, payload in files if name == "prompt_audio"]
            return FakeResponse()

    class FakeHttpx:
        Client = FakeClient

    monkeypatch.setattr(SoulXPodcastPipeline, "_httpx", staticmethod(lambda: FakeHttpx))

    request = GenerateRequest(
        task="text2audio",
        model="soulx-podcast-1.7b",
        backend="cuda",
        inputs={"prompt": "[S1] 你好。[S2] 欢迎回来。", "audio": str(speaker_a)},
        config={
            "output_dir": str(tmp_path / "outputs"),
            "prompt_audios": [str(speaker_a), str(speaker_b)],
            "prompt_texts": ["一号说话人。", "二号说话人。"],
        },
    )
    pipeline = SoulXPodcastPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    pipeline.run(request)

    assert captured["names"].count("prompt_texts") == 2
    assert captured["names"].count("prompt_audio") == 2
    assert captured["texts"] == ["一号说话人。", "二号说话人。"]
    assert captured["audio"] == [("a.wav", "audio/wav"), ("b.mp3", "audio/mpeg")]


def test_soulx_podcast_pipeline_rejects_mismatched_multi_speaker_inputs(tmp_path) -> None:
    speaker_a = tmp_path / "a.wav"
    speaker_b = tmp_path / "b.wav"
    speaker_a.write_bytes(b"a")
    speaker_b.write_bytes(b"b")
    request = GenerateRequest(
        task="text2audio",
        model="soulx-podcast-1.7b",
        backend="cuda",
        inputs={"prompt": "test", "audio": str(speaker_a)},
        config={
            "prompt_audios": [str(speaker_a), str(speaker_b)],
            "prompt_texts": ["only one"],
        },
    )
    pipeline = SoulXPodcastPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    with pytest.raises(ValueError, match="same length"):
        pipeline.prepare_conditions(request)

