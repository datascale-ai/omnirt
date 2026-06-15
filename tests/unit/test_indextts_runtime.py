from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omnirt.models.indextts.runtime import (
    IndexTTSStreamingRuntime,
    create_indextts_runtime_from_env,
)


def test_indextts_runtime_yields_first_segment_before_second_finishes(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        prompt = tmp_path / "prompt.wav"
        prompt.write_bytes(b"prompt")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\n", encoding="utf-8")
        calls: list[dict[str, object]] = []
        release_second = False

        class FakeTensor:
            def __init__(self, data: np.ndarray):
                self._data = data

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._data

        class FakeIndexTTS2:
            def __init__(self, **kwargs):
                calls.append({"init": kwargs})

            def infer(self, spk_audio_prompt, text, output_path, **kwargs):
                calls.append(
                    {
                        "spk_audio_prompt": spk_audio_prompt,
                        "text": text,
                        "output_path": output_path,
                        "kwargs": kwargs,
                    }
                )
                yield FakeTensor(np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32))
                yield "silence-placeholder"
                while not release_second:
                    import time

                    time.sleep(0.01)
                yield FakeTensor(np.array([0.5, 0.0], dtype=np.float32))

        fake_module = SimpleNamespace(IndexTTS2=FakeIndexTTS2)
        monkeypatch.setitem(sys.modules, "indextts.infer_v2", fake_module)

        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(prompt),
            sample_rate=16000,
            model_sample_rate=22050,
            chunk_ms=1000,
            max_text_tokens_per_segment=24,
            quick_streaming_tokens=24,
            interval_silence_ms=0,
            streaming_mode="segment",
            device="cpu",
            use_fp16=False,
        )

        stream = runtime.synthesize_pcm_stream("第一段，第二段。")
        first = await stream.__anext__()

        assert np.frombuffer(first, dtype="<i2").size > 0
        assert calls[0]["init"]["model_dir"] == str(model_dir)
        assert calls[1]["spk_audio_prompt"] == str(prompt)
        assert calls[1]["output_path"] is None
        assert calls[1]["kwargs"]["stream_return"] is True
        assert calls[1]["kwargs"]["max_text_tokens_per_segment"] == 24
        assert calls[1]["kwargs"]["quick_streaming_tokens"] == 24
        assert calls[1]["kwargs"]["interval_silence"] == 0

        release_second = True
        rest = [chunk async for chunk in stream]

        assert rest

    asyncio.run(run())


def test_indextts_segment_runtime_passes_emotion_config_to_infer(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        prompt = tmp_path / "prompt.wav"
        prompt.write_bytes(b"prompt")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\n", encoding="utf-8")
        calls: list[dict[str, object]] = []

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([0.0, 0.2, -0.2, 0.0], dtype=np.float32)

        class FakeIndexTTS2:
            def __init__(self, **kwargs):
                calls.append({"init": kwargs})

            def infer(self, spk_audio_prompt, text, output_path, **kwargs):
                calls.append({"kwargs": kwargs})
                yield FakeTensor()

        monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))
        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(prompt),
            sample_rate=16000,
            model_sample_rate=22050,
            chunk_ms=1000,
            streaming_mode="segment",
            device="cpu",
            use_fp16=False,
        )

        chunks = [
            chunk
            async for chunk in runtime.synthesize_pcm_stream(
                "你好。",
                config={
                    "emo_alpha": 0.6,
                    "use_emo_text": True,
                    "emo_text": "开心、自然",
                    "use_random": False,
                    "emo_vector": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "emo_audio_prompt": str(tmp_path / "emotion.wav"),
                },
            )
        ]

        assert chunks
        kwargs = calls[1]["kwargs"]
        assert kwargs["emo_alpha"] == 0.6
        assert kwargs["use_emo_text"] is True
        assert kwargs["emo_text"] == "开心、自然"
        assert kwargs["use_random"] is False
        assert kwargs["emo_vector"] == [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert kwargs["emo_audio_prompt"] == str(tmp_path / "emotion.wav")

    asyncio.run(run())


def test_indextts_token_window_uses_emotion_reference_prompt(tmp_path) -> None:
    runtime = IndexTTSStreamingRuntime(model_dir="x", cfg_path="x", prompt_audio="speaker.wav")
    emotion = tmp_path / "emotion.wav"
    emotion.write_bytes(b"RIFFemotion")

    assert runtime._emotion_audio_prompt_from_config({"emo_audio_prompt": str(emotion)}, "speaker.wav") == str(emotion)
    assert runtime._emotion_audio_prompt_from_config({}, "speaker.wav") == "speaker.wav"


def test_indextts_token_window_emotion_mix_uses_manual_vector_without_random() -> None:
    import torch

    runtime = IndexTTSStreamingRuntime(model_dir="x", cfg_path="x", prompt_audio="x")
    engine = SimpleNamespace(
        device="cpu",
        emo_num=[2] * 8,
        emo_matrix=[torch.ones(2, 3) * float(i + 1) for i in range(8)],
        spk_matrix=[torch.eye(2, 3) for _ in range(8)],
    )
    style = torch.tensor([[1.0, 0.0, 0.0]])

    vec = runtime._emotion_vector_from_config(
        engine,
        "你好。",
        {"emo_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "emo_alpha": 1.0},
    )
    emovec_mat, weight = runtime._emotion_mix_from_vector(engine, style, vec, False)

    assert float(torch.sum(weight)) == pytest.approx(0.8)
    assert torch.allclose(emovec_mat, torch.tensor([[0.8, 0.8, 0.8]]))

def test_indextts_runtime_status_reports_missing_inputs(tmp_path) -> None:
    runtime = IndexTTSStreamingRuntime(
        model_dir=str(tmp_path / "missing-model"),
        cfg_path=str(tmp_path / "missing-model" / "config.yaml"),
        prompt_audio=str(tmp_path / "prompt.wav"),
    )

    status = runtime.status()

    assert status["ready"] is False
    assert status["model_dir_exists"] is False
    assert status["cfg_path_exists"] is False
    assert status["prompt_audio_exists"] is False


def test_indextts_runtime_normalizes_ascend_device_and_defaults_to_fp16(monkeypatch, tmp_path) -> None:
    runtime = IndexTTSStreamingRuntime(
        model_dir=str(tmp_path / "model"),
        cfg_path=str(tmp_path / "model" / "config.yaml"),
        prompt_audio=str(tmp_path / "prompt.wav"),
        device="ascend",
        use_cuda_kernel=True,
    )

    status = runtime.status()

    assert runtime.device == "npu:0"
    assert runtime.use_fp16 is True
    assert runtime.use_cuda_kernel is False
    assert status["device"] == "npu:0"
    assert status["use_fp16"] is True
    assert status["use_cuda_kernel"] is False

    monkeypatch.setenv("OMNIRT_INDEXTTS_NPU_INDEX", "3")
    runtime = IndexTTSStreamingRuntime(
        model_dir=str(tmp_path / "model"),
        cfg_path=str(tmp_path / "model" / "config.yaml"),
        prompt_audio=str(tmp_path / "prompt.wav"),
        device="npu",
    )

    assert runtime.device == "npu:3"


def test_indextts_runtime_requires_torch_npu_before_loading_npu_engine(monkeypatch, tmp_path) -> None:
    import builtins

    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            raise AssertionError("engine must not be constructed without torch_npu")

    monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch_npu":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        device="npu:0",
    )

    with pytest.raises(RuntimeError, match="requires torch_npu"):
        runtime._load_engine()


def test_indextts_runtime_passes_npu_device_to_engine_when_torch_npu_exists(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    calls: list[dict[str, object]] = []

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        device="ascend:1",
        use_fp16=None,
        use_cuda_kernel=True,
    )

    runtime._load_engine()

    assert calls[0]["device"] == "npu:1"
    assert calls[0]["use_fp16"] is True
    assert calls[0]["use_cuda_kernel"] is False


def test_indextts_runtime_maps_quick_tokens_to_index_tts_infer_signature(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        prompt = tmp_path / "prompt.wav"
        prompt.write_bytes(b"prompt")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\n", encoding="utf-8")
        calls: list[dict[str, object]] = []

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([0.0, 0.2, -0.2, 0.0], dtype=np.float32)

        class FakeIndexTTS2:
            def __init__(self, **kwargs):
                calls.append({"init": kwargs})

            def infer(
                self,
                spk_audio_prompt,
                text,
                output_path,
                *,
                stream_return=False,
                max_text_tokens_per_segment=120,
                more_segment_before=0,
                **generation_kwargs,
            ):
                calls.append(
                    {
                        "stream_return": stream_return,
                        "max_text_tokens_per_segment": max_text_tokens_per_segment,
                        "more_segment_before": more_segment_before,
                        "generation_kwargs": generation_kwargs,
                    }
                )
                yield FakeTensor()

        monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))

        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(prompt),
            sample_rate=16000,
            model_sample_rate=22050,
            chunk_ms=1000,
            quick_streaming_tokens=7,
            streaming_mode="segment",
            device="cpu",
            use_fp16=False,
        )

        chunks = [chunk async for chunk in runtime.synthesize_pcm_stream("第一段，第二段。")]

        assert chunks
        assert calls[1]["stream_return"] is True
        assert calls[1]["more_segment_before"] == 7
        assert "quick_streaming_tokens" not in calls[1]["generation_kwargs"]

    asyncio.run(run())


def test_indextts_runtime_passes_low_latency_generation_config(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        prompt = tmp_path / "prompt.wav"
        prompt.write_bytes(b"prompt")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\n", encoding="utf-8")
        calls: list[dict[str, object]] = []

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([0.0, 0.2, -0.2, 0.0], dtype=np.float32)

        class FakeIndexTTS2:
            def __init__(self, **kwargs):
                calls.append({"init": kwargs})

            def infer(self, spk_audio_prompt, text, output_path, **kwargs):
                calls.append({"kwargs": kwargs})
                yield FakeTensor()

        monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))

        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(prompt),
            sample_rate=16000,
            model_sample_rate=22050,
            chunk_ms=1000,
            num_beams=1,
            top_p=0.75,
            top_k=20,
            temperature=0.7,
            repetition_penalty=8.0,
            max_mel_tokens=640,
            streaming_mode="segment",
            device="cpu",
            use_fp16=False,
        )

        chunks = [
            chunk
            async for chunk in runtime.synthesize_pcm_stream(
                "第一段，第二段。",
                config={"temperature": 0.6, "max_mel_tokens": 512},
            )
        ]

        assert chunks
        kwargs = calls[1]["kwargs"]
        assert kwargs["num_beams"] == 1
        assert kwargs["top_p"] == 0.75
        assert kwargs["top_k"] == 20
        assert kwargs["temperature"] == 0.6
        assert kwargs["repetition_penalty"] == 8.0
        assert kwargs["max_mel_tokens"] == 512

    asyncio.run(run())


def test_indextts_runtime_uses_system_voice_prompt(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        default_prompt = tmp_path / "default.wav"
        default_prompt.write_bytes(b"default")
        voice_dir = tmp_path / "voices" / "system" / "indextts-clear-cn"
        voice_dir.mkdir(parents=True)
        voice_prompt = voice_dir / "prompt.wav"
        voice_prompt.write_bytes(b"voice")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\n", encoding="utf-8")
        calls: list[dict[str, object]] = []

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([0.0, 0.2, -0.2, 0.0], dtype=np.float32)

        class FakeIndexTTS2:
            def __init__(self, **kwargs):
                calls.append({"init": kwargs})

            def infer(self, spk_audio_prompt, text, output_path, **kwargs):
                calls.append(
                    {
                        "spk_audio_prompt": spk_audio_prompt,
                        "text": text,
                        "output_path": output_path,
                        "kwargs": kwargs,
                    }
                )
                yield FakeTensor()

        monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))
        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(default_prompt),
            voices_dir=str(tmp_path / "voices"),
            sample_rate=16000,
            model_sample_rate=22050,
            chunk_ms=1000,
            streaming_mode="segment",
            device="cpu",
            use_fp16=False,
        )

        chunks = [chunk async for chunk in runtime.synthesize_pcm_stream("你好。", voice="indextts-clear-cn")]

        assert chunks
        assert calls[1]["spk_audio_prompt"] == str(voice_prompt)

    asyncio.run(run())


def test_indextts_runtime_rewrites_hub_assets_to_local_dirs(monkeypatch, tmp_path) -> None:
    w2v_dir = tmp_path / "facebook__w2v-bert-2.0"
    w2v_dir.mkdir()
    (w2v_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    maskgct_dir = tmp_path / "amphion__MaskGCT-ms"
    (maskgct_dir / "semantic_codec").mkdir(parents=True)
    (maskgct_dir / "semantic_codec" / "model.safetensors").write_bytes(b"maskgct")
    campplus_dir = tmp_path / "funasr__campplus"
    campplus_dir.mkdir()
    (campplus_dir / "campplus_cn_common.bin").write_bytes(b"campplus")
    bigvgan_dir = tmp_path / "nvidia__bigvgan_v2_22khz_80band_256x"
    bigvgan_dir.mkdir()
    (bigvgan_dir / "config.json").write_text("{}", encoding="utf-8")
    (bigvgan_dir / "bigvgan_generator.pt").write_bytes(b"bigvgan")

    calls: dict[str, object] = {}

    class FeatureExtractor:
        @staticmethod
        def from_pretrained(value, *args, **kwargs):
            calls["feature"] = value
            return "feature"

    def hub_download(repo_id, filename, *args, **kwargs):
        calls.setdefault("hub", []).append((repo_id, filename))
        return f"remote://{repo_id}/{filename}"

    class BigVGAN:
        @staticmethod
        def from_pretrained(value, *args, **kwargs):
            calls["bigvgan"] = value
            return "bigvgan"

    module = SimpleNamespace(
        SeamlessM4TFeatureExtractor=FeatureExtractor,
        hf_hub_download=hub_download,
    )
    bigvgan_module = SimpleNamespace(BigVGAN=BigVGAN)
    monkeypatch.setitem(sys.modules, "indextts.s2mel.modules.bigvgan.bigvgan", bigvgan_module)

    runtime = IndexTTSStreamingRuntime(
        w2v_bert_dir=str(w2v_dir),
        maskgct_dir=str(maskgct_dir),
        campplus_dir=str(campplus_dir),
        bigvgan_dir=str(bigvgan_dir),
    )

    runtime._patch_local_runtime_assets(module)

    assert FeatureExtractor.from_pretrained("facebook/w2v-bert-2.0") == "feature"
    assert calls["feature"] == str(w2v_dir)
    assert module.hf_hub_download("amphion/MaskGCT", "semantic_codec/model.safetensors") == str(maskgct_dir / "semantic_codec" / "model.safetensors")
    assert module.hf_hub_download("funasr/campplus", "campplus_cn_common.bin") == str(campplus_dir / "campplus_cn_common.bin")
    assert BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x") == "bigvgan"
    assert calls["bigvgan"] == str(bigvgan_dir)

def test_env_runtime_prefers_maskgct_dir_with_required_asset(monkeypatch, tmp_path) -> None:
    empty_dir = tmp_path / "amphion__MaskGCT"
    empty_dir.mkdir()
    populated_dir = tmp_path / "amphion__MaskGCT-ms"
    (populated_dir / "semantic_codec").mkdir(parents=True)
    (populated_dir / "semantic_codec" / "model.safetensors").write_bytes(b"maskgct")
    model_dir = tmp_path / "IndexTeam__IndexTTS-2"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")

    monkeypatch.setenv("OMNIRT_LOCAL_AUDIO_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNIRT_INDEXTTS_PROMPT_AUDIO", str(prompt))
    monkeypatch.delenv("OMNIRT_INDEXTTS_MASKGCT_DIR", raising=False)

    runtime = create_indextts_runtime_from_env()

    assert runtime.maskgct_dir == str(populated_dir)

def test_env_runtime_defaults_to_balanced_realtime_segments(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "IndexTeam__IndexTTS-2"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")

    monkeypatch.setenv("OMNIRT_LOCAL_AUDIO_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNIRT_INDEXTTS_PROMPT_AUDIO", str(prompt))
    monkeypatch.delenv("OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT", raising=False)
    monkeypatch.delenv("OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS", raising=False)
    monkeypatch.setenv("OMNIRT_INDEXTTS_NUM_BEAMS", "1")
    monkeypatch.setenv("OMNIRT_INDEXTTS_TOP_P", "0.75")

    runtime = create_indextts_runtime_from_env()
    status = runtime.status()

    assert runtime.max_text_tokens_per_segment == 80
    assert runtime.quick_streaming_tokens == 4
    assert runtime.num_beams == 1
    assert runtime.top_p == 0.75
    assert status["max_text_tokens_per_segment"] == 80
    assert status["num_beams"] == 1
    assert status["top_p"] == 0.75


def test_env_runtime_segment_mode_keeps_short_text_segments(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "IndexTeam__IndexTTS-2"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")

    monkeypatch.setenv("OMNIRT_LOCAL_AUDIO_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNIRT_INDEXTTS_PROMPT_AUDIO", str(prompt))
    monkeypatch.setenv("OMNIRT_INDEXTTS_STREAMING_MODE", "segment")
    monkeypatch.delenv("OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT", raising=False)

    runtime = create_indextts_runtime_from_env()
    status = runtime.status()

    assert runtime.streaming_mode == "segment"
    assert runtime.max_text_tokens_per_segment == 24
    assert status["max_text_tokens_per_segment"] == 24


def test_env_runtime_defaults_to_token_window_streaming(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "IndexTeam__IndexTTS-2"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")

    monkeypatch.setenv("OMNIRT_LOCAL_AUDIO_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNIRT_INDEXTTS_PROMPT_AUDIO", str(prompt))
    monkeypatch.delenv("OMNIRT_INDEXTTS_STREAMING_MODE", raising=False)

    runtime = create_indextts_runtime_from_env()
    status = runtime.status()

    assert runtime.streaming_mode == "token_window"
    assert runtime.num_beams == 1
    assert runtime.token_window_size == 40
    assert runtime.token_window_hop == 96
    assert runtime.token_window_context == 8
    assert runtime.token_window_overlap_ms == 60
    assert status["streaming_granularity"] == "token_window"
    assert status["model_internal_streaming"] is True
    assert status["token_window_size"] == 40
    assert status["token_window_hop"] == 96
    assert status["token_window_context"] == 8
    assert status["token_window_overlap_ms"] == 60


def test_env_runtime_supports_experimental_token_window_streaming(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "IndexTeam__IndexTTS-2"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("test: true\n", encoding="utf-8")
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")

    monkeypatch.setenv("OMNIRT_LOCAL_AUDIO_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("OMNIRT_INDEXTTS_PROMPT_AUDIO", str(prompt))
    monkeypatch.setenv("OMNIRT_INDEXTTS_STREAMING_MODE", "token_window")
    monkeypatch.setenv("OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE", "16")
    monkeypatch.setenv("OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP", "8")
    monkeypatch.setenv("OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS", "80")
    monkeypatch.setenv("OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT", "24")
    monkeypatch.setenv("OMNIRT_INDEXTTS_NUM_BEAMS", "3")

    runtime = create_indextts_runtime_from_env()
    status = runtime.status()

    assert runtime.streaming_mode == "token_window"
    assert runtime.token_window_size == 16
    assert runtime.token_window_hop == 8
    assert runtime.token_window_overlap_ms == 80
    assert runtime.token_window_context == 24
    assert runtime.num_beams == 1
    assert status["streaming_granularity"] == "token_window"
    assert status["model_internal_streaming"] is True
    assert status["streaming_experimental"] is True
    assert status["token_window_size"] == 16
    assert status["token_window_hop"] == 8
    assert status["token_window_overlap_ms"] == 80
    assert status["token_window_context"] == 24
    assert "experimental" in status["streaming_note"]

def test_indextts_runtime_warmup_loads_engine_and_runs_optional_text(monkeypatch, tmp_path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([0.0, 0.1], dtype=np.float32)

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            calls.append({"init": kwargs})

        def infer(self, spk_audio_prompt, text, output_path, **kwargs):
            calls.append({"text": text, "kwargs": kwargs})
            yield FakeTensor()

    monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=FakeIndexTTS2))

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        sample_rate=16000,
        model_sample_rate=22050,
        chunk_ms=1000,
        max_text_tokens_per_segment=6,
        quick_streaming_tokens=6,
        device="cpu",
        use_fp16=False,
    )

    runtime.warmup(text="预热", max_chunks=1)

    assert calls[0]["init"]["model_dir"] == str(model_dir)
    assert calls[1]["text"] == "预热"
    assert calls[1]["kwargs"]["stream_return"] is True
    assert calls[1]["kwargs"]["max_text_tokens_per_segment"] == 6


def test_indextts_runtime_dispatches_token_window_worker(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        prompt = tmp_path / "prompt.wav"
        prompt.write_bytes(b"prompt")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\n", encoding="utf-8")

        class FakeIndexTTS2:
            pass

        monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=lambda **kwargs: FakeIndexTTS2()))

        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(prompt),
            sample_rate=16000,
            model_sample_rate=16000,
            chunk_ms=1000,
            streaming_mode="token_window",
            device="cpu",
            use_fp16=False,
        )

        def fake_iter(engine, text, **kwargs):
            assert isinstance(engine, FakeIndexTTS2)
            assert text == "你好。"
            assert kwargs["token_window_size"] == 40
            assert kwargs["token_window_hop"] == 96
            assert kwargs["token_window_context"] == 8
            yield np.array([0, 1000, -1000, 0], dtype=np.int16)

        monkeypatch.setattr(runtime, "_iter_token_window_pcm", fake_iter)

        chunks = [chunk async for chunk in runtime.synthesize_pcm_stream("你好。")]

        assert chunks

    asyncio.run(run())



def test_token_window_allows_larger_followup_hop(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        prompt = tmp_path / "prompt.wav"
        prompt.write_bytes(b"prompt")
        model_dir = tmp_path / "IndexTTS-2"
        model_dir.mkdir()
        cfg = model_dir / "config.yaml"
        cfg.write_text("test: true\\n", encoding="utf-8")

        class FakeIndexTTS2:
            pass

        monkeypatch.setitem(sys.modules, "indextts.infer_v2", SimpleNamespace(IndexTTS2=lambda **kwargs: FakeIndexTTS2()))

        runtime = IndexTTSStreamingRuntime(
            model_dir=str(model_dir),
            cfg_path=str(cfg),
            prompt_audio=str(prompt),
            sample_rate=16000,
            model_sample_rate=16000,
            chunk_ms=1000,
            streaming_mode="token_window",
            token_window_size=32,
            token_window_hop=96,
            device="cpu",
            use_fp16=False,
        )

        def fake_iter(engine, text, **kwargs):
            assert kwargs["token_window_size"] == 32
            assert kwargs["token_window_hop"] == 96
            yield np.array([0, 1000, -1000, 0], dtype=np.int16)

        monkeypatch.setattr(runtime, "_iter_token_window_pcm", fake_iter)

        chunks = [chunk async for chunk in runtime.synthesize_pcm_stream("你好。")]

        assert chunks

    asyncio.run(run())



def test_token_window_streams_gpt_codes_with_single_generate_call(monkeypatch, tmp_path) -> None:
    import torch

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")

    class FakeTokenizer:
        def tokenize(self, text):
            return ["你"]

        def split_segments(self, tokens, max_text_tokens_per_segment, quick_streaming_tokens=0):
            return [tokens]

        def convert_tokens_to_ids(self, sent):
            return [1]

    class FakeInferenceModel:
        def __init__(self):
            self.generate_calls = 0
            self.embeddings = None
            self.text_pos_embedding = None

        def store_mel_emb(self, inputs_embeds):
            self.inputs_embeds = inputs_embeds

        def generate(self, inputs, *, streamer=None, **kwargs):
            self.generate_calls += 1
            streamer.put(inputs)
            for token in [2, 3, 4, 5, 6, 7, 8, 99]:
                streamer.put(torch.tensor([[token]], dtype=torch.long, device=inputs.device))
            streamer.end()
            return torch.tensor([[2, 3, 4, 5, 6, 7, 8, 99]], dtype=torch.long, device=inputs.device)

    class FakeGpt:
        start_mel_token = 1
        stop_mel_token = 99
        max_mel_tokens = 20
        accel_engine = None

        def __init__(self):
            self.inference_model = FakeInferenceModel()

        def merge_emovec(self, *args, **kwargs):
            return torch.ones((1, 4))

        def inference_speech(self, *args, **kwargs):
            raise AssertionError("token-window streaming should use one generate call")

        def get_conditioning(self, *args, **kwargs):
            return torch.ones((1, 1, 4))

        def speed_emb(self, tokens):
            return torch.zeros((tokens.shape[0], 4), device=tokens.device)

        def prepare_gpt_inputs(self, conds_latent, text_tokens):
            input_ids = torch.ones((1, 3), dtype=torch.long, device=text_tokens.device)
            inputs_embeds = torch.ones((1, 2, 4), device=text_tokens.device)
            attention_mask = torch.ones((1, 3), dtype=torch.long, device=text_tokens.device)
            return input_ids, inputs_embeds, attention_mask

    class FakeEngine:
        device = torch.device("cpu")
        dtype = None
        stop_mel_token = 99
        tokenizer = FakeTokenizer()
        gpt = FakeGpt()

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        sample_rate=16000,
        model_sample_rate=16000,
        chunk_ms=1000,
        streaming_mode="token_window",
        token_window_size=4,
        token_window_hop=4,
        token_window_context=0,
        device="cpu",
        use_fp16=False,
    )
    monkeypatch.setattr(runtime, "_prepare_prompt_context", lambda engine: (torch.ones((1, 1, 4)), None, None, None))
    monkeypatch.setattr(runtime, "_prepare_emo_context", lambda engine, prompt_audio=None: torch.ones((1, 1, 4)))
    decode_lengths = []

    def fake_decode(engine, *, codes, **kwargs):
        decode_lengths.append(int(codes.shape[-1]))
        return np.ones(int(codes.shape[-1]) * 1000, dtype=np.int16)

    monkeypatch.setattr(runtime, "_decode_codes_to_model_pcm", fake_decode)

    chunks = list(
        runtime._iter_token_window_pcm(
            FakeEngine(),
            "你好。",
            max_text_tokens_per_segment=24,
            quick_streaming_tokens=4,
            interval_silence_ms=0,
            token_window_size=4,
            token_window_hop=4,
            token_window_context=0,
            token_window_overlap_ms=0,
            generation_config={"max_mel_tokens": 20},
        )
    )

    assert chunks
    assert FakeEngine.gpt.inference_model.generate_calls == 1
    assert decode_lengths[:2] == [4, 7]


def test_token_window_generate_runs_inside_autocast(monkeypatch, tmp_path) -> None:
    import contextlib
    import torch

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\\n", encoding="utf-8")
    autocast_active = {"value": False}

    @contextlib.contextmanager
    def fake_autocast(*args, **kwargs):
        old = autocast_active["value"]
        autocast_active["value"] = bool(kwargs.get("enabled", True))
        try:
            yield
        finally:
            autocast_active["value"] = old

    class FakeInferenceModel:
        def __init__(self):
            self.generate_calls = 0
            self.embeddings = None
            self.text_pos_embedding = None

        def store_mel_emb(self, inputs_embeds):
            self.inputs_embeds = inputs_embeds

        def generate(self, inputs, *, streamer=None, **kwargs):
            self.generate_calls += 1
            assert autocast_active["value"] is True
            streamer.put(inputs)
            for token in [2, 99]:
                streamer.put(torch.tensor([[token]], dtype=torch.long, device=inputs.device))
            streamer.end()
            return torch.tensor([[2, 99]], dtype=torch.long, device=inputs.device)

    class FakeGpt:
        start_mel_token = 1
        stop_mel_token = 99
        max_mel_tokens = 20
        accel_engine = None

        def __init__(self):
            self.inference_model = FakeInferenceModel()

        def get_conditioning(self, *args, **kwargs):
            return torch.ones((1, 1, 4))

        def speed_emb(self, tokens):
            return torch.zeros((tokens.shape[0], 4), device=tokens.device)

        def prepare_gpt_inputs(self, conds_latent, text_tokens):
            input_ids = torch.ones((1, 3), dtype=torch.long, device=text_tokens.device)
            inputs_embeds = torch.ones((1, 2, 4), device=text_tokens.device)
            attention_mask = torch.ones((1, 3), dtype=torch.long, device=text_tokens.device)
            return input_ids, inputs_embeds, attention_mask

    class FakeEngine:
        dtype = torch.float16
        stop_mel_token = 99
        gpt = FakeGpt()

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        streaming_mode="token_window",
        device="cpu",
        use_fp16=True,
    )
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)

    batches = list(
        runtime._iter_streamed_speech_code_batches(
            FakeEngine(),
            text_tokens=torch.tensor([[1]], dtype=torch.int32),
            spk_cond_emb=torch.ones((1, 1, 4)),
            emo_cond_emb=torch.ones((1, 1, 4)),
            emovec=torch.ones((1, 4)),
            max_total_tokens=8,
            token_window_size=1,
            token_window_hop=1,
            generation_kwargs={"do_sample": True},
        )
    )

    assert batches
    assert FakeEngine.gpt.inference_model.generate_calls == 1

def test_token_window_decodes_bounded_context(monkeypatch, tmp_path) -> None:
    import torch

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")

    class FakeTokenizer:
        def tokenize(self, text):
            return ["你"]

        def split_segments(self, tokens, max_text_tokens_per_segment, quick_streaming_tokens=0):
            return [tokens]

        def convert_tokens_to_ids(self, sent):
            return [1]

    class FakeGpt:
        def __init__(self):
            self.calls = 0

        def merge_emovec(self, *args, **kwargs):
            return torch.ones((1, 4))

        def inference_speech(self, *args, max_generate_length=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return torch.arange(1, 9, dtype=torch.long).unsqueeze(0), torch.ones((1, 1, 4))
            if self.calls == 2:
                return torch.arange(9, 15, dtype=torch.long).unsqueeze(0), torch.ones((1, 1, 4))
            return torch.tensor([[99]], dtype=torch.long), torch.ones((1, 1, 4))

    class FakeEngine:
        device = torch.device("cpu")
        dtype = None
        stop_mel_token = 99
        tokenizer = FakeTokenizer()
        gpt = FakeGpt()

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        sample_rate=16000,
        model_sample_rate=16000,
        chunk_ms=1000,
        streaming_mode="token_window",
        token_window_size=8,
        token_window_hop=6,
        token_window_context=3,
        device="cpu",
        use_fp16=False,
    )
    monkeypatch.setattr(runtime, "_prepare_prompt_context", lambda engine: (torch.ones((1, 1, 4)), None, None, None))
    monkeypatch.setattr(runtime, "_prepare_emo_context", lambda engine, prompt_audio=None: torch.ones((1, 1, 4)))
    decode_lengths = []

    def fake_decode(engine, *, codes, **kwargs):
        decode_lengths.append(int(codes.shape[-1]))
        return np.ones(int(codes.shape[-1]) * 1000, dtype=np.int16)

    monkeypatch.setattr(runtime, "_decode_codes_to_model_pcm", fake_decode)

    chunks = list(
        runtime._iter_token_window_pcm(
            FakeEngine(),
            "你好。",
            max_text_tokens_per_segment=24,
            quick_streaming_tokens=4,
            interval_silence_ms=0,
            token_window_size=8,
            token_window_hop=6,
            token_window_context=3,
            token_window_overlap_ms=0,
            generation_config={"max_mel_tokens": 20},
        )
    )

    assert chunks
    assert decode_lengths == [8, 9]

def test_token_window_builds_emovec_inside_autocast(monkeypatch, tmp_path) -> None:
    import contextlib
    import torch

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")
    autocast_active = {"value": False}

    @contextlib.contextmanager
    def fake_autocast(*args, **kwargs):
        old = autocast_active["value"]
        autocast_active["value"] = True
        try:
            yield
        finally:
            autocast_active["value"] = old

    class FakeTokenizer:
        def tokenize(self, text):
            return ["你"]

        def split_segments(self, tokens, max_text_tokens_per_segment, quick_streaming_tokens=0):
            return [tokens]

        def convert_tokens_to_ids(self, sent):
            return [1]

    class FakeGpt:
        def merge_emovec(self, *args, **kwargs):
            assert autocast_active["value"] is True
            return torch.ones((1, 4))

        def inference_speech(self, *args, **kwargs):
            return torch.tensor([[2, 99]], dtype=torch.long), torch.ones((1, 1, 4))

    class FakeEngine:
        device = torch.device("cpu")
        dtype = torch.float16
        stop_mel_token = 99
        tokenizer = FakeTokenizer()
        gpt = FakeGpt()

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        sample_rate=16000,
        model_sample_rate=16000,
        chunk_ms=1000,
        streaming_mode="token_window",
        device="cpu",
        use_fp16=True,
    )
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    monkeypatch.setattr(
        runtime,
        "_prepare_prompt_context",
        lambda engine: (
            torch.ones((1, 1, 4)),
            torch.ones((1, 4)),
            torch.ones((1, 1, 4)),
            torch.ones((1, 80, 1)),
        ),
    )
    monkeypatch.setattr(runtime, "_prepare_emo_context", lambda engine, prompt_audio=None: torch.ones((1, 1, 4)))
    monkeypatch.setattr(
        runtime,
        "_decode_codes_to_model_pcm",
        lambda *args, **kwargs: np.array([0, 1000, -1000, 0], dtype=np.int16),
    )

    chunks = list(
        runtime._iter_token_window_pcm(
            FakeEngine(),
            "你好。",
            max_text_tokens_per_segment=24,
            quick_streaming_tokens=4,
            interval_silence_ms=0,
            token_window_size=16,
            token_window_hop=8,
            token_window_context=16,
            token_window_overlap_ms=0,
            generation_config={},
        )
    )

    assert chunks

def test_indextts_runtime_status_reports_segment_streaming_boundary(tmp_path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    model_dir = tmp_path / "IndexTTS-2"
    model_dir.mkdir()
    cfg = model_dir / "config.yaml"
    cfg.write_text("test: true\n", encoding="utf-8")

    runtime = IndexTTSStreamingRuntime(
        model_dir=str(model_dir),
        cfg_path=str(cfg),
        prompt_audio=str(prompt),
        streaming_mode="segment",
    )

    status = runtime.status()

    assert status["streaming"] is True
    assert status["streaming_granularity"] == "segment"
    assert status["model_internal_streaming"] is False
    assert "s2mel" in status["streaming_note"]
    assert "BigVGAN" in status["streaming_note"]
