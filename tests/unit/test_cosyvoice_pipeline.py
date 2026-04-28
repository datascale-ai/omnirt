from pathlib import Path

import numpy as np

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.cosyvoice.pipeline import CosyVoiceTritonPipeline


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
        id="cosyvoice3-triton-trtllm",
        task="text2audio",
        pipeline_cls=CosyVoiceTritonPipeline,
        default_backend="cuda",
        resource_hint={"min_vram_gb": 8, "dtype": "fp16"},
    )


def test_cosyvoice_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("cosyvoice3-triton-trtllm", task="text2audio")

    assert spec.task == "text2audio"
    assert spec.default_backend == "cuda"
    assert spec.capabilities.artifact_kind == "audio"
    assert "seed" in spec.capabilities.supported_config


def test_cosyvoice_pipeline_sends_offline_triton_request(tmp_path, monkeypatch) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"fake")
    captured = {}

    class FakeInferInput:
        def __init__(self, name, shape, dtype):
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.data = None

        def set_data_from_numpy(self, value):
            self.data = value

    class FakeRequestedOutput:
        def __init__(self, name):
            self.name = name

    class FakeTritonFinal:
        def __init__(self, value):
            self.bool_param = value

    class FakeTritonResponse:
        def __init__(self, final):
            self.parameters = {"triton_final_response": FakeTritonFinal(final)}

    class FakeResult:
        def __init__(self, audio=None, final=False):
            self.audio = audio
            self.final = final

        def get_response(self):
            return FakeTritonResponse(self.final)

        def as_numpy(self, name):
            assert name == "waveform"
            return self.audio

    class FakeClient:
        def __init__(self, url, verbose=False):
            captured["url"] = url
            captured["verbose"] = verbose
            self.callback = None

        def start_stream(self, callback, **kwargs):
            captured["stream_kwargs"] = kwargs
            self.callback = callback

        def async_stream_infer(
            self,
            model_name,
            inputs,
            request_id,
            outputs,
            enable_empty_final_response=False,
            parameters=None,
        ):
            captured["model_name"] = model_name
            captured["inputs"] = inputs
            captured["request_id"] = request_id
            captured["outputs"] = outputs
            captured["enable_empty_final_response"] = enable_empty_final_response
            captured["parameters"] = dict(parameters or {})
            self.callback(FakeResult(np.array([0.0, 0.25], dtype=np.float32)), None)
            self.callback(FakeResult(np.array([-0.25], dtype=np.float32)), None)
            self.callback(FakeResult(final=True), None)

        def stop_stream(self):
            captured["stopped"] = True

        def close(self):
            captured["closed"] = True

    class FakeGrpcModule:
        InferInput = FakeInferInput
        InferRequestedOutput = FakeRequestedOutput
        InferenceServerClient = FakeClient

    written = {}

    def fake_read(path):
        assert Path(path) == reference_audio
        return np.array([0.1, 0.2, 0.3], dtype=np.float32), 16000

    def fake_write(path, data, samplerate, subtype):
        written["path"] = Path(path)
        written["data"] = data
        written["samplerate"] = samplerate
        written["subtype"] = subtype
        Path(path).write_bytes(b"wav")

    monkeypatch.setattr(CosyVoiceTritonPipeline, "_grpc_module", staticmethod(lambda: FakeGrpcModule))
    monkeypatch.setattr(CosyVoiceTritonPipeline, "_np_to_triton_dtype", staticmethod(lambda dtype: f"TRITON_{dtype}"))
    monkeypatch.setattr(CosyVoiceTritonPipeline, "_read_audio", staticmethod(fake_read))
    monkeypatch.setattr(CosyVoiceTritonPipeline, "_write_audio", staticmethod(fake_write))

    request = GenerateRequest(
        task="text2audio",
        model="cosyvoice3-triton-trtllm",
        backend="cuda",
        inputs={
            "prompt": "生成一段确定性测试语音。",
            "audio": str(reference_audio),
            "reference_text": "参考音色。",
        },
        config={
            "server_addr": "8.92.9.146",
            "server_port": 8001,
            "output_dir": str(tmp_path / "outputs"),
            "seed": 1234,
            "request_id": "fixed-request",
        },
    )

    pipeline = CosyVoiceTritonPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert captured["url"] == "8.92.9.146:8001"
    assert captured["model_name"] == "cosyvoice3"
    assert captured["request_id"] == "fixed-request"
    assert captured["enable_empty_final_response"] is True
    assert captured["parameters"] == {"seed": 1234}
    assert [item.name for item in captured["inputs"]] == [
        "reference_wav",
        "reference_wav_len",
        "reference_text",
        "target_text",
    ]
    assert captured["inputs"][2].data.tolist() == [["参考音色。"]]
    assert captured["inputs"][3].data.tolist() == [["生成一段确定性测试语音。"]]
    assert written["samplerate"] == 24000
    assert written["subtype"] == "PCM_16"
    assert written["data"].tolist() == [0.0, 0.25, -0.25]
    assert captured["stopped"] is True
    assert captured["closed"] is True
    assert Path(result.outputs[0].path).exists()
    assert result.outputs[0].kind == "audio"
    assert result.outputs[0].mime == "audio/wav"
    assert result.metadata.config_resolved["server_addr"] == "8.92.9.146"
