"""CosyVoice3 wrapper backed by a Triton/TensorRT-LLM service."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import queue
import uuid
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest


@dataclass(frozen=True)
class CosyVoiceTritonConfig:
    server_addr: str
    server_port: int
    model_name: str
    service_accelerator: str
    service_profile: str
    service_runtime: Dict[str, Any]
    sample_rate: int
    output_path: Path
    request_id: str
    seed: Optional[int]
    reference_text: str
    target_text: str
    reference_audio: Path


COSYVOICE_SERVICE_PROFILES: Dict[str, Dict[str, Any]] = {
    "146-triton-trtllm": {
        "server_addr": "8.92.9.146",
        "server_port": 18001,
        "model_name": "cosyvoice3",
        "service_accelerator": "cuda",
        "runtime": {
            "host": "146",
            "gpu": "GPU1",
            "container": "cosyvoice-trt2504",
            "engine": "triton_trtllm",
            "llm_endpoint": "localhost:8000",
            "triton_http_port": 18000,
            "triton_grpc_port": 18001,
            "triton_metrics_port": 18002,
            "model_repo": "model_repo_cosyvoice3_copy",
            "token2wav_instances": 2,
            "vocoder_instances": 2,
            "kv_cache_free_gpu_memory_fraction": 0.2,
            "benchmark": {
                "date": "2026-04-28",
                "rtf": 0.1303,
                "avg_first_chunk_ms": 699.13,
                "avg_total_request_ms": 3029.77,
                "omnirt_denoise_loop_ms": 1969.611,
            },
        },
    },
    "146-local-stream-trt": {
        "server_url": "http://8.92.9.146:50000",
        "service_accelerator": "cuda",
        "runtime": {
            "host": "146",
            "engine": "local_cosyvoice_http",
            "flow_decoder_trt": True,
            "transformers": "4.51.3",
            "torch": "2.5.1+cu124",
            "token_hop_len": 8,
            "token_max_hop_len": 32,
            "stream_scale_factor": 2,
            "flow_n_timesteps": 10,
            "max_token_text_ratio": 6.0,
            "min_token_text_ratio": 2.0,
            "stop_token_mask": "all_stop_token_ids",
            "zero_shot_cache": "voiceops_warmup",
            "benchmark": {
                "date": "2026-06-23",
                "short_first_chunk_ms": 575,
                "medium_first_chunk_ms": 485,
                "short_audio_sec": 3.48,
                "medium_audio_sec": 8.2,
                "short_rtf_total": 0.664,
                "medium_rtf_total": 0.627,
            },
        },
    },
}


@register_model(
    id="cosyvoice3-triton-trtllm",
    task="text2audio",
    default_backend="cuda",
    resource_hint={
        "min_vram_gb": 8,
        "dtype": "fp16/bf16",
        "accelerator": "external CosyVoice service; CUDA/TensorRT-LLM reference or Ascend-hosted compatible service",
    },
    capabilities=ModelCapabilities(
        required_inputs=("prompt", "audio"),
        optional_inputs=("reference_text",),
        supported_config=(
            "server_addr",
            "server_port",
            "model_name",
            "service_accelerator",
            "service_profile",
            "token2wav_instances",
            "vocoder_instances",
            "kv_cache_free_gpu_memory_fraction",
            "triton_http_port",
            "triton_metrics_port",
            "stream_server_url",
            "token_hop_len",
            "token_max_hop_len",
            "stream_scale_factor",
            "flow_n_timesteps",
            "max_token_text_ratio",
            "min_token_text_ratio",
            "stop_token_mask",
            "zero_shot_cache_id",
            "sample_rate",
            "output_dir",
            "request_id",
            "seed",
        ),
        default_config={
            "service_profile": "custom",
            "server_addr": "127.0.0.1",
            "server_port": 8001,
            "model_name": "cosyvoice3",
            "service_accelerator": "cuda",
            "sample_rate": 24000,
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="audio",
        maturity="beta",
        tier="core",
        supports_batching=False,
        streaming=True,
        resident=True,
        service_adapter="text2audio.service.v1",
        backend_status={"cuda": "supported", "ascend": "planned", "cpu-stub": "validation-only"},
        chain_role="voice-generation",
        summary="CosyVoice3 text-to-audio generation through a CUDA Triton/TensorRT-LLM service endpoint.",
        example=(
            "omnirt generate --task text2audio --model cosyvoice3-triton-trtllm "
            "--prompt '你好，欢迎使用 OmniRT。' --audio reference.wav --reference-text '参考音色文本' "
            "--backend cuda --service-profile 146-triton-trtllm --server-addr 8.92.9.146 --server-port 18001 --seed 42"
        ),
    ),
)
class CosyVoiceTritonPipeline(BasePipeline):
    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        reference_audio = Path(str(req.inputs.get("audio", ""))).expanduser()
        if not reference_audio.exists():
            raise FileNotFoundError(reference_audio)
        target_text = str(req.inputs.get("prompt") or "")
        if not target_text:
            raise ValueError("CosyVoice target text is required as input 'prompt'.")
        return {
            "reference_audio": reference_audio,
            "reference_text": str(req.inputs.get("reference_text") or ""),
            "target_text": target_text,
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Dict[str, Any]) -> CosyVoiceTritonConfig:
        output_dir = self.resolve_output_dir(req)
        request_id = str(req.config.get("request_id") or uuid.uuid4())
        output_path = output_dir / f"{req.model}-{request_id}.wav"
        seed_value = req.config.get("seed")
        seed = int(seed_value) if seed_value is not None else None
        profile_name = str(req.config.get("service_profile") or "custom")
        service_profile = profile_name if profile_name in COSYVOICE_SERVICE_PROFILES else "custom"
        profile = COSYVOICE_SERVICE_PROFILES.get(service_profile, {})
        service_runtime = self._service_runtime(req.config, service_profile)
        return CosyVoiceTritonConfig(
            server_addr=str(req.config.get("server_addr") or profile.get("server_addr") or "127.0.0.1"),
            server_port=int(req.config.get("server_port") or profile.get("server_port") or 8001),
            model_name=str(req.config.get("model_name") or profile.get("model_name") or "cosyvoice3"),
            service_accelerator=str(req.config.get("service_accelerator") or profile.get("service_accelerator") or self._default_service_accelerator()),
            service_profile=service_profile,
            service_runtime=service_runtime,
            sample_rate=int(req.config.get("sample_rate", 24000)),
            output_path=output_path,
            request_id=request_id,
            seed=seed,
            reference_text=str(conditions["reference_text"]),
            target_text=str(conditions["target_text"]),
            reference_audio=conditions["reference_audio"],
        )

    def denoise_loop(self, latents: CosyVoiceTritonConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        del conditions, config
        grpcclient = self._grpc_module()
        waveform, sample_rate = self._read_audio(latents.reference_audio)
        waveform = self._to_mono_float32(waveform)
        if int(sample_rate) != 16000:
            waveform = self._resample(waveform, int(sample_rate), 16000)
        waveform = waveform.reshape(1, -1)
        lengths = self._numpy().array([[waveform.shape[1]]], dtype=self._numpy().int32)

        inputs = [
            grpcclient.InferInput("reference_wav", waveform.shape, self._np_to_triton_dtype(waveform.dtype)),
            grpcclient.InferInput("reference_wav_len", lengths.shape, self._np_to_triton_dtype(lengths.dtype)),
            grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
            grpcclient.InferInput("target_text", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(waveform)
        inputs[1].set_data_from_numpy(lengths)
        inputs[2].set_data_from_numpy(self._string_tensor(latents.reference_text))
        inputs[3].set_data_from_numpy(self._string_tensor(latents.target_text))
        outputs = [grpcclient.InferRequestedOutput("waveform")]

        completed_requests = queue.Queue()

        def _callback(result: Any, error: Any) -> None:
            completed_requests.put(error if error is not None else result)

        client = grpcclient.InferenceServerClient(url=f"{latents.server_addr}:{latents.server_port}", verbose=False)
        chunks = []
        try:
            parameters = {"seed": latents.seed} if latents.seed is not None else None
            client.start_stream(callback=_callback)
            client.async_stream_infer(
                latents.model_name,
                inputs,
                request_id=latents.request_id,
                outputs=outputs,
                enable_empty_final_response=True,
                parameters=parameters,
            )
            while True:
                item = completed_requests.get(timeout=200)
                if isinstance(item, Exception):
                    raise item
                response = item.get_response()
                final = response.parameters["triton_final_response"].bool_param
                if final:
                    break
                chunk = item.as_numpy("waveform").reshape(-1)
                if chunk.size:
                    chunks.append(chunk)
        finally:
            stop_stream = getattr(client, "stop_stream", None)
            if callable(stop_stream):
                stop_stream()
            close = getattr(client, "close", None)
            if callable(close):
                close()

        if not chunks:
            raise RuntimeError("CosyVoice Triton stream completed without waveform chunks.")
        audio = self._numpy().concatenate(chunks)
        return {"audio": audio, "latents": latents}

    def decode(self, latents: Dict[str, Any]) -> Dict[str, Any]:
        return latents

    def export(self, raw: Dict[str, Any], req: GenerateRequest) -> List[Artifact]:
        latents = raw["latents"]
        self._write_audio(latents.output_path, raw["audio"], latents.sample_rate, "PCM_16")
        return [
            Artifact(
                kind="audio",
                path=str(latents.output_path),
                mime="audio/wav",
                width=0,
                height=0,
            )
        ]

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: CosyVoiceTritonConfig) -> Dict[str, Any]:
        del conditions
        return {
            "server_addr": latents.server_addr,
            "server_port": latents.server_port,
            "model_name": latents.model_name,
            "service_accelerator": latents.service_accelerator,
            "service_profile": latents.service_profile,
            "service_runtime": latents.service_runtime,
            "sample_rate": latents.sample_rate,
            "output_dir": str(latents.output_path.parent),
            "request_id": latents.request_id,
            "seed": latents.seed,
        }

    @staticmethod
    def _numpy():
        try:
            import numpy as np
        except ImportError as exc:
            raise DependencyUnavailableError("numpy is required for CosyVoice Triton requests.") from exc
        return np

    def _default_service_accelerator(self) -> str:
        backend_name = str(getattr(self.runtime, "name", "") or "").strip().lower()
        return "ascend" if backend_name == "ascend" else "cuda"

    @classmethod
    def _service_runtime(cls, config: Dict[str, Any], service_profile: str) -> Dict[str, Any]:
        runtime = deepcopy(COSYVOICE_SERVICE_PROFILES.get(service_profile, {}).get("runtime", {}))
        overrides = {
            "token2wav_instances": config.get("token2wav_instances"),
            "vocoder_instances": config.get("vocoder_instances"),
            "kv_cache_free_gpu_memory_fraction": config.get("kv_cache_free_gpu_memory_fraction"),
            "triton_http_port": config.get("triton_http_port"),
            "triton_metrics_port": config.get("triton_metrics_port"),
            "stream_server_url": config.get("stream_server_url"),
            "token_hop_len": config.get("token_hop_len"),
            "token_max_hop_len": config.get("token_max_hop_len"),
            "stream_scale_factor": config.get("stream_scale_factor"),
            "flow_n_timesteps": config.get("flow_n_timesteps"),
            "max_token_text_ratio": config.get("max_token_text_ratio"),
            "min_token_text_ratio": config.get("min_token_text_ratio"),
            "stop_token_mask": config.get("stop_token_mask"),
            "zero_shot_cache_id": config.get("zero_shot_cache_id"),
        }
        for key, value in overrides.items():
            if value is not None:
                runtime[key] = value
        return runtime

    @staticmethod
    def _grpc_module():
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise DependencyUnavailableError("tritonclient[grpc] is required for CosyVoice Triton requests.") from exc
        return grpcclient

    @staticmethod
    def _np_to_triton_dtype(dtype: Any) -> str:
        try:
            from tritonclient.utils import np_to_triton_dtype
        except ImportError as exc:
            raise DependencyUnavailableError("tritonclient is required for CosyVoice Triton requests.") from exc
        return np_to_triton_dtype(dtype)

    @staticmethod
    def _read_audio(path: Path):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise DependencyUnavailableError("soundfile is required to read CosyVoice reference audio.") from exc
        return sf.read(str(path))

    @staticmethod
    def _write_audio(path: Path, data: Any, sample_rate: int, subtype: str) -> None:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise DependencyUnavailableError("soundfile is required to write CosyVoice audio artifacts.") from exc
        sf.write(str(path), data, sample_rate, subtype)

    @classmethod
    def _to_mono_float32(cls, waveform: Any):
        np = cls._numpy()
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1:
            raise ValueError("CosyVoice reference audio must be a mono or stereo waveform.")
        return audio.astype(np.float32, copy=False)

    @classmethod
    def _resample(cls, waveform: Any, source_rate: int, target_rate: int):
        if source_rate == target_rate:
            return waveform
        try:
            from scipy.signal import resample
        except ImportError as exc:
            raise DependencyUnavailableError("scipy is required to resample CosyVoice reference audio to 16 kHz.") from exc
        samples = int(len(waveform) * (float(target_rate) / float(source_rate)))
        return resample(waveform, samples).astype(cls._numpy().float32, copy=False)

    @classmethod
    def _string_tensor(cls, value: str):
        return cls._numpy().array([value], dtype=object).reshape((1, 1))
