"""Resident IndexTTS2 streaming runtime for OmniRT text2audio serving."""

from __future__ import annotations

import asyncio
import importlib
import os
import logging
import re
import time
from pathlib import Path
import queue
import random
from inspect import Parameter, signature
import threading
from typing import Any, AsyncIterator

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("uvicorn.error")

DEFAULT_SEGMENT_MAX_TEXT_TOKENS_PER_SEGMENT = 24
DEFAULT_TOKEN_WINDOW_MAX_TEXT_TOKENS_PER_SEGMENT = 80
DEFAULT_QUICK_STREAMING_TOKENS = 4
DEFAULT_DO_SAMPLE = True
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 30
DEFAULT_TEMPERATURE = 0.8
DEFAULT_NUM_BEAMS = 3
DEFAULT_REPETITION_PENALTY = 10.0
DEFAULT_MAX_MEL_TOKENS = 1500
DEFAULT_STREAMING_MODE = "token_window"
DEFAULT_TOKEN_WINDOW_SIZE = 40
DEFAULT_TOKEN_WINDOW_HOP = 96
DEFAULT_TOKEN_WINDOW_CONTEXT = 8
DEFAULT_TOKEN_WINDOW_OVERLAP_MS = 60
STREAMING_NOTE = (
    "IndexTTS2 stream_return yields complete text/audio segments after s2mel and BigVGAN finish; "
    "it is not model-internal 20ms streaming."
)
TOKEN_WINDOW_STREAMING_NOTE = (
    "experimental token_window streaming emits audio after speech-token windows before the full text segment finishes; "
    "it still re-runs s2mel/BigVGAN per window and is not guaranteed to be artifact-free."
)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _normalize_streaming_mode(value: str | None) -> str:
    mode = (value or DEFAULT_STREAMING_MODE).strip().lower().replace("-", "_")
    if mode in {"token", "token_window", "true", "model_internal"}:
        return "token_window"
    return "segment"


def _tensor_to_i16_pcm(audio: Any) -> np.ndarray:
    if isinstance(audio, (str, bytes, list, tuple, dict)):
        return np.zeros(0, dtype=np.int16)
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int16)
    arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, -32768, 32767).astype(np.int16)
    arr = arr.astype(np.float32, copy=False)
    peak = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if peak <= 1.5:
        arr = arr * 32767.0
    return np.clip(np.round(arr), -32768, 32767).astype(np.int16)


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if pcm.size == 0 or int(src_sr) == int(dst_sr):
        return pcm.copy()
    pcm_f = pcm.astype(np.float32) / 32768.0
    n_dst = max(1, int(round(pcm.size * int(dst_sr) / int(src_sr))))
    xi = np.linspace(0.0, pcm.size - 1.0, num=n_dst)
    out = np.interp(xi, np.arange(pcm.size), pcm_f)
    return np.clip(np.round(out * 32768.0), -32768, 32767).astype(np.int16)


def _speech_tokens_to_model_samples(token_count: int) -> int:
    return int(round(max(0, int(token_count)) * 1.72 * 256))


def _split_pcm_bytes(pcm: np.ndarray, sample_rate: int, chunk_ms: float) -> list[bytes]:
    samples_per_chunk = max(1, int(float(sample_rate) * (float(chunk_ms) / 1000.0)))
    out: list[bytes] = []
    for i in range(0, int(pcm.size), samples_per_chunk):
        part = pcm[i : i + samples_per_chunk]
        if part.size:
            out.append(part.astype("<i2", copy=False).tobytes())
    return out

def _indextts_emotion_kwargs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    out: dict[str, Any] = {}
    for key in ("emo_alpha", "emo_audio_prompt", "emo_vector", "use_emo_text", "emo_text", "use_random"):
        if config.get(key) is not None:
            out[key] = config[key]
    return out


def _clamp_unit_float(value: Any, default: float = 1.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, out))


def _find_most_similar_cosine(query_vector: Any, matrix: Any) -> int:
    similarities = F.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
    return int(torch.argmax(similarities).item())




class _TokenQueueStreamer:
    def __init__(self, *, prompt_length: int, max_queue: int = 0) -> None:
        self._prompt_remaining = max(0, int(prompt_length))
        self._queue: queue.Queue[int | BaseException | None] = queue.Queue(maxsize=max(0, int(max_queue)))

    def put(self, value: Any) -> None:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "reshape"):
            tokens = [int(item) for item in value.reshape(-1).tolist()]
        elif isinstance(value, (list, tuple)):
            tokens = [int(item) for item in value]
        else:
            tokens = [int(value)]
        if self._prompt_remaining:
            skip = min(self._prompt_remaining, len(tokens))
            tokens = tokens[skip:]
            self._prompt_remaining -= skip
        for token in tokens:
            self._queue.put(token)

    def put_error(self, exc: BaseException) -> None:
        self._queue.put(exc)

    def end(self) -> None:
        self._queue.put(None)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item


class IndexTTSStreamingRuntime:
    """Long-lived IndexTTS2 runtime that streams PCM as each model segment completes."""

    def __init__(
        self,
        *,
        model: str = "IndexTeam/IndexTTS-2",
        model_dir: str = "",
        cfg_path: str = "",
        prompt_audio: str = "",
        voices_dir: str = "",
        sample_rate: int = 16000,
        model_sample_rate: int = 22050,
        chunk_ms: float = 20.0,
        max_text_tokens_per_segment: int | None = None,
        quick_streaming_tokens: int = DEFAULT_QUICK_STREAMING_TOKENS,
        interval_silence_ms: int = 0,
        device: str = "auto",
        use_fp16: bool | None = None,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False,
        w2v_bert_dir: str = "",
        maskgct_dir: str = "",
        campplus_dir: str = "",
        bigvgan_dir: str = "",
        do_sample: bool = DEFAULT_DO_SAMPLE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
        num_beams: int = DEFAULT_NUM_BEAMS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_mel_tokens: int = DEFAULT_MAX_MEL_TOKENS,
        streaming_mode: str = DEFAULT_STREAMING_MODE,
        token_window_size: int = DEFAULT_TOKEN_WINDOW_SIZE,
        token_window_hop: int = DEFAULT_TOKEN_WINDOW_HOP,
        token_window_context: int = DEFAULT_TOKEN_WINDOW_CONTEXT,
        token_window_overlap_ms: int = DEFAULT_TOKEN_WINDOW_OVERLAP_MS,
    ) -> None:
        self.model = model
        self.model_dir = str(Path(model_dir).expanduser()) if model_dir else ""
        self.cfg_path = str(Path(cfg_path).expanduser()) if cfg_path else ""
        self.prompt_audio = str(Path(prompt_audio).expanduser()) if prompt_audio else ""
        self.voices_dir = str(Path(voices_dir).expanduser()) if voices_dir else ""
        self.sample_rate = int(sample_rate)
        self.model_sample_rate = int(model_sample_rate)
        self.chunk_ms = float(chunk_ms)
        self.streaming_mode = _normalize_streaming_mode(streaming_mode)
        if max_text_tokens_per_segment is None:
            max_text_tokens_per_segment = (
                DEFAULT_TOKEN_WINDOW_MAX_TEXT_TOKENS_PER_SEGMENT
                if self.streaming_mode == "token_window"
                else DEFAULT_SEGMENT_MAX_TEXT_TOKENS_PER_SEGMENT
            )
        self.max_text_tokens_per_segment = max(1, int(max_text_tokens_per_segment))
        self.quick_streaming_tokens = max(0, int(quick_streaming_tokens))
        self.interval_silence_ms = max(0, int(interval_silence_ms))
        self.device = device or "auto"
        self.use_fp16 = bool(self.device.startswith("cuda")) if use_fp16 is None else bool(use_fp16)
        self.use_cuda_kernel = bool(use_cuda_kernel)
        self.use_deepspeed = bool(use_deepspeed)
        self.w2v_bert_dir = str(Path(w2v_bert_dir).expanduser()) if w2v_bert_dir else ""
        self.maskgct_dir = str(Path(maskgct_dir).expanduser()) if maskgct_dir else ""
        self.campplus_dir = str(Path(campplus_dir).expanduser()) if campplus_dir else ""
        self.bigvgan_dir = str(Path(bigvgan_dir).expanduser()) if bigvgan_dir else ""
        self.do_sample = bool(do_sample)
        self.top_p = float(top_p)
        self.top_k = max(0, int(top_k))
        self.temperature = float(temperature)
        self.num_beams = 1 if self.streaming_mode == "token_window" else max(1, int(num_beams))
        self.repetition_penalty = float(repetition_penalty)
        self.max_mel_tokens = max(1, int(max_mel_tokens))
        self.token_window_size = max(4, int(token_window_size))
        self.token_window_hop = max(1, int(token_window_hop))
        self.token_window_context = max(0, int(token_window_context))
        self.token_window_overlap_ms = max(0, int(token_window_overlap_ms))
        self._engine: Any | None = None
        self._engine_lock = threading.Lock()

    def status(self) -> dict[str, object]:
        model_dir = Path(self.model_dir) if self.model_dir else Path("")
        cfg_path = Path(self.cfg_path) if self.cfg_path else Path("")
        prompt_audio = Path(self.prompt_audio) if self.prompt_audio else Path("")
        voices_dir = Path(self.voices_dir) if self.voices_dir else Path("")
        model_dir_exists = bool(self.model_dir) and model_dir.is_dir()
        cfg_path_exists = bool(self.cfg_path) and cfg_path.is_file()
        prompt_audio_exists = bool(self.prompt_audio) and prompt_audio.is_file()
        voices_dir_exists = bool(self.voices_dir) and voices_dir.is_dir()
        token_window = self.streaming_mode == "token_window"
        return {
            "id": "indextts",
            "model": self.model,
            "ready": model_dir_exists and cfg_path_exists and prompt_audio_exists,
            "model_dir": self.model_dir,
            "model_dir_exists": model_dir_exists,
            "cfg_path": self.cfg_path,
            "cfg_path_exists": cfg_path_exists,
            "prompt_audio": self.prompt_audio,
            "prompt_audio_exists": prompt_audio_exists,
            "voices_dir": self.voices_dir,
            "voices_dir_exists": voices_dir_exists,
            "sample_rate": self.sample_rate,
            "model_sample_rate": self.model_sample_rate,
            "chunk_ms": self.chunk_ms,
            "streaming": True,
            "streaming_mode": self.streaming_mode,
            "streaming_granularity": "token_window" if token_window else "segment",
            "model_internal_streaming": token_window,
            "streaming_experimental": token_window,
            "streaming_note": TOKEN_WINDOW_STREAMING_NOTE if token_window else STREAMING_NOTE,
            "max_text_tokens_per_segment": self.max_text_tokens_per_segment,
            "quick_streaming_tokens": self.quick_streaming_tokens,
            "interval_silence_ms": self.interval_silence_ms,
            "device": self.device,
            "w2v_bert_dir": self.w2v_bert_dir,
            "maskgct_dir": self.maskgct_dir,
            "campplus_dir": self.campplus_dir,
            "bigvgan_dir": self.bigvgan_dir,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "max_mel_tokens": self.max_mel_tokens,
            "token_window_size": self.token_window_size,
            "token_window_hop": self.token_window_hop,
            "token_window_context": self.token_window_context,
            "token_window_overlap_ms": self.token_window_overlap_ms,
        }

    def _generation_kwargs(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = dict(config or {})
        return {
            "do_sample": bool(config.get("do_sample") if config.get("do_sample") is not None else self.do_sample),
            "top_p": float(config.get("top_p") if config.get("top_p") is not None else self.top_p),
            "top_k": max(0, int(config.get("top_k") if config.get("top_k") is not None else self.top_k)),
            "temperature": float(config.get("temperature") if config.get("temperature") is not None else self.temperature),
            "num_beams": max(1, int(config.get("num_beams") if config.get("num_beams") is not None else self.num_beams)),
            "repetition_penalty": float(
                config.get("repetition_penalty")
                if config.get("repetition_penalty") is not None
                else self.repetition_penalty
            ),
            "max_mel_tokens": max(1, int(config.get("max_mel_tokens") if config.get("max_mel_tokens") is not None else self.max_mel_tokens)),
        }

    def _config_streaming_mode(self, config: dict[str, Any] | None = None) -> str:
        config = dict(config or {})
        return _normalize_streaming_mode(config.get("streaming_mode") or self.streaming_mode)

    def _resolve_voice_prompt(self, voice: str | None) -> str:
        voice_id = (voice or "").strip()
        if voice_id and re.fullmatch(r"[A-Za-z0-9_-]{3,80}", voice_id) and self.voices_dir:
            voices_root = Path(self.voices_dir)
            for source in ("clones", "system"):
                prompt = voices_root / source / voice_id / "prompt.wav"
                if prompt.is_file():
                    return str(prompt)
        return self.prompt_audio

    def _validate_inputs(self, prompt_audio: str | None = None) -> None:
        status = self.status()
        if not status["model_dir_exists"]:
            raise RuntimeError(f"IndexTTS model directory does not exist: {self.model_dir}")
        if not status["cfg_path_exists"]:
            raise RuntimeError(f"IndexTTS config does not exist: {self.cfg_path}")
        effective_prompt = prompt_audio if prompt_audio is not None else self.prompt_audio
        if not effective_prompt or not Path(effective_prompt).is_file():
            raise RuntimeError(f"IndexTTS prompt audio does not exist: {effective_prompt}")

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            module = importlib.import_module("indextts.infer_v2")
            cls = getattr(module, "IndexTTS2", None)
            if cls is None:
                raise RuntimeError("indextts.infer_v2 does not expose IndexTTS2")
            self._patch_local_runtime_assets(module)
            kwargs = {
                "cfg_path": self.cfg_path,
                "model_dir": self.model_dir,
                "use_fp16": self.use_fp16,
                "use_cuda_kernel": self.use_cuda_kernel,
                "use_deepspeed": self.use_deepspeed,
            }
            if self.device != "auto":
                kwargs["device"] = self.device
            self._engine = cls(**kwargs)
            return self._engine

    def _patch_local_runtime_assets(self, module: Any) -> None:
        self._patch_w2v_bert_runtime(module)
        self._patch_hf_hub_download(module)
        self._patch_bigvgan_runtime()

    def _patch_w2v_bert_runtime(self, module: Any) -> None:
        local_dir = self._existing_dir(self.w2v_bert_dir, "preprocessor_config.json")
        if local_dir is None:
            return
        self._patch_from_pretrained(module, "SeamlessM4TFeatureExtractor", local_dir)
        try:
            maskgct_utils = importlib.import_module("indextts.utils.maskgct_utils")
        except ImportError:
            return
        self._patch_from_pretrained(maskgct_utils, "Wav2Vec2BertModel", local_dir)

    def _patch_hf_hub_download(self, module: Any) -> None:
        original = getattr(module, "hf_hub_download", None)
        if not callable(original):
            return
        asset_map = self._local_hub_asset_map()
        if not asset_map:
            return

        def hf_hub_download(repo_id: str, filename: str, *args: Any, **kwargs: Any) -> str:
            path = asset_map.get((repo_id, filename))
            if path is not None:
                return str(path)
            return original(repo_id, filename, *args, **kwargs)

        module.hf_hub_download = hf_hub_download

    def _local_hub_asset_map(self) -> dict[tuple[str, str], Path]:
        out: dict[tuple[str, str], Path] = {}
        maskgct = self._existing_file(self.maskgct_dir, "semantic_codec/model.safetensors")
        if maskgct is not None:
            out[("amphion/MaskGCT", "semantic_codec/model.safetensors")] = maskgct
        campplus = self._existing_file(self.campplus_dir, "campplus_cn_common.bin")
        if campplus is not None:
            out[("funasr/campplus", "campplus_cn_common.bin")] = campplus
        return out

    def _patch_bigvgan_runtime(self) -> None:
        local_dir = self._existing_dir(self.bigvgan_dir, "config.json")
        if local_dir is None or not (local_dir / "bigvgan_generator.pt").is_file():
            return
        for module_name in (
            "indextts.s2mel.modules.bigvgan.bigvgan",
            "indextts.BigVGAN.bigvgan",
        ):
            try:
                bigvgan = importlib.import_module(module_name)
            except ImportError:
                continue
            self._patch_bigvgan_from_pretrained(bigvgan, local_dir)

    @staticmethod
    def _patch_bigvgan_from_pretrained(module: Any, local_dir: Path) -> None:
        cls = getattr(module, "BigVGAN", None)
        original = getattr(cls, "from_pretrained", None)
        if cls is None or not callable(original):
            return

        def from_pretrained(value: str, *args: Any, **kwargs: Any) -> Any:
            if value == "nvidia/bigvgan_v2_22khz_80band_256x":
                return original(str(local_dir), *args, **kwargs)
            return original(value, *args, **kwargs)

        cls.from_pretrained = from_pretrained

    @staticmethod
    def _existing_dir(value: str, required_file: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.is_dir() and (path / required_file).is_file():
            return path
        return None

    @staticmethod
    def _existing_file(value: str, relative_file: str) -> Path | None:
        if not value:
            return None
        path = Path(value) / relative_file
        if path.is_file():
            return path
        return None

    @staticmethod
    def _patch_from_pretrained(module: Any, attr_name: str, local_dir: Path) -> None:
        target = getattr(module, attr_name, None)
        original = getattr(target, "from_pretrained", None)
        if target is None or not callable(original):
            return

        def from_pretrained(value: str, *args: Any, **kwargs: Any) -> Any:
            if value == "facebook/w2v-bert-2.0":
                return original(str(local_dir), *args, **kwargs)
            return original(value, *args, **kwargs)

        target.from_pretrained = from_pretrained

    def warmup(self, *, text: str = "", max_chunks: int = 1) -> None:
        """Load the resident engine and optionally prime prompt/text inference."""
        self._validate_inputs(self.prompt_audio)
        engine = self._load_engine()
        warm_text = text.strip()
        if not warm_text:
            return
        infer = getattr(engine, "infer", None)
        if not callable(infer):
            raise RuntimeError("IndexTTS2 runtime does not expose infer().")
        infer_params = signature(infer).parameters
        kwargs: dict[str, Any] = {
            "stream_return": True,
            "max_text_tokens_per_segment": self.max_text_tokens_per_segment,
            "interval_silence": self.interval_silence_ms,
            **self._generation_kwargs(),
        }
        accepts_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in infer_params.values())
        if "more_segment_before" in infer_params:
            kwargs["more_segment_before"] = self.quick_streaming_tokens
        elif "quick_streaming_tokens" in infer_params or accepts_kwargs:
            kwargs["quick_streaming_tokens"] = self.quick_streaming_tokens
        iterator = infer(str(Path(self.prompt_audio)), warm_text, None, **kwargs)
        for idx, _ in enumerate(iterator):
            if idx + 1 >= max(1, int(max_chunks)):
                break

    async def synthesize_pcm_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        config = dict(config or {})
        prompt_audio = str(config.get("prompt_audio") or "").strip() or self._resolve_voice_prompt(voice)
        self._validate_inputs(prompt_audio)
        out_q: queue.Queue[bytes | BaseException | None] = queue.Queue()
        self._start_worker(text, config, out_q, prompt_audio)
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, out_q.get)
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item


    def _start_token_window_worker(
        self,
        text: str,
        config: dict[str, Any],
        out_q: "queue.Queue[bytes | BaseException | None]",
        prompt_audio: str,
    ) -> None:
        self._validate_inputs(prompt_audio)
        engine = self._load_engine()
        max_tokens = int(config.get("max_text_tokens_per_segment") or self.max_text_tokens_per_segment)
        quick_tokens = int(config.get("quick_streaming_tokens") or self.quick_streaming_tokens)
        silence_ms = int(config.get("interval_silence_ms") if config.get("interval_silence_ms") is not None else self.interval_silence_ms)
        window_size = max(4, int(config.get("token_window_size") or self.token_window_size))
        window_hop = max(1, int(config.get("token_window_hop") or self.token_window_hop))
        context_tokens = max(0, int(config.get("token_window_context") if config.get("token_window_context") is not None else self.token_window_context))
        overlap_ms = max(0, int(config.get("token_window_overlap_ms") if config.get("token_window_overlap_ms") is not None else self.token_window_overlap_ms))
        gen_config = dict(config)
        gen_config["num_beams"] = 1

        def worker() -> None:
            try:
                started = time.perf_counter()
                first_chunk_ms: float | None = None
                emitted_chunks = 0
                emitted_bytes = 0
                emitted_windows = 0
                for pcm in self._iter_token_window_pcm(
                    engine,
                    text,
                    prompt_audio=prompt_audio,
                    max_text_tokens_per_segment=max(1, max_tokens),
                    quick_streaming_tokens=max(0, quick_tokens),
                    interval_silence_ms=max(0, silence_ms),
                    token_window_size=window_size,
                    token_window_hop=window_hop,
                    token_window_context=context_tokens,
                    token_window_overlap_ms=overlap_ms,
                    generation_config=gen_config,
                ):
                    if pcm.size == 0:
                        continue
                    emitted_windows += 1
                    pcm = _resample_linear(pcm, self.model_sample_rate, self.sample_rate)
                    for chunk in _split_pcm_bytes(pcm, self.sample_rate, self.chunk_ms):
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started) * 1000.0
                        emitted_chunks += 1
                        emitted_bytes += len(chunk)
                        out_q.put(chunk)
                total_ms = (time.perf_counter() - started) * 1000.0
                log.info(
                    "IndexTTS token-window stream finished | first_chunk_ms=%s total_ms=%.1f windows=%d chunks=%d bytes=%d window=%d hop=%d overlap_ms=%d",
                    "none" if first_chunk_ms is None else f"{first_chunk_ms:.1f}",
                    total_ms,
                    emitted_windows,
                    emitted_chunks,
                    emitted_bytes,
                    window_size,
                    window_hop,
                    overlap_ms,
                )
            except BaseException as exc:  # noqa: BLE001
                out_q.put(exc)
            finally:
                out_q.put(None)

        threading.Thread(target=worker, name="omnirt-indextts-token-window", daemon=True).start()

    def _iter_token_window_pcm(
        self,
        engine: Any,
        text: str,
        *,
        prompt_audio: str | None = None,
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int,
        interval_silence_ms: int,
        token_window_size: int,
        token_window_hop: int,
        token_window_context: int,
        token_window_overlap_ms: int,
        generation_config: dict[str, Any],
    ):
        import torch

        effective_prompt_audio = prompt_audio or self.prompt_audio
        emo_prompt_audio = self._emotion_audio_prompt_from_config(generation_config, effective_prompt_audio)
        if prompt_audio is None:
            spk_cond_emb, style, prompt_condition, ref_mel = self._prepare_prompt_context(engine)
            emo_cond_emb = self._prepare_emo_context(engine, emo_prompt_audio)
        else:
            spk_cond_emb, style, prompt_condition, ref_mel = self._prepare_prompt_context(engine, effective_prompt_audio)
            emo_cond_emb = self._prepare_emo_context(engine, emo_prompt_audio)
        emotion_vector = self._emotion_vector_from_config(engine, text, generation_config)
        emotion_alpha = _clamp_unit_float(generation_config.get("emo_alpha"), 1.0) if emotion_vector is None else 1.0
        emotion_mix = self._emotion_mix_from_vector(engine, style, emotion_vector, bool(generation_config.get("use_random"))) if emotion_vector is not None else None
        text_tokens_list = engine.tokenizer.tokenize(text)
        segments = engine.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
        silence_pcm = np.zeros(int(self.model_sample_rate * interval_silence_ms / 1000.0), dtype=np.int16)
        gen_kwargs = self._generation_kwargs(generation_config)
        max_total_tokens = int(gen_kwargs.pop("max_mel_tokens"))
        gen_kwargs["num_beams"] = 1
        gen_kwargs.setdefault("do_sample", self.do_sample)
        overlap_samples = int(self.model_sample_rate * token_window_overlap_ms / 1000.0)

        for seg_idx, sent in enumerate(segments):
            text_tokens = engine.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=engine.device).unsqueeze(0)
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=engine.dtype is not None, dtype=engine.dtype):
                    emovec = engine.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emotion_alpha,
                    )
                    if emotion_mix is not None:
                        emovec_mat, weight_vector = emotion_mix
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
            if self._can_stream_speech_codes(engine):
                code_batches = self._iter_streamed_speech_code_batches(
                    engine,
                    text_tokens=text_tokens,
                    spk_cond_emb=spk_cond_emb,
                    emo_cond_emb=emo_cond_emb,
                    emovec=emovec,
                    max_total_tokens=max_total_tokens,
                    token_window_size=token_window_size,
                    token_window_hop=token_window_hop,
                    generation_kwargs=gen_kwargs,
                )
            else:
                code_batches = self._iter_iterative_speech_code_batches(
                    engine,
                    text_tokens=text_tokens,
                    spk_cond_emb=spk_cond_emb,
                    emo_cond_emb=emo_cond_emb,
                    emovec=emovec,
                    max_total_tokens=max_total_tokens,
                    token_window_size=token_window_size,
                    token_window_hop=token_window_hop,
                    generation_kwargs=gen_kwargs,
                )

            prefix = torch.empty((1, 0), dtype=torch.long, device=text_tokens.device)
            emitted_until = 0
            final_pcm: np.ndarray | None = None
            for new_codes, speech_conditioning_latent, finished in code_batches:
                new_codes = new_codes.to(text_tokens.device)
                if new_codes.ndim == 1:
                    new_codes = new_codes.unsqueeze(0)
                new_len = int(new_codes.shape[-1])
                if new_len > 0:
                    prefix = torch.cat([prefix, new_codes.long()], dim=1)
                if prefix.numel() == 0:
                    break
                final = bool(finished) or prefix.shape[-1] >= max_total_tokens
                if token_window_context > 0:
                    if new_len <= 0:
                        break
                    prefix_len = int(prefix.shape[-1])
                    decode_start = max(0, prefix_len - new_len - token_window_context)
                    decode_codes = prefix[:, decode_start:]
                    skip_tokens = max(0, int(decode_codes.shape[-1]) - new_len)
                    final_pcm = self._decode_codes_to_model_pcm(
                        engine,
                        codes=decode_codes,
                        code_lens=torch.tensor([decode_codes.shape[-1]], device=text_tokens.device, dtype=torch.long),
                        speech_conditioning_latent=speech_conditioning_latent,
                        text_tokens=text_tokens,
                        spk_cond_emb=spk_cond_emb,
                        emo_cond_emb=emo_cond_emb,
                        emovec=emovec,
                        prompt_condition=prompt_condition,
                        ref_mel=ref_mel,
                        style=style,
                    )
                    start_sample = min(_speech_tokens_to_model_samples(skip_tokens), final_pcm.size)
                    if final_pcm.size > start_sample:
                        yield final_pcm[start_sample:]
                else:
                    final_pcm = self._decode_codes_to_model_pcm(
                        engine,
                        codes=prefix,
                        code_lens=torch.tensor([prefix.shape[-1]], device=text_tokens.device, dtype=torch.long),
                        speech_conditioning_latent=speech_conditioning_latent,
                        text_tokens=text_tokens,
                        spk_cond_emb=spk_cond_emb,
                        emo_cond_emb=emo_cond_emb,
                        emovec=emovec,
                        prompt_condition=prompt_condition,
                        ref_mel=ref_mel,
                        style=style,
                    )
                    holdback = min(overlap_samples, max(0, final_pcm.size // 3)) if not final else 0
                    safe_until = final_pcm.size if final else max(emitted_until, final_pcm.size - holdback)
                    if safe_until > emitted_until:
                        yield final_pcm[emitted_until:safe_until]
                        emitted_until = safe_until
                if final:
                    break
            if token_window_context <= 0 and final_pcm is not None and emitted_until < final_pcm.size:
                yield final_pcm[emitted_until:]
            if interval_silence_ms > 0 and seg_idx < len(segments) - 1 and silence_pcm.size:
                yield silence_pcm

    def _emotion_vector_from_config(self, engine: Any, text: str, config: dict[str, Any]) -> list[float] | None:
        emo_vector = config.get("emo_vector")
        if config.get("use_emo_text"):
            emo_text = str(config.get("emo_text") or text).strip() or text
            emo_dict = engine.qwen_emo.inference(emo_text)
            emo_vector = list(emo_dict.values())
        if emo_vector is None:
            return None
        if not isinstance(emo_vector, (list, tuple)) or len(emo_vector) != 8:
            raise RuntimeError("IndexTTS emo_vector must contain 8 values")
        scale = _clamp_unit_float(config.get("emo_alpha"), 1.0)
        return [int(float(value) * scale * 10000) / 10000 for value in emo_vector]

    def _emotion_audio_prompt_from_config(self, config: dict[str, Any] | None, prompt_audio: str) -> str:
        value = str((config or {}).get("emo_audio_prompt") or "").strip()
        if value:
            return value
        return prompt_audio

    def _emotion_mix_from_vector(self, engine: Any, style: Any, emo_vector: list[float], use_random: bool) -> tuple[Any, Any]:
        import torch

        weight_vector = torch.tensor(emo_vector, device=engine.device)
        if use_random:
            random_index = [random.randint(0, max(0, int(count) - 1)) for count in engine.emo_num]
        else:
            random_index = [_find_most_similar_cosine(style, matrix) for matrix in engine.spk_matrix]
        emo_matrix = [matrix[int(index)].unsqueeze(0) for index, matrix in zip(random_index, engine.emo_matrix)]
        emo_matrix = torch.cat(emo_matrix, 0)
        emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix, 0).unsqueeze(0)
        return emovec_mat, weight_vector

    def _can_stream_speech_codes(self, engine: Any) -> bool:
        gpt = getattr(engine, "gpt", None)
        inference_model = getattr(gpt, "inference_model", None)
        return (
            gpt is not None
            and getattr(gpt, "accel_engine", None) is None
            and callable(getattr(gpt, "get_conditioning", None))
            and callable(getattr(gpt, "prepare_gpt_inputs", None))
            and callable(getattr(gpt, "speed_emb", None))
            and callable(getattr(inference_model, "generate", None))
            and callable(getattr(inference_model, "store_mel_emb", None))
        )

    def _iter_iterative_speech_code_batches(
        self,
        engine: Any,
        *,
        text_tokens: Any,
        spk_cond_emb: Any,
        emo_cond_emb: Any,
        emovec: Any,
        max_total_tokens: int,
        token_window_size: int,
        token_window_hop: int,
        generation_kwargs: dict[str, Any],
    ):
        import torch

        prefix = torch.empty((1, 0), dtype=torch.long, device=text_tokens.device)
        finished = False
        first_step = True
        while not finished and prefix.shape[-1] < max_total_tokens:
            step_tokens = token_window_size if first_step else token_window_hop
            step_tokens = max(1, min(step_tokens, max_total_tokens - int(prefix.shape[-1])))
            input_tokens = prefix if prefix.numel() else None
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=engine.dtype is not None, dtype=engine.dtype):
                    new_codes, speech_conditioning_latent = engine.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        input_tokens=input_tokens,
                        num_return_sequences=1,
                        max_generate_length=step_tokens,
                        **generation_kwargs,
                    )
            if not hasattr(new_codes, "ndim"):
                new_codes = new_codes.sequences
            new_codes = new_codes.to(text_tokens.device)
            if new_codes.ndim == 1:
                new_codes = new_codes.unsqueeze(0)
            new_codes = new_codes[:, :step_tokens]
            stop_positions = (new_codes[0] == engine.stop_mel_token).nonzero(as_tuple=False)
            if stop_positions.numel() > 0:
                new_codes = new_codes[:, : int(stop_positions[0].item())]
                finished = True
            if new_codes.numel() == 0:
                finished = True
            else:
                prefix = torch.cat([prefix, new_codes.long()], dim=1)
            yield new_codes, speech_conditioning_latent, finished or prefix.shape[-1] >= max_total_tokens
            first_step = False

    def _iter_streamed_speech_code_batches(
        self,
        engine: Any,
        *,
        text_tokens: Any,
        spk_cond_emb: Any,
        emo_cond_emb: Any,
        emovec: Any,
        max_total_tokens: int,
        token_window_size: int,
        token_window_hop: int,
        generation_kwargs: dict[str, Any],
    ):
        import torch

        gpt = engine.gpt
        with torch.no_grad():
            with torch.amp.autocast(text_tokens.device.type, enabled=engine.dtype is not None, dtype=engine.dtype):
                speech_conditioning_latent = gpt.get_conditioning(
                    spk_cond_emb.transpose(1, 2),
                    torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                )
                tmp = torch.zeros(text_tokens.size(0)).to(text_tokens.device)
                duration_emb = gpt.speed_emb(torch.zeros_like(tmp).long())
                duration_emb_half = gpt.speed_emb(torch.ones_like(tmp).long())
                conds_latent = torch.cat(
                    (speech_conditioning_latent + emovec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)),
                    1,
                )
                input_ids, inputs_embeds, attention_mask = gpt.prepare_gpt_inputs(conds_latent, text_tokens)
                gpt.inference_model.store_mel_emb(inputs_embeds)

        streamer = _TokenQueueStreamer(
            prompt_length=int(input_ids.shape[1]),
            max_queue=max(8, int(token_window_hop) * 4),
        )
        generate_kwargs = dict(generation_kwargs)
        max_length = int(input_ids.shape[1]) + int(max_total_tokens)

        def run_generate() -> None:
            try:
                with torch.no_grad():
                    with torch.amp.autocast(
                        text_tokens.device.type,
                        enabled=engine.dtype is not None,
                        dtype=engine.dtype,
                    ):
                        gpt.inference_model.generate(
                            input_ids,
                            bos_token_id=gpt.start_mel_token,
                            pad_token_id=gpt.stop_mel_token,
                            eos_token_id=gpt.stop_mel_token,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            num_return_sequences=1,
                            streamer=streamer,
                            **generate_kwargs,
                        )
            except BaseException as exc:  # noqa: BLE001
                streamer.put_error(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=run_generate, name="omnirt-indextts-gpt-stream", daemon=True)
        thread.start()
        batch: list[int] = []
        first_batch = True
        emitted = 0
        try:
            for token in streamer:
                finished = int(token) == int(engine.stop_mel_token)
                if not finished:
                    batch.append(int(token))
                    emitted += 1
                target = token_window_size if first_batch else token_window_hop
                target = max(1, int(target))
                if batch and (len(batch) >= target or finished or emitted >= max_total_tokens):
                    yield torch.tensor(batch, dtype=torch.long, device=text_tokens.device).unsqueeze(0), speech_conditioning_latent, finished or emitted >= max_total_tokens
                    batch = []
                    first_batch = False
                if finished or emitted >= max_total_tokens:
                    break
            if batch:
                yield torch.tensor(batch, dtype=torch.long, device=text_tokens.device).unsqueeze(0), speech_conditioning_latent, True
        finally:
            thread.join(timeout=1.0)

    def _prepare_prompt_context(self, engine: Any, prompt_audio: str | None = None) -> tuple[Any, Any, Any, Any]:
        import torch
        import torchaudio

        effective_prompt_audio = prompt_audio or self.prompt_audio
        if engine.cache_spk_cond is not None and engine.cache_spk_audio_prompt == effective_prompt_audio:
            return engine.cache_spk_cond, engine.cache_s2mel_style, engine.cache_s2mel_prompt, engine.cache_mel
        if engine.cache_spk_cond is not None:
            engine.cache_spk_cond = None
            engine.cache_s2mel_style = None
            engine.cache_s2mel_prompt = None
            engine.cache_mel = None
            if str(engine.device).startswith("cuda"):
                torch.cuda.empty_cache()
        audio, sr = engine._load_and_cut_audio(effective_prompt_audio, 15, False)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        inputs = engine.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(engine.device)
        attention_mask = inputs["attention_mask"].to(engine.device)
        spk_cond_emb = engine.get_emb(input_features, attention_mask)
        _, s_ref = engine.semantic_codec.quantize(spk_cond_emb)
        ref_mel = engine.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = engine.campplus_model(feat.unsqueeze(0))
        prompt_condition = engine.s2mel.models["length_regulator"](
            s_ref,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]
        engine.cache_spk_cond = spk_cond_emb
        engine.cache_s2mel_style = style
        engine.cache_s2mel_prompt = prompt_condition
        engine.cache_spk_audio_prompt = effective_prompt_audio
        engine.cache_mel = ref_mel
        return spk_cond_emb, style, prompt_condition, ref_mel

    def _prepare_emo_context(self, engine: Any, prompt_audio: str | None = None) -> Any:
        import torch

        effective_prompt_audio = prompt_audio or self.prompt_audio
        if engine.cache_emo_cond is not None and engine.cache_emo_audio_prompt == effective_prompt_audio:
            return engine.cache_emo_cond
        if engine.cache_emo_cond is not None:
            engine.cache_emo_cond = None
            if str(engine.device).startswith("cuda"):
                torch.cuda.empty_cache()
        emo_audio, _ = engine._load_and_cut_audio(effective_prompt_audio, 15, False, sr=16000)
        emo_inputs = engine.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"].to(engine.device)
        emo_attention_mask = emo_inputs["attention_mask"].to(engine.device)
        emo_cond_emb = engine.get_emb(emo_input_features, emo_attention_mask)
        engine.cache_emo_cond = emo_cond_emb
        engine.cache_emo_audio_prompt = effective_prompt_audio
        return emo_cond_emb

    def _decode_codes_to_model_pcm(
        self,
        engine: Any,
        *,
        codes: Any,
        code_lens: Any,
        speech_conditioning_latent: Any,
        text_tokens: Any,
        spk_cond_emb: Any,
        emo_cond_emb: Any,
        emovec: Any,
        prompt_condition: Any,
        ref_mel: Any,
        style: Any,
    ) -> np.ndarray:
        import torch

        use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
        with torch.no_grad():
            with torch.amp.autocast(text_tokens.device.type, enabled=engine.dtype is not None, dtype=engine.dtype):
                latent = engine.gpt(
                    speech_conditioning_latent,
                    text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=text_tokens.device),
                    emo_cond_emb,
                    cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_vec=emovec,
                    use_speed=use_speed,
                )
            dtype = None
            with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                latent = engine.s2mel.models["gpt_layer"](latent)
                s_infer = engine.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                s_infer = s_infer.transpose(1, 2)
                s_infer = s_infer + latent
                target_lengths = (code_lens * 1.72).long()
                cond = engine.s2mel.models["length_regulator"](
                    s_infer,
                    ylens=target_lengths,
                    n_quantizers=3,
                    f0=None,
                )[0]
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = engine.s2mel.models["cfm"].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                    ref_mel,
                    style,
                    None,
                    25,
                    inference_cfg_rate=0.7,
                )
                vc_target = vc_target[:, :, ref_mel.size(-1) :]
                wav = engine.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                wav = wav.squeeze(1)
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        return _tensor_to_i16_pcm(wav)

    def _start_worker(
        self,
        text: str,
        config: dict[str, Any],
        out_q: "queue.Queue[bytes | BaseException | None]",
        prompt_audio: str,
    ) -> None:
        if self._config_streaming_mode(config) == "token_window":
            self._start_token_window_worker(text, config, out_q, prompt_audio)
            return
        self._validate_inputs(prompt_audio)
        engine = self._load_engine()
        infer = getattr(engine, "infer", None)
        if not callable(infer):
            raise RuntimeError("IndexTTS2 runtime does not expose infer().")
        infer_params = signature(infer).parameters
        max_tokens = int(config.get("max_text_tokens_per_segment") or self.max_text_tokens_per_segment)
        quick_tokens = int(config.get("quick_streaming_tokens") or self.quick_streaming_tokens)
        silence_ms = int(config.get("interval_silence_ms") if config.get("interval_silence_ms") is not None else self.interval_silence_ms)

        def worker() -> None:
            try:
                started = time.perf_counter()
                first_chunk_ms: float | None = None
                emitted_chunks = 0
                emitted_bytes = 0
                emitted_segments = 0
                kwargs: dict[str, Any] = {
                    "stream_return": True,
                    "max_text_tokens_per_segment": max(1, max_tokens),
                    "interval_silence": max(0, silence_ms),
                    **self._generation_kwargs(config),
                    **_indextts_emotion_kwargs(config),
                }
                accepts_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in infer_params.values())
                if "more_segment_before" in infer_params:
                    kwargs["more_segment_before"] = max(0, quick_tokens)
                elif "quick_streaming_tokens" in infer_params or accepts_kwargs:
                    kwargs["quick_streaming_tokens"] = max(0, quick_tokens)
                iterator = infer(str(Path(prompt_audio)), text, None, **kwargs)
                for item in iterator:
                    pcm = _tensor_to_i16_pcm(item)
                    if pcm.size == 0:
                        continue
                    emitted_segments += 1
                    pcm = _resample_linear(pcm, self.model_sample_rate, self.sample_rate)
                    for chunk in _split_pcm_bytes(pcm, self.sample_rate, self.chunk_ms):
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started) * 1000.0
                        emitted_chunks += 1
                        emitted_bytes += len(chunk)
                        out_q.put(chunk)
                total_ms = (time.perf_counter() - started) * 1000.0
                log.info(
                    "IndexTTS stream finished | first_chunk_ms=%s total_ms=%.1f segments=%d chunks=%d bytes=%d max_tokens=%d quick_tokens=%d",
                    "none" if first_chunk_ms is None else f"{first_chunk_ms:.1f}",
                    total_ms,
                    emitted_segments,
                    emitted_chunks,
                    emitted_bytes,
                    max(1, max_tokens),
                    max(0, quick_tokens),
                )
            except BaseException as exc:  # noqa: BLE001
                out_q.put(exc)
            finally:
                out_q.put(None)

        threading.Thread(target=worker, name="omnirt-indextts-stream", daemon=True).start()


def _first_existing_local_audio_dir(root: str, *names: str, required_file: str = "") -> str:
    base = Path(root)
    for name in names:
        path = base / name
        if not path.is_dir():
            continue
        if required_file and not (path / required_file).is_file():
            continue
        return str(path)
    return str(base / names[0])


def create_indextts_runtime_from_env() -> IndexTTSStreamingRuntime:
    model = os.environ.get("OMNIRT_INDEXTTS_MODEL", "IndexTeam/IndexTTS-2").strip() or "IndexTeam/IndexTTS-2"
    root = os.environ.get("OMNIRT_LOCAL_AUDIO_MODEL_ROOT", "").strip() or os.environ.get(
        "OPENTALKING_LOCAL_AUDIO_MODEL_ROOT",
        "/data2/zhongyi/model/local-audio",
    ).strip()
    model_dir = os.environ.get("OMNIRT_INDEXTTS_MODEL_DIR", "").strip() or str(Path(root) / model.replace("/", "__"))
    cfg_path = os.environ.get("OMNIRT_INDEXTTS_CFG_PATH", "").strip() or str(Path(model_dir) / "config.yaml")
    streaming_mode = os.environ.get("OMNIRT_INDEXTTS_STREAMING_MODE", DEFAULT_STREAMING_MODE).strip() or DEFAULT_STREAMING_MODE
    max_text_tokens = _env_optional_int("OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT")

    return IndexTTSStreamingRuntime(
        model=model,
        model_dir=model_dir,
        cfg_path=cfg_path,
        prompt_audio=os.environ.get("OMNIRT_INDEXTTS_PROMPT_AUDIO", "").strip(),
        voices_dir=os.environ.get("OMNIRT_INDEXTTS_VOICES_DIR", "").strip() or str(Path(root) / "voices"),
        sample_rate=_env_int("OMNIRT_INDEXTTS_SAMPLE_RATE", 16000),
        model_sample_rate=_env_int("OMNIRT_INDEXTTS_MODEL_SAMPLE_RATE", 22050),
        chunk_ms=_env_float("OMNIRT_INDEXTTS_CHUNK_MS", 20.0),
        max_text_tokens_per_segment=max_text_tokens,
        quick_streaming_tokens=_env_int("OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS", DEFAULT_QUICK_STREAMING_TOKENS),
        interval_silence_ms=_env_int("OMNIRT_INDEXTTS_INTERVAL_SILENCE_MS", 0),
        device=os.environ.get("OMNIRT_INDEXTTS_DEVICE", "auto").strip() or "auto",
        use_fp16=_env_bool("OMNIRT_INDEXTTS_USE_FP16", True),
        use_cuda_kernel=_env_bool("OMNIRT_INDEXTTS_USE_CUDA_KERNEL", False),
        use_deepspeed=_env_bool("OMNIRT_INDEXTTS_USE_DEEPSPEED", False),
        w2v_bert_dir=os.environ.get("OMNIRT_INDEXTTS_W2V_BERT_DIR", "").strip()
        or str(Path(root) / "facebook__w2v-bert-2.0"),
        maskgct_dir=os.environ.get("OMNIRT_INDEXTTS_MASKGCT_DIR", "").strip()
        or _first_existing_local_audio_dir(root, "amphion__MaskGCT", "amphion__MaskGCT-ms", required_file="semantic_codec/model.safetensors"),
        campplus_dir=os.environ.get("OMNIRT_INDEXTTS_CAMPPLUS_DIR", "").strip()
        or str(Path(root) / "funasr__campplus"),
        bigvgan_dir=os.environ.get("OMNIRT_INDEXTTS_BIGVGAN_DIR", "").strip()
        or str(Path(root) / "nvidia__bigvgan_v2_22khz_80band_256x"),
        do_sample=_env_bool("OMNIRT_INDEXTTS_DO_SAMPLE", DEFAULT_DO_SAMPLE),
        top_p=_env_float("OMNIRT_INDEXTTS_TOP_P", DEFAULT_TOP_P),
        top_k=_env_int("OMNIRT_INDEXTTS_TOP_K", DEFAULT_TOP_K),
        temperature=_env_float("OMNIRT_INDEXTTS_TEMPERATURE", DEFAULT_TEMPERATURE),
        num_beams=_env_int("OMNIRT_INDEXTTS_NUM_BEAMS", DEFAULT_NUM_BEAMS),
        repetition_penalty=_env_float("OMNIRT_INDEXTTS_REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY),
        max_mel_tokens=_env_int("OMNIRT_INDEXTTS_MAX_MEL_TOKENS", DEFAULT_MAX_MEL_TOKENS),
        streaming_mode=streaming_mode,
        token_window_size=_env_int("OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE", DEFAULT_TOKEN_WINDOW_SIZE),
        token_window_hop=_env_int("OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP", DEFAULT_TOKEN_WINDOW_HOP),
        token_window_context=_env_int("OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT", DEFAULT_TOKEN_WINDOW_CONTEXT),
        token_window_overlap_ms=_env_int("OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS", DEFAULT_TOKEN_WINDOW_OVERLAP_MS),
    )
