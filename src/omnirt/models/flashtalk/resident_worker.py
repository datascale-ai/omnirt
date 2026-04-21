"""Resident worker implementation for FlashTalk execution."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import uuid
from typing import Any, Iterator, Optional

import numpy as np

from omnirt.core.types import Artifact, GenerateRequest, GenerateResult
from omnirt.models.flashtalk.components import DEFAULT_FLASHTALK_PROMPT
from omnirt.models.flashtalk.pipeline import FlashTalkPipeline, FlashTalkRuntimeConfig, probe_video_file
from omnirt.telemetry.report import build_run_report


@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _repo_on_path(path: Path) -> Iterator[None]:
    text = str(path)
    inserted = False
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


class FlashTalkResidentWorker:
    """Resident FlashTalk worker.

    Supports both a single-process in-memory worker and a torchrun-backed
    distributed resident worker where rank 0 exposes the gRPC service.
    """

    def __init__(self, *, runtime, model_spec, config, adapters) -> None:
        self.runtime = runtime
        self.model_spec = model_spec
        self.config = dict(config)
        self.adapters = list(adapters or [])
        self.runtime_config: Optional[FlashTalkRuntimeConfig] = None
        self._inference = None
        self._save_video = None
        self._pipeline = None
        self._started = False
        self._dist = None
        self._rank = 0
        self._world_size = 1
        self._pending_call: _PendingDistributedCall | None = None
        self._pending_lock = threading.Condition()
        self._shutdown_requested = False
        self.serves_rpc = True

    def start(self) -> None:
        if self._started:
            return
        runtime_config = FlashTalkPipeline.resolve_runtime_config(self.config)
        distributed_requested = self._distributed_requested(runtime_config)
        dist_module = self._resolve_distributed_context(runtime_config) if distributed_requested else None
        inference, save_video = self._load_runtime_modules(runtime_config.repo_path)
        self._pipeline = inference.get_pipeline(
            world_size=self._distributed_world_size(runtime_config),
            ckpt_dir=str(runtime_config.ckpt_dir),
            wav2vec_dir=str(runtime_config.wav2vec_dir),
            cpu_offload=runtime_config.cpu_offload,
            t5_quant=runtime_config.t5_quant,
            t5_quant_dir=str(runtime_config.t5_quant_dir) if runtime_config.t5_quant_dir is not None else None,
            wan_quant=runtime_config.wan_quant,
            wan_quant_include=runtime_config.wan_quant_include,
            wan_quant_exclude=runtime_config.wan_quant_exclude,
        )
        self.runtime_config = runtime_config
        self._inference = inference
        self._save_video = save_video
        if dist_module is not None:
            is_initialized = getattr(dist_module, "is_initialized", None)
            if not callable(is_initialized) or not is_initialized():
                raise RuntimeError("FlashTalk did not initialize torch.distributed under the resident worker.")
            self._dist = dist_module
            self._rank = int(dist_module.get_rank())
            self._world_size = int(dist_module.get_world_size())
            self.serves_rpc = self._rank == 0
            self._shutdown_requested = False
        self._started = True

    def ready(self) -> bool:
        return (
            self._started
            and self._pipeline is not None
            and self._inference is not None
            and self._save_video is not None
        )

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.start()
        if not self.ready():
            raise RuntimeError("FlashTalk resident worker failed to initialize.")
        if self._dist is not None:
            if not self.serves_rpc:
                raise RuntimeError("Only rank 0 accepts resident FlashTalk RPC requests.")
            call = _PendingDistributedCall(request=request)
            with self._pending_lock:
                while self._pending_call is not None:
                    self._pending_lock.wait(timeout=0.1)
                self._pending_call = call
                self._pending_lock.notify_all()
                while not call.done:
                    self._pending_lock.wait(timeout=0.1)
            if call.error is not None:
                raise call.error
            if call.result is None:
                raise RuntimeError("Distributed FlashTalk worker finished without producing a result.")
            return call.result
        return self._run_request(request, export_result=True)

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        if self._dist is None:
            return False
        self.serve_forever(timeout=timeout)
        return self._shutdown_requested

    def shutdown(self) -> None:
        if self._dist is not None:
            with self._pending_lock:
                self._shutdown_requested = True
                self._pending_lock.notify_all()
        self._pending_call = None
        self._shutdown_requested = False
        self._dist = None
        self._rank = 0
        self._world_size = 1
        self.serves_rpc = True
        self._pipeline = None
        self._inference = None
        self._save_video = None
        self._started = False

    def _run_request(self, request: GenerateRequest, *, export_result: bool) -> GenerateResult | None:
        assert self.runtime_config is not None
        assert self._inference is not None
        assert self._save_video is not None
        assert self._pipeline is not None

        conditions = self._prepare_conditions(request)
        seed = int(request.config.get("seed", 9999))
        output_dir = Path(request.config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_file = output_dir / f"{request.model}-{seed}-{int(time.time() * 1000)}.mp4"

        timings: dict[str, float] = {}
        started = time.perf_counter()
        self._inference.get_base_data(
            self._pipeline,
            input_prompt=conditions["prompt"],
            cond_image=str(conditions["image_path"]),
            base_seed=seed,
        )
        timings["prepare_conditions_ms"] = round((time.perf_counter() - started) * 1000, 3)

        denoise_started = time.perf_counter()
        generated_list = self._generate_video_frames(
            audio_path=conditions["audio_path"],
            audio_encode_mode=str(request.config.get("audio_encode_mode", "stream")),
            max_chunks=int(request.config.get("max_chunks", 0)),
        )
        timings["denoise_loop_ms"] = round((time.perf_counter() - denoise_started) * 1000, 3)

        artifacts: list[Artifact] = []
        if export_result:
            export_started = time.perf_counter()
            self._save_video(
                generated_list,
                str(save_file),
                str(conditions["audio_path"]),
                fps=self._inference.infer_params["tgt_fps"],
            )
            width, height, num_frames = probe_video_file(save_file)
            artifacts = [
                Artifact(
                    kind="video",
                    path=str(save_file),
                    mime="video/mp4",
                    width=width,
                    height=height,
                    num_frames=num_frames,
                )
            ]
            timings["export_ms"] = round((time.perf_counter() - export_started) * 1000, 3)

        report = build_run_report(
            run_id=str(uuid.uuid4()),
            request=request,
            backend_name=self.runtime.name,
            timings=timings,
            memory=self.runtime.memory_stats() if hasattr(self.runtime, "memory_stats") else {},
            backend_timeline=getattr(self.runtime, "backend_timeline", []),
            config_resolved={
                "repo_path": str(self.runtime_config.repo_path),
                "ckpt_dir": str(self.runtime_config.ckpt_dir),
                "wav2vec_dir": str(self.runtime_config.wav2vec_dir),
                "seed": seed,
                "output_dir": str(output_dir),
                "audio_encode_mode": str(request.config.get("audio_encode_mode", "stream")),
                "max_chunks": int(request.config.get("max_chunks", 0)),
                "launcher": self.runtime_config.launcher,
                "nproc_per_node": self.runtime_config.nproc_per_node,
            },
            artifacts=artifacts,
            error=None,
            execution_mode="persistent_worker",
        )
        return GenerateResult(outputs=artifacts, metadata=report) if export_result else None

    def serve_forever(self, timeout: float | None = None) -> None:
        if self._dist is None:
            if timeout:
                time.sleep(timeout)
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._shutdown_requested:
            if deadline is not None and time.monotonic() >= deadline:
                return
            self._run_distributed_step()

    def _load_runtime_modules(self, repo_path: Path):
        with _repo_on_path(repo_path), _temporary_cwd(repo_path):
            inference = importlib.import_module("flash_talk.inference")
            generate_video = importlib.import_module("generate_video")
        return inference, getattr(generate_video, "save_video")

    def _resolve_distributed_context(self, runtime_config: FlashTalkRuntimeConfig):
        if not self._distributed_requested(runtime_config):
            return None
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise NotImplementedError(
                "FlashTalk distributed resident worker requires torch.distributed to be available."
            ) from exc
        rank = os.environ.get("RANK")
        world_size = os.environ.get("WORLD_SIZE")
        if rank is None or world_size is None:
            raise NotImplementedError(
                "FlashTalk distributed resident worker must be launched under torchrun/accelerate so "
                "distributed environment variables are available."
            )
        return dist

    def _distributed_requested(self, runtime_config: FlashTalkRuntimeConfig) -> bool:
        return (
            runtime_config.launcher != "python"
            or runtime_config.nproc_per_node > 1
            or runtime_config.num_processes > 1
        )

    def _distributed_world_size(self, runtime_config: FlashTalkRuntimeConfig) -> int:
        if not self._distributed_requested(runtime_config):
            return 1
        world_size = os.environ.get("WORLD_SIZE")
        if world_size:
            return int(world_size)
        return max(runtime_config.nproc_per_node, runtime_config.num_processes, 1)

    def _run_distributed_step(self) -> None:
        assert self._dist is not None
        message = self._next_distributed_message()
        if message.get("op") == "shutdown":
            self._shutdown_requested = True
            return
        request_payload = message.get("request")
        if not isinstance(request_payload, dict):
            raise RuntimeError("FlashTalk distributed worker received an invalid request payload.")
        request = GenerateRequest.from_dict(request_payload)
        result: GenerateResult | None = None
        error_message: str | None = None
        try:
            result = self._run_request(request, export_result=self.serves_rpc)
        except Exception as exc:  # pragma: no cover - exercised via collective error tests with monkeypatching.
            error_message = f"{exc.__class__.__name__}: {exc}"
        if self.serves_rpc:
            with self._pending_lock:
                pending = self._pending_call
                self._pending_call = None
                if pending is not None:
                    if error_message:
                        pending.error = RuntimeError(error_message)
                    elif result is None:
                        pending.error = RuntimeError(
                            "Distributed FlashTalk worker finished without returning a coordinator result."
                        )
                    else:
                        pending.result = result
                    pending.done = True
                self._pending_lock.notify_all()

    def _next_distributed_message(self) -> dict[str, Any]:
        assert self._dist is not None
        if self.serves_rpc:
            with self._pending_lock:
                while self._pending_call is None and not self._shutdown_requested:
                    self._pending_lock.wait(timeout=0.1)
                if self._shutdown_requested:
                    payload = {"op": "shutdown"}
                else:
                    assert self._pending_call is not None
                    payload = {"op": "run", "request": self._pending_call.request.to_dict()}
        else:
            payload = None
        objects = [payload]
        self._dist.broadcast_object_list(objects, src=0)
        message = objects[0]
        if not isinstance(message, dict):
            raise RuntimeError("FlashTalk distributed worker received a malformed collective message.")
        return message

    def _prepare_conditions(self, request: GenerateRequest) -> dict[str, Any]:
        image_path = Path(str(request.inputs.get("image", ""))).expanduser()
        audio_path = Path(str(request.inputs.get("audio", ""))).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        return {
            "image_path": image_path,
            "audio_path": audio_path,
            "prompt": str(request.inputs.get("prompt") or DEFAULT_FLASHTALK_PROMPT),
        }

    def _generate_video_frames(self, *, audio_path: Path, audio_encode_mode: str, max_chunks: int):
        assert self._inference is not None
        assert self._pipeline is not None

        import librosa

        infer_params = self._inference.infer_params
        sample_rate = infer_params["sample_rate"]
        tgt_fps = infer_params["tgt_fps"]
        frame_num = infer_params["frame_num"]
        motion_frames_num = infer_params["motion_frames_num"]
        slice_len = frame_num - motion_frames_num

        generated_list = []
        human_speech_array_all, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

        if audio_encode_mode == "once":
            remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
            if remainder > 0:
                pad_length = human_speech_array_slice_len - remainder
                human_speech_array_all = np.concatenate(
                    [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
                )

            audio_embedding_all = self._inference.get_audio_embedding(self._pipeline, human_speech_array_all)
            audio_embedding_chunks_list = [
                audio_embedding_all[:, index * slice_len : index * slice_len + frame_num].contiguous()
                for index in range((audio_embedding_all.shape[1] - frame_num) // slice_len)
            ]

            for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
                if max_chunks > 0 and chunk_idx >= max_chunks:
                    break
                video = self._inference.run_pipeline(self._pipeline, audio_embedding_chunk)
                if chunk_idx != 0:
                    video = video[motion_frames_num:]
                generated_list.append(video.cpu())
            return generated_list

        cached_audio_duration = infer_params["cached_audio_duration"]
        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate(
                [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
            )
        human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            if max_chunks > 0 and chunk_idx >= max_chunks:
                break
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = self._inference.get_audio_embedding(self._pipeline, audio_array, audio_start_idx, audio_end_idx)
            video = self._inference.run_pipeline(self._pipeline, audio_embedding)
            video = video[motion_frames_num:]
            generated_list.append(video.cpu())
        return generated_list


class _PendingDistributedCall:
    def __init__(self, *, request: GenerateRequest) -> None:
        self.request = request
        self.result: GenerateResult | None = None
        self.error: Exception | None = None
        self.done = False
