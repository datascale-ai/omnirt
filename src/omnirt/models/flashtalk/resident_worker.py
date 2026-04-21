"""Resident worker implementation for FlashTalk execution.

Architecture
------------

There are three distinct concerns this module keeps separate:

* **Ingress (gRPC / in-process callers)** — :meth:`FlashTalkResidentWorker.submit`
  is safe to call from multiple threads concurrently. Each submission is
  wrapped in a :class:`_QueuedRequest` and placed on a thread-safe
  :class:`queue.Queue`. The caller waits on the request's own Event; it no
  longer spins on a shared Condition.

* **Engine (serial execution)** — on rank 0 a dedicated coordinator thread
  pulls one request at a time off the queue, broadcasts the payload to peer
  ranks via ``torch.distributed.broadcast_object_list``, runs the pipeline on
  rank 0, and resolves the caller's Event. Non-rank-0 processes run a mirror
  loop driven by the same broadcasts.

* **Observability** — ``queue.qsize()`` feeds the health endpoint and
  ``omnirt_worker_queue_depth`` gauge; :attr:`_inflight` feeds
  ``omnirt_worker_inflight``; per-chunk durations are recorded on the
  Prometheus registry if one is attached.

This separation is the smallest change that removes the single-slot lock and
lets the gRPC thread pool accept N concurrent ``submit`` calls without busy
waiting. Execution itself is still serial — adding continuous batching later
means replacing the ``queue.get()`` policy, not rewriting ingress.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import queue
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


class _QueuedRequest:
    """A submission waiting for the coordinator to pick it up.

    ``done`` is an Event (not a sleep-poll loop) — the submitting thread blocks
    on ``event.wait()`` until the coordinator sets it after writing ``result``
    or ``error``. This is the ingress half of the rank 0 split.
    """

    __slots__ = ("request", "result", "error", "done", "enqueued_at")

    def __init__(self, request: GenerateRequest) -> None:
        self.request = request
        self.result: GenerateResult | None = None
        self.error: Exception | None = None
        self.done = threading.Event()
        self.enqueued_at = time.monotonic()


# Alias preserved for tests that referenced the old name.
_PendingDistributedCall = _QueuedRequest


class FlashTalkResidentWorker:
    """Resident FlashTalk worker.

    Supports three modes:

    * Single-process in-memory worker (non-distributed).
    * torchrun-backed distributed resident worker, rank 0 serving the gRPC
      endpoint.
    * Non-rank-0 distributed participants that run a passive broadcast loop.
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
        self.serves_rpc = True

        # Rank 0 concurrency primitives.
        self._request_queue: "queue.Queue[_QueuedRequest | None]" = queue.Queue()
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown_requested = False
        self._inflight_lock = threading.Lock()
        self._inflight = 0
        self._last_error: str | None = None

        # Optional observability hooks; set by PersistentWorkerExecutor or tests.
        self.metrics = None  # PrometheusMetrics-compatible
        self.worker_id = f"flashtalk-resident-{id(self):x}"

    # ------------------------------------------------------------------ lifecycle

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
            if self.serves_rpc:
                self._coordinator_thread = threading.Thread(
                    target=self._coordinator_loop,
                    name=f"{self.worker_id}-coordinator",
                    daemon=True,
                )
                self._coordinator_thread.start()
        self._started = True

    def ready(self) -> bool:
        return (
            self._started
            and self._pipeline is not None
            and self._inference is not None
            and self._save_video is not None
        )

    def shutdown(self) -> None:
        if self._dist is not None and self.serves_rpc:
            self._shutdown_requested = True
            # Wake coordinator so it can observe the shutdown flag.
            self._request_queue.put(None)
            if self._coordinator_thread is not None:
                self._coordinator_thread.join(timeout=5.0)
        self._coordinator_thread = None
        self._drain_queue()
        self._shutdown_requested = False
        self._dist = None
        self._rank = 0
        self._world_size = 1
        self.serves_rpc = True
        self._pipeline = None
        self._inference = None
        self._save_video = None
        self._started = False
        with self._inflight_lock:
            self._inflight = 0

    def _drain_queue(self) -> None:
        """Fail any requests still queued when shutdown happens."""
        while True:
            try:
                item = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _QueuedRequest):
                item.error = RuntimeError("FlashTalk resident worker shut down before request was served.")
                item.done.set()

    # ------------------------------------------------------------------ ingress

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.start()
        if not self.ready():
            raise RuntimeError("FlashTalk resident worker failed to initialize.")
        if self._dist is not None:
            if not self.serves_rpc:
                raise RuntimeError("Only rank 0 accepts resident FlashTalk RPC requests.")
            queued = _QueuedRequest(request=request)
            self._request_queue.put(queued)
            self._report_queue_metrics()
            queued.done.wait()
            if queued.error is not None:
                raise queued.error
            if queued.result is None:
                raise RuntimeError("Distributed FlashTalk worker finished without producing a result.")
            return queued.result
        # Non-distributed path stays inline; callers that need concurrency run
        # multiple workers via WorkerPool instead.
        with self._inflight_scope():
            return self._run_request(request, export_result=True)

    # ------------------------------------------------------------------ observability

    def status_snapshot(self) -> dict[str, Any]:
        """Return a dict consumable by ``GrpcWorkerServer._handle_health``."""
        if not self._started:
            state = "loading"
        elif not self.ready():
            state = "degraded"
        elif self._last_error:
            state = "degraded"
        else:
            state = "ready"
        gpu_mem_used_gb: float | None = None
        memory_stats = getattr(self.runtime, "memory_stats", None)
        if callable(memory_stats):
            try:
                stats = memory_stats()
            except Exception:
                stats = {}
            peak_mb = stats.get("peak_mb") if isinstance(stats, dict) else None
            if isinstance(peak_mb, (int, float)) and peak_mb > 0:
                gpu_mem_used_gb = round(float(peak_mb) / 1024.0, 3)
        return {
            "state": state,
            "model_loaded": self.ready(),
            "queue_depth": self._request_queue.qsize(),
            "inflight": self._inflight,
            "last_error": self._last_error,
            "gpu_mem_used_gb": gpu_mem_used_gb,
            "worker_id": self.worker_id,
            "rank": self._rank,
            "world_size": self._world_size,
        }

    def _report_queue_metrics(self) -> None:
        metrics = self.metrics
        if metrics is None:
            return
        setter = getattr(metrics, "set_worker_queue_depth", None)
        if callable(setter):
            setter(
                worker_id=self.worker_id,
                model=self.model_spec.id if self.model_spec is not None else "unknown",
                depth=self._request_queue.qsize(),
            )

    def _report_inflight_metrics(self) -> None:
        metrics = self.metrics
        if metrics is None:
            return
        setter = getattr(metrics, "set_worker_inflight", None)
        if callable(setter):
            setter(
                worker_id=self.worker_id,
                model=self.model_spec.id if self.model_spec is not None else "unknown",
                count=self._inflight,
            )

    @contextmanager
    def _inflight_scope(self) -> Iterator[None]:
        with self._inflight_lock:
            self._inflight += 1
        self._report_inflight_metrics()
        try:
            yield
        finally:
            with self._inflight_lock:
                self._inflight = max(0, self._inflight - 1)
            self._report_inflight_metrics()

    # ------------------------------------------------------------------ coordinator

    def _coordinator_loop(self) -> None:
        """Rank 0 only: drain the ingress queue and broadcast one request at a time."""
        assert self._dist is not None
        while not self._shutdown_requested:
            try:
                item = self._request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._report_queue_metrics()
            if item is None:
                # shutdown sentinel
                payload = {"op": "shutdown"}
                objects = [payload]
                self._dist.broadcast_object_list(objects, src=0)
                break
            with self._inflight_scope():
                self._run_broadcast_step(item)
            self._report_queue_metrics()

    def _run_broadcast_step(self, item: _QueuedRequest) -> None:
        assert self._dist is not None
        payload = {"op": "run", "request": item.request.to_dict()}
        objects = [payload]
        self._dist.broadcast_object_list(objects, src=0)
        result: GenerateResult | None = None
        error_message: str | None = None
        try:
            result = self._run_request(item.request, export_result=True)
        except Exception as exc:  # pragma: no cover - exercised via collective error tests
            error_message = f"{exc.__class__.__name__}: {exc}"
        if error_message:
            item.error = RuntimeError(error_message)
            self._last_error = error_message
        elif result is None:
            item.error = RuntimeError(
                "Distributed FlashTalk worker finished without returning a coordinator result."
            )
            self._last_error = str(item.error)
        else:
            item.result = result
            self._last_error = None
        item.done.set()

    # ------------------------------------------------------------------ non-rank0 loop

    def serve_forever(self, timeout: float | None = None) -> None:
        """Passive broadcast loop used by non-rank-0 participants.

        Rank 0 does NOT enter this — its coordinator thread is already driving
        broadcasts. Non-rank-0 ranks block here for the life of the process,
        waking only to participate in the broadcast collective.
        """
        if self._dist is None:
            if timeout:
                time.sleep(timeout)
            return
        if self.serves_rpc:
            # Rank 0's coordinator thread handles execution; this method is a
            # no-op (or blocks for optional shutdown signalling).
            if timeout:
                time.sleep(timeout)
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return
            message = self._receive_broadcast()
            if message.get("op") == "shutdown":
                return
            request_payload = message.get("request")
            if not isinstance(request_payload, dict):
                raise RuntimeError("FlashTalk distributed worker received an invalid request payload.")
            request = GenerateRequest.from_dict(request_payload)
            try:
                self._run_request(request, export_result=False)
            except Exception:  # pragma: no cover - peer-rank errors surface on rank 0 via exceptions
                continue

    def _receive_broadcast(self) -> dict[str, Any]:
        assert self._dist is not None
        objects: list[Any] = [None]
        self._dist.broadcast_object_list(objects, src=0)
        message = objects[0]
        if not isinstance(message, dict):
            raise RuntimeError("FlashTalk distributed worker received a malformed collective message.")
        return message

    # ------------------------------------------------------------------ execution

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
        generated_list, generation_metrics = self._generate_video_frames(
            audio_path=conditions["audio_path"],
            audio_encode_mode=str(request.config.get("audio_encode_mode", "stream")),
            max_chunks=int(request.config.get("max_chunks", 0)),
        )
        timings["denoise_loop_ms"] = round((time.perf_counter() - denoise_started) * 1000, 3)
        timings.update(generation_metrics)

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
            base_artifact = Artifact(
                kind="video",
                path=str(save_file),
                mime="video/mp4",
                width=width,
                height=height,
                num_frames=num_frames,
            )
            transport = str(request.config.get("artifact_transport", "path")).strip() or "path"
            if transport == "inline_bytes":
                from omnirt.core.artifact_transport import pack_artifact

                artifacts = [pack_artifact(base_artifact, transport="inline_bytes")]
            else:
                artifacts = [base_artifact]
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

    # ------------------------------------------------------------------ helpers

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
        audio_embedding_times_ms: list[float] = []
        chunk_core_times_ms: list[float] = []
        chunk_copy_times_ms: list[float] = []
        chunk_total_times_ms: list[float] = []
        human_speech_array_all, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

        chunk_metric = getattr(self.metrics, "observe_worker_chunk_duration", None)

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
                chunk_started = time.perf_counter()
                core_started = time.perf_counter()
                video = self._inference.run_pipeline(self._pipeline, audio_embedding_chunk)
                chunk_core_times_ms.append((time.perf_counter() - core_started) * 1000)
                if chunk_idx != 0:
                    video = video[motion_frames_num:]
                copy_started = time.perf_counter()
                generated_list.append(video.cpu())
                chunk_copy_times_ms.append((time.perf_counter() - copy_started) * 1000)
                chunk_total_times_ms.append((time.perf_counter() - chunk_started) * 1000)
                if callable(chunk_metric):
                    chunk_metric(
                        worker_id=self.worker_id,
                        model=self.model_spec.id if self.model_spec is not None else "unknown",
                        seconds=time.perf_counter() - chunk_started,
                    )
            return generated_list, self._build_generation_metrics(
                audio_embedding_times_ms=audio_embedding_times_ms,
                chunk_core_times_ms=chunk_core_times_ms,
                chunk_copy_times_ms=chunk_copy_times_ms,
                chunk_total_times_ms=chunk_total_times_ms,
            )

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
            chunk_started = time.perf_counter()
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            embed_started = time.perf_counter()
            audio_embedding = self._inference.get_audio_embedding(self._pipeline, audio_array, audio_start_idx, audio_end_idx)
            audio_embedding_times_ms.append((time.perf_counter() - embed_started) * 1000)
            core_started = time.perf_counter()
            video = self._inference.run_pipeline(self._pipeline, audio_embedding)
            chunk_core_times_ms.append((time.perf_counter() - core_started) * 1000)
            video = video[motion_frames_num:]
            copy_started = time.perf_counter()
            generated_list.append(video.cpu())
            chunk_copy_times_ms.append((time.perf_counter() - copy_started) * 1000)
            chunk_total_times_ms.append((time.perf_counter() - chunk_started) * 1000)
            if callable(chunk_metric):
                chunk_metric(
                    worker_id=self.worker_id,
                    model=self.model_spec.id if self.model_spec is not None else "unknown",
                    seconds=time.perf_counter() - chunk_started,
                )
        return generated_list, self._build_generation_metrics(
            audio_embedding_times_ms=audio_embedding_times_ms,
            chunk_core_times_ms=chunk_core_times_ms,
            chunk_copy_times_ms=chunk_copy_times_ms,
            chunk_total_times_ms=chunk_total_times_ms,
        )

    def _build_generation_metrics(
        self,
        *,
        audio_embedding_times_ms: list[float],
        chunk_core_times_ms: list[float],
        chunk_copy_times_ms: list[float],
        chunk_total_times_ms: list[float],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "chunk_count": float(len(chunk_total_times_ms)),
        }
        if audio_embedding_times_ms:
            metrics["audio_embedding_ms_avg"] = round(sum(audio_embedding_times_ms) / len(audio_embedding_times_ms), 3)
        if chunk_core_times_ms:
            metrics["chunk_core_ms_avg"] = round(sum(chunk_core_times_ms) / len(chunk_core_times_ms), 3)
        if chunk_copy_times_ms:
            metrics["chunk_copy_ms_avg"] = round(sum(chunk_copy_times_ms) / len(chunk_copy_times_ms), 3)
        if chunk_total_times_ms:
            metrics["chunk_total_ms_avg"] = round(sum(chunk_total_times_ms) / len(chunk_total_times_ms), 3)
        if len(chunk_core_times_ms) > 1:
            steady = chunk_core_times_ms[1:]
            metrics["steady_chunk_core_ms_avg"] = round(sum(steady) / len(steady), 3)
        if len(chunk_total_times_ms) > 1:
            steady_total = chunk_total_times_ms[1:]
            metrics["steady_chunk_total_ms_avg"] = round(sum(steady_total) / len(steady_total), 3)
        return metrics
