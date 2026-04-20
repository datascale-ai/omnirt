"""Base pipeline execution skeleton."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from omnirt.backends.cpu_stub import denoise_guard
from omnirt.core.adapters import AdapterManager
from omnirt.core.registry import ModelSpec
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, InsufficientMemoryError
from omnirt.telemetry.log import get_logger
from omnirt.telemetry.report import build_run_report

LEGACY_OPTIMIZATION_CONFIG_KEYS = (
    "enable_model_cpu_offload",
    "enable_sequential_cpu_offload",
    "enable_group_offload",
    "group_offload_type",
    "group_offload_use_stream",
    "group_offload_disk_path",
    "enable_vae_slicing",
    "enable_vae_tiling",
    "channels_last",
    "fuse_qkv",
)

RESULT_CACHE_CONFIG_KEYS = ("use_result_cache",)


class BasePipeline(ABC):
    """Shared execution contract for all pipelines."""

    def __init__(self, *, runtime: Any, model_spec: ModelSpec, adapters: Optional[Iterable[Any]] = None) -> None:
        self.runtime = runtime
        self.model_spec = model_spec
        self.adapter_manager = AdapterManager()
        self.adapters = list(adapters or [])
        self.logger = get_logger()
        self.last_report = None
        self.components: Dict[str, Any] = {}
        self.loaded_adapters = []
        self._captured_latent = None

        if self.adapters:
            self.loaded_adapters = self.adapter_manager.load_all(self.adapters)

    @abstractmethod
    def prepare_conditions(self, req: GenerateRequest) -> Any:
        raise NotImplementedError

    @abstractmethod
    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        raise NotImplementedError

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: Any) -> Dict[str, Any]:
        return dict(req.config)

    def resolve_output_dir(self, req: GenerateRequest) -> Path:
        output_dir = Path(req.config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def adapter_fingerprint(self) -> tuple:
        """Stable identity of currently loaded LoRA adapters (path, scale tuples, sorted)."""

        if not self.adapters:
            return ()
        return tuple(sorted((str(getattr(a, "path", "")), float(getattr(a, "scale", 1.0))) for a in self.adapters))

    def pipeline_cache_key(self, *, source: Any, torch_dtype: Any, scheduler_name: str) -> tuple:
        """Shared cache key for Diffusers pipeline reuse across repeat runs."""

        return (str(source), str(torch_dtype), str(scheduler_name), self.adapter_fingerprint())

    def make_latent_callback(self, total_steps: int):
        """Return a Diffusers-style ``callback_on_step_end`` that captures the final-step latents.

        Pipelines pass this to ``pipeline(callback_on_step_end=self.make_latent_callback(steps))``.
        Only the last step is kept; intermediate steps are ignored to avoid memory churn.
        """

        last_index = max(int(total_steps) - 1, 0)

        def _callback(pipe, step, timestep, callback_kwargs):  # noqa: ARG001 - Diffusers signature
            if step == last_index:
                latents = callback_kwargs.get("latents") if isinstance(callback_kwargs, dict) else None
                if latents is not None:
                    try:
                        self._captured_latent = latents.detach().to("cpu").float().numpy()
                    except Exception:
                        self._captured_latent = None
            return callback_kwargs if isinstance(callback_kwargs, dict) else {}

        return _callback

    def _supports_callback_on_step_end(self, pipeline: Any) -> bool:
        try:
            signature = inspect.signature(pipeline.__call__)
        except (TypeError, ValueError, AttributeError):
            return False
        parameters = signature.parameters
        if "callback_on_step_end" in parameters:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    def _compute_latent_stats(self) -> Optional[Dict[str, float]]:
        if self._captured_latent is None:
            return None
        try:
            from omnirt.core.parity import latent_statistics

            return latent_statistics(self._captured_latent)
        except Exception:
            return None

    def ensure_resource_budget(self) -> None:
        minimum = self.model_spec.resource_hint.get("min_vram_gb")
        if minimum is None:
            return

        available = self.runtime.available_memory_gb()
        if available is None or available >= float(minimum):
            return

        raise InsufficientMemoryError(
            model=self.model_spec.id,
            estimated_gb=float(minimum),
            available_gb=float(available),
            hint="use smaller model or upgrade hardware; offload is planned for v0.2",
        )

    def apply_pipeline_optimizations(self, pipeline: Any, *, config: Dict[str, Any]) -> tuple[Any, bool]:
        """Apply a shared subset of Diffusers runtime optimizations.

        Returns ``(pipeline, placement_managed)`` where ``placement_managed`` indicates
        whether the pipeline now manages device placement/offload internally and callers
        should skip the eager ``runtime.to_device(...)`` step.
        """

        if config.get("enable_vae_slicing") and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if config.get("enable_vae_tiling") and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if config.get("channels_last"):
            self._apply_channels_last(pipeline)
        if config.get("fuse_qkv") and hasattr(pipeline, "fuse_qkv_projections"):
            pipeline.fuse_qkv_projections()

        if config.get("enable_model_cpu_offload") and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            return pipeline, True
        if config.get("enable_sequential_cpu_offload") and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
            return pipeline, True
        if config.get("enable_group_offload") and hasattr(pipeline, "enable_group_offload"):
            kwargs = {
                "offload_type": config.get("group_offload_type", "block_level"),
                "use_stream": bool(config.get("group_offload_use_stream", True)),
            }
            if config.get("group_offload_disk_path"):
                kwargs["offload_to_disk_path"] = config["group_offload_disk_path"]
            pipeline.enable_group_offload(**kwargs)
            return pipeline, True
        return pipeline, False

    def _apply_channels_last(self, pipeline: Any) -> None:
        try:
            torch = self._torch()  # type: ignore[attr-defined]
        except Exception:
            return

        for tag in ("unet", "vae", "transformer"):
            module = getattr(pipeline, tag, None)
            if module is None or not hasattr(module, "to"):
                continue
            try:
                module.to(memory_format=torch.channels_last)
            except Exception:
                continue

    def run(self, req: GenerateRequest) -> GenerateResult:
        run_id = str(uuid.uuid4())
        timings: Dict[str, float] = {}
        outputs: List[Artifact] = []
        self.runtime.backend_timeline = []
        self.runtime.reset_memory_stats()
        self._captured_latent = None
        self.ensure_resource_budget()

        sync_stages = {"denoise_loop", "decode"}

        def timed(stage: str, fn: Any) -> Any:
            started = time.perf_counter()
            self.logger.info("stage.start", extra={"stage": stage, "run_id": run_id, "model": req.model})
            try:
                return fn()
            finally:
                if stage in sync_stages:
                    try:
                        self.runtime.synchronize()
                    except Exception:
                        pass
                elapsed_ms = (time.perf_counter() - started) * 1000
                timings[f"{stage}_ms"] = round(elapsed_ms, 3)
                self.logger.info(
                    "stage.end",
                    extra={"stage": stage, "run_id": run_id, "model": req.model, "elapsed_ms": elapsed_ms},
                )

        try:
            conditions = timed("prepare_conditions", lambda: self.prepare_conditions(req))
            latents = timed("prepare_latents", lambda: self.prepare_latents(req, conditions))
            resolved_config = self.resolve_run_config(req, conditions, latents)
            denoise_guard(self.runtime)
            denoised = timed("denoise_loop", lambda: self.denoise_loop(latents, conditions, req.config))
            raw = timed("decode", lambda: self.decode(denoised))
            outputs = timed("export", lambda: self.export(raw, req))
            report = build_run_report(
                run_id=run_id,
                request=req,
                backend_name=self.runtime.name,
                timings=timings,
                memory=self.runtime.memory_stats(),
                backend_timeline=self.runtime.backend_timeline,
                config_resolved=resolved_config,
                artifacts=outputs,
                error=None,
                latent_stats=self._compute_latent_stats(),
            )
            self.last_report = report
            return GenerateResult(outputs=outputs, metadata=report)
        except Exception as exc:
            report = build_run_report(
                run_id=run_id,
                request=req,
                backend_name=self.runtime.name,
                timings=timings,
                memory=self.runtime.memory_stats(),
                backend_timeline=self.runtime.backend_timeline,
                config_resolved=locals().get("resolved_config", dict(req.config)),
                artifacts=outputs,
                error=str(exc),
                latent_stats=self._compute_latent_stats(),
            )
            self.last_report = report
            self.logger.error(
                "run.failed",
                extra={"run_id": run_id, "model": req.model, "backend": self.runtime.name, "error": report.error},
            )
            raise
