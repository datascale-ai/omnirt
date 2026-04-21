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
from omnirt.launcher import DEVICE_MAP_CONFIG_KEYS, resolve_config_device_map
from omnirt.middleware import (
    QUANTIZATION_CONFIG_KEYS,
    TEA_CACHE_CONFIG_KEYS,
    apply_quantization_runtime,
    apply_tea_cache_runtime,
)
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
 ) + QUANTIZATION_CONFIG_KEYS + TEA_CACHE_CONFIG_KEYS

RESULT_CACHE_CONFIG_KEYS = ("use_result_cache",) + QUANTIZATION_CONFIG_KEYS + TEA_CACHE_CONFIG_KEYS


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
        self._active_request: Optional[GenerateRequest] = None
        self._active_result_cache = None
        self._active_cache_hits: List[str] = []

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

    def from_pretrained_runtime_kwargs(self, *, config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        device_map = resolve_config_device_map(config)
        if device_map is not None:
            kwargs["device_map"] = device_map
        return kwargs

    def uses_managed_device_placement(self, config: Dict[str, Any]) -> bool:
        return resolve_config_device_map(config) is not None

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

    def _memory_scope(self) -> str:
        scope = str(self.model_spec.resource_hint.get("vram_scope", "per_device")).strip().lower()
        return scope if scope in {"per_device", "aggregate"} else "per_device"

    def _parallel_device_count(self, req: GenerateRequest) -> int:
        merged_config = dict(self.model_spec.capabilities.default_config)
        merged_config.update(req.config)

        for key in ("nproc_per_node", "num_processes"):
            value = merged_config.get(key)
            if value in (None, ""):
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count

        visible_devices = merged_config.get("visible_devices")
        if isinstance(visible_devices, str):
            count = len([item for item in visible_devices.split(",") if item.strip()])
            if count > 0:
                return count

        try:
            capabilities = self.runtime.capabilities()
        except Exception:
            return 1
        return max(int(getattr(capabilities, "device_count", 1) or 1), 1)

    def ensure_resource_budget(self, req: GenerateRequest) -> None:
        minimum = self.model_spec.resource_hint.get("min_vram_gb")
        if minimum is None:
            return

        available = self.runtime.available_memory_gb()
        if available is None:
            return

        estimated_gb = float(minimum)
        available_gb = float(available)
        if self._memory_scope() == "aggregate":
            available_gb *= float(self._parallel_device_count(req))

        if available_gb >= estimated_gb:
            return

        raise InsufficientMemoryError(
            model=self.model_spec.id,
            estimated_gb=estimated_gb,
            available_gb=available_gb,
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
        apply_quantization_runtime(pipeline, config=config)
        apply_tea_cache_runtime(pipeline, config=config)

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

    def inject_cached_prompt_embeddings(self, pipeline: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        cache = self._active_result_cache
        req = self._active_request
        if cache is None or req is None or not req.config.get("use_result_cache", True):
            return kwargs
        if req.inputs.get("prompt") in (None, "") or not hasattr(pipeline, "encode_prompt"):
            return kwargs

        cached = cache.lookup_embeddings(req)
        if cached is None:
            bundle = self._encode_prompt_bundle(pipeline, req=req, kwargs=kwargs)
            if bundle:
                cache.save_embeddings(req, bundle)
                return self._inject_prompt_bundle(dict(kwargs), bundle)
            return kwargs

        if "text_embedding" not in self._active_cache_hits:
            self._active_cache_hits.append("text_embedding")
        return self._inject_prompt_bundle(dict(kwargs), cached)

    def _encode_prompt_bundle(self, pipeline: Any, *, req: GenerateRequest, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            encode_signature = inspect.signature(pipeline.encode_prompt)
        except (TypeError, ValueError, AttributeError):
            return {}

        encode_kwargs = {
            "prompt": req.inputs.get("prompt"),
            "negative_prompt": req.inputs.get("negative_prompt"),
            "num_images_per_prompt": kwargs.get("num_images_per_prompt"),
            "max_sequence_length": kwargs.get("max_sequence_length"),
            "device": getattr(self.runtime, "device_name", None),
            "do_classifier_free_guidance": bool((kwargs.get("guidance_scale") or 0) > 1),
        }
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in encode_signature.parameters.values()):
            encode_kwargs = {k: v for k, v in encode_kwargs.items() if v is not None and k in encode_signature.parameters}
        else:
            encode_kwargs = {k: v for k, v in encode_kwargs.items() if v is not None}

        try:
            encoded = pipeline.encode_prompt(**encode_kwargs)
        except Exception:
            return {}

        if isinstance(encoded, dict):
            return {key: value for key, value in encoded.items() if key.endswith("embeds")}
        if isinstance(encoded, tuple):
            names_by_len = {
                2: ("prompt_embeds", "negative_prompt_embeds"),
                4: (
                    "prompt_embeds",
                    "negative_prompt_embeds",
                    "pooled_prompt_embeds",
                    "negative_pooled_prompt_embeds",
                ),
            }
            names = names_by_len.get(len(encoded))
            if names is not None:
                return {name: value for name, value in zip(names, encoded)}
        bundle = {}
        for name in (
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ):
            value = getattr(encoded, name, None)
            if value is not None:
                bundle[name] = value
        return bundle

    def _inject_prompt_bundle(self, kwargs: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
        kwargs.update(bundle)
        if "prompt_embeds" in bundle:
            kwargs.pop("prompt", None)
        if "negative_prompt_embeds" in bundle:
            kwargs.pop("negative_prompt", None)
        return kwargs

    def run(self, req: GenerateRequest, *, result_cache=None) -> GenerateResult:
        run_id = str(uuid.uuid4())
        timings: Dict[str, float] = {}
        outputs: List[Artifact] = []
        self.runtime.backend_timeline = []
        self.runtime.reset_memory_stats()
        self._captured_latent = None
        self._active_request = req
        self._active_result_cache = result_cache
        self._active_cache_hits = []
        self.ensure_resource_budget(req)

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
            for key in DEVICE_MAP_CONFIG_KEYS:
                if key in req.config:
                    resolved_config[key] = req.config[key]
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
                cache_hits=self._active_cache_hits,
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
                cache_hits=self._active_cache_hits,
            )
            self.last_report = report
            self.logger.error(
                "run.failed",
                extra={"run_id": run_id, "model": req.model, "backend": self.runtime.name, "error": report.error},
            )
            raise
        finally:
            self._active_request = None
            self._active_result_cache = None
