"""Executor backed by Diffusers modular pipelines."""

from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
import time
import uuid
from typing import Any, Dict, Iterable, Optional, Sequence

from omnirt.core.adapters import AdapterManager
from omnirt.core.media import load_image, load_mask, save_video_frames
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateResult, GenerateRequest
from omnirt.executors.base import Executor
from omnirt.executors.events import emit_event
from omnirt.launcher import resolve_config_device_map
from omnirt.middleware.backend_wrapper import BackendWrapperMiddleware
from omnirt.telemetry.report import build_run_report


class ModularExecutor(Executor):
    name = "modular"
    _COMPONENT_ATTRS = (
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "transformer",
        "transformer_2",
        "unet",
        "vae",
    )

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Any = None
        self.components_manager: Any = None
        self._torch_dtype: Any = None
        self._adapter_manager = AdapterManager()
        self._load_backend_timeline: list[Any] = []

    def load(self, *, runtime, model_spec, config, adapters) -> None:
        if self.pipeline is not None:
            return

        modular_pipeline_cls, components_manager_cls = self._diffusers_api()
        self.runtime = runtime
        self.model_spec = model_spec
        self.config = dict(config)
        self.adapters = list(adapters or [])
        self._torch_dtype = self._resolve_torch_dtype(self.config.get("dtype") or model_spec.resource_hint.get("dtype"))
        source = self._resolve_source(self.config, model_spec)
        self.components_manager = components_manager_cls()
        placement_managed = False
        if self.config.get("cpu_offload") and hasattr(self.components_manager, "enable_auto_cpu_offload"):
            self.components_manager.enable_auto_cpu_offload(device=runtime.device_name)
            placement_managed = True
        device_map = resolve_config_device_map(self.config)
        if device_map is not None:
            placement_managed = True

        from_pretrained_kwargs = self._filter_kwargs(
            modular_pipeline_cls.from_pretrained,
            {
                "components_manager": self.components_manager,
                "device_map": device_map,
            },
        )
        self.pipeline = modular_pipeline_cls.from_pretrained(source, **from_pretrained_kwargs)
        if hasattr(self.pipeline, "load_components"):
            load_components_kwargs = self._filter_kwargs(
                self.pipeline.load_components,
                {"torch_dtype": self._torch_dtype},
            )
            self.pipeline.load_components(**load_components_kwargs)
        self.pipeline = runtime.prepare_pipeline(self.pipeline, model_spec=model_spec, config=self.config)

        self.components = {}
        for name in self._COMPONENT_ATTRS:
            component = getattr(self.pipeline, name, None)
            if component is not None:
                self.components[name] = component
        self.apply_middleware([BackendWrapperMiddleware()])
        for name, component in self.components.items():
            setattr(self.pipeline, name, component)
        self._load_backend_timeline = list(getattr(runtime, "backend_timeline", []))
        if self.adapters:
            self._adapter_manager.load_all(self.adapters)
            self._adapter_manager.apply_to_pipeline(self.pipeline)

        if not placement_managed:
            try:
                self.pipeline = runtime.to_device(self.pipeline, dtype=self._torch_dtype)
            except Exception:
                pass

    def run(self, request, *, event_callback=None, cache=None) -> GenerateResult:
        if self.pipeline is None or self.runtime is None or self.model_spec is None:
            raise RuntimeError("ModularExecutor must be loaded before run().")

        runtime = self.runtime
        runtime.reset_memory_stats()
        run_id = str(uuid.uuid4())
        timings: Dict[str, float] = {}
        cache_hits: list[str] = []
        runtime.backend_timeline = list(self._load_backend_timeline)

        kwargs = self._build_call_kwargs(request, event_callback=event_callback)
        if cache is not None and request.config.get("use_result_cache", True):
            kwargs, cache_hit = self._apply_cached_prompt_embeddings(request, kwargs, cache, event_callback=event_callback)
            if cache_hit:
                cache_hits.append("text_embedding")

        emit_event(event_callback, "stage_start", "modular_pipeline", data={"model": request.model})
        emit_event(event_callback, "stage_start", "denoise", data={"model": request.model})
        started = time.perf_counter()
        result = self.pipeline(**self._filter_call_kwargs(kwargs))
        runtime.synchronize()
        timings["pipeline_call_ms"] = round((time.perf_counter() - started) * 1000, 3)
        emit_event(
            event_callback,
            "stage_end",
            "denoise",
            data={"model": request.model, "elapsed_ms": timings["pipeline_call_ms"]},
        )
        emit_event(event_callback, "stage_end", "modular_pipeline", data={"model": request.model})

        export_started = time.perf_counter()
        emit_event(event_callback, "stage_start", "export", data={"model": request.model})
        artifacts = self._export(result, request)
        timings["export_ms"] = round((time.perf_counter() - export_started) * 1000, 3)
        emit_event(
            event_callback,
            "stage_end",
            "export",
            data={"model": request.model, "elapsed_ms": timings["export_ms"]},
        )

        resolved_config = dict(request.config)
        if "model_path" not in resolved_config and self.model_spec.modular_pretrained_id:
            resolved_config["model_path"] = self.model_spec.modular_pretrained_id
        report = build_run_report(
            run_id=run_id,
            request=request,
            backend_name=runtime.name,
            timings=timings,
            memory=runtime.memory_stats(),
            backend_timeline=runtime.backend_timeline,
            config_resolved=resolved_config,
            artifacts=artifacts,
            error=None,
            cache_hits=cache_hits,
            execution_mode=self.name,
        )
        return GenerateResult(outputs=artifacts, metadata=report)

    def release(self) -> None:
        self.components.clear()
        self.pipeline = None
        self.components_manager = None
        self._load_backend_timeline = []

    def _diffusers_api(self):
        try:
            from diffusers import ComponentsManager, ModularPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with ModularPipeline support is required for modular execution."
            ) from exc
        return ModularPipeline, ComponentsManager

    def _resolve_source(self, config: dict[str, Any], model_spec) -> str:
        source = config.get("model_path") or model_spec.modular_pretrained_id
        if not source:
            raise ValueError(f"Model {model_spec.id!r} is missing modular_pretrained_id metadata.")
        return str(source)

    def _resolve_torch_dtype(self, dtype_name: Optional[str]):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required for modular execution.") from exc

        mapping = {
            None: torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        try:
            return mapping[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_name}") from exc

    def _build_generator(self, seed: Optional[int] | Sequence[Optional[int]]):
        if isinstance(seed, (list, tuple)):
            generators = [self._build_generator(item) for item in seed]
            return [generator for generator in generators if generator is not None] or None
        if seed is None:
            return None
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required for modular execution.") from exc
        for device_name in (self.runtime.device_name, "cpu"):
            try:
                return torch.Generator(device=device_name).manual_seed(int(seed))
            except Exception:
                continue
        return torch.Generator().manual_seed(int(seed))

    def _build_call_kwargs(self, request: GenerateRequest, *, event_callback=None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": request.inputs.get("prompt"),
            "negative_prompt": request.inputs.get("negative_prompt"),
            "generator": self._build_generator(request.config.get("seed")),
            "num_inference_steps": request.config.get("num_inference_steps"),
            "guidance_scale": request.config.get("guidance_scale"),
            "height": request.config.get("height"),
            "width": request.config.get("width"),
            "num_images_per_prompt": request.config.get("num_images_per_prompt"),
            "max_sequence_length": request.config.get("max_sequence_length"),
            "caption_upsample_temperature": request.config.get("caption_upsample_temperature"),
            "strength": request.config.get("strength"),
            "num_frames": request.inputs.get("num_frames"),
            "fps": request.inputs.get("fps"),
            "output_type": "pil",
        }
        raw_image = request.inputs.get("image")
        if raw_image is not None:
            loaded_image = self._load_image_input(raw_image)
            if self.model_spec.id in {"flux-depth", "flux-canny"}:
                kwargs["control_image"] = loaded_image
            else:
                kwargs["image"] = loaded_image
        raw_mask = request.inputs.get("mask")
        if raw_mask is not None:
            kwargs["mask_image"] = load_mask(str(raw_mask))

        if self._supports_callback_on_step_end():
            kwargs["callback_on_step_end"] = self._make_progress_callback(event_callback)
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        return kwargs

    def _load_image_input(self, raw_image: Any):
        if isinstance(raw_image, (list, tuple)):
            return [load_image(str(item)) for item in raw_image]
        return load_image(str(raw_image))

    def _make_progress_callback(self, event_callback):
        def _callback(_pipe, step, timestep, callback_kwargs):
            emit_event(
                event_callback,
                "stage_progress",
                "denoise",
                data={"step": int(step), "timestep": int(timestep) if timestep is not None else None},
            )
            return callback_kwargs if isinstance(callback_kwargs, dict) else {}

        return _callback

    def _apply_cached_prompt_embeddings(self, request, kwargs, cache, *, event_callback=None):
        if not hasattr(self.pipeline, "encode_prompt"):
            return kwargs, False
        if request.inputs.get("prompt") in (None, "") or isinstance(request.inputs.get("prompt"), (list, tuple)):
            return kwargs, False

        cached = cache.lookup_embeddings(request)
        if cached is not None:
            emit_event(event_callback, "cache_hit", "encode_prompt", data={"cache_type": "text_embedding"})
            return self._inject_prompt_bundle(dict(kwargs), cached), True

        emit_event(event_callback, "stage_start", "encode_prompt", data={"model": request.model})
        started = time.perf_counter()
        bundle = self._encode_prompt_bundle(request, kwargs)
        emit_event(
            event_callback,
            "stage_end",
            "encode_prompt",
            data={"model": request.model, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        if bundle:
            cache.save_embeddings(request, bundle)
            kwargs = self._inject_prompt_bundle(dict(kwargs), bundle)
        return kwargs, False

    def _encode_prompt_bundle(self, request, kwargs) -> dict[str, Any]:
        encode_kwargs = self._filter_kwargs(
            self.pipeline.encode_prompt,
            {
                "prompt": request.inputs.get("prompt"),
                "negative_prompt": request.inputs.get("negative_prompt"),
                "num_images_per_prompt": kwargs.get("num_images_per_prompt"),
                "max_sequence_length": kwargs.get("max_sequence_length"),
                "device": getattr(self.runtime, "device_name", None),
            },
        )
        encoded = self.pipeline.encode_prompt(**encode_kwargs)
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

    def _inject_prompt_bundle(self, kwargs: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
        kwargs.update(bundle)
        if "prompt_embeds" in bundle:
            kwargs.pop("prompt", None)
        if "negative_prompt_embeds" in bundle:
            kwargs.pop("negative_prompt", None)
        return kwargs

    def _supports_callback_on_step_end(self) -> bool:
        try:
            parameters = signature(self.pipeline.__call__).parameters
        except (TypeError, ValueError, AttributeError):
            return False
        if "callback_on_step_end" in parameters:
            return True
        return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())

    def _filter_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return self._filter_kwargs(self.pipeline.__call__, kwargs)

    def _filter_kwargs(self, callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            parameters = signature(callable_obj).parameters
        except (TypeError, ValueError, AttributeError):
            return {key: value for key, value in kwargs.items() if value is not None}
        if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return {key: value for key, value in kwargs.items() if value is not None}
        return {
            key: value
            for key, value in kwargs.items()
            if value is not None and key in parameters
        }

    def _export(self, result: Any, request: GenerateRequest) -> list[Artifact]:
        output_dir = Path(request.config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_value = request.config.get("seed", "random")
        seed_parts = seed_value if isinstance(seed_value, (list, tuple)) else None

        images = getattr(result, "images", None)
        if images is not None:
            artifacts: list[Artifact] = []
            for index, image in enumerate(list(images)):
                seed_part = seed_parts[index] if seed_parts and index < len(seed_parts) else seed_value
                file_path = output_dir / f"{request.model}-{seed_part}-{index}.png"
                image.save(file_path)
                artifacts.append(
                    Artifact(
                        kind="image",
                        path=str(file_path),
                        mime="image/png",
                        width=image.width,
                        height=image.height,
                    )
                )
            return artifacts

        frames = getattr(result, "frames", None)
        if frames is None:
            raise ValueError(f"Unexpected modular pipeline output for model {request.model!r}")
        sequence = list(frames[0] if frames and isinstance(frames[0], list) else frames)
        file_path = output_dir / f"{request.model}-{seed_value}.mp4"
        save_video_frames(file_path, sequence, fps=int(request.inputs.get("fps", 16)))
        first_frame = sequence[0]
        return [
            Artifact(
                kind="video",
                path=str(file_path),
                mime="video/mp4",
                width=first_frame.width,
                height=first_frame.height,
                num_frames=len(sequence),
            )
        ]
