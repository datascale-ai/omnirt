"""Generalist text-to-image family implementation backed by Diffusers."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.generalist_image.components import MODEL_CONFIGS, GeneralistImageModelConfig

class GeneralistImagePipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_key: Optional[tuple] = None
        self._last_seed = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        if not prompt:
            raise ValueError("text2image requires inputs.prompt")
        config = self._model_config()
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "model_source": req.config.get("model_path", config.source),
            "scheduler": req.config.get("scheduler", "native"),
            "height": int(req.config.get("height", config.default_config.get("height", 1024))),
            "width": int(req.config.get("width", config.default_config.get("width", 1024))),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
            "max_sequence_length": req.config.get("max_sequence_length"),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", self._default_steps()))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", self._default_guidance_scale()))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype", self._model_config().default_config.get("dtype")))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(
            source=conditions["model_source"],
            torch_dtype=torch_dtype,
            scheduler_name=conditions["scheduler"],
        )
        self._last_seed = seed
        return {
            "steps": steps,
            "seed": seed,
            "generator": generator,
            "guidance_scale": guidance_scale,
            "conditions": conditions,
            "pipeline": pipeline,
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        kwargs: Dict[str, Any] = {
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "generator": latents["generator"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "max_sequence_length": conditions.get("max_sequence_length"),
            "output_type": "pil",
        }
        callback_kwargs = {}
        if self._supports_callback_on_step_end(pipeline):
            callback_kwargs = {
                "callback_on_step_end": self.make_latent_callback(latents["steps"]),
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
        kwargs.update(callback_kwargs)
        filtered_kwargs = self._filter_call_kwargs(pipeline, kwargs)
        self._apply_runtime_overrides(pipeline, filtered_kwargs)
        result = pipeline(**filtered_kwargs)
        images = getattr(result, "images", None)
        if images is None and isinstance(result, tuple):
            images = result[0]
        if images is None:
            raise ValueError(f"Unexpected image pipeline output for model {self.model_spec.id!r}")
        return {
            "images": list(images),
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def decode(self, latents: Any) -> Any:
        return latents["images"]

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        output_dir = self.resolve_output_dir(req)
        artifacts: List[Artifact] = []
        for index, image in enumerate(raw):
            seed_part = self._last_seed if self._last_seed is not None else "random"
            file_path = output_dir / f"{req.model}-{seed_part}-{index}.png"
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

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: Any) -> Dict[str, Any]:
        return {
            "model_path": conditions["model_source"],
            "scheduler": conditions["scheduler"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "max_sequence_length": conditions.get("max_sequence_length"),
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "seed": latents["seed"],
            "dtype": req.config.get("dtype", self._model_config().default_config.get("dtype")),
            "output_dir": str(Path(req.config.get("output_dir", "outputs"))),
        }

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required to run image pipelines.") from exc
        return torch

    def _resolve_torch_dtype(self, dtype_name: Optional[str]):
        torch = self._torch()
        mapping = {
            None: torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        try:
            return mapping[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_name}") from exc

    def _build_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        torch = self._torch()
        for device_name in (self.runtime.device_name, "cpu"):
            try:
                return torch.Generator(device=device_name).manual_seed(int(seed))
            except Exception:
                continue
        return torch.Generator().manual_seed(int(seed))

    def _diffusers_pipeline_cls(self):
        try:
            diffusers = importlib.import_module("diffusers")
        except ImportError as exc:
            raise DependencyUnavailableError("diffusers is required for modern image-family execution.") from exc

        for candidate in self._model_config().class_candidates:
            pipeline_cls = getattr(diffusers, candidate, None)
            if pipeline_cls is not None:
                return pipeline_cls
        candidates = ", ".join(self._model_config().class_candidates)
        raise DependencyUnavailableError(f"diffusers does not provide any of the required classes: {candidates}")

    def _load_pipeline(self, *, source: str, torch_dtype: Any, scheduler_name: str):
        cache_key = self.pipeline_cache_key(source=source, torch_dtype=torch_dtype, scheduler_name=scheduler_name)
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(source, torch_dtype=torch_dtype)
        if scheduler_name != "native":
            raise ValueError(f"Unsupported scheduler for model {self.model_spec.id!r}: {scheduler_name}")
        self._wrap_pipeline_modules(pipeline)
        pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in self._model_config().module_tags:
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
        if self.adapters:
            self.adapter_manager.apply_to_pipeline(pipeline)

    def _filter_call_kwargs(self, pipeline: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(pipeline.__call__)
        except (TypeError, ValueError, AttributeError):
            return {key: value for key, value in kwargs.items() if value is not None}

        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return {key: value for key, value in kwargs.items() if value is not None}
        return {
            key: value
            for key, value in kwargs.items()
            if value is not None and key in parameters
        }

    def _apply_runtime_overrides(self, pipeline: Any, kwargs: Dict[str, Any]) -> None:
        guidance_scale = kwargs.get("guidance_scale")
        guider = getattr(pipeline, "guider", None)
        if guidance_scale is not None and guider is not None and hasattr(guider, "guidance_scale"):
            try:
                guider.guidance_scale = guidance_scale
            except Exception:
                pass

    def _model_config(self) -> GeneralistImageModelConfig:
        return MODEL_CONFIGS[self.model_spec.id]

    def _default_steps(self) -> int:
        if self.model_spec.id == "sana-1.6b":
            return 20
        return 30

    def _default_guidance_scale(self) -> float:
        if self.model_spec.id == "sana-1.6b":
            return 4.5
        if self.model_spec.id == "hidream-i1":
            return 5.0
        return 4.0


for model_id, model_config in MODEL_CONFIGS.items():
    register_model(
        id=model_id,
        task="text2image",
        default_backend="auto",
        resource_hint=model_config.resource_hint,
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            optional_inputs=("negative_prompt",),
            supported_config=(
                "model_path",
                "scheduler",
                "height",
                "width",
                "num_images_per_prompt",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "dtype",
                "output_dir",
                "max_sequence_length",
            ),
            default_config=model_config.default_config,
            supported_schedulers=("native",),
            adapter_kinds=("lora",),
            artifact_kind="image",
            maturity="beta",
            summary=model_config.summary,
            example=model_config.example,
        ),
    )(GeneralistImagePipeline)
