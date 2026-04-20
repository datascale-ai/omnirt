"""Stable Diffusion 3 family pipeline implementation backed by Diffusers."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.backends.overrides import ASCEND_ACCELERATION_CONFIG_KEYS
from omnirt.core.base_pipeline import BasePipeline, LEGACY_OPTIMIZATION_CONFIG_KEYS
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.sd3.components import (
    DEFAULT_SD3_MEDIUM_MODEL_SOURCE,
    DEFAULT_SD35_LARGE_MODEL_SOURCE,
    DEFAULT_SD35_LARGE_TURBO_MODEL_SOURCE,
)


@register_model(
    id="sd3-medium",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 24, "dtype": "fp16"},
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
        )
        + LEGACY_OPTIMIZATION_CONFIG_KEYS
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "fp16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Stable Diffusion 3 Medium text-to-image pipeline.",
        example="omnirt generate --task text2image --model sd3-medium --prompt \"a photo of a cat holding a sign\" --backend cuda",
    ),
)
@register_model(
    id="sd3.5-large",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 24, "dtype": "fp16"},
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
        )
        + LEGACY_OPTIMIZATION_CONFIG_KEYS
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "fp16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Stable Diffusion 3.5 Large text-to-image pipeline.",
        example="omnirt generate --task text2image --model sd3.5-large --prompt \"a photo of a cat holding a sign\" --backend cuda",
    ),
)
@register_model(
    id="sd3.5-large-turbo",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 24, "dtype": "fp16"},
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
        )
        + LEGACY_OPTIMIZATION_CONFIG_KEYS
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "fp16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Stable Diffusion 3.5 Large Turbo text-to-image pipeline.",
        example="omnirt generate --task text2image --model sd3.5-large-turbo --prompt \"a photo of a cat holding a sign\" --backend cuda",
    ),
)
class SD3Pipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_key: Optional[tuple] = None
        self._last_seed = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        if not prompt:
            raise ValueError("text2image requires inputs.prompt")
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "model_source": req.config.get("model_path", self._default_model_source()),
            "scheduler": req.config.get("scheduler", "native"),
            "height": int(req.config.get("height", 1024)),
            "width": int(req.config.get("width", 1024)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", self._default_steps()))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", self._default_guidance_scale()))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype"))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(
            source=conditions["model_source"],
            torch_dtype=torch_dtype,
            scheduler_name=conditions["scheduler"],
            config=req.config,
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
        callback_kwargs = {}
        if self._supports_callback_on_step_end(pipeline):
            callback_kwargs = {
                "callback_on_step_end": self.make_latent_callback(latents["steps"]),
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
        kwargs = {
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "generator": latents["generator"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "output_type": "pil",
            **callback_kwargs,
        }
        kwargs = self.inject_cached_prompt_embeddings(pipeline, kwargs)
        result = pipeline(**kwargs)
        return {
            "images": list(result.images),
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
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "seed": latents["seed"],
            "dtype": req.config.get("dtype", "fp16"),
            "output_dir": str(Path(req.config.get("output_dir", "outputs"))),
        }

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required to run the SD3 pipeline.") from exc
        return torch

    def _resolve_torch_dtype(self, dtype_name: Optional[str]):
        torch = self._torch()
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
            from diffusers import StableDiffusion3Pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with StableDiffusion3Pipeline support is required for SD3 execution."
            ) from exc
        return StableDiffusion3Pipeline

    def _load_pipeline(self, *, source: str, torch_dtype: Any, scheduler_name: str, config: Dict[str, Any]):
        cache_key = self.pipeline_cache_key(source=source, torch_dtype=torch_dtype, scheduler_name=scheduler_name)
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            **self.from_pretrained_runtime_kwargs(config=config),
        )
        if scheduler_name != "native":
            raise ValueError(f"Unsupported SD3 scheduler: {scheduler_name}")
        pipeline = self.runtime.prepare_pipeline(pipeline, model_spec=self.model_spec, config=config)
        self._wrap_pipeline_modules(pipeline)
        pipeline, placement_managed = self.apply_pipeline_optimizations(pipeline, config=config)
        if not placement_managed and not self.uses_managed_device_placement(config):
            pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"):
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
        self.adapter_manager.apply_to_pipeline(pipeline)

    def _default_model_source(self) -> str:
        if self.model_spec.id == "sd3-medium":
            return DEFAULT_SD3_MEDIUM_MODEL_SOURCE
        if self.model_spec.id == "sd3.5-large":
            return DEFAULT_SD35_LARGE_MODEL_SOURCE
        return DEFAULT_SD35_LARGE_TURBO_MODEL_SOURCE

    def _default_steps(self) -> int:
        if self.model_spec.id == "sd3.5-large-turbo":
            return 8
        return 28

    def _default_guidance_scale(self) -> float:
        if self.model_spec.id == "sd3.5-large-turbo":
            return 4.0
        return 7.0
