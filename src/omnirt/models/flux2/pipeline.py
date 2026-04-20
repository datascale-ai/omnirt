"""Flux2 pipeline implementation backed by Diffusers."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.flux2.components import DEFAULT_FLUX2_DEV_MODEL_SOURCE


@register_model(
    id="flux2.dev",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_images_per_prompt",
            "max_sequence_length",
            "caption_upsample_temperature",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        ),
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "max_sequence_length": 512, "dtype": "bf16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Flux 2 dev text-to-image pipeline.",
        example="omnirt generate --task text2image --model flux2.dev --prompt \"a paper dragon in a lantern shop\" --backend cuda",
    ),
)
@register_model(
    id="flux2-dev",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_images_per_prompt",
            "max_sequence_length",
            "caption_upsample_temperature",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        ),
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "max_sequence_length": 512, "dtype": "bf16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Alias for flux2.dev.",
        example="omnirt generate --task text2image --model flux2.dev --prompt \"a paper dragon in a lantern shop\" --backend cuda",
        alias_of="flux2.dev",
    ),
)
class Flux2Pipeline(BasePipeline):
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
            "model_source": req.config.get("model_path", DEFAULT_FLUX2_DEV_MODEL_SOURCE),
            "scheduler": req.config.get("scheduler", "native"),
            "height": int(req.config.get("height", 1024)),
            "width": int(req.config.get("width", 1024)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
            "max_sequence_length": int(req.config.get("max_sequence_length", 512)),
            "caption_upsample_temperature": req.config.get("caption_upsample_temperature"),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", 50))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", 2.5))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype", "bf16"))
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
        kwargs = {
            "prompt": conditions["prompt"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "generator": latents["generator"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "max_sequence_length": conditions["max_sequence_length"],
            "output_type": "pil",
        }
        if conditions["caption_upsample_temperature"] is not None:
            kwargs["caption_upsample_temperature"] = float(conditions["caption_upsample_temperature"])
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
            "max_sequence_length": conditions["max_sequence_length"],
            "caption_upsample_temperature": conditions["caption_upsample_temperature"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "seed": latents["seed"],
            "dtype": req.config.get("dtype", "bf16"),
            "output_dir": str(Path(req.config.get("output_dir", "outputs"))),
        }

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required to run the Flux2 pipeline.") from exc
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
            from diffusers import Flux2Pipeline as DiffusersFlux2Pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with Flux2 support is required for Flux2 execution. Install omnirt with runtime dependencies."
            ) from exc
        return DiffusersFlux2Pipeline

    def _load_pipeline(self, *, source: str, torch_dtype: Any, scheduler_name: str):
        cache_key = self.pipeline_cache_key(
            source=source, torch_dtype=torch_dtype, scheduler_name=scheduler_name
        )
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(source, torch_dtype=torch_dtype)
        if scheduler_name != "native":
            raise ValueError(f"Unsupported Flux2 scheduler: {scheduler_name}")
        self._wrap_pipeline_modules(pipeline)
        pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("text_encoder", "transformer", "vae"):
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
        self.adapter_manager.apply_to_pipeline(pipeline)
