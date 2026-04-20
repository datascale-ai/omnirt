"""Stable Diffusion image-to-image pipelines backed by Diffusers."""

from __future__ import annotations

import time
from typing import Any, Dict

from omnirt.core.media import load_image
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
from omnirt.models.sd15.pipeline import SD15Pipeline


@register_model(
    id="sd15",
    task="image2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 6, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("image", "prompt"),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_images_per_prompt",
            "num_inference_steps",
            "guidance_scale",
            "strength",
            "seed",
            "dtype",
            "output_dir",
        ),
        default_config={"scheduler": "euler-discrete", "height": 512, "width": 512, "strength": 0.8, "dtype": "fp16"},
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Stable Diffusion 1.5 image-to-image pipeline.",
        example="omnirt generate --task image2image --model sd15 --image input.png --prompt \"oil painting\" --backend cuda",
    ),
)
@register_model(
    id="sd21",
    task="image2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 8, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("image", "prompt"),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_images_per_prompt",
            "num_inference_steps",
            "guidance_scale",
            "strength",
            "seed",
            "dtype",
            "output_dir",
        ),
        default_config={"scheduler": "euler-discrete", "height": 768, "width": 768, "strength": 0.8, "dtype": "fp16"},
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Stable Diffusion 2.1 image-to-image pipeline.",
        example="omnirt generate --task image2image --model sd21 --image input.png --prompt \"oil painting\" --backend cuda",
    ),
)
class SD15ImageToImagePipeline(SD15Pipeline):
    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        image = req.inputs.get("image")
        if not prompt:
            raise ValueError("image2image requires inputs.prompt")
        if not image:
            raise ValueError("image2image requires inputs.image")
        source_image = load_image(str(image))
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "image": source_image,
            "model_source": req.config.get("model_path", self._default_model_source()),
            "scheduler": req.config.get("scheduler", "euler-discrete"),
            "height": int(req.config.get("height", source_image.height)),
            "width": int(req.config.get("width", source_image.width)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
            "strength": float(req.config.get("strength", 0.8)),
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        kwargs = {
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "image": conditions["image"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "strength": conditions["strength"],
            "generator": latents["generator"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "output_type": "pil",
        }
        if self._supports_callback_on_step_end(pipeline):
            kwargs["callback_on_step_end"] = self.make_latent_callback(latents["steps"])
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        result = pipeline(**self._filter_call_kwargs(pipeline, kwargs))
        return {
            "images": list(result.images),
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: Any) -> Dict[str, Any]:
        resolved = super().resolve_run_config(req, conditions, latents)
        resolved["strength"] = conditions["strength"]
        return resolved

    def _diffusers_pipeline_cls(self):
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers is required for SD1.5 image-to-image execution. Install omnirt with runtime dependencies."
            ) from exc
        return StableDiffusionImg2ImgPipeline
