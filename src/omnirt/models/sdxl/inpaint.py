"""SDXL inpainting pipelines backed by Diffusers."""

from __future__ import annotations

import time
from typing import Any, Dict

from omnirt.core.base_pipeline import RESULT_CACHE_CONFIG_KEYS
from omnirt.core.media import load_image, load_mask
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
from omnirt.models.sdxl.pipeline import SDXLPipeline


@register_model(
    id="sdxl-base-1.0",
    task="inpaint",
    default_backend="auto",
    execution_mode="modular",
    modular_pretrained_id="stabilityai/stable-diffusion-xl-base-1.0",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("image", "mask", "prompt"),
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
        )
        + RESULT_CACHE_CONFIG_KEYS,
        default_config={"scheduler": "euler-discrete", "height": 1024, "width": 1024, "strength": 1.0, "dtype": "fp16"},
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="stable",
        summary="SDXL inpainting pipeline with LoRA support.",
        example="omnirt generate --task inpaint --model sdxl-base-1.0 --image input.png --mask mask.png --prompt \"repair the damaged area\" --backend cuda",
    ),
)
class SDXLInpaintPipeline(SDXLPipeline):
    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        image = req.inputs.get("image")
        mask = req.inputs.get("mask")
        if not prompt:
            raise ValueError("inpaint requires inputs.prompt")
        if not image:
            raise ValueError("inpaint requires inputs.image")
        if not mask:
            raise ValueError("inpaint requires inputs.mask")
        source_image = load_image(str(image))
        source_mask = load_mask(str(mask))
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "image": source_image,
            "mask": source_mask,
            "model_source": req.config.get("model_path", self._default_model_source()),
            "scheduler": req.config.get("scheduler", self._default_scheduler()),
            "height": int(req.config.get("height", source_image.height)),
            "width": int(req.config.get("width", source_image.width)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
            "strength": float(req.config.get("strength", 1.0)),
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        kwargs = {
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "image": conditions["image"],
            "mask_image": conditions["mask"],
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
            from diffusers import StableDiffusionXLInpaintPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers is required for SDXL inpainting execution. Install omnirt with runtime dependencies."
            ) from exc
        return StableDiffusionXLInpaintPipeline
