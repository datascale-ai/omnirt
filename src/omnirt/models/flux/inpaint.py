"""Flux fill inpainting pipeline backed by Diffusers."""

from __future__ import annotations

import time
from typing import Any, Dict

from omnirt.backends.overrides import ASCEND_ACCELERATION_CONFIG_KEYS
from omnirt.core.base_pipeline import RESULT_CACHE_CONFIG_KEYS
from omnirt.core.media import load_image, load_mask
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
from omnirt.models.flux.components import DEFAULT_FLUX_FILL_MODEL_SOURCE
from omnirt.models.flux.pipeline import FluxPipeline


@register_model(
    id="flux-fill",
    task="inpaint",
    default_backend="auto",
    execution_mode="modular",
    modular_pretrained_id=DEFAULT_FLUX_FILL_MODEL_SOURCE,
    resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    capabilities=ModelCapabilities(
        required_inputs=("image", "mask", "prompt"),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_images_per_prompt",
            "max_sequence_length",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        )
        + RESULT_CACHE_CONFIG_KEYS
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "max_sequence_length": 512, "dtype": "bf16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="Flux Fill inpainting and outpainting pipeline.",
        example="omnirt generate --task inpaint --model flux-fill --image input.png --mask mask.png --prompt \"replace the missing area with folded paper texture\" --backend cuda",
    ),
)
class FluxFillPipeline(FluxPipeline):
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
            "model_source": req.config.get("model_path", DEFAULT_FLUX_FILL_MODEL_SOURCE),
            "scheduler": req.config.get("scheduler", "native"),
            "height": int(req.config.get("height", source_image.height)),
            "width": int(req.config.get("width", source_image.width)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
            "max_sequence_length": int(req.config.get("max_sequence_length", self._default_max_sequence_length())),
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
            "generator": latents["generator"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "max_sequence_length": conditions["max_sequence_length"],
            "output_type": "pil",
        }
        result = pipeline(**self._filter_call_kwargs(pipeline, kwargs))
        return {
            "images": list(result.images),
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _diffusers_pipeline_cls(self):
        try:
            from diffusers import FluxFillPipeline as DiffusersFluxFillPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with FluxFillPipeline support is required for Flux Fill execution."
            ) from exc
        return DiffusersFluxFillPipeline
