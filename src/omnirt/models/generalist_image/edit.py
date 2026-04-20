"""Modern image editing family implementation backed by Diffusers."""

from __future__ import annotations

import time
from typing import Any, Dict

from omnirt.backends.overrides import ASCEND_ACCELERATION_CONFIG_KEYS
from omnirt.core.media import load_image
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import GenerateRequest
from omnirt.models.generalist_image.components import EDIT_MODEL_CONFIGS, GeneralistImageModelConfig
from omnirt.models.generalist_image.pipeline import GeneralistImagePipeline


class GeneralistImageEditPipeline(GeneralistImagePipeline):
    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        if prompt is None and self.model_spec.id != "qwen-image-layered":
            raise ValueError("edit requires inputs.prompt")
        raw_image = req.inputs.get("image")
        if not raw_image:
            raise ValueError("edit requires inputs.image")
        images = self._load_images(raw_image)
        first_image = images[0] if isinstance(images, list) else images
        config = self._model_config()
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "image": images,
            "model_source": req.config.get("model_path", config.source),
            "scheduler": req.config.get("scheduler", "native"),
            "height": int(req.config.get("height", first_image.height)),
            "width": int(req.config.get("width", first_image.width)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
            "max_sequence_length": req.config.get("max_sequence_length"),
            "extra_call_kwargs": {
                key: req.config.get(key, config.default_config.get(key))
                for key in config.call_config_keys
                if req.config.get(key, config.default_config.get(key)) is not None
            },
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        kwargs: Dict[str, Any] = {
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "image": conditions["image"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "generator": latents["generator"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_images_per_prompt": conditions["num_images_per_prompt"],
            "max_sequence_length": conditions.get("max_sequence_length"),
            "output_type": "pil",
        }
        kwargs.update(conditions.get("extra_call_kwargs", {}))
        if self._supports_callback_on_step_end(pipeline):
            kwargs["callback_on_step_end"] = self.make_latent_callback(latents["steps"])
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        result = pipeline(**self._filter_call_kwargs(pipeline, kwargs))
        images = getattr(result, "images", None)
        if images is None and isinstance(result, tuple):
            images = result[0]
        if images is None:
            raise ValueError(f"Unexpected image edit pipeline output for model {self.model_spec.id!r}")
        if len(images) == 1 and isinstance(images[0], (list, tuple)):
            images = images[0]
        return {
            "images": list(images),
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _model_config(self) -> GeneralistImageModelConfig:
        return EDIT_MODEL_CONFIGS[self.model_spec.id]

    def _load_images(self, raw_image: Any):
        if isinstance(raw_image, (list, tuple)):
            return [load_image(str(path)) for path in raw_image]
        return load_image(str(raw_image))


for model_id, model_config in EDIT_MODEL_CONFIGS.items():
    register_model(
        id=model_id,
        task="edit",
        default_backend="auto",
        resource_hint=model_config.resource_hint,
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
                "seed",
                "dtype",
                "output_dir",
                "max_sequence_length",
                "true_cfg_scale",
                "layers",
                "resolution",
                "cfg_normalize",
                "use_en_prompt",
            )
            + ASCEND_ACCELERATION_CONFIG_KEYS,
            default_config=model_config.default_config,
            supported_schedulers=("native",),
            adapter_kinds=("lora",),
            artifact_kind="image",
            maturity="beta",
            summary=model_config.summary,
            example=model_config.example,
        ),
    )(GeneralistImageEditPipeline)
