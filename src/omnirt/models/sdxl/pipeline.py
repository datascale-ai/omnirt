"""SDXL pipeline implementation backed by Diffusers."""

from __future__ import annotations

import inspect
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.sdxl.components import DEFAULT_SDXL_MODEL_SOURCE, DEFAULT_SDXL_TURBO_MODEL_SOURCE
from omnirt.schedulers import build_scheduler


@register_model(
    id="sdxl-base-1.0",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
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
        ),
        default_config={"scheduler": "euler-discrete", "height": 1024, "width": 1024, "dtype": "fp16"},
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="stable",
        summary="SDXL base text-to-image pipeline with LoRA support.",
        example="omnirt generate --task text2image --model sdxl-base-1.0 --prompt \"a cinematic sci-fi city at sunrise\" --backend cuda",
    ),
)
@register_model(
    id="sdxl-turbo",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 10, "dtype": "fp16"},
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
        ),
        default_config={"scheduler": "euler-ancestral", "height": 1024, "width": 1024, "dtype": "fp16"},
        supported_schedulers=("euler-discrete", "euler-ancestral", "ddim", "dpm-solver"),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="SDXL Turbo low-latency text-to-image pipeline.",
        example="omnirt generate --task text2image --model sdxl-turbo --prompt \"a cinematic sci-fi city at sunrise\" --backend cuda",
    ),
)
class SDXLPipeline(BasePipeline):
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
            "scheduler": req.config.get("scheduler", self._default_scheduler()),
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
        result = pipeline(
            prompt=conditions["prompt"],
            negative_prompt=conditions.get("negative_prompt"),
            num_inference_steps=latents["steps"],
            guidance_scale=latents["guidance_scale"],
            generator=latents["generator"],
            height=conditions["height"],
            width=conditions["width"],
            num_images_per_prompt=conditions["num_images_per_prompt"],
            output_type="pil",
            **callback_kwargs,
        )
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
            raise DependencyUnavailableError("PyTorch is required to run the SDXL pipeline.") from exc
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
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers is required for SDXL execution. Install omnirt with runtime dependencies."
            ) from exc
        return StableDiffusionXLPipeline

    def _load_pipeline(
        self,
        *,
        source: str,
        torch_dtype: Any,
        scheduler_name: str,
        config: Dict[str, Any],
    ):
        cache_key = self.pipeline_cache_key(
            source=source, torch_dtype=torch_dtype, scheduler_name=scheduler_name
        )
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(source, **self._from_pretrained_kwargs(source, torch_dtype))
        scheduler_config = dict(config)
        scheduler_config.setdefault("scheduler", scheduler_name)
        if getattr(pipeline, "scheduler", None) is not None and hasattr(pipeline.scheduler, "config"):
            scheduler_config["scheduler_config"] = pipeline.scheduler.config
        pipeline.scheduler = build_scheduler(scheduler_config)
        self._wrap_pipeline_modules(pipeline)
        pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _from_pretrained_kwargs(self, source: str, torch_dtype: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        variant = self._detect_local_variant(source)
        if variant is not None:
            kwargs["variant"] = variant
        return kwargs

    def _detect_local_variant(self, source: str) -> Optional[str]:
        root = Path(source)
        if not root.is_dir():
            return None

        fp16_layout = {
            "unet": "diffusion_pytorch_model.fp16.safetensors",
            "vae": "diffusion_pytorch_model.fp16.safetensors",
            "text_encoder": "model.fp16.safetensors",
            "text_encoder_2": "model.fp16.safetensors",
        }
        if all((root / subdir / filename).is_file() for subdir, filename in fp16_layout.items()):
            return "fp16"
        return None

    def _default_model_source(self) -> str:
        if self.model_spec.id == "sdxl-turbo":
            return DEFAULT_SDXL_TURBO_MODEL_SOURCE
        return DEFAULT_SDXL_MODEL_SOURCE

    def _default_scheduler(self) -> str:
        if self.model_spec.id == "sdxl-turbo":
            return "euler-ancestral"
        return "euler-discrete"

    def _default_steps(self) -> int:
        if self.model_spec.id == "sdxl-turbo":
            return 4
        return 30

    def _default_guidance_scale(self) -> float:
        if self.model_spec.id == "sdxl-turbo":
            return 0.0
        return 7.5

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("text_encoder", "text_encoder_2", "unet", "vae"):
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
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
