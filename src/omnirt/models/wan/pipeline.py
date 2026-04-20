"""Wan 2.2 pipeline implementations backed by Diffusers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.media import load_image, save_video_frames
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.wan.components import DEFAULT_WAN2_2_I2V_MODEL_SOURCE, DEFAULT_WAN2_2_T2V_MODEL_SOURCE


@register_model(
    id="wan2.2-t2v-14b",
    task="text2video",
    default_backend="auto",
    resource_hint={"min_vram_gb": 20, "dtype": "bf16"},
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("negative_prompt", "num_frames", "fps"),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        ),
        default_config={"scheduler": "native", "num_inference_steps": 50, "guidance_scale": 5.0, "dtype": "bf16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="video",
        maturity="beta",
        summary="Wan 2.2 text-to-video pipeline.",
        example="omnirt generate --task text2video --model wan2.2-t2v-14b --prompt \"a glass whale gliding over a moonlit harbor\" --backend cuda",
    ),
)
@register_model(
    id="wan2.2-i2v-14b",
    task="image2video",
    default_backend="auto",
    resource_hint={"min_vram_gb": 20, "dtype": "bf16"},
    capabilities=ModelCapabilities(
        required_inputs=("image", "prompt"),
        optional_inputs=("negative_prompt", "num_frames", "fps"),
        supported_config=(
            "model_path",
            "scheduler",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        ),
        default_config={"scheduler": "native", "num_inference_steps": 50, "guidance_scale": 5.0, "dtype": "bf16"},
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="video",
        maturity="beta",
        summary="Wan 2.2 image-to-video pipeline.",
        example="omnirt generate --task image2video --model wan2.2-i2v-14b --image input.png --prompt \"turn this sketch into a moving cityscape\" --backend cuda",
    ),
)
class WanPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_key: Optional[tuple] = None
        self._last_seed = None
        self._last_fps = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        if not prompt:
            raise ValueError(f"{self.model_spec.task} requires inputs.prompt")

        conditions: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "num_frames": int(req.inputs.get("num_frames", 81)),
            "fps": int(req.inputs.get("fps", 16)),
            "height": int(req.config["height"]) if "height" in req.config else None,
            "width": int(req.config["width"]) if "width" in req.config else None,
            "model_source": req.config.get("model_path", self._default_model_source()),
            "scheduler": req.config.get("scheduler", "native"),
        }
        if self.model_spec.task == "image2video":
            image = req.inputs.get("image")
            if not image:
                raise ValueError("image2video requires inputs.image")
            if not Path(image).exists():
                raise FileNotFoundError(image)
            conditions["image"] = load_image(image)
        return conditions

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", 50))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", 5.0))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype", "bf16"))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(
            source=conditions["model_source"],
            torch_dtype=torch_dtype,
            scheduler_name=conditions["scheduler"],
        )
        self._last_seed = seed
        self._last_fps = conditions["fps"]
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
            "num_frames": conditions["num_frames"],
            "guidance_scale": latents["guidance_scale"],
            "num_inference_steps": latents["steps"],
            "generator": latents["generator"],
            "output_type": "pil",
        }
        if conditions.get("negative_prompt"):
            kwargs["negative_prompt"] = conditions["negative_prompt"]
        if conditions.get("height") is not None:
            kwargs["height"] = conditions["height"]
        if conditions.get("width") is not None:
            kwargs["width"] = conditions["width"]
        if "image" in conditions:
            kwargs["image"] = conditions["image"]
        result = pipeline(**kwargs)
        frames = list(result.frames[0] if result.frames and isinstance(result.frames[0], list) else result.frames)
        return {
            "frames": frames,
            "fps": conditions["fps"],
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def decode(self, latents: Any) -> Any:
        return latents["frames"]

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        output_dir = self.resolve_output_dir(req)
        seed_part = self._last_seed if self._last_seed is not None else "random"
        file_path = output_dir / f"{req.model}-{seed_part}.mp4"
        save_video_frames(file_path, raw, fps=self._last_fps or int(req.inputs.get("fps", 16)))
        first_frame = raw[0]
        return [
            Artifact(
                kind="video",
                path=str(file_path),
                mime="video/mp4",
                width=first_frame.width,
                height=first_frame.height,
                num_frames=len(raw),
            )
        ]

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: Any) -> Dict[str, Any]:
        return {
            "model_path": conditions["model_source"],
            "scheduler": conditions["scheduler"],
            "num_frames": conditions["num_frames"],
            "fps": conditions["fps"],
            "height": conditions["height"],
            "width": conditions["width"],
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
            raise DependencyUnavailableError("PyTorch is required to run the Wan pipeline.") from exc
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
            if self.model_spec.task == "image2video":
                from diffusers import WanImageToVideoPipeline as PipelineCls
            else:
                from diffusers import WanPipeline as PipelineCls
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with Wan support is required for Wan execution. Install omnirt with runtime dependencies."
            ) from exc
        return PipelineCls

    def _load_pipeline(self, *, source: str, torch_dtype: Any, scheduler_name: str):
        cache_key = self.pipeline_cache_key(
            source=source, torch_dtype=torch_dtype, scheduler_name=scheduler_name
        )
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(source, torch_dtype=torch_dtype)
        if scheduler_name != "native":
            raise ValueError(f"Unsupported Wan scheduler: {scheduler_name}")
        self._wrap_pipeline_modules(pipeline)
        pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("text_encoder", "image_encoder", "transformer", "transformer_2", "vae"):
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
        self.adapter_manager.apply_to_pipeline(pipeline)

    def _default_model_source(self) -> str:
        if self.model_spec.task == "image2video":
            return DEFAULT_WAN2_2_I2V_MODEL_SOURCE
        return DEFAULT_WAN2_2_T2V_MODEL_SOURCE
