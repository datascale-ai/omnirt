"""AnimateDiff SDXL text-to-video pipeline backed by Diffusers."""

from __future__ import annotations

import inspect
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.backends.overrides import ASCEND_ACCELERATION_CONFIG_KEYS
from omnirt.core.base_pipeline import BasePipeline, LEGACY_OPTIMIZATION_CONFIG_KEYS
from omnirt.core.media import save_video_frames
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.sdxl.components import DEFAULT_SDXL_MODEL_SOURCE


DEFAULT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE = "guoyww/animatediff-motion-adapter-sdxl-beta"


@register_model(
    id="animate-diff-sdxl",
    task="text2video",
    default_backend="auto",
    resource_hint={"min_vram_gb": 20, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("negative_prompt", "num_frames", "fps"),
        supported_config=(
            "model_path",
            "motion_adapter_path",
            "scheduler",
            "height",
            "width",
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
        artifact_kind="video",
        maturity="beta",
        summary="AnimateDiff SDXL text-to-video pipeline.",
        example="omnirt generate --task text2video --model animate-diff-sdxl --prompt \"a cinematic portrait with wind in the hair\" --backend cuda",
    ),
)
class AnimateDiffSDXLPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_key: Optional[tuple] = None
        self._last_seed = None
        self._last_fps = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        if not prompt:
            raise ValueError("text2video requires inputs.prompt")
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "num_frames": int(req.inputs.get("num_frames", 16)),
            "fps": int(req.inputs.get("fps", 8)),
            "model_source": req.config.get("model_path", DEFAULT_SDXL_MODEL_SOURCE),
            "motion_adapter_source": req.config.get("motion_adapter_path", DEFAULT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE),
            "scheduler": req.config.get("scheduler", "native"),
            "height": int(req.config.get("height", 1024)),
            "width": int(req.config.get("width", 1024)),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", 16))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", 7.5))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype", "fp16"))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(
            source=conditions["model_source"],
            motion_adapter_source=conditions["motion_adapter_source"],
            torch_dtype=torch_dtype,
            scheduler_name=conditions["scheduler"],
            config=req.config,
        )
        self._last_seed = seed
        self._last_fps = conditions["fps"]
        return {
            "steps": steps,
            "seed": seed,
            "generator": generator,
            "guidance_scale": guidance_scale,
            "pipeline": pipeline,
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        kwargs = {
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "num_frames": conditions["num_frames"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "generator": latents["generator"],
            "output_type": "pil",
        }
        if self._supports_callback_on_step_end(pipeline):
            kwargs["callback_on_step_end"] = self.make_latent_callback(latents["steps"])
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        result = pipeline(**self._filter_call_kwargs(pipeline, kwargs))
        frames = getattr(result, "frames", None)
        if frames is None and isinstance(result, tuple):
            frames = result[0]
        if frames is None:
            raise ValueError("Unexpected AnimateDiff SDXL pipeline output.")
        sequence = list(frames[0] if frames and isinstance(frames[0], list) else frames)
        return {
            "frames": sequence,
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def decode(self, latents: Any) -> Any:
        return latents["frames"]

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        output_dir = self.resolve_output_dir(req)
        seed_part = self._last_seed if self._last_seed is not None else "random"
        file_path = output_dir / f"{req.model}-{seed_part}.mp4"
        save_video_frames(file_path, raw, fps=self._last_fps or int(req.inputs.get("fps", 8)))
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
            "motion_adapter_path": conditions["motion_adapter_source"],
            "scheduler": conditions["scheduler"],
            "num_frames": conditions["num_frames"],
            "fps": conditions["fps"],
            "height": conditions["height"],
            "width": conditions["width"],
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
            raise DependencyUnavailableError("PyTorch is required to run AnimateDiff SDXL.") from exc
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
            from diffusers import AnimateDiffSDXLPipeline as DiffusersAnimateDiffSDXLPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with AnimateDiffSDXLPipeline support is required for AnimateDiff SDXL execution."
            ) from exc
        return DiffusersAnimateDiffSDXLPipeline

    def _motion_adapter_cls(self):
        try:
            from diffusers.models import MotionAdapter
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers MotionAdapter support is required for AnimateDiff SDXL execution."
            ) from exc
        return MotionAdapter

    def _load_pipeline(
        self,
        *,
        source: str,
        motion_adapter_source: str,
        torch_dtype: Any,
        scheduler_name: str,
        config: Dict[str, Any],
    ):
        cache_key = self.pipeline_cache_key(
            source=f"{source}|{motion_adapter_source}",
            torch_dtype=torch_dtype,
            scheduler_name=scheduler_name,
        )
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        if scheduler_name != "native":
            raise ValueError(f"Unsupported AnimateDiff SDXL scheduler: {scheduler_name}")

        motion_adapter_cls = self._motion_adapter_cls()
        pipeline_cls = self._diffusers_pipeline_cls()
        motion_adapter = motion_adapter_cls.from_pretrained(motion_adapter_source, torch_dtype=torch_dtype)
        pipeline = pipeline_cls.from_pretrained(source, motion_adapter=motion_adapter, torch_dtype=torch_dtype)
        pipeline = self.runtime.prepare_pipeline(pipeline, model_spec=self.model_spec, config=config)
        self._wrap_pipeline_modules(pipeline)
        pipeline, placement_managed = self.apply_pipeline_optimizations(pipeline, config=config)
        if not placement_managed:
            pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("text_encoder", "text_encoder_2", "unet", "motion_adapter", "vae"):
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
