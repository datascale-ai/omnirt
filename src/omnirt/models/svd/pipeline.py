"""SVD pipeline implementation backed by Diffusers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnirt.backends.overrides import ASCEND_ACCELERATION_CONFIG_KEYS
from omnirt.core.base_pipeline import BasePipeline, LEGACY_OPTIMIZATION_CONFIG_KEYS
from omnirt.core.media import load_image, save_video_frames
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.svd.components import DEFAULT_SVD_MODEL_SOURCE, DEFAULT_SVD_XT_MODEL_SOURCE
from omnirt.schedulers import build_scheduler


@register_model(
    id="svd",
    task="image2video",
    default_backend="auto",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("image",),
        optional_inputs=("num_frames", "fps"),
        supported_config=(
            "model_path",
            "scheduler",
            "frame_bucket",
            "motion_bucket_id",
            "decode_chunk_size",
            "noise_aug_strength",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        )
        + LEGACY_OPTIMIZATION_CONFIG_KEYS
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={
            "scheduler": "euler-discrete",
            "frame_bucket": 127,
            "decode_chunk_size": 8,
            "noise_aug_strength": 0.02,
            "dtype": "fp16",
        },
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        artifact_kind="video",
        maturity="stable",
        summary="Stable Video Diffusion base image-to-video pipeline.",
        example="omnirt generate --task image2video --model svd --image input.png --backend cuda",
    ),
)
@register_model(
    id="svd-xt",
    task="image2video",
    default_backend="auto",
    resource_hint={"min_vram_gb": 14, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("image",),
        optional_inputs=("num_frames", "fps"),
        supported_config=(
            "model_path",
            "scheduler",
            "frame_bucket",
            "motion_bucket_id",
            "decode_chunk_size",
            "noise_aug_strength",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "dtype",
            "output_dir",
        )
        + LEGACY_OPTIMIZATION_CONFIG_KEYS
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={
            "scheduler": "euler-discrete",
            "frame_bucket": 127,
            "decode_chunk_size": 8,
            "noise_aug_strength": 0.02,
            "dtype": "fp16",
        },
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        artifact_kind="video",
        maturity="stable",
        summary="Stable Video Diffusion XT image-to-video pipeline.",
        example="omnirt generate --task image2video --model svd-xt --image input.png --backend cuda",
    ),
)
class SVDPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_key: Optional[tuple] = None
        self._last_seed = None
        self._last_fps = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        image = req.inputs.get("image")
        if not image:
            raise ValueError("image2video requires inputs.image")
        if not Path(image).exists():
            raise FileNotFoundError(image)
        return {
            "image": load_image(image),
            "num_frames": int(req.inputs.get("num_frames", self._default_num_frames())),
            "fps": int(req.inputs.get("fps", 7)),
            "frame_bucket": int(req.config.get("frame_bucket", req.config.get("motion_bucket_id", 127))),
            "decode_chunk_size": int(req.config.get("decode_chunk_size", 8)),
            "noise_aug_strength": float(req.config.get("noise_aug_strength", 0.02)),
            "model_source": req.config.get("model_path", self._default_model_source()),
            "scheduler": req.config.get("scheduler", "euler-discrete"),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", 25))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", 3.0))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype"))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(
            source=conditions["model_source"],
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
            "conditions": conditions,
            "pipeline": pipeline,
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        result = pipeline(
            image=conditions["image"],
            num_frames=conditions["num_frames"],
            fps=conditions["fps"],
            motion_bucket_id=conditions["frame_bucket"],
            noise_aug_strength=conditions["noise_aug_strength"],
            decode_chunk_size=conditions["decode_chunk_size"],
            num_inference_steps=latents["steps"],
            min_guidance_scale=latents["guidance_scale"],
            max_guidance_scale=latents["guidance_scale"],
            generator=latents["generator"],
            output_type="pil",
        )
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
        save_video_frames(file_path, raw, fps=self._last_fps or int(req.inputs.get("fps", 7)))
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
            "frame_bucket": conditions["frame_bucket"],
            "decode_chunk_size": conditions["decode_chunk_size"],
            "noise_aug_strength": conditions["noise_aug_strength"],
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
            raise DependencyUnavailableError("PyTorch is required to run the SVD pipeline.") from exc
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
            from diffusers import StableVideoDiffusionPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers is required for SVD execution. Install omnirt with runtime dependencies."
            ) from exc
        return StableVideoDiffusionPipeline

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
        pipeline = pipeline_cls.from_pretrained(source, **self._from_pretrained_kwargs(source, torch_dtype, config=config))
        scheduler_config = dict(config)
        scheduler_config.setdefault("scheduler", scheduler_name)
        if getattr(pipeline, "scheduler", None) is not None and hasattr(pipeline.scheduler, "config"):
            scheduler_config["scheduler_config"] = pipeline.scheduler.config
        pipeline.scheduler = build_scheduler(scheduler_config)
        pipeline = self.runtime.prepare_pipeline(pipeline, model_spec=self.model_spec, config=config)
        self._wrap_pipeline_modules(pipeline)
        pipeline, placement_managed = self.apply_pipeline_optimizations(pipeline, config=config)
        if not placement_managed and not self.uses_managed_device_placement(config):
            pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _from_pretrained_kwargs(self, source: str, torch_dtype: Any, *, config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        kwargs.update(self.from_pretrained_runtime_kwargs(config=config))
        variant = self._detect_local_variant(source)
        if variant is not None:
            kwargs["variant"] = variant
        return kwargs

    def _detect_local_variant(self, source: str) -> Optional[str]:
        root = Path(source)
        if not root.is_dir():
            return None

        fp16_layout = {
            "image_encoder": "model.fp16.safetensors",
            "unet": "diffusion_pytorch_model.fp16.safetensors",
            "vae": "diffusion_pytorch_model.fp16.safetensors",
        }
        if all((root / subdir / filename).is_file() for subdir, filename in fp16_layout.items()):
            return "fp16"
        return None

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("image_encoder", "unet", "vae"):
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
        self.adapter_manager.apply_to_pipeline(pipeline)

    def _default_model_source(self) -> str:
        if self.model_spec.id == "svd":
            return DEFAULT_SVD_MODEL_SOURCE
        return DEFAULT_SVD_XT_MODEL_SOURCE

    def _default_num_frames(self) -> int:
        if self.model_spec.id == "svd":
            return 14
        return 25
