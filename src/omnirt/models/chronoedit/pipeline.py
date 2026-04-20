"""ChronoEdit image editing pipeline backed by Diffusers."""

from __future__ import annotations

import inspect
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.backends.overrides import ASCEND_ACCELERATION_CONFIG_KEYS
from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.media import load_image
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest


DEFAULT_CHRONOEDIT_MODEL_SOURCE = "nvidia/ChronoEdit-14B-Diffusers"


@register_model(
    id="chronoedit",
    task="edit",
    default_backend="auto",
    resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    capabilities=ModelCapabilities(
        required_inputs=("image", "prompt"),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "max_sequence_length",
            "enable_temporal_reasoning",
            "num_temporal_reasoning_steps",
            "seed",
            "dtype",
            "output_dir",
        )
        + ASCEND_ACCELERATION_CONFIG_KEYS,
        default_config={
            "height": 512,
            "width": 512,
            "num_frames": 9,
            "dtype": "bf16",
            "enable_temporal_reasoning": True,
            "num_temporal_reasoning_steps": 1,
        },
        supported_schedulers=("native",),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="beta",
        summary="ChronoEdit physically-consistent image editing pipeline.",
        example="omnirt generate --task edit --model chronoedit --image input.png --prompt \"turn this object into polished bronze while preserving structure\" --backend cuda",
    ),
)
class ChronoEditPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_key: Optional[tuple] = None
        self._last_seed = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        image = req.inputs.get("image")
        if not prompt:
            raise ValueError("edit requires inputs.prompt")
        if not image:
            raise ValueError("edit requires inputs.image")
        source_image = load_image(str(image))
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "image": source_image,
            "model_source": req.config.get("model_path", DEFAULT_CHRONOEDIT_MODEL_SOURCE),
            "height": int(req.config.get("height", source_image.height)),
            "width": int(req.config.get("width", source_image.width)),
            "num_frames": int(req.config.get("num_frames", 9)),
            "max_sequence_length": req.config.get("max_sequence_length"),
            "enable_temporal_reasoning": bool(req.config.get("enable_temporal_reasoning", True)),
            "num_temporal_reasoning_steps": int(req.config.get("num_temporal_reasoning_steps", 1)),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", 30))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", 4.0))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype", "bf16"))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(source=conditions["model_source"], torch_dtype=torch_dtype, config=req.config)
        self._last_seed = seed
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
            "image": conditions["image"],
            "prompt": conditions["prompt"],
            "negative_prompt": conditions.get("negative_prompt"),
            "height": conditions["height"],
            "width": conditions["width"],
            "num_frames": conditions["num_frames"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "generator": latents["generator"],
            "max_sequence_length": conditions.get("max_sequence_length"),
            "enable_temporal_reasoning": conditions["enable_temporal_reasoning"],
            "num_temporal_reasoning_steps": conditions["num_temporal_reasoning_steps"],
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
            raise ValueError("Unexpected ChronoEdit pipeline output.")
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
        artifacts: List[Artifact] = []
        seed_part = self._last_seed if self._last_seed is not None else "random"
        frame = raw[-1]
        file_path = output_dir / f"{req.model}-{seed_part}-final.png"
        frame.save(file_path)
        artifacts.append(
            Artifact(
                kind="image",
                path=str(file_path),
                mime="image/png",
                width=frame.width,
                height=frame.height,
            )
        )
        return artifacts

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: Any) -> Dict[str, Any]:
        return {
            "model_path": conditions["model_source"],
            "height": conditions["height"],
            "width": conditions["width"],
            "num_frames": conditions["num_frames"],
            "num_inference_steps": latents["steps"],
            "guidance_scale": latents["guidance_scale"],
            "max_sequence_length": conditions.get("max_sequence_length"),
            "enable_temporal_reasoning": conditions["enable_temporal_reasoning"],
            "num_temporal_reasoning_steps": conditions["num_temporal_reasoning_steps"],
            "seed": latents["seed"],
            "dtype": req.config.get("dtype", "bf16"),
            "output_dir": str(Path(req.config.get("output_dir", "outputs"))),
        }

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required to run ChronoEdit.") from exc
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
            from diffusers import ChronoEditPipeline as DiffusersChronoEditPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers with ChronoEditPipeline support is required for ChronoEdit execution."
            ) from exc
        return DiffusersChronoEditPipeline

    def _load_pipeline(self, *, source: str, torch_dtype: Any, config: Dict[str, Any]):
        cache_key = self.pipeline_cache_key(source=source, torch_dtype=torch_dtype, scheduler_name="native")
        if self._pipeline is not None and self._pipeline_key == cache_key:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(source, torch_dtype=torch_dtype)
        pipeline = self.runtime.prepare_pipeline(pipeline, model_spec=self.model_spec, config=config)
        self._wrap_pipeline_modules(pipeline)
        pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_key = cache_key
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("image_encoder", "text_encoder", "transformer", "vae"):
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
