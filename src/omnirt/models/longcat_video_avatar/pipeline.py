"""LongCat-Video-Avatar 1.5 wrapper pipeline backed by an external checkout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, GenerateRequest
from omnirt.launcher import resolve_launcher
from omnirt.models.flashtalk.pipeline import probe_video_file
from omnirt.models.longcat_video_avatar.components import (
    DEFAULT_LONGCAT_AVATAR_PROMPT,
    longcat_avatar_setting,
)


@dataclass(frozen=True)
class LongCatAvatarRuntimeConfig:
    accelerator: str
    repo_path: Path
    ckpt_dir: Path
    base_ckpt_dir: Optional[Path]
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]


@dataclass(frozen=True)
class LongCatAvatarLaunchConfig:
    accelerator: str
    repo_path: Path
    script_path: Path
    ckpt_dir: Path
    base_ckpt_dir: Optional[Path]
    input_json: Path
    request_file: Path
    response_file: Path
    stdout_log: Path
    save_file: Path
    request_id: str
    seed: int
    frames: int
    steps: int
    fps: int
    resolution: str
    model_type: str
    stage: str
    output_type: str
    placement: str
    cp_split_hw: str
    attention_profile: str
    attention_backend: str
    cross_attention_backend: Optional[str]
    bsa_band_blocks: Optional[int]
    bsa_block_q: int
    bsa_block_k: int
    cache_profile: str
    save_profile: str
    postprocess_mode: str
    warmup_steps: int
    max_sequence_length: int
    text_guidance_scale: float
    audio_guidance_scale: float
    use_distill: bool
    merge_lora: bool
    cache_static_inputs: bool
    precache_static_inputs: bool
    require_idle_npu: bool
    offline: bool
    hf_endpoint: Optional[str]
    ref_attn_map_impl: str
    text_encoder_device: str
    hccl_buffsize: str
    pytorch_npu_alloc_conf: str
    npu_fa_layout: Optional[str]
    attention_cache: bool
    attention_cache_interval: int
    attention_cache_step_start: int
    attention_cache_block_start: int
    attention_cache_block_end: int
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]
    extra_env: Dict[str, str]


@register_model(
    id="longcat-video-avatar-1.5",
    task="audio2video",
    default_backend="ascend",
    execution_mode="subprocess",
    resource_hint={
        "min_vram_gb": 256,
        "vram_scope": "aggregate",
        "dtype": "bf16",
        "accelerator": "CUDA GPU or Ascend 910B",
    },
    capabilities=ModelCapabilities(
        required_inputs=(),
        optional_inputs=("image", "audio", "prompt"),
        supported_config=(
            "model_path",
            "repo_path",
            "ckpt_dir",
            "base_ckpt_dir",
            "input_json",
            "worker_script",
            "seed",
            "output_dir",
            "python_executable",
            "launcher",
            "nproc_per_node",
            "num_processes",
            "accelerate_executable",
            "visible_devices",
            "accelerator",
            "device",
            "env_script",
            "ascend_env_script",
            "cuda_env_script",
            "frames",
            "num_frames",
            "steps",
            "sample_steps",
            "fps",
            "resolution",
            "model_type",
            "stage",
            "output_type",
            "placement",
            "audio_type",
            "bbox",
            "cp_split_hw",
            "attention_profile",
            "attention_backend",
            "cross_attention_backend",
            "bsa_band_blocks",
            "bsa_block_q",
            "bsa_block_k",
            "cache_profile",
            "save_profile",
            "save_mode",
            "postprocess_mode",
            "warmup_steps",
            "max_sequence_length",
            "text_guidance_scale",
            "audio_guidance_scale",
            "use_distill",
            "merge_lora",
            "cache_static_inputs",
            "precache_static_inputs",
            "require_idle_npu",
            "offline",
            "hf_endpoint",
            "ref_attn_map_impl",
            "text_encoder_device",
            "hccl_buffsize",
            "pytorch_npu_alloc_conf",
            "npu_fa_layout",
            "attention_cache",
            "attention_cache_interval",
            "attention_cache_step_start",
            "attention_cache_block_start",
            "attention_cache_block_end",
            "extra_env",
        ),
        default_config={
            "launcher": "torchrun",
            "nproc_per_node": 8,
            "frames": 249,
            "steps": 8,
            "fps": 25,
            "resolution": "480p",
            "model_type": "avatar-v1.5",
            "stage": "ai2v",
            "output_type": "latent",
            "placement": "dit-only",
            "cp_split_hw": "4,2",
            "attention_profile": "formal",
            "cache_profile": "faster",
            "save_profile": "copy_mux",
            "offline": True,
            "merge_lora": True,
            "use_distill": True,
            "cache_static_inputs": True,
            "precache_static_inputs": True,
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="video",
        maturity="beta",
        chain_role="avatar-render",
        realtime=False,
        tier="core",
        summary="LongCat-Video-Avatar 1.5 audio-driven avatar generation via external CUDA or Ascend checkouts.",
        example=(
            "OMNIRT_LONGCAT_AVATAR_REPO_PATH=/opt/model-repos/LongCat-Video "
            "omnirt generate --task audio2video --model longcat-video-avatar-1.5 "
            "--image speaker.png --audio voice.wav --backend ascend"
        ),
    ),
)
class LongCatVideoAvatarPipeline(BasePipeline):
    def ensure_resource_budget(self, req: GenerateRequest) -> None:
        if self._normalize_optional_string(req.config.get("visible_devices")):
            return
        super().ensure_resource_budget(req)

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        input_json = self._resolve_optional_file(req.config.get("input_json"))
        if input_json is not None:
            self._validate_input_json(input_json)
            return {"input_json": input_json, "generated_input_json": False}

        image_path = self._resolve_required_file(req.inputs.get("image"), "image")
        audio_path = self._resolve_required_file(req.inputs.get("audio"), "audio")
        prompt = str(req.inputs.get("prompt") or DEFAULT_LONGCAT_AVATAR_PROMPT)
        bbox = self._mapping_config(req.config.get("bbox"), "bbox")
        audio_type = str(req.config.get("audio_type", "para"))
        return {
            "image_path": image_path,
            "audio_path": audio_path,
            "prompt": prompt,
            "bbox": bbox,
            "audio_type": audio_type,
            "generated_input_json": True,
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> LongCatAvatarLaunchConfig:
        runtime_config = self.resolve_runtime_config(
            req.config,
            backend_name=getattr(self.runtime, "name", "auto"),
        )
        worker_script = (
            self._normalize_optional_string(req.config.get("worker_script"))
            or longcat_avatar_setting("worker_script")
            or self._default_worker_script(runtime_config.accelerator)
        )
        script_path = self._resolve_repo_relative_path(runtime_config.repo_path, worker_script)
        if not script_path.exists():
            raise FileNotFoundError(f"LongCat Avatar worker script not found: {script_path}")

        output_dir = self.resolve_output_dir(req).resolve()
        seed = int(req.config.get("seed", 301))
        now_ms = int(time.time() * 1000)
        save_file = output_dir / f"{req.model}-{seed}-{now_ms}.mp4"
        request_id = f"omnirt_{seed}_{now_ms}"
        input_json = self._prepare_input_json(req, conditions, output_dir)
        response_file = output_dir / f"{req.model}-{seed}-{now_ms}.response.jsonl"
        request_file = output_dir / f"{req.model}-{seed}-{now_ms}.request.jsonl"
        stdout_log = output_dir / f"{req.model}-{seed}-{now_ms}.log"
        attention = self._resolve_attention_config(req.config, accelerator=runtime_config.accelerator)

        launch = LongCatAvatarLaunchConfig(
            accelerator=runtime_config.accelerator,
            repo_path=runtime_config.repo_path,
            script_path=script_path,
            ckpt_dir=runtime_config.ckpt_dir,
            base_ckpt_dir=runtime_config.base_ckpt_dir,
            input_json=input_json,
            request_file=request_file,
            response_file=response_file,
            stdout_log=stdout_log,
            save_file=save_file,
            request_id=request_id,
            seed=seed,
            frames=int(req.config.get("frames", req.config.get("num_frames", 249))),
            steps=int(req.config.get("steps", req.config.get("sample_steps", 8))),
            fps=int(req.config.get("fps", 25)),
            resolution=str(req.config.get("resolution", "480p")),
            model_type=str(req.config.get("model_type", "avatar-v1.5")),
            stage=str(req.config.get("stage", "ai2v")),
            output_type=str(req.config.get("output_type", "latent")),
            placement=str(req.config.get("placement", "dit-only")),
            cp_split_hw=str(req.config.get("cp_split_hw", "4,2")),
            attention_profile=attention["profile"],
            attention_backend=attention["backend"],
            cross_attention_backend=attention["cross_backend"],
            bsa_band_blocks=attention["bsa_band_blocks"],
            bsa_block_q=int(req.config.get("bsa_block_q", 128)),
            bsa_block_k=int(req.config.get("bsa_block_k", 128)),
            cache_profile=str(req.config.get("cache_profile", "faster")),
            save_profile=str(req.config.get("save_profile", req.config.get("save_mode", "copy_mux"))),
            postprocess_mode=str(
                req.config.get("postprocess_mode", self._default_postprocess_mode(runtime_config.accelerator))
            ),
            warmup_steps=int(req.config.get("warmup_steps", 0)),
            max_sequence_length=int(req.config.get("max_sequence_length", 64)),
            text_guidance_scale=float(req.config.get("text_guidance_scale", 1.0)),
            audio_guidance_scale=float(req.config.get("audio_guidance_scale", 1.0)),
            use_distill=self._config_bool(req.config, "use_distill", True),
            merge_lora=self._config_bool(req.config, "merge_lora", True),
            cache_static_inputs=self._config_bool(req.config, "cache_static_inputs", True),
            precache_static_inputs=self._config_bool(req.config, "precache_static_inputs", True),
            require_idle_npu=self._config_bool(req.config, "require_idle_npu", False),
            offline=self._config_bool(req.config, "offline", True),
            hf_endpoint=self._normalize_optional_string(req.config.get("hf_endpoint")),
            ref_attn_map_impl=str(req.config.get("ref_attn_map_impl", "fast")),
            text_encoder_device=str(
                req.config.get("text_encoder_device", self._default_text_encoder_device(runtime_config.accelerator))
            ),
            hccl_buffsize=str(req.config.get("hccl_buffsize", "512")),
            pytorch_npu_alloc_conf=str(req.config.get("pytorch_npu_alloc_conf", "expandable_segments:True")),
            npu_fa_layout=self._normalize_optional_string(req.config.get("npu_fa_layout")),
            attention_cache=self._config_bool(req.config, "attention_cache", True),
            attention_cache_interval=int(req.config.get("attention_cache_interval", 4)),
            attention_cache_step_start=int(req.config.get("attention_cache_step_start", 1)),
            attention_cache_block_start=int(req.config.get("attention_cache_block_start", 0)),
            attention_cache_block_end=int(req.config.get("attention_cache_block_end", 48)),
            python_executable=runtime_config.python_executable,
            launcher=runtime_config.launcher,
            nproc_per_node=runtime_config.nproc_per_node,
            num_processes=runtime_config.num_processes,
            accelerate_executable=runtime_config.accelerate_executable,
            visible_devices=runtime_config.visible_devices,
            ascend_env_script=runtime_config.ascend_env_script,
            extra_env=self._string_mapping_config(req.config.get("extra_env"), "extra_env"),
        )
        self._write_request_file(launch)
        return launch

    def denoise_loop(self, latents: LongCatAvatarLaunchConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        del conditions, config
        env = self._build_env(latents)
        script_args = self._build_script_args(latents)
        launcher = resolve_launcher(latents.launcher)
        command = launcher.build_command(
            latents.script_path,
            python_executable=latents.python_executable,
            script_args=script_args,
            config={
                "nproc_per_node": latents.nproc_per_node,
                "num_processes": latents.num_processes,
                "accelerate_executable": latents.accelerate_executable,
            },
        )
        shell_command = launcher._build_shell_command(
            cwd=latents.repo_path,
            command=command,
            env_script=latents.ascend_env_script,
        )

        latents.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        with latents.stdout_log.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                ["bash", "-lc", shell_command],
                check=False,
                cwd=str(latents.repo_path),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            tail = latents.stdout_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-20:]
            detail = "\n".join(tail)
            raise RuntimeError(
                f"LongCat Avatar launch failed with exit code {completed.returncode}."
                + (f"\nRecent output:\n{detail}" if detail else "")
            )

        response_events = self._read_response_events(latents.response_file)
        save_file = self._response_save_file(response_events) or latents.save_file
        return {
            "save_file": save_file,
            "stdout_log": latents.stdout_log,
            "response_file": latents.response_file,
            "response_events": response_events,
        }

    def decode(self, latents: Any) -> Any:
        return latents

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        del req
        save_file = Path(raw["save_file"])
        if not save_file.exists():
            raise FileNotFoundError(f"LongCat Avatar output file missing after generation: {save_file}")
        width, height, num_frames = probe_video_file(save_file)
        return [
            Artifact(
                kind="video",
                path=str(save_file),
                mime="video/mp4",
                width=width,
                height=height,
                num_frames=num_frames,
            )
        ]

    def resolve_run_config(
        self,
        req: GenerateRequest,
        conditions: Any,
        latents: LongCatAvatarLaunchConfig,
    ) -> Dict[str, Any]:
        del req, conditions
        return {
            "accelerator": latents.accelerator,
            "repo_path": str(latents.repo_path),
            "model_path": str(latents.ckpt_dir),
            "ckpt_dir": str(latents.ckpt_dir),
            "base_ckpt_dir": str(latents.base_ckpt_dir) if latents.base_ckpt_dir else None,
            "input_json": str(latents.input_json),
            "request_file": str(latents.request_file),
            "response_file": str(latents.response_file),
            "stdout_log": str(latents.stdout_log),
            "output_dir": str(latents.save_file.parent),
            "seed": latents.seed,
            "frames": latents.frames,
            "steps": latents.steps,
            "fps": latents.fps,
            "resolution": latents.resolution,
            "model_type": latents.model_type,
            "stage": latents.stage,
            "output_type": latents.output_type,
            "placement": latents.placement,
            "cp_split_hw": latents.cp_split_hw,
            "attention_profile": latents.attention_profile,
            "attention_backend": latents.attention_backend,
            "cross_attention_backend": latents.cross_attention_backend,
            "bsa_band_blocks": latents.bsa_band_blocks,
            "bsa_block_q": latents.bsa_block_q,
            "bsa_block_k": latents.bsa_block_k,
            "cache_profile": latents.cache_profile,
            "save_profile": latents.save_profile,
            "postprocess_mode": latents.postprocess_mode,
            "offline": latents.offline,
            "python_executable": latents.python_executable,
            "launcher": latents.launcher,
            "nproc_per_node": latents.nproc_per_node,
            "num_processes": latents.num_processes,
            "accelerate_executable": latents.accelerate_executable,
            "visible_devices": latents.visible_devices,
            "env_script": latents.ascend_env_script,
            "ascend_env_script": latents.ascend_env_script,
            "extra_env_keys": sorted(latents.extra_env),
        }

    @staticmethod
    def resolve_runtime_config(config: Dict[str, Any], *, backend_name: str = "auto") -> LongCatAvatarRuntimeConfig:
        accelerator = LongCatVideoAvatarPipeline._normalize_accelerator(
            config.get("accelerator") or config.get("device") or backend_name
        )
        repo_path_value = config.get("repo_path") or longcat_avatar_setting("repo_path", required=True)
        repo_path = Path(str(repo_path_value)).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"LongCat Avatar repo_path not found: {repo_path}")

        ckpt_value = (
            config.get("ckpt_dir")
            or config.get("model_path")
            or longcat_avatar_setting("ckpt_dir", required=True)
        )
        ckpt_dir = LongCatVideoAvatarPipeline._resolve_repo_relative_path(repo_path, str(ckpt_value))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"LongCat Avatar ckpt_dir not found: {ckpt_dir}")

        base_ckpt_value = config.get("base_ckpt_dir") or longcat_avatar_setting("base_ckpt_dir")
        base_ckpt_dir = (
            LongCatVideoAvatarPipeline._resolve_repo_relative_path(repo_path, str(base_ckpt_value))
            if base_ckpt_value
            else None
        )
        if base_ckpt_dir is not None and not base_ckpt_dir.exists():
            raise FileNotFoundError(f"LongCat Avatar base_ckpt_dir not found: {base_ckpt_dir}")

        python_executable_value = config.get("python_executable") or longcat_avatar_setting(
            "python_executable",
            required=True,
        )
        python_executable = str(python_executable_value)
        if python_executable and not Path(python_executable).expanduser().exists():
            raise FileNotFoundError(f"LongCat Avatar python_executable not found: {python_executable}")

        env_script_override = LongCatVideoAvatarPipeline._normalize_optional_string(config.get("env_script"))
        ascend_env_override = LongCatVideoAvatarPipeline._normalize_optional_string(config.get("ascend_env_script"))
        cuda_env_override = LongCatVideoAvatarPipeline._normalize_optional_string(config.get("cuda_env_script"))
        env_script = (
            env_script_override
            or (cuda_env_override if accelerator == "cuda" else ascend_env_override)
            or (longcat_avatar_setting("cuda_env_script") if accelerator == "cuda" else None)
            or longcat_avatar_setting("env_script")
            or (longcat_avatar_setting("ascend_env_script") if accelerator == "ascend" else None)
        )
        if env_script and not Path(env_script).expanduser().exists():
            raise FileNotFoundError(f"LongCat Avatar env_script not found: {env_script}")

        nproc_per_node = int(config.get("nproc_per_node", 8))
        return LongCatAvatarRuntimeConfig(
            accelerator=accelerator,
            repo_path=repo_path,
            ckpt_dir=ckpt_dir,
            base_ckpt_dir=base_ckpt_dir,
            python_executable=python_executable,
            launcher=str(config.get("launcher", "torchrun")),
            nproc_per_node=nproc_per_node,
            num_processes=int(config.get("num_processes", nproc_per_node)),
            accelerate_executable=LongCatVideoAvatarPipeline._normalize_optional_string(
                config.get("accelerate_executable")
            ),
            visible_devices=(
                LongCatVideoAvatarPipeline._normalize_optional_string(config.get("visible_devices"))
                or longcat_avatar_setting("visible_devices")
            ),
            ascend_env_script=env_script,
        )

    @staticmethod
    def _append_flag(args: List[str], flag: str, enabled: bool) -> None:
        if enabled:
            args.append(flag)

    @staticmethod
    def _build_script_args(latents: LongCatAvatarLaunchConfig) -> List[str]:
        args: List[str] = [
            "--input-json",
            str(latents.input_json),
            "--checkpoint-dir",
            str(latents.ckpt_dir),
            "--model-type",
            latents.model_type,
            "--stage",
            latents.stage,
            "--resolution",
            latents.resolution,
            "--frames",
            str(latents.frames),
            "--steps",
            str(latents.steps),
            "--seed",
            str(latents.seed),
            "--fps",
            str(latents.fps),
            "--output-type",
            latents.output_type,
            "--placement",
            latents.placement,
            "--cp-split-hw",
            latents.cp_split_hw,
            "--request-file",
            str(latents.request_file),
            "--response-file",
            str(latents.response_file),
            "--save-mp4",
            str(latents.save_file),
            "--save-mode",
            latents.save_profile,
            "--warmup-steps",
            str(latents.warmup_steps),
            "--max-sequence-length",
            str(latents.max_sequence_length),
            "--text-guidance-scale",
            str(latents.text_guidance_scale),
            "--audio-guidance-scale",
            str(latents.audio_guidance_scale),
        ]
        if latents.base_ckpt_dir is not None:
            args.extend(["--base-checkpoint-dir", str(latents.base_ckpt_dir)])
        LongCatVideoAvatarPipeline._append_flag(args, "--use-distill", latents.use_distill)
        LongCatVideoAvatarPipeline._append_flag(args, "--merge-lora", latents.merge_lora)
        LongCatVideoAvatarPipeline._append_flag(args, "--cache-static-inputs", latents.cache_static_inputs)
        LongCatVideoAvatarPipeline._append_flag(args, "--precache-static-inputs", latents.precache_static_inputs)
        LongCatVideoAvatarPipeline._append_flag(args, "--require-idle-npu", latents.require_idle_npu)
        return args

    @staticmethod
    def _build_env(latents: LongCatAvatarLaunchConfig) -> Dict[str, str]:
        env = dict(os.environ)
        if latents.visible_devices:
            if latents.accelerator == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = latents.visible_devices
            else:
                env["ASCEND_RT_VISIBLE_DEVICES"] = latents.visible_devices
        if latents.launcher == "torchrun":
            env["GPU_NUM"] = str(latents.nproc_per_node)
            env["NPROC_PER_NODE"] = str(latents.nproc_per_node)
        env.update(
            {
                "LONGCAT_DEVICE_BACKEND": "cuda" if latents.accelerator == "cuda" else "npu",
                "LONGCAT_DIST_BACKEND": "nccl" if latents.accelerator == "cuda" else "hccl",
                "LONGCAT_MODEL_TYPE": latents.model_type,
                "LONGCAT_CP_SPLIT_HW": latents.cp_split_hw,
                "LONGCAT_ATTENTION_BACKEND": latents.attention_backend,
                "LONGCAT_REF_ATTN_MAP_IMPL": latents.ref_attn_map_impl,
                "LONGCAT_MERGE_LORA": LongCatVideoAvatarPipeline._env_bool(latents.merge_lora),
                "LONGCAT_AVATAR_TEXT_ENCODER_DEVICE": latents.text_encoder_device,
                "AVATAR_CACHE_PROFILE": latents.cache_profile,
                "LONGCAT_AVATAR_CACHE_PROFILE": latents.cache_profile,
                "LONGCAT_AVATAR_COPY_MUX_DIRECT": "1",
                "LONGCAT_AVATAR_STREAM_VAE_SAVE": "1",
                "LONGCAT_AVATAR_STREAM_VAE_ASYNC_WRITER": "1",
                "LONGCAT_AVATAR_STREAM_VAE_ASYNC_CPU_TENSOR": "1",
                "LONGCAT_AVATAR_SAVE_MODE": latents.save_profile,
                "LONGCAT_AVATAR_POSTPROCESS_MODE": latents.postprocess_mode,
                "LONGCAT_AVATAR_ATTENTION_CACHE": LongCatVideoAvatarPipeline._env_bool(latents.attention_cache),
                "LONGCAT_AVATAR_ATTENTION_CACHE_INTERVAL": str(latents.attention_cache_interval),
                "LONGCAT_AVATAR_ATTENTION_CACHE_STEP_START": str(latents.attention_cache_step_start),
                "LONGCAT_AVATAR_ATTENTION_CACHE_BLOCK_START": str(latents.attention_cache_block_start),
                "LONGCAT_AVATAR_ATTENTION_CACHE_BLOCK_END": str(latents.attention_cache_block_end),
            }
        )
        if latents.cross_attention_backend:
            env["LONGCAT_CROSS_ATTENTION_BACKEND"] = latents.cross_attention_backend
        if latents.accelerator == "ascend":
            env["HCCL_BUFFSIZE"] = latents.hccl_buffsize
            env["PYTORCH_NPU_ALLOC_CONF"] = latents.pytorch_npu_alloc_conf
        if latents.accelerator == "ascend" and latents.npu_fa_layout:
            env["LONGCAT_NPU_FA_LAYOUT"] = latents.npu_fa_layout
        if latents.offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["DIFFUSERS_OFFLINE"] = "1"
        if latents.hf_endpoint:
            env["HF_ENDPOINT"] = latents.hf_endpoint

        if latents.accelerator == "ascend" and "bsa" in latents.attention_backend.lower():
            env["LONGCAT_DISABLE_BSA"] = "0"
            env["LONGCAT_ASCEND_BSA_MASK_MODE"] = "band"
            if latents.bsa_band_blocks is not None:
                env["LONGCAT_ASCEND_BSA_BAND_BLOCKS"] = str(latents.bsa_band_blocks)
            env["LONGCAT_ASCEND_BSA_BLOCK_Q"] = str(latents.bsa_block_q)
            env["LONGCAT_ASCEND_BSA_BLOCK_K"] = str(latents.bsa_block_k)
            env.setdefault("LONGCAT_CROSS_ATTENTION_BACKEND", "npu_fusion")
        elif latents.accelerator == "ascend":
            env["LONGCAT_DISABLE_BSA"] = "1"

        env.update(latents.extra_env)
        return env

    @staticmethod
    def _write_request_file(latents: LongCatAvatarLaunchConfig) -> None:
        payload = {
            "id": latents.request_id,
            "input_json": str(latents.input_json),
            "frames": latents.frames,
            "steps": latents.steps,
            "fps": latents.fps,
            "seed": latents.seed,
            "output_type": latents.output_type,
            "cache_static_inputs": latents.cache_static_inputs,
            "precache_static_inputs": latents.precache_static_inputs,
            "save_mp4": str(latents.save_file),
            "save_mode": latents.save_profile,
        }
        latents.request_file.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n" + json.dumps({"id": "shutdown", "shutdown": True}) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_response_events(response_file: Path) -> List[Dict[str, Any]]:
        if not response_file.exists():
            return []
        events: List[Dict[str, Any]] = []
        for line in response_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise RuntimeError(f"LongCat Avatar worker error: {event['error']}")
            if str(event.get("status", "")).lower() == "error":
                raise RuntimeError(f"LongCat Avatar worker error: {event}")
            events.append(event)
        return events

    @staticmethod
    def _response_save_file(events: List[Dict[str, Any]]) -> Optional[Path]:
        for event in reversed(events):
            save = event.get("save")
            if isinstance(save, dict):
                path = save.get("path") or save.get("file")
                if path:
                    return Path(str(path))
            for key in ("save_file", "save_mp4", "output_path", "path"):
                value = event.get(key)
                if value:
                    return Path(str(value))
        return None

    @staticmethod
    def _prepare_input_json(req: GenerateRequest, conditions: Dict[str, Any], output_dir: Path) -> Path:
        input_json = conditions.get("input_json")
        if input_json is not None:
            return Path(input_json)
        item = {
            "prompt": conditions["prompt"],
            "cond_image": str(Path(conditions["image_path"]).resolve()),
            "cond_audio": {"person1": str(Path(conditions["audio_path"]).resolve())},
            "audio_type": conditions["audio_type"],
            "bbox": conditions["bbox"],
        }
        digest = LongCatVideoAvatarPipeline._input_json_digest(item)
        path = output_dir / f"{req.model}-input-{digest}.json"
        path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _resolve_attention_config(config: Dict[str, Any], *, accelerator: str) -> Dict[str, Any]:
        profile = str(config.get("attention_profile", "formal")).strip().lower() or "formal"
        explicit_backend = LongCatVideoAvatarPipeline._explicit_config_string(config, "attention_backend")
        explicit_cross = LongCatVideoAvatarPipeline._normalize_optional_string(config.get("cross_attention_backend"))
        explicit_band = LongCatVideoAvatarPipeline._normalize_optional_int(config.get("bsa_band_blocks"))

        if accelerator == "cuda":
            if explicit_backend and (explicit_backend in {"npu_fusion", "ascend_bsa"} or "bsa" in explicit_backend):
                raise ValueError("CUDA attention_backend must not be an Ascend-only backend.")
            if profile in {"sdpa", "cuda_sdpa"}:
                return {
                    "profile": "cuda_sdpa",
                    "backend": explicit_backend or "sdpa",
                    "cross_backend": explicit_cross,
                    "bsa_band_blocks": explicit_band,
                }
            if profile in {"formal", "dense", "cuda", "flash_attn", "flash-attn", "cuda_flash_attn"}:
                return {
                    "profile": "cuda_flash_attn",
                    "backend": explicit_backend or "flash_attn",
                    "cross_backend": explicit_cross,
                    "bsa_band_blocks": explicit_band,
                }
            raise ValueError(
                "CUDA attention_profile must be one of 'formal', 'cuda_flash_attn', or 'cuda_sdpa'."
            )

        if profile in {"preview_bsa128", "bsa128"}:
            return {
                "profile": "preview_bsa128",
                "backend": explicit_backend or "ascend_bsa",
                "cross_backend": explicit_cross or "npu_fusion",
                "bsa_band_blocks": explicit_band or 128,
            }
        if profile in {"preview_bsa256", "bsa256"}:
            return {
                "profile": "preview_bsa256",
                "backend": explicit_backend or "ascend_bsa",
                "cross_backend": explicit_cross or "npu_fusion",
                "bsa_band_blocks": explicit_band or 256,
            }
        if profile not in {"formal", "dense", "npu_fusion"}:
            raise ValueError(
                "attention_profile must be one of 'formal', 'preview_bsa128', or 'preview_bsa256'."
            )
        return {
            "profile": "formal",
            "backend": explicit_backend or "npu_fusion",
            "cross_backend": explicit_cross,
            "bsa_band_blocks": explicit_band,
        }

    @staticmethod
    def _resolve_required_file(value: Any, label: str) -> Path:
        if value in (None, ""):
            raise FileNotFoundError(f"LongCat Avatar requires {label!r} input unless config.input_json is provided.")
        path = Path(str(value)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LongCat Avatar {label} input not found: {path}")
        return path.resolve()

    @staticmethod
    def _resolve_optional_file(value: Any) -> Optional[Path]:
        text = LongCatVideoAvatarPipeline._normalize_optional_string(value)
        if text is None:
            return None
        path = Path(text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LongCat Avatar input_json not found: {path}")
        return path.resolve()

    @staticmethod
    def _validate_input_json(path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, (dict, list)):
            raise ValueError("LongCat Avatar input_json must contain a JSON object or list.")

    @staticmethod
    def _resolve_repo_relative_path(repo_path: Path, value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (repo_path / candidate)

    @staticmethod
    def _mapping_config(value: Any, label: str) -> Dict[str, Any]:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        raise TypeError(f"LongCat Avatar {label} must be a mapping or JSON object string.")

    @staticmethod
    def _string_mapping_config(value: Any, label: str) -> Dict[str, str]:
        mapping = LongCatVideoAvatarPipeline._mapping_config(value, label)
        return {str(key): str(item) for key, item in mapping.items()}

    @staticmethod
    def _normalize_optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _explicit_config_string(config: Dict[str, Any], key: str) -> Optional[str]:
        if key not in config:
            return None
        return LongCatVideoAvatarPipeline._normalize_optional_string(config.get(key))

    @staticmethod
    def _normalize_accelerator(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"cuda", "gpu", "nvidia"}:
            return "cuda"
        if text in {"ascend", "npu", "910b", "910b2"}:
            return "ascend"
        return "ascend"

    @staticmethod
    def _default_worker_script(accelerator: str) -> str:
        if accelerator == "cuda":
            return "run_cuda_avatar_worker.py"
        return "run_ascend_avatar_cp_worker.py"

    @staticmethod
    def _default_text_encoder_device(accelerator: str) -> str:
        if accelerator == "cuda":
            return "cuda"
        return "npu"

    @staticmethod
    def _default_postprocess_mode(accelerator: str) -> str:
        if accelerator == "cuda":
            return "cuda_uint8"
        return "npu_uint8"

    @staticmethod
    def _normalize_optional_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _config_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
        value = config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _env_bool(value: bool) -> str:
        return "1" if value else "0"

    @staticmethod
    def _input_json_digest(item: Dict[str, Any]) -> str:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]
