"""SoulX-LiveAct wrapper pipeline backed by an external Ascend checkout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, GenerateRequest
from omnirt.launcher import resolve_launcher
from omnirt.models.flashtalk.pipeline import probe_video_file
from omnirt.models.liveact.components import DEFAULT_LIVEACT_PROMPT, liveact_setting


@dataclass(frozen=True)
class LiveActRuntimeConfig:
    repo_path: Path
    ckpt_dir: Path
    wav2vec_dir: Path
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]
    platform: str


@dataclass(frozen=True)
class LiveActLaunchConfig:
    repo_path: Path
    script_path: Path
    ckpt_dir: Path
    wav2vec_dir: Path
    input_json: Path
    stdout_log: Path
    generated_file: Path
    save_file: Path
    size: str
    fps: int
    seed: int
    audio_cfg: float
    t5_cpu: bool
    rank0_t5_only: bool
    stop_after_text_context: bool
    offload_cache: bool
    fp8_kv_cache: bool
    block_offload: bool
    dura_print: bool
    steam_audio: bool
    mean_memory: bool
    sample_steps: int
    use_cache_vae: bool
    vae_path: Optional[Path]
    use_lightvae: bool
    condition_cache_dir: Optional[str]
    disable_condition_cache: bool
    prepare_text_cache: bool
    text_cache_device: str
    text_cache_visible_devices: Optional[str]
    force_text_cache: bool
    fast_export: bool
    disable_fast_export: bool
    fast_export_preset: str
    fast_export_crf: int
    sequence_parallel_degree: Optional[int]
    ulysses_degree: Optional[int]
    ring_degree: int
    stage_profile: bool
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]
    platform: str


@register_model(
    id="soulx-liveact-14b",
    task="audio2video",
    default_backend="ascend",
    execution_mode="legacy_call",
    resource_hint={
        "min_vram_gb": 64,
        "vram_scope": "aggregate",
        "dtype": "bf16",
        "accelerator": "Ascend 910B",
    },
    capabilities=ModelCapabilities(
        required_inputs=("image", "audio"),
        optional_inputs=("prompt",),
        supported_config=(
            "model_path",
            "repo_path",
            "ckpt_dir",
            "wav2vec_dir",
            "seed",
            "output_dir",
            "python_executable",
            "launcher",
            "nproc_per_node",
            "num_processes",
            "accelerate_executable",
            "visible_devices",
            "ascend_env_script",
            "platform",
            "size",
            "fps",
            "audio_cfg",
            "t5_cpu",
            "rank0_t5_only",
            "stop_after_text_context",
            "offload_cache",
            "fp8_kv_cache",
            "block_offload",
            "dura_print",
            "steam_audio",
            "mean_memory",
            "sample_steps",
            "use_cache_vae",
            "vae_path",
            "use_lightvae",
            "condition_cache_dir",
            "disable_condition_cache",
            "prepare_text_cache",
            "text_cache_device",
            "text_cache_visible_devices",
            "force_text_cache",
            "disable_text_cache_prepare",
            "fast_export",
            "disable_fast_export",
            "fast_export_preset",
            "fast_export_crf",
            "sequence_parallel_degree",
            "ulysses_degree",
            "ring_degree",
            "stage_profile",
            "edit_prompt",
        ),
        default_config={
            "launcher": "torchrun",
            "nproc_per_node": 4,
            "size": "416*720",
            "fps": 20,
            "prepare_text_cache": True,
            "text_cache_device": "npu",
            "t5_cpu": False,
            "rank0_t5_only": True,
            "sample_steps": 3,
            "steam_audio": True,
            "platform": "ascend_npu",
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="video",
        maturity="beta",
        summary="SoulX-LiveAct long-form audio-driven avatar video generation on Ascend.",
        example=(
            "OMNIRT_LIVEACT_REPO_PATH=/path/to/SoulX-LiveAct "
            "omnirt generate --task audio2video --model soulx-liveact-14b --image speaker.png "
            "--audio voice.wav --backend ascend --sample-steps 1 --rank0-t5-only"
        ),
    ),
)
class LiveActPipeline(BasePipeline):
    def ensure_resource_budget(self, req: GenerateRequest) -> None:
        if self._normalize_optional_string(req.config.get("visible_devices")):
            return
        super().ensure_resource_budget(req)

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        image_path = Path(str(req.inputs.get("image", ""))).expanduser()
        audio_path = Path(str(req.inputs.get("audio", ""))).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        prompt = str(req.inputs.get("prompt") or DEFAULT_LIVEACT_PROMPT)
        edit_prompt = req.config.get("edit_prompt") or {}
        if edit_prompt and not isinstance(edit_prompt, dict):
            raise TypeError("LiveAct edit_prompt must be a mapping when provided.")
        return {
            "image_path": image_path,
            "audio_path": audio_path,
            "prompt": prompt,
            "edit_prompt": edit_prompt,
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> LiveActLaunchConfig:
        runtime_config = self.resolve_runtime_config(req.config)
        script_path = runtime_config.repo_path / "generate.py"
        if not script_path.exists():
            raise FileNotFoundError(f"LiveAct entry script not found: {script_path}")

        output_dir = self.resolve_output_dir(req).resolve()
        seed = int(req.config.get("seed", 42))
        stdout_log = output_dir / f"{req.model}-{seed}-{int(time.time() * 1000)}.log"
        item = {
            "cond_image": str(conditions["image_path"].resolve()),
            "cond_audio": str(conditions["audio_path"].resolve()),
            "prompt": conditions["prompt"],
        }
        if conditions["edit_prompt"]:
            item["edit_prompt"] = conditions["edit_prompt"]
        input_json = output_dir / f"{req.model}-input-{self._input_json_digest(item)}.json"
        input_json.write_text(json.dumps([item], ensure_ascii=False, indent=2), encoding="utf-8")

        generated_name = f"{conditions['image_path'].stem}_{conditions['audio_path'].stem}.mp4"
        save_file = output_dir / f"{req.model}-{seed}-{int(time.time() * 1000)}.mp4"

        return LiveActLaunchConfig(
            repo_path=runtime_config.repo_path,
            script_path=script_path,
            ckpt_dir=runtime_config.ckpt_dir,
            wav2vec_dir=runtime_config.wav2vec_dir,
            input_json=input_json,
            stdout_log=stdout_log,
            generated_file=runtime_config.repo_path / generated_name,
            save_file=save_file,
            size=str(req.config.get("size", "416*720")),
            fps=int(req.config.get("fps", 20)),
            seed=seed,
            audio_cfg=float(req.config.get("audio_cfg", 1.0)),
            t5_cpu=bool(req.config.get("t5_cpu", False)),
            rank0_t5_only=bool(req.config.get("rank0_t5_only", True)),
            stop_after_text_context=bool(req.config.get("stop_after_text_context", False)),
            offload_cache=bool(req.config.get("offload_cache", False)),
            fp8_kv_cache=bool(req.config.get("fp8_kv_cache", False)),
            block_offload=bool(req.config.get("block_offload", False)),
            dura_print=bool(req.config.get("dura_print", False)),
            steam_audio=bool(req.config.get("steam_audio", True)),
            mean_memory=bool(req.config.get("mean_memory", False)),
            sample_steps=int(req.config.get("sample_steps", 3)),
            use_cache_vae=bool(req.config.get("use_cache_vae", False)),
            vae_path=self._resolve_optional_repo_path(runtime_config.repo_path, req.config.get("vae_path")),
            use_lightvae=bool(req.config.get("use_lightvae", False)),
            condition_cache_dir=self._normalize_optional_string(req.config.get("condition_cache_dir")),
            disable_condition_cache=bool(req.config.get("disable_condition_cache", False)),
            prepare_text_cache=bool(req.config.get("prepare_text_cache", True))
            and not bool(req.config.get("disable_text_cache_prepare", False)),
            text_cache_device=str(req.config.get("text_cache_device", "npu")),
            text_cache_visible_devices=self._normalize_optional_string(req.config.get("text_cache_visible_devices")),
            force_text_cache=bool(req.config.get("force_text_cache", False)),
            fast_export=bool(req.config.get("fast_export", False)),
            disable_fast_export=bool(req.config.get("disable_fast_export", False)),
            fast_export_preset=str(req.config.get("fast_export_preset", "veryfast")),
            fast_export_crf=int(req.config.get("fast_export_crf", 18)),
            sequence_parallel_degree=self._normalize_optional_int(req.config.get("sequence_parallel_degree")),
            ulysses_degree=self._normalize_optional_int(req.config.get("ulysses_degree")),
            ring_degree=int(req.config.get("ring_degree", 1)),
            stage_profile=bool(req.config.get("stage_profile", False)),
            python_executable=runtime_config.python_executable,
            launcher=runtime_config.launcher,
            nproc_per_node=runtime_config.nproc_per_node,
            num_processes=runtime_config.num_processes,
            accelerate_executable=runtime_config.accelerate_executable,
            visible_devices=runtime_config.visible_devices,
            ascend_env_script=runtime_config.ascend_env_script,
            platform=runtime_config.platform,
        )

    def denoise_loop(self, latents: LiveActLaunchConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        env = dict(os.environ)
        env["PLATFORM"] = latents.platform
        if latents.visible_devices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = latents.visible_devices
        if latents.launcher == "torchrun":
            env["GPU_NUM"] = str(latents.nproc_per_node)
        if latents.stage_profile:
            env["LIVEACT_STAGE_PROFILE"] = "1"

        script_args: List[str] = [
            "--ckpt_dir",
            str(latents.ckpt_dir),
            "--wav2vec_dir",
            str(latents.wav2vec_dir),
            "--input_json",
            str(latents.input_json),
            "--size",
            latents.size,
            "--fps",
            str(latents.fps),
            "--seed",
            str(latents.seed),
            "--audio_cfg",
            str(latents.audio_cfg),
            "--sample_steps",
            str(latents.sample_steps),
            "--ring_degree",
            str(latents.ring_degree),
            "--fast_export_preset",
            latents.fast_export_preset,
            "--fast_export_crf",
            str(latents.fast_export_crf),
        ]
        self._append_flag(script_args, "--t5_cpu", latents.t5_cpu)
        self._append_flag(script_args, "--rank0_t5_only", latents.rank0_t5_only)
        self._append_flag(script_args, "--stop_after_text_context", latents.stop_after_text_context)
        self._append_flag(script_args, "--offload_cache", latents.offload_cache)
        self._append_flag(script_args, "--fp8_kv_cache", latents.fp8_kv_cache)
        self._append_flag(script_args, "--block_offload", latents.block_offload)
        self._append_flag(script_args, "--dura_print", latents.dura_print)
        self._append_flag(script_args, "--steam_audio", latents.steam_audio)
        self._append_flag(script_args, "--mean_memory", latents.mean_memory)
        self._append_flag(script_args, "--use_cache_vae", latents.use_cache_vae)
        self._append_flag(script_args, "--use_lightvae", latents.use_lightvae)
        self._append_flag(script_args, "--disable_condition_cache", latents.disable_condition_cache)
        self._append_flag(script_args, "--fast_export", latents.fast_export)
        self._append_flag(script_args, "--disable_fast_export", latents.disable_fast_export)
        if latents.vae_path is not None:
            script_args.extend(["--vae_path", str(latents.vae_path)])
        if latents.condition_cache_dir:
            script_args.extend(["--condition_cache_dir", latents.condition_cache_dir])
        if latents.sequence_parallel_degree is not None:
            script_args.extend(["--sequence_parallel_degree", str(latents.sequence_parallel_degree)])
        if latents.ulysses_degree is not None:
            script_args.extend(["--ulysses_degree", str(latents.ulysses_degree)])

        launcher = resolve_launcher(latents.launcher)
        inner_command = launcher.build_command(
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
            command=inner_command,
            env_script=latents.ascend_env_script,
        )
        latents.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        with latents.stdout_log.open("w", encoding="utf-8") as log_file:
            if latents.prepare_text_cache:
                self._prepare_text_cache(latents, env, log_file)
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
                f"LiveAct launch failed with exit code {completed.returncode}."
                + (f"\nRecent output:\n{detail}" if detail else "")
            )

        return {
            "generated_file": latents.generated_file,
            "save_file": latents.save_file,
            "stdout_log": latents.stdout_log,
            "seed": latents.seed,
        }

    def decode(self, latents: Any) -> Any:
        return latents

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        generated_file = Path(raw["generated_file"])
        save_file = Path(raw["save_file"])
        if not generated_file.exists():
            raise FileNotFoundError(f"LiveAct output file missing after generation: {generated_file}")
        if generated_file.resolve() != save_file.resolve():
            save_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated_file, save_file)
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

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: LiveActLaunchConfig) -> Dict[str, Any]:
        return {
            "repo_path": str(latents.repo_path),
            "model_path": str(latents.ckpt_dir),
            "ckpt_dir": str(latents.ckpt_dir),
            "wav2vec_dir": str(latents.wav2vec_dir),
            "input_json": str(latents.input_json),
            "stdout_log": str(latents.stdout_log),
            "seed": latents.seed,
            "output_dir": str(latents.save_file.parent),
            "python_executable": latents.python_executable,
            "launcher": latents.launcher,
            "nproc_per_node": latents.nproc_per_node,
            "num_processes": latents.num_processes,
            "accelerate_executable": latents.accelerate_executable,
            "visible_devices": latents.visible_devices,
            "ascend_env_script": latents.ascend_env_script,
            "platform": latents.platform,
            "size": latents.size,
            "fps": latents.fps,
            "audio_cfg": latents.audio_cfg,
            "t5_cpu": latents.t5_cpu,
            "rank0_t5_only": latents.rank0_t5_only,
            "stop_after_text_context": latents.stop_after_text_context,
            "offload_cache": latents.offload_cache,
            "fp8_kv_cache": latents.fp8_kv_cache,
            "block_offload": latents.block_offload,
            "dura_print": latents.dura_print,
            "steam_audio": latents.steam_audio,
            "mean_memory": latents.mean_memory,
            "sample_steps": latents.sample_steps,
            "use_cache_vae": latents.use_cache_vae,
            "vae_path": str(latents.vae_path) if latents.vae_path else None,
            "use_lightvae": latents.use_lightvae,
            "condition_cache_dir": latents.condition_cache_dir,
            "disable_condition_cache": latents.disable_condition_cache,
            "prepare_text_cache": latents.prepare_text_cache,
            "text_cache_device": latents.text_cache_device,
            "text_cache_visible_devices": latents.text_cache_visible_devices,
            "force_text_cache": latents.force_text_cache,
            "fast_export": latents.fast_export,
            "disable_fast_export": latents.disable_fast_export,
            "fast_export_preset": latents.fast_export_preset,
            "fast_export_crf": latents.fast_export_crf,
            "sequence_parallel_degree": latents.sequence_parallel_degree,
            "ulysses_degree": latents.ulysses_degree,
            "ring_degree": latents.ring_degree,
            "stage_profile": latents.stage_profile,
        }

    def _prepare_text_cache(self, latents: LiveActLaunchConfig, env: Dict[str, str], log_file: Any) -> None:
        prepare_script = latents.repo_path / "prepare_text_cache.py"
        if not prepare_script.exists():
            raise FileNotFoundError(f"LiveAct text-cache script not found: {prepare_script}")
        if not latents.force_text_cache:
            cache_paths = self._text_context_cache_paths(latents.input_json)
            if cache_paths and all(path.exists() for path in cache_paths):
                joined = ", ".join(str(path) for path in cache_paths)
                log_file.write(f"[omnirt] prepare_text_cache skipped existing {joined}\n")
                log_file.flush()
                return

        text_cache_args: List[str] = [
            "--ckpt_dir",
            str(latents.ckpt_dir),
            "--input_json",
            str(latents.input_json),
            "--device",
            latents.text_cache_device,
        ]
        self._append_flag(text_cache_args, "--force", latents.force_text_cache)
        text_cache_launcher = resolve_launcher("python")
        text_cache_command = text_cache_launcher.build_command(
            prepare_script,
            python_executable=latents.python_executable,
            script_args=text_cache_args,
            config={},
        )
        shell_command = text_cache_launcher._build_shell_command(
            cwd=latents.repo_path,
            command=text_cache_command,
            env_script=latents.ascend_env_script,
        )
        prepare_env = dict(env)
        text_cache_visible_devices = latents.text_cache_visible_devices or self._first_visible_device(
            latents.visible_devices
        )
        if text_cache_visible_devices:
            prepare_env["ASCEND_RT_VISIBLE_DEVICES"] = text_cache_visible_devices
        prepare_env["GPU_NUM"] = "1"
        log_file.write("[omnirt] prepare_text_cache start\n")
        log_file.flush()
        completed = subprocess.run(
            ["bash", "-lc", shell_command],
            check=False,
            cwd=str(latents.repo_path),
            env=prepare_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.write(f"[omnirt] prepare_text_cache done status={completed.returncode}\n")
        log_file.flush()
        if completed.returncode != 0:
            tail = latents.stdout_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-20:]
            detail = "\n".join(tail)
            raise RuntimeError(
                f"LiveAct text-cache preparation failed with exit code {completed.returncode}."
                + (f"\nRecent output:\n{detail}" if detail else "")
            )

    @staticmethod
    def resolve_runtime_config(config: Dict[str, Any]) -> LiveActRuntimeConfig:
        repo_path_value = config.get("repo_path") or liveact_setting("repo_path", required=True)
        repo_path = Path(str(repo_path_value)).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"LiveAct repo_path not found: {repo_path}")

        ckpt_value = config.get("ckpt_dir") or config.get("model_path") or liveact_setting("ckpt_dir", required=True)
        wav2vec_value = config.get("wav2vec_dir") or liveact_setting("wav2vec_dir", required=True)
        ckpt_dir = LiveActPipeline._resolve_repo_relative_path(repo_path, str(ckpt_value))
        wav2vec_dir = LiveActPipeline._resolve_repo_relative_path(repo_path, str(wav2vec_value))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"LiveAct ckpt_dir not found: {ckpt_dir}")
        if not wav2vec_dir.exists():
            raise FileNotFoundError(f"LiveAct wav2vec_dir not found: {wav2vec_dir}")

        python_executable_value = config.get("python_executable") or liveact_setting("python_executable", required=True)
        python_executable = str(python_executable_value)
        ascend_env_override = LiveActPipeline._normalize_optional_string(config.get("ascend_env_script"))
        ascend_env_script = ascend_env_override or liveact_setting("ascend_env_script")
        if python_executable and not Path(python_executable).expanduser().exists():
            raise FileNotFoundError(f"LiveAct python_executable not found: {python_executable}")
        if ascend_env_script and not Path(ascend_env_script).expanduser().exists():
            raise FileNotFoundError(f"LiveAct ascend_env_script not found: {ascend_env_script}")

        return LiveActRuntimeConfig(
            repo_path=repo_path,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            python_executable=python_executable,
            launcher=str(config.get("launcher", "torchrun")),
            nproc_per_node=int(config.get("nproc_per_node", 4)),
            num_processes=int(config.get("num_processes", config.get("nproc_per_node", 4))),
            accelerate_executable=LiveActPipeline._normalize_optional_string(config.get("accelerate_executable")),
            visible_devices=LiveActPipeline._normalize_optional_string(config.get("visible_devices")),
            ascend_env_script=ascend_env_script,
            platform=str(config.get("platform", "ascend_npu")),
        )

    @staticmethod
    def _append_flag(args: List[str], flag: str, enabled: bool) -> None:
        if enabled:
            args.append(flag)

    @staticmethod
    def _resolve_repo_relative_path(repo_path: Path, value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (repo_path / candidate)

    @staticmethod
    def _resolve_optional_repo_path(repo_path: Path, value: Any) -> Optional[Path]:
        text = LiveActPipeline._normalize_optional_string(value)
        if text is None:
            return None
        return LiveActPipeline._resolve_repo_relative_path(repo_path, text)

    @staticmethod
    def _normalize_optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_optional_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _first_visible_device(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return next((part.strip() for part in value.split(",") if part.strip()), None)

    @staticmethod
    def _text_context_cache_paths(input_json: Path) -> List[Path]:
        data = json.loads(input_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        paths = []
        for data_idx, item in enumerate(data):
            if not isinstance(item, dict) or "prompt" not in item:
                continue
            paths.append(
                LiveActPipeline._text_context_cache_path(
                    input_json,
                    data_idx,
                    str(item["prompt"]),
                    item.get("edit_prompt", {}),
                )
            )
        return paths

    @staticmethod
    def _text_context_cache_path(input_json: Path, data_idx: int, prompt: str, edit_prompts: Any) -> Path:
        cache_key = json.dumps(
            {
                "input_json": os.path.abspath(str(input_json)),
                "data_idx": data_idx,
                "prompt": prompt,
                "edit_prompts": edit_prompts,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return Path("/tmp") / f"liveact_text_ctx_{hashlib.sha1(cache_key).hexdigest()[:16]}.pt"

    @staticmethod
    def _input_json_digest(item: Dict[str, Any]) -> str:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]
