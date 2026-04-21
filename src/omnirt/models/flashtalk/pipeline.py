"""SoulX-FlashTalk wrapper pipeline backed by an external Ascend checkout."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.launcher import resolve_launcher
from omnirt.models.flashtalk.components import (
    DEFAULT_FLASHTALK_PROMPT,
    flashtalk_setting,
)
from omnirt.models.flashtalk.resident_launch import (
    build_flashtalk_resident_worker_command,
    build_resident_worker_env,
    reserve_local_port,
)
from omnirt.workers import GrpcResidentWorkerProxy, ManagedGrpcResidentWorkerProxy


@dataclass(frozen=True)
class FlashTalkRuntimeConfig:
    resident_target: Optional[str]
    repo_path: Path
    ckpt_dir: Path
    wav2vec_dir: Path
    cpu_offload: bool
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]
    t5_quant: Optional[str]
    t5_quant_dir: Optional[Path]
    wan_quant: Optional[str]
    wan_quant_include: Optional[str]
    wan_quant_exclude: Optional[str]


@dataclass(frozen=True)
class FlashTalkLaunchConfig:
    repo_path: Path
    script_path: Path
    ckpt_dir: Path
    wav2vec_dir: Path
    save_file: Path
    prompt: str
    audio_encode_mode: str
    cpu_offload: bool
    max_chunks: int
    seed: int
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]
    t5_quant: Optional[str]
    t5_quant_dir: Optional[Path]
    wan_quant: Optional[str]
    wan_quant_include: Optional[str]
    wan_quant_exclude: Optional[str]


@register_model(
    id="soulx-flashtalk-14b",
    task="audio2video",
    default_backend="ascend",
    execution_mode="persistent_worker",
    resource_hint={
        "min_vram_gb": 64,
        "vram_scope": "aggregate",
        "dtype": "bf16",
        "accelerator": "Ascend 910B2",
    },
    capabilities=ModelCapabilities(
        required_inputs=("image", "audio"),
        optional_inputs=("prompt",),
        supported_config=(
            "model_path",
            "repo_path",
            "ckpt_dir",
            "wav2vec_dir",
            "resident_target",
            "resident_autostart",
            "seed",
            "output_dir",
            "audio_encode_mode",
            "cpu_offload",
            "max_chunks",
            "python_executable",
            "launcher",
            "nproc_per_node",
            "num_processes",
            "accelerate_executable",
            "visible_devices",
            "ascend_env_script",
            "t5_quant",
            "t5_quant_dir",
            "wan_quant",
            "wan_quant_include",
            "wan_quant_exclude",
        ),
        default_config={
            "audio_encode_mode": "stream",
            "seed": 9999,
            "launcher": "torchrun",
            "nproc_per_node": 8,
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="video",
        maturity="beta",
        summary="SoulX-FlashTalk talking-head avatar generation via image plus audio on Ascend.",
        example=(
            "OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk "
            "omnirt generate --task audio2video --model soulx-flashtalk-14b --image speaker.png "
            "--audio voice.wav --backend ascend"
        ),
    ),
)
class FlashTalkPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._launch: Optional[FlashTalkLaunchConfig] = None

    @classmethod
    def create_persistent_worker(cls, *, runtime, model_spec, config, adapters):
        resident_target = cls._normalize_optional_string(config.get("resident_target"))
        runtime_config = None
        if resident_target:
            autostart = bool(config.get("resident_autostart", False))
        else:
            runtime_config = cls.resolve_runtime_config(config)
            distributed_requested = (
                runtime_config.launcher != "python"
                or runtime_config.nproc_per_node > 1
                or runtime_config.num_processes > 1
            )
            if distributed_requested:
                resident_target = f"127.0.0.1:{reserve_local_port()}"
            autostart = bool(distributed_requested)
        if resident_target:
            if autostart:
                if runtime_config is None:
                    runtime_config = cls.resolve_runtime_config(config)
                project_root = Path(__file__).resolve().parents[4]
                worker_id = f"flashtalk-resident-{resident_target.rsplit(':', 1)[-1]}"
                return ManagedGrpcResidentWorkerProxy(
                    resident_target,
                    command=build_flashtalk_resident_worker_command(
                        runtime_config=runtime_config,
                        backend_name=getattr(runtime, "name", "auto"),
                        host=resident_target.rsplit(":", 1)[0],
                        port=int(resident_target.rsplit(":", 1)[1]),
                        worker_id=worker_id,
                    ),
                    cwd=project_root,
                    env=build_resident_worker_env(project_root=project_root),
                    env_script=runtime_config.ascend_env_script,
                    log_file=project_root / "outputs" / f"{worker_id}.log",
                )
            return GrpcResidentWorkerProxy(resident_target)
        from omnirt.models.flashtalk.resident_worker import FlashTalkResidentWorker

        return FlashTalkResidentWorker(runtime=runtime, model_spec=model_spec, config=config, adapters=adapters)

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        image_path = Path(str(req.inputs.get("image", ""))).expanduser()
        audio_path = Path(str(req.inputs.get("audio", ""))).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        prompt = str(req.inputs.get("prompt") or DEFAULT_FLASHTALK_PROMPT)
        return {"image_path": image_path, "audio_path": audio_path, "prompt": prompt}

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> FlashTalkLaunchConfig:
        runtime_config = self.resolve_runtime_config(req.config)
        script_path = runtime_config.repo_path / "generate_video.py"
        if not script_path.exists():
            raise FileNotFoundError(f"FlashTalk entry script not found: {script_path}")
        output_dir = self.resolve_output_dir(req)
        seed = int(req.config.get("seed", 9999))
        save_file = output_dir / f"{req.model}-{seed}-{int(time.time() * 1000)}.mp4"

        launch = FlashTalkLaunchConfig(
            repo_path=runtime_config.repo_path,
            script_path=script_path,
            ckpt_dir=runtime_config.ckpt_dir,
            wav2vec_dir=runtime_config.wav2vec_dir,
            save_file=save_file,
            prompt=str(conditions["prompt"]),
            audio_encode_mode=str(req.config.get("audio_encode_mode", "stream")),
            cpu_offload=runtime_config.cpu_offload,
            max_chunks=int(req.config.get("max_chunks", 0)),
            seed=seed,
            python_executable=runtime_config.python_executable,
            launcher=runtime_config.launcher,
            nproc_per_node=runtime_config.nproc_per_node,
            num_processes=runtime_config.num_processes,
            accelerate_executable=runtime_config.accelerate_executable,
            visible_devices=runtime_config.visible_devices,
            ascend_env_script=runtime_config.ascend_env_script,
            t5_quant=runtime_config.t5_quant,
            t5_quant_dir=runtime_config.t5_quant_dir,
            wan_quant=runtime_config.wan_quant,
            wan_quant_include=runtime_config.wan_quant_include,
            wan_quant_exclude=runtime_config.wan_quant_exclude,
        )
        self._launch = launch
        return launch

    def denoise_loop(self, latents: FlashTalkLaunchConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        env = dict(os.environ)
        if latents.visible_devices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = latents.visible_devices
        if latents.launcher == "torchrun":
            env["GPU_NUM"] = str(latents.nproc_per_node)

        script_args: List[str] = [
            "--ckpt_dir",
            str(latents.ckpt_dir),
            "--wav2vec_dir",
            str(latents.wav2vec_dir),
            "--save_file",
            str(latents.save_file),
            "--base_seed",
            str(latents.seed),
            "--input_prompt",
            latents.prompt,
            "--cond_image",
            str(conditions["image_path"]),
            "--audio_path",
            str(conditions["audio_path"]),
            "--audio_encode_mode",
            latents.audio_encode_mode,
            "--max_chunks",
            str(latents.max_chunks),
        ]
        if latents.cpu_offload:
            script_args.append("--cpu_offload")
        if latents.t5_quant:
            script_args.extend(["--t5_quant", latents.t5_quant])
        if latents.t5_quant_dir is not None:
            script_args.extend(["--t5_quant_dir", str(latents.t5_quant_dir)])
        if latents.wan_quant:
            script_args.extend(["--wan_quant", latents.wan_quant])
        if latents.wan_quant_include:
            script_args.extend(["--wan_quant_include", latents.wan_quant_include])
        if latents.wan_quant_exclude:
            script_args.extend(["--wan_quant_exclude", latents.wan_quant_exclude])

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
        try:
            completed = launcher.launch(
                cwd=latents.repo_path,
                command=inner_command,
                env=env,
                env_script=latents.ascend_env_script,
            )
        except subprocess.CalledProcessError as exc:
            tail = (exc.stdout or "").strip().splitlines()[-20:]
            detail = "\n".join(tail)
            raise RuntimeError(
                f"FlashTalk launch failed with exit code {exc.returncode}."
                + (f"\nRecent output:\n{detail}" if detail else "")
            ) from exc

        return {
            "save_file": latents.save_file,
            "stdout": completed.stdout,
            "seed": latents.seed,
            "prompt": latents.prompt,
        }

    def decode(self, latents: Any) -> Any:
        return latents

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        save_file = Path(raw["save_file"])
        if not save_file.exists():
            raise FileNotFoundError(f"FlashTalk output file missing after generation: {save_file}")
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

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: FlashTalkLaunchConfig) -> Dict[str, Any]:
        return {
            "repo_path": str(latents.repo_path),
            "model_path": str(latents.ckpt_dir),
            "ckpt_dir": str(latents.ckpt_dir),
            "wav2vec_dir": str(latents.wav2vec_dir),
            "seed": latents.seed,
            "output_dir": str(latents.save_file.parent),
            "audio_encode_mode": latents.audio_encode_mode,
            "cpu_offload": latents.cpu_offload,
            "max_chunks": latents.max_chunks,
            "python_executable": latents.python_executable,
            "launcher": latents.launcher,
            "nproc_per_node": latents.nproc_per_node,
            "num_processes": latents.num_processes,
            "accelerate_executable": latents.accelerate_executable,
            "visible_devices": latents.visible_devices,
            "ascend_env_script": latents.ascend_env_script,
            "t5_quant": latents.t5_quant,
            "t5_quant_dir": str(latents.t5_quant_dir) if latents.t5_quant_dir else None,
            "wan_quant": latents.wan_quant,
            "wan_quant_include": latents.wan_quant_include,
            "wan_quant_exclude": latents.wan_quant_exclude,
        }

    @staticmethod
    def resolve_runtime_config(config: Dict[str, Any]) -> FlashTalkRuntimeConfig:
        repo_path_value = config.get("repo_path") or flashtalk_setting("repo_path", required=True)
        repo_path = Path(str(repo_path_value)).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"FlashTalk repo_path not found: {repo_path}")

        ckpt_value = (
            config.get("ckpt_dir")
            or config.get("model_path")
            or flashtalk_setting("ckpt_dir", required=True)
        )
        wav2vec_value = config.get("wav2vec_dir") or flashtalk_setting("wav2vec_dir", required=True)
        ckpt_dir = FlashTalkPipeline._resolve_repo_relative_path(repo_path, str(ckpt_value))
        wav2vec_dir = FlashTalkPipeline._resolve_repo_relative_path(repo_path, str(wav2vec_value))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"FlashTalk ckpt_dir not found: {ckpt_dir}")
        if not wav2vec_dir.exists():
            raise FileNotFoundError(f"FlashTalk wav2vec_dir not found: {wav2vec_dir}")

        launcher = str(config.get("launcher", "torchrun"))
        nproc_per_node = int(config.get("nproc_per_node", 8))
        num_processes = int(config.get("num_processes", nproc_per_node))
        t5_quant_dir_value = config.get("t5_quant_dir")
        t5_quant_dir = (
            FlashTalkPipeline._resolve_repo_relative_path(repo_path, str(t5_quant_dir_value))
            if t5_quant_dir_value
            else None
        )
        python_executable_value = config.get("python_executable") or flashtalk_setting(
            "python_executable", required=True
        )
        python_executable = str(python_executable_value)
        accelerate_executable = FlashTalkPipeline._normalize_optional_string(config.get("accelerate_executable"))
        ascend_env_override = FlashTalkPipeline._normalize_optional_string(config.get("ascend_env_script"))
        ascend_env_script = ascend_env_override or flashtalk_setting("ascend_env_script")
        if python_executable and not Path(python_executable).expanduser().exists():
            raise FileNotFoundError(f"FlashTalk python_executable not found: {python_executable}")
        if ascend_env_script and not Path(ascend_env_script).expanduser().exists():
            raise FileNotFoundError(f"FlashTalk ascend_env_script not found: {ascend_env_script}")
        if t5_quant_dir is not None and not t5_quant_dir.exists():
            raise FileNotFoundError(f"FlashTalk t5_quant_dir not found: {t5_quant_dir}")

        return FlashTalkRuntimeConfig(
            resident_target=FlashTalkPipeline._normalize_optional_string(config.get("resident_target")),
            repo_path=repo_path,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            cpu_offload=bool(config.get("cpu_offload", False)),
            python_executable=python_executable,
            launcher=launcher,
            nproc_per_node=nproc_per_node,
            num_processes=num_processes,
            accelerate_executable=accelerate_executable,
            visible_devices=FlashTalkPipeline._normalize_optional_string(config.get("visible_devices")),
            ascend_env_script=ascend_env_script,
            t5_quant=FlashTalkPipeline._normalize_optional_string(config.get("t5_quant")),
            t5_quant_dir=t5_quant_dir,
            wan_quant=FlashTalkPipeline._normalize_optional_string(config.get("wan_quant")),
            wan_quant_include=FlashTalkPipeline._normalize_optional_string(config.get("wan_quant_include")),
            wan_quant_exclude=FlashTalkPipeline._normalize_optional_string(config.get("wan_quant_exclude")),
        )

    @staticmethod
    def _resolve_repo_relative_path(repo_path: Path, value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (repo_path / candidate)

    @staticmethod
    def _normalize_optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def probe_video_file(path: Path) -> tuple[int, int, int]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise DependencyUnavailableError("imageio is required to inspect generated video artifacts.") from exc

    reader = imageio.get_reader(str(path))
    try:
        first_frame = reader.get_next_data()
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        try:
            num_frames = int(reader.count_frames())
        except Exception:
            metadata = reader.get_meta_data()
            num_frames = int(metadata.get("nframes") or 0)
        return width, height, num_frames
    finally:
        reader.close()
