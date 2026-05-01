"""SoulX-FlashHead wrapper pipeline backed by an external checkout."""

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
from omnirt.models.flashhead.components import flashhead_setting


@dataclass(frozen=True)
class FlashHeadRuntimeConfig:
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


@dataclass(frozen=True)
class FlashHeadLaunchConfig:
    repo_path: Path
    script_path: Path
    ckpt_dir: Path
    wav2vec_dir: Path
    save_file: Path
    model_type: str
    audio_encode_mode: str
    sample_steps: Optional[int]
    vae_2d_split: bool
    latent_carry: bool
    npu_fusion_attention: bool
    profile: bool
    device: str
    python_executable: str
    launcher: str
    nproc_per_node: int
    num_processes: int
    accelerate_executable: Optional[str]
    visible_devices: Optional[str]
    ascend_env_script: Optional[str]


@register_model(
    id="soulx-flashhead-1.3b",
    task="audio2video",
    default_backend="ascend",
    execution_mode="subprocess",
    resource_hint={
        "min_vram_gb": 48,
        "vram_scope": "aggregate",
        "dtype": "bf16",
        "accelerator": "Ascend 910B2",
    },
    capabilities=ModelCapabilities(
        required_inputs=("image", "audio"),
        optional_inputs=(),
        supported_config=(
            "model_path",
            "repo_path",
            "ckpt_dir",
            "wav2vec_dir",
            "model_type",
            "seed",
            "output_dir",
            "audio_encode_mode",
            "sample_steps",
            "vae_2d_split",
            "latent_carry",
            "npu_fusion_attention",
            "profile",
            "device",
            "python_executable",
            "launcher",
            "nproc_per_node",
            "num_processes",
            "accelerate_executable",
            "visible_devices",
            "ascend_env_script",
        ),
        default_config={
            "model_type": "pro",
            "audio_encode_mode": "stream",
            "sample_steps": 2,
            "vae_2d_split": True,
            "latent_carry": False,
            "npu_fusion_attention": True,
            "device": "npu",
            "launcher": "torchrun",
            "nproc_per_node": 4,
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="video",
        maturity="beta",
        chain_role="avatar-render",
        realtime=False,
        summary="SoulX-FlashHead low-latency talking-head generation via image plus audio.",
        example=(
            "OMNIRT_FLASHHEAD_REPO_PATH=/path/to/SoulX-FlashHead "
            "omnirt generate --task audio2video --model soulx-flashhead-1.3b --image speaker.png "
            "--audio voice.wav --backend ascend"
        ),
    ),
)
class FlashHeadPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._launch: Optional[FlashHeadLaunchConfig] = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        image_path = Path(str(req.inputs.get("image", ""))).expanduser()
        audio_path = Path(str(req.inputs.get("audio", ""))).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        return {"image_path": image_path, "audio_path": audio_path}

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> FlashHeadLaunchConfig:
        runtime_config = self.resolve_runtime_config(req.config)
        script_path = runtime_config.repo_path / "generate_video.py"
        if not script_path.exists():
            raise FileNotFoundError(f"FlashHead entry script not found: {script_path}")
        output_dir = self.resolve_output_dir(req)
        seed = int(req.config.get("seed", 9999))
        save_file = output_dir / f"{req.model}-{seed}-{int(time.time() * 1000)}.mp4"

        launch = FlashHeadLaunchConfig(
            repo_path=runtime_config.repo_path,
            script_path=script_path,
            ckpt_dir=runtime_config.ckpt_dir,
            wav2vec_dir=runtime_config.wav2vec_dir,
            save_file=save_file,
            model_type=str(req.config.get("model_type", "pro")),
            audio_encode_mode=str(req.config.get("audio_encode_mode", "stream")),
            sample_steps=self._optional_int(req.config.get("sample_steps", 2)),
            vae_2d_split=self._config_bool(req.config, "vae_2d_split", True),
            latent_carry=self._config_bool(req.config, "latent_carry", False),
            npu_fusion_attention=self._config_bool(req.config, "npu_fusion_attention", True),
            profile=self._config_bool(req.config, "profile", False),
            device=str(req.config.get("device") or self._default_device()),
            python_executable=runtime_config.python_executable,
            launcher=runtime_config.launcher,
            nproc_per_node=runtime_config.nproc_per_node,
            num_processes=runtime_config.num_processes,
            accelerate_executable=runtime_config.accelerate_executable,
            visible_devices=runtime_config.visible_devices,
            ascend_env_script=runtime_config.ascend_env_script,
        )
        self._launch = launch
        return launch

    def denoise_loop(self, latents: FlashHeadLaunchConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        env = dict(os.environ)
        if latents.visible_devices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = latents.visible_devices
        env["FLASHHEAD_DEVICE"] = latents.device
        env["FLASHHEAD_VAE_2D_SPLIT"] = self._env_bool(latents.vae_2d_split)
        env["FLASHHEAD_LATENT_CARRY"] = self._env_bool(latents.latent_carry)
        env["FLASHHEAD_NPU_FUSION_ATTENTION"] = self._env_bool(latents.npu_fusion_attention)
        env["FLASHHEAD_PROFILE"] = self._env_bool(latents.profile)
        if latents.sample_steps is not None:
            env["FLASHHEAD_SAMPLE_STEPS"] = str(latents.sample_steps)

        script_args: List[str] = [
            "--ckpt_dir",
            str(latents.ckpt_dir),
            "--wav2vec_dir",
            str(latents.wav2vec_dir),
            "--model_type",
            latents.model_type,
            "--cond_image",
            str(conditions["image_path"]),
            "--audio_path",
            str(conditions["audio_path"]),
            "--audio_encode_mode",
            latents.audio_encode_mode,
            "--save_file",
            str(latents.save_file),
        ]
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
                f"FlashHead launch failed with exit code {exc.returncode}."
                + (f"\nRecent output:\n{detail}" if detail else "")
            ) from exc

        return {
            "save_file": latents.save_file,
            "stdout": completed.stdout,
            "model_type": latents.model_type,
            "sample_steps": latents.sample_steps,
        }

    def decode(self, latents: Any) -> Any:
        return latents

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        save_file = Path(raw["save_file"])
        if not save_file.exists():
            raise FileNotFoundError(f"FlashHead output file missing after generation: {save_file}")
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

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: FlashHeadLaunchConfig) -> Dict[str, Any]:
        return {
            "repo_path": str(latents.repo_path),
            "model_path": str(latents.ckpt_dir),
            "ckpt_dir": str(latents.ckpt_dir),
            "wav2vec_dir": str(latents.wav2vec_dir),
            "model_type": latents.model_type,
            "output_dir": str(latents.save_file.parent),
            "audio_encode_mode": latents.audio_encode_mode,
            "sample_steps": latents.sample_steps,
            "vae_2d_split": latents.vae_2d_split,
            "latent_carry": latents.latent_carry,
            "npu_fusion_attention": latents.npu_fusion_attention,
            "profile": latents.profile,
            "device": latents.device,
            "python_executable": latents.python_executable,
            "launcher": latents.launcher,
            "nproc_per_node": latents.nproc_per_node,
            "num_processes": latents.num_processes,
            "accelerate_executable": latents.accelerate_executable,
            "visible_devices": latents.visible_devices,
            "ascend_env_script": latents.ascend_env_script,
        }

    @staticmethod
    def resolve_runtime_config(config: Dict[str, Any]) -> FlashHeadRuntimeConfig:
        repo_path_value = config.get("repo_path") or flashhead_setting("repo_path", required=True)
        repo_path = Path(str(repo_path_value)).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"FlashHead repo_path not found: {repo_path}")

        ckpt_value = config.get("ckpt_dir") or config.get("model_path") or flashhead_setting("ckpt_dir", required=True)
        wav2vec_value = config.get("wav2vec_dir") or flashhead_setting("wav2vec_dir", required=True)
        ckpt_dir = FlashHeadPipeline._resolve_repo_relative_path(repo_path, str(ckpt_value))
        wav2vec_dir = FlashHeadPipeline._resolve_repo_relative_path(repo_path, str(wav2vec_value))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"FlashHead ckpt_dir not found: {ckpt_dir}")
        if not wav2vec_dir.exists():
            raise FileNotFoundError(f"FlashHead wav2vec_dir not found: {wav2vec_dir}")

        launcher = str(config.get("launcher", "torchrun"))
        nproc_per_node = int(config.get("nproc_per_node", 4))
        num_processes = int(config.get("num_processes", nproc_per_node))
        python_executable_value = config.get("python_executable") or flashhead_setting(
            "python_executable", required=True
        )
        python_executable = str(python_executable_value)
        accelerate_executable = FlashHeadPipeline._normalize_optional_string(config.get("accelerate_executable"))
        ascend_env_override = FlashHeadPipeline._normalize_optional_string(config.get("ascend_env_script"))
        ascend_env_script = ascend_env_override or flashhead_setting("ascend_env_script")
        if python_executable and not Path(python_executable).expanduser().exists():
            raise FileNotFoundError(f"FlashHead python_executable not found: {python_executable}")
        if ascend_env_script and not Path(ascend_env_script).expanduser().exists():
            raise FileNotFoundError(f"FlashHead ascend_env_script not found: {ascend_env_script}")

        return FlashHeadRuntimeConfig(
            repo_path=repo_path,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            python_executable=python_executable,
            launcher=launcher,
            nproc_per_node=nproc_per_node,
            num_processes=num_processes,
            accelerate_executable=accelerate_executable,
            visible_devices=FlashHeadPipeline._normalize_optional_string(config.get("visible_devices")),
            ascend_env_script=ascend_env_script,
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

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
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

    def _default_device(self) -> str:
        device_name = str(getattr(self.runtime, "device_name", "") or "").strip().lower()
        if device_name in {"cuda", "npu"}:
            return device_name
        backend_name = str(getattr(self.runtime, "name", "") or "").strip().lower()
        if backend_name == "cuda":
            return "cuda"
        return "npu"


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
