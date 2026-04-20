"""SoulX-FlashTalk wrapper pipeline backed by an external Ascend checkout."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.flashtalk.components import (
    DEFAULT_FLASHTALK_ASCEND_ENV_SCRIPT,
    DEFAULT_FLASHTALK_CKPT_DIR,
    DEFAULT_FLASHTALK_PROMPT,
    DEFAULT_FLASHTALK_REPO_PATH,
    DEFAULT_FLASHTALK_WAV2VEC_DIR,
    resolve_flashtalk_python,
)


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
    resource_hint={"min_vram_gb": 64, "dtype": "bf16", "machine": "8.92.7.86", "accelerator": "Ascend 910B2"},
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
            "audio_encode_mode",
            "cpu_offload",
            "max_chunks",
            "python_executable",
            "launcher",
            "nproc_per_node",
            "visible_devices",
            "ascend_env_script",
            "t5_quant",
            "t5_quant_dir",
            "wan_quant",
            "wan_quant_include",
            "wan_quant_exclude",
        ),
        default_config={
            "repo_path": DEFAULT_FLASHTALK_REPO_PATH,
            "ckpt_dir": DEFAULT_FLASHTALK_CKPT_DIR,
            "wav2vec_dir": DEFAULT_FLASHTALK_WAV2VEC_DIR,
            "audio_encode_mode": "stream",
            "seed": 9999,
            "launcher": "torchrun",
            "nproc_per_node": 8,
            "ascend_env_script": DEFAULT_FLASHTALK_ASCEND_ENV_SCRIPT,
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="video",
        maturity="beta",
        summary="SoulX-FlashTalk talking-head avatar generation via image plus audio on Ascend.",
        example=(
            "omnirt generate --task audio2video --model soulx-flashtalk-14b --image speaker.png "
            "--audio voice.wav --backend ascend --repo-path /home/<user>/SoulX-FlashTalk"
        ),
    ),
)
class FlashTalkPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._launch: Optional[FlashTalkLaunchConfig] = None

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
        repo_path = Path(str(req.config.get("repo_path", DEFAULT_FLASHTALK_REPO_PATH))).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"FlashTalk repo_path not found: {repo_path}")
        script_path = repo_path / "generate_video.py"
        if not script_path.exists():
            raise FileNotFoundError(f"FlashTalk entry script not found: {script_path}")

        ckpt_value = req.config.get("ckpt_dir") or req.config.get("model_path") or DEFAULT_FLASHTALK_CKPT_DIR
        wav2vec_value = req.config.get("wav2vec_dir", DEFAULT_FLASHTALK_WAV2VEC_DIR)
        ckpt_dir = self._resolve_repo_relative_path(repo_path, str(ckpt_value))
        wav2vec_dir = self._resolve_repo_relative_path(repo_path, str(wav2vec_value))
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"FlashTalk ckpt_dir not found: {ckpt_dir}")
        if not wav2vec_dir.exists():
            raise FileNotFoundError(f"FlashTalk wav2vec_dir not found: {wav2vec_dir}")

        output_dir = self.resolve_output_dir(req)
        seed = int(req.config.get("seed", 9999))
        save_file = output_dir / f"{req.model}-{seed}-{int(time.time() * 1000)}.mp4"
        launcher = str(req.config.get("launcher", "torchrun"))
        nproc_per_node = int(req.config.get("nproc_per_node", 8))
        t5_quant_dir_value = req.config.get("t5_quant_dir")
        t5_quant_dir = self._resolve_repo_relative_path(repo_path, str(t5_quant_dir_value)) if t5_quant_dir_value else None
        python_executable = str(req.config.get("python_executable") or resolve_flashtalk_python())
        ascend_env_script = self._normalize_optional_string(req.config.get("ascend_env_script", DEFAULT_FLASHTALK_ASCEND_ENV_SCRIPT))
        if python_executable and not Path(python_executable).expanduser().exists():
            raise FileNotFoundError(f"FlashTalk python_executable not found: {python_executable}")
        if ascend_env_script and not Path(ascend_env_script).expanduser().exists():
            raise FileNotFoundError(f"FlashTalk ascend_env_script not found: {ascend_env_script}")
        if t5_quant_dir is not None and not t5_quant_dir.exists():
            raise FileNotFoundError(f"FlashTalk t5_quant_dir not found: {t5_quant_dir}")

        launch = FlashTalkLaunchConfig(
            repo_path=repo_path,
            script_path=script_path,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            save_file=save_file,
            prompt=str(conditions["prompt"]),
            audio_encode_mode=str(req.config.get("audio_encode_mode", "stream")),
            cpu_offload=bool(req.config.get("cpu_offload", False)),
            max_chunks=int(req.config.get("max_chunks", 0)),
            seed=seed,
            python_executable=python_executable,
            launcher=launcher,
            nproc_per_node=nproc_per_node,
            visible_devices=self._normalize_optional_string(req.config.get("visible_devices")),
            ascend_env_script=ascend_env_script,
            t5_quant=self._normalize_optional_string(req.config.get("t5_quant")),
            t5_quant_dir=t5_quant_dir,
            wan_quant=self._normalize_optional_string(req.config.get("wan_quant")),
            wan_quant_include=self._normalize_optional_string(req.config.get("wan_quant_include")),
            wan_quant_exclude=self._normalize_optional_string(req.config.get("wan_quant_exclude")),
        )
        self._launch = launch
        return launch

    def denoise_loop(self, latents: FlashTalkLaunchConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        env = dict(os.environ)
        if latents.visible_devices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = latents.visible_devices
        if latents.launcher == "torchrun":
            env["GPU_NUM"] = str(latents.nproc_per_node)

        inner_command: List[str] = [latents.python_executable]
        if latents.launcher == "torchrun":
            inner_command.extend(["-m", "torch.distributed.run", f"--nproc_per_node={latents.nproc_per_node}"])
        elif latents.launcher != "python":
            raise ValueError(f"Unsupported FlashTalk launcher: {latents.launcher}")
        inner_command.extend(
            [
                str(latents.script_path),
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
        )
        if latents.cpu_offload:
            inner_command.append("--cpu_offload")
        if latents.t5_quant:
            inner_command.extend(["--t5_quant", latents.t5_quant])
        if latents.t5_quant_dir is not None:
            inner_command.extend(["--t5_quant_dir", str(latents.t5_quant_dir)])
        if latents.wan_quant:
            inner_command.extend(["--wan_quant", latents.wan_quant])
        if latents.wan_quant_include:
            inner_command.extend(["--wan_quant_include", latents.wan_quant_include])
        if latents.wan_quant_exclude:
            inner_command.extend(["--wan_quant_exclude", latents.wan_quant_exclude])

        shell_command = self._build_shell_command(
            repo_path=latents.repo_path,
            inner_command=inner_command,
            ascend_env_script=latents.ascend_env_script,
        )
        try:
            completed = subprocess.run(
                ["bash", "-lc", shell_command],
                check=True,
                cwd=str(latents.repo_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
            "visible_devices": latents.visible_devices,
            "ascend_env_script": latents.ascend_env_script,
            "t5_quant": latents.t5_quant,
            "t5_quant_dir": str(latents.t5_quant_dir) if latents.t5_quant_dir else None,
            "wan_quant": latents.wan_quant,
            "wan_quant_include": latents.wan_quant_include,
            "wan_quant_exclude": latents.wan_quant_exclude,
        }

    def _resolve_repo_relative_path(self, repo_path: Path, value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (repo_path / candidate)

    def _normalize_optional_string(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _build_shell_command(self, *, repo_path: Path, inner_command: List[str], ascend_env_script: Optional[str]) -> str:
        commands = [f"cd {shlex.quote(str(repo_path))}"]
        if ascend_env_script:
            commands.append(f"source {shlex.quote(ascend_env_script)}")
        commands.append(" ".join(shlex.quote(part) for part in inner_command))
        return " && ".join(commands)


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
