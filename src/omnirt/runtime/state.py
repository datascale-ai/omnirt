"""Runtime state persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from omnirt.runtime.manifest import RuntimeManifest
from omnirt.runtime.paths import runtime_state_dir


@dataclass(frozen=True)
class RuntimeState:
    name: str
    device: str
    profile: str
    manifest_path: str
    runtime_dir: str
    repo_path: str
    ckpt_dir: str
    wav2vec_dir: str
    env_script: str
    venv_activate: str
    python: str
    torchrun: str
    server_path: str
    nproc_per_node: int
    repo_marker_dir: str

    @classmethod
    def from_manifest(cls, manifest: RuntimeManifest) -> "RuntimeState":
        return cls(
            name=manifest.name,
            device=manifest.device,
            profile=manifest.profile,
            manifest_path=str(manifest.manifest_path),
            runtime_dir=str(manifest.runtime_dir),
            repo_path=str(manifest.repo_dir),
            ckpt_dir=manifest.ckpt_dir,
            wav2vec_dir=manifest.wav2vec_dir,
            env_script=str(manifest.env_script or ""),
            venv_activate=str(manifest.activate_path),
            python=str(manifest.python_path),
            torchrun=str(manifest.torchrun_path),
            server_path=str(manifest.server_path),
            nproc_per_node=manifest.nproc_per_node,
            repo_marker_dir=manifest.repo_marker_dir or "",
        )

    @property
    def state_path(self) -> Path:
        return runtime_state_path(self.name, self.device)

    def to_env(self, *, prefix: str | None = None) -> dict[str, str]:
        if prefix is None:
            prefix = f"OMNIRT_{self.name.upper()}_"
        env = {
            f"{prefix}REPO_PATH": self.repo_path,
            f"{prefix}SERVER_PATH": self.server_path,
            f"{prefix}CKPT_DIR": self.ckpt_dir,
            f"{prefix}WAV2VEC_DIR": self.wav2vec_dir,
            f"{prefix}DEVICE": self.device,
            f"{prefix}ENV_SCRIPT": self.env_script,
            f"{prefix}VENV_ACTIVATE": self.venv_activate,
            f"{prefix}PYTHON": self.python,
            f"{prefix}TORCHRUN": self.torchrun,
            f"{prefix}NPROC_PER_NODE": str(self.nproc_per_node),
        }
        if self.device == "ascend":
            env[f"{prefix}ASCEND_ENV_SCRIPT"] = self.env_script
        return env


def runtime_state_path(name: str, device: str) -> Path:
    return runtime_state_dir(name, device) / "state.yaml"


def write_state(state: RuntimeState) -> Path:
    path = state.state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(state), sort_keys=True), encoding="utf-8")
    return path


def load_state(name: str, device: str = "ascend") -> RuntimeState:
    path = runtime_state_path(name, device)
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return _state_from_mapping(data)


def _state_from_mapping(data: dict[str, Any]) -> RuntimeState:
    name = str(data["name"])
    return RuntimeState(
        name=name,
        device=str(data["device"]),
        profile=str(data.get("profile") or data["device"]),
        manifest_path=str(data["manifest_path"]),
        runtime_dir=str(data["runtime_dir"]),
        repo_path=str(data["repo_path"]),
        ckpt_dir=str(data.get("ckpt_dir") or ""),
        wav2vec_dir=str(data.get("wav2vec_dir") or ""),
        env_script=str(data.get("env_script") or ""),
        venv_activate=str(data["venv_activate"]),
        python=str(data["python"]),
        torchrun=str(data["torchrun"]),
        server_path=str(data["server_path"]),
        nproc_per_node=int(data["nproc_per_node"]),
        repo_marker_dir=str(data.get("repo_marker_dir") or ("flash_talk" if name == "flashtalk" else "")),
    )


def status_checks(state: RuntimeState) -> list[tuple[str, Path, bool]]:
    repo_path = Path(state.repo_path)
    checks = [
        ("state", state.state_path, state.state_path.exists()),
        ("runtime_dir", Path(state.runtime_dir), Path(state.runtime_dir).exists()),
        ("python", Path(state.python), Path(state.python).is_file()),
        ("torchrun", Path(state.torchrun), Path(state.torchrun).is_file()),
        ("repo_path", repo_path, repo_path.is_dir()),
        ("server_path", Path(state.server_path), Path(state.server_path).is_file()),
    ]
    if state.env_script:
        checks.append(("env_script", Path(state.env_script), Path(state.env_script).is_file()))
    if state.repo_marker_dir:
        checks.append((state.repo_marker_dir, repo_path / state.repo_marker_dir, (repo_path / state.repo_marker_dir).is_dir()))
    if state.ckpt_dir:
        checks.append(("ckpt_dir", _repo_relative(repo_path, state.ckpt_dir), _repo_relative(repo_path, state.ckpt_dir).is_dir()))
    if state.wav2vec_dir:
        checks.append(("wav2vec_dir", _repo_relative(repo_path, state.wav2vec_dir), _repo_relative(repo_path, state.wav2vec_dir).is_dir()))
    return checks


def _repo_relative(repo_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_path / path
