from __future__ import annotations

from pathlib import Path

from omnirt.runtime.manifest import load_manifest
from omnirt.runtime.paths import project_root
from omnirt.runtime.state import RuntimeState, load_state, write_state


def test_load_flashtalk_runtime_manifest_uses_project_home_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OMNIRT_HOME", raising=False)

    manifest = load_manifest("flashtalk", "ascend")

    assert manifest.name == "flashtalk"
    assert manifest.device == "ascend"
    assert manifest.runtime_dir == project_root() / ".omnirt" / "runtimes" / "flashtalk" / "ascend"
    assert manifest.repo_dir == project_root() / ".omnirt" / "model-repos" / "SoulX-FlashTalk"
    assert manifest.ckpt_dir == "models/SoulX-FlashTalk-14B"
    assert manifest.server_path == Path("model_backends/flashtalk/flashtalk_ws_server.py").resolve()
    assert manifest.repo_marker_dir == "flash_talk"


def test_load_musetalk_cuda_runtime_manifest(monkeypatch) -> None:
    monkeypatch.delenv("OMNIRT_HOME", raising=False)
    monkeypatch.delenv("OMNIRT_MUSETALK_REPO", raising=False)

    manifest = load_manifest("musetalk", "cuda")

    assert manifest.name == "musetalk"
    assert manifest.device == "cuda"
    assert manifest.repo_url == "https://github.com/TMElyralab/MuseTalk.git"
    assert manifest.repo_marker_dir == "musetalk"
    assert manifest.checkpoint_url is None
    assert manifest.wav2vec_repo_id is None
    assert manifest.python_version == "3.10"
    assert manifest.repo_dir == (project_root() / ".omnirt" / "model-repos" / "MuseTalk").resolve()
    assert manifest.server_path == Path("model_backends/musetalk/musetalk_ws_server.py").resolve()


def test_manifest_overrides_runtime_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))

    manifest = load_manifest("flashtalk", "ascend").with_overrides(
        repo_dir=tmp_path / "SoulX-FlashTalk",
        ckpt_dir=tmp_path / "models" / "SoulX-FlashTalk-14B",
        wav2vec_dir=tmp_path / "models" / "chinese-wav2vec2-base",
    )

    assert manifest.repo_dir == tmp_path / "SoulX-FlashTalk"
    assert manifest.resolved_ckpt_dir == tmp_path / "models" / "SoulX-FlashTalk-14B"
    assert manifest.resolved_wav2vec_dir == tmp_path / "models" / "chinese-wav2vec2-base"


def test_clone_or_update_skips_nonempty_dir_without_git(monkeypatch, tmp_path: Path) -> None:
    """Pre-populated checkpoint dirs (e.g. symlinks to shared weights) must not require .git."""
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    manifest = load_manifest("flashtalk", "ascend")
    installer = RuntimeInstaller(manifest)
    ckpt = manifest.resolved_ckpt_dir
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "placeholder.bin").write_bytes(b"x")

    installer._clone_or_update(
        "https://example.com/weights.git",
        ckpt,
        update=True,
        marker_dir=None,
        label="FlashTalk checkpoint",
    )

    assert not (ckpt / ".git").exists()
    assert not any("git" in c and "clone" in c for c in installer.commands)


def test_plan_skips_existing_repo_checkpoint_and_wav2vec(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    manifest = load_manifest("flashtalk", "ascend")
    (manifest.repo_dir / "flash_talk").mkdir(parents=True)
    manifest.resolved_ckpt_dir.mkdir(parents=True)
    (manifest.resolved_ckpt_dir / "model.safetensors").write_bytes(b"x")
    manifest.resolved_wav2vec_dir.mkdir(parents=True)
    (manifest.resolved_wav2vec_dir / "config.json").write_text("{}", encoding="utf-8")

    commands = RuntimeInstaller(manifest).plan_commands(update=True)

    assert ["skip", "SoulX-FlashTalk checkout", str(manifest.repo_dir), "contains flash_talk/"] in commands
    assert ["skip", "FlashTalk checkpoint", str(manifest.resolved_ckpt_dir), "already exists"] in commands
    assert ["skip", "wav2vec", str(manifest.resolved_wav2vec_dir), "already exists"] in commands


def test_musetalk_plan_only_manages_repo_and_venv(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    manifest = load_manifest("musetalk", "cuda")
    (manifest.repo_dir / "musetalk").mkdir(parents=True)

    commands = RuntimeInstaller(manifest).plan_commands(update=True)

    assert ["skip", "MuseTalk checkout", str(manifest.repo_dir), "contains musetalk/"] in commands
    assert not any("checkpoint" in " ".join(command) for command in commands)
    assert not any("wav2vec" in command for command in commands for command in command)



def test_musetalk_plan_installs_torch_before_other_requirements_with_uv(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    manifest = load_manifest("musetalk", "cuda")

    commands = RuntimeInstaller(manifest).plan_commands(update=True)
    assert ["/usr/bin/uv", "venv", "--seed", "--python", "3.10", str(manifest.venv_dir)] in commands
    pip_installs = [
        command
        for command in commands
        if command[:4] == [str(manifest.python_path), "-m", "pip", "install"]
    ]
    uv_installs = [command for command in commands if command[:3] == ["/usr/bin/uv", "pip", "install"]]

    assert len(pip_installs) == 3
    assert pip_installs[1][:6] == [str(manifest.python_path), "-m", "pip", "install", "--timeout", "120"]
    assert "--no-deps" in pip_installs[1]
    assert "torch==2.0.1+cu118" in pip_installs[1]
    assert "torchvision==0.15.2+cu118" in pip_installs[1]
    assert "torchaudio==2.0.2+cu118" in pip_installs[1]
    assert "-r" not in pip_installs[1]
    assert "-r" in pip_installs[2]
    assert "--no-build-isolation" in pip_installs[2]
    assert not uv_installs


def test_musetalk_plan_falls_back_to_pip_when_uv_is_missing(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("shutil.which", lambda name: None)
    manifest = load_manifest("musetalk", "cuda")

    commands = RuntimeInstaller(manifest).plan_commands(update=True)
    pip_installs = [
        command
        for command in commands
        if command[:4] == [str(manifest.python_path), "-m", "pip", "install"]
    ]

    assert len(pip_installs) == 3
    assert "--no-deps" in pip_installs[1]
    assert "torch==2.0.1+cu118" in pip_installs[1]
    assert "torchvision==0.15.2+cu118" in pip_installs[1]
    assert "torchaudio==2.0.2+cu118" in pip_installs[1]
    assert "-r" in pip_installs[2]
    assert "--no-build-isolation" in pip_installs[2]


def test_install_requirements_retries_transient_failures(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    manifest = load_manifest("musetalk", "cuda")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    RuntimeInstaller(manifest)._install_requirements()

    assert calls[0] == calls[1]


def test_run_uses_configurable_install_timeout(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMNIRT_RUNTIME_INSTALL_TIMEOUT", "7")
    manifest = load_manifest("musetalk", "cuda")
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    RuntimeInstaller(manifest)._run(["python", "-m", "pip", "install", "torch"])

    assert seen["timeout"] == 7


def test_plan_recreate_venv_keeps_model_resources(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    manifest = load_manifest("flashtalk", "ascend")
    manifest.python_path.parent.mkdir(parents=True)
    manifest.python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    manifest.resolved_ckpt_dir.mkdir(parents=True)
    (manifest.resolved_ckpt_dir / "model.safetensors").write_bytes(b"x")

    commands = RuntimeInstaller(manifest).plan_commands(update=False, recreate_venv=True)

    assert ["recreate-venv", str(manifest.venv_dir)] in commands
    assert ["skip", "FlashTalk checkpoint", str(manifest.resolved_ckpt_dir), "already exists"] in commands


def test_recreate_venv_does_not_delete_model_resources(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import RuntimeInstaller

    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    manifest = load_manifest("flashtalk", "ascend")
    manifest.venv_dir.mkdir(parents=True)
    (manifest.venv_dir / "old").write_text("old", encoding="utf-8")
    manifest.resolved_ckpt_dir.mkdir(parents=True)
    model_file = manifest.resolved_ckpt_dir / "model.safetensors"
    model_file.write_bytes(b"x")
    installer = RuntimeInstaller(manifest)
    installer._run = lambda command, **kwargs: installer.commands.append(command)  # type: ignore[method-assign]

    installer._prepare_venv(recreate=True)

    assert not (manifest.venv_dir / "old").exists()
    assert model_file.exists()
    assert str(manifest.venv_dir) in installer.commands[0]


def test_patch_soulx_t5_import_default_device(monkeypatch, tmp_path: Path) -> None:
    from omnirt.runtime.installer import _patch_soulx_wan_t5_for_cpu_torch

    repo = tmp_path / "SoulX-FlashTalk"
    t5 = repo / "flash_talk" / "wan" / "modules" / "t5.py"
    t5.parent.mkdir(parents=True)
    t5.write_text(
        "import torch\n\n__all__ = ['T5EncoderModel']\n\nclass T5EncoderModel:\n"
        "    def __init__(self, device=torch.cuda.current_device()):\n"
        "        self.device = device\n",
        encoding="utf-8",
    )

    assert _patch_soulx_wan_t5_for_cpu_torch(repo) is True
    text = t5.read_text(encoding="utf-8")
    assert "def _omnirt_default_torch_device_index()" in text
    assert "device=torch.cuda.current_device()" not in text
    assert "_omnirt_default_torch_device_index()" in text
    assert _patch_soulx_wan_t5_for_cpu_torch(repo) is False


def test_runtime_state_roundtrip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNIRT_HOME", str(tmp_path / "home"))
    manifest = load_manifest("flashtalk", "ascend")
    state = RuntimeState.from_manifest(manifest)

    path = write_state(state)
    loaded = load_state("flashtalk", "ascend")

    assert path == tmp_path / "home" / "runtimes" / "flashtalk" / "ascend" / "state.yaml"
    assert loaded == state
    assert loaded.to_env()["OMNIRT_FLASHTALK_REPO_PATH"] == str(manifest.repo_dir)
