from __future__ import annotations

from pathlib import Path

from omnirt.launcher import resolve_config_device_map, resolve_device_map, resolve_devices, resolve_launcher


def test_resolve_device_map_supports_balanced_and_mapping() -> None:
    assert resolve_device_map("balanced") == "balanced"
    assert resolve_device_map('{"unet": 0, "vae": 1}') == {"unet": 0, "vae": 1}
    assert resolve_device_map("unet:0, vae:cuda:1") == {"unet": 0, "vae": "cuda:1"}


def test_resolve_config_device_map_infers_balanced_from_devices() -> None:
    assert resolve_devices("cuda:0,cuda:1") == ("cuda:0", "cuda:1")
    assert resolve_config_device_map({"devices": "cuda:0,cuda:1"}) == "balanced"
    assert resolve_config_device_map({"devices": "cuda:0"}) is None


def test_torchrun_launcher_builds_distributed_command() -> None:
    launcher = resolve_launcher("torchrun")

    command = launcher.build_command(
        Path("/tmp/generate_video.py"),
        python_executable="/tmp/python",
        script_args=["--foo", "bar"],
        config={"nproc_per_node": 8},
    )

    assert command[:4] == ["/tmp/python", "-m", "torch.distributed.run", "--nproc_per_node=8"]
    assert command[-3:] == ["/tmp/generate_video.py", "--foo", "bar"]
