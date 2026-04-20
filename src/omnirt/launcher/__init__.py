"""Launcher exports."""

from __future__ import annotations

from omnirt.launcher.accelerate import AccelerateLauncher
from omnirt.launcher.base import Launcher
from omnirt.launcher.device_map import DEVICE_MAP_CONFIG_KEYS, resolve_config_device_map, resolve_device_map, resolve_devices
from omnirt.launcher.inprocess import InProcessLauncher
from omnirt.launcher.torchrun import TorchrunLauncher


def resolve_launcher(name: str) -> Launcher:
    normalized = str(name).strip().lower()
    if normalized == "python":
        return InProcessLauncher()
    if normalized == "torchrun":
        return TorchrunLauncher()
    if normalized == "accelerate":
        return AccelerateLauncher()
    raise ValueError(f"Unsupported launcher: {name}")


__all__ = [
    "AccelerateLauncher",
    "DEVICE_MAP_CONFIG_KEYS",
    "InProcessLauncher",
    "Launcher",
    "TorchrunLauncher",
    "resolve_config_device_map",
    "resolve_device_map",
    "resolve_devices",
    "resolve_launcher",
]
