"""Helpers for Diffusers device_map configuration."""

from __future__ import annotations

import json
from typing import Any, Iterable

DEVICE_MAP_CONFIG_KEYS = ("device_map", "devices")


def resolve_devices(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Iterable):
        return tuple(str(part).strip() for part in value if str(part).strip())
    raise ValueError("devices must be a comma-separated string or a list of device labels.")


def resolve_device_map(value: Any) -> str | dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return {str(key): _coerce_mapping_value(item) for key, item in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"balanced", "auto", "sequential"}:
            return lowered
        if text.startswith("{"):
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("device_map JSON must decode to an object mapping component names to devices.")
            return {str(key): _coerce_mapping_value(item) for key, item in loaded.items()}
        mapping: dict[str, Any] = {}
        for entry in text.split(","):
            part = entry.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    "device_map must be 'balanced', a JSON object, or comma-separated assignments like 'unet:0,vae:1'."
                )
            key, raw_value = part.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if not key or raw_value == "":
                raise ValueError("device_map assignments must include both a component name and a device target.")
            mapping[key] = _coerce_mapping_value(raw_value)
        if mapping:
            return mapping
    raise ValueError(
        "device_map must be 'balanced', 'auto', 'sequential', a JSON object, or comma-separated assignments."
    )


def resolve_config_device_map(config: dict[str, Any] | None) -> str | dict[str, Any] | None:
    if not config:
        return None
    if "device_map" in config:
        return resolve_device_map(config.get("device_map"))
    devices = resolve_devices(config.get("devices"))
    if len(devices) > 1:
        return "balanced"
    return None


def _coerce_mapping_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        return text
    return value
