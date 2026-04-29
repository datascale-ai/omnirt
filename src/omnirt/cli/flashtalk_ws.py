"""Lightweight FlashTalk-compatible WebSocket launcher.

This module intentionally avoids importing the top-level :mod:`omnirt` package so
it can run inside the model vendor environment, where only FlashTalk runtime
dependencies are installed.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ENV_PREFIX = "OMNIRT_FLASHTALK_"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def _setting(name: str, args: argparse.Namespace) -> str | None:
    value = getattr(args, name, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    env_name = ENV_PREFIX + name.upper()
    env_value = os.environ.get(env_name)
    if env_value and env_value.strip():
        return env_value.strip()
    merged = {}
    merged.update(_read_yaml(_project_root() / "configs" / "flashtalk.yaml"))
    merged.update(_read_yaml(Path.home() / ".omnirt" / "flashtalk.yaml"))
    yaml_value = merged.get(name)
    if isinstance(yaml_value, str) and yaml_value.strip():
        return yaml_value.strip()
    return None


def _required_setting(name: str, args: argparse.Namespace) -> str:
    value = _setting(name, args)
    if value is None:
        raise SystemExit(
            f"error: FlashTalk setting {name!r} is required. Pass --{name.replace(chr(95), chr(45))} "
            f"or set {ENV_PREFIX + name.upper()}."
        )
    return value


def _resolve_repo_path(args: argparse.Namespace) -> Path:
    return Path(_required_setting("repo_path", args)).expanduser().resolve()


def _resolve_repo_relative(repo_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_path / path).resolve()


def build_flashtalk_ws_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "flashtalk_server.py",
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--ckpt_dir",
        str(args.ckpt_dir),
        "--wav2vec_dir",
        str(args.wav2vec_dir),
    ]
    if getattr(args, "cpu_offload", False):
        argv.append("--cpu_offload")
    for attr, flag in (
        ("t5_quant", "--t5_quant"),
        ("t5_quant_dir", "--t5_quant_dir"),
        ("wan_quant", "--wan_quant"),
        ("wan_quant_include", "--wan_quant_include"),
        ("wan_quant_exclude", "--wan_quant_exclude"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            argv.extend([flag, str(value)])
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnirt-flashtalk-ws")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo-path")
    parser.add_argument("--ckpt-dir")
    parser.add_argument("--wav2vec-dir")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--t5-quant", choices=["int8", "fp8"])
    parser.add_argument("--t5-quant-dir")
    parser.add_argument("--wan-quant", choices=["int8", "fp8"])
    parser.add_argument("--wan-quant-include")
    parser.add_argument("--wan-quant-exclude")
    parser.add_argument(
        "--upstream-ws-url",
        help="Proxy FlashTalk-compatible WS traffic to an already-running backend. Useful when NPUs are occupied.",
    )
    return parser


async def _proxy_one_connection(client_ws, upstream_ws_url: str) -> None:
    from websockets.asyncio.client import connect

    async with connect(upstream_ws_url, max_size=50 * 1024 * 1024) as upstream_ws:
        async def client_to_upstream() -> None:
            async for message in client_ws:
                await upstream_ws.send(message)

        async def upstream_to_client() -> None:
            async for message in upstream_ws:
                await client_ws.send(message)

        left = asyncio.create_task(client_to_upstream())
        right = asyncio.create_task(upstream_to_client())
        done, pending = await asyncio.wait({left, right}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()


async def _run_proxy_server(args: argparse.Namespace) -> None:
    from websockets.asyncio.server import serve

    async def handler(websocket) -> None:
        await _proxy_one_connection(websocket, str(args.upstream_ws_url))

    async with serve(handler, args.host, args.port, max_size=50 * 1024 * 1024):
        await asyncio.Future()


def run_proxy_server(args: argparse.Namespace) -> int:
    asyncio.run(_run_proxy_server(args))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.upstream_ws_url:
        return run_proxy_server(args)

    repo_path = _resolve_repo_path(args)
    if not (repo_path / "flashtalk_server.py").exists():
        raise SystemExit(f"error: FlashTalk server not found under {repo_path}")

    args.ckpt_dir = str(_resolve_repo_relative(repo_path, _required_setting("ckpt_dir", args)))
    args.wav2vec_dir = str(_resolve_repo_relative(repo_path, _required_setting("wav2vec_dir", args)))
    if args.t5_quant is None:
        args.t5_quant = _setting("t5_quant", args)
    if args.t5_quant_dir is None:
        t5_quant_dir = _setting("t5_quant_dir", args)
        if t5_quant_dir is not None:
            args.t5_quant_dir = str(_resolve_repo_relative(repo_path, t5_quant_dir))
        elif args.t5_quant is not None:
            args.t5_quant_dir = args.ckpt_dir
    if args.wan_quant is None:
        args.wan_quant = _setting("wan_quant", args)
    if args.wan_quant_include is None:
        args.wan_quant_include = _setting("wan_quant_include", args)
    if args.wan_quant_exclude is None:
        args.wan_quant_exclude = _setting("wan_quant_exclude", args)

    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(repo_path))
    sys.argv = build_flashtalk_ws_argv(args)
    try:
        os.chdir(repo_path)
        server_module = importlib.import_module("flashtalk_server")
        return int(server_module.main() or 0)
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
