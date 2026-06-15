from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from omnirt.cli.main import build_flashtalk_ws_argv, build_parser, default_flashtalk_ws_server_path


def test_build_parser_accepts_serve_flashtalk_ws_protocol() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "serve",
        "--protocol",
        "flashtalk-ws",
        "--host",
        "0.0.0.0",
        "--port",
        "8765",
        "--repo-path",
        "/path/to/SoulX-FlashTalk",
        "--server-path",
        "/path/to/omnirt/model_backends/flashtalk/flashtalk_ws_server.py",
        "--ckpt-dir",
        "models/SoulX-FlashTalk-14B",
        "--wav2vec-dir",
        "models/chinese-wav2vec2-base",
        "--t5-quant",
        "int8",
        "--wan-quant",
        "fp8",
    ])

    assert args.command == "serve"
    assert args.protocol == "flashtalk-ws"
    assert args.host == "0.0.0.0"
    assert args.port == 8765
    assert args.repo_path == "/path/to/SoulX-FlashTalk"
    assert args.server_path == "/path/to/omnirt/model_backends/flashtalk/flashtalk_ws_server.py"
    assert args.ckpt_dir == "models/SoulX-FlashTalk-14B"
    assert args.wav2vec_dir == "models/chinese-wav2vec2-base"
    assert args.t5_quant == "int8"
    assert args.wan_quant == "fp8"


def test_build_flashtalk_ws_argv_maps_omnirt_args_to_upstream_server() -> None:
    args = Namespace(
        host="0.0.0.0",
        port=8765,
        server_path="/path/to/omnirt/model_backends/flashtalk/flashtalk_ws_server.py",
        ckpt_dir="/models/SoulX-FlashTalk-14B",
        wav2vec_dir="/models/chinese-wav2vec2-base",
        cpu_offload=True,
        t5_quant="int8",
        t5_quant_dir="/models/SoulX-FlashTalk-14B",
        wan_quant="fp8",
        wan_quant_include="blocks.*",
        wan_quant_exclude="head",
    )

    argv = build_flashtalk_ws_argv(args)

    assert argv == [
        "/path/to/omnirt/model_backends/flashtalk/flashtalk_ws_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        "8765",
        "--ckpt_dir",
        "/models/SoulX-FlashTalk-14B",
        "--wav2vec_dir",
        "/models/chinese-wav2vec2-base",
        "--cpu_offload",
        "--t5_quant",
        "int8",
        "--t5_quant_dir",
        "/models/SoulX-FlashTalk-14B",
        "--wan_quant",
        "fp8",
        "--wan_quant_include",
        "blocks.*",
        "--wan_quant_exclude",
        "head",
    ]


def test_default_flashtalk_ws_server_path_points_to_model_backend() -> None:
    path = default_flashtalk_ws_server_path()

    assert path == Path("model_backends/flashtalk/flashtalk_ws_server.py").resolve()


def test_lightweight_parser_accepts_upstream_proxy_url() -> None:
    from omnirt.cli.flashtalk_ws import build_parser

    args = build_parser().parse_args([
        "--host",
        "0.0.0.0",
        "--port",
        "8766",
        "--upstream-ws-url",
        "ws://127.0.0.1:8765",
    ])

    assert args.upstream_ws_url == "ws://127.0.0.1:8765"


def test_runtime_parser_accepts_install_status_and_env() -> None:
    parser = build_parser()

    install = parser.parse_args([
        "runtime",
        "install",
        "flashtalk",
        "--device",
        "cuda",
        "--home",
        "/path/to/omnirt-runtime",
        "--repo-dir",
        "/path/to/SoulX-FlashTalk",
        "--ckpt-dir",
        "/models/SoulX-FlashTalk-14B",
        "--wav2vec-dir",
        "/models/chinese-wav2vec2-base",
        "--recreate-venv",
        "--dry-run",
    ])
    status = parser.parse_args(["runtime", "status", "flashtalk", "--device", "cuda"])
    env = parser.parse_args(["runtime", "env", "flashtalk", "--device", "cuda", "--shell"])

    assert install.command == "runtime"
    assert install.runtime_command == "install"
    assert install.name == "flashtalk"
    assert install.device == "cuda"
    assert install.home == "/path/to/omnirt-runtime"
    assert install.repo_dir == "/path/to/SoulX-FlashTalk"
    assert install.ckpt_dir == "/models/SoulX-FlashTalk-14B"
    assert install.wav2vec_dir == "/models/chinese-wav2vec2-base"
    assert install.recreate_venv is True
    assert install.dry_run is True
    assert status.runtime_command == "status"
    assert env.runtime_command == "env"
    assert env.shell is True
