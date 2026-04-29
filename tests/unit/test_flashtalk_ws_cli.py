from __future__ import annotations

from argparse import Namespace

from omnirt.cli.main import build_flashtalk_ws_argv, build_parser


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
        "/srv/SoulX-FlashTalk",
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
    assert args.repo_path == "/srv/SoulX-FlashTalk"
    assert args.ckpt_dir == "models/SoulX-FlashTalk-14B"
    assert args.wav2vec_dir == "models/chinese-wav2vec2-base"
    assert args.t5_quant == "int8"
    assert args.wan_quant == "fp8"


def test_build_flashtalk_ws_argv_maps_omnirt_args_to_upstream_server() -> None:
    args = Namespace(
        host="0.0.0.0",
        port=8765,
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
        "flashtalk_server.py",
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
