import json

from omnirt.core.registry import ModelCapabilities, ModelSpec
from omnirt.cli.main import (
    build_parser,
    flashtalk_worker_config_from_args,
    main,
    parse_remote_worker_specs,
    render_model_summary,
    request_from_args,
)


def test_build_parser_accepts_generate_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate", "--config", "request.yaml"])

    assert args.command == "generate"
    assert args.config == "request.yaml"


def test_build_parser_accepts_serve_avatar_ws_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve-avatar-ws", "--host", "0.0.0.0", "--port", "8765", "--compat", "flashtalk"])

    assert args.command == "serve-avatar-ws"
    assert args.host == "0.0.0.0"
    assert args.port == 8765
    assert args.compat == "flashtalk"


def test_request_from_args_builds_text2image_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cuda",
            "--num-inference-steps",
            "20",
            "--seed",
            "9",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "text2image"
    assert request.model == "sdxl-base-1.0"
    assert request.inputs["prompt"] == "hello"
    assert request.config["num_inference_steps"] == 20
    assert request.config["seed"] == 9


def test_main_emits_json(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "request.yaml"
    config_path.write_text(
        "task: text2image\nmodel: sdxl-base-1.0\nbackend: cuda\ninputs:\n  prompt: hello\nconfig: {}\n",
        encoding="utf-8",
    )

    def fake_generate(request, backend=None):
        class Result:
            def to_dict(self):
                return {"outputs": [], "metadata": {"run_id": "1", "task": request.task, "model": request.model}}

        return Result()

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(["generate", "--config", str(config_path), "--json"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["metadata"]["model"] == "sdxl-base-1.0"


def test_main_supports_direct_cli_arguments(monkeypatch, capsys) -> None:
    def fake_generate(request, backend=None):
        class Result:
            def to_dict(self):
                return {"outputs": [], "metadata": {"run_id": "1", "task": request.task, "model": request.model}}

        return Result()

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cuda",
            "--json",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["metadata"]["task"] == "text2image"


def test_request_from_args_builds_image2video_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "image2video",
            "--model",
            "svd-xt",
            "--image",
            "frame.png",
            "--num-frames",
            "12",
            "--fps",
            "8",
            "--frame-bucket",
            "96",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "image2video"
    assert request.inputs["image"] == "frame.png"
    assert request.inputs["num_frames"] == 12
    assert request.config["frame_bucket"] == 96


def test_request_from_args_builds_image2image_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "image2image",
            "--model",
            "sd15",
            "--image",
            "input.png",
            "--prompt",
            "watercolor",
            "--strength",
            "0.6",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "image2image"
    assert request.inputs["image"] == "input.png"
    assert request.inputs["prompt"] == "watercolor"
    assert request.config["strength"] == 0.6


def test_request_from_args_builds_inpaint_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "inpaint",
            "--model",
            "sdxl-base-1.0",
            "--image",
            "input.png",
            "--mask",
            "mask.png",
            "--prompt",
            "repair the wall",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "inpaint"
    assert request.inputs["image"] == "input.png"
    assert request.inputs["mask"] == "mask.png"
    assert request.inputs["prompt"] == "repair the wall"


def test_request_from_args_builds_text2video_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "text2video",
            "--model",
            "wan2.2-t2v-14b",
            "--prompt",
            "a neon train crossing snowy mountains",
            "--num-frames",
            "81",
            "--fps",
            "16",
            "--guidance-scale",
            "5.0",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "text2video"
    assert request.inputs["prompt"] == "a neon train crossing snowy mountains"
    assert request.inputs["num_frames"] == 81
    assert request.inputs["fps"] == 16
    assert request.config["guidance_scale"] == 5.0


def test_request_from_args_builds_audio2video_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "audio2video",
            "--model",
            "soulx-flashtalk-14b",
            "--image",
            "speaker.png",
            "--audio",
            "voice.wav",
            "--prompt",
            "talking head",
            "--repo-path",
            "/path/to/SoulX-FlashTalk",
            "--resident-target",
            "127.0.0.1:50071",
            "--resident-autostart",
            "--launcher",
            "python",
            "--audio-encode-mode",
            "once",
            "--cpu-offload",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "audio2video"
    assert request.inputs["image"] == "speaker.png"
    assert request.inputs["audio"] == "voice.wav"
    assert request.config["repo_path"] == "/path/to/SoulX-FlashTalk"
    assert request.config["resident_target"] == "127.0.0.1:50071"
    assert request.config["resident_autostart"] is True
    assert request.config["launcher"] == "python"
    assert request.config["audio_encode_mode"] == "once"
    assert request.config["cpu_offload"] is True


def test_request_from_args_builds_liveact_audio2video_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "audio2video",
            "--model",
            "soulx-liveact-14b",
            "--image",
            "speaker.png",
            "--audio",
            "voice.wav",
            "--repo-path",
            "/path/to/SoulX-LiveAct",
            "--size",
            "416*720",
            "--fps",
            "20",
            "--sample-steps",
            "1",
            "--condition-cache-dir",
            "/tmp/liveact_condition_cache_lightvae",
            "--text-cache-visible-devices",
            "2",
            "--text-cache-device",
            "npu",
            "--force-text-cache",
            "--vae-path",
            "models/vae/lightvaew2_1.pth",
            "--use-lightvae",
            "--use-cache-vae",
            "--rank0-t5-only",
            "--steam-audio",
            "--stage-profile",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "audio2video"
    assert request.model == "soulx-liveact-14b"
    assert request.inputs["image"] == "speaker.png"
    assert request.inputs["audio"] == "voice.wav"
    assert request.config["repo_path"] == "/path/to/SoulX-LiveAct"
    assert request.config["size"] == "416*720"
    assert request.config["fps"] == 20
    assert request.config["sample_steps"] == 1
    assert request.config["condition_cache_dir"] == "/tmp/liveact_condition_cache_lightvae"
    assert request.config["text_cache_visible_devices"] == "2"
    assert request.config["text_cache_device"] == "npu"
    assert request.config["force_text_cache"] is True
    assert request.config["vae_path"] == "models/vae/lightvaew2_1.pth"
    assert request.config["use_lightvae"] is True
    assert request.config["use_cache_vae"] is True
    assert request.config["rank0_t5_only"] is True
    assert "t5_cpu" not in request.config
    assert request.config["steam_audio"] is True
    assert request.config["stage_profile"] is True
    assert "resident_autostart" not in request.config


def test_request_from_args_accepts_flashhead_runtime_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "audio2video",
            "--model",
            "soulx-flashhead-1.3b",
            "--image",
            "speaker.png",
            "--audio",
            "voice.wav",
            "--model-type",
            "pro",
            "--sample-steps",
            "2",
            "--no-vae-2d-split",
            "--latent-carry",
            "--npu-fusion-attention",
            "--profile",
        ]
    )

    request = request_from_args(args, parser)

    assert request.config["model_type"] == "pro"
    assert request.config["sample_steps"] == 2
    assert request.config["vae_2d_split"] is False
    assert request.config["latent_carry"] is True
    assert request.config["npu_fusion_attention"] is True
    assert request.config["profile"] is True
    assert "resident_autostart" not in request.config


def test_request_from_args_accepts_presets_and_scheduler() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sd15",
            "--prompt",
            "hello",
            "--preset",
            "fast",
            "--scheduler",
            "ddim",
        ]
    )

    request = request_from_args(args, parser)

    assert request.config["preset"] == "fast"
    assert request.config["scheduler"] == "ddim"


def test_request_from_args_accepts_quantization_and_tea_cache() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sd15",
            "--prompt",
            "hello",
            "--cache",
            "tea_cache",
            "--quantization",
            "int8",
            "--quantization-backend",
            "torchao",
            "--enable-layerwise-casting",
            "--layerwise-casting-storage-dtype",
            "fp8_e4m3fn",
            "--layerwise-casting-compute-dtype",
            "bf16",
            "--enable-tea-cache",
            "--tea-cache-ratio",
            "0.25",
            "--tea-cache-interval",
            "2",
        ]
    )

    request = request_from_args(args, parser)

    assert request.config["cache"] == "tea_cache"
    assert request.config["quantization"] == "int8"
    assert request.config["quantization_backend"] == "torchao"
    assert request.config["enable_layerwise_casting"] is True
    assert request.config["layerwise_casting_storage_dtype"] == "fp8_e4m3fn"
    assert request.config["layerwise_casting_compute_dtype"] == "bf16"
    assert request.config["enable_tea_cache"] is True
    assert request.config["tea_cache_ratio"] == 0.25
    assert request.config["tea_cache_interval"] == 2


def test_build_parser_accepts_serve_redis_and_otlp_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--redis-url",
            "redis://cache:6379/0",
            "--otlp-endpoint",
            "http://collector:4318/v1/traces",
        ]
    )

    assert args.redis_url == "redis://cache:6379/0"
    assert args.otlp_endpoint == "http://collector:4318/v1/traces"


def test_build_parser_accepts_worker_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["worker", "--host", "0.0.0.0", "--port", "50061", "--worker-id", "worker-a"])

    assert args.command == "worker"
    assert args.port == 50061
    assert args.worker_id == "worker-a"


def test_build_parser_accepts_resident_flashtalk_worker_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "resident-flashtalk-worker",
            "--host",
            "0.0.0.0",
            "--port",
            "50071",
            "--worker-id",
            "ft-a",
            "--repo-path",
            "/path/to/SoulX-FlashTalk",
            "--launcher",
            "python",
        ]
    )

    assert args.command == "resident-flashtalk-worker"
    assert args.port == 50071
    assert args.worker_id == "ft-a"
    assert args.repo_path == "/path/to/SoulX-FlashTalk"
    assert args.launcher == "python"


def test_parse_remote_worker_specs_supports_models_and_tags() -> None:
    parsed = parse_remote_worker_specs(["worker-a=127.0.0.1:50061@sdxl-base-1.0,flux-dev#gpu,cn-shanghai"])

    assert parsed == [
        {
            "worker_id": "worker-a",
            "address": "127.0.0.1:50061",
            "models": ("sdxl-base-1.0", "flux-dev"),
            "tags": ("gpu", "cn-shanghai"),
        }
    ]


def test_request_from_args_accepts_device_placement_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--device-map",
            "balanced",
            "--devices",
            "cuda:0,cuda:1",
        ]
    )

    request = request_from_args(args, parser)

    assert request.config["device_map"] == "balanced"
    assert request.config["devices"] == "cuda:0,cuda:1"


def test_request_from_args_accepts_accelerate_launcher() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "audio2video",
            "--model",
            "soulx-flashtalk-14b",
            "--image",
            "speaker.png",
            "--audio",
            "voice.wav",
            "--launcher",
            "accelerate",
        ]
    )

    request = request_from_args(args, parser)

    assert request.config["launcher"] == "accelerate"


def test_request_from_args_accepts_accelerate_config() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "audio2video",
            "--model",
            "soulx-flashtalk-14b",
            "--image",
            "speaker.png",
            "--audio",
            "voice.wav",
            "--launcher",
            "accelerate",
            "--num-processes",
            "4",
            "--accelerate-executable",
            "/tmp/accelerate",
        ]
    )

    request = request_from_args(args, parser)

    assert request.config["num_processes"] == 4
    assert request.config["accelerate_executable"] == "/tmp/accelerate"


def test_flashtalk_worker_config_from_args_collects_runtime_fields() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "resident-flashtalk-worker",
            "--repo-path",
            "/path/to/SoulX-FlashTalk",
            "--ckpt-dir",
            "models/SoulX-FlashTalk-14B",
            "--wav2vec-dir",
            "models/chinese-wav2vec2-base",
            "--launcher",
            "accelerate",
            "--num-processes",
            "4",
            "--accelerate-executable",
            "/tmp/accelerate",
            "--audio-encode-mode",
            "once",
            "--cpu-offload",
            "--visible-devices",
            "0,1,2,3",
            "--output-dir",
            "/tmp/out",
            "--seed",
            "42",
        ]
    )

    config = flashtalk_worker_config_from_args(args)

    assert config == {
        "repo_path": "/path/to/SoulX-FlashTalk",
        "ckpt_dir": "models/SoulX-FlashTalk-14B",
        "wav2vec_dir": "models/chinese-wav2vec2-base",
        "launcher": "accelerate",
        "num_processes": 4,
        "accelerate_executable": "/tmp/accelerate",
        "audio_encode_mode": "once",
        "cpu_offload": True,
        "visible_devices": "0,1,2,3",
        "output_dir": "/tmp/out",
        "seed": 42,
    }


def test_main_prints_clean_omnirt_errors(monkeypatch, capsys) -> None:
    from omnirt.core.types import BackendUnavailableError

    def fake_generate(request, backend=None):
        raise BackendUnavailableError("cuda missing")

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cuda",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "error: cuda missing" in captured.err


def test_main_validate_emits_json(monkeypatch, capsys) -> None:
    class Validation:
        ok = True

        def to_dict(self):
            return {
                "ok": True,
                "request": {"task": "text2image", "model": "sdxl-base-1.0"},
                "resolved_backend": "cpu-stub",
                "resolved_inputs": {"prompt": "hello"},
                "resolved_config": {"num_inference_steps": 20},
                "model": "sdxl-base-1.0",
                "issues": [],
            }

    monkeypatch.setattr("omnirt.cli.main.validate", lambda request, backend=None: Validation())

    exit_code = main(
        [
            "validate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cpu-stub",
            "--json",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["resolved_backend"] == "cpu-stub"


def test_main_bench_emits_json(monkeypatch, capsys) -> None:
    class Report:
        def to_dict(self):
            return {
                "scenario": "text2image_sdxl_concurrent4",
                "total_requests": 10,
                "concurrency": 4,
                "warmup": 1,
                "total_duration_s": 1.5,
                "throughput_rps": 6.7,
                "latency_ms": {"p50": 120.0, "p95": 150.0, "p99": 160.0},
                "ttft_ms": {"p50": 30.0, "p95": 40.0, "p99": 45.0},
                "peak_vram": 12.0,
                "cache_hit_ratio": 0.5,
                "batch_size_mean": 1.5,
                "batched_request_ratio": 0.5,
                "execution_mode_breakdown": {"modular": 10},
            }

    monkeypatch.setattr("omnirt.cli.main.run_bench", lambda scenario: Report())

    exit_code = main(["bench", "--scenario", "text2image_sdxl_concurrent4", "--json"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["scenario"] == "text2image_sdxl_concurrent4"


def test_main_dry_run_uses_validation_only(monkeypatch, capsys) -> None:
    calls = {"generate": 0}

    class Validation:
        ok = True
        request = type("Request", (), {"task": "text2image", "model": "sd15"})()
        resolved_backend = "cpu-stub"
        resolved_inputs = {"prompt": "hello"}
        resolved_config = {"num_inference_steps": 20}
        issues = []

        def to_dict(self):
            return {
                "ok": True,
                "request": {"task": "text2image", "model": "sd15"},
                "resolved_backend": "cpu-stub",
                "resolved_inputs": self.resolved_inputs,
                "resolved_config": self.resolved_config,
                "issues": [],
            }

    monkeypatch.setattr("omnirt.cli.main.validate", lambda request, backend=None: Validation())

    def fake_generate(request, backend=None):
        calls["generate"] += 1
        raise AssertionError("generate should not be called during dry-run")

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sd15",
            "--prompt",
            "hello",
            "--backend",
            "cpu-stub",
            "--dry-run",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert calls["generate"] == 0
    assert "resolved_backend=cpu-stub" in stdout


def test_main_models_emits_json(monkeypatch, capsys) -> None:
    spec = ModelSpec(
        id="sd15",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="beta", summary="Stable Diffusion 1.5"),
    )
    monkeypatch.setattr("omnirt.cli.main.list_available_models", lambda include_aliases=False: [spec])

    exit_code = main(["models", "--json"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(stdout)
    assert payload[0]["id"] == "sd15"
    assert payload[0]["status"] == "public/beta"
    assert "fast" in payload[0]["presets"]


def test_render_model_summary_includes_statuses() -> None:
    primary = ModelSpec(
        id="sdxl-base-1.0",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="stable", summary="SDXL base"),
    )
    variants = {
        "text2image": primary,
        "image2image": ModelSpec(
            id="sdxl-base-1.0",
            task="image2image",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="stable", summary="SDXL image-to-image"),
        ),
        "inpaint": ModelSpec(
            id="sdxl-base-1.0",
            task="inpaint",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="beta", summary="SDXL inpaint"),
        ),
    }

    summary = render_model_summary(primary, variants=variants)

    assert "status=public/stable" in summary
    assert "supported_tasks=text2image (public/stable), image2image (public/stable), inpaint (preview/beta)" in summary


def test_main_models_text_output_includes_status(monkeypatch, capsys) -> None:
    spec = ModelSpec(
        id="sd15",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="beta", summary="Stable Diffusion 1.5"),
    )
    monkeypatch.setattr("omnirt.cli.main.list_available_models", lambda include_aliases=False: [spec])

    exit_code = main(["models"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "sd15\ttext2image\tpublic/beta\tStable Diffusion 1.5" in stdout


def test_main_models_markdown_format_groups_by_digital_human_chain_and_lists_aliases(monkeypatch, capsys) -> None:
    sdxl = ModelSpec(
        id="sdxl-base-1.0",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="stable", summary="SDXL base"),
    )
    flux2_canonical = ModelSpec(
        id="flux2.dev",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="beta", summary="Flux2 dev"),
    )
    flux2_alias = ModelSpec(
        id="flux2-dev",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="beta", summary="Flux2 dev", alias_of="flux2.dev"),
    )
    svd = ModelSpec(
        id="svd-xt",
        task="image2video",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="stable", summary="SVD XT"),
    )
    monkeypatch.setattr(
        "omnirt.cli.main.list_available_models",
        lambda include_aliases=False: [sdxl, flux2_canonical, flux2_alias, svd],
    )

    exit_code = main(["models", "--format", "markdown"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "# OmniRT supported models" in stdout
    assert "## Avatar asset generation" in stdout
    assert "## Video and idle assets" in stdout
    assert "| `sdxl-base-1.0` | `text2image` | stable | no | SDXL base |" in stdout
    assert "| `flux2.dev` | `text2image` | beta | no | Flux2 dev |" in stdout
    assert "## Aliases" in stdout
    assert "| `flux2-dev` | `flux2.dev` |" in stdout
    # Alias rows must not appear in the task table.
    model_sections = stdout.split("## Aliases", 1)[0]
    assert "| `flux2-dev` |" not in model_sections


def test_main_model_detail_json_includes_supported_task_statuses(monkeypatch, capsys) -> None:
    primary = ModelSpec(
        id="sdxl-base-1.0",
        task="text2image",
        pipeline_cls=object,
        default_backend="auto",
        capabilities=ModelCapabilities(maturity="stable", summary="SDXL base"),
    )
    variants = {
        "text2image": primary,
        "image2image": ModelSpec(
            id="sdxl-base-1.0",
            task="image2image",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="stable", summary="SDXL image-to-image"),
        ),
        "inpaint": ModelSpec(
            id="sdxl-base-1.0",
            task="inpaint",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="beta", summary="SDXL inpaint"),
        ),
    }
    monkeypatch.setattr("omnirt.cli.main.describe_model", lambda model_id: primary)
    monkeypatch.setattr("omnirt.cli.main.list_model_variants", lambda model_id: variants)

    exit_code = main(["models", "sdxl-base-1.0", "--json"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(stdout)
    assert payload["status"] == "public/stable"
    assert payload["supported_task_statuses"] == {
        "text2image": "public/stable",
        "image2image": "public/stable",
        "inpaint": "preview/beta",
    }
