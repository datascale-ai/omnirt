import json

from omnirt.core.registry import ModelCapabilities, ModelSpec
from omnirt.cli.main import build_parser, main, request_from_args


def test_build_parser_accepts_generate_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate", "--config", "request.yaml"])

    assert args.command == "generate"
    assert args.config == "request.yaml"


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
            "/srv/SoulX-FlashTalk",
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
    assert request.config["repo_path"] == "/srv/SoulX-FlashTalk"
    assert request.config["launcher"] == "python"
    assert request.config["audio_encode_mode"] == "once"
    assert request.config["cpu_offload"] is True


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
    assert "fast" in payload[0]["presets"]
