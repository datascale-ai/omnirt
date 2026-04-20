from omnirt.core.registry import ModelCapabilities, clear_registry, register_model
from omnirt.core.types import GenerateRequest
from omnirt.core.validation import validate_request


def test_validate_request_applies_preset_and_preserves_explicit_override() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("num_inference_steps", "guidance_scale", "dtype"),
            default_config={"num_inference_steps": 30, "guidance_scale": 7.5},
        ),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"preset": "fast", "guidance_scale": 9.0},
        )
    )

    assert validation.ok is True
    assert validation.resolved_config["num_inference_steps"] == 20
    assert validation.resolved_config["guidance_scale"] == 9.0
    assert any(issue.level == "warning" and "Applied preset" in issue.message for issue in validation.issues)

    clear_registry()


def test_validate_request_reports_unknown_preset() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"preset": "ultra"},
        )
    )

    assert validation.ok is False
    assert "Unknown preset" in validation.format_errors()

    clear_registry()


def test_validate_request_rejects_missing_image_path() -> None:
    clear_registry()

    @register_model(
        id="dummy-video",
        task="image2video",
        capabilities=ModelCapabilities(required_inputs=("image",)),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="image2video",
            model="dummy-video",
            backend="cpu-stub",
            inputs={"image": "/tmp/does-not-exist.png"},
            config={},
        )
    )

    assert validation.ok is False
    assert "does not exist locally" in validation.format_errors()

    clear_registry()


def test_validate_request_rejects_missing_audio_path() -> None:
    clear_registry()

    @register_model(
        id="dummy-audio-video",
        task="audio2video",
        capabilities=ModelCapabilities(required_inputs=("image", "audio")),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="audio2video",
            model="dummy-audio-video",
            backend="cpu-stub",
            inputs={"image": "/tmp/does-not-exist.png", "audio": "/tmp/does-not-exist.wav"},
            config={},
        )
    )

    assert validation.ok is False
    assert "Input audio does not exist locally" in validation.format_errors()

    clear_registry()


def test_validate_request_rejects_inpaint_strength_out_of_range(tmp_path) -> None:
    clear_registry()
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    image_path.write_bytes(b"fake")
    mask_path.write_bytes(b"fake")

    @register_model(
        id="dummy-image",
        task="inpaint",
        capabilities=ModelCapabilities(
            required_inputs=("image", "mask", "prompt"),
            supported_config=("strength",),
        ),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="inpaint",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"image": str(image_path), "mask": str(mask_path), "prompt": "repair"},
            config={"strength": 1.5},
        )
    )

    assert validation.ok is False
    assert "expected a float between 0 and 1" in validation.format_errors()

    clear_registry()


def test_validate_request_reports_supported_tasks_for_existing_model() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="inpaint",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"image": "x.png", "mask": "m.png", "prompt": "repair"},
            config={},
        )
    )

    assert validation.ok is False
    assert "supports tasks [text2image]" in validation.format_errors()

    clear_registry()


def test_validate_request_resolves_repo_relative_flashtalk_paths(tmp_path) -> None:
    clear_registry()
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    image_path = tmp_path / "face.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")

    @register_model(
        id="dummy-flashtalk",
        task="audio2video",
        capabilities=ModelCapabilities(
            required_inputs=("image", "audio"),
            supported_config=("repo_path", "ckpt_dir", "wav2vec_dir"),
            default_config={"ckpt_dir": "models/SoulX-FlashTalk-14B", "wav2vec_dir": "models/chinese-wav2vec2-base"},
        ),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="audio2video",
            model="dummy-flashtalk",
            backend="cpu-stub",
            inputs={"image": str(image_path), "audio": str(audio_path)},
            config={"repo_path": str(repo_path)},
        )
    )

    assert validation.ok is True

    clear_registry()


def test_validate_request_rejects_multiple_offload_flags() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("enable_model_cpu_offload", "enable_sequential_cpu_offload"),
        ),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"enable_model_cpu_offload": True, "enable_sequential_cpu_offload": True},
        )
    )

    assert validation.ok is False
    assert "mutually exclusive" in validation.format_errors()

    clear_registry()


def test_validate_request_rejects_invalid_group_offload_type() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("enable_group_offload", "group_offload_type"),
        ),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"enable_group_offload": True, "group_offload_type": "weird"},
        )
    )

    assert validation.ok is False
    assert "group_offload_type must be either" in validation.format_errors()

    clear_registry()
