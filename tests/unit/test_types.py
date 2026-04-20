from pathlib import Path

from omnirt import requests
from omnirt.core.types import GenerateRequest, GenerateResult, ImageToVideoRequest, RunReport, TextToImageRequest, TextToVideoRequest


def test_generate_request_round_trip() -> None:
    payload = {
        "task": "text2image",
        "model": "sdxl-base-1.0",
        "backend": "cuda",
        "inputs": {"prompt": "hello"},
        "config": {"seed": 7},
        "adapters": [{"kind": "lora", "path": "/tmp/demo.safetensors", "scale": 0.75}],
    }

    request = GenerateRequest.from_dict(payload)

    assert request.to_dict() == payload


def test_generate_request_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "request.yaml"
    config_path.write_text(
        "task: text2image\nmodel: sdxl-base-1.0\ninputs:\n  prompt: hello\nconfig:\n  seed: 1\n",
        encoding="utf-8",
    )

    request = GenerateRequest.from_file(config_path)

    assert request.task == "text2image"
    assert request.inputs["prompt"] == "hello"


def test_generate_result_round_trip() -> None:
    report = RunReport(run_id="1", task="text2image", model="sdxl-base-1.0", backend="cuda")
    result = GenerateResult(outputs=[], metadata=report)

    assert GenerateResult.from_dict(result.to_dict()) == result


def test_generate_request_supports_text2video() -> None:
    payload = {
        "task": "text2video",
        "model": "wan2.2-t2v-14b",
        "backend": "cuda",
        "inputs": {"prompt": "storm over a futuristic harbor", "num_frames": 81},
        "config": {"seed": 3},
    }

    request = GenerateRequest.from_dict(payload)

    assert request.task == "text2video"
    assert request.inputs["num_frames"] == 81


def test_typed_requests_capture_task_specific_inputs() -> None:
    image_request = TextToImageRequest(model="sd15", prompt="hello", config={"seed": 1})
    video_request = TextToVideoRequest(model="wan2.2-t2v-14b", prompt="hello", num_frames=81, fps=16)
    image_video_request = ImageToVideoRequest(model="svd", image="frame.png", prompt="animate")

    assert image_request.task == "text2image"
    assert image_request.inputs["prompt"] == "hello"
    assert video_request.inputs["fps"] == 16
    assert image_video_request.inputs["image"] == "frame.png"


def test_request_helpers_are_ergonomic() -> None:
    request = requests.text2image(model="flux2.dev", prompt="hello", width=1024, guidance_scale=2.5)

    assert request.task == "text2image"
    assert request.inputs["prompt"] == "hello"
    assert request.config["width"] == 1024
    assert request.config["guidance_scale"] == 2.5
