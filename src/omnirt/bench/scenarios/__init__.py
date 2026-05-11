"""Built-in benchmark scenarios."""

from __future__ import annotations

from omnirt.core.types import GenerateRequest

from ..runner import BenchScenario


_SCENARIOS = {
    "core_audio2video_flashtalk_smoke": BenchScenario(
        name="core_audio2video_flashtalk_smoke",
        request_template=GenerateRequest(
            task="audio2video",
            model="soulx-flashtalk-14b",
            backend="ascend",
            inputs={"image": "input.png", "audio": "input.wav"},
            config={"audio_encode_mode": "once", "max_chunks": 1},
        ),
        concurrency=1,
        total_requests=10,
        warmup=0,
    ),
    "adjacent_text2image_sdxl_concurrent4": BenchScenario(
        name="adjacent_text2image_sdxl_concurrent4",
        request_template=GenerateRequest(
            task="text2image",
            model="sdxl-base-1.0",
            backend="auto",
            inputs={"prompt": "a cinematic portrait of a traveler under neon rain"},
            config={"width": 1024, "height": 1024, "num_inference_steps": 30, "guidance_scale": 5.0},
        ),
        concurrency=4,
        total_requests=100,
        warmup=2,
        batch_window_ms=50,
        max_batch_size=4,
    ),
    "text2image_sdxl_concurrent4": BenchScenario(
        name="text2image_sdxl_concurrent4",
        request_template=GenerateRequest(
            task="text2image",
            model="sdxl-base-1.0",
            backend="auto",
            inputs={"prompt": "a cinematic portrait of a traveler under neon rain"},
            config={"width": 1024, "height": 1024, "num_inference_steps": 30, "guidance_scale": 5.0},
        ),
        concurrency=4,
        total_requests=100,
        warmup=2,
        batch_window_ms=50,
        max_batch_size=4,
    ),
}


def get_bench_scenario(name: str) -> BenchScenario:
    try:
        return _SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SCENARIOS))
        raise ValueError(f"Unknown bench scenario {name!r}. Available: {known}") from exc


def list_bench_scenarios() -> list[str]:
    return sorted(_SCENARIOS)
