# Text to Video

Generate an MP4 from a prompt. OmniRT wraps video tasks in the same request contract as image tasks; the artifact is always exported via `imageio-ffmpeg`.

## Minimal example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import text2video

    result = generate(text2video(
        model="wan2.2-t2v-14b",
        prompt="aerial view over Shanghai skyline at dusk, cinematic",
        num_frames=81,
        fps=16,
        preset="balanced",
    ))
    print(result.artifacts[0].path)   # path to the exported MP4
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task text2video \
      --model wan2.2-t2v-14b \
      --prompt "aerial view over Shanghai skyline at dusk, cinematic" \
      --num-frames 81 --fps 16 --preset balanced
    ```

=== "HTTP"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "text2video",
        "model": "wan2.2-t2v-14b",
        "inputs": {"prompt": "aerial view over Shanghai skyline at dusk"},
        "config": {"num_frames": 81, "fps": 16, "preset": "balanced"}
      }'
    ```

## Key parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `prompt` | `str` | **required** | text prompt |
| `num_frames` | `int?` | model default | typically `81` for Wan2.2; experimental video models have their own defaults |
| `fps` | `int?` | model default | output frames per second |
| `negative_prompt` | `str?` | `None` | negative prompt (if supported) |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | see [Presets](../features/presets.md) |
| `num_inference_steps` | `int?` | preset | explicit denoise override |
| `guidance_scale` | `float?` | preset | CFG |
| `width` / `height` | `int?` | model default | video models impose strict aspect / divisor constraints |
| `seed` | `int?` | random | reproducibility |

## Supported models

- **Adjacent main path**: `wan2.2-t2v-14b` (24 GB+)
- **Asset motion**: `animate-diff-sdxl`
- **Compatibility / experimental**: `cogvideox-*`, `hunyuan-video*`, and similar families remain in the registry, but are not default release baselines

Full list: `omnirt models --task text2video`.

## Common recipes

=== "Wan2.2 baseline"

    ```bash
    omnirt generate --task text2video --model wan2.2-t2v-14b \
      --prompt "..." --num-frames 81 --fps 16 --preset balanced
    ```

=== "Asset motion"

    ```bash
    omnirt generate --task text2video --model animate-diff-sdxl \
      --prompt "..." --num-frames 16 --fps 8 --preset balanced
    ```

## Troubleshooting

!!! warning

    - **MP4 encode fails** — make sure `imageio-ffmpeg` is installed via runtime extras (`pip install '.[runtime]'`)
    - **OOM** — video is the most memory-intensive task; drop `num_frames` first, then `width/height`, then `preset=low-vram`
    - **Temporal flicker / inconsistency** — raise `num_inference_steps` or switch to `preset=quality`; some models are scheduler-sensitive (see [Architecture](../../developer_guide/architecture.md))
    - **Ascend falls back to eager on video models** — expected; recorded in `RunReport.backend_timeline`. See [Ascend Backend](../deployment/ascend.md).
