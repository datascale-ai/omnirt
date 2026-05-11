# Text to Image

Generate a PNG from a text prompt. The cheapest and most mature task to run end-to-end on OmniRT.

## Minimal example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import text2image

    result = generate(text2image(
        model="sdxl-base-1.0",
        prompt="a lighthouse in fog, cinematic, 35mm film",
        preset="fast",
    ))
    print(result.artifacts[0].path)
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task text2image \
      --model sdxl-base-1.0 \
      --prompt "a lighthouse in fog, cinematic, 35mm film" \
      --preset fast \
      --backend auto
    ```

=== "YAML + CLI"

    ```yaml
    # request.yaml
    task: text2image
    model: flux2.dev
    backend: auto
    inputs:
      prompt: "a cinematic sci-fi city at sunrise"
    config:
      preset: balanced
      width: 1024
      height: 1024
    ```

    ```bash
    omnirt generate --config request.yaml --json
    ```

=== "HTTP"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "text2image",
        "model": "sdxl-base-1.0",
        "inputs": {"prompt": "a lighthouse in fog"},
        "config": {"preset": "fast"}
      }'
    ```

## Key parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `prompt` | `str` | **required** | text prompt |
| `negative_prompt` | `str?` | `None` | negative prompt; honored by SD / SDXL / SD3 |
| `width` / `height` | `int?` | model default | output size; models enforce 8/16/32-multiple constraints |
| `preset` | `fast` / `balanced` / `quality` / `low-vram` | `balanced` | bundled steps / precision / guidance; see [Presets](../features/presets.md) |
| `num_inference_steps` | `int?` | preset | explicit denoise step override |
| `guidance_scale` | `float?` | preset | classifier-free guidance |
| `num_images_per_prompt` | `int?` | `1` | batch images per prompt |
| `seed` | `int?` | random | fix randomness for reproducibility |
| `scheduler` | `str?` | model default | see [Architecture → Scheduler layer](../../developer_guide/architecture.md) |
| `dtype` | `fp16` / `bf16` / `fp32` | `fp16` | compute dtype; Ascend defaults to `bf16` |
| `adapters` | `list[AdapterRef]?` | `[]` | LoRA / ControlNet adapters |

## Supported models

Typical quality / speed tradeoffs:

- **High quality**: `flux2.dev` (≥ 24 GB VRAM), `sdxl-base-1.0` + `sdxl-refiner-1.0`
- **Balanced**: `sdxl-base-1.0`, `qwen-image`
- **Compatibility / experimental**: `sd15`, `sd21`, and `sd3-*` stay in the registry, but are not the recommended starting point for the digital-human main path

Full list: `omnirt models` or [Supported Models](../models/supported_models.md).

## Common recipes

=== "Speed-first"

    ```bash
    omnirt generate --task text2image --model sdxl-base-1.0 \
      --prompt "..." --preset fast --backend cuda
    ```

=== "Quality-first"

    ```bash
    omnirt generate --task text2image --model flux2.dev \
      --prompt "..." --preset quality --width 1024 --height 1024 \
      --backend cuda --dtype bf16
    ```

=== "Low-VRAM"

    ```bash
    omnirt generate --task text2image --model sdxl-base-1.0 \
      --prompt "..." --preset low-vram --dtype fp16
    ```

## Troubleshooting

!!! warning "Common issues"

    - **`ValidationError: width must be multiple of 8`** — most SD-family models require multiples of 8; Flux2 is stricter (16 / 32)
    - **`CUDA out of memory`** — switch to `--preset low-vram` or reduce `width/height`; or set `OMNIRT_DISABLE_COMPILE=1` to skip `torch.compile`
    - **`adapter not supported for this model`** — check `omnirt models <model_id>`'s `adapters` field; LoRA / ControlNet compatibility is declared in `ModelCapabilities`

Running `omnirt validate` catches the first two without touching a GPU.
