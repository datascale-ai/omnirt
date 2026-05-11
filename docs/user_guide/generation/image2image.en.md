# Image to Image

Repaint an input image under a new prompt and export a PNG. Ideal for stylization, touch-up, or partial rewrites.

## Minimal example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import image2image

    result = generate(image2image(
        model="sdxl-base-1.0",
        image="inputs/portrait.png",
        prompt="oil painting, impressionist, vivid colors",
        strength=0.7,
        preset="balanced",
    ))
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task image2image \
      --model sdxl-base-1.0 \
      --image inputs/portrait.png \
      --prompt "oil painting, impressionist, vivid colors" \
      --strength 0.7 --preset balanced
    ```

=== "HTTP"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "image2image",
        "model": "sdxl-base-1.0",
        "inputs": {
          "image": "inputs/portrait.png",
          "prompt": "oil painting, impressionist, vivid colors"
        },
        "config": {"strength": 0.7, "preset": "balanced"}
      }'
    ```

## Key parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | **required** | path or URI to input image |
| `prompt` | `str` | **required** | rewrite prompt |
| `strength` | `float` ∈ `(0, 1]` | `0.8` | rewrite intensity; lower keeps more of the original |
| `negative_prompt` | `str?` | `None` | negative prompt |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | see [Presets](../features/presets.md) |
| `num_inference_steps` | `int?` | preset | explicit step override |
| `guidance_scale` | `float?` | preset | CFG |
| `seed` | `int?` | random | reproducibility |

## Supported models

- **Recommended**: `sdxl-base-1.0`
- **Refiner**: `sdxl-refiner-1.0` (typically used as a second pass over `sdxl-base-1.0`)
- **Compatibility / experimental**: `sd15` and `sd21` remain available, but are not the default starting point for new digital-human asset pipelines

Run `omnirt models --task image2image` for the full list.

## Common recipes

- **Light stylization**: `strength=0.4`, `preset=balanced`
- **Heavy rewrite**: `strength=0.85`, `preset=quality`
- **SDXL two-pass polish**: draft with `sdxl-base-1.0`, then run `image2image` on `sdxl-refiner-1.0` at `strength=0.3`

## Troubleshooting

!!! warning

    - **Invalid input dimensions** — validate first: `omnirt validate --task image2image --model <id> --image <path> --prompt "…"`
    - **`strength` outside (0, 1]** — caught at validation time
    - **Color shift / ghosting** — usually `strength` too high or too few steps; try `strength=0.55, preset=balanced`
