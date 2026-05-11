# Presets

`omnirt` ships four out-of-the-box presets: `fast`, `balanced`, `quality`, `low-vram`.
Each preset is a small set of `config` overrides. The request's `config` is merged in this order, with later layers winning:

1. **Base preset** — steps / dtype shared by all tasks
2. **Task preset** — per-task guidance-scale defaults (`text2image` / `text2video` / `image2video`)
3. **Model preset** — narrow model-specific overrides (today, the Flux2 family)
4. **User-provided `config`** — always wins

The merge happens in `resolve_preset(task, model, preset)` in [src/omnirt/core/presets.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/presets.py).

## Base preset

| Preset | `num_inference_steps` | `dtype` |
|---|---|---|
| `fast` | 20 | — |
| `balanced` | — | — |
| `quality` | 40 | — |
| `low-vram` | 18 | `fp16` |

`balanced` keeps each pipeline's own default step count. `low-vram` forces 18 steps and fp16 so memory-constrained devices still finish.

## Task preset — `guidance_scale`

| Preset | `text2image` | `text2video` | `image2video` |
|---|---|---|---|
| `fast` | 5.5 | 4.0 | 3.0 |
| `balanced` | 7.5 | 5.0 | 3.5 |
| `quality` | 8.0 | 6.0 | 4.0 |
| `low-vram` | — | 4.0 | 3.0 |

`image2image` / `inpaint` / `edit` reuse the `text2image` entries (see `_TASK_PRESET_ALIASES`). `low-vram` on `text2image` only contributes the base preset's steps + dtype and does not override guidance.

## Model preset — `flux2.dev` / `flux2-dev`

| Preset | `guidance_scale` | `max_sequence_length` | Other |
|---|---|---|---|
| `fast` | 2.0 | 384 | — |
| `balanced` | 2.5 | 512 | — |
| `quality` | 3.0 | 512 | — |
| `low-vram` | 2.0 | 256 | `dtype=fp16` |

Flux is sensitive to high guidance; these values override the task preset to stay in the useful range.

## Usage

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" \
  --preset fast
```

Python equivalent:

```python
from omnirt import requests, generate

req = requests.text2image(model="sdxl-base-1.0", prompt="a lighthouse in fog", preset="fast")
result = generate(req, backend="cuda")
```

## Inspecting the merged result

```bash
omnirt validate \
  --task text2image --model flux2.dev \
  --prompt "..." \
  --preset low-vram \
  --backend cpu-stub
```

The validator returns a `resolved_config` that shows exactly what the preset produced, so you can see how the merge resolved for the specific model you chose.
