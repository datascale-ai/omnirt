# Presets

`omnirt` 提供四个开箱即用的 preset：`fast`、`balanced`、`quality`、`low-vram`。
每个 preset 是一组 `config` 覆写；请求的 `config` 会按以下顺序合并，后者覆盖前者：

1. **Base preset** — `fast / balanced / quality / low-vram` 共用的基础项（步数、dtype）
2. **Task preset** — 同一 preset 在 `text2image / text2video / image2video` 上的差异（guidance scale）
3. **Model preset** — 少数模型（目前是 Flux2 家族）的专门覆写
4. **用户显式 `config`** — 最终胜出

合并逻辑位于 [src/omnirt/core/presets.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/presets.py) 的 `resolve_preset(task, model, preset)`。

## Base preset

| Preset | `num_inference_steps` | `dtype` |
|---|---|---|
| `fast` | 20 | — |
| `balanced` | — | — |
| `quality` | 40 | — |
| `low-vram` | 18 | `fp16` |

`balanced` 只保留各 pipeline 的默认 `num_inference_steps`，不做覆写。`low-vram` 在低显存环境下把步数压到 18 并强制 fp16。

## Task preset — `guidance_scale`

| Preset | `text2image` | `text2video` | `image2video` |
|---|---|---|---|
| `fast` | 5.5 | 4.0 | 3.0 |
| `balanced` | 7.5 | 5.0 | 3.5 |
| `quality` | 8.0 | 6.0 | 4.0 |
| `low-vram` | — | 4.0 | 3.0 |

`image2image` / `inpaint` / `edit` 当作 `text2image` 处理（见 `_TASK_PRESET_ALIASES`）。`low-vram` 在 `text2image` 下不做 guidance 覆写，只沿用 base preset 的步数 + dtype。

## Model preset — `flux2.dev` / `flux2-dev`

| Preset | `guidance_scale` | `max_sequence_length` | 其它 |
|---|---|---|---|
| `fast` | 2.0 | 384 | — |
| `balanced` | 2.5 | 512 | — |
| `quality` | 3.0 | 512 | — |
| `low-vram` | 2.0 | 256 | `dtype=fp16` |

Flux 家族对 guidance 敏感，需要更低的 CFG 才能避免过饱和；这组值覆盖了对应 task preset 的 `guidance_scale`。

## 使用示例

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" \
  --preset fast
```

等价 Python：

```python
from omnirt import requests, generate

req = requests.text2image(model="sdxl-base-1.0", prompt="a lighthouse in fog", preset="fast")
result = generate(req, backend="cuda")
```

## 验证 preset 效果

```bash
omnirt validate \
  --task text2image --model flux2.dev \
  --prompt "..." \
  --preset low-vram \
  --backend cpu-stub
```

校验器会返回 `resolved_config`，里面带合并后的最终值，可以直观看到 preset 改了什么。
