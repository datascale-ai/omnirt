# 文本到图像（text2image）

从一段 prompt 生成一张 PNG。OmniRT 上最成熟、最便宜跑通的任务面。

## 最小示例

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

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | `str` | **必填** | 文本 prompt |
| `negative_prompt` | `str?` | `None` | 负向 prompt；对 SD / SDXL / SD3 有效 |
| `width` / `height` | `int?` | 模型默认 | 输出尺寸；不同模型对 8/16/32 倍数有不同要求 |
| `preset` | `fast` / `balanced` / `quality` / `low-vram` | `balanced` | 整组步数 / 精度 / 引导强度；详见 [预设](../features/presets.md) |
| `num_inference_steps` | `int?` | preset 决定 | 显式覆盖去噪步数 |
| `guidance_scale` | `float?` | preset 决定 | CFG 引导强度 |
| `num_images_per_prompt` | `int?` | `1` | 同一 prompt 批量生成数量 |
| `seed` | `int?` | 随机 | 固定随机性用于复现 |
| `scheduler` | `str?` | 模型默认 | 见 [架构 → Scheduler 层](../../developer_guide/architecture.md) |
| `dtype` | `fp16` / `bf16` / `fp32` | `fp16` | 计算精度；Ascend 默认 `bf16` |
| `adapters` | `list[AdapterRef]?` | `[]` | LoRA / ControlNet 适配器清单 |

## 支持模型

按质量 / 速度的典型权衡：

- **高质量**：`flux2.dev`（需 ≥ 24 GB 显存）、`sdxl-base-1.0` + `sdxl-refiner-1.0`
- **平衡**：`sdxl-base-1.0`、`qwen-image`
- **兼容 / experimental**：`sd15`、`sd21`、`sd3-*` 仍保留 registry，但不作为数字人主线推荐起点

完整清单：`omnirt models` 或 [模型清单](../models/supported_models.md)。

## 常见组合

=== "速度优先"

    ```bash
    omnirt generate --task text2image --model sdxl-base-1.0 \
      --prompt "..." --preset fast --backend cuda
    ```

=== "质量优先"

    ```bash
    omnirt generate --task text2image --model flux2.dev \
      --prompt "..." --preset quality --width 1024 --height 1024 \
      --backend cuda --dtype bf16
    ```

=== "低显存"

    ```bash
    omnirt generate --task text2image --model sdxl-base-1.0 \
      --prompt "..." --preset low-vram --dtype fp16
    ```

## 错误与排查

!!! warning "常见问题"

    - **`ValidationError: width must be multiple of 8`** — 大多数 SD 系模型要求 `width` / `height` 是 8 的倍数；Flux2 更严格（16 / 32 倍数）
    - **`CUDA out of memory`** — 切到 `--preset low-vram` 或降低 `width/height`；或设置 `OMNIRT_DISABLE_COMPILE=1` 跳过 `torch.compile`
    - **`adapter not supported for this model`** — 检查 `omnirt models <model_id>` 的 `adapters` 字段；LoRA / ControlNet 的兼容性由 `ModelCapabilities` 声明

一次 `omnirt validate` 就能提前抓到前两类错误，无需占用 GPU。
