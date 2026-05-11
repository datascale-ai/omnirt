# 图像到图像（image2image）

在一张输入图的基础上，按 prompt 重绘一张 PNG。适合风格化、精修、局部改写。

## 最小示例

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

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `str` | **必填** | 输入图路径或 URI |
| `prompt` | `str` | **必填** | 重绘 prompt |
| `strength` | `float` ∈ `(0, 1]` | `0.8` | 改写强度；越低越保留原图结构 |
| `negative_prompt` | `str?` | `None` | 负向 prompt |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | 见 [预设](../features/presets.md) |
| `num_inference_steps` | `int?` | preset 决定 | 显式覆盖去噪步数 |
| `guidance_scale` | `float?` | preset 决定 | CFG 引导强度 |
| `seed` | `int?` | 随机 | 固定随机性 |

## 支持模型

- **推荐**：`sdxl-base-1.0`
- **精修**：`sdxl-refiner-1.0`（通常作为 `sdxl-base-1.0` 的二次推理）
- **兼容 / experimental**：`sd15`、`sd21` 仍保留，但不作为新数字人素材链路默认起点

运行 `omnirt models --task image2image` 查看完整清单。

## 常见组合

- **轻度风格化**：`strength=0.4`, `preset=balanced`
- **重度重绘**：`strength=0.85`, `preset=quality`
- **SDXL 双阶段精修**：先 `sdxl-base-1.0` 出草图，再 `sdxl-refiner-1.0` 跑一次 `image2image` with `strength=0.3`

## 错误与排查

!!! warning

    - **输入图尺寸不合法** — 先校验：`omnirt validate --task image2image --model <id> --image <path> --prompt "…"`
    - **`strength` 超出 (0, 1]** — OmniRT 在 validation 阶段就会报错
    - **颜色漂移/鬼影** — 多半是 `strength` 过高或 `num_inference_steps` 过少；尝试 `strength=0.55, preset=balanced`
