# 文本到视频（text2video）

从一段 prompt 生成一个 MP4。OmniRT 把视频任务封装成与图像任务一致的请求契约，产物固定通过 `imageio-ffmpeg` 导出。

## 最小示例

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
    print(result.artifacts[0].path)   # 输出的 MP4 路径
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

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | `str` | **必填** | 文本 prompt |
| `num_frames` | `int?` | 模型默认 | 帧数；Wan2.2 通常 `81`，不同 experimental 视频模型会有自己的默认值 |
| `fps` | `int?` | 模型默认 | 输出帧率 |
| `negative_prompt` | `str?` | `None` | 负向 prompt（部分模型支持） |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | 见 [预设](../features/presets.md) |
| `num_inference_steps` | `int?` | preset | 显式覆盖去噪步数 |
| `guidance_scale` | `float?` | preset | CFG 引导强度 |
| `width` / `height` | `int?` | 模型默认 | 输出分辨率；视频模型对宽高有更强约束 |
| `seed` | `int?` | 随机 | 固定随机性 |

## 支持模型

- **相邻主线**：`wan2.2-t2v-14b`（24 GB+）
- **素材/动效**：`animate-diff-sdxl`
- **兼容 / experimental**：`cogvideox-*`、`hunyuan-video*` 等仍保留 registry，但不作为默认发布基线

完整清单：`omnirt models --task text2video`。

## 常见组合

=== "Wan2.2 基线"

    ```bash
    omnirt generate --task text2video --model wan2.2-t2v-14b \
      --prompt "..." --num-frames 81 --fps 16 --preset balanced
    ```

=== "素材动效"

    ```bash
    omnirt generate --task text2video --model animate-diff-sdxl \
      --prompt "..." --num-frames 16 --fps 8 --preset balanced
    ```

## 错误与排查

!!! warning

    - **MP4 编码失败** — 检查 `imageio-ffmpeg` 已随 runtime extras 安装（`pip install '.[runtime]'`）
    - **显存不足** — `text2video` 是显存消耗最密集的任务；优先降 `num_frames`，其次降 `width/height`，最后才切 `preset=low-vram`
    - **颜色一致性差 / 画面抖动** — 提高 `num_inference_steps` 或切 `preset=quality`；个别模型对 `scheduler` 敏感，参考 [架构](../../developer_guide/architecture.md) 的 scheduler 列表
    - **Ascend 上跑视频模型回退到 eager** — 正常现象，记录在 `RunReport.backend_timeline`，见 [Ascend 后端](../deployment/ascend.md)
