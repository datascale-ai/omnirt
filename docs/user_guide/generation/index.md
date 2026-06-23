# 生成任务

本章按任务面组织，覆盖 OmniRT 支持的全部公开任务。每一页结构相同：**最小示例**（Python / CLI / HTTP 三种入口）→ **关键参数** → **支持模型** → **常见组合** → **错误与排查**。

| 任务面 | 典型入口 | 典型模型 | 页面 |
|---|---|---|---|
| `text2image` | 文字 → 单张图 | `sdxl-base-1.0`、`flux2.dev`、`qwen-image` | [文本到图像](text2image.md) |
| `text2audio` | 文字 + 参考音频 → 语音 | `vllm-omni-speech`, `cosyvoice3-triton-trtllm` | [文本到音频](text2audio.md) |
| `audio2text` | 音频 → 文本 | `sensevoice-small` | [音频到文本](audio2text.md) |
| `image2image` | 图 + prompt → 图 | `sdxl-base-1.0`、`sdxl-refiner-1.0` | [图像到图像](image2image.md) |
| `text2video` | 文字 → 视频 | `wan2.2-t2v-14b`、`animate-diff-sdxl` | [文本到视频](text2video.md) |
| `image2video` | 首帧 + prompt → 视频 | `svd-xt`、`wan2.2-i2v-14b` | [图像到视频](image2video.md) |
| `audio2video` | 音频 + portrait → 视频 | `soulx-flashtalk-14b`、`soulx-flashhead-1.3b`、`soulx-liveact-14b` | [数字人](talking_head.md) |

!!! tip "不知道从哪个任务开始？"
    如果目标是数字人产品，先读 [文本到音频](text2audio.md) 和 [数字人](talking_head.md)。如果只是熟悉请求契约，再用 [文本到图像](text2image.md) 做最低成本的本地练习。
