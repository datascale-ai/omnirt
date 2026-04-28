# 生成任务

本章按任务面组织，覆盖 OmniRT 支持的全部公开任务。每一页结构相同：**最小示例**（Python / CLI / HTTP 三种入口）→ **关键参数** → **支持模型** → **常见组合** → **错误与排查**。

| 任务面 | 典型入口 | 典型模型 | 页面 |
|---|---|---|---|
| `text2image` | 文字 → 单张图 | `sd15`、`sdxl-base-1.0`、`flux2.dev`、`qwen-image` | [文本到图像](text2image.md) |
| `text2audio` | 文字 + 参考音频 → 语音 | `cosyvoice3-triton-trtllm` | [文本到音频](text2audio.md) |
| `image2image` | 图 + prompt → 图 | `sd15`、`sdxl-base-1.0` | [图像到图像](image2image.md) |
| `text2video` | 文字 → 视频 | `wan2.2-t2v-14b`、`cogvideox-2b`、`hunyuan-video` | [文本到视频](text2video.md) |
| `image2video` | 首帧 + prompt → 视频 | `svd-xt`、`wan2.2-i2v-14b` | [图像到视频](image2video.md) |
| `audio2video` | 音频 + portrait → 视频 | `soulx-flashtalk-14b` | [数字人](talking_head.md) |

!!! tip "不知道从哪个任务开始？"
    先读 [文本到图像](text2image.md) —— OmniRT 上最成熟、最便宜跑通的任务面，也是 `omnirt validate` / `omnirt generate` 的教学默认项。
