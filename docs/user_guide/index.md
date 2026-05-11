# 使用指南

本指南面向**用 OmniRT 构建应用**的开发者。你会在这里找到：

- **快速开始** — [环境准备与第一条请求](../getting_started/quickstart.md)
- **生成任务** — [文本到音频](generation/text2audio.md)、[数字人](generation/talking_head.md)、[文本到图像](generation/text2image.md)、[图像到图像](generation/image2image.md)、[文本到视频](generation/text2video.md)、[图像到视频](generation/image2video.md)
- **运行入口** — [CLI](serving/cli.md)、[Python API](serving/python_api.md)、[HTTP 服务](serving/http_server.md)
- **部署指南** — [CUDA](deployment/cuda.md)、[Ascend 后端](deployment/ascend.md)、[国内部署](deployment/china_mirrors.md)、[Docker](deployment/docker.md)
- **模型** — [模型清单](models/supported_models.md)、[支持状态](models/support_status.md)、[路线图](models/roadmap.md)
- **特性** — [预设](features/presets.md)、[请求校验](features/validation.md)、[服务协议](features/service_schema.md)、[派发与队列](features/dispatch_queue.md)、[遥测](features/telemetry.md)

!!! tip "还在熟悉 OmniRT？"
    先从 [快速开始](../getting_started/quickstart.md) 跑通一次 `omnirt validate` 与一次 `omnirt generate`，再回到本指南按需深入。

!!! info "需要参考级 API 文档？"
    完整的 Python API 参考正在整理中；在此之前，[Python API 使用指南](serving/python_api.md) 已经覆盖了 `generate`、`validate`、`pipeline` 与全部 typed request helpers。
