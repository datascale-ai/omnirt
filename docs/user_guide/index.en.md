# User Guide

This guide is for developers **building applications on OmniRT**. You'll find:

- **Getting Started** — [install and first request](../getting_started/quickstart.md)
- **Generation** — [text to audio](generation/text2audio.md), [talking head](generation/talking_head.md), [text to image](generation/text2image.md), [image to image](generation/image2image.md), [text to video](generation/text2video.md), [image to video](generation/image2video.md)
- **Serving** — [CLI](serving/cli.md), [Python API](serving/python_api.md), [HTTP Server](serving/http_server.md)
- **Deployment** — [CUDA](deployment/cuda.md), [Ascend backend](deployment/ascend.md), [domestic deployment](deployment/china_mirrors.md), [Docker](deployment/docker.md)
- **Models** — [supported models](models/supported_models.md), [support status](models/support_status.md), [roadmap](models/roadmap.md)
- **Features** — [presets](features/presets.md), [validation](features/validation.md), [service schema](features/service_schema.md), [dispatch & queue](features/dispatch_queue.md), [telemetry](features/telemetry.md)

!!! tip "Still new to OmniRT?"
    Start with [Quickstart](../getting_started/quickstart.md) — one `omnirt validate` plus one `omnirt generate` — before diving deeper.

!!! info "Looking for API reference?"
    A full auto-generated reference is in progress. For now the [Python API guide](serving/python_api.md) covers `generate`, `validate`, `pipeline`, and every typed request helper.
