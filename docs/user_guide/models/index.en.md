# Models

OmniRT's models are owned by the **registry**: every model uses `@register_model` to declare its supported tasks, accepted adapters, minimum VRAM, and recommended preset.

From this stage on, model count is no longer the main goal. Maintenance is organized around the digital-human pipeline:

- **Core**: TTS, audio-driven avatars, realtime / resident workers, deployment, and benchmarks. These require real-hardware smoke evidence.
- **Adjacent**: avatar assets, backgrounds, idle video material, and post-processing for digital-human products.
- **Experimental**: existing general image / video integrations. They keep registry entries and basic tests, but do not promise dual-backend validation.

This section is organized around three tables:

| Page | Purpose |
|---|---|
| [Supported Models](supported_models.md) | auto-generated full registry (equivalent to `omnirt models`) |
| [Support Status](support_status.md) | manually curated digital-human priorities, real-hardware smoke, and contraction status |
| [Roadmap](roadmap.md) | digital-human main line, adjacent capabilities, and experimental model boundaries |

!!! tip "Query models from the CLI"
    `omnirt models` lists everything; `omnirt models <id>` dumps a model's `ModelCapabilities` (tasks, adapters, VRAM, recommended preset).
