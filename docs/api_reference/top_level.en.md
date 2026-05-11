# `omnirt`

Top-level functions — the recommended Python entry points. The auto-rendered signatures and docstrings live on the default-locale page to avoid duplicate anchors across languages: [→ API Reference / omnirt](../../api_reference/top_level/).

Summary:

- **`omnirt.generate(request, *, backend=None)`** — Execute a `GenerateRequest` and return a `GenerateResult`.
- **`omnirt.validate(request, *, backend=None)`** — Validate a request without touching hardware.
- **`omnirt.pipeline(model, *, backend=None)`** — Build an `OmniModelPipeline` for repeated use.
- **`omnirt.list_available_models(*, include_aliases=True, tier=None)`** — List `ModelSpec` entries, optionally filtered by `core`, `adjacent`, or `experimental`.
- **`omnirt.describe_model(model_id, *, task=None)`** — Return the `ModelSpec` for one registry id.
- **`omnirt.available_presets()`** — Tuple of the preset names recognized by the runtime.

For task-oriented examples see [Python API guide](../user_guide/serving/python_api.md).
