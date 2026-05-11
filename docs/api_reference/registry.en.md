# `omnirt.core.registry`

The model registry: the `@register_model` decorator, `ModelSpec`, `ModelCapabilities`, and discovery APIs. Auto-rendered reference: [→ API Reference / omnirt.core.registry](../../api_reference/registry/).

Summary:

- **`@register_model(id, *, task, capabilities=...)`** — decorator applied to a `BasePipeline` subclass to expose it in the registry.
- **`ModelSpec`** — entry in the registry (id, task, pipeline class, capabilities).
- **`ModelCapabilities`** — per-model metadata: inputs, config surface, adapters, maturity, maintenance `tier`, `alias_of`, and `summary`.
- **`register_model` / `get_model` / `list_models`** — programmatic API; the same data is what powers `omnirt models` and `omnirt.list_available_models`.

For how to onboard a new model, see [Model Onboarding](../developer_guide/model_onboarding.md).
