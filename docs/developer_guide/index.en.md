# Developer Guide

For developers **contributing to OmniRT** — adding models, adding backends, or understanding how the runtime fits together.

- **[Contributing](contributing.md)** — dev setup, tests, PR workflow, documentation conventions
- **[Architecture](architecture.md)** — how the interface layer, engine, executors, middleware, observability, and distributed extensions fit together
- **[Legacy Optimization Guide](legacy_optimization_guide.md)** — offload, layout, quantization, and TeaCache knobs for `legacy_call` families
- **[Benchmark Baseline](benchmark_baseline.md)** — bench scenarios, JSON metrics, and release acceptance guidance
- **[FlashTalk Resident Benchmark](flashtalk_resident_benchmark.md)** — first real-hardware resident benchmark on `Ascend 910B2 x8`
- **[Model onboarding](model_onboarding.md)** — how to register a new model family and pass validation
- **[Backend onboarding](backend_onboarding.md)** — how to implement `BackendRuntime` and wire in a new hardware backend

!!! tip "First contribution?"
    Start with [Contributing](contributing.md) and [Architecture](architecture.md), then pick [Model Onboarding](model_onboarding.md) or [Backend Onboarding](backend_onboarding.md) based on your goal.
