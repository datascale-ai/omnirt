# Features

Beyond the `GenerateRequest` contract, OmniRT provides several cross-cutting capabilities for validation, queuing, caching, observability, and service integration.

| Feature | Purpose | Page |
|---|---|---|
| **Presets** | `fast` / `balanced` / `quality` / `low-vram` tags that bundle steps / dtype / guidance | [Presets](presets.md) |
| **Request validation** | Catch contract errors before using real hardware | [Validation](validation.md) |
| **Service schema** | Field-level reference for `GenerateRequest` / `GenerateResult` / `RunReport` | [Service Schema](service_schema.md) |
| **Runtime Profile / Capability Manifest** | Model composition, resources, warmup, streaming, resident mode, and backend status | [Runtime Profiles](runtime_profiles.md) |
| **Dispatch queue** | Async engine, concurrency, dynamic batching, JobStore, remote workers | [Dispatch & Queue](dispatch_queue.md) |
| **Telemetry** | `RunReport`, Prometheus, OTLP traces, SSE / WebSocket event streams | [Telemetry](telemetry.md) |
