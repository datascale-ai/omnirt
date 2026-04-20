# 开发者指南

本指南面向**向 OmniRT 贡献代码**的开发者：想接入新模型、新后端、或者理解运行时内部分层。

- **[参与贡献](contributing.md)** — 开发环境、测试、PR 流程、文档约定
- **[架构说明](architecture.md)** — 接口层、engine、executor、middleware、观测与分布式扩展如何协作
- **[Legacy 优化指南](legacy_optimization_guide.md)** — `legacy_call` 家族可用的 offload、layout、量化与 TeaCache 开关
- **[Benchmark 基线](benchmark_baseline.md)** — bench 场景、JSON 指标和 release 验收口径
- **[模型接入](model_onboarding.md)** — 如何把一个新模型族注册进 registry 并通过校验
- **[后端接入](backend_onboarding.md)** — 如何实现 `BackendRuntime` 把新硬件接进来

!!! tip "第一次贡献？"
    先读 [参与贡献](contributing.md) 和 [架构说明](architecture.md)，再按你的目标选 [模型接入](model_onboarding.md) 或 [后端接入](backend_onboarding.md)。
