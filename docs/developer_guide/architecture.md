# OmniRT 架构说明

OmniRT 当前已经从“单进程 pipeline 包装器”演进成一个带 queue、executor、观测和远程 worker 扩展位的生成运行时。

## 主体分层

### 1. 接口层

- Python API：`omnirt.generate(...)`、`omnirt.validate(...)`
- CLI：`generate / validate / models / serve / bench / worker`
- HTTP：原生 `/v1/generate` 与 OpenAI 兼容路由

这一层的职责是把外部输入统一归一成 `GenerateRequest`。

### 2. 契约与注册层

- `GenerateRequest` / `GenerateResult` / `RunReport`
- registry / presets / validation
- 模型别名解析

这一层决定“一个请求是否合法、最终解析成什么执行配置”。

### 3. Engine 与派发层

- `OmniEngine`
- `JobQueue`
- `RequestBatcher`
- `InMemoryJobStore` / `RedisJobStore`
- `Controller`

这一层负责：

- 同步与异步入口统一
- 本地队列和 job 生命周期
- batching
- 将请求留在本地执行，或转发给远程 worker

### 4. Executor 层

当前有三条执行路径：

| execution_mode | 说明 |
|---|---|
| `modular` | 面向已迁移家族的组件化执行路径 |
| `legacy_call` | 对现有 Diffusers pipeline 的包装执行 |
| `subprocess` | 外部脚本 / 仓库驱动的执行模式，如 FlashTalk |

`ModelSpec.execution_mode` 决定 engine 最终落到哪条路径。

### 5. 模型 / 启动器 / 后端层

- 模型家族实现位于 `src/omnirt/models/`
- launcher 负责 `python / torchrun / accelerate`
- backend 负责运行时设备与编译封装

这里同时承接：

- `device_map` / `devices`
- legacy 官方优化开关
- quantization / layerwise casting / TeaCache

### 6. 观测层

- `RunReport`
- Prometheus metrics
- Trace recorder + OTLP exporter
- SSE / WebSocket / Realtime 事件流

这一层让一次执行既能被“请求内返回”，也能被“服务外部抓取”。

## 同步执行路径

1. 接口层创建 `GenerateRequest`
2. validation 解析模型、任务和 config
3. `OmniEngine.run_sync()` 选择本地 executor 或远程 worker
4. executor 运行模型并生成 `GenerateResult`
5. telemetry 填充 `RunReport`

## 异步执行路径

1. `POST /v1/generate` 带 `async_run=true`
2. engine 创建 job，写入 JobStore
3. queue / batcher / controller 处理该 job
4. 事件持续写入 job stream
5. 客户端从 `GET /v1/jobs/{id}`、SSE、WebSocket 或 trace 路由消费

## 分布式扩展点

当前已经落地的扩展点：

- gRPC worker transport
- `Controller` 路由远程 worker
- `RedisJobStore`
- OTLP/HTTP trace 导出

当前仍然保持轻量的部分：

- gRPC transport 是最小 unary RPC，不是完整控制面
- `ROCm / XPU` 仍是实验性 backend 占位

## 稳定契约

OmniRT 对外最重要的稳定面是：

- `GenerateRequest`
- `GenerateResult`
- `RunReport.schema_version`

内部 executor、middleware、launcher 可以继续演进，但这三项应尽量保持向后兼容。

## 相关

- [服务协议](../user_guide/features/service_schema.md)
- [派发与队列](../user_guide/features/dispatch_queue.md)
- [遥测](../user_guide/features/telemetry.md)
