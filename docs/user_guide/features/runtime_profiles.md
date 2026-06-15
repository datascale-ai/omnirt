# Runtime Profile 与 Capability Manifest

OmniRT 的运行时侧核心概念是：

- `Runtime Profile`：一组模型、后端、资源、预热、并发和降级配置。
- `Model Capability Manifest`：模型支持的任务、输入输出、流式能力、常驻能力、硬件后端和成熟度声明。
- `Benchmark Scenario`：用于测 TTFF、首包、端到端耗时、显存、并发和稳定性的标准压测场景。
- `Integration Recipe`：面向 OpenTalking、Agent 框架、自研前端的接入示例。

这些概念替代业务“场景包”在 OmniRT 核心中的位置。业务场景、Persona、知识库和客户页面属于上层系统。

## Model Capability Manifest

查看单个模型：

```bash
omnirt models indextts --manifest
```

查看 Core 模型能力：

```bash
omnirt models --tier core --manifest
```

Manifest 字段：

| 字段 | 含义 |
|---|---|
| `model` / `task` | registry id 与任务面 |
| `tier` / `role` / `maturity` | 维护层级、链路角色和成熟度 |
| `inputs` / `optional_inputs` / `outputs` | 输入输出契约 |
| `config` / `default_config` | 可配置项与默认值 |
| `streaming` | 是否提供流式服务语义 |
| `resident` | 是否推荐常驻 worker / 常驻服务 |
| `service_adapter` | 服务化 adapter 名称，如 `text2audio.service.v1` |
| `backends` | CUDA / Ascend / CPU stub 支持状态 |

## Runtime Profile

示例见 `examples/profiles/realtime-avatar-local.yaml`。

校验：

```bash
omnirt profile validate examples/profiles/realtime-avatar-local.yaml
omnirt profile validate examples/profiles/realtime-avatar-local.yaml --json
```

Profile 不负责启动业务页面，它只描述运行时需要启动哪些模型服务、使用哪个后端、占用什么端口、如何预热、最大并发是多少，以及繁忙或失败时降级到哪个模型。

## Text2Audio Adapter

TTS 模型优先走 service-backed adapter，而不是强制塞进离线 `omnirt generate` 主路径。统一入口：

- `GET /v1/text2audio/models`
- `GET /v1/text2audio/health`
- `GET /v1/text2audio/metrics`
- `POST /v1/text2audio/warmup`
- `POST /v1/text2audio/stream`

`/v1/text2audio/stream` 接收统一 JSON：

```json
{
  "model": "indextts",
  "text": "你好，我是 OmniRT 实时语音。",
  "speaker_profile": "default-female",
  "prompt_audio": "/models/voices/default.wav",
  "reference_text": "参考音色文本",
  "audio_format": "pcm_s16le",
  "stream": true,
  "config": {
    "streaming_mode": "token_window",
    "temperature": 0.8
  }
}
```

当前 IndexTTS 已支持该通用入口，同时保留 `/v1/text2audio/indextts` 兼容路径。

## Integration Recipes

- `examples/integrations/opentalking`
- `examples/integrations/agent-service`
- `examples/integrations/http-cli-demo`

OpenTalking 是重点参考接入方之一，但不是 OmniRT 的唯一目标用户。
