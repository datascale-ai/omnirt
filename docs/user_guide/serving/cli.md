# CLI 文档

OmniRT 当前公开 6 个顶层 CLI 命令：

- `omnirt generate`
- `omnirt validate`
- `omnirt models`
- `omnirt serve`
- `omnirt bench`
- `omnirt worker`

更细的参数表见 [CLI 参考](../../cli_reference/index.md)。这里优先给任务导向示例。

## 请求结构

CLI 与 `GenerateRequest` 共用一份请求结构：

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

经验规则：

- `inputs` 放语义输入，如 `prompt`、`image`、`mask`、`audio`
- `config` 放执行配置，如 `preset`、`scheduler`、`device_map`、`quantization`

## `omnirt generate`

直接执行一次请求：

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" \
  --preset fast
```

从文件执行：

```bash
omnirt generate --config request.yaml --json
```

只做解析和校验，不真正推理：

```bash
omnirt generate --config request.yaml --dry-run
```

## `omnirt validate`

```bash
omnirt validate \
  --task text2image \
  --model flux2.dev \
  --prompt "a poster with bold typography" \
  --backend cpu-stub
```

适合在真机执行前确认：

- 模型和任务是否匹配
- 输入字段是否齐全
- 默认 backend / config 是否被正确解析

## `omnirt models`

```bash
omnirt models
omnirt models sdxl-base-1.0
omnirt models --format markdown
```

## `omnirt serve`

```bash
omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

如果你想把默认放置策略透传到所有服务请求：

```bash
omnirt serve --device-map balanced --devices cuda:0,cuda:1
```

## `omnirt serve --protocol flashtalk-ws`

启动 FlashTalk 兼容 WebSocket 服务，供 OpenTalking 等实时数字人客户端接入：

```bash
omnirt serve \
  --protocol flashtalk-ws \
  --host 0.0.0.0 \
  --port 8765 \
  --repo-path /path/to/SoulX-FlashTalk \
  --ckpt-dir models/SoulX-FlashTalk-14B \
  --wav2vec-dir models/chinese-wav2vec2-base
```

如果模型环境没有完整安装 OmniRT 依赖，优先使用脚本的轻量入口：

```bash
bash scripts/start_flashtalk_ws.sh
```

完整配置见 [FlashTalk 兼容 WebSocket](flashtalk_ws.md)。

## `omnirt worker`

```bash
omnirt worker \
  --host 0.0.0.0 \
  --port 50061 \
  --worker-id sdxl-a \
  --backend cuda
```

与 `serve` 搭配：

```bash
omnirt serve \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0,sdxl-refiner-1.0'
```

## `omnirt bench`

内置场景：

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --warmup 2 \
  --output bench.json
```

自定义请求：

```bash
omnirt bench \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic portrait under neon rain" \
  --concurrency 4 \
  --total 20 \
  --batch-window-ms 50 \
  --max-batch-size 4
```

## Legacy / 运行时优化示例

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse" \
  --enable-layerwise-casting \
  --quantization int8 \
  --quantization-backend torchao \
  --enable-tea-cache
```

这类配置是否真正生效，建议配合 `RunReport` 和 `/metrics` 一起看。详见 [Legacy 优化指南](../../developer_guide/legacy_optimization_guide.md)。
