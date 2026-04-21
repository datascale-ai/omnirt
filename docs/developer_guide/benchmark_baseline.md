# Benchmark 基线

这份文档记录 OmniRT 当前推荐的 benchmark 口径，以及发布前应归档哪些结果。它的重点不是给出“所有机器都适用的一组绝对数字”，而是让不同阶段、不同提交的结果可重复比较。

## 统一输出格式

`omnirt bench` 当前会输出这些核心字段：

| 字段 | 含义 |
|---|---|
| `throughput_rps` | 吞吐（请求 / 秒） |
| `latency_ms.p50 / p95 / p99` | 端到端时延分位数 |
| `ttft_ms.p50 / p95 / p99` | 首个流式事件延迟分位数 |
| `peak_vram` | 本轮样本里的峰值显存 / 内存值 |
| `cache_hit_ratio` | 命中结果缓存的请求比例 |
| `batch_size_mean` | 平均 batch 大小 |
| `batched_request_ratio` | 被合批请求的比例 |
| `execution_mode_breakdown` | `modular / legacy_call / subprocess / persistent_worker` 分布 |

建议每次都把 JSON 落盘：

```bash
omnirt bench ... --output bench.json --json
```

## 内置场景

当前内置场景：

- `text2image_sdxl_concurrent4`

示例：

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --warmup 2 \
  --output bench-sdxl-c4.json
```

## P2 收尾基线：结果缓存

目标是验证同 prompt 重复请求时，prompt embedding 缓存是否真正命中。

建议命令：

```bash
omnirt bench \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic portrait of a traveler under neon rain" \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 30 \
  --concurrency 1 \
  --total 50 \
  --warmup 1 \
  --batch-window-ms 0 \
  --max-batch-size 1 \
  --output bench-cache.json
```

观察点：

- `cache_hit_ratio` 应该明显高于 0
- 结合 `RunReport.cache_hits`，确认命中项包含 `text_embedding`
- 如果要做更细的阶段级分析，再结合 `/v1/jobs/{id}/trace` 或结构化日志看 `encode_prompt` 耗时

## P3 收尾基线：动态 batching

目标是验证 `text2image` modular 路径在并发场景下是否真的合批。

对照组：

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --batch-window-ms 0 \
  --max-batch-size 1 \
  --output bench-nobatch.json
```

实验组：

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --batch-window-ms 50 \
  --max-batch-size 4 \
  --output bench-batch.json
```

观察点：

- `throughput_rps` 是否提升
- `batch_size_mean` 是否大于 1
- `batched_request_ratio` 是否显著高于 0
- `latency_ms.p95` 是否仍在你的服务目标范围内

## 发布前建议归档的信息

每份 benchmark 结果都建议和这些上下文一起保存：

- commit SHA
- 硬件型号与数量
- 后端（CUDA / Ascend / 其他）
- 驱动、Torch、Diffusers 版本
- 模型来源与权重精度
- 完整 CLI 命令
- 产出的 JSON 报告

## CI 与真机基线怎么分工

- CI / 本地 fake runtime：验证字段是否齐全、报告结构是否稳定、指标有没有回归为 0
- 真机 benchmark：验证吞吐、时延、显存和多卡收益

不要把 CPU stub 或 fake runtime 的数值当作发布性能结论；它更适合做契约和回归检查。

## 解释结果时的建议

- 先横向比同一硬件上的两个提交
- 再纵向比同一提交在“不开缓存 / 开缓存”“不开 batching / 开 batching”下的差异
- 避免跨机器、跨驱动、跨权重来源直接比较绝对数值

## 相关

- [派发与队列](../user_guide/features/dispatch_queue.md)
- [遥测](../user_guide/features/telemetry.md)
- [Legacy 优化指南](legacy_optimization_guide.md)
