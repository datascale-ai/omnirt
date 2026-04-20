# Legacy 优化指南

本文面向仍运行在 `execution_mode="legacy_call"` 的家族，例如 `sd15`、`sd3`、`svd`、`animatediff_sdxl`、`chronoedit`、`generalist_image`、`video_family`。

这些优化项的目标不是“保证所有模型都生效”，而是把上游 Diffusers 和运行时已经能接受的能力，统一暴露到 OmniRT 的请求配置里。

## 设计原则

- **先保证可运行，再逐步加速**：先确认 baseline 能跑，再叠加 offload、layout、cache、量化
- **尽量一次只开一类优化**：这样回归更容易定位
- **所有优化都是 best-effort**：如果上游 pipeline / 组件不支持某个 hook，OmniRT 会跳过，而不是强行报错

## 可用配置项

| 配置项 | 典型值 | 作用 | 备注 |
|---|---|---|---|
| `enable_model_cpu_offload` | `true` | 模型级 CPU offload | 优先用于单卡显存不够但还能接受更高延迟的场景 |
| `enable_sequential_cpu_offload` | `true` | 更激进的顺序 offload | 比 model offload 更省显存，通常也更慢 |
| `enable_group_offload` | `true` | 分组 offload | 需要 pipeline 暴露 `enable_group_offload()` |
| `group_offload_type` | `block_level` | 分组 offload 颗粒度 | 与上游实现保持一致 |
| `group_offload_use_stream` | `true` | offload 传输是否用 stream | 主要影响吞吐和抖动 |
| `group_offload_disk_path` | `/path/to/cache` | 允许 offload 到磁盘 | 仅在上游实现支持时生效 |
| `enable_vae_slicing` | `true` | VAE slicing | 常见于图像大分辨率或视频 decode 压力场景 |
| `enable_vae_tiling` | `true` | VAE tiling | 通常和 slicing 一起尝试 |
| `channels_last` | `true` | `torch.channels_last` 内存布局 | 常见于卷积密集模型 |
| `fuse_qkv` | `true` | QKV 融合 | 仅组件支持 `fuse_qkv_projections()` 时有效 |
| `quantization` | `int8` / `fp8` / `nf4` | best-effort 量化入口 | 与 `quantization_backend` 配合使用 |
| `quantization_backend` | `torchao` | 指定量化后端 | 当前优先尝试 `torchao` |
| `enable_layerwise_casting` | `true` | 分层 casting | 适合显存吃紧但想保留较高计算 dtype 的场景 |
| `layerwise_casting_storage_dtype` | `fp8_e4m3fn` | 分层存储 dtype | 是否真正生效取决于后端和组件 |
| `layerwise_casting_compute_dtype` | `bf16` | 分层计算 dtype | 常与 `enable_layerwise_casting` 搭配 |
| `cache` / `enable_tea_cache` | `tea_cache` / `true` | TeaCache 入口 | 命中前提取决于模型组件是否暴露兼容 hook |
| `tea_cache_ratio` | `0.2` | TeaCache 复用比例提示 | 建议从小值开始调 |
| `tea_cache_interval` | `2` | TeaCache 间隔提示 | 视频任务可适当调大 |

## 推荐起手式

### 显存优先

```yaml
config:
  enable_model_cpu_offload: true
  enable_vae_slicing: true
  enable_vae_tiling: true
```

适合 12 GB 到 24 GB 这类边界显存机器。先追求“能跑”，再看是否还能接受时延。

### 吞吐优先

```yaml
config:
  channels_last: true
  fuse_qkv: true
```

适合已经能稳定运行、正在挤吞吐的单机环境。建议配合 [Benchmark 基线](benchmark_baseline.md) 一起看收益。

### 实验性压显存

```yaml
config:
  quantization: int8
  quantization_backend: torchao
  enable_layerwise_casting: true
  layerwise_casting_compute_dtype: bf16
```

这类配置已经有统一接缝，但仍建议你在目标硬件上单独验证精度和性能。

## 调参建议

1. 先只打开 `enable_vae_slicing` / `enable_vae_tiling`
2. 显存仍然不够，再尝试 `enable_model_cpu_offload`
3. 需要更激进压显存时，再考虑 `enable_sequential_cpu_offload` 或 `group_offload`
4. 量化、layerwise casting、TeaCache 放到最后，因为这几类的收益和副作用更依赖具体模型

## 如何判断有没有生效

- 看 `RunReport.config_resolved` 是否包含你传入的配置
- 看 `RunReport.cache_hits` 是否出现 `text_embedding`
- 看 `RunReport.device_placement`、`memory`、`timings` 有没有随配置变化
- 开启服务模式时，再结合 `/metrics` 和 `/v1/jobs/{id}/trace` 看阶段耗时变化

## 已知边界

- `legacy_call` 家族并不保证全部支持 `device_map` 风格的托管放置
- `torchao`、TeaCache、layerwise casting 都依赖可选运行时依赖和上游组件方法
- 即使配置被接受，也不等于所有子模块都真正执行了相同优化

## 相关

- [架构说明](architecture.md)
- [Benchmark 基线](benchmark_baseline.md)
- [遥测](../user_guide/features/telemetry.md)
