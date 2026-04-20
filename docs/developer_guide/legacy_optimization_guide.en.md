# Legacy Optimization Guide

This guide targets families still running with `execution_mode="legacy_call"`, such as `sd15`, `sd3`, `svd`, `animatediff_sdxl`, `chronoedit`, `generalist_image`, and `video_family`.

The goal is not to guarantee that every knob works for every model. The goal is to expose the optimizations already understood by upstream Diffusers and the runtime through one consistent OmniRT request surface.

## Design principles

- **make the model run first, optimize second**: confirm the baseline path before stacking offload, layout, cache, or quantization
- **turn on one class of optimization at a time**: this keeps regressions explainable
- **everything is best-effort**: if an upstream pipeline or component does not expose a compatible hook, OmniRT skips it instead of forcing a hard failure

## Available config keys

| Config key | Typical value | Purpose | Notes |
|---|---|---|---|
| `enable_model_cpu_offload` | `true` | model-level CPU offload | good first choice when VRAM is tight and extra latency is acceptable |
| `enable_sequential_cpu_offload` | `true` | more aggressive sequential offload | saves more VRAM and is usually slower |
| `enable_group_offload` | `true` | grouped offload | requires the pipeline to expose `enable_group_offload()` |
| `group_offload_type` | `block_level` | grouped-offload granularity | follows upstream semantics |
| `group_offload_use_stream` | `true` | use stream for offload transfer | mainly affects throughput and jitter |
| `group_offload_disk_path` | `/path/to/cache` | allow disk-backed offload | only works when the upstream implementation supports it |
| `enable_vae_slicing` | `true` | VAE slicing | common for large images or video decode pressure |
| `enable_vae_tiling` | `true` | VAE tiling | often paired with slicing |
| `channels_last` | `true` | `torch.channels_last` memory layout | common for convolution-heavy models |
| `fuse_qkv` | `true` | QKV fusion | only effective when the component exposes `fuse_qkv_projections()` |
| `quantization` | `int8` / `fp8` / `nf4` | best-effort quantization entry point | pair with `quantization_backend` |
| `quantization_backend` | `torchao` | select the quantization backend | currently `torchao` is tried first |
| `enable_layerwise_casting` | `true` | layerwise casting | useful when VRAM is constrained but you want a higher compute dtype |
| `layerwise_casting_storage_dtype` | `fp8_e4m3fn` | layerwise storage dtype | real effect depends on backend and component support |
| `layerwise_casting_compute_dtype` | `bf16` | layerwise compute dtype | commonly paired with `enable_layerwise_casting` |
| `cache` / `enable_tea_cache` | `tea_cache` / `true` | TeaCache entry point | actual reuse depends on compatible component hooks |
| `tea_cache_ratio` | `0.2` | TeaCache reuse hint | start small |
| `tea_cache_interval` | `2` | TeaCache interval hint | video workloads often benefit from larger values |

## Recommended starting points

### VRAM first

```yaml
config:
  enable_model_cpu_offload: true
  enable_vae_slicing: true
  enable_vae_tiling: true
```

Good for 12 GB to 24 GB boundary-VRAM machines. Aim for "it runs" before tuning latency.

### Throughput first

```yaml
config:
  channels_last: true
  fuse_qkv: true
```

Good once the model is already stable and you want more throughput. Measure it together with [Benchmark Baseline](benchmark_baseline.md).

### Experimental memory reduction

```yaml
config:
  quantization: int8
  quantization_backend: torchao
  enable_layerwise_casting: true
  layerwise_casting_compute_dtype: bf16
```

These hooks are available, but you should still validate quality and performance on target hardware.

## Tuning order

1. start with `enable_vae_slicing` and `enable_vae_tiling`
2. if VRAM is still insufficient, try `enable_model_cpu_offload`
3. if you need a more aggressive reduction, move to `enable_sequential_cpu_offload` or `group_offload`
4. leave quantization, layerwise casting, and TeaCache for last because their gains and tradeoffs are more model-specific

## How to tell whether it worked

- inspect `RunReport.config_resolved` for the final applied config
- inspect `RunReport.cache_hits` for entries such as `text_embedding`
- compare `RunReport.device_placement`, `memory`, and `timings`
- in server mode, combine `/metrics` and `/v1/jobs/{id}/trace` to see stage-level changes

## Known boundaries

- `legacy_call` families do not guarantee full support for managed `device_map` placement
- `torchao`, TeaCache, and layerwise casting all depend on optional runtime dependencies and compatible upstream methods
- a config key being accepted does not mean every submodule applied the same optimization

## Related

- [Architecture](architecture.md)
- [Benchmark Baseline](benchmark_baseline.md)
- [Telemetry](../user_guide/features/telemetry.md)
