# Benchmark Baseline

This document captures the recommended benchmark methodology for OmniRT and what should be archived before a release. The goal is not one absolute set of numbers for every machine. The goal is reproducible comparisons across phases and commits.

## Standard output shape

`omnirt bench` currently emits these core fields:

| Field | Meaning |
|---|---|
| `throughput_rps` | throughput in requests per second |
| `latency_ms.p50 / p95 / p99` | end-to-end latency percentiles |
| `ttft_ms.p50 / p95 / p99` | time-to-first-event percentiles |
| `peak_vram` | peak memory / VRAM observed in this run |
| `cache_hit_ratio` | fraction of requests that hit the result cache |
| `batch_size_mean` | average batch size |
| `batched_request_ratio` | fraction of requests that were merged into a batch |
| `execution_mode_breakdown` | distribution across `modular / legacy_call / subprocess / persistent_worker` |

Always persist the JSON:

```bash
omnirt bench ... --output bench.json --json
```

## Built-in scenario

Current built-in scenario:

- `text2image_sdxl_concurrent4`

Example:

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --warmup 2 \
  --output bench-sdxl-c4.json
```

## P2 close-out baseline: result cache

The goal is to verify that prompt-embedding reuse actually hits on repeated same-prompt requests.

Suggested command:

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

What to inspect:

- `cache_hit_ratio` should be clearly above 0
- inspect `RunReport.cache_hits` and confirm `text_embedding` appears
- for stage-level analysis, pair the run with `/v1/jobs/{id}/trace` or structured logs and inspect `encode_prompt`

## P3 close-out baseline: dynamic batching

The goal is to verify that the modular `text2image` path actually batches under concurrency.

Control:

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --batch-window-ms 0 \
  --max-batch-size 1 \
  --output bench-nobatch.json
```

Experiment:

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --batch-window-ms 50 \
  --max-batch-size 4 \
  --output bench-batch.json
```

What to inspect:

- whether `throughput_rps` improves
- whether `batch_size_mean` rises above 1
- whether `batched_request_ratio` is materially above 0
- whether `latency_ms.p95` still fits your service target

## What to archive before release

Each benchmark artifact should be stored with:

- commit SHA
- hardware model and count
- backend (`CUDA / Ascend / ...`)
- driver, Torch, and Diffusers versions
- model source and weight precision
- full CLI command
- the JSON report itself

## CI baseline vs real-hardware baseline

- CI / local fake runtime: verify report structure, non-zero metrics, and schema stability
- real hardware benchmark: verify throughput, latency, VRAM, and multi-device gains

Do not treat CPU stub or fake-runtime numbers as release performance claims. They are best used for contract and regression checks.

## How to compare results

- compare two commits on the same machine first
- then compare one commit across "cache off vs cache on" or "batching off vs batching on"
- avoid cross-machine, cross-driver, or cross-weight-source absolute comparisons

## Related

- [Dispatch & Queue](../user_guide/features/dispatch_queue.md)
- [Telemetry](../user_guide/features/telemetry.md)
- [Legacy Optimization Guide](legacy_optimization_guide.md)
