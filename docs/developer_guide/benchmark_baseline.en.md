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

## SoulX-LiveAct Ascend Baseline

The `soulx-liveact-14b` script-backed wrapper has completed an Ascend real-hardware validation run. Test scope:

- Inputs: `examples/image/1.png` + `examples/audio/1.wav`
- Resolution / FPS: `416*720`, `fps=20`
- Inference: `--sample-steps 1 --rank0-t5-only --use-lightvae --vae-path models/vae/lightvaew2_1.pth --use-cache-vae --stage-profile`
- Placement: `--text-cache-visible-devices 2 --visible-devices 2,3,4,5`, meaning one NPU prepares the T5 text cache before the 4-NPU inference job
- Output video: `416x720`, `20fps`, `755 frames`, `37.75s`

| run | cache state | wall_s | stage total avg | Key stages |
|---|---|---:|---:|---|
| `cold` | text cache rebuilt; condition cache miss | 190 | 112.9584s | `prepare_text_cache total=11.10s`, `sample_model_forward avg=9.8508s`, `vae_decode avg=21.3054s` |
| `warm` | text cache hit; condition cache hit; still initialized T5 | 207 | 132.6558s | `prepare_text_cache total=10.09s`, `sample_model_forward avg=16.9334s`, `vae_decode avg=22.7612s` |
| `warm2` | text cache skipped; condition cache hit | 169 | 121.2429s | `sample_model_forward avg=15.9815s`, `vae_decode avg=23.3559s`, `export avg=13.5125s` |

Use `warm2` as the current warm baseline. The first `warm` run measured `207s` because the wrapper still called upstream `prepare_text_cache.py` on a cache hit. That script initializes T5 before checking for the hit, adding about `10s`. The wrapper now checks `/tmp/liveact_text_ctx_*.pt` before launching that script and skips it when all expected cache files exist. This path does not use CPU T5.

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
- [FlashHead Benchmark](flashhead_benchmark.md)
