# FlashTalk Resident Benchmark

This document captures the first real-hardware benchmark for `soulx-flashtalk-14b` under the `persistent_worker` resident serving path, so later commits can compare against the same baseline.

## Test environment

- Date: `2026-04-21`
- Machine: `internal Ascend validation host`
- Accelerator: `Ascend 910B2 x8`
- Serving path: `omnirt serve` + `persistent_worker` + resident `torchrun`
- Model stack:
  `soulx-flashtalk-14b`
  `SoulX-FlashTalk-14B`
  `chinese-wav2vec2-base`

## Benchmark configuration

This run matches the “speed-first” realtime profile documented in the paired LiveAct benchmark summary:

- `FLASHTALK_HEIGHT=704`
- `FLASHTALK_WIDTH=416`
- `FLASHTALK_FRAME_NUM=29`
- `FLASHTALK_MOTION_FRAMES_NUM=1`
- `FLASHTALK_SAMPLE_STEPS=2`
- `FLASHTALK_COLOR_CORRECTION_STRENGTH=0`
- `audio_encode_mode=stream`
- Inputs:
  `examples/woman2.jpg`
  `examples/cantonese_16k.wav`

The resident server was also configured with:

- `max_concurrency=1`
- `pipeline_cache_size=1`

## Metric definitions

Three timing layers matter here:

- `cold request`
  First request, including model load, distributed initialization, and resident worker warmup.
- `steady_chunk_core_ms_avg`
  Hot-path average for the core `run_pipeline(...)` chunk work. This is the closest metric to the “steady chunk” numbers in `SESSION_SUMMARY`.
- `steady_chunk_total_ms_avg`
  Hot-path average chunk time including audio embedding and `video.cpu()` handling.

Do not compare `denoise_loop_ms / chunk_count` directly with the old `Generate video chunk-x done` logs; the scopes differ.

## Summary

### Cold start

| Scenario | End-to-end |
|---|---:|
| `max_chunks=1` | `88.409s` |
| `max_chunks=3` | `91.196s` |
| Full audio (`937` frames) | `121.029s` |

### Hot path

| Scenario | End-to-end | Notes |
|---|---:|---|
| `max_chunks=1` | `2.672s` | single hot chunk request |
| `max_chunks=3` | `5.514s` | used to estimate steady chunk cost |
| Full audio (`937` frames) | `37.377s` | full resident hot-path video |

### Hot chunk metrics

Hot resident telemetry for `max_chunks=3`:

| Metric | Value |
|---|---:|
| `audio_embedding_ms_avg` | `21.259 ms` |
| `chunk_core_ms_avg` | `894.051 ms` |
| `steady_chunk_core_ms_avg` | `891.002 ms` |
| `chunk_copy_ms_avg` | `33.339 ms` |
| `chunk_total_ms_avg` | `957.137 ms` |
| `steady_chunk_total_ms_avg` | `953.662 ms` |

## Comparison with the standalone script

On the same internal validation host, with the same `704x416 + 29/1 + 2 step + stream` configuration, the direct `generate_video.py` run reported:

- `chunk-0`: `6.59s`
- `chunk-1`: `0.89s`
- `chunk-2`: `0.90s`

That means:

- `steady_chunk_core_ms_avg ≈ 891ms`
  is effectively aligned with the standalone script’s `0.89s` to `0.90s`.
- The remaining resident overhead sits mostly in:
  - audio embedding
  - `video.cpu()` and per-chunk tail work

In other words, the `persistent_worker` path is not slowing down the core FlashTalk DiT generation loop.

## Conclusion

- `soulx-flashtalk-14b` in OmniRT’s `persistent_worker` resident path is now in a usable state.
- Under the realtime profile, the core generation speed matches the previous standalone best case.
- Further optimization work should focus on:
  - `video.cpu()` / result collection
  - chunk-level non-core overhead
  - longer-audio and more-diverse sample stability runs

## Related

- [Benchmark Baseline](benchmark_baseline.md)
- [Architecture](architecture.md)
- [Support Status](../user_guide/models/support_status.md)
