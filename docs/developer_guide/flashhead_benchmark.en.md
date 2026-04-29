# FlashHead Benchmark

This document records the first real-hardware benchmark for `soulx-flashhead-1.3b` through OmniRT's `subprocess` wrapper. It is different from the external SoulX-FlashHead resident benchmark: this page measures the cold-start end-to-end path where OmniRT launches `generate_video.py`.

## Environment

- Date: `2026-04-28`
- Machine: `internal Ascend validation host`
- Accelerator: `Ascend 910B2`
- Entry point: `omnirt generate` + `subprocess` + `torchrun`
- Models:
  `soulx-flashhead-1.3b`
  `SoulX-FlashHead-1_3B`
  `wav2vec2-base-960h`
- External checkout: `/path/to/SoulX-FlashHead`
- OmniRT test checkout: `/path/to/omnirt`

## Baseline Config

This run uses the quality-oriented 910B profile:

- `model_type=pro`
- `audio_encode_mode=stream`
- `FLASHHEAD_SAMPLE_STEPS=2`
- `FLASHHEAD_VAE_2D_SPLIT=1`
- `FLASHHEAD_LATENT_CARRY=0`
- `FLASHHEAD_NPU_FUSION_ATTENTION=1`
- Inputs:
  `examples/girl.png`
  `bench_results/bench_10s.wav`
- Output:
  `512x512`
  `25 FPS`
  `10.0s`
  `250 frames`

!!! note "Device visibility"
    Set `ASCEND_RT_VISIBLE_DEVICES` before starting the OmniRT parent process. Passing only `--visible-devices` is not enough, because the resource-budget check runs before the external `torchrun` process starts.

## Command Template

2 NPU:

```bash
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
ASCEND_RT_VISIBLE_DEVICES=2,3 \
PYTHONPATH=/path/to/omnirt/src \
/path/to/flashhead-venv/bin/python -m omnirt generate \
  --task audio2video \
  --model soulx-flashhead-1.3b \
  --backend ascend \
  --image /path/to/SoulX-FlashHead/examples/girl.png \
  --audio /path/to/SoulX-FlashHead/bench_results/bench_10s.wav \
  --repo-path /path/to/SoulX-FlashHead \
  --ckpt-dir models/SoulX-FlashHead-1_3B \
  --wav2vec-dir models/wav2vec2-base-960h \
  --python-executable /path/to/flashhead-venv/bin/python \
  --ascend-env-script /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --launcher torchrun \
  --nproc-per-node 2 \
  --visible-devices 2,3 \
  --sample-steps 2 \
  --vae-2d-split \
  --npu-fusion-attention \
  --output-dir outputs/flashhead_bench_2npu \
  --json
```

For 4 NPU, set both `ASCEND_RT_VISIBLE_DEVICES` and `--visible-devices` to `2,3,4,5`, then set `--nproc-per-node 4`.

## Results

| Config | Wall time | `denoise_loop_ms` | `export_ms` | Output |
|---|---:|---:|---:|---|
| 2 NPU cold start | `82.96s` | `69,501.215 ms` | `264.686 ms` | `512x512 / 10s / 250 frames` |
| 4 NPU cold start | `84.08s` | `69,963.908 ms` | `237.519 ms` | `512x512 / 10s / 250 frames` |

Interpretation:

- The OmniRT `subprocess` wrapper can complete real `audio2video` generation on 910B.
- In cold-start mode, 4 NPU does not beat 2 NPU; distributed initialization, model loading, data preparation, and first-operator warmup offset the extra parallelism.
- To measure steady-state multi-NPU gains, benchmark a future `persistent_worker` / resident path instead of a single cold-start script launch.

## Artifact Checks

| Config | RunReport `run_id` | Remote output directory | Checks |
|---|---|---|---|
| 2 NPU | `8361015e-f1d6-4ad7-8c3c-6f3680354fa1` | `outputs/flashhead_bench_20260428_180909` | `ffprobe`: `512x512 / 25 FPS / 10.0s / 250 frames`; no `blackdetect` / `freezedetect` warnings |
| 4 NPU | `79ebe868-609a-4f5f-a571-6366d984aeb2` | `outputs/flashhead_bench_20260428_181056` | `ffprobe`: `512x512 / 25 FPS / 10.0s / 250 frames`; no `blackdetect` / `freezedetect` warnings |

## Notes

- Running the OmniRT CLI in the remote venv requires `protobuf` and `grpcio`; this run installed them into `/path/to/flashhead-venv` from the Tsinghua PyPI mirror.
- This page tracks the OmniRT `subprocess` cold-start path, not service-mode hot latency.
- `latent_carry=false` is the default quality profile. `latent_carry=true` can reduce part of VAE encode overhead, but prior adaptation notes observed style drift, so it is not the default display profile.

## Related

- [Benchmark Baseline](benchmark_baseline.md)
- [FlashTalk Resident Benchmark](flashtalk_resident_benchmark.md)
- [Support Status](../user_guide/models/support_status.md)
