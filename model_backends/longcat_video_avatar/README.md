# LongCat-Video-Avatar 1.5 Backend

This directory documents the external runtime expected by OmniRT's
`longcat-video-avatar-1.5` adapter. The adapter supports both an Ascend profile
and a CUDA/GPU profile; Ascend remains the default backend in the model
registry.

OmniRT does not vendor the upstream repository, virtual environment, or
checkpoints. Keep those assets outside the source tree and point OmniRT at them
with environment variables or `configs/longcat_video_avatar.yaml`.

## Boundary

- OmniRT owns request validation, model registry metadata, input JSON creation,
  subprocess launch, response parsing, and artifact reporting.
- The external LongCat checkout owns model code, CUDA or CANN dependencies,
  attention kernels, LoRA merge behavior, and checkpoint loading.
- The worker entrypoint should read OmniRT's request JSONL, write response JSONL,
  and save the final MP4 to the `save_mp4` path supplied by OmniRT.

## Expected Layout

```text
/opt/model-repos/LongCat-Video/
  run_ascend_avatar_cp_worker.py
  weights/
    LongCat-Video-Avatar-1.5/
    LongCat-Video/
```

Configure OmniRT:

```bash
export OMNIRT_LONGCAT_AVATAR_REPO_PATH=/opt/model-repos/LongCat-Video
export OMNIRT_LONGCAT_AVATAR_CKPT_DIR=weights/LongCat-Video-Avatar-1.5
export OMNIRT_LONGCAT_AVATAR_BASE_CKPT_DIR=weights/LongCat-Video
export OMNIRT_LONGCAT_AVATAR_PYTHON=/opt/omnirt-runtimes/longcat-avatar/bin/python
export OMNIRT_LONGCAT_AVATAR_ASCEND_ENV_SCRIPT=/usr/local/Ascend/ascend-toolkit/set_env.sh
```

Equivalent YAML can be placed in `configs/longcat_video_avatar.yaml` or
`~/.omnirt/longcat_video_avatar.yaml`.

## Domestic Environment Notes

- Prefer transferring already verified local checkpoints into the target machine
  instead of downloading during startup.
- If downloading is unavoidable, use a domestic PyPI mirror, a local wheelhouse,
  and a Hugging Face mirror such as `HF_ENDPOINT=https://hf-mirror.com`.
- Use a fresh virtual environment matched to the accelerator stack. Ascend needs
  CANN plus `torch-npu`; CUDA needs a matching PyTorch CUDA wheel and should not
  install `torch-npu`.
- Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `DIFFUSERS_OFFLINE=1`
  enabled for production runs once all weights are present.

## OmniRT Invocation

Single-person input:

```bash
omnirt generate \
  --task audio2video \
  --model longcat-video-avatar-1.5 \
  --backend ascend \
  --image speaker.png \
  --audio voice.wav \
  --config '{"visible_devices":"0,1,2,3,4,5,6,7","nproc_per_node":8}'
```

CUDA/GPU input:

```bash
omnirt generate \
  --task audio2video \
  --model longcat-video-avatar-1.5 \
  --backend cuda \
  --image speaker.png \
  --audio voice.wav \
  --config '{"visible_devices":"0,1,2,3","nproc_per_node":4,"attention_profile":"cuda_flash_attn"}'
```

Multi-person input should use a LongCat-compatible JSON file:

```bash
omnirt generate \
  --task audio2video \
  --model longcat-video-avatar-1.5 \
  --backend ascend \
  --config '{"input_json":"/opt/inputs/multi_person.json","visible_devices":"0,1,2,3,4,5,6,7"}'
```

For multi-person audio, check the upstream `audio_type` semantics. `para` means
parallel speaker audio; `add` means sequential composition.

## Performance Profiles

The default Ascend profile is the formal quality-safe path:

```json
{
  "attention_profile": "formal",
  "attention_backend": "npu_fusion",
  "cp_split_hw": "4,2",
  "cache_profile": "faster",
  "save_profile": "copy_mux",
  "frames": 249,
  "steps": 8,
  "resolution": "480p"
}
```

This sets:

- `LONGCAT_ATTENTION_BACKEND=npu_fusion`
- `LONGCAT_CP_SPLIT_HW=4,2`
- `AVATAR_CACHE_PROFILE=faster`
- `LONGCAT_AVATAR_STREAM_VAE_SAVE=1`
- `LONGCAT_AVATAR_STREAM_VAE_ASYNC_WRITER=1`
- `LONGCAT_AVATAR_STREAM_VAE_ASYNC_CPU_TENSOR=1`
- `LONGCAT_AVATAR_SAVE_MODE=copy_mux`
- `HCCL_BUFFSIZE=512`

CANN 9.1 and BSA should be treated as explicit preview profiles until quality is
accepted for the target scenario:

```json
{"attention_profile":"preview_bsa128"}
```

or:

```json
{"attention_profile":"preview_bsa256"}
```

These profiles set self-attention to `ascend_bsa` and keep cross-attention on
`npu_fusion`.

The CUDA profile is selected with `--backend cuda` or `{"accelerator":"cuda"}`:

```json
{
  "attention_profile": "cuda_flash_attn",
  "attention_backend": "flash_attn",
  "visible_devices": "0,1,2,3",
  "nproc_per_node": 4
}
```

CUDA runs set:

- `CUDA_VISIBLE_DEVICES=<visible_devices>`
- `LONGCAT_DEVICE_BACKEND=cuda`
- `LONGCAT_DIST_BACKEND=nccl`
- `LONGCAT_ATTENTION_BACKEND=flash_attn` by default

Use `{"attention_profile":"cuda_sdpa"}` when the CUDA checkout should avoid
FlashAttention and rely on SDPA instead.

## Worker Protocol

OmniRT writes a JSONL request file:

```json
{"id":"omnirt_301_0000000000000","input_json":"/opt/inputs/request.json","frames":249,"steps":8,"fps":25,"seed":301,"output_type":"latent","cache_static_inputs":true,"precache_static_inputs":true,"save_mp4":"/opt/outputs/out.mp4","save_mode":"copy_mux"}
{"id":"shutdown","shutdown":true}
```

The worker should write JSONL response events. OmniRT accepts any final event
containing one of these output fields:

```json
{"id":"omnirt_301_0000000000000","status":"done","save":{"path":"/opt/outputs/out.mp4"}}
```

or:

```json
{"status":"done","save_file":"/opt/outputs/out.mp4"}
```

Any event with `error` or `status: "error"` is treated as a failed run.

## Patch Requirements

The upstream LongCat-Video project is CUDA-oriented. A CUDA checkout used by
this adapter should include:

- A worker script such as `run_cuda_avatar_worker.py`.
- NCCL distributed launch support.
- `LONGCAT_ATTENTION_BACKEND=flash_attn` or `sdpa`.
- LoRA merge/fuse support before the hot path.

An Ascend checkout used by this adapter should additionally include:

- `torch-npu`/HCCL distributed launch support.
- `LONGCAT_ATTENTION_BACKEND=npu_fusion` for the dense formal path.
- Optional `LONGCAT_ATTENTION_BACKEND=ascend_bsa` for preview BSA self-attention.
- `LONGCAT_CROSS_ATTENTION_BACKEND=npu_fusion` when BSA is enabled for
  self-attention.
- LoRA merge/fuse support before the hot path, controlled by
  `LONGCAT_MERGE_LORA=1` or the worker's `--merge-lora` flag.
- Explicit fp32 casts in modulation/final-layer time embeddings instead of hard
  dtype assertions.
- Streamed VAE save and copy-mux export support for lower end-to-end latency.

See `patches/README.md` for patch-level guidance.
