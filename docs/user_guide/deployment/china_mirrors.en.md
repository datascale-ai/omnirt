# China Deployment Guide

This guide documents the recommended OmniRT workflow for environments where public model downloads may be slow, blocked, or unreliable.

The key principle is simple:

`OmniRT production and smoke tests should use local model directories, not runtime downloads.`

## Why this matters

In many mainland China environments:

- `huggingface.co` may resolve in DNS but still time out on HTTPS
- large model downloads are unstable or extremely slow
- GPU and NPU servers often have weaker outbound internet access than development laptops
- temporary runtime downloads make CI and hardware smoke tests flaky

Because of that, `model_path` should be treated as the primary deployment mode.

## Recommended deployment pattern

1. Download model snapshots on a machine with stable external access.
2. Store those snapshots in a shared internal location.
3. Sync them to GPU or Ascend servers ahead of time.
4. Point OmniRT to local directories through `config.model_path` or environment variables.

Recommended directory layout:

```text
/data/models/omnirt/
  sdxl-base-1.0/
  svd/
  svd-xt/
  flux2-dev/
  wan2.2-t2v-14b/
  wan2.2-i2v-14b/
  qwen-image/
```

## Integration test policy

OmniRT integration smoke tests intentionally require local model directories.

- `OMNIRT_SDXL_MODEL_SOURCE` must point to an existing local directory
- `OMNIRT_SDXL_REFINER_MODEL_SOURCE` must point to the SDXL refiner local directory
- `OMNIRT_SVD_MODEL_SOURCE` must point to the base SVD local directory
- `OMNIRT_SVD_XT_MODEL_SOURCE` must point to the SVD-XT local directory
- `OMNIRT_FLUX_FILL_MODEL_SOURCE` must point to the Flux Fill local directory
- `OMNIRT_FLUX_DEPTH_MODEL_SOURCE` must point to the Flux Depth local directory
- `OMNIRT_FLUX_CANNY_MODEL_SOURCE` must point to the Flux Canny local directory
- `OMNIRT_FLUX_KONTEXT_MODEL_SOURCE` must point to the Flux Kontext local directory
- `OMNIRT_QWEN_IMAGE_EDIT_MODEL_SOURCE` must point to the Qwen Image Edit local directory
- `OMNIRT_QWEN_IMAGE_EDIT_PLUS_MODEL_SOURCE` must point to the Qwen Image Edit Plus local directory
- `OMNIRT_QWEN_IMAGE_LAYERED_MODEL_SOURCE` must point to the Qwen Image Layered local directory
- `OMNIRT_CHRONOEDIT_MODEL_SOURCE` must point to the ChronoEdit local directory
- `OMNIRT_KOLORS_MODEL_SOURCE` must point to the Kolors local directory
- `OMNIRT_PIXART_SIGMA_MODEL_SOURCE` must point to the PixArt Sigma local directory
- `OMNIRT_BRIA_3_2_MODEL_SOURCE` must point to the Bria 3.2 local directory
- `OMNIRT_LUMINA_T2X_MODEL_SOURCE` must point to the Lumina-T2X local directory
- `OMNIRT_ANIMATEDIFF_SDXL_MODEL_SOURCE` can optionally point to the base SDXL local directory used by AnimateDiff SDXL
- `OMNIRT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE` must point to the AnimateDiff SDXL motion adapter local directory
- `OMNIRT_MOCHI_MODEL_SOURCE` must point to the Mochi local directory
- `OMNIRT_SKYREELS_V2_MODEL_SOURCE` must point to the SkyReels V2 local directory
- if the directory is missing, the test skips
- the tests do not treat a Hugging Face repo id as a valid smoke-test source

This keeps hardware validation deterministic in restricted networks.

## Preparing local snapshots

On a machine with outbound access, use:

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir /data/models/omnirt/sdxl-base-1.0
```

Example for SVD:

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-video-diffusion-img2vid \
  --output-dir /data/models/omnirt/svd
```

Example for SVD-XT:

```bash
python scripts/prepare_modelscope_snapshot.py \
  --repo-id ai-modelscope/stable-video-diffusion-img2vid-xt \
  --output-dir /data/models/omnirt/svd-xt \
  --download-file image_encoder/model.fp16.safetensors \
  --download-file unet/diffusion_pytorch_model.fp16.safetensors \
  --download-file vae/diffusion_pytorch_model.fp16.safetensors
```

If a mirror is available, set the endpoint before running:

```bash
export HF_ENDPOINT=https://<your-mirror-host>
```

If you prefer downloading directly from Modelers, clone the repository into a local directory:

```bash
python scripts/prepare_modelers_snapshot.py \
  --repo-id MindSpore-Lab/SDXL_Base1_0 \
  --output-dir /data/models/omnirt/SDXL_Base1_0
```

For verified ModelScope downloads, use:

```bash
python scripts/prepare_modelscope_snapshot.py \
  --repo-id ai-modelscope/stable-video-diffusion-img2vid-xt \
  --output-dir /data/models/omnirt/svd-xt \
  --download-file image_encoder/model.fp16.safetensors \
  --download-file unet/diffusion_pytorch_model.fp16.safetensors \
  --download-file vae/diffusion_pytorch_model.fp16.safetensors
```

This follows Modelers' documented Git download flow:

`git clone https://modelers.cn/<username>/<model_name>.git`

After downloading, validate the layout before syncing:

```bash
python scripts/check_model_layout.py \
  --task sdxl \
  --model-dir /data/models/omnirt/sdxl-base-1.0
```

For SVD:

```bash
python scripts/check_model_layout.py \
  --task svd \
  --model-dir /data/models/omnirt/svd-xt
```

This is especially useful for non-Hugging-Face sources such as mirrored archives or Modelers downloads.

## Verified domestic sources

As of `2026-04-20`, these sources have been validated with real hardware smoke tests:

- `SDXL`
  Source: `modelers.cn` repo `MindSpore-Lab/SDXL_Base1_0`
  Verified local layouts on:
  `/data/models/omnirt/SDXL_Base1_0`
  `$HOME/models/omnirt/SDXL_Base1_0`
- `SVD-XT`
  Source: ModelScope repo `ai-modelscope/stable-video-diffusion-img2vid-xt`
  Verified local layouts on:
  `/data/models/omnirt/svd-xt-ms`
  `$HOME/models/omnirt/svd-xt-ms`

Important note for `SVD`:

- `modelers.cn` repo `MindSpore-Lab/svd` is a `ckpt` repository, not a Diffusers directory.
- It is not directly compatible with the current OmniRT `SVDPipeline`.
- For current OmniRT validation, prefer the ModelScope `SVD-XT` source above.

## Verified smoke commands

CUDA validation host:

```bash
export OMNIRT_SDXL_MODEL_SOURCE=/data/models/omnirt/SDXL_Base1_0
export OMNIRT_SVD_XT_MODEL_SOURCE=/data/models/omnirt/svd-xt-ms
export OMNIRT_DISABLE_COMPILE=1

$VENV_PYTHON -m pytest tests/integration/test_sdxl_cuda.py -q
$VENV_PYTHON -m pytest tests/integration/test_svd_cuda.py -q
```

Ascend validation host:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}
export OMNIRT_SDXL_MODEL_SOURCE=$HOME/models/omnirt/SDXL_Base1_0
export OMNIRT_SVD_XT_MODEL_SOURCE=$HOME/models/omnirt/svd-xt-ms

$VENV_PYTHON -m pytest tests/integration/test_sdxl_ascend.py -q
$VENV_PYTHON -m pytest tests/integration/test_svd_ascend.py -q
```

## Syncing to target servers

Use the provided sync helper:

```bash
bash scripts/sync_model_dir.sh \
  /data/models/omnirt/sdxl-base-1.0 \
  user@<cuda-host>:/data/models/omnirt/sdxl-base-1.0
```

You can do the same for Ascend:

```bash
bash scripts/sync_model_dir.sh \
  /data/models/omnirt/svd-xt \
  user@<ascend-host>:$HOME/models/omnirt/svd-xt
```

## Running OmniRT with local models

CLI example:

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic sci-fi city at sunrise" \
  --backend cuda \
  --model-path /data/models/omnirt/sdxl-base-1.0
```

Environment-variable example for smoke tests:

```bash
export OMNIRT_SDXL_MODEL_SOURCE=/data/models/omnirt/sdxl-base-1.0
export OMNIRT_SDXL_REFINER_MODEL_SOURCE=/data/models/omnirt/sdxl-refiner-1.0
export OMNIRT_SVD_MODEL_SOURCE=/data/models/omnirt/svd
export OMNIRT_SVD_XT_MODEL_SOURCE=/data/models/omnirt/svd-xt
export OMNIRT_FLUX_FILL_MODEL_SOURCE=/data/models/omnirt/flux-fill
export OMNIRT_FLUX_DEPTH_MODEL_SOURCE=/data/models/omnirt/flux-depth
export OMNIRT_FLUX_CANNY_MODEL_SOURCE=/data/models/omnirt/flux-canny
export OMNIRT_FLUX_KONTEXT_MODEL_SOURCE=/data/models/omnirt/flux-kontext
export OMNIRT_QWEN_IMAGE_EDIT_MODEL_SOURCE=/data/models/omnirt/qwen-image-edit
export OMNIRT_QWEN_IMAGE_EDIT_PLUS_MODEL_SOURCE=/data/models/omnirt/qwen-image-edit-plus
export OMNIRT_QWEN_IMAGE_LAYERED_MODEL_SOURCE=/data/models/omnirt/qwen-image-layered
export OMNIRT_CHRONOEDIT_MODEL_SOURCE=/data/models/omnirt/chronoedit
export OMNIRT_KOLORS_MODEL_SOURCE=/data/models/omnirt/kolors
export OMNIRT_PIXART_SIGMA_MODEL_SOURCE=/data/models/omnirt/pixart-sigma
export OMNIRT_BRIA_3_2_MODEL_SOURCE=/data/models/omnirt/bria-3.2
export OMNIRT_LUMINA_T2X_MODEL_SOURCE=/data/models/omnirt/lumina-t2x
export OMNIRT_ANIMATEDIFF_SDXL_MODEL_SOURCE=/data/models/omnirt/sdxl-base-1.0
export OMNIRT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE=/data/models/omnirt/animatediff-sdxl-motion-adapter
export OMNIRT_MOCHI_MODEL_SOURCE=/data/models/omnirt/mochi
export OMNIRT_SKYREELS_V2_MODEL_SOURCE=/data/models/omnirt/skyreels-v2
pytest \
  tests/integration/test_sdxl_cuda.py \
  tests/integration/test_sdxl_refiner_cuda.py \
  tests/integration/test_svd_base_cuda.py \
  tests/integration/test_svd_cuda.py \
  tests/integration/test_flux_fill_cuda.py \
  tests/integration/test_image_edit_cuda.py \
  tests/integration/test_generalist_text2image_cuda.py \
  tests/integration/test_structured_edit_cuda.py \
  tests/integration/test_extended_video_cuda.py
```

## Python dependency strategy

For restricted networks, prefer:

- internal PyPI mirrors
- prebuilt wheel caches
- environment reuse on target servers

Typical example:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e '.[runtime,dev]'
```

For Ascend servers, avoid large ad-hoc environment rebuilds when a working `torch_npu` environment already exists.

## Server-specific guidance

### CUDA hosts

- preinstall `torch`, `diffusers`, `transformers`, `safetensors`, `accelerate`
- validate with local model directories only
- avoid relying on runtime Hugging Face downloads

### Ascend hosts

- source `set_env.sh` before running OmniRT
- reuse the known-good `torch_npu` virtualenv when possible
- install only the minimal missing packages if the environment is otherwise healthy

## What to avoid

- using Hugging Face repo ids directly in hardware smoke tests
- assuming public internet access from GPU or NPU servers
- downloading model weights during CI
- rebuilding full Python environments on every verification run

## Suggested operational workflow

1. Maintain a connected machine for model snapshot preparation.
2. Mirror or snapshot approved model versions into a local model store.
3. Sync model directories into each hardware environment.
4. Run OmniRT smoke tests against those local directories.
5. Only update model snapshots intentionally, not implicitly during runtime.
