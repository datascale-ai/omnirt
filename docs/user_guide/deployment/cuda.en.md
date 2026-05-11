# CUDA Deployment (NVIDIA GPU)

OmniRT's public baseline on NVIDIA GPUs: single-card inference on Ampere and newer.

## Hardware requirements

| Item | Requirement | Notes |
|---|---|---|
| GPU | Ampere+ | A100 / L40S / RTX 3090 / 4090; `torch.compile` is only stable on Ampere+ |
| VRAM | per-model `resource_hint.min_vram_gb` | check exact values with `omnirt models <id>`: SDXL ≥ 12 GB, SVD ≥ 14 GB, Flux2 / Wan2.2 ≥ 24 GB; SD1.5 and similar experimental models are compatibility references |
| Driver | ≥ 535 (pairs with CUDA 12.1) | verify with `nvidia-smi` |
| CUDA Toolkit | 12.1 or 12.4 | must match the PyTorch wheel |
| PyTorch | 2.1+ official CUDA wheel | e.g. `torch==2.5.1+cu121` |

## Install

=== "pip"

    ```bash
    # 1. Install the matching CUDA PyTorch wheel (pick the index on pytorch.org)
    python -m pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121

    # 2. Install OmniRT plus runtime extras
    python -m pip install -e '.[runtime,dev]'
    ```

=== "uv"

    ```bash
    uv pip install torch==2.5.1 torchvision==0.20.1 \
      --index https://download.pytorch.org/whl/cu121
    uv pip install -e '.[runtime,dev]'
    ```

=== "From source"

    ```bash
    git clone https://github.com/datascale-ai/omnirt.git
    cd omnirt
    python -m pip install -e '.[runtime,dev]'
    ```

## Smoke test

```bash
# Confirm CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Dry-run validate the request contract against cpu-stub
omnirt validate --task text2image --model sdxl-base-1.0 --prompt "a lighthouse" --backend cpu-stub

# Run a real generation
omnirt generate --task text2image --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" --backend cuda --preset fast
```

## Production tuning

- **`torch.compile`**: on by default, stable on Ampere+. If compilation fails, set `OMNIRT_DISABLE_COMPILE=1` to skip — failures are recorded in `RunReport.backend_timeline`.
- **Device visibility**: `CUDA_VISIBLE_DEVICES=0` (single card) or `0,1` (multi-card — but note: **multi-GPU parallelism, USP, and CFG sharding are not yet public features**, see [PLAN.md](https://github.com/datascale-ai/omnirt/blob/main/PLAN.md)).
- **VRAM peak**: inspect `RunReport.memory`; on OOM, switch to `--preset low-vram` or drop `width/height` / `num_frames`.
- **Telemetry**: `omnirt.middleware.telemetry` emits stage timings, peak memory, and fallback events — see [Telemetry](../features/telemetry.md).
- **Serving**: for FastAPI deployment see [HTTP Server](../serving/http_server.md); production defaults should use `--model-tier core --model-tier adjacent`.

## Known issues

!!! warning

    - **`torch.compile` crashes on older cards** — `OMNIRT_DISABLE_COMPILE=1` falls back to eager; each fallback is tracked
    - **`flashinfer` / custom attention kernel miss** — failed kernel overrides automatically fall back to eager attention; check `kernel_override` entries in `RunReport.backend_timeline`
    - **Slow Triton compile with older versions** — upgrade to the `triton` version PyTorch recommends
