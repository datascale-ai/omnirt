# FlashTalk-compatible WebSocket

OmniRT can expose SoulX-FlashTalk through a FlashTalk-compatible WebSocket server. This is useful when an existing realtime avatar stack, such as OpenTalking, already speaks the `init` / `AUDI` / `VIDX` protocol and you want OmniRT to own the model service.

The service is intentionally configured by environment variables so machine-specific paths stay outside the repository. The walkthrough below assumes every script is run from your OmniRT checkout. The external FlashTalk checkout, virtual environment, and model weights can live in any directory and are passed in through environment variables.

## 910B Quick Start

First make sure no old service is already listening on port 8765 and that the eight 910B cards have enough free HBM:

```bash
ss -ltnp | grep ':8765' || true
pgrep -af 'flashtalk_server.py|torchrun|omnirt.*flashtalk' || true
npu-smi info
```

If a service is already listening on `0.0.0.0:8765`, check the connection instead of starting a second copy:

```bash
cd /path/to/omnirt
/path/to/flashtalk-venv/bin/python - <<'PY'
import asyncio
from websockets.asyncio.client import connect

async def main():
    async with connect('ws://127.0.0.1:8765', open_timeout=5, close_timeout=2):
        print('connected')

asyncio.run(main())
PY
```

If the port is free, start with the following minimal configuration. The Ascend/CANN environment script is required; without it, `torch_npu` can fail with `libhccl.so: cannot open shared object file`.

```bash
cd /path/to/omnirt

export OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk
export OMNIRT_FLASHTALK_CKPT_DIR=models/SoulX-FlashTalk-14B
export OMNIRT_FLASHTALK_WAV2VEC_DIR=models/chinese-wav2vec2-base
export OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh
export OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate
export OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python
export OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun
export OMNIRT_FLASHTALK_HOST=0.0.0.0
export OMNIRT_FLASHTALK_PORT=8765
export OMNIRT_FLASHTALK_NPROC_PER_NODE=8
export OMNIRT_FLASHTALK_CMD_DIR=$PWD/outputs/flashtalk-cmd
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29517

bash scripts/start_flashtalk_ws.sh
```

A successful startup prints `Pipeline loaded successfully` for every rank and `WebSocket server starting on 0.0.0.0:8765` from rank 0.

## Background Startup

For a long-running service, keep logs and the pid under the OmniRT checkout's `outputs/` directory:

```bash
cd /path/to/omnirt
mkdir -p outputs

nohup env \
  OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk \
  OMNIRT_FLASHTALK_CKPT_DIR=models/SoulX-FlashTalk-14B \
  OMNIRT_FLASHTALK_WAV2VEC_DIR=models/chinese-wav2vec2-base \
  OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh \
  OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate \
  OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python \
  OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun \
  OMNIRT_FLASHTALK_HOST=0.0.0.0 \
  OMNIRT_FLASHTALK_PORT=8765 \
  OMNIRT_FLASHTALK_NPROC_PER_NODE=8 \
  OMNIRT_FLASHTALK_CMD_DIR=$PWD/outputs/flashtalk-cmd \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT=29517 \
  bash scripts/start_flashtalk_ws.sh \
  > outputs/omnirt-flashtalk-ws.log 2>&1 &
echo $! > outputs/omnirt-flashtalk-ws.pid

tail -f outputs/omnirt-flashtalk-ws.log
```

To stop the service, kill the recorded torchrun parent process first:

```bash
kill "$(cat outputs/omnirt-flashtalk-ws.pid)"
```

If the pid file is missing, use `pgrep -af 'flashtalk_server.py|torchrun'` to find the matching process before stopping it manually.

## Realtime Parameters

The helper script preserves and forwards the upstream FlashTalk `FLASHTALK_*` environment variables. For a 910B realtime avatar path, this block is a good starting point:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

export FLASHTALK_HEIGHT=704
export FLASHTALK_WIDTH=416
export FLASHTALK_FRAME_NUM=29
export FLASHTALK_MOTION_FRAMES_NUM=1
export FLASHTALK_SAMPLE_STEPS=2
export FLASHTALK_COLOR_CORRECTION_STRENGTH=0
export FLASHTALK_AUDIO_LOUDNESS_NORM=0
export FLASHTALK_JPEG_QUALITY=55
export FLASHTALK_JPEG_WORKERS=4
export FLASHTALK_IDLE_CACHE_DIR=$PWD/outputs/idle_cache
export FLASHTALK_WARMUP=0
export FLASHTALK_WARMUP_ON_INIT=0
```

To warm up with a reference image, set `FLASHTALK_WARMUP=1` and `FLASHTALK_WARMUP_REF_IMAGE=/path/to/SoulX-FlashTalk/assets/flashtalk-demo-warmup.png`.

## Entrypoints

The helper script defaults to the lightweight entrypoint:

```bash
OMNIRT_FLASHTALK_ENTRYPOINT=lightweight bash scripts/start_flashtalk_ws.sh
```

This runs `src/omnirt/cli/flashtalk_ws.py`. It avoids importing the full OmniRT package and is best for vendor model environments that only have FlashTalk dependencies installed.

When the environment has OmniRT and its dependencies installed, you can use the formal CLI entrypoint:

```bash
OMNIRT_FLASHTALK_ENTRYPOINT=cli bash scripts/start_flashtalk_ws.sh
```

This is equivalent to:

```bash
omnirt serve \
  --protocol flashtalk-ws \
  --host 0.0.0.0 \
  --port 8765 \
  --repo-path /path/to/SoulX-FlashTalk \
  --ckpt-dir models/SoulX-FlashTalk-14B \
  --wav2vec-dir models/chinese-wav2vec2-base
```

On 910B, keep `OMNIRT_FLASHTALK_NPROC_PER_NODE=8`. A single-card startup can OOM while loading the T5/Wan weights.

## Optional Quantization Flags

The script forwards optional quantization settings to the upstream FlashTalk server:

```bash
export OMNIRT_FLASHTALK_T5_QUANT=int8
export OMNIRT_FLASHTALK_T5_QUANT_DIR=/path/to/t5-int8
export OMNIRT_FLASHTALK_WAN_QUANT=fp8
export OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE='blocks.*'
export OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE=head
```

## Connect OpenTalking

OpenTalking can keep using its FlashTalk remote mode while OmniRT provides the model service:

```bash
OPENTALKING_FLASHTALK_MODE=remote
OPENTALKING_FLASHTALK_WS_URL=ws://omnirt-host:8765
```

No OpenTalking code changes are required for this compatibility path.

## Troubleshooting

`ImportError: libhccl.so: cannot open shared object file` means the Ascend/CANN environment was not loaded. Set `OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh`.

`NPU out of memory` usually means another service is already using HBM, or the service was started with `OMNIRT_FLASHTALK_NPROC_PER_NODE=1`. Check `npu-smi info`, `pgrep -af 'flashtalk_server.py|torchrun'`, and `ss -ltnp | grep ':8765'`.

`Address already in use` means a service is already listening on 8765. Run the connection check above first; stop the old service only when you intentionally want to restart it.

`Wav2Vec2Model LOAD REPORT` with `UNEXPECTED` keys can appear with the current FlashTalk wav2vec loading path. If every rank later prints `Pipeline loaded successfully` and the connection check passes, continue with OpenTalking integration.
