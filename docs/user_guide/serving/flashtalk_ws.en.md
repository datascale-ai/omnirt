# FlashTalk-compatible WebSocket

OmniRT can expose SoulX-FlashTalk through a FlashTalk-compatible WebSocket server. This is useful when an existing realtime avatar stack, such as OpenTalking, already speaks the `init` / `AUDI` / `VIDX` protocol and you want OmniRT to own the model service.

The service is intentionally configured by environment variables so machine-specific paths stay outside the repository.

## Start with the helper script

Set the external FlashTalk checkout and runtime paths, then run the helper script:

```bash
cd /path/to/omnirt

export OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk
export OMNIRT_FLASHTALK_CKPT_DIR=models/SoulX-FlashTalk-14B
export OMNIRT_FLASHTALK_WAV2VEC_DIR=models/chinese-wav2vec2-base
export OMNIRT_FLASHTALK_HOST=0.0.0.0
export OMNIRT_FLASHTALK_PORT=8765
export OMNIRT_FLASHTALK_NPROC_PER_NODE=8
export OMNIRT_FLASHTALK_CMD_DIR=$PWD/outputs/flashtalk-cmd

bash scripts/start_flashtalk_ws.sh
```

If the FlashTalk runtime lives in a dedicated virtual environment, point the script at that environment's launcher. On Ascend/CANN hosts, set OMNIRT_FLASHTALK_ENV_SCRIPT as well:

```bash
export OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun
export OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python
export OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate
export OMNIRT_FLASHTALK_ENV_SCRIPT=/usr/local/Ascend/ascend-toolkit/set_env.sh
```

`OMNIRT_FLASHTALK_CMD_DIR` is important on shared machines. It keeps FlashTalk's command files in a directory owned by the serving user instead of a shared `/tmp` path.

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

For multi-card serving, wrap the same entrypoint with `torchrun` or keep `OMNIRT_FLASHTALK_NPROC_PER_NODE` above `1`.

## Optional quantization flags

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

## Check the service

From the serving machine:

```bash
python - <<'PY'
import asyncio
from websockets.asyncio.client import connect

async def main():
    async with connect('ws://127.0.0.1:8765', open_timeout=5, close_timeout=2):
        print('connected')

asyncio.run(main())
PY
```

This only checks the WebSocket listener. A full avatar session still requires a client to send the FlashTalk-compatible `init` and audio messages.
