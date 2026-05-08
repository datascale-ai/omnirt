# Wav2Lip WebSocket (FlashTalk-compatible)

This backend exposes **the same WebSocket protocol** as SoulX FlashTalk (`AUDI` / `VIDX`, JSON `init` / `init_ok`), so **OpenTalking** can keep:

- `OPENTALKING_DEFAULT_MODEL=flashtalk`
- `OPENTALKING_FLASHTALK_MODE=remote`
- `OPENTALKING_FLASHTALK_WS_URL=ws://<host>:<port>`

while the server runs **Wav2Lip** lip-sync instead of SoulX.

**Deployment walkthroughs (Ascend + NVIDIA GPU, end-to-end):** [`docs/user_guide/serving/wav2lip_ws.md`](../../docs/user_guide/serving/wav2lip_ws.md) (简体中文).

## Layout

| Path | Purpose |
|------|---------|
| [`wav2lip_ws_server.py`](wav2lip_ws_server.py) | Protocol-compatible WS server |
| [`../../scripts/start_wav2lip_ws.sh`](../../scripts/start_wav2lip_ws.sh) | Launcher (no SoulX repo required) |
| [`requirements-wav2lip.txt`](requirements-wav2lip.txt) | Pip deps for CUDA or CPU-oriented installs (see user guide) |
| [`requirements-wav2lip-ascend.txt`](requirements-wav2lip-ascend.txt) | Ascend 910 / `torch_npu` + CANN-matched wheels |

## Prerequisites

1. **Rudrabha/Wav2Lip** clone (default: `<omnirt>/models/repos/Wav2Lip`), or set `OMNIRT_WAV2LIP_REPO`.
2. **Weights** (default: `<omnirt>/models/wav2lip/wav2lip_gan.pth`), or set `OMNIRT_WAV2LIP_CHECKPOINT`.
3. **Python 3.9+** venv created **on the machine that runs the server** — do not copy `.venv` trees from other hosts (broken `home` / prefix paths).
4. Install backend deps from the file that matches your hardware:
   - **NVIDIA GPU (typical):** `pip install -r model_backends/wav2lip/requirements-wav2lip.txt`
   - **Ascend NPU:** configure Huawei wheel indices, then `pip install -r model_backends/wav2lip/requirements-wav2lip-ascend.txt` (align torch/torch-npu with [`requirements-ascend.txt`](../flashtalk/requirements-ascend.txt)).

Do **not** follow Rudrabha’s pinned `torch==1.1.0` `requirements.txt`. This repo patches **`models/repos/Wav2Lip/audio.py`** for **librosa≥0.10**. If imports still fail, install additional packages from the upstream Wav2Lip `requirements.txt` (face-alignment, etc.) into the same env.

Set **`OMNIRT_WAV2LIP_PYTHON`** to your venv’s `python` when calling `start_wav2lip_ws.sh`.

## Protocol defaults

Controlled via environment so OpenTalking’s chunk sizing stays consistent:

| Variable | Default | Notes |
|----------|---------|--------|
| `OMNIRT_WAV2LIP_FRAME_NUM` | 33 | |
| `OMNIRT_WAV2LIP_MOTION_FRAMES_NUM` | 8 | `slice_len = 33 - 8 = 25` |
| `OMNIRT_WAV2LIP_FPS` | 25 | |
| Audio chunk | `slice_len * 16000 // fps` | **16000** int16 samples per chunk |

The server returns these fields in `init_ok`; the client derives chunk sample counts from `slice_len`, `fps`, and 16 kHz sampling — matching OpenTalking’s `FlashTalkWSClient` behaviour.

## Launch

From the OmniRT repo root:

```bash
export OMNIRT_WAV2LIP_PYTHON=/path/to/venv/bin/python   # recommended
export OMNIRT_WAV2LIP_REPO=/path/to/Wav2Lip              # optional
export OMNIRT_WAV2LIP_CHECKPOINT=/path/to/wav2lip_gan.pth  # optional
bash scripts/start_wav2lip_ws.sh
```

Background:

```bash
OMNIRT_WAV2LIP_BACKGROUND=1 bash scripts/start_wav2lip_ws.sh --background
```

## Relation to `start_flashtalk_ws.sh`

[`scripts/start_flashtalk_ws.sh`](../../scripts/start_flashtalk_ws.sh) assumes a **SoulX-FlashTalk** checkout (`flash_talk/` package, checkpoints, wav2vec).  
Wav2Lip-only deployments should use **`start_wav2lip_ws.sh`** instead.

## OpenTalking

Same remote FlashTalk settings as OmniRT + SoulX; only the WS URL changes to your Wav2Lip host.

## Smoke test

With the server running:

```bash
python3 model_backends/wav2lip/smoke_ws_client.py --url ws://127.0.0.1:8765
```

## Troubleshooting

- **`Face not detected`**: adjust `OMNIRT_WAV2LIP_PADS` (default `0 10 0 0`) or use a clearer face crop in the avatar reference image.
- **PCM length errors**: do not change `slice_len` / `fps` on only one side; OpenTalking and this server must agree (via `init_ok`).
- **Broken venv after copy**: recreate with `python3 -m venv` on the target machine — see [`wav2lip_ws.md`](../../docs/user_guide/serving/wav2lip_ws.md).
