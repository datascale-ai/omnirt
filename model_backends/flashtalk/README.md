# FlashTalk Backend

This backend contains OmniRT's FlashTalk-compatible WebSocket assets.

OmniRT provides the WebSocket entrypoint and launcher. The external SoulX-FlashTalk checkout still provides the `flash_talk` Python package, model code, assets, and checkpoint layout.

## Prepare CUDA Runtime

The CUDA runtime follows the official SoulX-FlashTalk GPU setup: PyTorch
`2.7.1` from the CUDA 12.8 wheel index plus `flash_attn==2.8.0.post2` installed
with `--no-build-isolation`.

```bash
python -m omnirt.cli.main runtime install flashtalk --device cuda \
  --ckpt-dir .omnirt/model-repos/SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
  --wav2vec-dir .omnirt/model-repos/SoulX-FlashTalk/models/chinese-wav2vec2-base \
  --no-update \
  --recreate-venv
```

The Ascend compatibility patch is **not** applied for `--device cuda`; the GPU
runtime should stay close to the official upstream CUDA code path.

## Prepare Ascend 910B Runtime

From the OmniRT repository root (paths relative to that root):

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend \
  --ckpt-dir .omnirt/model-repos/SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
  --wav2vec-dir .omnirt/model-repos/SoulX-FlashTalk/models/chinese-wav2vec2-base \
  --no-update \
  --recreate-venv
```

The command creates a dedicated FlashTalk model environment, installs Ascend runtime dependencies, clones or updates the model checkout, skips existing checkpoint directories, and writes runtime state under `.omnirt/runtimes/flashtalk/ascend/state.yaml` in the current OmniRT checkout.

By default runtime artifacts live under `.omnirt/`. Use `--home ./runtime-data` or `OMNIRT_HOME=./runtime-data` to choose another location (relative to the current shell directory when you invoke the CLI). Use `--repo-dir`, `--ckpt-dir`, and `--wav2vec-dir` to point at existing directories.

```bash
bash scripts/start_flashtalk_ws.sh
```

`prepare_ascend_910b.sh` remains as a compatibility wrapper around the runtime installer.

## Key Files

- `flashtalk_ws_server.py`: OmniRT's FlashTalk-compatible WebSocket server entrypoint.
- `requirements-gpu.txt`: CUDA dependencies for the official SoulX-FlashTalk GPU path.
- `requirements-ascend.txt`: Ascend 910B Python dependencies for the FlashTalk model environment.
- `runtime.cuda.yaml`: OmniRT-managed CUDA runtime manifest.
- `prepare_ascend_910b.sh`: Compatibility wrapper around `omnirt runtime install`.
- `prepare_cuda.sh`: Compatibility wrapper around `omnirt runtime install --device cuda`.

Do not place virtual environments, model checkpoints, or full upstream repository copies in this directory. Runtime artifacts belong in `.omnirt/` or a user-selected `--home` directory.
