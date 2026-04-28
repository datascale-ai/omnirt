# Talking Head (audio2video)

Given a face portrait plus an audio clip, produce an MP4 where lips, head motion, or long-form avatar animation are aligned to the speech. OmniRT supports this task via `soulx-flashtalk-14b`, `soulx-flashhead-1.3b`, and `soulx-liveact-14b`.

## Minimal example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import audio2video

    result = generate(audio2video(
        model="soulx-flashtalk-14b",
        image="inputs/portrait.png",
        audio="inputs/speech.wav",
        preset="balanced",
    ))
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task audio2video \
      --model soulx-flashtalk-14b \
      --image inputs/portrait.png \
      --audio inputs/speech.wav \
      --preset balanced
    ```

=== "HTTP"

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "audio2video",
        "model": "soulx-flashtalk-14b",
        "inputs": {
          "image": "inputs/portrait.png",
          "audio": "inputs/speech.wav"
        },
        "config": {"preset": "balanced"}
      }'
    ```

## Key parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | **required** | path to the face portrait |
| `audio` | `str` | **required** | audio path (prefer `.wav`; any ffmpeg-decodable format works) |
| `prompt` | `str?` | `None` | optional hint (expression / emotion) |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | see [Presets](../features/presets.md) |
| `fps` | `int?` | model default | output frame rate |
| `repo_path` | `str?` | config | external repo checkout path; can also come from `OMNIRT_FLASHTALK_REPO_PATH` / `OMNIRT_FLASHHEAD_REPO_PATH` or YAML config |
| `ckpt_dir` | `str?` | config | checkpoint directory, absolute or relative to `repo_path` |
| `wav2vec_dir` | `str?` | config | wav2vec checkpoint directory |
| `model_type` | `pro`/`lite` | `pro` | FlashHead model type |
| `sample_steps` | `int?` | `2` | FlashHead sample-step override, passed through `FLASHHEAD_SAMPLE_STEPS` |
| `vae_2d_split` | `bool` | `true` | FlashHead 910B quality-profile VAE split strategy |
| `latent_carry` | `bool` | `false` | FlashHead experimental fastest mode; may cause style drift |

## Supported models

| Model | Inputs | Output | VRAM |
|---|---|---|---|
| `soulx-flashtalk-14b` | portrait + audio | MP4 | ≥ 20 GB |
| `soulx-flashhead-1.3b` | portrait + audio | MP4 | ≥ 48 GB aggregate |
| `soulx-liveact-14b` | portrait + audio | MP4 | 4-card Ascend 910B recommended |

!!! info "SoulX avatar models are script-backed"
    `soulx-flashtalk-14b`, `soulx-flashhead-1.3b`, and `soulx-liveact-14b` require an external SoulX checkout, model checkpoint directory, wav2vec directory, and matching Python environment. For restricted networks see the "script-backed model mirrors" section of [Domestic Deployment](../deployment/china_mirrors.md).

## FlashHead Ascend Recommendation

`soulx-flashhead-1.3b` follows the 910B adaptation note and defaults to the quality-oriented profile:

```bash
omnirt generate \
  --task audio2video \
  --model soulx-flashhead-1.3b \
  --image inputs/portrait.png \
  --audio inputs/speech.wav \
  --backend ascend \
  --repo-path /path/to/SoulX-FlashHead \
  --ckpt-dir models/SoulX-FlashHead-1_3B \
  --wav2vec-dir models/wav2vec2-base-960h \
  --python-executable /path/to/venv/bin/python \
  --ascend-env-script /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --launcher torchrun \
  --nproc-per-node 4 \
  --visible-devices 2,3,4,5 \
  --sample-steps 2 \
  --vae-2d-split \
  --npu-fusion-attention
```

## LiveAct Ascend Recommendation

`soulx-liveact-14b` launches `generate.py` from an external SoulX-LiveAct checkout. The wrapper sets `PLATFORM=ascend_npu` by default, prepares text context with `prepare_text_cache.py` on a single NPU, then launches the 4-card inference job. With explicit placement, use `--text-cache-visible-devices 2 --visible-devices 2,3,4,5` for the 1-card T5 + 4-card inference split. Add `--sample-steps 1` for quick smoke tests. For LightVAE, pair `--vae-path models/vae/lightvaew2_1.pth --use-lightvae --use-cache-vae` and warm `--condition-cache-dir`.

## Troubleshooting

!!! warning

    - **Audio sample-rate mismatch** — SoulX avatar models typically expect 16 kHz mono. Other formats are resampled automatically, but long audio amplifies error.
    - **Poor portrait alignment** — frontal, upper-body, eyes-visible portraits give the best output.
    - **External repo clone fails** — behind the GFW, route through `GHPROXY` or supply an offline `repo_path`.
    - **FlashHead style drift** — keep `latent_carry=false` first; it is an experimental speed knob, not the default display profile.
    - **Ascend much slower than CUDA** — check `FLASHHEAD_NPU_FUSION_ATTENTION`, `visible_devices`, CANN env loading, and whether the external checkout includes the NPU adaptation patches.
    - **LiveAct reports CUDA device unavailable** — confirm `PLATFORM=ascend_npu`; the OmniRT wrapper sets it, but manual external runs often miss it.
    - **LiveAct T5 OOM** — prefer the default single-NPU text-context cache path; set `--text-cache-visible-devices` explicitly and avoid loading T5 inside the 4-card inference process.
