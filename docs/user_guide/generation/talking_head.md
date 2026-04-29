# 数字人（audio2video / talking head）

给一张人脸 portrait + 一段音频，生成口型、头部动作或长时数字人动画对齐的 MP4。OmniRT 通过 `soulx-flashtalk-14b`、`soulx-flashhead-1.3b` 和 `soulx-liveact-14b` 支持这一任务面。

## 最小示例

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

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | `str` | **必填** | 人脸 portrait 路径 |
| `audio` | `str` | **必填** | 音频路径（推荐 `.wav`；支持 ffmpeg 能解的格式） |
| `prompt` | `str?` | `None` | 可选提示（表情 / 情绪） |
| `preset` | `fast`/`balanced`/`quality`/`low-vram` | `balanced` | 见 [预设](../features/presets.md) |
| `fps` | `int?` | 模型默认 | 输出帧率 |
| `repo_path` | `str?` | 配置项 | 外部仓库 checkout 路径；也可通过 `OMNIRT_FLASHTALK_REPO_PATH` / `OMNIRT_FLASHHEAD_REPO_PATH` 或 YAML 配置提供 |
| `ckpt_dir` | `str?` | 配置项 | 权重目录，支持相对 `repo_path` 的路径 |
| `wav2vec_dir` | `str?` | 配置项 | wav2vec 权重目录 |
| `model_type` | `pro`/`lite` | `pro` | FlashHead 模型类型 |
| `sample_steps` | `int?` | `2` | FlashHead 采样步数覆盖，对应 `FLASHHEAD_SAMPLE_STEPS` |
| `vae_2d_split` | `bool` | `true` | FlashHead 910B 质量档默认 VAE split 策略 |
| `latent_carry` | `bool` | `false` | FlashHead 实验极速模式；可能引入风格漂移 |

## 支持模型

| 模型 | 输入 | 输出 | 显存 |
|---|---|---|---|
| `soulx-flashtalk-14b` | portrait + audio | MP4 | ≥ 20 GB |
| `soulx-flashhead-1.3b` | portrait + audio | MP4 | ≥ 48 GB aggregate |
| `soulx-liveact-14b` | portrait + audio | MP4 | 4 卡 Ascend 910B 推荐 |

!!! info "SoulX 数字人模型是 script-backed 模型"
    当前 `soulx-flashtalk-14b`、`soulx-flashhead-1.3b` 与 `soulx-liveact-14b` 都需要外部 SoulX 仓库 checkout、模型权重目录、wav2vec 目录和对应 Python 环境。内网/离线环境请参考 [国内部署](../deployment/china_mirrors.md) 的 "script-backed 模型镜像" 小节。

## 实时 WebSocket 接入

上面的 `generate` / HTTP 示例适合离线生成 MP4。如果要把 OmniRT 作为实时数字人模型服务接到 [OpenTalking](https://github.com/zyairehhh/opentalking) 等前端，需要先准备外部 SoulX-FlashTalk checkout、模型权重、wav2vec 权重、FlashTalk Python 环境和 Ascend/CANN 环境脚本，再启动 FlashTalk 兼容 WebSocket：

```bash
bash scripts/start_flashtalk_ws.sh
```

完整配置和启动前检查见 [FlashTalk 兼容 WebSocket](../serving/flashtalk_ws.md)。OpenTalking 侧保持 `OPENTALKING_FLASHTALK_MODE=remote`，并把 `OPENTALKING_FLASHTALK_WS_URL` 指向 OmniRT 服务即可。

## FlashHead Ascend 推荐配置

`soulx-flashhead-1.3b` 参考 910B 适配记录，默认采用质量优先配置：

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

## LiveAct Ascend 推荐配置

`soulx-liveact-14b` 使用外部 SoulX-LiveAct checkout 的 `generate.py`。Ascend 上默认设置 `PLATFORM=ascend_npu`，默认会先用单张 NPU 运行 `prepare_text_cache.py`，再启动 4 卡推理；显式多卡时可用 `--text-cache-visible-devices 2 --visible-devices 2,3,4,5` 固定为 1 卡 T5 + 4 卡推理。快速 smoke 可加 `--sample-steps 1`。若使用 LightVAE，请同时设置 `--vae-path models/vae/lightvaew2_1.pth --use-lightvae --use-cache-vae`，并预热 `--condition-cache-dir`。

## 错误与排查

!!! warning

    - **音频采样率不匹配** — SoulX 数字人模型通常要求 16 kHz 单声道；非此格式会自动 resample，但过长音频会放大误差
    - **portrait 对齐差** — 正面、上半身、眼睛可见效果最好
    - **外部仓库克隆失败** — 在国内网络下走 `GHPROXY` 或离线提供 `repo_path`
    - **FlashHead 输出风格漂移** — 先保持 `latent_carry=false`；该模式虽快，但适合作为实验开关而不是默认展示档
    - **Ascend 上速度显著低于 CUDA** — 检查 `FLASHHEAD_NPU_FUSION_ATTENTION`、`visible_devices`、CANN 环境和外部仓库的 NPU 适配补丁是否生效
    - **LiveAct 启动时报 CUDA 设备不可用** — 确认 `PLATFORM=ascend_npu` 已设置；OmniRT wrapper 默认会设置，但手工运行外部仓库时容易漏掉
    - **LiveAct T5 OOM** — 优先使用默认的单卡 NPU text context cache；显式设置 `--text-cache-visible-devices`，并避免让 T5 和 4 卡主推理同时加载
