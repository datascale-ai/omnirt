# FlashHead Benchmark

这份文档记录 `soulx-flashhead-1.3b` 在 OmniRT `subprocess` 包装路径下的首轮真机 benchmark。它和外部 SoulX-FlashHead 适配文档的 resident 数据口径不同：这里测的是 OmniRT 调用 `generate_video.py` 的冷启动端到端链路。

## 测试环境

- 日期：`2026-04-28`
- 机器：`内部 Ascend 验证主机`
- 加速器：`Ascend 910B2`
- 运行入口：`omnirt generate` + `subprocess` + `torchrun`
- 模型：
  `soulx-flashhead-1.3b`
  `SoulX-FlashHead-1_3B`
  `wav2vec2-base-960h`
- 外部仓库：`/path/to/SoulX-FlashHead`
- OmniRT 测试目录：`/path/to/omnirt`

## 基准配置

这次对齐的是质量优先的 910B 配置：

- `model_type=pro`
- `audio_encode_mode=stream`
- `FLASHHEAD_SAMPLE_STEPS=2`
- `FLASHHEAD_VAE_2D_SPLIT=1`
- `FLASHHEAD_LATENT_CARRY=0`
- `FLASHHEAD_NPU_FUSION_ATTENTION=1`
- 输入：
  `examples/girl.png`
  `bench_results/bench_10s.wav`
- 输出：
  `512x512`
  `25 FPS`
  `10.0s`
  `250 frames`

!!! note "设备可见性"
    真机运行时需要在启动 OmniRT 父进程前设置 `ASCEND_RT_VISIBLE_DEVICES`，不能只传 `--visible-devices`。资源预算检查发生在外部 `torchrun` 启动前，如果父进程仍看到被占用的默认 NPU，会错误地按默认卡计算可用显存。

## 命令模板

2 卡：

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

4 卡只需把 `ASCEND_RT_VISIBLE_DEVICES`、`--visible-devices` 改成 `2,3,4,5`，并将 `--nproc-per-node` 改成 `4`。

## 结果摘要

| 配置 | wall time | `denoise_loop_ms` | `export_ms` | 输出 |
|---|---:|---:|---:|---|
| 2 卡冷启动 | `82.96s` | `69,501.215 ms` | `264.686 ms` | `512x512 / 10s / 250 frames` |
| 4 卡冷启动 | `84.08s` | `69,963.908 ms` | `237.519 ms` | `512x512 / 10s / 250 frames` |

这组数据说明：

- 当前 OmniRT `subprocess` 包装链路已经能在 910B 上完成真实 `audio2video` 生成。
- 冷启动口径下，4 卡没有优于 2 卡；分布式初始化、模型加载、数据准备和首轮算子预热会抵消多卡收益。
- 要衡量多卡稳态收益，应继续做 `persistent_worker` / resident 路径，而不是只看单次脚本冷启动。

## 产物校验

| 配置 | RunReport `run_id` | 远端输出目录 | 校验 |
|---|---|---|---|
| 2 卡 | `8361015e-f1d6-4ad7-8c3c-6f3680354fa1` | `outputs/flashhead_bench_20260428_180909` | `ffprobe`: `512x512 / 25 FPS / 10.0s / 250 frames`; `blackdetect` / `freezedetect` 未输出告警 |
| 4 卡 | `79ebe868-609a-4f5f-a571-6366d984aeb2` | `outputs/flashhead_bench_20260428_181056` | `ffprobe`: `512x512 / 25 FPS / 10.0s / 250 frames`; `blackdetect` / `freezedetect` 未输出告警 |

## 已知注意点

- 远端运行 OmniRT CLI 需要 `protobuf` 和 `grpcio`；本次在 `/path/to/flashhead-venv` 中通过清华源补装。
- 当前文档记录的是 OmniRT `subprocess` 冷启动链路，不代表服务化热态时延。
- `latent_carry=false` 是默认质量档；`latent_carry=true` 虽可减少部分 VAE encode 开销，但参考适配记录里存在风格漂移，不作为默认展示配置。

## 相关

- [Benchmark 基线](benchmark_baseline.md)
- [FlashTalk Resident Benchmark](flashtalk_resident_benchmark.md)
- [当前支持状态](../user_guide/models/support_status.md)
