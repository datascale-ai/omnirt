# FlashTalk Resident Benchmark

这份文档记录 `soulx-flashtalk-14b` 在 `persistent_worker` 常驻模式下的首轮真机 benchmark，方便后续提交继续按同一口径比较。

## 测试环境

- 日期：`2026-04-21`
- 机器：`内部 Ascend 验证主机`
- 加速器：`Ascend 910B2 x8`
- 运行入口：`omnirt serve` + `persistent_worker` + resident `torchrun`
- 模型：
  `soulx-flashtalk-14b`
  `SoulX-FlashTalk-14B`
  `chinese-wav2vec2-base`

## 基准配置

这次对齐的是对应 LiveAct benchmark summary 里的“速度优先”实时档：

- `FLASHTALK_HEIGHT=704`
- `FLASHTALK_WIDTH=416`
- `FLASHTALK_FRAME_NUM=29`
- `FLASHTALK_MOTION_FRAMES_NUM=1`
- `FLASHTALK_SAMPLE_STEPS=2`
- `FLASHTALK_COLOR_CORRECTION_STRENGTH=0`
- `audio_encode_mode=stream`
- 输入：
  `examples/woman2.jpg`
  `examples/cantonese_16k.wav`

resident 服务启动时还带了：

- `max_concurrency=1`
- `pipeline_cache_size=1`

## 口径说明

这组数据里最重要的是区分三层时间：

- `cold request`
  第一次请求，包含模型加载、分布式初始化和 resident worker 预热。
- `steady_chunk_core_ms_avg`
  热态下每个 chunk 的 `run_pipeline(...)` 主体时间，最接近 `SESSION_SUMMARY` 里的“稳态 chunk”。
- `steady_chunk_total_ms_avg`
  热态下每个 chunk 的总时间，额外包含音频 embedding 和 `video.cpu()` 收尾。

不要直接拿 `denoise_loop_ms / chunk_count` 和旧日志里的 `Generate video chunk-x done` 对比；两者口径不同。

## 结果摘要

### 冷启动

| 场景 | 端到端 |
|---|---:|
| `max_chunks=1` | `88.409s` |
| `max_chunks=3` | `91.196s` |
| 完整音频（`937` 帧） | `121.029s` |

### 热启动

| 场景 | 端到端 | 备注 |
|---|---:|---|
| `max_chunks=1` | `2.672s` | 单 chunk 热态请求 |
| `max_chunks=3` | `5.514s` | 用于估算 steady chunk |
| 完整音频（`937` 帧） | `37.377s` | resident 完整视频热态 |

### 热态 chunk 指标

`max_chunks=3` 热态 resident telemetry：

| 指标 | 数值 |
|---|---:|
| `audio_embedding_ms_avg` | `21.259 ms` |
| `chunk_core_ms_avg` | `894.051 ms` |
| `steady_chunk_core_ms_avg` | `891.002 ms` |
| `chunk_copy_ms_avg` | `33.339 ms` |
| `chunk_total_ms_avg` | `957.137 ms` |
| `steady_chunk_total_ms_avg` | `953.662 ms` |

## 与直接脚本对比

同一台内部验证主机、同一套 `704x416 + 29/1 + 2 step + stream` 配置下，直接运行 `generate_video.py` 的日志显示：

- `chunk-0`: `6.59s`
- `chunk-1`: `0.89s`
- `chunk-2`: `0.90s`

这说明：

- `steady_chunk_core_ms_avg ≈ 891ms`
  已经和直接脚本的 `0.89s` 到 `0.90s` 对齐。
- resident 比直接脚本多出来的 chunk 开销，主要落在：
  - `audio embedding`
  - `video.cpu()` / chunk 收尾

换句话说，当前 `persistent_worker` 路径没有把 FlashTalk 的核心 DiT 生成速度拖慢。

## 结论

- `soulx-flashtalk-14b` 在 `omnirt` 里的 `persistent_worker` 常驻模式已经达到可用状态。
- 实时配置下，核心生成速度已经对齐旧的 standalone 最优档。
- 若继续优化，下一阶段更值得盯的是：
  - `video.cpu()` / 结果收集
  - chunk 级外围开销
  - 更长音频和更多样本的稳定性压测

## 相关

- [Benchmark 基线](benchmark_baseline.md)
- [架构说明](architecture.md)
- [当前支持状态](../user_guide/models/support_status.md)
