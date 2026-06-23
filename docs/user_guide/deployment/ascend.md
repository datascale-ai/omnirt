# Ascend 后端部署

OmniRT 原生支持华为昇腾 Atlas / 910 / 910B 系列。Ascend 后端与 CUDA 共享同一套外部契约（`GenerateRequest` / `GenerateResult` / `RunReport`），但编译路径更保守，失败会自动回退到 eager。

## 硬件与系统要求

| 项目 | 要求 |
|---|---|
| 设备 | Atlas 300I Pro / 800I / 800T / 910 / 910B 系列 |
| CANN | 8.0.RC2 及以上，与机器的 driver / firmware 对齐 |
| torch_npu | 与 CANN 版本匹配；`torch==2.1.0` + `torch_npu==2.1.0.post6` 为当前验证组合 |
| 驱动 / 固件 | 由 `Ascend-hdk-*` 安装包提供，必须与 CANN 同一大版本 |
| 系统工具 | `Ascend-toolkit-*` 的 `set_env.sh` 必须在启动前 `source` |
| Python | 3.10+；当前 CI 使用 3.11 |

## 安装

=== "pip"

    ```bash
    # 0. 确认 CANN 已预置（一般由运维安装）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 1. 装对应版本的 torch + torch_npu
    python -m pip install torch==2.1.0 torchvision==0.16.0
    python -m pip install torch_npu==2.1.0.post6

    # 2. 装 OmniRT 本体 + runtime
    python -m pip install -e '.[runtime,dev]'

    # 3. 烟测
    python -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch.npu.device_count())"
    omnirt generate --task audio2video --model soulx-flashtalk-14b \
      --image input.png --audio input.wav --backend ascend
    ```

=== "离线 wheel"

    ```bash
    # 内网环境：先在有网机器 download，再 copy 到目标机
    python -m pip download torch==2.1.0 torchvision==0.16.0 \
      torch_npu==2.1.0.post6 -d ./wheels
    # 目标机：
    python -m pip install --no-index --find-links ./wheels \
      torch torchvision torch_npu
    python -m pip install -e '.[runtime,dev]'
    ```

## 执行模型

- **后端名**：`ascend`
- **设备名**：`npu`
- **编译尝试**：`BackendRuntime.wrap_module(...)` 会优先调用 `torch_npu.npu.graph_mode()`
- **回退行为**：如果 graph-mode 初始化失败或模块不兼容，运行时会记录一条 `backend_timeline` 并保留 eager 模块继续执行
- **显存管理**：`torch_npu` 的 `empty_cache` 在每个 pipeline 阶段结束后被触发

## 设备可见性

```bash
ASCEND_RT_VISIBLE_DEVICES=0 omnirt generate ...      # 单卡
ASCEND_RT_VISIBLE_DEVICES=0,1 omnirt generate ...    # 多卡（目前公开 API 只用第一张；多卡并行不是公开能力）
```

`ASCEND_RT_VISIBLE_DEVICES` 是 CUDA 上 `CUDA_VISIBLE_DEVICES` 的等价物。

## vLLM-Omni 语音服务

TTS 模型如果已经由 vLLM-Omni/vLLM-Ascend 托管，OmniRT 推荐使用 `vllm-omni-speech` provider 通过 HTTP 调用 `/v1/audio/speech`，而不是把模型权重直接加载进 OmniRT 进程。这样可以把 Ascend runtime、vLLM worker、模型 chunking 和 OmniRT 的调度/观测边界分开。

910B 服务进程通常先准备环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Qwen3-TTS 示例：

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code
```

OmniRT 侧调用：

```bash
omnirt generate \
  --task text2audio \
  --model vllm-omni-speech \
  --prompt "你好，这是昇腾上的 vLLM-Omni 语音服务。" \
  --server-url http://127.0.0.1:8091 \
  --upstream-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --voice vivian \
  --language Chinese
```

如果用 OpenAI-compatible `/v1/audio/speech`，请求里的 `model` 可以直接写上游模型名；OmniRT 会自动落到 `vllm-omni-speech` 并转发。

## 已验证模型

下表反映最近一轮 Ascend smoke 的覆盖情况。完整清单以 [支持状态](../models/support_status.md) 为准。

| 模型 | 任务 | CANN | 备注 |
|---|---|---|---|
| `soulx-flashtalk-14b` | `audio2video` | 8.0.RC2+ | Core；常驻 worker 路径已完成真机 benchmark |
| `soulx-flashhead-1.3b` | `audio2video` | 8.0.RC2+ | Core；script-backed 冷启动包装 |
| `vllm-omni-speech` | `text2audio` | 8.3.RC1+ | 外部 vLLM-Omni/vLLM-Ascend 服务适配面已接入；真实 910B benchmark 单独记录 |
| `sdxl-base-1.0` | `text2image` | 8.0.RC2 | 稳定 |
| `svd-xt` | `image2video` | 8.0.RC2 | 部分算子回退到 eager |
| `wan2.2-t2v-14b` | `text2video` | 8.0.RC2+ | 初步验证；建议 `preset=balanced` |

## 校验流程

仓库提供 Ascend smoke tests，仅当以下前置满足时执行：

- `torch_npu` 已安装
- diffusers runtime 依赖已安装（`pip install '.[runtime]'`）
- 模型源通过对应 `OMNIRT_*_MODEL_SOURCE` / `OMNIRT_FLASHTALK_*` 环境变量提供
- 运行在 Ascend-capable host 上

前置不满足时测试会 **skip**，不会产生噪声式失败。

```bash
# 本地触发 Ascend smoke（如果前置齐全）
pytest tests/integration/test_ascend_smoke.py -q
```

## 常见问题

!!! warning

    - **`RuntimeError: graph mode init failed`** — 某些算子不被当前 CANN 版本支持；OmniRT 已自动回退 eager，失败条目记录在 `RunReport.backend_timeline`。无需处理，但可以在 log 里确认是否是预期的算子
    - **显存不释放**：跨请求复用 pipeline 的场景（例如 FastAPI 服务）下，Ascend `empty_cache` 不会立即归还系统内存；通过 `max_concurrency=1` + `pipeline_cache_size=1` 强制释放，见 [HTTP 服务](../serving/http_server.md)
    - **`torch==2.1.0` 与 diffusers 最新版本冲突**：锁定 diffusers 版本为 `0.37.x`（runtime extras 已声明）
    - **精度**：Ascend 默认 `bf16`，对部分模型（FlashTalk、Flux2）的数值稳定性比 CUDA 差，可显式 `--dtype fp16` 或 `--dtype fp32` 试错
    - **国内网络拉不到模型**：见 [国内部署](china_mirrors.md) 的 ModelScope / HF-Mirror / 离线快照流程

## 相关

- [CUDA 部署](cuda.md) — 对照两个后端的区别
- [国内部署](china_mirrors.md) — 镜像与离线策略
- [Docker 部署](docker.md) — Ascend 镜像模板
- [架构说明](../../developer_guide/architecture.md) — 后端层、`backend_timeline` 字段详情
