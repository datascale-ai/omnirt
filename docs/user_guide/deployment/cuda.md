# CUDA 部署（NVIDIA GPU）

OmniRT 在 NVIDIA GPU 上的公开基线：Ampere 及以上的单卡推理。

## 硬件要求

| 项目 | 要求 | 备注 |
|---|---|---|
| GPU | Ampere 及以上 | A100 / L40S / RTX 3090 / 4090 等；`torch.compile` 对 Ampere+ 才稳定 |
| 显存 | 按模型 `resource_hint.min_vram_gb` | `omnirt models <id>` 查看精确值：SDXL ≥ 12 GB、SVD ≥ 14 GB、Flux2 / Wan2.2 ≥ 24 GB；SD1.5 等 experimental 模型仅作兼容参考 |
| 驱动 | ≥ 535（与 CUDA 12.1 配套） | `nvidia-smi` 检查 |
| CUDA Toolkit | 12.1 或 12.4 | 与 PyTorch wheel 对齐 |
| PyTorch | 2.1+ 官方 CUDA wheel | 例：`torch==2.5.1+cu121` |

## 安装

=== "pip"

    ```bash
    # 1. 装匹配的 CUDA PyTorch wheel（从 pytorch.org 选索引）
    python -m pip install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121

    # 2. 装 OmniRT 本体 + runtime extras
    python -m pip install -e '.[runtime,dev]'
    ```

=== "uv"

    ```bash
    uv pip install torch==2.5.1 torchvision==0.20.1 \
      --index https://download.pytorch.org/whl/cu121
    uv pip install -e '.[runtime,dev]'
    ```

=== "源码"

    ```bash
    git clone https://github.com/datascale-ai/omnirt.git
    cd omnirt
    python -m pip install -e '.[runtime,dev]'
    ```

## 冒烟测试

```bash
# 确认 CUDA 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 用 cpu-stub 做一次 dry-run 校验请求契约
omnirt validate --task text2image --model sdxl-base-1.0 --prompt "a lighthouse" --backend cpu-stub

# 跑一次真实生成
omnirt generate --task text2image --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" --backend cuda --preset fast
```

## FlashTalk CUDA

`soulx-flashtalk-14b` 使用独立 SoulX-FlashTalk runtime。GPU 路径参考官方教程：CUDA 12.8 PyTorch wheel、`flash_attn==2.8.0.post2 --no-build-isolation`，并且不应用 Ascend 代码补丁。

```bash
python -m omnirt.cli.main runtime install flashtalk --device cuda
OMNIRT_FLASHTALK_DEVICE=cuda OMNIRT_FLASHTALK_VISIBLE_DEVICES=0 bash scripts/start_flashtalk_ws.sh
```

更多 WebSocket、OpenTalking 接入和权重路径配置见 [FlashTalk 兼容 WebSocket](../serving/flashtalk_ws.md)。

## 生产配置建议

- **`torch.compile`**：默认开启，Ampere+ 稳定。如果遇到编译失败，设 `OMNIRT_DISABLE_COMPILE=1` 跳过；失败会记录在 `RunReport.backend_timeline`。
- **设备可见性**：`CUDA_VISIBLE_DEVICES=0`（单卡）或 `0,1`（多卡，但注意：**多卡并行 / USP / CFG 分片当前不是公开能力**，详见 [PLAN.md](https://github.com/datascale-ai/omnirt/blob/main/PLAN.md)）。
- **显存峰值**：通过 `RunReport.memory` 观察；如触发 OOM，切 `--preset low-vram` 或降 `width/height` / `num_frames`。
- **遥测接入**：`omnirt.middleware.telemetry` 把阶段计时、峰值显存、后端回退写入结构化日志，见 [遥测](../features/telemetry.md)。
- **服务化**：生产接入 FastAPI 服务见 [HTTP 服务](../serving/http_server.md)，默认建议 `--model-tier core --model-tier adjacent`。

## 已知问题

!!! warning

    - **老卡上 `torch.compile` 崩**：`OMNIRT_DISABLE_COMPILE=1` 回退 eager；每次回退记录在 `RunReport.backend_timeline`
    - **`flashinfer` / 自定义 attention kernel 未命中**：kernel override 失败会自动回退到 eager attention；查 `RunReport.backend_timeline` 里的 `kernel_override` 条目
    - **`triton` 旧版本编译慢**：升级到 PyTorch 推荐的 `triton` 版本
