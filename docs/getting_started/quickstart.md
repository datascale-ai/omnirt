# 快速开始

这份指南覆盖从全新 checkout 到一次可校验的 OmniRT 请求，以及本地文档站预览的最短路径。

OmniRT 的数字人主线优先沉淀 Ascend / 910B 的私有化部署、真机 smoke 和 benchmark 路径；CUDA 仍是模型生态最成熟的开发、验证和兼容后端。下面的环境说明按“先能本地校验，再选择 Ascend 或 CUDA 真机运行”的顺序展开。

## 环境要求

| 项目 | 最低要求 | 备注 |
|---|---|---|
| Python | 3.10 以上 | 建议 3.11 / 3.12，与 CI 的 `python-version: "3.11"` 对齐 |
| 操作系统 | Linux x86_64 | macOS 与 Windows 可用于开发 + `cpu-stub`，实机推理走 Linux |
| 内存 | ≥ 16 GB（系统） | 模型本身依赖 GPU / NPU 显存 |

OmniRT 自身只声明 `torch>=2.1` + diffusers + safetensors（见 [pyproject.toml](https://github.com/datascale-ai/omnirt/blob/main/pyproject.toml) 的 `runtime` extras）。CUDA 与 Ascend 的**专用 wheel 和底层驱动**由你按下表自行安装；`pip install '.[runtime,dev]'` 不会替你装 CUDA 工具链或 CANN。

### CPU stub（本地开发、请求校验、CI）

足够跑 `omnirt validate` / `--dry-run` / `pytest tests/unit tests/parity`，不涉及真实推理：

```bash
python -m pip install -e '.[dev]'
```

`--backend cpu-stub` 已经覆盖大多数单元路径，不需要 GPU / NPU。

### Ascend 环境（华为 Atlas / 910 / 910B）

| 项目 | 要求 |
|---|---|
| 设备 | Atlas 300I Pro / 800I / 800T / 910 / 910B 系列 |
| CANN | 8.0.RC2 以上，与所在机器的 driver/firmware 对齐 |
| torch_npu | 与 CANN 版本匹配的 wheel；`torch==2.1.0` + `torch_npu==2.1.0.post6` 是当前验证组合 |
| 驱动/固件 | 由 Ascend 侧 `Ascend-hdk-*` 安装包提供，版本必须与 CANN 保持同一大版本 |
| 系统工具 | `Ascend-toolkit-*` 的 `set_env.sh` 必须在启动前 `source` |

推荐安装顺序：

```bash
# 0. 先确认 CANN 已经在机器上（一般由运维预置）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1. 装对应版本的 torch + torch_npu
python -m pip install torch==2.1.0 torchvision==0.16.0
python -m pip install torch_npu==2.1.0.post6 -f https://download.pytorch.org/whl/torch_stable.html

# 2. OmniRT 本体 + runtime
python -m pip install -e '.[runtime,dev]'

# 3. 烟测
python -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch.npu.device_count())"
omnirt generate --task audio2video --model soulx-flashtalk-14b \
  --image input.png --audio input.wav --backend ascend
```

注意：

- `torch_npu` 的 `graph_mode` 对部分算子仍会失败，OmniRT 会自动把失败模块回退到 eager 并在 `RunReport.backend_timeline` 里记录（见 [架构说明](../developer_guide/architecture.md)）。
- Ascend 后端的更深入细节、已知限制与 smoke 入口见 [Ascend 后端](../user_guide/deployment/ascend.md)。
- 国内环境通常拉不到 `huggingface.co`；走内网模型镜像 / 离线快照的完整流程见 [国内部署](../user_guide/deployment/china_mirrors.md)。
- Ascend 设备可见性用 `ASCEND_RT_VISIBLE_DEVICES` 控制（等价于 CUDA 的 `CUDA_VISIBLE_DEVICES`）。

### CUDA 环境（NVIDIA GPU）

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA Ampere 及以上（A100 / L40S / RTX 3090 / 4090 等） |
| 显存 | 按模型 `resource_hint.min_vram_gb`，SDXL ≥ 12 GB、SVD ≥ 14 GB、Flux2 ≥ 24 GB（`omnirt models <id>` 查看每个模型的确切值） |
| NVIDIA 驱动 | ≥ 535（与 CUDA 12.1 配套） |
| CUDA Toolkit | 12.1 或 12.4 |
| PyTorch | 2.1+，官方 CUDA wheel 优先，例如 `torch==2.5.1+cu121` |

推荐安装顺序：

```bash
# 1. 先装匹配的 CUDA PyTorch wheel（从 pytorch.org 选索引）
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2. 再装 OmniRT 本体与其余 runtime 依赖
python -m pip install -e '.[runtime,dev]'

# 3. 烟测
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
omnirt generate --task text2image --model sdxl-base-1.0 --prompt "a lighthouse" --backend cuda --preset fast
```

提示：

- `torch.compile` 对 Ampere+ 才稳定；老卡可通过 `OMNIRT_DISABLE_COMPILE=1` 跳过编译一路走 eager。
- 多卡并行、USP、CFG 分片目前不是公开能力（见 [PLAN.md](https://github.com/datascale-ai/omnirt/blob/main/PLAN.md)）。
- 编译与显存日志会出现在 `RunReport.backend_timeline` 里。

## 安装

通用开发环境：

```bash
python -m pip install -e '.[dev]'
```

如果要真正执行模型推理，再安装 runtime 依赖（先装好上面的 CUDA / Ascend 底层 wheel）：

```bash
python -m pip install -e '.[runtime,dev]'
```

如果要维护文档站：

```bash
python -m pip install -e '.[docs]'
```

如果你希望把代码、runtime、文档工具放在同一个环境里：

```bash
python -m pip install -e '.[runtime,dev,docs]'
```

## 查看 CLI

```bash
python -m omnirt --help
omnirt models
omnirt models flux2.dev
```

`omnirt models` 是查看实时模型 registry 的最快方式，不需要先读源码。

## 先校验第一条请求

在真正使用 GPU / NPU 之前，建议先做一次纯校验：

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold title" \
  --backend cpu-stub
```

也可以直接校验配置文件：

```bash
omnirt validate --config request.yaml --json
```

## 跑通第一条生成请求

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

YAML 请求示例：

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

执行方式：

```bash
omnirt generate --config request.yaml --json
```

## 运行测试

本地快速覆盖：

```bash
pytest tests/unit tests/parity
```

错误路径集成测试：

```bash
pytest tests/integration/test_error_paths.py
```

依赖真实硬件的 CUDA / Ascend smoke tests 已接入 CI；本地若缺少对应 runtime 包或模型目录，会自动跳过。

## 预览文档站

```bash
mkdocs serve
```

使用严格模式构建静态站点：

```bash
mkdocs build --strict
```

GitHub Pages 发布说明见 [文档发布](../community/publishing_docs.md)。
