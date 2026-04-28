# OmniRT

<p align="center">
  <strong>面向 CUDA 与 Ascend 后端的统一图像 / 视频 / 数字人生成运行时</strong>
</p>

<p align="center">
  <a href="./README.en.md">English</a> ·
  <a href="https://datascale-ai.github.io/omnirt/">文档站点</a> ·
  <a href="https://datascale-ai.github.io/omnirt/en/">English Docs</a> ·
  <a href="https://github.com/datascale-ai/omnirt">GitHub</a>
</p>

<p align="center">
  <a href="https://github.com/datascale-ai/omnirt/stargazers"><img src="https://img.shields.io/github/stars/datascale-ai/omnirt?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/datascale-ai/omnirt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datascale-ai/omnirt" alt="License"></a>
  <a href="https://pypi.org/project/omnirt/"><img src="https://img.shields.io/pypi/v/omnirt.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/backend-CUDA%20%7C%20Ascend%20%7C%20cpu--stub-8A2BE2" alt="Backends">
</p>

---

OmniRT 是一个开源的多模态生成运行时，把 **文本→图像 / 图像→图像 / 文本→视频 / 图像→视频 / 音频→数字人** 等任务统一到同一套请求契约、同一组 CLI / Python API / HTTP 服务，以及可替换的硬件后端之上。切换不同模型家族时，你**不需要**重新适应一整套新的运行时接口。

## ✨ 核心亮点

- **统一请求契约** — `GenerateRequest`、`GenerateResult`、`RunReport` 三个对象覆盖全部任务面
- **跨后端运行时** — 同一份请求可在 `cuda` / `ascend` / `cpu-stub` 上完成校验与执行
- **三种入口** — Python API、CLI (`omnirt generate / validate / models`)、FastAPI 服务
- **16+ 模型家族** — SD1.5 / SDXL / SD3 / FLUX / FLUX2 / WAN / SVD / AnimateDiff / ChronoEdit / FlashTalk / FlashHead 等
- **产物标准化** — 图像统一导出为 PNG，视频统一导出为 MP4，每次运行都会生成一份 `RunReport`
- **离线与国内环境友好** — 同时支持本地目录、Hugging Face、ModelScope、Modelers 快照
- **LoRA 灵活加载** — 本地 safetensors 与 `hf://` 单文件引用并存
- **异步派发** — `queue` / `worker` / `policies` 支持批量请求与多模型排队执行
- **可插拔遥测** — `middleware.telemetry` 可将运行指标接入你现有的观测体系
- **安全默认** — `--dry-run` 与 `validate` 能让你在真机运行前尽早发现问题

## 🎯 公开任务面

| 任务 | 说明 | 典型输出 |
|---|---|---|
| `text2image` | 文本驱动图像生成 | PNG |
| `image2image` | 图像引导图像生成 | PNG |
| `text2video` | 文本驱动视频生成 | MP4 |
| `image2video` | 首帧引导视频生成 | MP4 |
| `audio2video` | 音频驱动说话数字人 | MP4 |

`inpaint`、`edit`、`video2video` 仍在持续演进，目前先通过模型特定入口提供。

## 🚀 快速开始

```bash
# 最小安装(含开发工具链)
pip install -e '.[dev]'

# 查看 CLI
omnirt --help

# 运行本地契约与解析层测试
pytest
```

如果需要实际运行模型，再按需安装以下扩展：

```bash
# 运行模型(diffusers / transformers / safetensors / torch)
pip install -e '.[runtime,dev]'

# 启动 HTTP 服务
pip install -e '.[server]'

# 构建 / 预览文档
pip install -e '.[docs]'
```

完整的入门流程（包括首次 `validate` / `generate`、YAML 请求格式、preset，以及 `hf://` 单文件 LoRA 引用）见 [docs/getting_started/quickstart.md](./docs/getting_started/quickstart.md)。

## 🐍 Python API

```python
from omnirt import generate, requests

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    preset="balanced",
)
result = generate(req, backend="cuda")
print(result.artifacts, result.report)
```

更多 helper（包括各任务的 typed request、`pipeline(...)` 便捷封装，以及 `RunReport` 字段说明）见 [docs/user_guide/serving/python_api.md](./docs/user_guide/serving/python_api.md)。

## 🖥️ 命令行

```bash
# 列出全部已注册模型
omnirt models

# 查看单个模型的元信息(min_vram_gb、推荐 preset 等)
omnirt models flux2.dev

# 先做一次请求校验(不会真正跑模型)
omnirt validate request.yaml

# 真机生成
omnirt generate request.yaml --backend cuda --out ./out
```

CLI 参考见 [docs/cli_reference/index.md](./docs/cli_reference/index.md)。

## 🧩 已支持模型

权威清单由 registry 实时生成，建议以 CLI 输出为准：

```bash
omnirt models
```

对应的文档镜像见 [docs/user_guide/models/supported_models.md](./docs/user_guide/models/supported_models.md)，当前接入快照见 [support_status.md](./docs/user_guide/models/support_status.md)。

| 类别 | 代表模型 |
|---|---|
| 图像 | `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sd15`, `sd21`, `sd3`, `flux.dev`, `flux2.dev`, `kolors`, `pixart-sigma`, `bria-3.2`, `lumina-t2x` |
| 图像编辑 | `flux-depth`, `flux-canny`, `flux-fill`, `flux-kontext`, `qwen-image-edit*`, `chronoedit` |
| 视频 | `svd-xt`, `wan*`, `animate-diff-sdxl`, `mochi`, `skyreels-v2`, `hunyuan-video-1.5-*`, `helios-*` |
| 数字人 | `flashtalk`, `flashhead` |

`image2image` 的推荐起点是 `sdxl-base-1.0`、`sdxl-refiner-1.0`、`sd15`、`sd21`。

## 🧱 架构速览

```
┌──────────────────────────────────────────────────────────────┐
│  CLI / Python API / FastAPI Server                           │
├──────────────────────────────────────────────────────────────┤
│  requests (typed helpers)  →  GenerateRequest                │
│                               ↓                              │
│                        dispatch / scheduler / queue          │
│                               ↓                              │
│                          engine  +  middleware               │
│                               ↓                              │
│  backends:  cuda   |   ascend   |   cpu-stub          │
│                               ↓                              │
│                models:  sdxl · flux · wan · svd · …          │
│                               ↓                              │
│                    GenerateResult + RunReport (PNG / MP4)    │
└──────────────────────────────────────────────────────────────┘
```

关于架构分层、后端抽象和模型适配的更多细节，见 [docs/developer_guide/architecture.md](./docs/developer_guide/architecture.md)。

## 🧪 测试与验证

- `pytest tests/unit tests/parity` — 覆盖本地契约层与指标层
- `pytest tests/integration/test_error_paths.py` — 覆盖低显存、坏权重等错误路径
- CUDA / Ascend smoke tests 在缺少硬件、运行时依赖或本地模型目录时会自动跳过

真正的端到端生成仍依赖目标硬件环境、运行时依赖和模型权重。

## 📦 当前状态

- `sdxl-base-1.0` 与 `svd-xt` 已在 CUDA 和 Ascend 双后端完成真机 smoke
- `image2image` 已正式公开；`sdxl-refiner-1.0` 已具备 smoke 用例，真机验证仍在进行中
- `flux-fill`、`flux-kontext`、`qwen-image-edit*` 等编辑模型已经接入 smoke 入口，待补齐已验证的本地模型目录
- 更完整的路线图见 [docs/user_guide/models/roadmap.md](./docs/user_guide/models/roadmap.md)

## 🚢 部署形态

你可以根据硬件条件与部署规模选择合适的部署形态：

| 形态 | 适用场景 | 文档 |
|---|---|---|
| CUDA 单机 | NVIDIA GPU 本地推理 / 开发机 | [cuda.md](./docs/user_guide/deployment/cuda.md) |
| Ascend 单机 | 昇腾 910 / 310P 等 NPU | [ascend.md](./docs/user_guide/deployment/ascend.md) |
| Docker | 容器化隔离、CI/CD、可复制环境 | [docker.md](./docs/user_guide/deployment/docker.md) |
| 分布式服务 | 多卡 / 多机 / 高并发在线服务 | [distributed_serving.md](./docs/user_guide/deployment/distributed_serving.md) |

### 按网络环境选择模型源

OmniRT 对模型来源做了统一抽象，你可以根据网络可达性灵活切换：

| 网络环境 | 推荐模型源 | 建议 |
|---|---|---|
| 可直连 Hugging Face | `hf://` 或 `huggingface.co` repo id | 默认方案，可获得最完整的模型矩阵与 `hf://` 单文件 LoRA 支持 |
| 国内 / Hugging Face 受限 | ModelScope、HF-Mirror、Modelers | 可通过镜像或 `modelscope://` 路径加载，使用体验与 HF 路径等价 |
| 完全离线 / 内网 | 本地模型目录 + 离线快照 | 可先在有网机器上用 [`prepare_model_snapshot.py`](./scripts/prepare_model_snapshot.py) / [`prepare_modelscope_snapshot.py`](./scripts/prepare_modelscope_snapshot.py) / [`prepare_modelers_snapshot.py`](./scripts/prepare_modelers_snapshot.py) 拉取快照，再用 [`sync_model_dir.sh`](./scripts/sync_model_dir.sh) 同步到目标机器 |

镜像配置、环境变量和完整的离线流程见 [docs/user_guide/deployment/china_mirrors.md](./docs/user_guide/deployment/china_mirrors.md)（覆盖 HF-Mirror / ModelScope / Modelers 三类镜像源）。

## 📚 文档导航

- **用户指南**
  - 快速开始:[docs/getting_started/quickstart.md](./docs/getting_started/quickstart.md)
  - CLI 参考:[docs/cli_reference/index.md](./docs/cli_reference/index.md)
  - Python API:[docs/user_guide/serving/python_api.md](./docs/user_guide/serving/python_api.md)
  - HTTP 服务:[docs/user_guide/serving/http_server.md](./docs/user_guide/serving/http_server.md)
  - 预设 Presets:[docs/user_guide/features/presets.md](./docs/user_guide/features/presets.md)
  - 请求校验:[docs/user_guide/features/validation.md](./docs/user_guide/features/validation.md)
  - 服务协议 Schema:[docs/user_guide/features/service_schema.md](./docs/user_guide/features/service_schema.md)
  - 派发与队列:[docs/user_guide/features/dispatch_queue.md](./docs/user_guide/features/dispatch_queue.md)
  - 遥测:[docs/user_guide/features/telemetry.md](./docs/user_guide/features/telemetry.md)
- **开发者指南**
  - 架构说明:[docs/developer_guide/architecture.md](./docs/developer_guide/architecture.md)
  - 模型接入:[docs/developer_guide/model_onboarding.md](./docs/developer_guide/model_onboarding.md)
  - 后端接入:[docs/developer_guide/backend_onboarding.md](./docs/developer_guide/backend_onboarding.md)
  - Benchmark 基线:[docs/developer_guide/benchmark_baseline.md](./docs/developer_guide/benchmark_baseline.md)
  - Legacy 优化指南:[docs/developer_guide/legacy_optimization_guide.md](./docs/developer_guide/legacy_optimization_guide.md)
  - 贡献指南:[docs/developer_guide/contributing.md](./docs/developer_guide/contributing.md)
- **API 参考**:[docs/api_reference/index.md](./docs/api_reference/index.md)

## 🔧 工具脚本

| 脚本 | 用途 |
|---|---|
| [`scripts/prepare_model_snapshot.py`](./scripts/prepare_model_snapshot.py) | 准备离线 Hugging Face 模型快照 |
| [`scripts/prepare_modelers_snapshot.py`](./scripts/prepare_modelers_snapshot.py) | 拉取 Modelers 仓库快照 |
| [`scripts/prepare_modelscope_snapshot.py`](./scripts/prepare_modelscope_snapshot.py) | 准备 ModelScope 仓库及大文件 |
| [`scripts/check_model_layout.py`](./scripts/check_model_layout.py) | 检查本地模型目录布局 |
| [`scripts/sync_model_dir.sh`](./scripts/sync_model_dir.sh) | 把模型目录同步到远程服务器 |

## 🤝 参与贡献

欢迎提交 Issue 和 PR。提交前请先阅读 [贡献指南](./docs/developer_guide/contributing.md)，并运行 `pytest` 与 `pre-commit run -a`，确保本地检查通过。

## 📄 许可证

本项目基于 [MIT License](./LICENSE) 发布。
