# OmniRT

[English](./README.md) | [文档站点](https://datascale-ai.github.io/omnirt/) | [英文文档](https://datascale-ai.github.io/omnirt/en/)

面向 CUDA 与 Ascend 后端的开源图像、视频与音频驱动数字人生成运行时。

OmniRT 为不同模型家族提供统一的 CLI、Python API、请求校验流程、产物导出协议和后端抽象，让你在切换模型时不需要重学整套运行时接口。

## 亮点

- 以 `GenerateRequest`、`GenerateResult`、`RunReport` 为核心的统一请求 / 返回契约
- 面向用户的 CLI：`omnirt generate`、`omnirt validate`、`omnirt models`
- Python 侧同时提供 `requests.*` helper 和 `pipeline(...)` 便捷封装
- 支持 `cuda`、`ascend` 和 `cpu-stub` 三类后端模式
- 标准 PNG / MP4 产物导出
- 模型来源既支持本地目录，也支持 Hugging Face repo id
- LoRA safetensors 既支持本地文件，也支持 Hugging Face 单文件引用
- 适合本地模型目录、离线快照和受限网络环境的部署方式

## 当前公开任务面

| 任务面 | 说明 | 典型输出 |
|---|---|---|
| `text2image` | 文本驱动图像生成 | PNG |
| `image2image` | 图像引导图像生成 | PNG |
| `text2video` | 文本驱动视频生成 | MP4 |
| `image2video` | 首帧引导视频生成 | MP4 |
| `audio2video` | 音频驱动说话头像生成 | MP4 |

通过 CLI 可以最快查看当前实时 registry：

```bash
omnirt models
omnirt models flux2.dev
```

## 已支持模型

权威清单由 registry 自动生成，见
[docs/_generated/models.md](./docs/_generated/models.md)（英文：[docs/_generated/models.en.md](./docs/_generated/models.en.md)），
也可本地运行 `omnirt models` 查看。

当前公开接口已经足以支撑生成、校验、模型发现和产物导出。`image2image` 是正式公开任务面，推荐优先使用 `sdxl-base-1.0`、`sdxl-refiner-1.0`、`sd15`、`sd21`。`inpaint`、`edit`、`video2video` 仍在演进。

## 快速开始

```bash
python -m pip install -e '.[dev]'
python -m omnirt --help
pytest
```

Runtime extras（`'.[runtime,dev]'`）和 docs extras（`'.[docs]'`）需要时再单独安装。

完整走查——安装变体、第一条 `validate` / `generate`、YAML 请求、preset、`hf://` 单文件 LoRA 引用——见 [docs/getting-started.md](./docs/getting-started.md)。

## Python API

```python
from omnirt import generate, requests, validate

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    preset="balanced",
)
result = generate(req, backend="cuda")
```

完整参考——每个 task 的 typed request helper、`pipeline(...)` 便捷封装、RunReport 字段——见 [docs/python-api.md](./docs/python-api.md)。

## 测试与验证

- `pytest tests/unit tests/parity` 覆盖本地契约层与指标层
- `pytest tests/integration/test_error_paths.py` 覆盖低显存与坏权重错误路径
- CUDA / Ascend smoke tests 会在缺少硬件、运行时包或本地模型目录时自动跳过

真正的端到端生成仍然依赖目标硬件、运行时库和模型权重。

## 当前状态

- `sdxl-base-1.0` 和 `svd-xt` 已经完成 CUDA / Ascend 双后端真机 smoke
- `image2image` 已公开支持；其中 `sdxl-refiner-1.0` 的 CUDA / Ascend smoke 用例已经具备，但仍待补齐已验证的本地模型目录与真机验证结果
- `flux-fill`、`flux-kontext`、`qwen-image-edit*` 等编辑模型已经具备接入和 smoke 用例入口，但仍待补齐已验证的本地模型目录与真机验证结果
- 更完整的路线图见 [docs/user_guide/models/roadmap.md](./docs/user_guide/models/roadmap.md)，当前接入快照见 [docs/user_guide/models/support_status.md](./docs/user_guide/models/support_status.md)

## 文档

- 文档站点：<https://datascale-ai.github.io/omnirt/>
- 英文文档：<https://datascale-ai.github.io/omnirt/en/>
- 模型接入说明：[docs/developer_guide/model_onboarding.md](./docs/developer_guide/model_onboarding.md)
- 当前支持状态：[docs/user_guide/models/support_status.md](./docs/user_guide/models/support_status.md)
- 模型支持路线图：[docs/user_guide/models/roadmap.md](./docs/user_guide/models/roadmap.md)
- 中国区部署说明：[docs/user_guide/deployment/china_mirrors.md](./docs/user_guide/deployment/china_mirrors.md)
- 架构说明：[docs/developer_guide/architecture.md](./docs/developer_guide/architecture.md)
- 服务协议：[docs/user_guide/features/service_schema.md](./docs/user_guide/features/service_schema.md)
- Presets：[docs/user_guide/features/presets.md](./docs/user_guide/features/presets.md)
- Legacy 优化指南：[docs/developer_guide/legacy_optimization_guide.md](./docs/developer_guide/legacy_optimization_guide.md)
- Benchmark 基线：[docs/developer_guide/benchmark_baseline.md](./docs/developer_guide/benchmark_baseline.md)
- 接口改进决策记录：[docs/adr/0002-interface-improvements.md](./docs/adr/0002-interface-improvements.md)

## 工具脚本

- 准备离线模型快照：[scripts/prepare_model_snapshot.py](./scripts/prepare_model_snapshot.py)
- 拉取 Modelers 仓库快照：[scripts/prepare_modelers_snapshot.py](./scripts/prepare_modelers_snapshot.py)
- 准备 ModelScope 仓库及指定大文件快照：[scripts/prepare_modelscope_snapshot.py](./scripts/prepare_modelscope_snapshot.py)
- 检查本地模型目录布局：[scripts/check_model_layout.py](./scripts/check_model_layout.py)
- 同步模型目录到服务器：[scripts/sync_model_dir.sh](./scripts/sync_model_dir.sh)
