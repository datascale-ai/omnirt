# 模型接入指南

本文档给出把一个 Diffusers 风格新模型接入 `omnirt` 的最短路径。OmniRT 当前聚焦数字人垂直链路；新增模型默认进入 `experimental`，只有能服务数字人主链路或素材生产，并且有明确验证证据时，才进入 `adjacent` 或 `core`。

## 0. 先确定维护层级

接入前先写清楚模型为什么进入 registry：

| Tier | 准入标准 |
|---|---|
| `core` | TTS、音频驱动数字人、实时 / 常驻 worker、部署和 benchmark 主线；必须有单测、真机 smoke、benchmark 和部署说明 |
| `adjacent` | 角色资产、背景、idle 视频、可控修图或后处理；必须能服务数字人产品链路，并有按场景设计的 smoke |
| `experimental` | 兼容保留或探索性泛模型；可以有 registry 和单测，但不进入默认生产服务、硬件 CI 或主线 benchmark |

升层规则：`experimental -> adjacent` 需要清楚的数字人用例和至少一个真实后端 smoke；`adjacent -> core` 需要端到端链路证据、benchmark、部署文档和维护人承诺。

## 1. 新增 pipeline 类

在 `src/omnirt/models/<model_name>/` 下创建 `pipeline.py`，继承 `BasePipeline` 并实现五阶段：

```python
from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model


@register_model(
    id="my-model",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("negative_prompt",),
        supported_config=(
            "model_path", "scheduler", "height", "width",
            "num_inference_steps", "guidance_scale", "seed", "dtype", "output_dir",
        ),
        default_config={
            "scheduler": "euler-discrete",
            "height": 1024, "width": 1024, "dtype": "fp16",
        },
        supported_schedulers=("euler-discrete", "ddim", "dpm-solver", "euler-ancestral"),
        adapter_kinds=("lora",),
        artifact_kind="image",
        maturity="experimental",   # experimental | beta | stable
        tier="experimental",        # core | adjacent | experimental
        summary="One-sentence description that shows up in `omnirt models`.",
        example='omnirt generate --task text2image --model my-model --prompt "..." --backend cuda',
    ),
)
class MyPipeline(BasePipeline):
    def prepare_conditions(self, req): ...
    def prepare_latents(self, req, conditions): ...
    def denoise_loop(self, latents, conditions, config): ...
    def decode(self, latents): ...
    def export(self, raw, req): ...
```

### `ModelCapabilities` 字段速览

`capabilities` 字段把模型的可用配置暴露给 `omnirt models`、`omnirt validate` 和 Python API：

| 字段 | 用途 |
|---|---|
| `required_inputs` | `inputs` 里必须出现的 key；缺失时 `validate()` 直接报错 |
| `optional_inputs` | `inputs` 里允许但非必须的 key |
| `supported_config` | `config` 里允许的 key；白名单外的值会被标记为 warning |
| `default_config` | 用户未显式指定时的默认值（height/width/dtype 等） |
| `supported_schedulers` | 本 pipeline 真正测过的 scheduler id；`--scheduler` 超出范围时报警 |
| `adapter_kinds` | 支持的 adapter 种类（目前仅 `"lora"`） |
| `artifact_kind` | `"image"` / `"video"`；决定导出器选择 |
| `maturity` | `experimental` / `beta` / `stable`；CLI `models` 列表会显示 |
| `tier` | `core` / `adjacent` / `experimental`；决定生产服务、CI 和 benchmark 的默认优先级 |
| `summary` / `example` | 给 `omnirt models` 和文档用的简介和示例命令 |

完整定义见 [src/omnirt/core/registry.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/registry.py)。

## 2. 复用后端契约

通过 `self.runtime.wrap_module(submodule, tag="unet")` 包装对后端敏感的子模块。这样新模型无需写专属后端代码，就能拿到 compile / override / eager 三级回退，并在 `RunReport.backend_timeline` 里有每步的记录。

## 3. 注册 alias（可选）

同一个 pipeline 可以同时暴露多个 id。把别名版本用 `alias_of` 标出，它会在 `omnirt models --format markdown` 里被放进独立的「Aliases」表，不会污染主清单：

```python
@register_model(
    id="flux2.dev",
    task="text2image",
    capabilities=ModelCapabilities(summary="Flux 2 dev text-to-image pipeline."),
)
@register_model(
    id="flux2-dev",
    task="text2image",
    capabilities=ModelCapabilities(
        summary="Flux 2 dev text-to-image pipeline.",
        alias_of="flux2.dev",
    ),
)
class Flux2Pipeline(BasePipeline):
    ...
```

装饰器可以堆叠，每次调用都会把一份条目写进 registry；`get_model("flux2-dev")` 和 `get_model("flux2.dev")` 解析到同一个类。

## 4. 规格与约定

- 权重：仅加载 `safetensors`
- Adapters：在 `__init__` 校验路径存在，在 `prepare_conditions` 阶段真正 materialize pipeline 后应用一次；单文件 adapter 既可以来自本地，也可以来自 Hugging Face
- 产物：`export` 返回带完整文件路径的 `Artifact`，CLI 依赖 `Artifact.path` 回显

## 5. 把模型接入注册入口

在 [src/omnirt/models/\_\_init\_\_.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/models/__init__.py) 的 `_BUILTIN_MODEL_IDS` 里加上你的 id，并在 `ensure_registered()` 里导入你的 `pipeline` 模块。这样 `omnirt models` 启动时会保证你的 pipeline 被加载。

## 6. 补测试

- 单测：用 `tests/unit/test_sd15_pipeline.py` 作为模板，靠 fake runtime + fake Diffusers 对象跑完 `BasePipeline.run()`
- Integration：`tests/integration/test_<model>_{cuda,ascend}.py`，依赖硬件的 case 在 `conftest.py` 里有自动 skip
- Parity（可选）：如果要参与跨后端验证，用 `tests/parity/test_parity.py` 的 latent 统计 / PSNR helper

测试准入建议：

- `experimental`：至少要有 registry / validate / pipeline 单测
- `adjacent`：在 experimental 基础上补数字人场景 smoke，并确认不会进入默认生产 tier 之外的 CI
- `core`：必须有端到端 smoke、benchmark 记录、部署文档和服务侧验证路径

## 7. 文档

- 新增模型**不需要**手工改 `docs/_generated/models.md` —— 跑一遍 `python scripts/generate_models_doc.py` 即可，`ModelCapabilities.summary` 会自动出现在表里
- 如果模型有特殊部署要求或已知限制，写进 [支持状态](../user_guide/models/support_status.md) 的「部分支持」或「已接入但仍待真机 smoke」小节
- 如果要把模型标成 `adjacent` 或 `core`，同步更新 [支持状态](../user_guide/models/support_status.md) 和 [路线图](../user_guide/models/roadmap.md)，说明它服务的数字人链路、验证范围和剩余风险

## 参考实现

数字人主线优先参考 `src/omnirt/models/flashtalk/`、`src/omnirt/models/flashhead/`、`src/omnirt/models/liveact/` 和 `src/omnirt/models/cosyvoice/`。素材和编辑能力可以参考 `src/omnirt/models/sdxl/`、`src/omnirt/models/svd/`、`src/omnirt/models/flux/`、`src/omnirt/models/flux2/` 和 `src/omnirt/models/generalist_image/`。`src/omnirt/models/sd15/`、`sd3/` 与其它泛模型保留为 experimental 兼容参考，不应作为新模型接入的默认方向。
