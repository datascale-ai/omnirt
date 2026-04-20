# 参与贡献

本页面给到 OmniRT 仓库的常规 PR 工作流、开发环境准备、测试入口与文档约定。

## 开发环境

```bash
git clone https://github.com/datascale-ai/omnirt.git
cd omnirt

# 开发依赖
python -m pip install -e '.[dev]'

# 要跑 runtime / server / docs 任何一块时再加对应 extras
python -m pip install -e '.[runtime,dev]'   # 真实推理
python -m pip install -e '.[server,dev]'    # 服务化
python -m pip install -e '.[docs]'          # 文档站
```

推荐 Python 版本 **3.11**（与 CI 对齐；3.10+ 应该也 ok）。

## pre-commit

```bash
python -m pip install pre-commit
pre-commit install
```

当前唯一的 hook 是 **`generate-models-doc`**：它会在你动到 `src/omnirt/core/registry.py`、`src/omnirt/models/**` 或 `docs/user_guide/models/supported_models.md` 时重新生成 registry 文档，并用 `--check` 确保没有漂移。

## 本地测试

=== "快速（单元 + parity）"

    ```bash
    pytest tests/unit tests/parity
    ```

    CI 的 `unit-and-parity` job 跑的就是这些。无需 GPU / NPU。

=== "错误路径"

    ```bash
    pytest tests/integration/test_error_paths.py
    ```

    覆盖低显存、坏权重、不兼容 adapter 等失败路径。

=== "CUDA smoke（需要 NVIDIA GPU）"

    ```bash
    OMNIRT_SDXL_MODEL_SOURCE=/path/to/sdxl \
    OMNIRT_SVD_MODEL_SOURCE=/path/to/svd \
    pytest tests/integration/test_sdxl_cuda.py tests/integration/test_svd_cuda.py
    ```

=== "Ascend smoke（需要 Ascend 设备）"

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    OMNIRT_SDXL_MODEL_SOURCE=/path/to/sdxl \
    OMNIRT_SVD_MODEL_SOURCE=/path/to/svd \
    pytest tests/integration/test_sdxl_ascend.py tests/integration/test_svd_ascend.py
    ```

硬件 smoke 在缺少前置依赖 / 模型源 / 硬件时会自动 skip，不会产生噪声失败。

## 文档

本站的文档 CI（`docs-lint` job）会依次做三件事：

1. **`scripts/generate_models_doc.py --check`** — registry 文档未过期
2. **`scripts/check_bilingual_parity.py`** — 每篇中文都有英文兄弟，长度不怪
3. **`mkdocs build --strict`** — 链接、引用、nav 全部合法

本地一次性复现：

```bash
python -m pip install -e '.[dev,docs]'
python scripts/generate_models_doc.py --check
python scripts/check_bilingual_parity.py
mkdocs build --strict
```

双语写作约定：中文文件是 `foo.md`，英文兄弟是 `foo.en.md`。详见 [文档发布](../community/publishing_docs.md)。

## PR 流程

1. **fork → 分支**：按 `feat/xxx`、`fix/xxx`、`docs/xxx` 命名
2. **commit 信息**：首行 ≤ 72 字，祈使句；正文解释"为什么"
3. **测试覆盖**：新增公开行为必须附 `tests/unit` 或 `tests/parity` 用例；硬件相关功能至少写一个 skippable smoke
4. **文档**：改动公开接口 / 请求 schema / 新模型时需同步更新 `docs/`
5. **提 PR**：在描述里说明改动动机与影响范围，附带本地 `pytest tests/unit tests/parity --maxfail=1` 的输出

## 新能力的接入点

- **新模型** → 读 [模型接入](model_onboarding.md)
- **新后端** → 读 [后端接入](backend_onboarding.md)
- **新任务面** → 动 `omnirt.core.types` + `omnirt.core.validation` + `omnirt.requests`，并提前开 ADR
- **新 feature（比如新 preset、新 adapter 种类）** → 在 PR 描述里写清动机 / 取舍，必要时先在 GitHub issue 或 Discussion 里走一轮共识

## 参考

- [架构说明](architecture.md) — 代码组织与当前 runtime 拓扑
- [模型接入](model_onboarding.md) — `@register_model` 与 `ModelCapabilities`
- [后端接入](backend_onboarding.md) — `BackendRuntime.wrap_module` 契约
- [pyproject.toml](https://github.com/datascale-ai/omnirt/blob/main/pyproject.toml) — extras 与脚本入口
