# 后端接入

把一个新硬件后端（例如 Rebellions NPU、MetaX GPU、Apple Silicon）接进 OmniRT，需要实现 `BackendRuntime` 的抽象接口。现有实现在 [src/omnirt/backends/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/backends) 下有三个参考：`cuda.py` / `ascend.py` / `cpu_stub.py`。

## 契约

后端层只对 `omnirt.core.base_pipeline` 暴露一组小而固定的方法，在 [src/omnirt/backends/base.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/base.py) 里定义：

| 方法 | 作用 | 必实现 |
|---|---|---|
| `is_available()` | 检测宿主机是否能真跑此后端 | 是 |
| `capabilities()` | 返回 `Capabilities`（dtype / compile / override 选项） | 是 |
| `_compile(module, tag)` | 尝试用硬件相关的编译路径包裹一个模块 | 是 |
| `prepare_pipeline(pipeline, *, model_spec, config)` | 对 pipeline 做后端相关的初始化（kernel 注册、attention 替换等） | 可选 |
| `wrap_module(module, tag)` | 调度 `_compile` → `override` → `eager` 的三级回退，并把每一步写入 `backend_timeline` | **已由 base 提供，一般不重写** |
| `register_override(tag, override)` | 注册一个手写的 kernel 替换，用作 `_compile` 失败的回退 | 可选 |
| `reset_memory_stats()` / `memory_stats()` / `available_memory_gb()` | 把显存采样喂给 `RunReport.memory` | 可选（默认 no-op） |
| `synchronize()` | 阻塞到加速器完成（为了准确的阶段计时） | 可选（CUDA / Ascend 已实现） |
| `to_device(tensor_or_module, dtype=None)` | 把张量 / 模块搬到目标设备 | 可选（base 已有默认） |

## `wrap_module` 的三级回退

每一次 `wrap_module` 都会在 `self.backend_timeline` 追加一条 `BackendTimelineEntry`，记录 **compile → kernel_override → eager** 三级的每一次尝试与原因。这个时间线最终随 `RunReport.backend_timeline` 返回给调用方，是"为什么这个模块回退到 eager"的唯一真相。

**关键不变量**：

- 新后端只需要实现 `_compile`；基类 `wrap_module` 自动处理 override + eager 回退
- 如果你的硬件完全没有编译路径（像 `cpu_stub`），直接在 `_compile` 里 `raise NotImplementedError`，基类会走到 override → eager
- **一个都不要 swallow**：所有异常都以 `BackendAttempt.reason=str(exc)` 的形式暴露；日志 / 观测可以基于这个字段报警

## 最小后端模板

```python
# src/omnirt/backends/mybackend.py
from __future__ import annotations
from typing import Any, Dict

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import Capabilities


class MyBackend(BackendRuntime):
    name = "mybackend"
    device_name = "my_device"  # 与 torch 或你的设备运行时一致

    def is_available(self) -> bool:
        try:
            import my_device_runtime   # noqa: F401
        except Exception:
            return False
        return my_device_runtime.is_available()

    def capabilities(self) -> Capabilities:
        return Capabilities(
            dtype_options=["fp16", "bf16"],
            compile_available=True,
            supports_graph_mode=False,
        )

    def _compile(self, module: Any, tag: str) -> Any:
        import my_device_runtime
        return my_device_runtime.compile(module)  # 必须原地 / 返回新的可调用对象

    def prepare_pipeline(self, pipeline: Any, *, model_spec: Any, config: Dict[str, Any]) -> Any:
        # 可选：替换 attention 实现 / 注册 kernel override
        self.register_override("attention", my_device_runtime.fused_attention)
        return pipeline

    # 其余方法（memory_stats / synchronize 等）按需实现
```

## 注册到分发器

`omnirt.backends.__init__` 里的 `resolve_backend(name)` 根据字符串名字返回后端实例。加入新后端需要：

1. 在 `src/omnirt/backends/__init__.py` 里 `import MyBackend` 并加入 dispatch 分支
2. 在 [src/omnirt/core/types.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/types.py) 的 `BackendName` 字面量联合里加入 `"mybackend"`
3. 在 `src/omnirt/cli/main.py` 的 `--backend choices` 列表里加入 `"mybackend"`
4. 在 `omnirt/server/app.py` 不需要改动（它只透传 `backend` 字符串）

## 验证清单

- [ ] `is_available()` 在缺失驱动的机器上返回 `False`，不抛异常
- [ ] `capabilities()` 返回的 `dtype_options` 覆盖你实际支持的所有 dtype
- [ ] `_compile` 的失败路径写到 `backend_timeline`，可被 `omnirt generate --json` 观察
- [ ] `memory_stats()` 的返回 key 与 CUDA / Ascend 现有实现对齐（`peak_bytes`、`allocated_bytes` 等）
- [ ] 至少一个 skippable smoke test 覆盖：跑 `sd15` + `text2image` 单张图，对比 latent 统计与 CPU stub / CUDA 的差异（parity）

## 参考实现

- [cuda.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/cuda.py) — `torch.compile` + fused attention override
- [ascend.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/ascend.py) — `torch_npu.npu.graph_mode` + 严格的 eager 回退
- [cpu_stub.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/cpu_stub.py) — 最小实现，用来走通 validation 与 parity 路径

## 相关

- [架构说明](architecture.md) — 后端层在当前 runtime 拓扑里的位置
- [模型接入](model_onboarding.md) — 新模型如何声明它依赖的后端能力
- [遥测](../user_guide/features/telemetry.md) — `backend_timeline` 字段与如何读它
