# Backend Onboarding

Adding a new hardware backend to OmniRT (for example Rebellions NPU, MetaX GPU, or Apple Silicon) requires implementing the `BackendRuntime` abstract contract. Three reference implementations live under [src/omnirt/backends/](https://github.com/datascale-ai/omnirt/tree/main/src/omnirt/backends): `cuda.py`, `ascend.py`, and `cpu_stub.py`.

## Contract

The backend layer exposes a small, fixed surface to `omnirt.core.base_pipeline`, defined in [src/omnirt/backends/base.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/base.py):

| Method | Purpose | Required |
|---|---|---|
| `is_available()` | Can this backend actually run on the host? | Yes |
| `capabilities()` | Return `Capabilities` (dtype / compile / override options) | Yes |
| `_compile(module, tag)` | Attempt a hardware-specific compile wrapper on a module | Yes |
| `prepare_pipeline(pipeline, *, model_spec, config)` | Do backend-specific pipeline init (kernel registration, attention swaps) | Optional |
| `wrap_module(module, tag)` | Orchestrates `_compile` → `override` → `eager` with timeline tracking | **Provided by base; usually don't override** |
| `register_override(tag, override)` | Register a hand-written kernel as the compile-fallback | Optional |
| `reset_memory_stats()` / `memory_stats()` / `available_memory_gb()` | Feed VRAM samples into `RunReport.memory` | Optional (no-op default) |
| `synchronize()` | Block until the accelerator finishes (for accurate stage timing) | Optional (CUDA / Ascend implement it) |
| `to_device(tensor_or_module, dtype=None)` | Move tensor / module to the device | Optional (base default exists) |

## The three-level `wrap_module` fallback

Every `wrap_module` call appends a `BackendTimelineEntry` to `self.backend_timeline` with every attempt at **compile → kernel_override → eager**, including the exact failure reason. The entry is returned inside `RunReport.backend_timeline` and is the single source of truth for "why did this module fall back to eager".

**Key invariants:**

- A new backend only needs to implement `_compile`; the base class handles override + eager fallback
- If your hardware has no compile path (like `cpu_stub`), `raise NotImplementedError` from `_compile` and the base class routes through override → eager
- **Don't swallow exceptions**: every failure is exposed as `BackendAttempt.reason=str(exc)`, so logs and observability can alert on it

## Minimal backend template

```python
# src/omnirt/backends/mybackend.py
from __future__ import annotations
from typing import Any, Dict

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import Capabilities


class MyBackend(BackendRuntime):
    name = "mybackend"
    device_name = "my_device"  # match your torch / device runtime

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
        return my_device_runtime.compile(module)  # must return a callable

    def prepare_pipeline(self, pipeline: Any, *, model_spec: Any, config: Dict[str, Any]) -> Any:
        # Optional: swap attention / register kernel overrides
        self.register_override("attention", my_device_runtime.fused_attention)
        return pipeline

    # Optional: memory_stats / synchronize / etc.
```

## Wiring into dispatch

`omnirt.backends.__init__` exposes `resolve_backend(name)`, which maps a string to a backend instance. Adding a new backend requires:

1. Import `MyBackend` in `src/omnirt/backends/__init__.py` and add a dispatch branch
2. Extend the `BackendName` literal union in [src/omnirt/core/types.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/types.py) with `"mybackend"`
3. Extend the `--backend` choices list in `src/omnirt/cli/main.py`
4. `omnirt/server/app.py` needs no change (it passes the `backend` string through)

## Verification checklist

- [ ] `is_available()` returns `False` (does not raise) on hosts that lack the driver
- [ ] `capabilities()` covers every dtype your backend actually supports
- [ ] Failing `_compile` paths show up in `backend_timeline`, visible via `omnirt generate --json`
- [ ] `memory_stats()` return keys match the CUDA / Ascend shape (`peak_bytes`, `allocated_bytes`, etc.)
- [ ] At least one skippable smoke test exercises `sd15` + `text2image` for a single image, and validates parity (latent stats) against CPU stub or CUDA

## Reference implementations

- [cuda.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/cuda.py) — `torch.compile` plus fused-attention overrides
- [ascend.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/ascend.py) — `torch_npu.npu.graph_mode` with strict eager fallback
- [cpu_stub.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/backends/cpu_stub.py) — minimal implementation for validation and parity

## Related

- [Architecture](architecture.md) — where the backend layer sits in the current runtime topology
- [Model Onboarding](model_onboarding.md) — declaring backend capability requirements for a model
- [Telemetry](../user_guide/features/telemetry.md) — reading `backend_timeline` fields
