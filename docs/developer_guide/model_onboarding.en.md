# Model Onboarding

This guide is the shortest path for onboarding a new Diffusers-style model into `omnirt`. OmniRT now focuses on the digital-human vertical. New models default to `experimental`; they should only move to `adjacent` or `core` when they serve the digital-human main path or asset-production loop and have clear validation evidence.

## 0. Choose the maintenance tier first

Before adding code, write down why the model belongs in the registry:

| Tier | Admission criteria |
|---|---|
| `core` | TTS, audio-driven avatars, realtime / resident workers, deployment, and benchmark mainline; requires unit tests, real-hardware smoke, benchmark, and deployment notes |
| `adjacent` | avatar assets, backgrounds, idle video, controlled edits, or post-processing; must serve a digital-human product path and have scenario-specific smoke coverage |
| `experimental` | compatibility or exploratory general models; may have registry and unit coverage, but does not enter default production serving, hardware CI, or mainline benchmarks |

Promotion rules: `experimental -> adjacent` requires a concrete digital-human use case and at least one real-backend smoke; `adjacent -> core` requires end-to-end path evidence, benchmark results, deployment docs, and an owner commitment.

## 1. Write the pipeline class

Create `src/omnirt/models/<model_name>/pipeline.py`, subclass `BasePipeline`, and implement the five stages:

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

### `ModelCapabilities` field reference

`capabilities` exposes the model's configuration surface to `omnirt models`, `omnirt validate`, and the Python API:

| Field | Purpose |
|---|---|
| `required_inputs` | Keys that must appear in `inputs`; `validate()` errors if missing |
| `optional_inputs` | Keys allowed but not required in `inputs` |
| `supported_config` | Allowed keys in `config`; anything outside the list is flagged as a warning |
| `default_config` | Fallback values when the user does not provide them (height/width/dtype…) |
| `supported_schedulers` | Scheduler ids actually tested for this pipeline; `--scheduler` outside the list emits a warning |
| `adapter_kinds` | Supported adapter kinds (currently only `"lora"`) |
| `artifact_kind` | `"image"` or `"video"`; selects the exporter |
| `maturity` | `experimental` / `beta` / `stable`; shown in `omnirt models` |
| `tier` | `core` / `adjacent` / `experimental`; controls the default production, CI, and benchmark priority |
| `summary` / `example` | Short description and sample command for docs and CLI |

Full definition: [src/omnirt/core/registry.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/core/registry.py).

## 2. Reuse the backend contract

Wrap backend-sensitive submodules through `self.runtime.wrap_module(submodule, tag="unet")`. This gives the new model compile / override / eager fallback behavior without any model-specific backend code, and every attempt lands in `RunReport.backend_timeline`.

## 3. Register aliases (optional)

Stacking `@register_model(...)` decorators lets one pipeline class expose several ids. Mark the alias variant with `alias_of` so it shows up in a dedicated "Aliases" block in `omnirt models --format markdown` instead of cluttering the main list:

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

Each decorator invocation writes its own registry entry; `get_model("flux2-dev")` and `get_model("flux2.dev")` resolve to the same class.

## 4. Formats and conventions

- Weights: `safetensors` only.
- Adapters: validate the path in `__init__`, then apply once inside `prepare_conditions` after the real pipeline is materialized; single-file adapters may come from local paths or Hugging Face.
- Artifacts: `export` returns `Artifact` records with complete file paths; the CLI echoes `Artifact.path`.

## 5. Plug the model into registration

Add your id to `_BUILTIN_MODEL_IDS` and import your `pipeline` module inside `ensure_registered()` in [src/omnirt/models/\_\_init\_\_.py](https://github.com/datascale-ai/omnirt/blob/main/src/omnirt/models/__init__.py). That way `omnirt models` loads your pipeline on startup.

## 6. Add tests

- Unit: copy `tests/unit/test_sd15_pipeline.py` as the template. Fake runtime + fake Diffusers object cover `BasePipeline.run()` end-to-end.
- Integration: `tests/integration/test_<model>_{cuda,ascend}.py`. Hardware-dependent cases are auto-skipped via the shared `conftest.py`.
- Parity (optional): if the model participates in cross-backend verification, wire into the latent-statistics / PSNR helpers in `tests/parity/test_parity.py`.

Recommended gates by tier:

- `experimental`: at least registry / validate / pipeline unit tests
- `adjacent`: add a digital-human scenario smoke and confirm it is not treated as broader-than-tier production CI
- `core`: end-to-end smoke, benchmark record, deployment docs, and a service-side validation path are required

## 7. Documentation

- You **do not** manually edit `docs/_generated/models.en.md`. Run `python scripts/generate_models_doc.py` and the `ModelCapabilities.summary` line flows into the table automatically.
- If the model has deployment quirks or known limitations, document them in the "Partial support" or "Awaiting hardware smoke" section of [Support Status](../user_guide/models/support_status.md).
- If you mark a model as `adjacent` or `core`, update [Support Status](../user_guide/models/support_status.md) and [Roadmap](../user_guide/models/roadmap.md) with the digital-human path it serves, the validation scope, and remaining risks.

## Reference pipelines

For the digital-human main path, start with `src/omnirt/models/flashtalk/`, `src/omnirt/models/flashhead/`, `src/omnirt/models/liveact/`, and `src/omnirt/models/cosyvoice/`. For adjacent asset and editing paths, use `src/omnirt/models/sdxl/`, `src/omnirt/models/svd/`, `src/omnirt/models/flux/`, `src/omnirt/models/flux2/`, and `src/omnirt/models/generalist_image/`. `src/omnirt/models/sd15/`, `sd3/`, and other general model families remain experimental compatibility references and should not be the default direction for new onboarding.
