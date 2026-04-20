# Contributing

This page covers the standard PR workflow for the OmniRT repo — dev setup, tests, and documentation conventions.

## Dev environment

```bash
git clone https://github.com/datascale-ai/omnirt.git
cd omnirt

# Dev deps
python -m pip install -e '.[dev]'

# Add extras only when needed
python -m pip install -e '.[runtime,dev]'   # real inference
python -m pip install -e '.[server,dev]'    # FastAPI server
python -m pip install -e '.[docs]'          # docs site
```

Recommended Python: **3.11** (matches CI; 3.10+ should also work).

## pre-commit

```bash
python -m pip install pre-commit
pre-commit install
```

The sole current hook is **`generate-models-doc`**: it regenerates `docs/user_guide/models/supported_models.md` whenever you touch `src/omnirt/core/registry.py`, `src/omnirt/models/**`, or that doc, and runs `--check` so drift is caught before review.

## Local tests

=== "Fast (unit + parity)"

    ```bash
    pytest tests/unit tests/parity
    ```

    This is what CI's `unit-and-parity` job runs. No GPU / NPU required.

=== "Error paths"

    ```bash
    pytest tests/integration/test_error_paths.py
    ```

    Covers low-VRAM, bad-weight, incompatible-adapter, and other failure paths.

=== "CUDA smoke (requires NVIDIA GPU)"

    ```bash
    OMNIRT_SDXL_MODEL_SOURCE=/path/to/sdxl \
    OMNIRT_SVD_MODEL_SOURCE=/path/to/svd \
    pytest tests/integration/test_sdxl_cuda.py tests/integration/test_svd_cuda.py
    ```

=== "Ascend smoke (requires Ascend)"

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    OMNIRT_SDXL_MODEL_SOURCE=/path/to/sdxl \
    OMNIRT_SVD_MODEL_SOURCE=/path/to/svd \
    pytest tests/integration/test_sdxl_ascend.py tests/integration/test_svd_ascend.py
    ```

Hardware smoke tests skip automatically when prerequisites / model sources / hardware are missing.

## Documentation

The docs CI (`docs-lint` job) runs three things in order:

1. **`scripts/generate_models_doc.py --check`** — the registry doc is up to date
2. **`scripts/check_bilingual_parity.py`** — every Chinese page has an English sibling and lengths look reasonable
3. **`mkdocs build --strict`** — all links, references, and nav entries are valid

Reproduce locally:

```bash
python -m pip install -e '.[dev,docs]'
python scripts/generate_models_doc.py --check
python scripts/check_bilingual_parity.py
mkdocs build --strict
```

Bilingual convention: the Chinese source is `foo.md` and its English sibling is `foo.en.md`. See [Publishing Docs](../community/publishing_docs.md).

## PR workflow

1. **Fork → branch**: use `feat/xxx`, `fix/xxx`, or `docs/xxx` naming
2. **Commit messages**: first line ≤ 72 chars, imperative mood; the body explains *why*
3. **Test coverage**: new public behavior must ship with `tests/unit` or `tests/parity` cases; hardware-specific features need at least one skippable smoke
4. **Docs**: update `docs/` whenever public API, request schema, or model support changes
5. **Open the PR**: describe motivation and scope; paste the output of `pytest tests/unit tests/parity --maxfail=1`

## Extension points

- **New model** → read [Model Onboarding](model_onboarding.md)
- **New backend** → read [Backend Onboarding](backend_onboarding.md)
- **New task surface** → touch `omnirt.core.types` + `omnirt.core.validation` + `omnirt.requests`; open an ADR first
- **New feature (preset, adapter kind, etc.)** → spell out the motivation and trade-offs in the PR description; when alignment matters, socialize in a GitHub issue or Discussion first

## References

- [Architecture](architecture.md) — repo layout and the current runtime topology
- [Model Onboarding](model_onboarding.md) — `@register_model` + `ModelCapabilities`
- [Backend Onboarding](backend_onboarding.md) — `BackendRuntime.wrap_module` contract
- [pyproject.toml](https://github.com/datascale-ai/omnirt/blob/main/pyproject.toml) — extras and entry points
