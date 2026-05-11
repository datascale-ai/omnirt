# Request Validation

`omnirt.validate()` and the `omnirt validate` command check whether a request would run **without touching the hardware**: model existence, task compatibility, `config` field legality, dimension constraints, adapter / model compatibility.

## Why validate before running on real hardware

- **Cheap**: validation only reads the registry, presets, and schema — no weight loading, no GPU time
- **Predictable**: a `ValidationError` points to the exact field, e.g. "`width` must be a multiple of 8", instead of a cryptic CUDA runtime error
- **Consistent**: the same request validates identically on `cpu-stub`, `cuda`, and `ascend`

## Minimal examples

=== "Python"

    ```python
    from omnirt import validate
    from omnirt.requests import text2image

    req = text2image(model="sdxl-base-1.0", prompt="a lighthouse", width=513)  # 513 isn't a multiple of 8
    result = validate(req, backend="cpu-stub")
    print(result.ok)          # False
    print(result.errors)      # [("config.width", "must be a multiple of 8"), ...]
    ```

=== "CLI"

    ```bash
    omnirt validate \
      --task text2image \
      --model sdxl-base-1.0 \
      --prompt "a lighthouse" \
      --width 513 \
      --backend cpu-stub
    # Non-zero exit code on failure; --json for machine-readable output
    ```

=== "YAML + CLI"

    ```bash
    omnirt validate --config request.yaml --json
    ```

=== "HTTP (dry-run)"

    There is no separate `/v1/validate` endpoint yet; set `"dry_run": true` inside `config` to run validation only:

    ```bash
    curl -sS http://localhost:8000/v1/generate \
      -H 'Content-Type: application/json' \
      -d '{
        "task": "text2image",
        "model": "sdxl-base-1.0",
        "inputs": {"prompt": "a lighthouse"},
        "config": {"width": 513, "dry_run": true}
      }'
    ```

## What validation covers

1. **Model existence** — whether `model` (or an alias) is in the registry
2. **Task match** — whether the model supports the task (via `ModelCapabilities.tasks`)
3. **Required fields** — `prompt` (text2image / text2video), `image` (image2image / image2video), `audio` (audio2video), etc.
4. **Numerical constraints** — `width` / `height` multiples, `strength ∈ (0, 1]`, `num_frames` / `fps` ranges
5. **Preset legality** — `preset ∈ {fast, balanced, quality, low-vram}` and defined for this task + model
6. **Adapter compatibility** — LoRA / ControlNet references must appear in `ModelCapabilities.adapters`
7. **Backend reachability** — `backend=cuda` with `torch.cuda.is_available()=False` fails unless `backend=auto`

## In production

The FastAPI server runs the same `validate()` before inference; a failure returns `400 Bad Request` with the `errors` array.

## Related

- [Python API](../serving/python_api.md) — full `validate()` / `generate()` signatures
- [Service Schema](service_schema.md) — field-level reference for `GenerateRequest`
- [CLI](../serving/cli.md) — `omnirt validate` subcommand flags
