# 请求校验

`omnirt.validate()` 与 `omnirt validate` 命令在**不占用硬件**的前提下检查一次请求能否跑通：模型是否存在、任务是否匹配、`config` 字段是否合法、尺寸约束是否满足、adapter 是否与模型兼容。

## 为什么在真机前做

- **便宜**：校验只读 registry、presets、schema；不加载权重、不碰 GPU
- **可预测**：把 `ValidationError` 明确指向字段，告诉你"`width` 必须是 8 的倍数"而不是一条 CUDA runtime error
- **一致**：同一个 request 在 CPU-stub / CUDA / Ascend 上会得到相同的校验结果

## 最小示例

=== "Python"

    ```python
    from omnirt import validate
    from omnirt.requests import text2image

    req = text2image(model="sdxl-base-1.0", prompt="a lighthouse", width=513)  # 513 非 8 的倍数
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
    # Exit code non-zero if validation fails; --json for machine-readable output
    ```

=== "YAML + CLI"

    ```bash
    omnirt validate --config request.yaml --json
    ```

=== "HTTP（dry-run）"

    HTTP 侧当前没有独立 `/v1/validate` 端点；在请求 `config` 里加 `"dry_run": true` 即可让 engine 仅做校验：

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

## 校验覆盖什么

1. **模型存在性**：`model` 是否在 registry（或其 alias）里
2. **任务匹配性**：`model` 是否支持 `task`（通过 `ModelCapabilities.tasks`）
3. **必填字段**：`prompt`（text2image / text2video）、`image`（image2image / image2video）、`audio`（audio2video）等
4. **数值约束**：`width` / `height` 的倍数、`strength ∈ (0, 1]`、`num_frames` / `fps` 的合法范围
5. **Preset 合法性**：`preset ∈ {fast, balanced, quality, low-vram}` 且该 preset 对当前 task + model 有定义
6. **Adapter 兼容性**：LoRA / ControlNet 是否出现在 `ModelCapabilities.adapters` 里
7. **后端可达性**：`backend=cuda` 但 `torch.cuda.is_available()=False` 时会报错（除非写明 `backend=auto`）

## 在服务化场景里用

FastAPI 服务的 `/v1/generate` 在正式推理前会复用同一套 `validate()`；一旦 `ValidationResult.ok=False`，返回 `400 Bad Request`，响应体包含 `errors` 列表。

## 相关

- [Python API](../serving/python_api.md) — 完整 `validate()` / `generate()` 签名
- [服务协议](service_schema.md) — `GenerateRequest` 字段级参考
- [CLI](../serving/cli.md) — `omnirt validate` 子命令参数
