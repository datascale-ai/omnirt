# Python API

OmniRT 暴露了一组精简但清晰的 Python 接口，用于请求构造、请求校验、直接执行，以及以模型为中心的便捷封装。

当前正式公开的任务 helper 包括 `text2image`、`image2image`、`text2video`、`image2video` 和 `audio2video`。`image2image` 推荐从 `sdxl-base-1.0`、`sdxl-refiner-1.0` 开始；`sd15`、`sd21` 仅作为 experimental 兼容参考。

## 基础导入

```python
import omnirt
from omnirt import generate, requests, validate
from omnirt.core.types import GenerateRequest, GenerateResult, RunReport
```

常用的顶层导出包括：

- `requests`
- `generate(...)`
- `validate(...)`
- `pipeline(...)`
- `list_available_models(...)`
- `describe_model(...)`

## Typed request helpers

`requests` 模块提供了按任务组织的 helper，用来构造符合包结构约定的 `GenerateRequest` 对象。

```python
from omnirt import requests

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    width=1024,
    height=1024,
    preset="balanced",
)
```

`image2image` 使用同一套公开 helper：

```python
img2img = requests.image2image(
    model="sdxl-base-1.0",
    image="input.png",
    prompt="cinematic concept art",
    strength=0.8,
)
```

此外还公开了这些请求类型：

- `TextToImageRequest`
- `TextToVideoRequest`
- `ImageToImageRequest`
- `InpaintRequest`
- `EditRequest`
- `ImageToVideoRequest`
- `AudioToVideoRequest`

## 请求校验

```python
from omnirt import validate

validation = validate(req, backend="cpu-stub")
print(validation.ok)
print(validation.resolved_backend)
print(validation.resolved_config)
```

在真正开始长耗时执行之前，校验是查看默认值解析结果、后端选择和请求错误的最安全入口。

## 模型发现

`list_available_models(...)` 支持按维护层级过滤，方便服务侧只暴露数字人主线或相邻素材能力：

```python
import omnirt

core_models = omnirt.list_available_models(tier="core")
adjacent_models = omnirt.list_available_models(tier="adjacent")
```

## 直接执行

```python
from omnirt import generate

result = generate(req, backend="cuda")
```

`generate(...)` 支持这些输入形式：

- `GenerateRequest`
- 符合请求结构的普通字典
- YAML 或 JSON 请求文件路径

与权重来源相关的约定：

- `config["model_path"]` 可以是本地模型目录，也可以是 Hugging Face repo id
- `AdapterRef.path` 可以是本地 `.safetensors`，也可以是 `hf://owner/repo/path/to/file.safetensors`

## `pipeline(...)` 便捷封装

如果你想以模型为中心来调用，并让关键字参数自动落入 `inputs` / `config`，可以使用 `pipeline(...)`。

```python
import omnirt

pipe = omnirt.pipeline("sdxl-base-1.0", backend="cpu-stub")
validation = pipe.validate(prompt="a lighthouse in fog", preset="fast")
```

通过 pipeline 直接执行：

```python
result = pipe(prompt="a lighthouse in fog", preset="fast")
```

如果传入未知关键字参数，会直接抛出 `ValueError`，避免无声吞掉不支持的选项。

## 返回结果模型

`generate(...)` 返回一个 `GenerateResult`：

```python
result: GenerateResult
outputs = result.outputs
report: RunReport = result.metadata
```

`RunReport` 里最值得关注的字段包括：

- `task`、`model` 和最终解析出的 `backend`
- 各阶段 `timings`
- 内存观测信息
- 用于记录编译和回退过程的 `backend_timeline`
- `config_resolved`
- 导出的 `artifacts`
- 如有失败，对应的 `error`

如果你要把 OmniRT 接进服务侧，请继续阅读 [服务协议](../features/service_schema.md)。
