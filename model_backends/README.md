# OmniRT Model Backends

OmniRT keeps the framework environment separate from heavyweight model runtime environments.

The OmniRT environment should stay small: CLI, HTTP/gRPC server, request schemas, registry, orchestration, and lightweight adapters live in `src/omnirt`. Model-specific dependencies such as `torch-npu`, `flash_attn`, TensorRT-LLM, or vendor SDKs belong to a backend environment prepared per model family.

## Layout

```text
model_backends/
  <backend-name>/
    README.md
    requirements-*.txt
    prepare_*.sh
    start_*.sh
    *_server.py
```

Use this directory for scripts and docs needed to prepare or start an external model backend. Do not commit virtual environments, downloaded checkpoints, or complete upstream model repositories.

Runtime Manager stores generated artifacts outside `model_backends/`. By default they live under the OmniRT checkout at `.omnirt/`; choose another location with `omnirt runtime ... --home ./runtime-data` or `OMNIRT_HOME=./runtime-data` (paths are resolved from the shell working directory when you invoke the CLI).

The matching OmniRT adapter code remains under `src/omnirt/models/<model-name>/`.

## Runtime Boundary

OmniRT can call a model backend through several mechanisms:

- CLI or subprocess launch in the model backend environment.
- HTTP, gRPC, or WebSocket calls to an already running model service.
- Local resident workers when the model runtime can be imported safely in a model-specific process.

See also [`wav2lip/README.md`](wav2lip/README.md) for the FlashTalk-protocol-compatible **Wav2Lip** WebSocket service (OpenTalking `remote flashtalk`), and [`musetalk/README.md`](musetalk/README.md) for the same protocol backed by **MuseTalk v1.5** (OpenTalking adapter + weights under `models/`).

Each backend README should document:

- Supported hardware and Python version.
- Required external repository, if any.
- Environment preparation command.
- Checkpoint download locations.
- OmniRT environment variables needed to launch or connect.
