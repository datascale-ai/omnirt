# Serving

OmniRT offers four common entry points, all sharing the same `GenerateRequest` contract:

| Entry point | Best for | Page |
|---|---|---|
| **Python API** | embedding in existing Python apps, notebook experiments | [Python API](python_api.md) |
| **CLI** | scripted batches, one-off `validate` / `generate` | [CLI](cli.md) |
| **HTTP server** | microservice, multi-tenant, OpenAI-compatible API, Prometheus / OTLP hooks | [HTTP Server](http_server.md) |
| **Worker server** | gRPC execution node used by `serve --remote-worker` | [Distributed Serving](../deployment/distributed_serving.md) |

!!! tip "Recommended order"
    Start in Python or CLI to validate the contract; then deploy the HTTP server for concurrency, batching, and policy tuning. Add remote workers only when you need to split execution across processes or hosts.
