# Serving

OmniRT offers five common entry points. The normal generation paths share the same `GenerateRequest` contract; the FlashTalk-compatible WebSocket targets existing realtime avatar clients:

| Entry point | Best for | Page |
|---|---|---|
| **Python API** | embedding in existing Python apps, notebook experiments | [Python API](python_api.md) |
| **CLI** | scripted batches, one-off `validate` / `generate` | [CLI](cli.md) |
| **HTTP server** | microservice, multi-tenant, OpenAI-compatible API, Prometheus / OTLP hooks | [HTTP Server](http_server.md) |
| **Worker server** | gRPC execution node used by `serve --remote-worker` | [Distributed Serving](../deployment/distributed_serving.md) |
| **FlashTalk WebSocket** | connecting [OpenTalking](https://github.com/zyairehhh/opentalking) and other realtime avatar clients that already speak the FlashTalk WS protocol | [FlashTalk-compatible WebSocket](flashtalk_ws.md) |

!!! tip "Recommended order"
    For offline generation, start in Python or CLI to validate the contract, then deploy the HTTP server for concurrency, batching, and policy tuning. For an existing realtime avatar frontend, use the FlashTalk-compatible WebSocket path to connect the service first.
