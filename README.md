# LLM Sidecar Proxy

OpenAI-compatible local sidecar proxy for tracing agent workflows and enforcing
per-model output-token quotas.

## Run

```bash
./start-server.sh
```

The default `config.yaml` uses the deterministic smart mock provider and stores
state in `./llm_proxy.db`.

Override host or port when needed:

```bash
HOST=127.0.0.1 PORT=9000 ./start-server.sh
```

## Demo

In another shell:

```bash
uv run python examples/quota_demo.py
```

The demo sends five OpenAI-style chat completion requests under one trace ID.
The fourth request crosses the `gpt-4.1-mini` output-token quota, and the fifth
request is blocked with an OpenAI-style `429`.

## Endpoints

- `POST /v1/chat/completions`
- `GET /healthz`
- `GET /internal/traces`
- `GET /internal/traces/{trace_id}`
- `GET /internal/usage/models`
- `POST /internal/usage/models/{model}/reset`

## Migrations

Persistence uses SQLAlchemy ORM models with Alembic migrations. The app runs
migrations on startup for the configured SQLite database.

To run migrations manually:

```bash
uv run alembic upgrade head
```

## Real Provider Variant

Set `GROQ_API_KEY`, keep the `groq` provider in `config.yaml`, then run:

```bash
uv run python examples/quota_demo.py --provider groq
```

## Tests

```bash
uv run pytest
```
