# LLM Sidecar Proxy Plan

## One-Line Pitch

Build an OpenAI-compatible LLM sidecar proxy that tracks model output-token usage, opens a per-model circuit breaker when quota is exhausted, and records agent-workflow traces across multiple LLM calls.

## MVP Goal

Create a small, credible AI infrastructure project that can be showcased on Twitter.

The MVP should demonstrate:

- An application or agent workflow sends multiple OpenAI-style chat completion requests through the proxy.
- The proxy records all LLM calls under one workflow trace ID.
- The proxy tracks output-token usage per model.
- When a model crosses its configured output-token quota, the proxy opens a circuit breaker.
- Future requests for that model are blocked with an OpenAI-style `429` error.
- In-flight upstream requests for that model are cancelled within the same sidecar process when possible.

The MVP is intentionally single-container and local-first. Distributed sidecars, Redis quota state, Postgres trace storage, trace export, spend estimation, and dashboards are future scope.

## Core Story

The demo story is an agent launch-planning workflow for the proxy itself.

Example flow:

```txt
trace_id = launch-demo-001

planner_agent       success
researcher_agent    success
writer_agent        success
reviewer_agent      success, opens circuit after crossing quota
final_editor_agent  blocked by circuit breaker
```

This lets the project be framed as:

> I am building an OpenAI-compatible LLM sidecar proxy that gives agent workflows a local safety layer: trace every model call, track output-token usage, and shut off future calls once a model budget is exhausted.

## MVP Scope

### 1. OpenAI-Compatible Endpoint

Support:

```txt
POST /v1/chat/completions
```

The proxy accepts an OpenAI-style chat completion request and returns an OpenAI-style chat completion response.

For MVP, only non-streaming responses are supported.

If a request sets:

```json
{ "stream": true }
```

the proxy returns a clear unsupported-feature error.

### 2. Provider Selection

The active provider is selected per request using:

```txt
X-LLM-Provider: mock
```

If the header is missing, the configured default provider is used.

MVP providers:

- `smart_mock`
- `openai_compatible` configured for Groq

The deterministic demo should use `smart_mock` by default.

The Groq provider is included to prove the proxy can sit in front of a real OpenAI-compatible upstream.

### 3. Smart Mock Provider

The smart mock provider should:

- Return OpenAI-shaped chat completion responses.
- Avoid calling a paid LLM API.
- Read `X-Span-Name` and the last user message.
- Return plausible deterministic responses for the launch-planner agent flow.
- Return usage metadata, including input, output, and total tokens.
- Simulate enough output-token usage to trigger the circuit breaker during the demo.

Known demo span names:

```txt
planner_agent
researcher_agent
writer_agent
reviewer_agent
final_editor_agent
```

Unknown span names should return a generic mock response.

### 4. Groq OpenAI-Compatible Provider

Groq support should be minimal and transparent.

Config example:

```yaml
providers:
  default: mock

  mock:
    type: smart_mock
    default_model: gpt-4.1-mini

  groq:
    type: openai_compatible
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
    default_model: llama-3.1-8b-instant
```

Proxy behavior:

- Forward `POST /v1/chat/completions` to `{base_url}/chat/completions`.
- Use the provider API key from the configured environment variable.
- Return the upstream response body unchanged to the client.
- Extract usage metadata internally for tracing and circuit breaker accounting.

The Twitter demo should not require Groq. It should run with mock by default.

### 5. Trace Model

The MVP uses tracing semantics similar to distributed tracing:

```txt
One agent workflow = one trace_id.
Each LLM call inside the workflow = one span.
```

Headers:

```txt
X-Trace-ID: launch-demo-001
X-Span-Name: planner_agent
```

If `X-Trace-ID` is missing, the proxy generates one.

If `X-Span-Name` is missing, the proxy generates or defaults the span name.

`GET /internal/traces/{trace_id}` returns the full workflow trace with all spans.

### 6. Span Fields

Each span should store:

```txt
span_id
span_name
started_at
ended_at
latency_ms
provider
upstream_base_url
model
status: success | error | blocked | cancelled
http_status_code
input_tokens
output_tokens
total_tokens
usage_source: provider | estimated | mock
quota_limit
quota_used_before
quota_used_after
circuit_opened_by_this_span
request_body
response_body
error_message
```

For MVP, body logging is enabled in the single shipped config because this is a local demo project.

### 7. Trace Summary Endpoint

`GET /internal/traces` returns trace summaries, not full span bodies.

Example shape:

```json
[
  {
    "trace_id": "launch-demo-001",
    "created_at": "2026-05-30T12:00:00Z",
    "updated_at": "2026-05-30T12:00:04Z",
    "span_count": 5,
    "status": "blocked",
    "models": ["gpt-4.1-mini"],
    "total_input_tokens": 320,
    "total_output_tokens": 540
  }
]
```

`GET /internal/traces/{trace_id}` returns the complete trace and all spans.

### 8. Output-Token Circuit Breaker

The MVP does not use reservations.

There is:

- No default output-token reservation.
- No requirement for clients to send `max_tokens` or `max_completion_tokens`.
- No cost estimation.

The proxy tracks actual output-token usage after responses.

For each configured model:

```yaml
models:
  gpt-4.1-mini:
    output_token_limit: 500
```

Behavior:

- If a model has no configured quota, treat quota as infinite.
- Track usage for all models, even if quota is infinite.
- If usage becomes `>= output_token_limit`, open the circuit for that model.
- The response that crosses the quota is still returned successfully to the client.
- Future requests for that model are blocked.
- Any other in-flight upstream requests for that model in the same sidecar process should be cancelled if possible.

Open circuit response:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
```

```json
{
  "error": {
    "message": "Output token circuit is open for model gpt-4.1-mini",
    "type": "quota_exceeded",
    "code": "output_quota_exceeded"
  }
}
```

The blocked call should still be recorded as a span with status `blocked`.

### 9. Circuit Reset

The circuit can reset in two ways:

- Manual internal reset endpoint.
- Quota period rollover.

Supported period values:

```yaml
quota:
  period: manual
```

and:

```yaml
quota:
  period: daily
```

```yaml
quota:
  period: monthly
```

Period key examples:

```txt
manual
2026-05-30
2026-05
```

For demo, use `manual`.

Reset endpoint:

```txt
POST /internal/usage/models/{model}/reset
```

### 10. Storage

MVP is single-container and local-first.

Use SQLite for:

- workflow traces
- spans
- per-model usage and circuit state

Use in-memory state for:

- active upstream requests per model
- process-local cancellation

Quota/circuit state should survive sidecar restarts.

SQLite is not intended to be shared across many sidecar containers in MVP.

### 11. Token Usage

Enforcement is based only on output tokens.

Trace metadata should include:

- input tokens
- output tokens
- total tokens

Usage source priority:

1. Provider usage, if present.
2. Mock usage, for smart mock responses.
3. Estimated usage, if provider usage is missing.

Fallback estimation:

- Use `tiktoken` when a model encoding can be resolved.
- Fall back to `ceil(chars / 4)` when `tiktoken` cannot map the model.

### 12. Internal Endpoints

MVP endpoints:

```txt
GET /healthz
GET /internal/traces
GET /internal/traces/{trace_id}
GET /internal/usage/models
POST /internal/usage/models/{model}/reset
```

Authentication is out of scope for MVP.

### 13. Demo Script

Include:

```txt
examples/quota_demo.py
```

The script should:

- Use one trace ID for the whole launch-planner workflow.
- Send multiple LLM calls with different `X-Span-Name` values.
- Use `X-LLM-Provider: mock` by default.
- Accept `--provider groq` for the real-provider variant.
- Print a clear workflow summary.
- Print the full trace URL.

Example output:

```txt
Trace: launch-demo-001

planner_agent       success   output=120   used=120/500
researcher_agent    success   output=130   used=250/500
writer_agent        success   output=160   used=410/500
reviewer_agent      success   output=120   used=530/500 circuit opened
final_editor_agent  blocked   circuit already open

Open full trace:
http://localhost:8000/internal/traces/launch-demo-001
```

### 14. Configuration

Use YAML config for providers, models, quota period, and local storage.

Use env vars for secrets and config path.

Example:

```yaml
server:
  host: 0.0.0.0
  port: 8000

database:
  url: sqlite:///./llm_proxy.db

tracing:
  log_full_body: true

quota:
  period: manual

providers:
  default: mock

  mock:
    type: smart_mock
    default_model: gpt-4.1-mini

  groq:
    type: openai_compatible
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
    default_model: llama-3.1-8b-instant

models:
  gpt-4.1-mini:
    output_token_limit: 500
  llama-3.1-8b-instant:
    output_token_limit: 1000
```

## Out Of Scope For MVP

- Streaming responses.
- Cutting off a streaming response mid-generation.
- Redis distributed quota state.
- Multiple sidecar containers.
- Cross-sidecar cancellation.
- Postgres trace storage.
- Periodic trace export.
- Dashboard UI.
- Cost estimation.
- Spend limits.
- Per-user or per-agent quotas.
- API keys for clients.
- Production authentication.
- Provider fallback.
- Retries with backoff and jitter.
- Prompt evals.
- Semantic caching.
- Provider-specific schema translation for Anthropic, Gemini, etc.
- Prometheus metrics.

## Follow-Up Scope

### Multi-Sidecar Mode

- Redis-backed quota and circuit state.
- Redis pub/sub or another coordinator for cross-sidecar circuit-open events.
- Cross-sidecar cancellation where possible.

### Shared Trace Storage

- Postgres trace backend.
- Periodic export from local SQLite to Postgres, object storage, ClickHouse, or OpenSearch.
- Trace retention policies.

### Richer Controls

- Per-user quotas.
- Per-agent quotas.
- Spend estimation and spend budgets.
- Pricing table.
- Cost fields in traces.
- Retry and timeout policy.

### Streaming

- Support `stream: true`.
- Count output tokens during streaming.
- Close the stream when quota is exhausted.

### Observability

- Dashboard for traces, spans, usage, latency, and circuit state.
- Prometheus metrics.
- OpenTelemetry export.

## Implementation Milestones

### Milestone 1: Skeleton

- FastAPI app.
- Config loader.
- `/healthz`.
- SQLite initialization.
- Basic `POST /v1/chat/completions`.

### Milestone 2: Smart Mock

- Provider interface.
- Smart mock provider.
- OpenAI-shaped mock responses.
- Token usage metadata.

### Milestone 3: Tracing

- Trace and span persistence.
- `X-Trace-ID` and `X-Span-Name` support.
- `GET /internal/traces`.
- `GET /internal/traces/{trace_id}`.

### Milestone 4: Circuit Breaker

- Per-model usage persistence.
- Circuit-open logic at `used >= limit`.
- `429` blocked responses.
- `GET /internal/usage/models`.
- Reset endpoint.

### Milestone 5: Demo

- `examples/quota_demo.py`.
- Tune mock output usage so the circuit opens during the launch-planner workflow.
- Print summary and trace URL.

### Milestone 6: Groq

- Minimal `openai_compatible` provider.
- Groq config.
- `--provider groq` demo path.
- Usage extraction from provider responses.

