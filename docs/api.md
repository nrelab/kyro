# API Reference

## POST /v1/chat/completions

Request:

```json
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hello"}],
  "prompt": null,
  "stream": false,
  "max_tokens": 64,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": null,
  "response_format": {"type": "json_object"}
}
```

- `messages` or `prompt` is required (400 otherwise).
- `model` must equal the configured `--model-name` (400 otherwise).
- `max_tokens` must be within `1..=KYRO_MAX_TOKENS_CAP`; `temperature` in
  `[0,2]`; `top_p` in `(0,1]`; message count ≤ `KYRO_MAX_MESSAGES`; prompt
  size ≤ `KYRO_MAX_PROMPT_BYTES`.
- Optional `priority` (0–100, default 0): higher-priority requests are
  dequeued by the scheduler before lower-priority ones; FIFO within equal
  priority.
- Optional `tools`: OpenAI-style tool/function definitions
  (`[{"type": "function", "function": {"name", "description", "parameters"}}]`,
  max 64). Tool schemas are folded into the prompt so the model can
  emit tool calls; responses are not auto-executed.
- Non-streaming requests are bounded by `KYRO_REQUEST_TIMEOUT_SECS`
  (default 600); exceeding it returns `504` with an
  `{"error": {"message", "type"}}` body. Streaming requests are
  bounded by the same deadline: the SSE stream terminates once
  it is reached.
- Requests received before model loading completes return `503`
  ("Engine is not ready").
- `stream: true` returns `text/event-stream` chunks
  (`chat.completion.chunk`), decoded incrementally, then a final chunk with
  `finish_reason: "stop"`.
- Errors return `{"error": {"message", "type"}}` with a 4xx status.

Response (non-streaming): standard `chat.completion` object with decoded
`choices[0].message.content`.

## GET /v1/models

Lists the served model in OpenAI list format.

## GET /health

`200 OK` plain text.

## GET /ready

Readiness probe: `200` once the model is loaded and the worker loop is
running, `503` otherwise. The flag also flips to `503` if the worker
circuit breaker trips (10 consecutive iteration failures). Use as a
Kubernetes readiness probe instead of `/health` to avoid routing
traffic to a cold or failed engine.

## POST /v1/cancel

Cancels an in-flight or queued request by ID.

Request:

```json
{"request_id": 123456}
```

- The `request_id` is returned in the `X-Request-Id` response header of
  the original `POST /v1/chat/completions` call (both streaming and
  non-streaming).
- `200` if the request was found and cancelled; `400` if no such request
  exists.
- Cancelling a running request frees its KV-cache blocks immediately; the
  client connection is closed by the server.

## GET /metrics

Prometheus text exposition: request counters by model label, queue depth,
prefix-cache hits/misses, token counters, TTFT/TBT histograms, KV-cache
usage, worker error counters, circuit-breaker state.

## Distributed Tracing (optional)

Build with the `otlp` feature and set `KYRO_OTLP_ENDPOINT` (or
`--otlp-endpoint`) to export `tracing` spans to an OpenTelemetry
collector (e.g. Jaeger, Tempo) over gRPC:

```bash
cargo run --release --features otlp -- \
  --otlp-endpoint http://localhost:4317
```

The `chat_completions` span carries `model` and `stream` attributes;
the provider flushes on graceful shutdown.
