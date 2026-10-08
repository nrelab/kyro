# Deployment

## Docker

```bash
docker build -t kyro:latest .
docker run --rm -p 3000:3000 \
  -v /path/to/models:/models:ro \
  -e KYRO_MODEL_PATH=/models/llama3 \
  -e KYRO_TOKENIZER_PATH=/models/llama3/tokenizer.json \
  -e KYRO_MODEL_NAME=llama3 \
  kyro:latest
```

The runtime image is Debian-slim, non-root (`kyro` user), and ships a
`curl`-based `/health` healthcheck.

### Multi-arch builds

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t kyro:latest --push .
```

GPU builds require the CUDA runtime and `--features cuda`; the default image
runs on CPU only.

## Docker Compose

See `deploy/docker-compose.yml`. GPU passthrough is commented in; enable the
`deploy.resources` block with an NVIDIA container toolkit.

## Kubernetes

See `deploy/k8s-deployment.yaml`: Deployment with readiness/liveness
probes on `/health`, GPU limit of 1, read-only model volume, and a ClusterIP
Service on port 3000.

## Environment variables

| Variable | Default | Notes |
| -------- | ------- | ----- |
| `KYRO_MODEL_PATH` | — | Safetensors dir or `.gguf` file |
| `KYRO_TOKENIZER_PATH` | — | `tokenizer.json` |
| `KYRO_MODEL_NAME` | `kyro` | Advertised/served model id |
| `KYRO_HOST` / `KYRO_PORT` | `0.0.0.0` / `3000` | Bind |
| `KYRO_MAX_TOKENS_CAP` | 4096 | Per-request cap |
| `KYRO_MAX_PROMPT_BYTES` | 65536 | Per-request prompt cap |
| `KYRO_MAX_MESSAGES` | 256 | Per-request message cap |
| `KYRO_REQUEST_TIMEOUT_SECS` | 600 | Per-request timeout (streaming and non-streaming) |
| `KYRO_OTLP_ENDPOINT` | — | OpenTelemetry collector (requires `otlp` cargo feature) |
| `RUST_LOG` | `info` | Tracing filter |

## Production checklist

Before exposing Kyro to traffic:

- [ ] **Model and tokenizer verified**: engine starts with `KYRO_MODEL_PATH`; `/ready` returns 200 after startup.
- [ ] **Resource limits set**: `KYRO_MAX_TOKENS_CAP`, `KYRO_MAX_PROMPT_BYTES`, `KYRO_MAX_MESSAGES` sized for the deployment.
- [ ] **Timeout configured**: `KYRO_REQUEST_TIMEOUT_SECS` matches the client's patience (default 600s).
- [ ] **Kubernetes probes**: readiness → `GET /ready`; liveness → `GET /health`.
- [ ] **Metrics scraped**: Prometheus target on `:3000/metrics`; Grafana dashboard imported from `deploy/grafana-dashboard.json`.
- [ ] **Alerting installed**: rules from `docs/slos.md` loaded into Prometheus (error rate, TTFT/TBT, queue depth, KV cache, circuit breaker).
- [ ] **Graceful shutdown**: SIGTERM/SIGINT handled; rolling deploys drain in-flight requests.
- [ ] **Reverse proxy in front**: TLS termination and authentication (Kyro itself is unauthenticated).
- [ ] **KV cache sized**: monitor `kyro_kv_cache_usage_percent`; alert above 90%.
- [ ] **OTLP tracing** (optional): build with `--features otlp` and set `KYRO_OTLP_ENDPOINT` for distributed tracing.

## Security notes

Run behind an authenticating reverse proxy; API input limits above are the
first line of defense. Model files are validated at startup; invalid paths
fail fast with a clear error.
