# Kyro LLM Engine

![CI](https://github.com/nrelab/kyro/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-70%25-green)

Kyro is a high-throughput LLM serving engine written in Rust, inspired by vLLM and TGI. It leverages the `candle` ML framework for efficient tensor operations and `tokio` for high-concurrency async scheduling.

## Key Features

- **Continuous Batching**: Iteration-level scheduling to maximize GPU throughput and eliminate queue wait times.
- **PagedAttention**: Virtual memory management for KV cache, eliminating memory fragmentation and enabling long-context serving.
- **Prefix Caching (Radix Cache)**: Automatic reuse of KV cache for common prefixes (system prompts, multi-turn history), enabling near-zero Time-To-First-Token (TTFT).
- **Chunked Prefill**: Eliminates "Prefill Stall" by interleaving large prompt processing with active decode steps.
- **Speculative Decoding**: Planned. Draft-model module exists (`src/speculative.rs`) but is not yet integrated into the worker loop or API.
- **Distributed Inference**: Planned. The engine currently runs single-device; `DistributedContext` exists but is a stub (no NCCL, no TP/PP).
- **Quantization Support**: GGUF weight loading via candle; FP8 and AWQ are on the roadmap (src/model/quantization is currently a simulation stub).
- **Constrained Decoding**: Structured JSON-mode and Regex-constrained output via grammar-based sampling.
- **Multi-LoRA Support**: Planned. LoRA math exists (`src/model/lora.rs`) but weight loading, API parameters, and scheduler integration are not yet implemented.
- **Observability**: Real-time Prometheus metrics for TTFT, TBT (Time Between Tokens), and KV cache utilization; optional OpenTelemetry (OTLP) trace export behind the `otlp` cargo feature.

## Architecture

1. **Frontend (Axum)**: Handles HTTP requests, streaming SSE, and health/metrics endpoints.
2. **Scheduler (Continuous Batching)**: Manages request queues, prefix caching, and chunked prefill scheduling.
3. **Model (Candle)**: Optimized Transformer blocks with GGUF quantization support (FP8/AWQ planned), PagedAttention kernels, and LoRA adapters (integration pending).4. **KV Cache (PagedAttention)**: Manages logical-to-physical block mapping via a **Reference-Counted BlockManager**, ensuring cached prefixes are protected from overwrite.
5. **Distributed (planned)**: Multi-node/multi-GPU synchronization via NCCL `All-Reduce` is not yet implemented.

## Getting Started

### Running the Engine

```bash
cargo run --release
```

The API will be available at `http://localhost:3000/v1/chat/completions`.

### Model Configuration

Startup is configured via CLI flags or environment variables:

| Flag | Environment variable | Description |
| ---- | -------------------- | ----------- |
| `--model-path` | `KYRO_MODEL_PATH` | Path to a Safetensors model directory or a `.gguf` file |
| `--tokenizer-path` | `KYRO_TOKENIZER_PATH` | Path to a HuggingFace `tokenizer.json` |
| `--model-name` | `KYRO_MODEL_NAME` | Served model name for request validation (default: `kyro`) |
| `--host` | `KYRO_HOST` | Bind address (default: `0.0.0.0`) |
| `--port` | `KYRO_PORT` | Bind port (default: `3000`) |
| — | `KYRO_MAX_TOKENS_CAP` | Max allowed `max_tokens` per request (default: 4096) |
| — | `KYRO_MAX_PROMPT_BYTES` | Max prompt size in bytes (default: 65536) |
| — | `KYRO_MAX_MESSAGES` | Max messages per chat request (default: 256) |
| — | `KYRO_REQUEST_TIMEOUT_SECS` | Per-request timeout for non-streaming completions (default: 600) |
| — | `KYRO_OTLP_ENDPOINT` | OpenTelemetry collector endpoint, e.g. `http://localhost:4317` (requires `otlp` cargo feature) |

Example with a real model:

```bash
cargo run --release -- \
  --model-path /models/Llama-3-8B \
  --tokenizer-path /models/Llama-3-8B/tokenizer.json \
  --model-name llama3
```

If `--model-path` is omitted, the engine starts with a dummy model for
development. An invalid model path or tokenizer fails startup with a clear
error.

The chat completions endpoint accepts OpenAI-compatible `messages` (or a raw
`prompt`), streams SSE chunks when `stream: true`, and validates the requested
`model` against the configured name.

### Benchmarking

To stress test the engine under concurrent load:

```bash
python benchmarks/stress_test.py
```

## API Documentation

- **POST** `/v1/chat/completions`: OpenAI-compatible completions endpoint. Supports `messages`, `prompt`, streaming SSE, and `response_format: {"type": "json_object"}` for grammar-masked decoding.
- **GET** `/v1/models`: Lists the served model.
- **GET** `/health`: Liveness and readiness probe.
- **GET** `/metrics`: Prometheus-formatted engine metrics.

Operational limits, chunking, quantization paths, and distributed-test notes are documented in [docs/limits.md](docs/limits.md) and [docs/architecture.md](docs/architecture.md). Full request/response shapes and error cases are in [docs/api.md](docs/api.md); deployment, Docker/Compose/Kubernetes, and security guidance in [docs/deployment.md](docs/deployment.md).
