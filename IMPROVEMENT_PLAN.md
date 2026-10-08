# Kyro LLM Engine: Improvement Plan & Roadmap

**Date:** October 7, 2026  
**Status:** Ready for implementation  
**Estimated Effort:** 12–16 weeks across three milestones  
**Repository:** nrelab/kyro

---

## Executive Summary

Kyro is a well-architected, early-stage LLM serving engine with a solid core runtime. The architecture (continuous batching, prefix caching, chunked prefill) is production-ready at a single-GPU level. However, several advertised features are incomplete or non-functional, and observability/resilience gaps remain.

**Key Issues:**
- Distributed inference (TP/PP) is advertised but not implemented.
- Quantization claims are partially misleading (GGUF is real; AWQ/FP8 are mocks).
- LoRA and speculative decoding are partially coded but not integrated.
- Model ecosystem is limited to Llama.
- Observability is partial (metrics exist; tracing and alerting are missing).
- Production resilience patterns are incomplete.

**Strategy:** Ship production blockers first, then expand features and operational maturity in parallel milestones.

---

## Current Implementation Status

| Area | Status | Notes |
|------|--------|-------|
| **Testing** | ✅ Done | 92 tests, 72.00% coverage, CI gate at ≥70%; scheduler modules >94%. |
| **Distributed Inference** | ❌ Open | `src/distributed.rs` is a stub; no NCCL or weight sharding. |
| **Quantization** | 🔶 Partial | GGUF functional; AWQ/FP8 are no-op casts. Audit documented in code. |
| **LoRA** | 🔶 Partial | Module exists; not integrated into forward pass or API. |
| **Speculative Decoding** | 🔶 Partial | Module exists; not active in worker loop. |
| **Model Ecosystem** | ❌ Open | Llama only; Mistral, Qwen, etc. not supported. |
| **Observability** | ✅ Done | Core metrics, Grafana dashboard, SLO/alert rules, KV-cache gauge, optional OTLP tracing (`otlp` feature), structured request spans. |
| **Error Handling** | ✅ Done | Circuit breaker (trips after 10 errors, drains requests, flips `/ready` to 503), transient-error retry with backoff, graceful SIGINT shutdown, readiness probe. |
| **Deployment Docs** | ✅ Done | Troubleshooting guide, production checklist, env var table, SLO/alerting docs. |
| **API Compatibility** | ✅ Done | Request cancellation (`POST /v1/cancel` + `X-Request-Id`), priority queuing (0–100), request timeouts (streaming + non-streaming), `tools`/`functions` parameter support. |

---

## Three-Milestone Roadmap

### Milestone 1: Production Blockers
**Window:** Weeks 1–8 (Oct 12 – Nov 27, 2026)  
**Theme:** Stabilize the runtime and remove product blockers.

**Goals:**
- Reliable serving on single and multi-GPU hardware
- Truthful feature claims and clear documentation
- Strong test coverage for critical paths
- Clear baseline for production deployments

**Included Issues:**
1. Implement Distributed Inference (TP/PP)
2. Expand Test Suite & Coverage Gates
3. Audit & Complete Quantization Support

**Exit Criteria:**
- ✅ Distributed inference works end-to-end on 2+ GPUs; throughput scales >80%
- ✅ Critical module coverage exceeds 75%; scheduler >90%
- ✅ All advertised quantization paths are either real or explicitly unsupported
- ✅ Engineering team can deploy stable baseline without blocking issues

**Owners:** Distributed Systems, QA/Testing, Model Optimization

---

### Milestone 2: High-Value Features
**Window:** Weeks 5–12 (Nov 16, 2026 – Jan 8, 2027)  
**Theme:** Unlock the most important capabilities that drive adoption.

**Goals:**
- Convert partially implemented modules into active features
- Expand model ecosystem to 3+ architectures
- Production-grade observability and visibility
- Clear, actionable dashboards and alerting

**Included Issues:**
4. Integrate LoRA Support
5. Integrate Speculative Decoding
6. Expand Model Ecosystem (Mistral, Qwen, etc.)
7. Complete Observability & Tracing

**Exit Criteria:**
- ✅ LoRA weights load, apply, and work with mixed-adapter workloads
- ✅ Speculative decoding produces correct outputs with >1.5x measured speedup
- ✅ At least 3 model families run end-to-end; architecture trait established
- ✅ Structured logs, trace IDs, OTLP export, and Grafana dashboard in place

**Owners:** Model Adaptation, Performance Optimization, Model Support, DevOps/Observability

---

### Milestone 3: Production Hardening & Ops
**Window:** Weeks 9–16 (Jan 4 – Feb 5, 2027)  
**Theme:** Operational maturity and deployment readiness.

**Goals:**
- Graceful failure modes and resilience patterns
- Comprehensive deployment and troubleshooting guidance
- Full API compatibility with OpenAI ecosystem
- Clear operational model and SLO targets

**Included Issues:**
8. Improve Error Handling & Resilience
9. Production Deployment Guide & Checklist
10. Enhanced API Compatibility

**Exit Criteria:**
- ✅ Readiness probes, graceful shutdown, retry logic, circuit breakers implemented
- ✅ Production deployment guide, Kubernetes examples, security best practices documented
- ✅ API supports functions/tools and priority queueing
- ✅ SLOs and alerting rules are published and actionable

**Owners:** Runtime/DevOps, Documentation, API/Platform

---

## Execution Details

### Issue 1: Distributed Inference (Tensor & Pipeline Parallelism)
**Effort:** 4–6 weeks | **Owner:** Distributed Systems  
**Acceptance Criteria:**
- TP/PP work end-to-end on 2+ GPUs
- Throughput scaling >80% with 2 GPUs
- Integration test passes in CI
- Documentation covers configuration and examples

**Key Tasks:**
- Implement NCCL initialization and AllReduce primitives
- Implement tensor parallelism: weight sharding, forward/backward splits, AllReduce
- Implement pipeline parallelism: layer assignment, activation checkpointing
- Add unit tests for sharding and communication patterns
- Add 2–4 GPU integration test
- Benchmark single-GPU vs. multi-GPU throughput and latency

---

### Issue 2: Test Suite & Coverage Gates
**Effort:** 2–3 weeks | **Owner:** QA/Testing  
**Acceptance Criteria:**
- >75% coverage on scheduler/cache modules
- All critical tests pass in CI
- Coverage badge added to README

**Key Tasks:**
- Add unit tests for `continuous_batching.rs`, `block_manager.rs`, `radix_cache.rs`
- Add concurrency, cancellation, timeout integration tests
- Enforce CI gate: `--fail-under-lines 75`
- Add coverage badge

---

### Issue 3: Quantization Audit & Completion
**Effort:** 2–3 weeks | **Owner:** Model Optimization  
**Acceptance Criteria:**
- All advertised quantization paths are real or unsupported
- GGUF end-to-end example runs
- Quantization benchmarks published
- Compatibility matrix documented

**Key Tasks:**
- Audit and document which quantization modes are real vs. mock
- Either implement AWQ/FP8 fully or remove from README
- Add GGUF loading example
- Add quantization benchmarks
- Write `docs/quantization.md` with compatibility matrix

---

### Issue 4: LoRA Integration
**Effort:** 2–3 weeks | **Owner:** Model Adaptation  
**Acceptance Criteria:**
- LoRA weights load and apply correctly
- Multi-request mixed-adapter workload is stable
- Request-level adapter selection works
- Example runs end-to-end

**Key Tasks:**
- Complete `src/model/lora.rs`: weight loading, merging
- Integrate LoRA projections into model forward pass
- Add scheduler tracking for active adapter per request
- Add API parameter for adapter selection
- Write LoRA example and unit tests

---

### Issue 5: Speculative Decoding Integration
**Effort:** 2–3 weeks | **Owner:** Performance Optimization  
**Acceptance Criteria:**
- Outputs are identical to non-speculative baseline
- Measured speedup >1.5x (claim is 2x)
- API parameter enables/disables speculative decoding

**Key Tasks:**
- Implement draft model loading and management
- Add speculative decoding logic to worker sampling loop
- Implement verification and rejection handling
- Add API parameter `use_speculative: bool`
- Write example and benchmarks

---

### Issue 6: Model Ecosystem Expansion
**Effort:** 3–4 weeks | **Owner:** Model Support  
**Acceptance Criteria:**
- At least 3 model families work end-to-end
- Architecture trait defined and adopted for all models
- Compatibility matrix published

**Key Tasks:**
- Define `ModelArchitecture` trait
- Refactor Llama to use trait
- Implement Mistral/Mixtral and Qwen architectures
- Add examples for each architecture
- Write architecture addition guide

---

### Issue 7: Observability & Tracing
**Effort:** 2–3 weeks | **Owner:** DevOps/Observability  
**Acceptance Criteria:**
- Structured logging with trace IDs and lifecycle events
- Optional OTLP exporter
- Grafana dashboard
- SLO definitions and alerting rules

**Key Tasks:**
- Add `tracing` instrumentation to request lifecycle
- Implement optional OTLP exporter
- Create/improve Grafana dashboard
- Write SLO definitions in `docs/slos.md`
- Write alert templates and runbooks

---

### Issue 8: Error Handling & Resilience
**Effort:** 1–2 weeks | **Owner:** Runtime/DevOps  
**Acceptance Criteria:**
- Complete readiness probe
- Graceful shutdown drains in-flight requests
- Circuit breaker for scheduler failures
- Standardized error codes

**Key Tasks:**
- Implement readiness probe
- Add graceful shutdown handler
- Add circuit breaker
- Implement retry logic with backoff
- Standardize error codes and messages

---

### Issue 9: Production Deployment Guide
**Effort:** 1–2 weeks | **Owner:** DevOps/Documentation  
**Acceptance Criteria:**
- Comprehensive deployment guide published
- Kubernetes manifests provided
- Security best practices documented
- Troubleshooting guide covers common issues

**Key Tasks:**
- Write `docs/deployment.md` with hardware, config, tuning
- Write `docs/kubernetes.md` with K8s examples
- Write `docs/security.md` with best practices
- Enhance `docs/troubleshooting.md`
- Create Docker Compose examples

---

### Issue 10: Enhanced API Compatibility
**Effort:** 2–3 weeks | **Owner:** API/Platform  
**Acceptance Criteria:**
- `functions`/`tools` support is compatible with OpenAI
- Priority queueing is implemented
- Request management is complete (cancellation, timeouts)
- Versioning and compatibility strategy is documented

**Key Tasks:**
- Add schema validation for functions/tools
- Implement function calling compatibility
- Add priority field and queue in scheduler
- Verify request cancellation and timeout handling
- Write API versioning guide

---

## Dependencies & Parallelization

**Critical Path:**
- Milestone 1 must complete before Milestone 2 starts (shared infrastructure)
- Milestone 2 items can parallelize (different teams)
- Milestone 3 can start in parallel with late Milestone 2 work

**Key Handoffs:**
- Milestone 1 → Milestone 2: Stable runtime, working CI infrastructure
- Milestone 2 → Milestone 3: Feature stability, clarity on error paths

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Multi-GPU scaling fails | Medium | Critical | Early CI testing; load tests with 4+ GPUs |
| Test coverage regressions | Medium | High | Enforce coverage gates; expand scheduler tests |
| LoRA/speculative features conflict | Low | Medium | Integration tests for feature combinations |
| Quantization implementation incomplete | Low–Medium | Medium | Audit AWQ/FP8; complete or mark unsupported |
| Silent worker failures in production | Medium | High | Enhanced logging, health probes, SLO dashboards |

---

## Success Criteria: End of Roadmap

- ✅ Multi-GPU serving is production-ready with >80% scaling efficiency
- ✅ Feature claims are truthful; unsupported features are explicitly labeled
- ✅ Critical modules have >75% test coverage; scheduler >90%
- ✅ LoRA and speculative decoding work end-to-end with verified performance gains
- ✅ At least 3 model families are supported
- ✅ Observability includes structured logs, trace IDs, OTLP export, and actionable dashboards
- ✅ Graceful degradation, circuit breakers, and retry logic are in place
- ✅ Production deployment is documented with Kubernetes examples, security guide, and troubleshooting
- ✅ API is compatible with OpenAI ecosystem (tools, priority queueing, versioning)

---

## How to Use This Roadmap

1. **Create GitHub Issues:** See `docs/ISSUES_BACKLOG.md` for detailed issue templates.
2. **Create Milestones:** Create three GitHub milestones mapping to the dates above.
3. **Assign Owners:** Assign each issue to the responsible team/engineer.
4. **Track Progress:** Update issue status and link dependencies.
5. **Update Backlog:** As work progresses, update this document and the backlog.

---

## References

- Detailed issue backlog: `docs/ISSUES_BACKLOG.md`
- Original gap analysis: `IMPROVEMENT_PLAN.md` (historical reference)
- Implementation status: `docs/implementation_status.md`
