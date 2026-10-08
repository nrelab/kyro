# Kyro SLOs & Alerting Guide

Recommended Service Level Objectives and Prometheus alerting rules for production Kyro deployments.

## Service Level Objectives

| SLO | Target | Measurement |
|-----|--------|-------------|
| Availability | 99.9% | `GET /health` success rate over 30 days |
| TTFT p99 | < 2s | `kyro_ttft_ms` histogram, prompts ≤ 4K tokens |
| TBT p99 | < 100ms | `kyro_tbt_ms` histogram |
| Request success rate | > 99% | Non-5xx responses / total requests |
| Queue depth | < 50 | `kyro_queue_depth` sustained |
| KV cache usage | < 90% | `kyro_kv_cache_usage_percent` (updated per worker iteration and on `/metrics` scrape) |

## Prometheus Alerting Rules

Add to your Prometheus rule groups:

```yaml
groups:
  - name: kyro
    rules:
      - alert: KyroHighErrorRate
        expr: |
          sum(rate(kyro_requests_total{status=~"5.."}[5m]))
            / sum(rate(kyro_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kyro 5xx error rate above 1%"

      - alert: KyroHighTTFT
        expr: |
          histogram_quantile(0.99, sum(rate(kyro_ttft_ms_bucket[5m])) by (le)) > 2000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Kyro p99 TTFT exceeds 2s"

      - alert: KyroHighTBT
        expr: |
          histogram_quantile(0.99, sum(rate(kyro_tbt_ms_bucket[5m])) by (le)) > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Kyro p99 TBT exceeds 100ms"

      - alert: KyroQueueBacklog
        expr: kyro_queue_depth > 50
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Kyro scheduler queue depth above 50"

      - alert: KyroKVCachePressure
        expr: kyro_kv_cache_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kyro KV cache usage above 90%"

      - alert: KyroWorkerLoopDown
        expr: rate(kyro_tokens_total[5m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Kyro worker appears stalled (no tokens generated)"

      - alert: KyroWorkerCircuitBreakerTripped
        expr: kyro_worker_circuit_breaker_tripped == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Kyro worker circuit breaker tripped; engine is not ready"

      - alert: KyroWorkerErrorBurst
        expr: rate(kyro_worker_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kyro worker logging repeated iteration errors"
```

## Grafana Dashboard

Import `deploy/grafana-dashboard.json` into Grafana for a pre-built view of:
- Request and token throughput
- TTFT / TBT percentiles
- Queue depth and KV cache pressure
- Prefix cache hit rate
- Per-model request rates

## Error Budget

With a 99.9% availability SLO over 30 days, the error budget is ~43.2 minutes of downtime. Track burn rate:

```promql
# Fast burn: budget exhausted in ~2 days
1 - (sum(rate(kyro_requests_total{status!~"5.."}[1h])) / sum(rate(kyro_requests_total[1h]))) > 0.0025
```
