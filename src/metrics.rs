#![allow(dead_code)]

use anyhow::Result;
use prometheus::{Counter, Gauge, Histogram, Registry};
use std::sync::Arc;

pub struct EngineMetrics {
    pub total_requests: Counter,
    pub requests_by_model: prometheus::CounterVec,
    pub queue_depth: Gauge,
    pub cache_hits: Gauge,
    pub cache_misses: Gauge,
    pub total_tokens_generated: Counter,
    pub token_latency: Histogram,
    pub time_to_first_token: Histogram,
    pub time_between_tokens: Histogram,
    pub kv_cache_usage: Gauge,
    pub worker_errors_total: Counter,
    pub worker_tripped: Gauge,
}

impl EngineMetrics {
    /// Registers all engine metrics (requests, tokens, latency, cache,
    /// and worker circuit-breaker gauges/counters) with `registry`.
    pub fn new(registry: &Registry) -> Result<Arc<Self>> {
        let total_requests = prometheus::register_counter_with_registry!(
            "kyro_requests_total",
            "Total number of requests processed",
            registry
        )?;

        let total_tokens_generated = prometheus::register_counter_with_registry!(
            "kyro_tokens_total",
            "Total number of tokens generated",
            registry
        )?;

        let token_latency = prometheus::register_histogram_with_registry!(
            "kyro_token_latency_seconds",
            "Latency per token generation",
            registry
        )?;

        let time_to_first_token = prometheus::register_histogram_with_registry!(
            "kyro_ttft_ms",
            "Time to first token",
            registry
        )?;

        let time_between_tokens = prometheus::register_histogram_with_registry!(
            "kyro_tbt_ms",
            "Time between tokens",
            registry
        )?;

        let kv_cache_usage = prometheus::register_gauge_with_registry!(
            "kyro_kv_cache_usage_percent",
            "KV cache utilization percentage",
            registry
        )?;

        let requests_by_model = prometheus::register_counter_vec_with_registry!(
            "kyro_requests_by_model_total",
            "Total requests, labeled by served model name",
            &["model"],
            registry
        )?;

        let queue_depth = prometheus::register_gauge_with_registry!(
            "kyro_queue_depth",
            "Number of requests waiting or running in the scheduler",
            registry
        )?;

        let cache_hits = prometheus::register_gauge_with_registry!(
            "kyro_prefix_cache_hits_total",
            "Number of requests that reused a cached prefix",
            registry
        )?;

        let cache_misses = prometheus::register_gauge_with_registry!(
            "kyro_prefix_cache_misses_total",
            "Number of requests that required a cold prefill",
            registry
        )?;

        let worker_errors_total = prometheus::register_counter_with_registry!(
            "kyro_worker_errors_total",
            "Total number of worker loop iteration errors",
            registry
        )?;

        let worker_tripped = prometheus::register_gauge_with_registry!(
            "kyro_worker_circuit_breaker_tripped",
            "Whether the worker circuit breaker is tripped (1) or not (0)",
            registry
        )?;

        Ok(Arc::new(Self {
            total_requests,
            requests_by_model,
            queue_depth,
            cache_hits,
            cache_misses,
            total_tokens_generated,
            token_latency,
            time_to_first_token,
            time_between_tokens,
            kv_cache_usage,
            worker_errors_total,
            worker_tripped,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All metrics, including the worker error/circuit-breaker gauges, are
    /// registered and can be updated and gathered from the registry.
    #[test]
    fn metrics_register_and_update() {
        let registry = Registry::new();
        let m = EngineMetrics::new(&registry).unwrap();

        m.total_requests.inc();
        m.total_tokens_generated.inc_by(5.0);
        m.queue_depth.set(3.0);
        m.cache_hits.set(1.0);
        m.cache_misses.set(2.0);
        m.kv_cache_usage.set(42.0);
        m.requests_by_model.with_label_values(&["kyro"]).inc();
        m.time_to_first_token.observe(12.0);
        m.time_between_tokens.observe(3.0);
        m.token_latency.observe(0.5);

        let gathered = registry.gather();
        let names: Vec<_> = gathered.iter().map(|mf| mf.name().to_string()).collect();
        for expected in [
            "kyro_requests_total",
            "kyro_tokens_total",
            "kyro_queue_depth",
            "kyro_prefix_cache_hits_total",
            "kyro_prefix_cache_misses_total",
            "kyro_kv_cache_usage_percent",
            "kyro_requests_by_model_total",
            "kyro_ttft_ms",
            "kyro_tbt_ms",
            "kyro_token_latency_seconds",
            "kyro_worker_errors_total",
            "kyro_worker_circuit_breaker_tripped",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "missing metric {}",
                expected
            );
        }
    }

    #[test]
    fn duplicate_registration_fails() {
        let registry = Registry::new();
        let _m = EngineMetrics::new(&registry).unwrap();
        assert!(EngineMetrics::new(&registry).is_err());
    }
}
