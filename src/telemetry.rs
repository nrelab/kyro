//! Optional OpenTelemetry (OTLP) tracing integration.
//!
//! Compile with the `otlp` cargo feature and set `KYRO_OTLP_ENDPOINT`
//! (or `--otlp-endpoint`) to export traces to a collector.

#[cfg(feature = "otlp")]
use anyhow::Result;

/// Initialize an OTLP trace pipeline and install a global tracing
/// subscriber that bridges `tracing` spans to OpenTelemetry.
///
/// The returned provider must be kept alive for the process lifetime;
/// call `shutdown()` on it to flush pending spans.
#[cfg(feature = "otlp")]
pub fn init(
    endpoint: &str,
    service_name: &str,
) -> Result<opentelemetry_sdk::trace::SdkTracerProvider> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use opentelemetry_sdk::Resource;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_attributes(vec![KeyValue::new(
                    "service.name",
                    service_name.to_string(),
                )])
                .build(),
        )
        .build();

    let tracer = provider.tracer("kyro");
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(telemetry)
        .with(EnvFilter::from_default_env())
        .try_init()
        .map_err(|e| anyhow::anyhow!("failed to install tracing subscriber: {}", e))?;

    Ok(provider)
}

/// Without the `otlp` feature, OTLP initialization is a no-op error.
#[cfg(not(feature = "otlp"))]
pub fn init(_endpoint: &str, _service_name: &str) -> anyhow::Result<()> {
    Err(anyhow::anyhow!(
        "OTLP support not compiled in; rebuild with --features otlp"
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_without_feature_reports_error() {
        #[cfg(not(feature = "otlp"))]
        {
            assert!(super::init("http://localhost:4317", "kyro").is_err());
        }
    }
}
