use anyhow::Result;
use kyro::distributed::DistributedContext;
use kyro::model::loader::{LoadedModel, ModelLoader};
use kyro::scheduler::block_manager::BlockManager;
use kyro::scheduler::continuous_batching::Scheduler;
use kyro::worker::Worker;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Boots the Kyro engine: initializes logging, loads the model, starts the
/// worker loop (wired to the shared readiness flag), and serves the API.
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(Level::INFO.into()),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting Kyro LLM Engine...");

    // 1. Hardware detection
    let device = kyro::device::get_device()?;
    info!("Using device: {:?}", device);

    // 2. Initialize Distributed Context, Block Manager, Scheduler, and Metrics
    let dist = Arc::new(DistributedContext::new());
    info!(
        "Distributed context initialized (Rank: {}, World Size: {})",
        dist.rank, dist.world_size
    );

    let block_manager = BlockManager::new(16, 1024, 256);
    let scheduler_cfg = kyro::scheduler::continuous_batching::SchedulerConfig::default();
    let scheduler = Arc::new(Mutex::new(Scheduler::new(block_manager, scheduler_cfg)));
    let notify = Arc::new(Notify::new());

    let registry = Arc::new(prometheus::Registry::new());
    let metrics = kyro::metrics::EngineMetrics::new(&registry)?;
    info!("Scheduler and Metrics initialized.");

    // 3. Centralized, validated configuration
    let config = kyro::config::AppConfig::from_env_and_args()?;
    info!("Configuration: {:?}", config);

    let tokenizer_path = config.tokenizer_path.clone();
    let tokenizer = match &tokenizer_path {
        Some(path) => {
            let tok = kyro::api::tokenizer::LuminaTokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", path, e))?;
            info!("Tokenizer loaded from {}", path);
            Some(Arc::new(tok))
        }
        None => {
            info!("No tokenizer path configured; raw token IDs will be returned.");
            None
        }
    };

    // 4. Load model through ModelLoader when a path is configured; otherwise use the dummy model.
    let loaded_model = match &config.model_path {
        Some(path) => {
            let loader = ModelLoader::new(path)
                .map_err(|e| anyhow::anyhow!("Invalid model path '{}': {}", path, e))?;
            let model = loader
                .load(&device, dist)
                .map_err(|e| anyhow::anyhow!("Failed to load model from '{}': {}", path, e))?;
            info!("Model loaded from {}", path);
            model
        }
        None => {
            info!("No model path configured; running with dummy model.");
            let cfg = kyro::model::config::LlamaConfig::llama_7b();
            LoadedModel::Standard(kyro::model::llama::LlamaModel::dummy(&cfg)?)
        }
    };

    // 5. Start Worker Loop
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    ready.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut worker = Worker::new(loaded_model, scheduler.clone(), device, metrics.clone())
        .with_ready(ready.clone());
    let worker_notify = notify.clone();
    tokio::spawn(async move {
        if let Err(e) = worker.run_loop(worker_notify).await {
            tracing::error!("Worker loop failed: {:?}", e);
        }
    });

    // 6. Start API Server
    let registry_arc = registry.clone();
    let app_state = Arc::new(
        kyro::api::openai::AppState::new(scheduler, notify, tokenizer, config.model_name.clone())
            .with_metrics(metrics.clone(), registry_arc.clone())
            .with_limits(
                config.max_tokens_cap,
                config.max_prompt_bytes,
                config.max_messages,
            )
            .with_readiness(ready)
            .with_timeout(std::time::Duration::from_secs(config.request_timeout_secs)),
    );
    let app = kyro::api::openai::app(app_state);
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Kyro API serving on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("Shutdown signal received; draining in-flight requests");
        })
        .await?;

    Ok(())
}
