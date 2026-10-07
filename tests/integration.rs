use kyro::api::openai::{self, AppState};
use kyro::api::tokenizer::LuminaTokenizer;
use kyro::metrics::EngineMetrics;
use kyro::model::loader::LoadedModel;
use kyro::model::{config::LlamaConfig, llama::LlamaModel, loader::ModelLoader};
use kyro::scheduler::block_manager::BlockManager;
use kyro::scheduler::continuous_batching::{Scheduler, SchedulerConfig};
use kyro::worker::Worker;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

/// Build a small WordLevel tokenizer saved to a temp file so tests exercise the
/// real file-based tokenizer loading path.
fn make_test_tokenizer() -> (LuminaTokenizer, tempfile::TempPath) {
    let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    vocab.insert("[UNK]".to_string(), 0);
    vocab.insert("hello".to_string(), 1);
    vocab.insert("world".to_string(), 2);
    vocab.insert("good".to_string(), 3);
    vocab.insert("morning".to_string(), 4);
    vocab.insert("user".to_string(), 5);
    vocab.insert("assistant".to_string(), 6);
    vocab.insert("what".to_string(), 7);
    vocab.insert("is".to_string(), 8);
    vocab.insert("the".to_string(), 9);
    for i in 0..200 {
        vocab.entry(format!("token_{}", i)).or_insert(100 + i);
    }

    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    let path = tempfile::NamedTempFile::new().unwrap();
    tokenizer.save(path.path(), false).unwrap();
    let tok = LuminaTokenizer::from_file(path.path()).unwrap();
    (tok, path.into_temp_path())
}

/// Spins up a scheduler, worker (sharing a readiness flag), and app state
/// backed by a dummy model, for use by the HTTP-level integration tests.
fn setup_engine() -> (Arc<AppState>, tempfile::TempPath) {
    let (tokenizer, tmp) = make_test_tokenizer();
    let block_manager = BlockManager::new(16, 1024, 256);
    let scheduler = Arc::new(Mutex::new(Scheduler::new(
        block_manager,
        SchedulerConfig::default(),
    )));
    let notify = Arc::new(Notify::new());

    let registry = prometheus::Registry::new();
    let metrics = EngineMetrics::new(&registry).unwrap();
    let cfg = LlamaConfig::llama_7b();
    let model = LoadedModel::Standard(LlamaModel::dummy(&cfg).unwrap());
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let mut worker = Worker::new(model, scheduler.clone(), candle_core::Device::Cpu, metrics)
        .with_ready(ready.clone());
    let worker_notify = notify.clone();
    tokio::spawn(async move {
        let _ = worker.run_loop(worker_notify).await;
    });

    let state = Arc::new(
        AppState::new(
            scheduler,
            notify,
            Some(Arc::new(tokenizer)),
            "kyro".to_string(),
        )
        .with_readiness(ready),
    );
    (state, tmp)
}

#[tokio::test]
async fn test_prompt_encoding() {
    let (tok, _tmp) = make_test_tokenizer();
    let ids = tok.encode("hello world").unwrap();
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn test_token_decoding() {
    let (tok, _tmp) = make_test_tokenizer();
    let text = tok.decode(&[1, 2]).unwrap();
    assert_eq!(text, "hello world");
}

#[tokio::test]
async fn test_model_loading_failure() {
    // Nonexistent path must produce a clear error, not a panic.
    let loader = ModelLoader::new("/nonexistent/model/dir");
    assert!(loader.is_err());

    // A .gguf path pointing at a real but invalid file must fail at load time.
    let path = std::env::temp_dir().join("kyro_bad_model.gguf");
    std::fs::write(&path, b"not a gguf").unwrap();
    let loader = ModelLoader::new(&path).unwrap();
    let dist = Arc::new(kyro::distributed::DistributedContext::new());
    let result = loader.load(&candle_core::Device::Cpu, dist);
    assert!(result.is_err());
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_model_name_validation() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let body = serde_json::json!({
        "model": "unsupported-model",
        "messages": [{"role": "user", "content": "hello world"}]
    });
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_non_streaming_completion() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let body = serde_json::json!({
        "model": "kyro",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hello world"}]
    });
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty());
    assert!(!content.starts_with("token_") || content.contains(' '));
    assert_eq!(json["model"], "kyro");
}

#[tokio::test]
async fn test_streaming_sse_completion() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let body = serde_json::json!({
        "model": "kyro",
        "stream": true,
        "max_tokens": 3,
        "messages": [{"role": "user", "content": "hello world"}]
    });
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("finish_reason"));
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let (state, _tmp) = setup_engine();
    // Attach a fresh registry to the state so /metrics responds 200.
    let registry = std::sync::Arc::new(prometheus::Registry::new());
    let metrics = kyro::metrics::EngineMetrics::new(&registry).unwrap();
    let state = std::sync::Arc::new(
        AppState::new(
            state.scheduler.clone(),
            state.notify.clone(),
            Some(state.tokenizer.clone().unwrap()),
            "kyro".to_string(),
        )
        .with_metrics(metrics, registry),
    );
    let app = openai::app(state);
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .uri("/metrics")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("kyro_requests_total"));
}

#[tokio::test]
async fn test_invalid_temperature_rejected() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let body = serde_json::json!({
        "model": "kyro",
        "temperature": 5.0,
        "messages": [{"role": "user", "content": "hello"}]
    });
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_json_response_format_accepted() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let body = serde_json::json!({
        "model": "kyro",
        "max_tokens": 2,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "hello world"}]
    });
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn test_models_endpoint() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .uri("/v1/models")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["data"][0]["id"], "kyro");
}

#[tokio::test]
async fn test_ready_endpoint_returns_200_when_ready() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .uri("/ready")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn test_ready_endpoint_returns_503_when_not_ready() {
    let (state, _tmp) = setup_engine();
    let not_ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let state = Arc::new(
        AppState::new(
            state.scheduler.clone(),
            state.notify.clone(),
            state.tokenizer.clone(),
            "kyro".to_string(),
        )
        .with_readiness(not_ready),
    );
    let app = openai::app(state);
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .uri("/ready")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 503);
}

#[tokio::test]
async fn test_request_timeout_returns_504() {
    let (state, _tmp) = setup_engine();
    let state = Arc::new(
        AppState::new(
            state.scheduler.clone(),
            state.notify.clone(),
            state.tokenizer.clone(),
            "kyro".to_string(),
        )
        .with_readiness(Arc::new(std::sync::atomic::AtomicBool::new(true)))
        .with_timeout(std::time::Duration::from_millis(10)),
    );
    let app = openai::app(state);
    let body = serde_json::json!({
        "model": "kyro",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hello world"}]
    });
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 504);
}

#[tokio::test]
async fn test_concurrent_requests() {
    let (state, _tmp) = setup_engine();
    let app = Arc::new(openai::app(state));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let body = serde_json::json!({
                "model": "kyro",
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "hello world"}]
            });
            tower::ServiceExt::oneshot(
                app.as_ref().clone(),
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
        }));
    }

    for handle in handles {
        assert_eq!(handle.await.unwrap(), 200);
    }
}

#[tokio::test]
async fn test_cancel_unknown_request() {
    let (state, _tmp) = setup_engine();
    let app = openai::app(state);
    let body = serde_json::json!({"request_id": 999999});
    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/cancel")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn test_cancel_inflight_request() {
    // Build an engine whose KV pool is too small for the prompt, so the
    // request deterministically stays in the waiting queue (block
    // allocation fails and the scheduler leaves it queued).
    let (tokenizer, _tmp) = make_test_tokenizer();
    let block_manager = BlockManager::new(16, 1024, 256);
    let scheduler = Arc::new(Mutex::new(Scheduler::new(
        block_manager,
        SchedulerConfig::default(),
    )));
    let notify = Arc::new(Notify::new());
    let registry = prometheus::Registry::new();
    let metrics = kyro::metrics::EngineMetrics::new(&registry).unwrap();
    let cfg = LlamaConfig::llama_7b();
    let model = LoadedModel::Standard(LlamaModel::dummy(&cfg).unwrap());
    let mut worker = Worker::new(model, scheduler.clone(), candle_core::Device::Cpu, metrics);
    let worker_notify = notify.clone();
    tokio::spawn(async move {
        let _ = worker.run_loop(worker_notify).await;
    });
    let state = Arc::new(
        AppState::new(
            scheduler.clone(),
            notify.clone(),
            Some(Arc::new(tokenizer)),
            "kyro".to_string(),
        )
        .with_readiness(Arc::new(std::sync::atomic::AtomicBool::new(true)))
        .with_limits(4096, 10 * 1024 * 1024, 256),
    );
    let app = Arc::new(openai::app(state.clone()));

    // ~20K tokens => 1250 blocks needed > 1024 available.
    let long_prompt = "hello world ".repeat(10_000);
    let app_bg = app.clone();
    let bg = tokio::spawn(async move {
        let body = serde_json::json!({
            "model": "kyro",
            "max_tokens": 4,
            "prompt": long_prompt
        });
        tower::ServiceExt::oneshot(
            app_bg.as_ref().clone(),
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
    });

    // Wait until the request is visible in the waiting queue.
    let mut request_id = None;
    for _ in 0..1000 {
        let sched = scheduler.lock().await;
        let id = sched.waiting_queue.front().map(|r| r.id);
        drop(sched);
        if let Some(id) = id {
            request_id = Some(id);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let request_id = request_id.expect("request should be queued");

    let body = serde_json::json!({"request_id": request_id});
    let response = tower::ServiceExt::oneshot(
        app.as_ref().clone(),
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/cancel")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);

    // The scheduler must no longer track the request.
    {
        let sched = scheduler.lock().await;
        assert!(sched.running_queue.is_empty());
        assert!(sched.waiting_queue.is_empty());
    }

    // Background request completes after cancellation (channel closed).
    let status = bg.await.unwrap();
    assert_eq!(status, 200);
}
