#![allow(dead_code)]

use crate::metrics::EngineMetrics;
use crate::model::loader::LoadedModel;
use crate::scheduler::continuous_batching::Scheduler;
use candle_core::{Device, Result, Tensor};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};

/// Stops the worker loop after repeated consecutive failures so a
/// persistent fault (OOM, device error) does not spin the engine.
pub struct CircuitBreaker {
    max_consecutive_errors: u32,
    consecutive_errors: u32,
    tripped: bool,
}

impl CircuitBreaker {
    /// Creates a breaker that trips after `max_consecutive_errors` failures
    /// in a row (floored at 1).
    pub fn new(max_consecutive_errors: u32) -> Self {
        Self {
            max_consecutive_errors: max_consecutive_errors.max(1),
            consecutive_errors: 0,
            tripped: false,
        }
    }

    /// Resets the consecutive error count after a successful iteration.
    pub fn record_success(&mut self) {
        self.consecutive_errors = 0;
    }

    /// Records a failure. Returns true when the breaker trips.
    pub fn record_error(&mut self) -> bool {
        self.consecutive_errors += 1;
        if self.consecutive_errors >= self.max_consecutive_errors {
            self.tripped = true;
        }
        self.tripped
    }

    /// Returns whether the breaker has tripped.
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    /// Returns the current count of consecutive failures.
    pub fn consecutive_errors(&self) -> u32 {
        self.consecutive_errors
    }
}

pub struct Worker {
    pub model: LoadedModel,
    pub scheduler: Arc<Mutex<Scheduler>>,
    pub device: Device,
    pub metrics: Arc<EngineMetrics>,
    breaker: CircuitBreaker,
    ready: Option<Arc<AtomicBool>>,
}

impl Worker {
    pub fn new(
        model: LoadedModel,
        scheduler: Arc<Mutex<Scheduler>>,
        device: Device,
        metrics: Arc<EngineMetrics>,
    ) -> Self {
        Self {
            model,
            scheduler,
            device,
            metrics,
            breaker: CircuitBreaker::new(10),
            ready: None,
        }
    }

    /// Share the engine readiness flag so the breaker can take the
    /// engine out of rotation when it trips.
    pub fn with_ready(mut self, ready: Arc<AtomicBool>) -> Self {
        self.ready = Some(ready);
        self
    }

    /// Consecutive iteration failures before the breaker trips.
    pub fn with_error_threshold(mut self, max_consecutive_errors: u32) -> Self {
        self.breaker = CircuitBreaker::new(max_consecutive_errors);
        self
    }

    /// Runs the worker loop, ticking the scheduler/model until the circuit
    /// breaker trips and the loop returns an error.
    pub async fn run_loop(&mut self, notify: Arc<Notify>) -> anyhow::Result<()> {
        loop {
            match self.tick(&notify).await {
                Ok(TickOutcome::Idle) => {}
                Ok(TickOutcome::Processed) => {
                    self.breaker.record_success();
                }
                Err(e) => {
                    self.metrics.worker_errors_total.inc();
                    let tripped = self.breaker.record_error();
                    tracing::warn!(
                        error = ?e,
                        consecutive = self.breaker.consecutive_errors(),
                        "worker iteration failed"
                    );
                    if tripped {
                        tracing::error!(
                            "circuit breaker tripped after {} consecutive errors; stopping worker",
                            self.breaker.consecutive_errors()
                        );
                        self.metrics.worker_tripped.set(1.0);
                        if let Some(ready) = &self.ready {
                            ready.store(false, Ordering::SeqCst);
                        }
                        {
                            let mut scheduler = self.scheduler.lock().await;
                            let ids: Vec<u64> = scheduler
                                .running_queue
                                .iter()
                                .map(|r| r.id)
                                .chain(scheduler.waiting_queue.iter().map(|r| r.id))
                                .collect();
                            // Dropping each Request drops its token_sender, ending client streams.
                            for id in ids {
                                scheduler.cancel_request(id);
                            }
                        }
                        return Err(anyhow::anyhow!(
                            "circuit breaker tripped after {} consecutive errors",
                            self.breaker.consecutive_errors()
                        ));
                    }
                    // Transient failure: brief backoff before retrying.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Runs a single scheduling/inference iteration, returning whether work
    /// was processed or the scheduler was idle.
    async fn tick(&mut self, notify: &Arc<Notify>) -> anyhow::Result<TickOutcome> {
        // Phase 1: Schedule (short lock hold)
        let (to_prefill, _, mut work_batch) = {
            let mut scheduler = self.scheduler.lock().await;
            let (to_prefill, to_decode) = scheduler.schedule();

            if to_prefill.is_empty() && to_decode.is_empty() {
                drop(scheduler);
                notify.notified().await;
                return Ok(TickOutcome::Idle);
            }

            // Extract all work items upfront - clone data, don't hold lock during compute
            let mut batch = Vec::new();

            for req_id in &to_prefill {
                if let Some(req) = scheduler.running_queue.iter().find(|r| r.id == *req_id) {
                    let chunk_start = req.prefill_cursor;
                    let chunk_end = (chunk_start
                        + crate::scheduler::continuous_batching::PREFILL_CHUNK_SIZE)
                        .min(req.prompt_tokens.len());

                    if chunk_start < req.prompt_tokens.len() {
                        let chunk_tokens: Vec<u32> =
                            req.prompt_tokens[chunk_start..chunk_end].to_vec();
                        batch.push(WorkItem {
                            req_id: *req_id,
                            input: chunk_tokens,
                            is_last_chunk: chunk_end == req.prompt_tokens.len(),
                            temperature: req.temperature,
                            top_p: req.top_p,
                            is_prefill: true,
                            grammar: req.grammar_processor.as_ref().map(|g| g.constraint.clone()),
                        });
                    } else if req.cached_prefix_len == req.prompt_tokens.len() {
                        let last_token: u32 = req.prompt_tokens.last().copied().unwrap_or(0);
                        batch.push(WorkItem {
                            req_id: *req_id,
                            input: vec![last_token],
                            is_last_chunk: true,
                            temperature: req.temperature,
                            top_p: req.top_p,
                            is_prefill: true,
                            grammar: req.grammar_processor.as_ref().map(|g| g.constraint.clone()),
                        });
                    }
                }
            }

            for req_id in &to_decode {
                if let Some(req) = scheduler.running_queue.iter().find(|r| r.id == *req_id) {
                    let last_token: u32 = req.generated_tokens.last().copied().unwrap_or(0);
                    batch.push(WorkItem {
                        req_id: *req_id,
                        input: vec![last_token],
                        is_last_chunk: false,
                        temperature: req.temperature,
                        top_p: req.top_p,
                        is_prefill: false,
                        grammar: req.grammar_processor.as_ref().map(|g| g.constraint.clone()),
                    });
                }
            }

            (to_prefill, to_decode, batch)
        }; // Lock released here - compute can overlap with new request insertion

        // Phase 2: Execute model inference (no lock held)
        let mut results = Vec::with_capacity(work_batch.len());
        for item in work_batch.drain(..) {
            let input = Tensor::new(item.input.as_slice(), &self.device)?
                .unsqueeze(0)?
                .to_dtype(candle_core::DType::U32)?;

            let logits = match &mut self.model {
                LoadedModel::Standard(m) => m.forward(&input, 0)?,
                LoadedModel::Quantized(q) => q.forward(&input, 0)?,
            };

            let next_token = if !logits.dims().is_empty() && logits.dims()[0] > 0 {
                let logits = match &item.grammar {
                    Some(crate::api::grammar::GrammarConstraint::Json) => {
                        let mut proc = crate::api::grammar::GrammarLogitsProcessor::new(
                            item.grammar.clone().unwrap(),
                        );
                        let vocab = logits.dims().last().copied().unwrap_or(0);
                        match proc.apply_grammar_mask(&last_pos(&logits)?, vocab) {
                            Ok(masked) => masked,
                            Err(_) => return Err(anyhow::anyhow!("grammar mask failed")),
                        }
                    }
                    _ => last_pos(&logits)?,
                };
                self.sample(&logits, item.temperature, item.top_p)?
            } else {
                rand::rng().random_range(0..100)
            };

            results.push(ComputeResult {
                req_id: item.req_id,
                token: next_token,
                is_last_chunk: item.is_last_chunk,
            });
        }

        // Phase 3: Update scheduler state (short lock hold)
        {
            let mut scheduler = self.scheduler.lock().await;

            for res in results {
                if let Some(req) = scheduler
                    .running_queue
                    .iter_mut()
                    .find(|r| r.id == res.req_id)
                {
                    req.generated_tokens.push(res.token);
                    if let Some(sender) = &req.token_sender {
                        let _ = sender.send(res.token);
                    }
                    self.metrics.total_tokens_generated.inc();
                }
            }

            // Advance prefills and cleanup finished requests
            for req_id in &to_prefill {
                scheduler.advance_prefill_cursor(*req_id);
            }

            let finished: Vec<u64> = scheduler
                .running_queue
                .iter()
                .filter(|r| r.generated_tokens.len() >= r.max_tokens)
                .map(|r| r.id)
                .collect();

            for id in finished {
                scheduler.finish_request(id);
            }
        }

        tokio::task::yield_now().await;

        Ok(TickOutcome::Processed)
    }

    fn sample_last_position(&self, logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
        // logits may be [vocab], [seq, vocab], or [batch, seq, vocab]; sample
        // from the last sequence position, which corresponds to the prediction
        // for the next token.
        let logits = match logits.dims().len() {
            1 => logits.clone(),
            2 => logits.get(logits.dims()[0] - 1)?,
            _ => {
                let seq = logits.dims()[1];
                logits.get(0)?.get(seq - 1)?
            }
        };
        self.sample(&logits, temperature, top_p)
    }

    fn sample(&self, logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
        let dims = logits.dims();

        if dims.is_empty() || dims.iter().all(|&d| d == 0) {
            return Ok(rand::rng().random_range(0..100));
        }

        let logits = if dims.len() == 1 {
            logits.clone()
        } else {
            logits.get(0)?
        };

        let logits = match logits.flatten_all() {
            Ok(l) => l,
            Err(_) => return Ok(rand::rng().random_range(0..100)),
        };
        let logits = logits.to_dtype(candle_core::DType::F32)?;

        if temperature <= 0.0 {
            return if logits.dims()[0] > 0 {
                Ok(logits.argmax(0)?.to_scalar::<u32>()?)
            } else {
                Ok(rand::rng().random_range(0..100))
            };
        }

        let logits = (&logits / (temperature as f64))?;
        let prs = candle_nn::ops::softmax(&logits, 0)?;
        let prs: Vec<f32> = match prs.to_vec1() {
            Ok(p) if !p.is_empty() => p,
            _ => return Ok(rand::rng().random_range(0..100)),
        };

        if top_p < 1.0 {
            let mut indexed_prs: Vec<(usize, f32)> = prs.into_iter().enumerate().collect();
            indexed_prs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0;
            let mut cut_off = indexed_prs.len();
            for (i, (_, p)) in indexed_prs.iter().enumerate() {
                cumsum += p;
                if cumsum > top_p {
                    cut_off = i + 1;
                    break;
                }
            }
            indexed_prs.truncate(cut_off);

            let total_p: f32 = indexed_prs.iter().map(|(_, p)| p).sum();
            let mut rng = rand::rng();
            let mut r: f32 = rng.random::<f32>() * total_p;

            for (id, p) in &indexed_prs {
                r -= p;
                if r <= 0.0 {
                    return Ok(*id as u32);
                }
            }
            Ok(indexed_prs[0].0 as u32)
        } else {
            let mut rng = rand::rng();
            let mut r: f32 = rng.random::<f32>();
            for (id, &p) in prs.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return Ok(id as u32);
                }
            }
            Ok((prs.len() - 1) as u32)
        }
    }
}

struct WorkItem {
    req_id: u64,
    input: Vec<u32>,
    is_last_chunk: bool,
    temperature: f32,
    top_p: f32,
    is_prefill: bool,
    grammar: Option<crate::api::grammar::GrammarConstraint>,
}

fn last_pos(logits: &Tensor) -> Result<Tensor> {
    match logits.dims().len() {
        1 => Ok(logits.clone()),
        2 => logits.get(logits.dims()[0] - 1),
        _ => {
            let seq = logits.dims()[1];
            logits.get(0)?.get(seq - 1)
        }
    }
}

struct ComputeResult {
    req_id: u64,
    token: u32,
    is_last_chunk: bool,
}

enum TickOutcome {
    Idle,
    Processed,
}

#[cfg(test)]
mod breaker_tests {
    use super::*;

    /// Breaker trips once the consecutive error count reaches the threshold.
    #[test]
    fn trips_after_threshold() {
        let mut cb = CircuitBreaker::new(3);
        assert!(!cb.record_error());
        assert!(!cb.record_error());
        assert!(cb.record_error());
        assert!(cb.is_tripped());
        assert_eq!(cb.consecutive_errors(), 3);
    }

    /// A success resets the consecutive error count back to zero.
    #[test]
    fn success_resets_counter() {
        let mut cb = CircuitBreaker::new(3);
        cb.record_error();
        cb.record_error();
        cb.record_success();
        assert_eq!(cb.consecutive_errors(), 0);
        assert!(!cb.record_error());
        assert!(!cb.record_error());
        assert!(cb.record_error());
        assert!(cb.is_tripped());
    }

    /// A threshold of 0 is floored to 1, so a single error trips the breaker.
    #[test]
    fn threshold_floor_of_one() {
        let mut cb = CircuitBreaker::new(0);
        assert!(cb.record_error());
        assert!(cb.is_tripped());
    }

    /// Once tripped, the breaker remains tripped on further errors.
    #[test]
    fn stays_tripped() {
        let mut cb = CircuitBreaker::new(2);
        cb.record_error();
        cb.record_error();
        assert!(cb.is_tripped());
        cb.record_error();
        assert!(cb.is_tripped());
    }
}
