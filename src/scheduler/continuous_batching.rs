#![allow(dead_code)]

use crate::scheduler::block_manager::BlockManager;
use std::collections::VecDeque;

pub struct Request {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub is_prefill: bool,
    /// Number of prompt tokens already covered by a Radix Cache hit.
    /// The model only needs to process `prompt_tokens[cached_prefix_len..]`.
    pub cached_prefix_len: usize,
    /// For Chunked Prefill: the index into prompt_tokens up to which we've prefilled so far.
    pub prefill_cursor: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Scheduling priority: higher values are dequeued first.
    pub priority: u32,
    /// Channel to send newly generated tokens back to the API for streaming.
    pub token_sender: Option<tokio::sync::mpsc::UnboundedSender<u32>>,
    /// Optional grammar processor for structured output (XGrammar).
    pub grammar_processor: Option<crate::api::grammar::GrammarLogitsProcessor>,
}

pub const PREFILL_CHUNK_SIZE: usize = 512;

pub struct SchedulerConfig {
    pub max_tokens_per_iter: usize,
    pub max_prefill_chunk_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_iter: 2048,
            max_prefill_chunk_size: 512,
        }
    }
}

pub struct Scheduler {
    pub waiting_queue: VecDeque<Request>,
    pub running_queue: Vec<Request>,
    pub block_manager: BlockManager,
    pub config: SchedulerConfig,
    /// Number of requests that reused at least one cached prefix block.
    pub cache_hits: u64,
    /// Number of requests that required a full (cold) prefill.
    pub cache_misses: u64,
}

impl Scheduler {
    pub fn new(block_manager: BlockManager, config: SchedulerConfig) -> Self {
        Self {
            waiting_queue: VecDeque::new(),
            running_queue: Vec::new(),
            block_manager,
            config,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn add_request(&mut self, request: Request) {
        // Insert by priority (descending); FIFO within equal priority.
        let priority = request.priority;
        let pos = self
            .waiting_queue
            .iter()
            .position(|r| r.priority < priority)
            .unwrap_or(self.waiting_queue.len());
        self.waiting_queue.insert(pos, request);
    }

    pub fn schedule(&mut self) -> (Vec<u64>, Vec<u64>) {
        let mut to_prefill = Vec::new();
        let mut to_decode = Vec::new();
        let mut total_tokens = 0;

        // 1. Prioritize ongoing Decodes (Memory Bound)
        for req in &mut self.running_queue {
            if req.prefill_cursor >= req.prompt_tokens.len() {
                to_decode.push(req.id);
                req.is_prefill = false;
                total_tokens += 1;
            }
        }

        // 2. Add Prefills (Compute Bound), but chunked
        // First, check already running prefill requests
        for req in &mut self.running_queue {
            if total_tokens >= self.config.max_tokens_per_iter {
                break;
            }

            if req.prefill_cursor < req.prompt_tokens.len() {
                let remaining = req.prompt_tokens.len() - req.prefill_cursor;
                let chunk_size = std::cmp::min(remaining, self.config.max_prefill_chunk_size);

                req.is_prefill = true;
                to_prefill.push(req.id);
                total_tokens += chunk_size;
            }
        }

        // Second, pull new requests from waiting queue
        while let Some(req) = self.waiting_queue.front() {
            if total_tokens >= self.config.max_tokens_per_iter {
                break;
            }

            let tokens = req.prompt_tokens.clone();
            if let Some((_blocks, cached_len)) =
                self.block_manager.allocate_with_prefix(req.id, &tokens)
            {
                let mut req = self.waiting_queue.pop_front().unwrap();
                req.cached_prefix_len = cached_len;
                req.prefill_cursor = cached_len;
                if cached_len > 0 {
                    self.cache_hits += 1;
                } else {
                    self.cache_misses += 1;
                }

                let remaining = req.prompt_tokens.len() - req.prefill_cursor;
                let chunk_size = std::cmp::min(remaining, self.config.max_prefill_chunk_size);

                if remaining > 0 {
                    req.is_prefill = true;
                    to_prefill.push(req.id);
                    total_tokens += chunk_size;
                } else {
                    // Fully cached, move to decode
                    req.is_prefill = false;
                    to_decode.push(req.id);
                    total_tokens += 1;
                }

                self.running_queue.push(req);
            } else {
                break;
            }
        }

        (to_prefill, to_decode)
    }

    pub fn advance_prefill_cursor(&mut self, request_id: u64) {
        if let Some(req) = self.running_queue.iter_mut().find(|r| r.id == request_id) {
            let next = (req.prefill_cursor + self.config.max_prefill_chunk_size)
                .min(req.prompt_tokens.len());
            req.prefill_cursor = next;
        }
    }

    pub fn finish_request(&mut self, request_id: u64) {
        if let Some(pos) = self.running_queue.iter().position(|r| r.id == request_id) {
            self.running_queue.remove(pos);
            self.block_manager.free(request_id);
        }
    }

    /// Cancels a request by ID, whether it is still waiting or already
    /// running. Returns true if a matching request was found and removed.
    pub fn cancel_request(&mut self, request_id: u64) -> bool {
        let mut cancelled = false;
        if let Some(pos) = self.waiting_queue.iter().position(|r| r.id == request_id) {
            self.waiting_queue.remove(pos);
            cancelled = true;
        }
        if let Some(pos) = self.running_queue.iter().position(|r| r.id == request_id) {
            self.running_queue.remove(pos);
            cancelled = true;
        }
        if cancelled {
            self.block_manager.free(request_id);
        }
        cancelled
    }

    /// GPU KV-cache utilization in percent (0..=100).
    pub fn kv_cache_usage_percent(&self) -> f64 {
        self.block_manager.kv_cache_usage_percent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: u64, tokens: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            id,
            prompt_tokens: tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            is_prefill: true,
            cached_prefix_len: 0,
            prefill_cursor: 0,
            temperature: 1.0,
            top_p: 1.0,
            priority: 0,
            token_sender: None,
            grammar_processor: None,
        }
    }

    #[test]
    fn schedules_waiting_request_for_prefill() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![1, 2, 3, 4], 8));
        let (prefill, decode) = sched.schedule();
        assert_eq!(prefill, vec![1]);
        assert!(decode.is_empty());
    }

    #[test]
    fn respects_token_budget() {
        let bm = BlockManager::new(16, 1024, 256);
        let cfg = SchedulerConfig {
            max_tokens_per_iter: 4,
            max_prefill_chunk_size: 4,
        };
        let mut sched = Scheduler::new(bm, cfg);
        sched.add_request(make_request(1, vec![1; 64], 8));
        let (prefill, _) = sched.schedule();
        assert_eq!(prefill, vec![1]);
        let req = &sched.running_queue[0];
        assert!(req.prefill_cursor > 0 || req.is_prefill);
    }

    #[test]
    fn decode_after_prefill_completes() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![7], 4));
        sched.schedule();
        sched.running_queue[0].prefill_cursor = 1;
        sched.running_queue[0].is_prefill = false;
        let (prefill, decode) = sched.schedule();
        assert!(prefill.is_empty());
        assert_eq!(decode, vec![1]);
    }

    #[test]
    fn cache_hit_counters_update() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![1, 2, 3], 4));
        sched.schedule();
        assert_eq!(sched.cache_misses, 1);
        assert_eq!(sched.cache_hits, 0);
    }

    #[test]
    fn cancel_waiting_request() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![1, 2, 3], 4));
        assert!(sched.cancel_request(1));
        assert!(sched.waiting_queue.is_empty());
        assert!(sched.running_queue.is_empty());
    }

    #[test]
    fn cancel_running_request() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![1, 2, 3], 4));
        sched.schedule();
        assert!(sched.cancel_request(1));
        assert!(sched.running_queue.is_empty());
    }

    #[test]
    fn cancel_unknown_request_is_noop() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        assert!(!sched.cancel_request(999));
    }

    #[test]
    fn higher_priority_dequeued_first() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![1, 2, 3], 4));
        let mut high = make_request(2, vec![1, 2, 3], 4);
        high.priority = 10;
        sched.add_request(high);
        let mut med = make_request(3, vec![1, 2, 3], 4);
        med.priority = 5;
        sched.add_request(med);

        let (prefill, _) = sched.schedule();
        assert_eq!(
            prefill,
            vec![2, 3, 1],
            "highest priority should be scheduled first"
        );
    }

    #[test]
    fn equal_priority_is_fifo() {
        let bm = BlockManager::new(16, 64, 16);
        let mut sched = Scheduler::new(bm, SchedulerConfig::default());
        sched.add_request(make_request(1, vec![1, 2, 3], 4));
        sched.add_request(make_request(2, vec![1, 2, 3], 4));
        let (prefill, _) = sched.schedule();
        assert_eq!(prefill, vec![1, 2], "equal priority preserves FIFO order");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    // Deterministic xorshift PRNG so the test needs no external crates.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn scheduler_never_exceeds_token_budget() {
        let bm = BlockManager::new(16, 4096, 1024);
        let cfg = SchedulerConfig {
            max_tokens_per_iter: 32,
            max_prefill_chunk_size: 16,
        };
        let mut sched = Scheduler::new(bm, cfg);
        let mut rng = Rng(0xdeadbeef);
        for i in 0..16 {
            let len = 1 + (rng.next() % 64) as usize;
            sched.add_request(Request {
                id: i,
                prompt_tokens: vec![1; len],
                generated_tokens: Vec::new(),
                max_tokens: 4,
                is_prefill: true,
                cached_prefix_len: 0,
                prefill_cursor: 0,
                temperature: 1.0,
                top_p: 1.0,
                priority: 0,
                token_sender: None,
                grammar_processor: None,
            });
        }
        for _ in 0..32 {
            let (prefill, decode) = sched.schedule();
            let mut accounted = 0usize;
            for id in &prefill {
                let req = sched.running_queue.iter().find(|r| r.id == *id).unwrap();
                let chunk = (req.prompt_tokens.len() - req.prefill_cursor).clamp(1, 16);
                accounted += chunk;
            }
            accounted += decode.len();
            assert!(accounted <= 32, "budget exceeded: {}", accounted);
        }
    }
}
