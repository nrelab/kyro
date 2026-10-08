use criterion::{criterion_group, criterion_main, Criterion};
use kyro::scheduler::block_manager::BlockManager;
use kyro::scheduler::continuous_batching::{Request, Scheduler, SchedulerConfig};

fn scheduler_bench(c: &mut Criterion) {
    c.bench_function("schedule_32_requests", |b| {
        b.iter(|| {
            let bm = BlockManager::new(16, 4096, 1024);
            let mut sched = Scheduler::new(bm, SchedulerConfig::default());
            for i in 0..32u64 {
                sched.add_request(Request {
                    id: i,
                    prompt_tokens: vec![1; 128],
                    generated_tokens: Vec::new(),
                    max_tokens: 8,
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
            sched.schedule()
        })
    });
}

criterion_group!(benches, scheduler_bench);
criterion_main!(benches);
