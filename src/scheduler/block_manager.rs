#![allow(dead_code)]

use crate::scheduler::radix_cache::RadixCache;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

pub struct PhysicalBlock {
    pub id: BlockId,
    pub device_id: usize,
    pub ref_count: usize,
}

pub struct BlockManager {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub num_cpu_blocks: usize,
    pub free_gpu_blocks: Vec<BlockId>,
    pub free_cpu_blocks: Vec<BlockId>,
    pub block_table: HashMap<u64, Vec<BlockId>>, // request_id -> block_ids
    pub radix_cache: RadixCache,
    // Tracks the prompt token sequence for each request (needed for cache insertion on free)
    pub prompt_table: HashMap<u64, Vec<u32>>,
    /// Tracks the reference count of each physical block
    pub ref_counts: Vec<usize>,
}

impl BlockManager {
    pub fn new(block_size: usize, num_gpu_blocks: usize, num_cpu_blocks: usize) -> Self {
        // Reserve 30% of GPU blocks as the Radix Cache's eviction budget
        let cache_capacity = num_gpu_blocks / 3;
        let free_gpu_blocks = (0..num_gpu_blocks).map(BlockId).collect();
        let free_cpu_blocks = (0..num_cpu_blocks).map(BlockId).collect();
        Self {
            block_size,
            num_gpu_blocks,
            num_cpu_blocks,
            free_gpu_blocks,
            free_cpu_blocks,
            block_table: HashMap::new(),
            radix_cache: RadixCache::new(cache_capacity),
            prompt_table: HashMap::new(),
            ref_counts: vec![0; num_gpu_blocks],
        }
    }

    /// Allocates blocks for a request, first checking the Radix Cache for any
    /// prefix hit to avoid recomputation. Returns (allocated_blocks, cached_token_count).
    pub fn allocate_with_prefix(
        &mut self,
        request_id: u64,
        prompt_tokens: &[u32],
    ) -> Option<(Vec<BlockId>, usize)> {
        // 1. Check Radix Cache for prefix hit
        let (cached_blocks, cached_token_count) = self.radix_cache.match_prefix(prompt_tokens);

        // Increment ref count for cached blocks
        for block in &cached_blocks {
            self.ref_counts[block.0] += 1;
        }

        // 2. Calculate how many NEW blocks we need (beyond the cache hit)
        let remaining_tokens = prompt_tokens.len().saturating_sub(cached_token_count);
        let new_blocks_needed = remaining_tokens.div_ceil(self.block_size);

        // 3. Check we have enough free GPU blocks for the remainder
        // First try to evict from the radix cache if needed
        let mut evicted: Vec<(Vec<u32>, Vec<BlockId>)> = Vec::new();
        if self.free_gpu_blocks.len() < new_blocks_needed {
            evicted = self.radix_cache.evict_lru();
            for (_tokens, blocks) in &evicted {
                for block in blocks {
                    self.ref_counts[block.0] -= 1;
                    if self.ref_counts[block.0] == 0 {
                        self.free_gpu_blocks.push(*block);
                    }
                }
            }
        }

        if self.free_gpu_blocks.len() < new_blocks_needed {
            // Roll back ref counts on failure
            for block in &cached_blocks {
                self.ref_counts[block.0] -= 1;
            }
            // Restore evicted cache entries so a failed allocation does not
            // silently drop cached prefixes.
            for (tokens, blocks) in evicted {
                for block in &blocks {
                    self.ref_counts[block.0] += 1;
                    if let Some(pos) = self.free_gpu_blocks.iter().position(|&b| b == *block) {
                        self.free_gpu_blocks.remove(pos);
                    }
                }
                self.radix_cache.insert_unchecked(&tokens, &blocks);
            }
            return None; // Out of memory even after eviction
        }

        // 4. Allocate new blocks for the uncached suffix
        let mut all_blocks = cached_blocks;
        for _ in 0..new_blocks_needed {
            let block = self.free_gpu_blocks.pop().unwrap();
            self.ref_counts[block.0] = 1; // 1 for the request
            all_blocks.push(block);
        }

        self.block_table.insert(request_id, all_blocks.clone());
        self.prompt_table.insert(request_id, prompt_tokens.to_vec());

        Some((all_blocks, cached_token_count))
    }

    /// Legacy allocate (no prefix caching). Kept for compatibility.
    pub fn allocate(&mut self, request_id: u64, num_tokens: usize) -> Option<Vec<BlockId>> {
        let num_blocks = num_tokens.div_ceil(self.block_size);
        if self.free_gpu_blocks.len() < num_blocks {
            return None;
        }
        let mut allocated = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block = self.free_gpu_blocks.pop().unwrap();
            self.ref_counts[block.0] = 1;
            allocated.push(block);
        }
        self.block_table.insert(request_id, allocated.clone());
        Some(allocated)
    }

    /// Frees a request's blocks back into the Radix Cache (not the free pool directly).
    /// The LRU eviction policy manages when blocks are truly released.
    pub fn free(&mut self, request_id: u64) {
        if let Some(blocks) = self.block_table.remove(&request_id) {
            let tokens = self.prompt_table.remove(&request_id);

            for block in &blocks {
                self.ref_counts[block.0] -= 1;
            }

            if let Some(tokens) = tokens {
                // Deposit blocks into the Radix Cache for future prefix reuse
                // We increment ref counts because the cache now 'owns' a reference
                for block in &blocks {
                    self.ref_counts[block.0] += 1;
                }
                self.radix_cache.insert(&tokens, &blocks);
            } else {
                // If no token trace, release blocks with ref count 0 to free pool
                for block in blocks {
                    if self.ref_counts[block.0] == 0 {
                        self.free_gpu_blocks.push(block);
                    }
                }
            }
        }
    }

    /// GPU block utilization in percent (0..=100). Blocks held by
    /// requests or by the Radix Cache both count as used.
    pub fn kv_cache_usage_percent(&self) -> f64 {
        if self.num_gpu_blocks == 0 {
            return 0.0;
        }
        let used = self.num_gpu_blocks - self.free_gpu_blocks.len();
        (used as f64 / self.num_gpu_blocks as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_free() {
        let mut bm = BlockManager::new(16, 8, 4);
        let blocks = bm.allocate(1, 32).expect("should allocate");
        assert_eq!(blocks.len(), 2);
        bm.free(1);
        let blocks2 = bm.allocate(2, 128).expect("should reuse freed blocks");
        assert!(blocks2.len() >= 8);
    }

    #[test]
    fn returns_none_when_out_of_blocks() {
        let mut bm = BlockManager::new(16, 1, 0);
        assert!(bm.allocate(1, 16 * 64).is_none());
    }

    #[test]
    fn prefix_cache_hit_on_second_request() {
        let mut bm = BlockManager::new(4, 32, 0);
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (blocks, cached) = bm.allocate_with_prefix(1, &prompt).unwrap();
        assert_eq!(cached, 0);
        assert_eq!(blocks.len(), 2);
        bm.free(1);
        let (_blocks2, cached2) = bm.allocate_with_prefix(2, &prompt).unwrap();
        assert!(cached2 > 0, "expected a prefix cache hit, got {}", cached2);
    }

    #[test]
    fn failed_allocation_leaves_manager_usable() {
        let mut bm = BlockManager::new(4, 4, 0);
        let prompt = vec![1, 2, 3, 4];
        bm.allocate_with_prefix(1, &prompt).unwrap();
        bm.free(1);
        let huge = vec![9u32; 16 * 64];
        assert!(bm.allocate_with_prefix(3, &huge).is_none());
        // Manager must still serve a valid allocation afterwards.
        assert!(bm.allocate_with_prefix(2, &prompt[..4]).is_some());
    }

    #[test]
    fn failed_allocation_preserves_cache_and_refcounts() {
        let mut bm = BlockManager::new(4, 4, 0);
        let prompt = vec![1, 2, 3, 4];
        bm.allocate_with_prefix(1, &prompt).unwrap();
        bm.free(1);
        let cached_before = bm.radix_cache.num_cached_blocks;
        let refcounts_before: Vec<usize> = bm.ref_counts.clone();

        let huge = vec![9u32; 16 * 64];
        assert!(bm.allocate_with_prefix(3, &huge).is_none());

        assert_eq!(bm.ref_counts, refcounts_before);
        assert_eq!(bm.radix_cache.num_cached_blocks, cached_before);
        assert_eq!(bm.free_gpu_blocks.len(), 3);
        // Cache still serves the original prefix.
        let (_, cached) = bm.allocate_with_prefix(2, &prompt).unwrap();
        assert!(cached > 0);
    }

    #[test]
    fn free_unknown_request_is_noop() {
        let mut bm = BlockManager::new(4, 4, 0);
        bm.free(999);
        assert_eq!(bm.free_gpu_blocks.len(), 4);
    }

    #[test]
    fn kv_usage_zero_when_empty() {
        let bm = BlockManager::new(16, 100, 0);
        assert_eq!(bm.kv_cache_usage_percent(), 0.0);
    }

    #[test]
    fn kv_usage_counts_allocated_blocks() {
        let mut bm = BlockManager::new(16, 100, 0);
        bm.allocate(1, 16 * 25).unwrap(); // 25 blocks
        let usage = bm.kv_cache_usage_percent();
        assert!((usage - 25.0).abs() < 1e-6, "got {}", usage);
        // Legacy allocate leaves no token trace, so free returns
        // blocks to the free pool.
        bm.free(1);
        assert_eq!(bm.kv_cache_usage_percent(), 0.0);
    }

    #[test]
    fn kv_usage_counts_cached_blocks() {
        let mut bm = BlockManager::new(4, 100, 0);
        let prompt = vec![1u32; 8];
        bm.allocate_with_prefix(1, &prompt).unwrap();
        bm.free(1);
        // Blocks are deposited into the radix cache and still count as used.
        let usage = bm.kv_cache_usage_percent();
        assert!(usage > 0.0, "expected cached blocks to count as used");
    }
}
