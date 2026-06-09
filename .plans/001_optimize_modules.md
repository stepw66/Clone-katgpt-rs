# 001: Optimize 11 Modules per optimization.md Guidelines

## Scope
Optimize cache_prune, dash_attn, data_probe, distill, fold, gdn2, hla, hybrid_oct_pq, iso_quant, kvarn, octopus.

## GOAT Gate
- Feature flag: `opt_001` (default OFF)
- Each optimization must pass existing tests
- Promote to default if benchmarks show gain

## Tasks

### P0 — Highest Impact (hot-path allocations, data layout)

- [x] `cache_prune/sat.rs`: Replace `&mut [Vec<f32>]` with flat `&mut [f32]` layout (TODO already noted)
- [x] `dash_attn/block_topk.rs`: `argtopk` pre-allocate pairs buffer, reuse across calls
- [x] `dash_attn/channel_aware.rs`: Pre-allocate `routing_query` in `VortexScratch`, avoid per-indexer allocation
- [x] `dash_attn/routing.rs`: `score_blocks_entmax_into` — documented scratch reuse tradeoff (active_indices/bias are small, sorted/probs are the expensive scratch to preserve)
- [x] `dash_attn/value_energy.rs`: Fuse norm computation (chunked 4-wide loop) and dot product in indexer
- [x] `dash_attn/block_topk.rs`: Chunked dot product in `forward_indexer` for auto-vectorization

### P1 — Medium Impact (pre-compute, cache reuse)

- [ ] `cache_prune/rolling_hash.rs`: `add_segment` uses `Vec::with_capacity` for hash_buf (good), but `find_matches` re-allocates `hash_buf` each call — accept as scratch param
- [ ] `data_probe/geometry.rs`: Pre-allocate covariance + centered matrices, accept scratch params
- [ ] `data_probe/markov.rs`: `stationary_distribution` already uses swap — good; `generate_markov_chain` clones transition_buf — pre-allocate candidates
- [ ] `dash_attn/chunk_summary.rs`: `mean_pool_keys_into` already zero-alloc — good
- [ ] `dash_attn/entmax.rs`: `entmax_1p5` allocating variant creates Vec — already has `_into` variant

### P2 — Lower Impact (style, minor)

- [ ] `cache_prune/sensitivity.rs`: `StrictDetector::detect` allocates Vec<bool> — add `_into` variant
- [ ] `data_probe/nll.rs`: Already has `_into` variant — good
- [ ] `data_probe/typical_set.rs`: `regime_distribution` already avoids double-compute — good

## Defer to Sub-agent Tasks (larger modules)

- [ ] `distill/ilc.rs`: kmeans inner-loop allocation, squared_distance SIMD
- [ ] `fold/chain_folder.rs`: Pre-allocate indexed/decisions scratch buffers
- [ ] `gdn2/kernel.rs`: Branch removal in hot inner loop
- [ ] `hla/kernel.rs`: SIMD transpose matvec (5 occurrences)
- [ ] `hybrid_oct_pq/kv_cache.rs`: Flatten Vec<Vec<Vec<...>>> to contiguous storage
- [ ] `iso_quant/kv_cache.rs`: Flatten Vec<Vec<...>> storage
- [ ] `kvarn/kv_cache.rs`: Per-tile Vec allocation → scratch buffer reuse
- [ ] `kvarn/var_norm.rs`: Per-iteration allocation → scratch buffer reuse
- [ ] `octopus/kv_cache.rs`: Flatten Vec<Vec<Vec<...>>> + Vec<Vec<f32>>
- [ ] `octopus/encode.rs`: Precompute oct_decode directions
