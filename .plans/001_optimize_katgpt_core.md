# Plan 001: Optimize katgpt-core

## Scope
Optimize hot-path code in `crates/katgpt-core/src/` following `.contexts/optimization.md` guidelines.

## Analysis

### Identified Optimizations

#### 1. `irrep_pruner.rs` — O(k) linear scan in `is_in_top_k` (HOT)
- [x] Replace `is_in_top_k` O(k) linear scan with O(1) bitmap lookup
- Current: for each `is_valid` call, scans up to `valid_count` sorted indices
- Fix: pre-compute `HashSet` or bitmap of top-k indices in `set_logits`, use O(1) contains check
- Impact: `batch_is_valid` calls `is_in_top_k` per candidate → O(candidates × k) → O(candidates) with bitmap

#### 2. `irrep_pruner.rs` — Full sort for top-k (TODO already noted)
- [x] Replace `sort_unstable_by` with `select_nth_unstable` for partial sort
- Current: O(n log n) full sort when only need top-k
- Fix: `select_nth_unstable_by` gives top-k in O(n) average
- Impact: vocab sizes up to 256k → significant for each `set_logits` call

#### 3. `irrep_pruner.rs` — FFT planner allocation per call
- [x] Cache FFT planner in struct to avoid reallocation
- Current: `FftPlanner::new()` allocates per `spectral_flatness` call
- Fix: store planner in `IrrepPruner` struct, reuse across calls

#### 4. `slod.rs` — `sq_dist` scalar when `simd_dist_sq` exists
- [x] Replace scalar `sq_dist` with `crate::simd::simd_dist_sq`
- Already SIMD-accelerated version exists in simd.rs

#### 5. `slod.rs` — `zscore` allocates Vec
- [x] Add `zscore_into` with pre-allocated buffer, use in `build_laplacian`
- `build_laplacian` calls `zscore` multiple times → allocation per call

#### 6. `slod.rs` — `mad_peak_picker` sorts composite twice for median/MAD
- [x] Use `select_nth_unstable` for median (O(n) vs O(n log n))

#### 7. `linoss.rs` — `nearest_token` brute-force scan
- [x] Use `simd_dot_f32` instead of scalar zip-sum for dot product
- Also avoid sigmoid computation — just compare raw dots (sigmoid is monotonic)

#### 8. `linoss.rs` — `parallel_scan` allocates 7 vectors per call
- [x] Pre-allocate scratch in struct or accept scratch buffers
- Current: allocates a,b,c,d,bias_y,bias_z + pa,pb,pc,pd,pby,pbz = 12 vectors per call

#### 9. `attention.rs` — `s_tile` initialized per k_tile iteration
- [x] Hoist `s_tile` outside k_tile loop, only clear needed region

#### 10. `slod.rs` — `build_laplacian` hot-path allocation
- [x] Accept pre-allocated scratch buffers instead of allocating Vecs internally
- Multiple `vec![0.0; n]` allocations inside loops

#### 11. `simd.rs` — `simd_gram_f32` potential cache optimization
- [x] Already well-implemented; no changes needed

## Not Changing (already optimal)
- `simd.rs` dot products: already 4-accumulator unrolled, SIMD-accelerated
- `coda.rs` fused kernels: already eliminate intermediate writes, use `get_unchecked`
- `roofline.rs`: compute-only, not hot path
- `dirichlet.rs`: thin wrapper around simd kernels
- `shard_embedding.rs`: already uses simd_dot_f32
- `parallax_attn.rs`: already uses column-sum factorization
