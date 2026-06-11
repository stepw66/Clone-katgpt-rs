# Plan 001: Optimize katgpt-core ✅ DONE

Commit: `078f9141`

## Scope
Optimize hot-path code in `crates/katgpt-core/src/` following `.contexts/optimization.md` guidelines.

## Completed Optimizations

| # | File | Change | Impact |
|---|------|--------|--------|
| 1 | `irrep_pruner.rs` | O(1) bitmap top-k lookup | O(k) → O(1) per is_valid |
| 2 | `irrep_pruner.rs` | `select_nth_unstable` partial sort | O(n log n) → O(n) per set_logits |
| 3 | `irrep_pruner.rs` | Cached FFT planner | Eliminates planner alloc per call |
| 4 | `slod.rs` | `simd_dist_sq` for distance | Scalar → SIMD |
| 5 | `slod.rs` | `zscore_into` zero-alloc | Eliminates 3 Vec allocs per boundary_scan |
| 6 | `slod.rs` | `select_nth_unstable` for median | O(n log n) → O(n) per mad_peak_picker |
| 7 | `linoss.rs` | SIMD dot + skip sigmoid in nearest_token | SIMD + removes exp/div per token |
| 8 | `linoss.rs` | `ParallelScanScratch` reusable buffers | 12 Vec allocs eliminated per scan |
| 9 | `attention.rs` | Remove redundant s_tile re-clear | Removes BR×BC writes per k_tile |

## Not Changed (already optimal)
- `simd.rs` dot products: 4-accumulator unrolled SIMD
- `coda.rs` fused kernels: eliminate intermediate writes
- `roofline.rs`: compute-only, not hot path
- `dirichlet.rs`: thin wrapper around simd kernels
- `shard_embedding.rs`: already uses simd_dot_f32
- `parallax_attn.rs`: already uses column-sum factorization

## Not Changing (already optimal)
- `simd.rs` dot products: already 4-accumulator unrolled, SIMD-accelerated
- `coda.rs` fused kernels: already eliminate intermediate writes, use `get_unchecked`
- `roofline.rs`: compute-only, not hot path
- `dirichlet.rs`: thin wrapper around simd kernels
- `shard_embedding.rs`: already uses simd_dot_f32
- `parallax_attn.rs`: already uses column-sum factorization
