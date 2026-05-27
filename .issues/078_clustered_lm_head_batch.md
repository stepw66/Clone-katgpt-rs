# clustered_lm_head: batch per-token dot products via simd_matmul_rows

## Status
⏸️ **DEFERRED** — Requires profiling to confirm dispatch overhead is significant. Optimization depends on actual cluster layout; scattered clusters may negate benefits of batching. Code has TODO comment documenting the optimization path.

## Severity
🟡 MEDIUM — activated for large vocabularies

## Location
`src/transformer.rs:1335-1350` (`clustered_lm_head` Stage 2)

## Problem
The clustered LM head computes logits for selected cluster tokens via individual `simd_dot_f32` calls:
```rust
for &token_idx in cluster_tokens {
    let row_off = token_idx * n_embd;
    let dot = crate::simd::simd_dot_f32(
        &lm_head[row_off..row_off + n_embd],
        &hidden[..n_embd],
        n_embd,
    );
    unsafe { *logits.get_unchecked_mut(token_idx) = dot; }
}
```

Each call has function dispatch overhead. For large clusters (e.g., top-5 clusters covering 50% of vocab = 10K+ tokens), this means 10K+ individual function calls.

## Proposed Fix
Gather the selected token rows into a contiguous buffer and use `simd_matmul_rows` for batch processing:

```rust
// Gather selected row offsets into contiguous weight slice
// Then use simd_matmul_rows on the gathered weights
let selected_rows: Vec<usize> = selected_clusters.iter()
    .flat_map(|&c| cluster_map[c].iter().filter(|&&t| t < vocab_size).copied())
    .collect();

// Batch matmul: compute all selected logits at once
for (out_idx, &token_idx) in selected_rows.iter().enumerate() {
    // Still need per-row due to sparse output, but could group by cluster
    // for better sequential weight access
}
```

Alternatively, if clusters are contiguous in vocab (or can be made so), use `simd_matmul_rows` directly on the cluster weight slice.

**Note**: This optimization depends on the cluster layout. If clusters are scattered, the gather itself may negate benefits. Profile before implementing.

## Estimated Impact
- **10-30% faster** LM head for large clusters
- Reduced function call overhead
- Better instruction cache behavior

## Acceptance Criteria
- [ ] Profile clustered LM head to confirm dispatch overhead is significant
- [ ] If confirmed, batch dot products using `simd_matmul_rows` where possible
- [ ] All clustered LM head tests pass
