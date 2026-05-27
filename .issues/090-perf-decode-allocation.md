# 🔴 Perf: Eliminate heap allocations from decode hot path

## Summary
Multiple components allocate `Vec` on every decode token or per (head × layer) during autoregressive generation. Heap allocation latency (~50-100ns each) directly adds to per-token latency. With 6+ allocations per token across 8-32 layers, this accounts for ~500ns-1µs of avoidable overhead.

## Affected Paths

### 1. `katgpt-core/types.rs` — `sample_token()` (L1635)
**Issue**: Allocates CDF `Vec::with_capacity(n)` every decode step. The zero-alloc `sample_token_into()` already exists but the allocating variant is still callable.
**Fix**: Add doc warning on `sample_token()` pointing to `sample_token_into()`. Audit all call sites to ensure `_into` variant is used in decode loops.

### 2. `dash_attn/forward.rs` — summaries collection (L160)
**Issue**: `let summaries: Vec<&Vec<f32>> = ... .collect()` allocates per decode token per layer.
**Fix**: Use an iterator directly or a stack-allocated array (n_chunks is bounded and small).

### 3. `dash_attn/routing.rs` — `score_blocks_entmax` (L27-34, L98, L126, L132)
**Issue**: Allocates 3+ `Vec`s per call (`RoutingScratch::new` + `.to_vec()` on results). Called once per head per layer per decode step.
**Fix**: The `_into` variant already exists — use `score_blocks_entmax_into` in `compute_routing_bias` (L155-158). Also fix `per_head.iter().map(|r| r.probs.clone()).collect()` at L161 to use `r.probs.as_slice()` references.

### 4. `sp_kv/utility_predictor.rs` — `predict()` (L72)
**Issue**: `vec![0.0f32; n_kv_heads]` per call. Called per token per layer.
**Fix**: Accept `&mut [f32]` output buffer parameter (the `_into` pattern already used elsewhere).

### 5. `ega_attn.rs` — `gate_attention` (L161)
**Issue**: `let mut gate_buf = vec![0.0; seq_len]` per call per head.
**Fix**: Accept pre-allocated `gate_buf: &mut [f32]` from caller context.

### 6. `transformer.rs` — `PagedKVCache::ensure_pages` (L3395)
**Issue**: Allocates `deficits: Vec<usize>` and `new_pages: Vec<Vec<usize>>` per token during paged decode.
**Fix**: Cache these as fields on `PagedKVCache` or use stack arrays (n_layer is typically small).

## References
- Optimization guideline: "Don't allocate inside hot loops"
- Pre-allocated scratch buffers: `&mut [T]` parameters instead of allocating inside hot loops
