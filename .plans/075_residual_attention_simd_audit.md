# Plan 075: SIMD Residual + Attention Audit (Issue 062)

## Status: COMPLETED

## Summary

Wire SIMD primitives into all remaining scalar loops in `src/transformer.rs`, `src/types.rs`, and `src/speculative/sampling.rs`. Issue 057/058 already added `simd_dot_f32`, `simd_scale_inplace`, and related kernels — this plan closes the gap on residual adds, embedding adds, raven attention, and softmax max-finding.

## Tasks

- [x] T1: Add `simd_add_inplace(dst: &mut [f32], src: &[f32])` to `src/simd.rs` — NEON `vaddq_f32`, AVX2 `_mm256_add_ps`
- [x] T2: Add `simd_add_into(dst: &mut [f32], a: &[f32], b: &[f32])` to `src/simd.rs` — zip add a+b into dst
- [x] T3: Add `simd_max_f32(x: &[f32]) -> f32` to `src/simd.rs` — NEON `vmaxq_f32`, AVX2 `_mm256_max_ps` + horizontal reduction
- [x] T4: Add `simd_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32)` to `src/simd.rs` — dst = decay*dst + write*src
- [x] T5: Wire `simd_add_into` into embedding add loops (5 forward variants + prefill multi-layer init)
- [x] T6: Wire `simd_add_inplace` into all residual add loops (attn + MLP × 5 forward variants + domain_latent injection)
- [x] T7: Wire `simd_dot_f32` into `raven_readout_into` Q·K dot product
- [x] T8: Wire `simd_fused_decay_write` into `raven_update` decay/write loop
- [x] T9: Wire `simd_scale_inplace` into `sample_residual_distribution_into` normalize
- [x] T10: Wire `simd_max_f32` into `softmax` and `softmax_scaled` pass 1
- [x] T11: Add unit tests for all new SIMD kernels (`simd_add_inplace`, `simd_add_into`, `simd_max_f32`, `simd_fused_decay_write`)
- [x] T12: Add benchmark `bench_residual_simd_062` for component-level timing
- [x] T13: Run `cargo clippy --fix --allow-dirty`, fix warnings, run full test suite

## Architecture

### New SIMD Primitives (src/simd.rs)

| Function | NEON | AVX2 | Scalar Fallback |
|----------|------|------|-----------------|
| `simd_add_inplace(dst, src)` | `vaddq_f32` | `_mm256_add_ps` | `dst[i] += src[i]` |
| `simd_add_into(dst, a, b)` | `vaddq_f32` | `_mm256_add_ps` | `dst[i] = a[i] + b[i]` |
| `simd_max_f32(x) -> f32` | `vmaxq_f32` + reduce | `_mm256_max_ps` + reduce | scalar cmp |
| `simd_fused_decay_write(dst, decay, src, write)` | `vfmaq_f32` | `_mm256_fmadd_ps` | scalar FMA |

### Call Site Inventory

**Embedding Add (`simd_add_into`)** — 1× per forward variant:
- `forward_base` L552-558
- `forward_paged` L1475-1481
- `forward_raven` L2138-2144
- `forward_turboquant` L2323-2329
- `forward_prefill` L1090-1098 (multi-layer), L1112-1118 (single-layer)

**Residual Add (`simd_add_inplace`)** — 2× per forward variant × n_layer:
- `forward_base` L639-643 (attn), L691-695 (MLP)
- `forward_paged` L1529-1533 (attn), L1575-1579 (MLP)
- `forward_raven` L2230-2234 (attn), L2276-2280 (MLP)
- `forward_turboquant` L2408-2412 (attn), L2454-2458 (MLP)
- `forward_prefill` L1228-1232 (attn), L1280-1284 (MLP)

**Domain Latent (`simd_add_inplace`)** — gated behind `domain_latent` feature:
- `forward_base` L590-593 (k += dl.embedding, v += dl.embedding)
- `forward_prefill` L1141-1144 (k += dl.embedding, v += dl.embedding)

**Raven Readout (`simd_dot_f32`)** — per-head per-layer:
- `raven_readout_into` L2054-2060 (Q·K dot loop)

**Raven Update (`simd_fused_decay_write`)** — per-slot:
- `raven_update` L2023-2026 (keys), L2027-2030 (values)

**Softmax Max (`simd_max_f32`)** — every softmax call:
- `softmax` L615-619 (pass 1 max-finding)
- `softmax_scaled` L652-656 (pass 1 max-finding)

**Residual Normalize (`simd_scale_inplace`)** — speculative sampling:
- `sample_residual_distribution_into` L39-41 (normalize scratch)

## Files to Modify

| File | Changes |
|------|---------|
| `src/simd.rs` | Add `simd_add_inplace`, `simd_add_into`, `simd_max_f32`, `simd_fused_decay_write` + tests |
| `src/transformer.rs` | Wire SIMD into embedding/residual/raven loops |
| `src/types.rs` | Wire `simd_max_f32` into `softmax` / `softmax_scaled` |
| `src/speculative/sampling.rs` | Wire `simd_scale_inplace` into normalize |
| `tests/bench_residual_simd_062.rs` | Component benchmark (new) |

## Expected Improvement

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| residual add (n_embd=64) | scalar 64× add | NEON 16 ops / AVX2 8 ops | ~4-8× fewer inst |
| embedding add (n_embd=64) | scalar 64× add | NEON 16 ops / AVX2 8 ops | ~4-8× fewer inst |
| raven_readout Q·K (kv_dim=64) | manual dot loop | `simd_dot_f32` | proven +25% from 057 |
| raven_update (kv_dim=64) | scalar FMA | SIMD fused | ~4-8× fewer inst |
| softmax max (block_size=128) | scalar cmp | SIMD max | ~4-8× fewer inst |
| residual normalize (vocab=256) | scalar 256× mul | `simd_scale_inplace` | already proven |

## Related

- Issue 062 (residual attention SIMD) — issue closed + removed
- `.agent/optimization.md` — hot-path patterns reference
- Issue 057 — SIMD dot + zero-alloc (CLOSED, +25-32% forward)
- Issue 058 — SIMD scale + extract paths (CLOSED)