# Plan 263: Cumprodsum Primitive + Dual-Mode SSD Block Decomposition

**Date:** 2026-06-13
**Status:** 🔵 ACTIVE
**Research:** [231 Semiseparable State Space Duality](../.research/231_Semiseparable_State_Space_Duality_Modelless.md)
**Depends On:** GDN2 (Plan 105), Diagonal Gate (existing), ConstraintPruner trait

---

## Goal

Implement the **cumprodsum primitive** (atomic 1-SS matrix multiplication from SSD paper Section 3.2.2) and the **dual-mode block decomposition** (Section 6) as the unifying computation strategy for all attention/SSM variants. This is the DRY foundation: one primitive replaces GDN2 decay, LinOSS oscillation, standard cumsum, and attention mask computation.

## Why

The SSD paper proves all sequence models are semiseparable matrix multiplications. Our GDN2 with diagonal gate IS the SSD layer — we just didn't have the primitive abstraction. The cumprodsum unifies:
- `cumsum` (a=1): standard causal mask
- `cumprod` (x=0): pure exponential decay
- GDN2 diagonal decay: matrix-valued cumprodsum
- LinOSS oscillation: complex-eigenvalue cumprodsum

The block decomposition gives adaptive CPU/SIMD/GPU routing: small chunks → SIMD, large chunks → GPU tensor cores.

---

## Tasks

### Phase 1: Cumprodsum Primitive (Foundation) ✅
- [x] Create `src/cumprodsum.rs` module with:
  - `cumprodsum_scalar(a, x, h_init, out)` — O(T) scalar recurrence
  - `cumprodsum_batched(a, x, h_init, out, n_channels)` — channel-batched version
  - `segsum(a, out)` — segment sum (exp(segsum) produces 1-SS mask, per Listing 1)
  - `influence(a, from, to)` — cumulative decay product for pruning
  - `context_freshness(a)` — mean influence for adaptive CoT
- [ ] SIMD-vectorize the scalar version (process 4 or 8 channels in parallel)
- [x] Zero-allocation: all outputs written to pre-allocated `&mut [f32]` slices
- [x] Unit tests: verify against manual computation for T=8, T=64, T=256
- [x] Unit tests: verify cumsum special case (a=1), cumprod special case (x=0)
- [x] Unit tests: verify batched matches scalar, GDN2 diagonal decay equivalence
- [x] **21/21 tests passing**

### Phase 2: Dual-Mode Block Decomposition
- [ ] Create `src/ssd_block.rs` module with:
  - `SsdBlockConfig { block_len, state_dim, head_dim }` — configuration
  - `ssd_block_forward(x, a, b, c, config, out)` — the 4-step block decomposition:
    1. Diagonal blocks: quadratic attention per chunk (matmul)
    2. Right factors: input → state per chunk
    3. Center factors: inter-chunk cumprodsum recurrence
    4. Left factors: state → output per chunk
- [ ] Gate behind `ssd_block` feature flag
- [ ] Adaptive chunk-size routing based on sequence length:
  - T < 256: full quadratic (standard attention)
  - 256 ≤ T < 2048: block_len=64 (SIMD sweet spot)
  - T ≥ 2048: block_len=128 (GPU tensor core sweet spot)
- [ ] Integration test: verify SSD block output matches naive quadratic computation
- [ ] Benchmark: SSD block vs standard attention at T=512, 1024, 2048, 4096

### Phase 3: Semiseparable Pruner
- [ ] Create `src/pruners/ss_pruner.rs` implementing `ConstraintPruner`:
  - `SemiseparablePruner { decay_factors, threshold }`
  - Uses cumprodsum to compute temporal influence along token paths
  - Prunes branches where cumulative influence < threshold
- [ ] Gate behind `ss_pruner` feature flag
- [ ] Unit test: verify pruning of far-range low-influence branches
- [ ] Integration test: DDTree with vs without SS pruner — measure branching reduction

### Phase 4: Adaptive Thinking Budget
- [ ] Extend `ThinkingBandit` (or FreqBandit) with cumprodsum freshness signal:
  - `context_freshness = mean(cumprodsum(decay_factors))`
  - `thinking_budget = base + max_extra * sigmoid(beta * (freshness - threshold))`
- [ ] Before/after benchmark: thinking budget allocation on fresh vs stale context

### Phase 5: GOAT Gate Validation
- [ ] Create `examples/ssd_demo.rs` — before/after comparison:
  - Standard attention vs SSD block decomposition (timing + output match)
  - DDTree with vs without SS pruner (branching factor + quality)
  - Adaptive CoT vs fixed thinking budget (quality + latency)
- [ ] If SSD block is faster at T ≥ 512 → promote `ssd_block` to default
- [ ] If SS pruner reduces branching without quality loss → promote `ss_pruner` to default
- [ ] If SSD block is slower → demote, create issue for optimization

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Module location | `src/cumprodsum.rs`, `src/ssd_block.rs` | Modelless engine, not crate-level |
| Feature gating | `ssd_block`, `ss_pruner` | GOAT gate — prove gain before default |
| SIMD strategy | Process N channels in parallel via wide loads | Cumprodsum is per-channel independent |
| Chunk size | Adaptive: 64 for CPU, 128 for GPU | Matches SSD paper crossover benchmarks |
| Normalization | Sigmoid, never softmax | Per project rules |
| Allocation | Zero — all pre-allocated scratch buffers | Per optimization guidelines |

---

## Expected Gains

| Metric | Before | After (expected) | Why |
|--------|--------|------------------|-----|
| Attention compute @ T=2048 | O(T²N) full attention | O(TN²) block decomp | Matmul-dominated, 2-8× faster |
| DDTree branching | Full exploration | Pruned far-range | Cumprodsum influence threshold |
| Memory @ T=8192 | O(T²) KV cache | O(TN) block decomp | Constant state between chunks |
| Thinking budget | Fixed | Adaptive | Context freshness routing |

---

## CPU/GPU/ANE Routing Table

| Operation | CPU/SIMD | GPU | ANE | Threshold |
|-----------|----------|-----|-----|-----------|
| Cumprodsum (N ≤ 64) | ✅ Best | ❌ | ❌ | Always CPU |
| SSD block Q=64 | ✅ SIMD | ❌ Overhead | ❌ | T < 2048 |
| SSD block Q=128 | ✅ Slower | ✅ Tensor cores | ❌ | T ≥ 2048 + GPU available |
| Quadratic attention T<256 | ✅ | ❌ | ❌ | Always CPU |
| SS pruner | ✅ Scalar | ❌ | ❌ | Always CPU |

---

## References

- [SSD Paper](https://arxiv.org/abs/2405.21060) — Section 3.2.2 (cumprodsum), Section 6 (block decomposition)
- [Research 231](../.research/231_Semiseparable_State_Space_Duality_Modelless.md)
- [Research 070](../.research/070_Gated_DeltaNet_2_Decoupled_Erase_Write_Linear_Attention.md) — GDN2 (existing SSD implementation)
- [Plan 105](105_gated_deltanet_2_recurrent_attention.md) — GDN2 implementation plan
