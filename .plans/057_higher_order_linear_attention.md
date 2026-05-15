# Plan 057: Higher-order Linear Attention — O(1) Inference Cache

**Branch:** `develop/feature/057_higher_order_linear_attention`
**Depends on:** Plan 010 (Multilayer Transformer), Plan 020 (Raven KV Cache — pattern reference)
**Research:** `.research/28_Higher_order_Linear_Attention.md`
**Goal:** Implement second-order HLA (symmetric + asymmetric AHLA) as an alternative to standard KV-cache attention. Achieve O(1) per-token memory independent of sequence length. Benchmark against flat KV, Raven, and TurboQuant to quantify the tradeoff: constant memory vs quality (models must be trained with HLA from scratch).

---

## Tasks

### Phase 1: Types & State

- [ ] T1: Define `HlaHeadState` struct in `src/types.rs` — symmetric second-order: `sk [hd×hd]`, `cqv [hd×hd]`, `mq [hd]`, `g [hd×hd]`, `h [hd]` + `new(hd)` + `reset()`
- [ ] T2: Define `AhlaHeadState` struct in `src/types.rs` — asymmetric second-order: `pkv [hd×hd]`, `mk [hd]`, `e [hd×hd]`, `n [hd]` + `new(hd)` + `reset()`
- [ ] T3: Define `MultiLayerHlaCache` — `layers: Vec<HlaLayerState>` where `HlaLayerState` holds per-head states (respecting GQA: SK shared per kv_group, CQV/mQ/G/h per q_head)
- [ ] T4: Define `MultiLayerAhlaCache` — same layer structure but with AHLA states (PKV/mK shared per kv_group, E/n per q_head)
- [ ] T5: Add `new()` / `reset()` for both caches, compute per-head layout from `Config`
- [ ] T6: Register `#[cfg(feature = "hla_attention")]` gate on all new types

### Phase 2: Attention Kernels

- [ ] T7: Implement `hla_state_update()` — streaming recurrence for symmetric HLA. Enforce order: compute G,h with OLD CQV,mQ BEFORE updating SK,CQV,mQ. Zero-alloc using pre-allocated temp buffers
- [ ] T8: Implement `hla_attention_head()` — readout `o_t = q_tᵀ(SK·CQV − G)`. Two-stage: matvec `u = q_tᵀ·SK` then `u·CQV − q_tᵀ·G`. Zero-alloc
- [ ] T9: Implement `ahla_step()` — combined update+readout for AHLA. `PKV += kvᵀ`, `r = qᵀPKV`, `E += kr`, `o = qᵀE`. Zero-alloc
- [ ] T10: Add optional normalization to both: `o_t / (denom + ε)` where denom uses mQ/h (symmetric) or n (AHLA)
- [ ] T11: Add optional exponential decay γ: `SK = γ·SK_prev + kkᵀ`, same for all accumulators
- [ ] T12: Verify GQA correctness — shared SK/PKV per kv_group, per-head CQV/mQ/G/h or E/n

### Phase 3: Forward Integration

- [ ] T13: Add `forward_hla()` function — same structure as `forward_base()` but uses `MultiLayerHlaCache`. Skip KV store, use `hla_state_update()` + `hla_attention_head()` instead of `attention_head()` loop
- [ ] T14: Add `forward_ahla()` function — same but uses `MultiLayerAhlaCache` and `ahla_step()`
- [ ] T15: Pre-allocate HLA-specific temp buffers in `ForwardContext` (behind feature gate): `hla_u [head_dim]`, `hla_k_cqv [head_dim]`
- [ ] T16: Ensure `generate_into()` works with both new forward functions (generic over cache type or match on attention mode)

### Phase 4: Benchmarks (Before/After)

- [ ] T17: `bench_hla_vs_flat_cache()` — compare `forward()` (flat KV) vs `forward_hla()` (symmetric) vs `forward_ahla()` (asymmetric). Measure: tok/s, µs/step, memory/layer (bytes). Run at positions 1, 16, 64, 128, 256 to show constant vs growing cost
- [ ] T18: `bench_hla_memory()` — measure total cache allocation: `std::mem::size_of_val()` for each cache type. Report bytes/layer for all configs (micro, game, bpe, small_target)
- [ ] T19: `bench_hla_quality()` — perplexity proxy: run forward on fixed prompt, compare logit divergence between SDPA and HLA/AHLA on random weights. This is a sanity check, not a quality claim (models must be trained with HLA)
- [ ] T20: Add HLA/AHLA rows to existing benchmark CSV output and timeseries

### Phase 5: Documentation & Polish

- [ ] T21: Update `README.md` — add HLA section under Architecture, note "requires training from scratch"
- [ ] T22: Update `Cargo.toml` feature flags section in README with `hla_attention`
- [ ] T23: Fix all clippy warnings under `hla_attention` feature: `cargo clippy --features hla_attention --fix --allow-dirty`
- [ ] T24: Commit with message `feat(hla): second-order linear attention — O(1) inference cache`

---

## Architecture

```text
src/types.rs
├── HlaHeadState          — symmetric 2nd-order state (SK, CQV, mQ, G, h)
├── AhlaHeadState         — asymmetric 2nd-order state (PKV, mK, E, n)
├── MultiLayerHlaCache    — per-layer, per-head HLA states (GQA-aware)
└── MultiLayerAhlaCache   — per-layer, per-head AHLA states (GQA-aware)

src/transformer.rs
├── hla_state_update()    — streaming recurrence (symmetric)
├── hla_attention_head()  — readout: qᵀ(SK·CQV − G) (symmetric)
├── ahla_step()           — combined update+readout (asymmetric)
├── forward_hla()         — full forward with HLA cache
└── forward_ahla()        — full forward with AHLA cache

src/benchmark.rs
├── bench_hla_vs_flat_cache()   — throughput comparison
├── bench_hla_memory()          — memory comparison
└── bench_hla_quality()         — logit divergence sanity check
```

### State Layout (GQA-aware)

For a config with `n_head=8`, `n_kv_head=2`, `head_dim=8`:

```text
Per KV-group (shared):  SK [8×8] = 64 floats, or PKV [8×8] = 64 floats
Per Q-head (unique):    CQV [8×8] + mQ [8] + G [8×8] + h [8] = 200 floats (symmetric)
                        E [8×8] + n [8] = 72 floats (AHLA)

Total per layer:
  Symmetric: 2 × 64 + 8 × 200 = 1728 floats = 6.9 KB
  AHLA:      2 × 64 + 8 × 72  = 704 floats  = 2.8 KB
  Flat KV:   256 × 16 × 2     = 8192 floats = 32.8 KB (at block_size=256)
```

### Forward Flow (HLA variant)

```text
1. Embedding: x = wte[token] + wpe[pos]
2. For each layer:
   a. RMSNorm → save residual → RMSNorm
   b. QKV projections (same as standard)
   c. For each head:
      - Extract q_h, k_h, v_h (respecting GQA grouping)
      - hla_state_update(state, q_h, k_h, v_h, hd)   ← UPDATE state
      - attn_out[h] = hla_attention_head(q_h, state, hd) ← READOUT
   d. Output projection + residual
   e. MLP (unchanged)
3. LM Head (unchanged)
```

---

## Key Design Decisions

1. **AHLA implemented first** — Lower state cost (O(d·dv) vs O(d²)), simpler 4-tuple, better for tiny head_dims. Symmetric HLA adds expressivity via data-dependent metric SK.
2. **Feature gate `hla_attention`** — All new code behind `#[cfg(feature = "hla_attention")]`. Default off. Zero cost when disabled.
3. **NOT a drop-in replacement** — Models trained with SDPA will produce different outputs with HLA. Document this clearly. HLA is a training architecture choice, not an inference optimization for pretrained SDPA models.
4. **Zero-alloc hot path** — All temp buffers pre-allocated in `ForwardContext`. No Vec::push or heap allocation in `hla_state_update()` or `ahla_step()`.
5. **Second-order only** — Third-order adds minimal value for head_dim 4-16 but significant complexity (3 corrected cross-summaries, segment maps). Not worth it for our configs.
6. **GQA-aware from day one** — SK/PKV shared per kv_group, CQV/E per q_head. Correct GQA layout is non-negotiable.
7. **Optional decay γ** — Exponential decay for recency bias. Default γ=1.0 (no decay). Configurable per-config.
8. **Config extends, not replaces** — Add `hla_mode: HlaMode` enum to Config: `Standard`, `Hla`, `Ahla`. No breaking change to existing configs.

### Config Extension

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum HlaMode {
    #[default]
    Standard,  // SDPA with KV cache (current behavior)
    Hla,       // Symmetric second-order HLA
    Ahla,      // Asymmetric second-order AHLA
}

// In Config:
pub hla_mode: HlaMode,              // which attention variant
pub hla_normalize: bool,            // divide by masked denominator
pub hla_decay: f32,                 // γ ∈ (0,1], default 1.0
```

---

## Expected Outcomes

### Success Criteria

1. ✅ `forward_hla()` and `forward_ahla()` compile and run without panics
2. ✅ Memory per layer is constant (does not grow with position)
3. ✅ HLA/AHLA states pass round-trip test: update then readout produces finite, non-NaN output
4. ✅ Benchmark shows constant µs/step across positions 1-256 (vs growing for flat KV)
5. ⚠️ Logit divergence from SDPA is non-trivial (expected — different operator)
6. ✅ AHLA state < symmetric HLA state < flat KV cache for all configs

### What This Proves

- ✅ O(1) inference memory is achievable with second-order prefix statistics
- ✅ The algebraic identities from the HLA paper are implementable in Rust
- ✅ GQA-aware state layout works correctly
- ✅ AHLA is the practical choice for tiny models (lower state, simpler code)

### What This Does NOT Prove

- ❌ HLA produces better outputs than SDPA (requires training from scratch)
- ❌ HLA is faster than SDPA for short sequences (O(d²) overhead at short seq_len)
- ❌ Quality parity with SDPA on pretrained weights (guaranteed to differ)

---

## Benchmark Targets

| Metric | Flat KV | Symmetric HLA | AHLA |
|--------|---------|---------------|------|
| Memory/layer (micro, hd=4) | 128 floats | 80 floats/head × 4 = 320 | 16 floats/head × 4 = 64 |
| Memory/layer (bpe, hd=8) | 4096 floats | 200 floats/head × 4 = 800 | 72 floats/head × 4 = 288 |
| µs/step at pos=1 | ~10 | ~12 (matmul overhead) | ~8 |
| µs/step at pos=256 | ~40 (O(N) scan) | ~12 (constant) | ~8 (constant) |
| Context window | block_size | ∞ (streaming) | ∞ (streaming) |

### Before/After Benchmark Matrix

```text
Config       | Position | Flat KV µs | HLA µs | AHLA µs | Flat mem | HLA mem | AHLA mem
micro (hd=4) |        1 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
micro (hd=4) |       16 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
game (hd=8)  |        1 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
game (hd=8)  |      170 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
bpe (hd=8)   |        1 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
bpe (hd=8)   |      256 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
```

(The `???` values will be filled by T17 benchmarks.)

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 010 (Multilayer) | `forward_base()` is the template for `forward_hla()` |
| Plan 020 (Raven) | Similar O(1) cache pattern but Raven is heuristic, HLA is exact algebra |
| Plan 043 (TurboQuant) | Complementary — TQ compresses KV entries, HLA eliminates them entirely |
| Plan 044 (PFlash) | Prefill parallelism — HLA chunk-parallel scan is future training work |
| Plan 050 (Feature Gate) | `hla_attention` follows same gate pattern |
| Plan 055 (MTP Drafter) | MTP could use HLA for longer draft context windows |

---

## Risks

1. **Correctness of GQA state sharing** — SK shared per kv_group means multiple heads write to the same `sk` buffer. Must verify no data race in sequential code (fine for single-threaded inference). Future: parallel head computation needs per-group SK accumulation then merge.
   - Mitigation: Unit test that verifies `forward_hla()` output matches manual single-head computation.

2. **Update order bug** — Computing G,h with NEW CQV,mQ instead of OLD produces wrong output. This is the #1 correctness trap.
   - Mitigation: Explicit comment + assertion pattern. Separate `update_cross_terms()` and `update_accumulators()` functions.

3. **Quality unknown** — We can't verify HLA quality without training from scratch. Random weights will show divergence from SDPA but that proves nothing.
   - Mitigation: Logit divergence benchmark is a sanity check (finite, non-NaN), not a quality claim.

4. **O(d²) overhead at short sequences** — For seq_len=1 and head_dim=16, SK matmul (16²=256 ops) is more expensive than single-position attention (16 ops).
   - Mitigation: Document the break-even point. For our configs (hd=4-8), HLA is competitive from the start.

5. **Feature gate combinatorics** — `hla_attention` + `domain_latent` + `sparse_mlp` combinations need to compile.
   - Mitigation: CI builds with `--all-features` and individual feature combinations.