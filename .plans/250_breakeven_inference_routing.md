# Plan 250: Breakeven Complexity Inference Routing

**Date:** 2026-06
**Status:** Active
**Research:** 218_Breakeven_Complexity_Inference_Router.md
**Feature Flag:** `breakeven_routing`

---

## Goal

Implement cost-aware inference routing based on breakeven complexity N* = B / (C_classical - C_surrogate), applying the PDE solver breakeven framework to LLM inference tier selection. The key insight: approximation methods (speculative decode, sparse attention, quantized KV) become MORE valuable as inference gets harder (longer sequences, higher QPS, complex prompts).

---

## Architecture

```
BreakevenBandit (NEW)
├── Tracks per-tier breakeven N* (upfront cost / per-token savings)
├── Observes wallclock timing per tier
├── Routes to tier that has amortized its setup cost
└── Sigmoid-gated tier transitions (not softmax)

FidelityMatcher (NEW)
├── Computes error-matched KV compression level
├── Answers: "What compression gives same perplexity as full attention at position N?"
└── Uses existing SpectralQuant / TurboQuant / ShardKV as backends

Integration
├── BreakevenBandit hooks into InferenceRouter.forward()
├── Coexists with TriggerGate (QPS), trust signal, RV gate
└── Breakeven signal OVERRIDES when N* < observed_inferences
```

---

## Tasks

### Phase 1: Core Types & Breakeven Computation

- [x] T1: Create `src/breakeven/mod.rs` with `BreakevenTierPair` enum (CpuOnly→CpuGpu, CpuGpu→CpuGpuAne, CpuOnly→Speculative)
- [x] T2: Implement `BreakevenTracker` struct tracking:
  - `upfront_cost_us: u64` — one-time setup cost for tier activation
  - `per_token_cost_us: u64` — wallclock per token at this tier
  - `per_token_cost_baseline_us: u64` — wallclock per token at lower tier
  - `total_tokens_at_tier: u64` — tokens processed at this tier
  - `breakeven_n: f64` — computed N* = upfront / (baseline - current)
- [x] T3: Implement `BreakevenTracker::is_amortized()` → bool (total_tokens > breakeven_n)
- [x] T4: Implement `BreakevenTracker::update(timing_us)` — incremental cost tracking
- [x] T5: Implement `BreakevenTracker::remaining_to_amortize()` → f64

### Phase 2: BreakevenBandit

- [x] T6: Create `BreakevenBandit` struct with per-tier-pair `BreakevenTracker` instances
- [x] T7: Implement `BreakevenBandit::select_tier(current_load, prompt_length) -> ComputeTier`:
  - If no tier has amortized → stay at current tier (avoid unnecessary promotion)
  - If a higher tier has amortized AND load warrants it → promote
  - If a higher tier has NOT amortized but load is high → promote with warning
- [x] T8: Implement `BreakevenBandit::observe_tier_timing(tier, timing_us)` — update tracker
- [x] T9: Implement sigmoid gating for tier transitions: `σ(α × (tokens - N*))` where α controls transition sharpness
- [x] T10: Implement `BreakevenBandit::reset()` — clear all trackers (e.g., on model change)

### Phase 3: FidelityMatcher (Error-Matched KV Compression)

- [x] T11: Create `FidelityMatcher` struct with:
  - Reference perplexity at each sequence position (computed once from calibration run)
  - Per-compression-level perplexity at each position (computed once)
  - Mapping: position → optimal compression level
- [x] T12: Implement `FidelityMatcher::error_matched_level(pos, target_perplexity_delta) -> CompressionLevel`
- [x] T13: Implement calibration procedure: run forward pass with full attention, record per-position perplexity as baseline
- [x] T14: Implement compression sweep: run with each compression level, record per-position perplexity delta

### Phase 4: Integration with InferenceRouter

- [x] T15: Add `BreakevenBandit` field to `InferenceRouter` (feature-gated behind `breakeven_routing`)
- [x] T16: Hook breakeven signal into `InferenceRouter::forward()`:
  - After TriggerGate evaluates tier, check breakeven override
  - If breakeven says "not amortized" AND load is borderline → defer promotion
  - If breakeven says "amortized" AND load is high → promote aggressively
- [x] T17: Hook tier timing into `BreakevenBandit::observe_tier_timing()` in the timing block at end of `forward()`
- [x] T18: Add breakeven stats to `RouterStats`:
  - `breakeven_n: Vec<(ComputeTier, f64)>` — per-tier breakeven threshold
  - `breakeven_amortized: Vec<(ComputeTier, bool)>` — which tiers are amortized

### Phase 5: Tests & Benchmarks

- [x] T19: Unit test — `BreakevenTracker` correctly computes N* from known costs
- [x] T20: Unit test — `BreakevenTracker::is_amortized()` flips at exactly N* tokens
- [x] T21: Unit test — `BreakevenBandit` selects amortized tier over non-amortized
- [x] T22: Unit test — `FidelityMatcher` returns higher compression for later positions
- [x] T23: Integration test — `InferenceRouter` with breakeven routes differently than without
- [x] T24: Benchmark — measure overhead of breakeven computation (< 100ns per forward call)
- [ ] T25: GOAT proof — demonstrate breakeven routing saves ≥5% wallclock on long sequences vs QPS-only routing

### Phase 6: Feature Gate & Documentation

- [x] T26: Add `breakeven_routing` feature flag to `Cargo.toml`
- [x] T27: Add to `lib.rs` conditional module: `#[cfg(feature = "breakeven_routing")] pub mod breakeven;`
- [x] T28: Update README.md with breakeven routing section
- [x] T29: Create benchmark document at `.benchmarks/046_breakeven_routing_goat.md`

---

## Success Criteria

1. **Breakeven N* computed correctly** — matches paper's equation with wallclock timings
2. **Overhead < 100ns per forward** — breakeven tracking must be negligible
3. **GOAT gate: ≥5% wallclock savings** on long sequences (≥512 tokens) vs QPS-only routing
4. **Feature-gated** — no impact when disabled
5. **Zero allocation in hot path** — all tracking uses pre-allocated arrays

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/breakeven/mod.rs` | NEW — BreakevenTracker, BreakevenBandit |
| `src/breakeven/fidelity.rs` | NEW — FidelityMatcher |
| `src/inference_router.rs` | MODIFY — add breakeven signal integration |
| `src/lib.rs` | MODIFY — add conditional module |
| `Cargo.toml` | MODIFY — add feature flag |

## Plasma Path Alignment

| Layer | Breakeven Role |
|-------|----------------|
| **Plasma** (bit-plane ternary) | Lowest tier for KV cache — breakeven N* determines when worth enabling |
| **Hot** (CPU SIMD) | Default tier — always amortized (zero setup) |
| **Warm** (GPU dispatch) | Breakeven N* = GPU compile time / per-token savings |
| **Cold** (ANE + speculative) | Breakeven N* = ANE compile + draft model load / per-token savings |
| **Freeze** (KV cache dump) | Breakeven N* = serialization cost / replay savings |
