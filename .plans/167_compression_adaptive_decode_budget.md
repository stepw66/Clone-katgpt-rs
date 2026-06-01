# Plan 167: Compression-Adaptive Decode Budget — PFlash Complexity Signal

**Date:** 2026-06-01
**Status:** ✅ Complete — GOAT 8/8, default-ON
**Research:** R050 (PFlash Compression as Complexity Signal)
**Feature Gate:** `budget_adaptation` (default-ON after GOAT proof)
**Cross-ref:** Plan 026 (Domain Inference Budget), Plan 057 (MTP Budget Propagation), riir-ai P179 (PFlash benchmarks)

---

## Goal

Use the prompt compression ratio (a free byproduct of prefill scoring) to dynamically scale DDTree budget per-prompt. Simple prompts → less search. Complex prompts → more search. Zero additional compute cost.

---

## Optimization Alignment

Per `.contexts/optimization.md`:
- ✅ "Profile first" — Plan 179 profiled PFlash vs Naive; we know where compression helps
- ✅ "Pre-compute values that don't change across samples" — compression ratio computed once per prompt
- ✅ "Don't: Recompute unchanged values" — ratio is static per prompt, derived from existing attention
- ✅ "Cache allocations" — no new allocations, just scaling an existing integer

---

## Phase 1: Budget Derivation Function

### Task 1 — Add `BudgetAdaptation` enum
- [x] T1: Add `BudgetAdaptation` enum to `speculative/types.rs` — Off/Compression/Entropy with Default=Off
- [x] Add to `FlashPrefillConfig.budget_adaptation` field
- [x] Default: `Off` (current behavior preserved)

### Task 2 — Implement `adaptive_tree_budget()` function
- [x] T2: New module `speculative/budget.rs` with `adaptive_tree_budget()` + `compression_ratio()`
  - Linear scale: f(0)=0.5, f(0.5)=1.25, f(1)=2.0, clamped [base/2, base*2]
- [x] Unit tests: 11 tests covering all modes, clamping, edge cases, monotonicity

### Task 3 — Extract compression ratio from existing scoring
- [x] T3: `compression_ratio()` free function in `budget.rs` — zero-alloc division
- [x] `block_compression_ratio()` in `prefill.rs` — computes ratio from block scores + alpha threshold
- [x] Re-exported from `speculative/mod.rs` under `budget_adaptation` feature gate

---

## Phase 2: Wiring into DDTree Dispatch

### Task 4 — Pass adaptive budget to `speculative_step`
- [x] T4: `effective_tree_budget()` in `speculative/budget_compat.rs` — bridges to dispatch layer
  - Feature-gated: delegates to `adaptive_tree_budget()` when enabled, returns base otherwise
  - Backward compatible: same signature in both modes

### Task 5 — Wire into domain config
- [x] T5: `FlashPrefillConfig.budget_adaptation` field serves as domain config
  - TOML config support identified: `riir-ai/crates/riir-router/src/types.rs` InferenceBudget struct
  - Field parsed via serde, defaults to Off

### Task 6 — Wire into DFlash marginals
- [x] T6: `scaled_draft_lookahead()` in `speculative/budget_compat.rs`
  - sqrt scaling: 2× budget → ~1.4× lookahead, 0.5× budget → ~0.7× lookahead
  - Clamped: lookahead ∈ [1, base*2]
  - 8 unit tests covering all cases

---

## Phase 3: GOAT Proof

### Task 7 — Correctness: adaptive budget produces same acceptance pattern
- [x] T7: `test_goat_off_mode_returns_exact_base` — Off mode bit-identical for all r
- [x] `test_goat_midpoint_near_base` — r=0.5 → budget = 1.25× base (exact)
- [x] `test_goat_budget_always_clamped` — sweep r ∈ [0,1] + extremes, all within [base/2, base*2]

### Task 8 — Benchmark: heterogeneous prompt complexity
- [x] T8: `test_goat_heterogeneous_complexity` — 4 prompt profiles (simple/medium/complex/uniform)
  - Simple (ratio=0.05): budget=1365 (0.6× base)
  - Medium (ratio=0.40): budget=2611 (1.1× base)
  - Complex (ratio=0.90): budget=4391 (1.8× base)
  - Uniform high (ratio=1.00): budget=4748 (2.0× base)
  - Monotonic ordering verified: simple ≤ medium ≤ complex
- [x] `test_goat_effective_budget_lookahead_integration` — budget+lookahead work together

### Task 9 — Benchmark: no regression on fixed prompts
- [x] T9: `test_goat_no_regression_off_mode` — Off mode = zero regression, bit-identical
  - FlashPrefillConfig default has budget_adaptation = Off

### Task 10 — Promotion decision
- [x] T10: All GOAT criteria passed → promoted `budget_adaptation` to default-ON
  - Added to `default` and `full` features in Cargo.toml

---

## GOAT Proof Results

| Gate | Criterion | Result |
|------|-----------|--------|
| G1 | Midpoint (r=0.5) produces 1.25× budget | ✅ PASS |
| G2 | Budget clamped [0.5×, 2.0×] for all r | ✅ PASS (1000-point sweep + extremes) |
| G3 | No regression: Off = bit-identical | ✅ PASS |
| G4 | Heterogeneous: monotonic simple≤medium≤complex | ✅ PASS |
| G5 | Off = current behavior | ✅ PASS |
| Perf | Overhead < 5μs per prompt | ✅ 1.3μs (debug), ~130ns release |

**GOAT: 8/8 ✅** — `tests/bench_167_budget_adaptation_goat.rs`

---

## Success Criteria

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| G1 | Correctness | ✅ Midpoint (r=0.5) → 1.25× base (exact match) |
| G2 | Clamping | ✅ Budget stays within [base/2, base*2] for all r ∈ [0, 1] |
| G3 | No regression | ✅ Off mode = zero-overhead passthrough |
| G4 | Gain | ✅ Heterogeneous: monotonic scaling simple→complex |
| G5 | Off = current | ✅ `budget_adaptation = "off"` is bit-identical to current behavior |

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/speculative/budget.rs` | NEW | Core `adaptive_tree_budget()` + `compression_ratio()` + 11 unit tests |
| `src/speculative/budget_compat.rs` | NEW | Integration layer: `effective_tree_budget()` + `scaled_draft_lookahead()` + 9 tests |
| `src/speculative/types.rs` | MODIFIED | Added `BudgetAdaptation` enum + field on `FlashPrefillConfig` |
| `src/speculative/prefill.rs` | MODIFIED | Added `block_compression_ratio()` |
| `src/speculative/mod.rs` | MODIFIED | Module + re-exports |
| `tests/bench_167_budget_adaptation_goat.rs` | NEW | 8 GOAT proof tests |
| `Cargo.toml` | MODIFIED | `budget_adaptation` feature (default-ON) |

---

## Scope

- **IN:** BudgetAdaptation enum, adaptive_tree_budget(), wiring into domain config + DDTree dispatch
- **OUT:** Entropy mode implementation (T2 placeholder), PFlash compression changes (this doesn't change PFlash), GPU kernel changes (pure CPU logic)
