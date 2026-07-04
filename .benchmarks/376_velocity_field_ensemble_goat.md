# Benchmark 376: Velocity-Field Ensemble — Phase 3 GOAT Gate

**Date:** 2026-07-04
**Plan:** [376_velocity_field_ensemble_primitive.md](../.plans/376_velocity_field_ensemble_primitive.md)
**Research:** [375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md](../.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md)
**Source paper:** [arXiv:2602.20070](https://arxiv.org/abs/2602.20070) — Coeurdoux et al., ICML 2026 SPIGM
**Bench (G3+G4):** `crates/katgpt-core/benches/bench_376_velocity_field_ensemble_goat.rs`
**Bench (G2 PoC):** `crates/katgpt-core/benches/bench_376_velocity_field_ensemble_poc.rs`
**Test (G3 alloc):** `crates/katgpt-core/tests/velocity_field_ensemble_alloc_check.rs`

---

## Summary

All four GOAT gates PASS. `velocity_field_ensemble` is **promoted to
default-on** in `crates/katgpt-core/Cargo.toml`. The primitive ships as a
first-class modelless ensemble-combination primitive alongside KARC (same
P×P Cholesky math, different basis construction).

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G1 mechanics** | η recovery `\|η - η*\|_∞ < 1e-4` | **9/9 unit tests PASS** | ✅ PASS |
| **G2 cross-domain quality** | ensemble ≥ single-best on ≥ 2/3 metrics | **3/3 wins, 3.5× MSE reduction** | ✅ PASS |
| **G3 no-regression** | 0 warnings + 0 hot-path allocs + combo clean | **all three verified** | ✅ PASS |
| **G4 latency** | fit ≤50µs / eval ≤200ns / batch(1000) ≤5ms | **6.27µs / 21ns / 20µs** | ✅ PASS |

**Promotion:** `velocity_field_ensemble` added to `default = [...]` in
`crates/katgpt-core/Cargo.toml`. Verified: `cargo check -p katgpt-core --lib`
(default features) compiles clean; 9/9 unit tests pass with default features
only (no explicit `--features`).

---

## G1 — Mechanics

9 unit tests in `crates/katgpt-core/src/velocity_field_ensemble.rs::tests`:

| Test | Purpose |
|------|---------|
| `test_fit_recovers_known_eta` | Synthetic η recovery: `\|η - [0.5, 0.3, 0.2]\|_∞ < 1e-4` (P=3, N=50). |
| `test_eval_is_linear_combination` | Signed weights (η can be negative; not a probabilistic mixture). |
| `test_gram_symmetric` | Gram symmetry after accumulate (K[i,j] == K[j,i]). |
| `test_chosen_lambda_stabilizes_ill_conditioned_gram` | Duplicate fields + λ > 0 → finite η. |
| `test_eval_batch_reuses_scratch` | Batch correctness; scratch reset per element. |
| `test_schedule_linear` | Linear schedule: (α,β) endpoints + γ = 1. |
| `test_schedule_trigonometric` | Trig schedule: (α,β) endpoints + γ = π/2. |
| `test_stochastic_interpolant_step_no_drift_no_noise` | Pure transport: x_{t+h} = (β_t/β_{t+h})·x_t. |
| `test_stochastic_interpolant_step_with_drift` | Drift contribution verified analytically. |

All 9 PASS. Headline: `test_fit_recovers_known_eta` confirms the ridge solve
recovers the known combination weights to < 1e-4.

```
running 9 tests
test velocity_field_ensemble::tests::test_fit_recovers_known_eta ... ok
... (all 9 ok)
test result: ok. 9 passed; 0 failed
```

---

## G2 — Cross-Domain Quality

See `.benchmarks/376_velocity_field_ensemble_poc.md` for the full PoC report.
Summary:

| Regime | (a) single-best MSE | (b) ensemble MSE | (c) from-scratch MSE | (b) wins |
|---|---|---|---|---|
| **Related sources** (W_i = W* + Δ_i) | 0.218 | **0.063** | 0.0027 | 3/3 metrics |
| Unrelated sources (null) | 1.333 | 0.810 | 0.0027 | 3/3 (but near-chance) |

Regime 1 (paper's claim regime) supports the cross-domain composition claim:
ensemble achieves **3.5× MSE reduction** over the single-best source, with
η = [+0.362, +0.305, +0.229] correctly down-weighting the more biased sources.
Regime 2 (null) confirms the claim is conditional on source-target relatedness.

---

## G3 — No-Regression

Three checks:

1. **Zero warnings, single-feature:** `cargo check -p katgpt-core --features velocity_field_ensemble --lib` → 0 warnings, 0 errors.

2. **Combo check (the merkle_root lesson):** `cargo check --workspace --all-features` → clean. No feature-combo regressions.

3. **Zero hot-path allocs:** `crates/katgpt-core/tests/velocity_field_ensemble_alloc_check.rs` — `CountingAllocator` wrapping `System`. After warmup:
   - `eval_into`: **0 allocs / 1000 calls**
   - `eval_batch_into`: **0 allocs / 100 batches of 100**
   
   The one-time `EnsembleFitScratch::new()` allocates 3 P×P Vecs (P=8 → 64
   floats each); subsequent `fit_into` / `eval_into` / `eval_batch_into` calls
   allocate nothing.

```
running 1 test
test g3_eval_and_batch_zero_alloc_after_warmup ... ok
test result: ok. 1 passed; 0 failed
```

**Implementation note (parallel-test fragility):** the alloc check uses a
single `#[test]` function (not two) because the `#[global_allocator]` counting
pattern is fragile under parallel test execution — sibling tests' setup
allocations get counted by the shared `ALLOC_COUNT`. Single-function is
inherently serial (matches the `tests/karc_alloc_check.rs` convention).

---

## G4 — Latency

`crates/katgpt-core/benches/bench_376_velocity_field_ensemble_goat.rs`,
release build, median over 10–100 batches.

| Operation | Target | p50 | Headroom | Status |
|---|---|---|---|---|
| `fit_into` (N=50, P=8, D=8) | ≤ 50 µs | **6.27 µs** | 8× | ✅ PASS |
| `eval_into` (single, P=8, D=8) | ≤ 200 ns | **21 ns** | 9.5× | ✅ PASS |
| `eval_batch_into` (N=1000, P=8, D=8) | ≤ 5 ms | **20 µs** | 250× | ✅ PASS |

```
--- G4: fit_into latency (N=50, P=8, D=8) ---
  fit_into p50: 6268 ns  (target ≤ 50 µs = 50000 ns)

--- G4: eval_into latency (single call, P=8, D=8) ---
  eval_into p50: 21 ns  (target ≤ 200 ns)

--- G4: eval_batch_into latency (N_batch=1000, P=8, D=8) ---
  eval_batch_into(N=1000) p50: 20µs  (target ≤ 5 ms)

=== ALL G3+G4 GATES PASS ===
```

The latency targets had generous headroom by design (plasma-tier budget). The
actual numbers are far inside budget — `eval_into` at 21ns is essentially the
cost of P=8 closure calls + D=8 multiply-adds. The batch op at 20µs/1000
states = 20ns/state confirms the per-state cost is dominated by the field
evaluations, not loop overhead.

---

## Promotion

**Decision:** PROMOTE `velocity_field_ensemble` to default-on.

**Rationale (per AGENTS.md feature flag discipline):**
1. ✅ All 4 GOAT gates PASS.
2. ✅ The gain is **modelless** — closed-form `ridge_solve_direct_f32`, no
   training, no backprop, no gradient descent.
3. ✅ No stack-slot conflict — this is a new slot ("ensemble combination"),
   distinct from KARC (delay-basis ridge) and committed_field_blend
   (sigmoid-voting).
4. ✅ Zero runtime cost unless a caller explicitly constructs an ensemble and
   calls `fit_into` / `eval_into`.

**Change:** `crates/katgpt-core/Cargo.toml` — added `"velocity_field_ensemble"`
to the `default = [...]` list; updated the feature's inline comment from
"Opt-in until G1–G4 GOAT gate passes" to "DEFAULT-ON (Plan 376 Phase 3)".

**Verification:**
- `cargo check -p katgpt-core --lib` (default features) → clean.
- `cargo test -p katgpt-core --lib velocity_field_ensemble` (default features)
  → 9/9 PASS.

---

## Run reproducibility

```bash
# G3 alloc test (debug build, single test to avoid parallel-test interference)
CARGO_TARGET_DIR=/tmp/vfe_376 cargo test -p katgpt-core \
    --features velocity_field_ensemble \
    --test velocity_field_ensemble_alloc_check

# G3 + G4 GOAT bench (release build)
CARGO_TARGET_DIR=/tmp/vfe_376 cargo build --release -p katgpt-core \
    --features velocity_field_ensemble \
    --bench bench_376_velocity_field_ensemble_goat
/tmp/vfe_376/release/deps/bench_376_velocity_field_ensemble_goat-* --nocapture

# G2 PoC bench (release build)
CARGO_TARGET_DIR=/tmp/vfe_376 cargo build --release -p katgpt-core \
    --features velocity_field_ensemble \
    --bench bench_376_velocity_field_ensemble_poc
/tmp/vfe_376/release/deps/bench_376_velocity_field_ensemble_poc-* --nocapture
```

Deterministic LCG seeds; bit-reproducible run-to-run.

---

## TL;DR

All G1–G4 PASS with comfortable headroom. `velocity_field_ensemble` promoted
to default-on as a first-class modelless ensemble-combination primitive. The
cross-domain composition claim is supported by the defend-wrong PoC (3.5× MSE
reduction in the related-sources regime). Phases 4–6 (heterogeneous-d, LatCal
commitment, UQ conformal floor) remain deferred follow-ups.
