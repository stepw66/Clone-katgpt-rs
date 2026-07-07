# Benchmark 411 S2: SSMax Built-in Rolling-Δ Estimator GOAT Gate

**Date**: 2026-07-07
**Plan**: `.plans/411_ssmax_goldshare.md` Stretch S2
**Research**: `.research/392_*` (arXiv:2607.01538, Gollapudi et al., *Drowning in Documents at Million Token Scale*)
**Target dir**: `CARGO_TARGET_DIR=/tmp/ssmax_s2_check` (kept for follow-up; clean with `rm -rf /tmp/ssmax_s2_check`)
**Feature**: `ssmax_adaptive` (opt-in, implies `ssmax_temperature`)

## Primitive

`RollingDeltaEstimator` — lock-free EMA of the `max(logits) − mean(logits)` proxy
for the gold-distractor logit gap `Δ`. Produces `SsmaxMode::Adaptive` on demand
via `to_mode()`. Warm-start `Δ = 1.0` → `s_L = 1.0` (matches `Fixed { s_l: 1.0 }`).

**Key modelless insight**: at inference time we don't know which key is "gold",
but `max(logits) − mean(logits)` is an observable proxy for the paper's
`Δ = s_{t⋆} − s̄_{distractor}` (the gold key tends to be the max; distractors
cluster near the mean). The estimator observes this proxy per attention row and
maintains an EMA, producing `s_L = 1/Δ_typical` (clamped to `[0.1, 10.0]`).

## Summary verdict

| Gate | Target | Verdict | Key evidence |
|------|--------|---------|--------------|
| G1 (convergence) | `|Δ_est − Δ_true| / Δ_true < 10%` after 50 obs | ✅ PASS | Rel err 1.0–2.6% across N ∈ {64, 1k, 10k} |
| G2 (retrieval parity) | `cos_estimator ≥ 0.95 · cos_analytical` | ✅ PASS | Estimator *beats* analytical at every N (0.975 vs 0.972) |
| G3 (latency) | `observe_row + to_mode < 100µs` at N=10k | ✅ PASS | 6.55 µs/call (15× headroom) |
| G4 (alloc-free) | 0 allocs / 1000 calls | ✅ PASS | 0 allocations |
| G5 (no-regression) | warm-start s_L = 1.0 (matches Fixed) | ✅ PASS | Bit-identical |

**All gates PASS.** The estimator is eligible for promotion from opt-in
(`ssmax_adaptive`) to default-on. Decision: **keep opt-in for now** — the
estimator is a Phase 6 refinement, and promotion should wait for riir-ai
runtime validation on real (non-synthetic) attention distributions. The
synthetic task confirms the mechanism works; real-world parity needs a PoC
per research skill §3.6.

## G1 (convergence) — max-mean proxy vs true Δ

**Bench**: `benches/bench_411_ssmax_adaptive_goat.rs` → G1 section
**Setup**: synthetic retrieval task, Δ = 0.5 (gold-distractor pre-softmax gap).
Estimator with α = 0.3 observes 50 forward passes with varying seeds (same Δ).
Target: `|Δ_est − 0.5| / 0.5 < 10%`.

| N | true_delta | est_delta | rel_err% | converged? |
|------|------------|-----------|----------|------------|
| 64 | 0.500000 | 0.486998 | 2.60 | YES |
| 1000 | 0.500000 | 0.494472 | 1.11 | YES |
| 10000 | 0.500000 | 0.494935 | 1.01 | YES |

**Result**: PASS. The max-mean proxy converges to within 1–3% of the true Δ.
The small underestimate (Δ_est ≈ 0.49 vs Δ_true = 0.50) is expected: the gold
logit inflates the mean slightly, so `max − mean < max − distractor_mean`.

## G2 (retrieval parity) — estimator vs analytical s_L=1/Δ

**Bench**: `benches/bench_411_ssmax_adaptive_goat.rs` → G2 section
**Setup**: same retrieval task, but each key has a distinct one-hot value
vector `v_j = e_{j mod d_model}` (d_model=16). The attention output
`o = Σ_j α_j v_j` should point toward `v_gold`. Measure cosine similarity
`cos(o, v_gold)` with (a) no SSMax, (b) analytical `s_L = 1/Δ`, (c) estimator.
Target: `cos_estimator ≥ 0.95 · cos_analytical`.

| N | base_cos_sim | analytical_cos | estimator_cos | parity? |
|------|-------------|----------------|---------------|---------|
| 64 | 0.286570 | 0.972081 | 0.977192 | ✓ |
| 1000 | 0.254275 | 0.971646 | 0.975302 | ✓ |
| 10000 | 0.250215 | 0.970448 | 0.975019 | ✓ |

**Result**: PASS. The estimator *beats* the analytical oracle at every N. This
is because the max-mean proxy slightly underestimates Δ (giving a slightly
higher `s_L = 1/Δ_est > 1/Δ_true`), which sharpens a bit more aggressively —
and the synthetic task rewards aggressive sharpening. On real distributions the
relationship may differ, but the parity target (95% of analytical) is met with
~0.3% margin.

## G3 (latency) — observe_row + to_mode overhead

**Bench**: `benches/bench_411_ssmax_adaptive_goat.rs` → G3 section
**Setup**: 10,000 iterations of `observe_row(10k logits) + to_mode()` at N=10k.
Target: < 100µs/call.

```
observe_row + to_mode at N=10000: 6548.6 ns/call (6.55 µs/call)
Target: < 100,000 ns (100 µs)
```

**Result**: PASS. 6.55 µs/call — 15× under the 100µs target. The O(N) single
pass for max+sum dominates; the CAS loop is negligible (uncontended).

## G4 (alloc-free) — 0 allocations over 1000 calls

**Bench**: `benches/bench_411_ssmax_adaptive_goat.rs` → G4 section
**Setup**: CountingAllocator, 1000 × `observe_row(10k logits) + to_mode()`.

```
Allocations: 0
```

**Result**: PASS. Zero allocations — the estimator is a fixed-size struct
(`AtomicU64` + `f64`), `observe_row` is a single-pass stack computation, and
`to_mode` returns a `Copy` type.

## G5 (no-regression) — warm-start matches Fixed { s_l: 1.0 }

**Bench**: `benches/bench_411_ssmax_adaptive_goat.rs` → G5 section
**Setup**: construct `RollingDeltaEstimator::default()` (no observations),
check `to_mode().resolve_s_l()` == `Fixed { s_l: 1.0 }.resolve_s_l()`.

```
Warm-start s_L = 1.000000, Fixed s_L = 1.000000
```

**Result**: PASS. Bit-identical. Before any observation, the estimator behaves
exactly like `Fixed { s_l: 1.0 }` — the truly modelless default.

## Design notes

### Why EMA (not ring buffer)?

- EMA: single `f64` (8 bytes), `AtomicU64` CAS loop, zero allocation.
- Ring buffer: needs heap allocation for the window, `Mutex` or lock-free
  index management — strictly more complex for marginal accuracy gain.
- The EMA's exponential forgetting is the right model for non-stationary
  attention distributions (the "typical" Δ drifts as context changes).

### Why AtomicU64 (not papaya)?

- The plan text says "papaya hashmap per layer". Papaya is for key→value maps.
  A single estimator is one value — `AtomicU64` is simpler, faster, and
  dependency-free.
- Per-layer storage (the "papaya per layer" part) is caller-managed: callers
  who want per-layer estimators use `Vec<RollingDeltaEstimator>` or a papaya
  `HashMap<LayerId, RollingDeltaEstimator>` if they need lock-free layer lookup.
  The estimator itself doesn't impose a storage strategy.

### Why max-mean (not top-1 minus top-2)?

- `max − mean` captures the *average* separation, which is what the dilution
  bound uses (`Δ = s_{t⋆} − s̄_{distractor}` where `s̄` is the mean distractor
  score).
- `top-1 − top-2` captures the *nearest competitor* gap, which is relevant for
  argmax preservation but not for the dilution bound's `(N−1)` denominator.
- `max − mean` is also more stable (less sensitive to individual noise spikes).

### Warm-start rationale

`Δ = 1.0` gives `s_L = 1/1.0 = 1.0`, matching `Fixed { s_l: 1.0 }`. This means:
- Before any observation, the estimator is a no-op (same as Fixed default).
- The estimator only changes behavior after it has observed enough data to
  have a confident estimate.
- This is the "zero runtime cost unless invoked" property — the estimator
  doesn't change SSMax's default behavior until a caller actively feeds it
  observations.
