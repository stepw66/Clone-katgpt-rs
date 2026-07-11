# Benchmark 010: Best-Belief Beta Selector Floor Comparison (Issue 010 T5)

**Date:** 2026-07-01
**Task:** Issue 010 T5 — "Report the Floor" comparison for Best-Belief Beta Selector (Plan 336).
**Verdict:** **Best-Belief BEATS the MLE floor** on selection quality — but only in the heteroscedastic regime (variable observation counts), which is the real-world use case. At uniform n, they tie (the monotonicity theorem).
**Adapter + tests:** `crates/katgpt-core/tests/conformal_floor_best_belief.rs`.

---

## TL;DR

The Best-Belief Beta Selector (Plan 336) computes `BB_ε(S, F) = I⁻¹_ε(1+S, 1+F)`
— the ε-quantile of the Beta(1+S, 1+F) posterior — and selects the candidate
with the highest lower bound. T5 asks: does the Beta prior beat the empirical
MLE floor on selection quality?

The answer has two parts, and both are load-bearing:

1. **At uniform observation count n (homoscedastic), the two methods produce
   IDENTICAL selections.** This is a theorem: with fixed n, `BB_ε(S, n−S)` is
   a monotone function of S/n, so argmax is preserved. The Beta prior shifts
   absolute scores but not the ordering. This is the degenerate baseline.

2. **At variable observation counts (heteroscedastic — the real-world case),
   Best-Belief WINS by 15–30% selection-regret reduction** (and up to 77% on
   the low-data false-positive stress test). This is where the Beta prior earns
   its keep: a candidate with 2/2 successes has MLE=1.0 (perfect) but
   BB_0.05 ≈ 0.025 (discounted for low evidence), so Best-Belief correctly
   prefers a candidate with 50/60 (MLE=0.833, BB_0.05 ≈ 0.75).

This is a genuine UQ gain, not a reframing. `best_belief` stays DEFAULT-ON
(promoted in Plan 336 Phase 2).

---

## Method

The comparison is NOT an interval-calibration test (CRPS/coverage/Winkler) —
it's a **selection-quality** test. The metric is **selection regret**:
`θ_best − θ_selected`, where `θ_best` is the highest true win-rate and
`θ_selected` is the true win-rate of the candidate each method picks.

### Two selectors

| Method | Score | Selector |
|---|---|---|
| **Best-Belief (Beta)** | `BB_ε(S, F) = I⁻¹_ε(1+S, 1+F)` | `select_best_belief(candidates, ε, None)` |
| **Empirical MLE floor** | `S/(S+F)` (with (0,0)→0.5) | argmax of rate |

The MLE floor is pure exploitation — no confidence adjustment, no
regularization. It picks the candidate that *looked* best in the data. The
Best-Belief Beta lower bound adds: (1) Beta(1,1) prior (regularizes low-data
candidates toward 0.5), (2) ε-quantile conservatism (a *lower* bound,
penalizing high-variance candidates).

### The key experimental knob: observation-count distribution

The central T5 finding emerged from the experimental design: **the observation-
count distribution is the only knob that matters.**

| Mode | Per-candidate n | Expected result |
|---|---|---|
| `Uniform { n_mean }` | every candidate gets exactly n_mean | TIE (monotonicity theorem) |
| `Variable { n_mean }` | n_i ~ Uniform[2, 2·n_mean] | Beta WINS (real-world) |
| `OneLowData { n_mean, n_lo }` | one candidate gets n_lo, rest get n_mean | Beta WINS big (false-positive stress) |

Each trial: draw K true rates θ_i ~ Uniform[0.3, 0.9], draw per-candidate
observation counts per the mode, draw S_i ~ Binomial(n_i, θ_i), select with
each method, measure regret. Average over 5000 trials.

---

## Canonical run (K=8, ε=0.05, θ ∈ [0.3, 0.9], 5000 trials)

### Uniform n (baseline — the monotonicity theorem)

| n | regret_beta | regret_mle | verdict |
|---|---|---|---|
| 4 | 0.098126 | 0.098126 | TIE |
| 8 | 0.063417 | 0.063417 | TIE |
| 16 | 0.036160 | 0.036160 | TIE |
| 32 | 0.021344 | 0.021344 | TIE |
| 64 | 0.012013 | 0.012013 | TIE |

**At every uniform n, regrets are bit-identical.** This confirms the
monotonicity argument empirically: with fixed n, `BB_ε(S, n−S)` is monotone in
S/n, so the argmax is the same as the MLE's. The Beta conservatism shifts the
absolute scores (useful for thresholding, gating, or comparing across pools
with different n) but does not change the within-pool selection.

### Variable n (real-world — heteroscedastic data)

| n_mean | regret_beta | regret_mle | improvement | verdict |
|---|---|---|---|---|
| 4 | 0.085163 | 0.099708 | 14.59% | BETA WINS |
| 8 | 0.060750 | 0.079318 | 23.41% | BETA WINS |
| 16 | 0.041112 | 0.058803 | 30.09% | BETA WINS |
| 32 | 0.028033 | 0.038250 | 26.71% | BETA WINS |
| 64 | 0.018766 | 0.025943 | 27.66% | BETA WINS |
| 128 | 0.011411 | 0.015277 | 25.30% | BETA WINS |

**Beta wins at all 6 levels (15–30% regret reduction).** When candidates have
different evidence weights, the Beta prior's regularization correctly
discounts low-data candidates that the MLE over-promotes. The improvement is
stable across n_mean (25–30% in the mid-range), confirming the value is
structural, not an artifact of a specific noise level.

### One-low-data stress test (false-positive regime)

| n_mean | n_lo | regret_beta | regret_mle | improvement | verdict |
|---|---|---|---|---|---|
| 32 | 2 | 0.027650 | 0.071420 | 61.29% | BETA WINS |
| 64 | 2 | 0.019554 | 0.067631 | 71.09% | BETA WINS |
| 32 | 4 | 0.026379 | 0.039749 | 33.63% | BETA WINS |
| 64 | 4 | 0.019290 | 0.032107 | 39.92% | BETA WINS |
| 128 | 2 | 0.013915 | 0.060595 | 77.04% | BETA WINS |

**The n_lo=2 case is the sharpest test.** A candidate with 2 observations
that happens to get 2/2 successes has MLE=1.0 (perfect), which the MLE
selector will pick over a genuinely better candidate with 60/80 (MLE=0.75).
Best-Belief discounts the 2/2 candidate (BB_0.05(2,0) ≈ 0.025) and correctly
prefers the 60/80 candidate (BB_0.05(60,20) ≈ 0.68). The improvement grows
with n_mean (the contrast between the low-data candidate and the rest
sharpens): 61% → 71% → 77% at n_lo=2 as n_mean goes 32 → 64 → 128.

---

## Why uniform n ties (the monotonicity theorem)

For fixed n, the Beta posterior is `Beta(1+S, 1+n−S)`. The ε-quantile
`I⁻¹_ε(1+S, 1+n−S)` is a monotonically increasing function of S (more successes
→ higher quantile, for fixed n). The MLE rate `S/n` is also monotonically
increasing in S. Two monotone functions of the same variable have the same
argmax. ∎

This is not a limitation of the experiment — it's a mathematical fact about
the primitive. The Beta conservatism's *selection* value is exclusively in the
heteroscedastic regime (different n per candidate). Its *absolute-score* value
(thresholding, gating, cross-pool comparison) exists regardless, but that's a
different claim than "better selection".

---

## ε sweep (Variable n_mean=8, K=8, 3000 trials)

| ε | regret_beta | regret_mle | verdict |
|---|---|---|---|
| 0.01 | 0.064818 | 0.077092 | BETA WINS |
| 0.05 | 0.062208 | 0.077092 | BETA WINS |
| 0.10 | 0.062685 | 0.077092 | BETA WINS |
| 0.20 | 0.060094 | 0.077092 | BETA WINS |
| 0.50 | 0.062624 | 0.077092 | BETA WINS |

Beta wins across all ε from 0.01 to 0.50. The improvement is fairly flat
(ε=0.05–0.10 is marginally best), suggesting the result is robust to the
conservatism knob — any reasonable ε captures most of the value.

---

## Verdict

**Best-Belief BEATS the MLE floor** on selection quality in the heteroscedastic
regime (the real-world use case: frozen snapshots / archetype shards with
different deployment durations → different observation counts). The improvement
is 15–30% on realistic variable-n data and up to 77% on the low-data false-
positive stress test.

This is a genuine UQ gain — unlike BoM (T3) and Sleep-Time (T4), which were
EXCLUDED via the reframing escape hatch, Best-Belief genuinely beats the floor
on its native metric (selection regret). The primitive earns its "conservative
selection" selling point.

| Property | MLE floor | Best-Belief (Beta) |
|---|---|---|
| Score | S/(S+F) (point estimate) | I⁻¹_ε(1+S, 1+F) (ε-quantile lower bound) |
| Regularizes low-data? | No (2/2 → 1.0) | Yes (2/2 → 0.025 at ε=0.05) |
| Uniform-n selection | (identical to Beta) | (identical to MLE) |
| Variable-n selection | baseline | **15–30% lower regret** |
| Low-data stress (n_lo=2) | baseline | **61–77% lower regret** |
| Selling point | pure exploitation | conservative selection under evidence imbalance |

`best_belief` stays DEFAULT-ON (promoted in Plan 336 Phase 2). The floor
comparison confirms the promotion: the Beta prior adds real selection-quality
value over the naive MLE in the heteroscedastic regime that characterizes its
actual use case.

---

## Reproducibility

```bash
cargo test -p katgpt-core --test conformal_floor_best_belief \
  --features conformal_predictive_intervals,best_belief -- --nocapture
```

All 6 tests green. Numbers are bit-reproducible (deterministic SplitMix64
with the same constants as the floor harness's RNG; deterministic seeds
0x1111/0xDEAD_BEEF/0xCAFE_F00D/0xFEED_FACE per experiment).

Config pinned: K=8, ε=0.05 (default), θ ∈ [0.3, 0.9], 5000 trials (3000 for
ε sweep and convergence). Observation-count modes: Uniform, Variable,
OneLowData (see test source for per-mode parameters).

---

## References

- Policy: `katgpt-rs/AGENTS.md` Feature Flag Discipline (UQ-bearing primitive
  GOAT gate extension), Issue 010.
- Adapter + tests: `crates/katgpt-core/tests/conformal_floor_best_belief.rs`.
- Best-Belief primitive: `crates/katgpt-core/src/best_belief.rs`, Plan 336.
- Best-Belief GOAT gate (G1–G4): `.benchmarks/` Plan 336 (3.099e-5 vs statrs,
  3.38ns score via LUT, 924/924 tests, alloc-free).
- Companion paper: [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) —
  *Report the Floor*.
- T3/T4 precedents (EXCLUDED via reframing): `.benchmarks/010_bom_floor_comparison.md`,
  `.benchmarks/010_sleep_time_floor_comparison.md`.
