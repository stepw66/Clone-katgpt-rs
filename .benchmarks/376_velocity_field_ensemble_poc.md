# Benchmark 376: Velocity-Field Ensemble — Phase 2 Cross-Domain PoC

**Date:** 2026-07-04
**Plan:** [376_velocity_field_ensemble_primitive.md](../.plans/376_velocity_field_ensemble_primitive.md)
**Research:** [375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md](../.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md)
**Source paper:** [arXiv:2602.20070](https://arxiv.org/abs/2602.20070) — Coeurdoux et al., ICML 2026 SPIGM
**Bench:** `crates/katgpt-core/benches/bench_376_velocity_field_ensemble_poc.rs`

---

## Summary

Phase 2 defend-wrong PoC (AGENTS.md §3.6) for the velocity-field ensemble's
cross-domain composition claim. Three competitors head-to-head on a held-out
target domain, two source regimes.

**G2 verdict: PASS (3/3 metrics in the paper's claim regime).**

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G2 (related regime)** | ensemble beats single-best on ≥ 2/3 metrics | **3/3 wins, 3.5× MSE reduction** | ✅ PASS |
| **G2 (unrelated regime)** | informational (null regime) | 3/3 wins but weak absolute quality | ℹ️ informational |

Eligible to proceed to Phase 3 GOAT gate (G1 mechanics, G3 no-regression, G4
latency). The cross-domain composition claim stands for the paper's claim regime
(related sources).

---

## Setup

Synthetic linear velocity-field domain (D=8). Target = fixed random linear
field `b*(x) = W* x` with `W* ~ U(-0.5, +0.5)^{8×8}`. Three source drafters
in two regimes:

- **Regime 1 (related — paper's claim regime):** each source `W_i = W* + Δ_i`,
  `Δ_i ~ U(-0.3, +0.3)`. Models cross-domain composition where sources share
  structure with the target (the F-MNIST → MNIST case from Appendix E).
- **Regime 2 (unrelated — null regime):** each source `W_i` independent random
  `~ U(-0.5, +0.5)`. The null case; no structural relation to the target.

Data: N_train = 200 fit pairs, N_test = 200 held-out pairs. Each pair
`(x_n, İ_t_n)` with `x_n ~ N(0, I_D)` and `İ_t_n = W* x_n + ε_n`, label noise
σ_noise = 0.05. Deterministic LCG (seed `0x376_5EED_3765`); bit-reproducible.

### Competitors

- **(a) single-best source:** for each source `i`, evaluate `b_i` alone on
  train; pick the source with the best train MSE; report its test metrics.
  Paper's "frozen, no-adaptation" baseline (§3.3).
- **(b) cross-domain ensemble (this primitive):** ridge-solve η over the 3
  sources on train (λ=1e-4); report test metrics of `Σ η_i b_i(x)`.
- **(c) target-trained-from-scratch:** solve a single linear `W_approx` directly
  from the 200 train pairs via per-row least-squares (8 ridge solves of size 8,
  reusing `ridge_solve_direct_f32`). Closed-form analog of "train a fresh model
  on target data" — reference upper bound.

### Metrics (all on held-out test set)

1. **MSE** — `mean_n ‖b̂(x_n) − İ_t_n‖² / D` (primary regression metric).
2. **top-1 agreement** — fraction of test pairs where `argmax_k b̂(x_n)[k] ==
   argmax_k İ_t_n[k]`.
3. **mean rank** — mean over test pairs of the rank of the true argmax action
   in the predicted ranking (1 = perfect).
4. **NLL** — `-log(σ(s_true) / Σ_k σ(s_k))` (sigmoid-normalized categorical;
   reported only, does not gate).

---

## Regime 1: related sources (paper's claim regime)

```
D=8, N_sources=3, N_train=200, N_test=200, σ_bias=0.3, σ_noise=0.05
per-source test MSE: [0.21820, 0.26748, 0.30882]  (best=0 → competitor a)
ensemble η = [+0.3618, +0.3051, +0.2286]

Competitor metrics (held-out test set):
(a) single-best source            MSE=   0.21820  top1=0.665  rank= 1.47  NLL= 1.671
(b) cross-domain ensemble         MSE=   0.06300  top1=0.765  rank= 1.28  NLL= 1.685
(c) from-scratch (target)         MSE=   0.00268  top1=0.940  rank= 1.07  NLL= 1.665
```

### Verdict

| Metric | (a) single-best | (b) ensemble | (c) from-scratch | (b) vs (a) |
|---|---|---|---|---|
| **MSE** ↓ | 0.21820 | **0.06300** | 0.00268 | **b wins (3.5× reduction)** |
| **top-1** ↑ | 0.665 | **0.765** | 0.940 | **b wins (+0.10)** |
| **rank** ↓ | 1.47 | **1.28** | 1.07 | **b wins (−0.19)** |
| NLL ↓ | 1.671 | 1.685 | 1.665 | a wins (within noise) |

**G2 PASS — 3/3 primary metric wins.**

### Interpretation

The ensemble's solved weights `[+0.362, +0.305, +0.229]` show it correctly
down-weights the more biased sources (source 3 had the worst solo MSE 0.309
and got the lowest weight 0.229). The MSE reduction from 0.218 → 0.063 (3.5×)
demonstrates the regression-optimal combination cancels per-source biases —
exactly the paper's mechanism.

The gap to from-scratch (c) is ~24× (0.063 vs 0.00268): the from-scratch model
has direct access to W* via 200 target pairs, while the ensemble is constrained
to the 3 sources' span. This is the expected cross-domain ceiling — sources
that are W* + bias do not perfectly span W*, so the residual error is bounded
below by the projection of W* onto span(W_1, W_2, W_3).

---

## Regime 2: unrelated sources (null regime)

```
D=8, N_sources=3, N_train=200, N_test=200, σ_bias=0.3, σ_noise=0.05
per-source test MSE: [1.33269, 1.47601, 1.59788]  (best=0 → competitor a)
ensemble η = [+0.1804, +0.0416, +0.0068]

Competitor metrics (held-out test set):
(a) single-best source            MSE=   1.33269  top1=0.140  rank= 4.22  NLL= 2.120
(b) cross-domain ensemble         MSE=   0.81010  top1=0.165  rank= 4.20  NLL= 2.075
(c) from-scratch (target)         MSE=   0.00268  top1=0.940  rank= 1.07  NLL= 1.665
```

### Verdict

| Metric | (a) single-best | (b) ensemble | (c) from-scratch | (b) vs (a) |
|---|---|---|---|---|
| **MSE** ↓ | 1.33269 | **0.81010** | 0.00268 | b wins (1.6×) |
| **top-1** ↑ | 0.140 | **0.165** | 0.940 | b wins (+0.025) |
| **rank** ↓ | 4.22 | **4.20** | 1.07 | b wins (−0.02) |
| NLL ↓ | 2.120 | 2.075 | 1.665 | b wins |

**G2 PASS technically (3/3 wins), but the absolute quality is poor.**

### Interpretation — honest reading

The regime-2 PASS is technically true but qualitatively weak:

- Absolute MSE 0.81 is **300× worse** than from-scratch (0.00268). The ensemble
  is barely better than predicting zero.
- top-1 = 0.165 is **near chance** (1/8 = 0.125 for uniform random). The
  ensemble picks the right action only ~17% of the time.
- The 1.6× MSE improvement over single-best is mostly because the single-best
  baseline is itself terrible (random matrices).

The solved weights `[+0.180, +0.042, +0.007]` collapse toward source 0 — the
ridge solve correctly identifies that 2 of the 3 sources are unhelpful and
down-weights them. This is the right behavior (regularization working), but the
result confirms: **with no source-target structural relation, no combination
helps meaningfully.** This is exactly the paper's caveat — cross-domain
composition requires related sources.

**This regime is informational, not gating.** The paper makes no claim for
unrelated sources; the bench's G2 gating only checks Regime 1.

---

## Defend-wrong audit (AGENTS.md §3.6)

The PoC was designed to potentially refute the cross-domain claim:

1. **Held-out test set** — metrics are on fresh pairs, not fit pairs. Rules out
   pure interpolation.
2. **Two regimes** — Regime 1 tests the claim; Regime 2 is the null. If
   Regime 1 had failed, the claim would be refuted.
3. **Three competitors** — (a) is the honest baseline (best single source),
   (c) is the honest upper bound (from-scratch with same closed-form math).
4. **NLL uses sigmoid** (per AGENTS rule) — reported but not gating, since it's
   mathematically equivalent to softmax-NLL on the velocity logits.
5. **No cherry-picking** — fixed seed, fixed σ_bias=0.3, fixed N=200. Numbers
   are bit-reproducible.

The PoC did NOT refute the claim in the related regime. The cross-domain
composition works as the math predicts: ridge solve finds the regression-
optimal signed combination, which is by construction ≥ any single source. The
empirical question was whether this holds on held-out data with noise — it does.

---

## Honest caveats

1. **Linear fields only.** The PoC uses linear `b(x) = W x`. Real drafters (LLM,
   HLA, KARC) are nonlinear; the linear regime is the easiest case for ridge
   combination. Nonlinear fields may behave differently. **The PoC validates the
   math, not the nonlinear-drafter use case** — that validation requires real
   game drafters (deferred to riir-ai Plan 385 runtime integration).

2. **σ_bias = 0.3 is moderate.** The sources are 60%-biased versions of W*.
   Tighter bias (σ_bias = 0.1) would give the ensemble more headroom; looser
   bias (σ_bias = 0.5) would push it toward the unrelated regime. The chosen
   value is a realistic "trained on related game" setting.

3. **N_train = 200 is small.** With more pairs, the from-scratch (c) gap would
   shrink further (more data → better W_approx). The ensemble's gap to (c)
   reflects the structural limit (sources' span), not data scarcity.

4. **NLL is near uniform.** All competitors have NLL ≈ 1.67–2.12, close to
   `log(8) ≈ 2.08` (uniform categorical). This is because the velocity logits
   have small dynamic range (entries ~O(0.1–1.0)); sigmoid normalization
   doesn't separate them strongly. NLL is not a discriminating metric here;
   MSE and top-1 carry the signal.

5. **The regime-2 "PASS" is misleading in isolation.** It would be dishonest to
   cite "3/3 wins in regime 2" as evidence the ensemble helps with unrelated
   sources. The absolute quality (MSE 0.81, top-1 0.165) shows it barely helps.
   The honest framing: the ridge solve always finds *some* improvement over
   single-best (by construction), but the improvement is meaningful only when
   sources share structure with the target.

---

## Run reproducibility

```bash
CARGO_TARGET_DIR=/tmp/vfe_376 cargo build --release -p katgpt-core \
    --features velocity_field_ensemble --bench bench_376_velocity_field_ensemble_poc
/tmp/vfe_376/release/deps/bench_376_velocity_field_ensemble_poc-* --nocapture
```

Deterministic LCG seed `0x376_5EED_3765`. Same seed → same numbers, bit-for-bit.

---

## TL;DR

Phase 2 PoC supports the cross-domain composition claim in the paper's claim
regime (related sources): **3/3 primary metric wins, 3.5× MSE reduction over
single-best, η correctly down-weights biased sources.** The null regime
(unrelated sources) confirms the claim is conditional on source-target
relatedness. Eligible for Phase 3 GOAT gate (G1 mechanics ✅, G3 no-regression,
G4 latency) and promotion decision.
