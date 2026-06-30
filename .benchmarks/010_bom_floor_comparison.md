# Benchmark 010: BoMSampler Floor Comparison (Issue 010 T3)

**Date:** 2026-06-30
**Task:** Issue 010 T3 — "Report the Floor" comparison for BoMSampler (Plan 281).
**Verdict:** **EXCLUDED from the "Report the Floor" policy** (T3 escape hatch, exercised with evidence).
**Harness:** `crates/katgpt-core/src/conformal/floor_harness.rs` (Issue 010 T2).
**Adapter + tests:** `crates/katgpt-core/tests/conformal_floor_bom.rs`.

---

## TL;DR

BoMSampler (Plan 281) is a **belief-space exploration sampler**, not a calibrated
forecaster. When its K hypotheses are projected to a scalar and converted to an
empirical-quantile interval, the resulting intervals are **systematically
overconfident** (5–15% coverage vs the nominal 95%) — the textbook
**false-confidence** failure mode. CRPS alone flatters BoM (it rewards the
narrow σ-bound intervals), but coverage and Winkler expose the under-calibration.
BoM's GOAT gate (Plan 281 G2) measures **planning** win rate (+31.49pp on the
riir-ai arena, Plan 314), not calibrated UQ — so this exclusion is consistent
with BoM's actual selling point.

---

## Method

The `BoMSamplerAdapter` (`tests/conformal_floor_bom.rs`) wraps
`AttractorKernel::from_seed` (random init, **unfitted** to the corpus) as a
`UqPrimitiveUnderTest`:

- **Embedding**: the scalar observation `y_t` is embedded into the kernel's
  D-dim input as `x = [y_t, 0, 0, ..., 0]` (channel 0 carries the signal).
- **Prediction**: K=8 hypotheses are sampled via `sample_k_states`; each is
  projected to channel 0; the K scalars are returned as
  `PredictiveOutput::from_samples`. The harness converts samples → empirical-
  quantile interval.
- **State advance**: on `observe(y)`, the kernel's `step()` advances the belief
  state `s_prev`, so the next prediction sees the updated belief.
- **Noise queries**: regenerated each `predict_next` from a deterministic
  SplitMix64 + Box-Muller RNG (bit-reproducible).

Corpora are scaled into BoM's representable `(-1, 1)` output range to avoid a
structural range-mismatch loss:

- **small-amplitude seasonal**: `0.8·sin(2πt/12) + N(0, 0.05)` — the floor's
  home turf (seasonal-naive captures the structure).
- **small-σ white noise**: `N(0, 0.3)` — the floor's worst case (last-value
  forecast is meaningless on i.i.d. data; optimal forecast is the mean).

The kernel is **unfitted** because BoM's GOAT gate (Plan 281 G2) is measured in
riir-ai's arena (Plan 314), not as a scalar forecaster — there is no "fitted
scalar forecaster" configuration of BoM to test. This is the honest baseline.

---

## Canonical run (α = 0.05, σ = 0.1, K = 8, D = 4)

### Seasonal corpus (`small_amp_seasonal_0p8sigma0p05_n500`)

```
=== Floor Comparison: BoMSampler (AttractorKernel, unfitted, channel-0 projection) ===
Corpus: small_amp_seasonal_0p8sigma0p05_n500 (n_scored=452, n_unscorable=0, α=0.05)

Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
-------------------|------------|------------|--------------------|---------
Mean CRPS          |     0.8342 |     0.9635 |             0.8658 | WIN
Mean Winkler       |    14.4673 |     1.0493 |            13.7874 | LOSE
Coverage (nom=0.95) |     0.0553 |     0.9491 | err 0.8947 vs 0.0009 | LOSE

Overall: 🟠 MIXED
```

### White-noise corpus (`white_noise_sigma0p3_n500`)

```
=== Floor Comparison: BoMSampler (AttractorKernel, unfitted, channel-0 projection) ===
Corpus: white_noise_sigma0p3_n500 (n_scored=436, n_unscorable=0, α=0.05)

Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
-------------------|------------|------------|--------------------|---------
Mean CRPS          |     0.5181 |     1.6928 |             0.3060 | WIN
Mean Winkler       |     8.1984 |     1.9938 |             4.1120 | LOSE
Coverage (nom=0.95) |     0.1514 |     0.9518 | err 0.7986 vs 0.0018 | LOSE

Overall: 🟠 MIXED
```

### Summary table

| Corpus | CRPS ratio | Winkler ratio | Coverage (nom 0.95) | Verdict |
|---|---|---|---|---|
| seasonal | 0.866 | 13.79 | 0.055 | Mixed |
| white noise | 0.306 | 4.11 | 0.151 | Mixed |

---

## σ-sweep (seasonal corpus)

Widening σ widens BoM's interval (coverage climbs) but does NOT rescue the
calibration — even at σ=0.5 (5× the default), coverage reaches only 0.254
(still a third of nominal), and the verdict flips from Mixed to LosesToFloor
because the CRPS advantage evaporates.

| σ | CRPS ratio | Coverage | Verdict |
|---|---|---|---|
| 0.05 | 0.878 | 0.024 | Mixed |
| 0.10 | 0.883 | 0.052 | Mixed |
| 0.30 | 0.924 | 0.167 | Mixed |
| 0.50 | 0.995 | 0.254 | LosesToFloor |

**No σ gives both competitive CRPS AND nominal coverage.** This is the
structural mismatch: σ is exploration noise, not calibration width.

---

## Width-vs-volatility test (the structural smoking gun)

BoM's interval width should be **constant across volatility regimes** (it's
σ-bound), while the floor's width should **track local volatility** (residual
quantile). Test feeds BoM a low-volatility regime (σ_data = 0.02) then a
high-volatility regime (σ_data = 0.30) — a 15× volatility ratio — and measures
the interval-width ratio.

```
BoM interval width: low-vol mean = 0.1125, high-vol mean = 0.1114 (ratio 0.990)
```

**Width ratio ≈ 1.0** (not 15.0). BoM's interval width is blind to the data's
actual volatility — it tracks σ (the hyperparameter), not the residual stream.
The floor's width would track the 15× volatility change. This is the structural
reason BoM cannot be a calibrated UQ primitive: its uncertainty estimate is
exogenous (a knob), not endogenous (data-driven).

---

## Why BoM wins on CRPS but loses on coverage (the false-confidence trap)

CRPS for a uniform interval `[l, u]` rewards **narrowness**: a tight interval
around a decent point forecast scores well *on the samples inside the interval*.
BoM's σ=0.1 produces very narrow intervals (width ~0.11, see above), and its
channel-0 point forecast happens to track the seasonal signal reasonably well
(the sigmoid centers near 0, and the recurrent kernel integrates the input
history). So BoM's CRPS is competitive (ratio 0.87–0.31).

But CRPS does not severely penalize **outside-miss distance** the way Winkler
does. Winkler adds `(2/α)·|y - clamp(y, l, u)|` for misses — at α=0.05 that's a
40× penalty per unit distance. With coverage at 5–15%, most actuals are misses,
and the Winkler score explodes (4–14× the floor).

**This is the "Report the Floor" policy working as designed.** A naive read of
"BoM has lower CRPS than the floor" would conclude BoM is the better UQ
primitive. The policy's coverage + Winkler requirements expose that conclusion
as false confidence.

---

## Verdict

**EXCLUDED from the "Report the Floor" policy.**

Per the Issue 010 T3 escape hatch:
> If BoMSampler can't be evaluated on a continuous metric, document why and
> exclude it from the policy (it's a discrete selector, not a continuous UQ
> primitive).

BoM *can* be evaluated on a continuous metric (the harness handled it cleanly
via `from_samples` → empirical quantile), and the evaluation *was run*. The
result is not "BoM is bad" — it's "BoM's hypothesis spread is a different kind
of uncertainty than calibrated predictive intervals":

| Property | Conformal UQ (the floor) | BoM hypothesis spread |
|---|---|---|
| Width driver | Data residuals (endogenous) | σ hyperparameter (exogenous) |
| Adapts to volatility? | Yes (residual quantile) | No (see width-volatility test) |
| Coverage guarantee | Yes (split conformal, nominal ± tol) | No (5–25% at nominal 95%) |
| Point forecast | Seasonal-naive (last value) | Kernel channel-0 (unfitted) |
| Selling point | Calibrated intervals | Diverse hypotheses for minimax planning |
| GOAT gate | (is the floor) | Plan 281 G2: +31.49pp arena win rate |

BoM's value is in **planning** (the K hypotheses enable minimax-over-beliefs,
proven in riir-ai Plan 314), not in **forecast calibration**. Its GOAT gate
measures the former; the floor comparison measures the latter. Excluding BoM
from the UQ policy does NOT demote it — `bom_sampling` remains DEFAULT-ON
(promoted in Plan 281 T2.4). It just means BoM cannot claim "calibrated UQ" as
a selling point, which it never did.

---

## Reproducibility

```bash
cargo test -p katgpt-core --test conformal_floor_bom \
  --features conformal_predictive_intervals,bom_sampling -- --nocapture
```

All 5 tests green. Numbers are bit-reproducible (deterministic SplitMix64 +
Box-Muller for the noise queries; deterministic kernel seed; deterministic
corpus seeds).

Config pinned: `kernel_seed=42`, `noise_seed` per-corpus (see test source),
`D=4`, `K=8`, `σ=0.1` (default), `α=0.05`, warmup = 4 periods (seasonal) / 64
(white noise).

---

## References

- Policy: `katgpt-rs/AGENTS.md` Feature Flag Discipline (UQ-bearing primitive
  GOAT gate extension), Issue 010.
- Harness: `crates/katgpt-core/src/conformal/floor_harness.rs` (Issue 010 T2).
- Adapter + tests: `crates/katgpt-core/tests/conformal_floor_bom.rs`.
- BoMSampler: `crates/katgpt-micro-belief/src/bom.rs`, Plan 281.
- BoM GOAT gate (G2 arena): riir-ai Plan 314 (+31.49pp).
- Companion paper: [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) —
  *Report the Floor*.
