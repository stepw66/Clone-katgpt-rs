# Benchmark 376 — Velocity-Field Ensemble UQ Conformal Floor (Phase 6)

**Date:** 2026-07-04
**Plan:** [katgpt-rs/.plans/376_velocity_field_ensemble_primitive.md](../.plans/376_velocity_field_ensemble_primitive.md) Phase 6
**Issue:** `038_velocity_field_ensemble_uq_conformal_floor` (resolved, removed — this benchmark is the canonical record; recover via `git show fce6e44b^:.issues/038_velocity_field_ensemble_uq_conformal_floor.md`)
**Rule:** "Report the Floor" (Research 322, Plan 340, Issue 010) — any UQ-bearing primitive MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1) on CRPS / coverage / Winkler.

---

## TL;DR

**Verdict: ✅ BEATS FLOOR.** The velocity-field ensemble (configured as a 1D forecaster with two linear closure fields, fit on N=200 AR(1) pairs, sampling via Gaussian noise around the regression-optimal drift) beats the canonical conformal-naive floor on **both** lower-is-better metrics (CRPS, Winkler) with coverage within tie tolerance. This refutes the prior expectation that the ensemble would lose (the ensemble's induced distribution is Gaussian, but the floor's seasonal-naive point forecast is worse than the ensemble's drift-aware point forecast, and that point-quality advantage carries the win).

**Implication:** the UQ gate (Issue 010 "Report the Floor" rule) **passes** for the velocity-field ensemble on the AR(1) corpus. If a future caller adds a UQ claim to the primitive, this benchmark satisfies the mandatory floor comparison. The primitive currently makes NO UQ claim (it ships as a non-UQ algebraic combiner per Plan 376 Phase 3) — this benchmark is a pre-validation, not a claim addition.

---

## Setup

### Corpus

AR(1) stationary process: `x_{t+1} = φ·x_t + ε`, `ε ~ N(0, σ²)`, with `φ=0.7`, `σ=0.5`, deterministic seed `0x1234_5678_9ABC_DEF0` (SplitMix64).

- N_TRAIN = 200 (used to fit the ensemble + estimate residual std).
- N_TEST = 200 (the floor-comparison corpus; first 32 are warmup).
- n_scored = 168 (200 − 32 warmup).

AR(1) was chosen because it is the simplest non-trivial stochastic process where:
- The floor (seasonal naive: forecast = last observation) is a reasonable but suboptimal baseline.
- The ensemble can learn the optimal drift `b(x) = (φ−1)·x` from training pairs.
- Ground-truth residuals are Gaussian (so the ensemble's Gaussian-noise sampler is well-specified).

### Competitors

#### (a) Velocity-Field Ensemble (the primitive under test)

- **Fields:** 2 closure fields — `b_0(x) = x` (identity), `b_1(x) = 1.0` (constant). These span the AR(1) drift space: `drift = (φ−1)·x + ε ≈ a·x + b`.
- **Fit:** ridge solve (λ=1e-3) on N_TRAIN pairs `(x_t, drift_t = x_{t+1} − x_t)`.
- **Noise calibration:** residual std estimated from training residuals (drift − predicted drift).
- **Sampling:** for each `predict_next`, evaluate the drift at the current observed state `x_t`, generate M=64 samples `x_pred = x_t + drift + noise_sigma · ξ` where `ξ ~ N(0, 1)`.
- **Static fit regime:** the ensemble is fit once on the training prefix; it does NOT refit online (contrast with the floor, which adapts conformally to every observation).

#### (b) Conformal-Naive Floor (the canonical baseline)

`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1`, `exp_lambda=0.0`, `HStep` residual mode, capacity 256 (the Issue 010 canonical config). Adapts online: residual pool updates after every observation.

### Metrics (per Issue 010)

- **Mean interval-CRPS** (lower is better) — primary.
- **Mean Winkler interval score** (lower is better) — penalizes width + outside-miss.
- **Empirical coverage** at α=0.05 (nominal 0.95) — should converge to nominal.

---

## Results

```
=== Floor Comparison: VFE (2 linear closure fields, static-fit, Gaussian-noise sampler) ===
Corpus: ar1_phi0.7_sigma0.5_n200 (n_scored=168, n_unscorable=0, α=0.05)

Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
-------------------|------------|------------|--------------------|---------
Mean CRPS          |     1.9388 |     2.0794 |             0.9324 | WIN
Mean Winkler       |     2.3181 |     2.4616 |             0.9417 | WIN
Coverage (nom=0.95) |     0.9524 |     0.9583 | err 0.0024 vs 0.0083 | tie

Overall: ✅ BEATS FLOOR — primitive adds UQ value
```

| Metric | Primitive | Floor | Ratio | Verdict |
|---|---|---|---|---|
| Mean CRPS | 1.9388 | 2.0794 | 0.9324 | **WIN** (< 0.95 threshold) |
| Mean Winkler | 2.3181 | 2.4616 | 0.9417 | **WIN** (< 0.95 threshold) |
| Coverage (nom 0.95) | 0.9524 | 0.9583 | err 0.0024 vs 0.0083 | **tie** (within ±0.02) |

**Win margins:** CRPS 6.8% better, Winkler 5.8% better. Both exceed the 5% BEAT_THRESHOLD.

---

## Why the ensemble wins (analysis)

The prior expectation was that the ensemble would lose because its induced distribution is Gaussian (parametric) while the floor uses non-parametric empirical quantile calibration. **The prior was wrong.** The ensemble wins for a different reason:

1. **Point-quality advantage dominates.** The ensemble's regression-optimal drift `b̂(x) ≈ (φ−1)·x` is a strictly better point forecast than the floor's "last observation" (`b̂(x) = 0` in drift space, i.e., forecast = x_t). On AR(1) with `φ=0.7`, the optimal one-step forecast is `φ·x_t`, which requires learning the slope. The floor cannot learn the slope — it always predicts `x_t`. The ensemble learns the slope via the ridge solve.

2. **The Gaussian noise is well-specified for AR(1).** AR(1) innovations are Gaussian by construction in this corpus. The ensemble's `noise_sigma` (estimated from training residuals) matches the true innovation std. So the Gaussian sampling produces correctly-calibrated intervals — no distributional misspecification.

3. **The floor's conformal calibration cannot recover from the point-forecast bias.** Conformal prediction corrects miscalibrated intervals by quantile-shifting residuals, but it cannot fix a biased point forecast. The floor's residuals are centered around `(φ−1)·x_t` (the drift the floor misses), which inflates both CRPS and Winkler.

### When would the ensemble lose?

The ensemble's advantage is point-quality. It would lose in regimes where:
- **The point forecast is already optimal** (e.g., pure white noise — both the ensemble and the floor predict the mean, and the floor's non-parametric calibration is tighter).
- **The innovations are strongly non-Gaussian** (e.g., heavy-tailed or multimodal — the ensemble's Gaussian-noise sampler would be misspecified, while the floor's empirical quantiles adapt).
- **The process is non-stationary** and the ensemble's static fit goes stale (the floor adapts online; the ensemble does not — this is a known limitation of the static-fit regime tested here).

These regimes are not tested in this benchmark. A future re-gate on a richer corpus (e.g., Lorenz-63, real-world time series) should verify the win holds outside the friendly AR(1)+Gaussian setup.

---

## Caveats

1. **Single corpus, single seed.** This is a synthetic AR(1) corpus with one deterministic seed. The win is demonstrated, not exhaustively characterized. A production UQ claim would want multi-corpus validation.
2. **Static-fit regime.** The ensemble does not refit online. The floor does. This is a fair comparison for the "fit once, deploy" use case but may not hold for drifting distributions.
3. **Gaussian noise sampler.** The adapter wraps the ensemble's deterministic drift with a Gaussian noise sampler. The stochastic interpolator (`stochastic_interpolant_step_into`) is NOT directly used here — the adapter uses a simpler `drift + σ·ξ` formulation because the interpolator's schedule (α_t, β_t, γ_t over t∈[0,1]) is designed for the generation interpolant, not for one-step forecasting. Using the integrator directly would require a different benchmark setup (multi-step trajectory generation with a known target distribution).
4. **No claim added.** This benchmark satisfies the Issue 010 gate but does NOT add a UQ claim to the primitive. The primitive's module doc and Plan 376 still describe it as a non-UQ algebraic combiner. Adding a UQ claim is a separate decision that requires updating the module doc, the Cargo.toml comment, and the Plan 376 status.

---

## Reproduction

```sh
cd /Users/katopz/git/katgpt-rs
cargo test -p katgpt-core \
  --features velocity_field_ensemble,conformal_predictive_intervals \
  --test velocity_field_ensemble_uq_floor -- --ignored --nocapture
```

Test file: `crates/katgpt-core/tests/velocity_field_ensemble_uq_floor.rs`
Adapter: `VfeForecastAdapter` (in the test file)

---

## Cross-References

- **Plan 376 Phase 6:** `.plans/376_velocity_field_ensemble_primitive.md`
- **Issue 038 (the gate, resolved + removed):** `.benchmarks/376_uq_floor.md` (this file) is the canonical record. The original issue tracked the conformal-floor gate per the UQ-bearing primitive GOAT extension (Issue 010); recoverable via `git show fce6e44b^:.issues/038_velocity_field_ensemble_uq_conformal_floor.md`.
- **Issue 010 (the rule):** `.benchmarks/010_report_the_floor_consolidated.md`
- **Plan 340 (the floor):** `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1, default-on)
- **Floor harness:** `crates/katgpt-core/src/conformal/floor_harness.rs` — `run_floor_comparison`, `UqPrimitiveUnderTest`, `FloorAdapter`
- **Sibling benchmarks (other primitives' floor comparisons):** `.benchmarks/010_best_belief_floor_comparison.md`, `.benchmarks/010_bom_floor_comparison.md`, `.benchmarks/010_sleep_time_floor_comparison.md`
