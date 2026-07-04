# Issue 038 — Velocity-Field Ensemble: UQ Conformal Floor (Issue 010 §"Report the Floor")

**Filed:** 2026-07-04
**Priority:** P3 (deferred — only mandatory if/when the ensemble makes a UQ claim; the primitive ships today as a non-UQ algebraic combiner with no calibrated-distribution claim)
**Origin:** katgpt-rs Plan 376 Phase 6 (deferred) — `.plans/376_velocity_field_ensemble_primitive.md`
**Blocks:** Any UQ-bearing claim about the velocity-field ensemble (e.g., "calibrated ensemble", "principled uncertainty", "predictive distribution over trajectories"). **Blocked by:** Nothing (the floor already ships — Plan 340 Phase 1, 2026-06-30).
**Type:** Benchmark / GOAT-gate extension (mandatory gate per Issue 010 before any UQ claim).

---

## Problem

The Velocity-Field Ensemble primitive (Plan 376, **promoted to default-on** in commit `b2091151`, 2026-07-04) currently ships as an **algebraic combiner** — no UQ claim is made. The Phase 2 PoC (`.benchmarks/376_velocity_field_ensemble_poc.md`) validated G2 cross-domain quality on MSE / top-1 / mean-rank, NOT on CRPS / coverage / Winkler.

Per the **"Report the Floor" rule** adopted 2026-06-28 (Research 322 / Plan 340 / Issue 010), any primitive that claims a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty MUST benchmark against the **conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, `m=1`, plain split conformal) — on CRPS / coverage / Winkler score. If the primitive cannot beat the floor, the GOAT gate FAILS for any UQ-bearing claim.

The velocity-field ensemble is **grandfathered** under the Issue 010 transition clause (it predates the rule's enforcement date 2026-06-30 — the primitive shipped 2026-07-04 but the gate was already enforceable). Per Issue 010: grandfathered UQ-bearing primitives "must include the floor at their next re-gate." **However**, the primitive currently makes NO UQ claim, so it is not yet UQ-bearing. This issue tracks the gate that becomes mandatory the moment anyone adds a UQ claim.

## Scope

The stochastic interpolator (`stochastic_interpolant_step_into` in `velocity_field_ensemble.rs`) does produce a stochastic trajectory — `x_{t+1} = x_t + D*_t · b̂(x_t) · dt + γ_t · √dt · ξ`. If integrated over many noise samples `ξ ~ N(0, I)`, this induces a distribution over terminal states `x_1`. The question this issue answers: **is that distribution calibrated?**

This is a benchmark issue, not an implementation issue. The floor already ships. The ensemble + integrator already ship. The work is: run both on a shared UQ benchmark, compare on the right metrics, and decide whether the UQ claim stands.

## Proposed direction (not committed)

### 1. Pick a UQ benchmark

Candidate: the bom_arena QMC benchmark referenced in riir-ai Plan 370 (Quasi-Monte Carlo). Alternative: a synthetic 2D Gaussian-mixture target where the ground-truth distribution is analytically known (so CRPS is computable exactly, not estimated).

Decision deferred to whoever picks this up — the key constraint is that the ground-truth distribution must be known so CRPS / coverage / Winkler are well-defined.

### 2. Two competitors

| Competitor | Description |
|---|---|
| **(a) Velocity-field ensemble + interpolator** | Fit η on N_train pairs, integrate to `t=1` over M noise samples, induce empirical distribution, score against ground truth. |
| **(b) Conformal-naive floor** | `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, `m=1`, plain split conformal). Score on the same ground truth. |

### 3. Metrics (per Issue 010 spec)

- **CRPS** (Continuous Ranked Probability Score) — primary.
- **Empirical coverage** at α = 0.8, 0.9, 0.95 — nominal vs achieved.
- **Winkler score** — interval score penalizing both width and miss.

### 4. Decision rule

- **If (a) does NOT beat (b)** on at least CRPS → drop the UQ claim. The primitive stays as a non-UQ algebraic combiner (still valuable — see Phase 3 G2: 3.5× MSE reduction over single-best source).
- **If (a) beats (b)** → UQ claim stands. Document in `.benchmarks/376_uq_floor.md`. Re-gate the GOAT with the floor included.

## Tasks

- [ ] **T1** Pick a UQ benchmark with known ground-truth distribution. Document the choice in `.benchmarks/376_uq_floor.md` (create the file).
- [ ] **T2** Implement the ensemble + integrator harness: fit η, integrate to `t=1` over M=1000 noise samples, induce empirical CDF.
- [ ] **T3** Implement the conformal-naive floor harness on the same benchmark. Reuse `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` from Plan 340.
- [ ] **T4** Compute CRPS / coverage / Winkler for both. Print verdict table.
- [ ] **T5** Decision per the rule above. If UQ claim dropped, add a `## UQ Status: NON-UQ` section to the primitive's module doc + the Plan 376 README. If UQ claim stands, re-gate the GOAT with the floor as a permanent fixture.
- [ ] **T6** If re-gated: extend the G2 PoC bench (`bench_376_velocity_field_ensemble_poc.rs`) to include the floor as a fourth competitor on the UQ metrics. Update `.benchmarks/376_velocity_field_ensemble_poc.md`.

## Non-Goals

- ❌ Implementing new UQ machinery in the primitive — the integrator already ships; this is a benchmark of what it produces.
- ❌ LatCal commitment of `η` — that's `riir-chain/.issues/003_*` (Phase 5, filed same day).
- ❌ Heterogeneous-d velocity fields — that's Plan 376 Phase 4 (deferred, only if a use case emerges).
- ❌ Runtime wiring into NPC ticks — that's riir-ai Plan 385.

## Honest caveats (documented up front)

1. **The ensemble may well lose to the floor.** Velocity-field ensembles are designed for *point* quality (regression-optimal drift), not *distributional* calibration. The interpolator's induced distribution is a side effect of the SDE discretization, not a designed-in calibration. If the floor wins, that's the honest answer — the primitive ships as non-UQ.
2. **The PoC is linear.** Plan 376 Phase 2 used linear velocity fields. A nonlinear-drafter PoC (deferred to riir-ai Plan 385) might change the UQ picture. This issue should run on the linear PoC first (cheaper, isolates the math) and re-run if/when Plan 385 ships a nonlinear validation.
3. **Coverage is a weak metric on small samples.** M=1000 noise samples may be too few for stable 0.95-coverage estimates. Consider M=10000 if variance is high.

## Cross-References

- **Source plan:** `.plans/376_velocity_field_ensemble_primitive.md` Phase 6 (deferred, this issue resolves it).
- **The rule:** `.issues/010_*` (consolidated; see `.benchmarks/010_report_the_floor_consolidated.md` for the cross-primitive summary). Issue 010 is **FULLY CLOSED** but the rule it codified is permanently enforceable — this issue is an instance of the rule, not a re-opening of 010.
- **The floor:** Plan 340 Phase 1 (shipped 2026-06-30) — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`.
- **Sibling primitives (grandfathered, must include floor at next re-gate):** BoMSampler (Plan 281), Sleep-Time Anticipator (Plan 334), Best-Belief Beta Selector (Plan 336 — already floored in `.benchmarks/010_best_belief_floor_comparison.md`), KARC+overlay.
- **Sibling issue:** `riir-chain/.issues/003_*` (Phase 5 LatCal commitment, filed same day).
- **Nonlinear follow-up:** `riir-ai/.plans/385_*` (runtime integration — may change the UQ picture if it ships nonlinear validation).

## TL;DR

Velocity-Field Ensemble (Plan 376, default-on since 2026-07-04) ships today as a non-UQ algebraic combiner — no calibrated-distribution claim. Per the Issue 010 "Report the Floor" rule, the moment anyone adds a UQ claim, the GOAT gate MUST benchmark against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340) on CRPS / coverage / Winkler. This issue tracks that benchmark. Honest expectation: the ensemble may lose to the floor (it's designed for point quality, not distributional calibration) — if so, the UQ claim is dropped and the primitive stays non-UQ. P3 — only mandatory if/when a UQ claim is added.
