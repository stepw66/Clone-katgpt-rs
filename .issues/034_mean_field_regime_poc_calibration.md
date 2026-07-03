# Issue 034: Mean-Field Regime Classifier — PoC calibration + paper-exact DMFT simulator

**Date:** 2026-07-03
**Source:** Plan 371 Phase 5 T5.1 defend-wrong PoC (INCONCLUSIVE verdict)
**Blocking:** Promotion of `mean_field_regime` feature to default-on
**Severity:** Medium (primitive ships opt-in and is mathematically sound; the PoC simulator is the weak link)

---

## Problem

The Plan 371 Phase 5 T5.1 defend-wrong PoC (`riir-poc/benches/mean_field_regime_poc.rs`) produced an **INCONCLUSIVE** verdict:

- **19/25 grid points match** (76%) — majority correct.
- **Only 1/4 distinct regimes correctly identified** — `NoiseSustainedOscillation` is consistently right; `Static`, `IrregularSwitching`, `GlobalLimitCycle` are not.
- **6 mismatches**, clustered at:
  - (a) **g=1.0 boundary** (5/6 mismatches): the classifier's `chaos_threshold = 1.0` treats g=1.0 as "not chaotic" (`g > 1.0` is false), but the simplified ODE shows oscillation at g=1.0. This is a `chaos_threshold` calibration sensitivity.
  - (b) **Intermediate β** (1/6 mismatches, plus contributing to the g=1.0 row): the classifier correctly detects the Hopf instability direction (complex eigenvalues with positive real part) but calls it `GlobalLimitCycle` instead of `IrregularSwitching`. This is a `hopf_margin` calibration issue — the trace T exceeds the 0.1 margin, triggering the limit-cycle verdict, but the actual dynamics show irregular switching.

## Root cause analysis

**The PoC simulator is the weak link, not the classifier.** The simplified ODE uses rough approximations:
- `χ̄ ≈ 1 − Σ²/3` (small-Σ curvature of tanh) — breaks down at high input variance.
- `Q_fp ≈ g²·Σ²·χ̄` (rough self-consistency) — does not reproduce the paper's exact DMFT bifurcation structure.
- `G_eff = χ̄/(1−β·χ̄)` clamped at denominator 0.05 — introduces a hard nonlinearity near β·χ̄ → 1.

These approximations mean the simplified ODE does not cleanly span all four regimes the way the paper's exact DMFT does. The `NoiseSustainedOscillation` regime (the bulk of the grid) is correctly identified because it's the "easy" case (stable focus + chaotic bulk). The boundary regimes (Static near g_c, IrregularSwitching near Hopf, GlobalLimitCycle past Hopf) are where the approximations break down.

**The classifier's closed-form Hopf discriminant is mathematically correct** — it correctly identifies when the 2×2 Jacobian has complex eigenvalues with positive real part (paper Eq. 56). The issue is calibrating the *margins* (`hopf_margin`, `switching_margin`, `chaos_threshold`) to match a specific ODE's regime boundaries, which requires either:
1. A paper-exact DMFT simulator (the paper's self-consistency equations, not my rough approximations), OR
2. A real-game-domain validation (run the classifier on actual NPC crowd data and check whether the regime predictions match observed crowd behavior).

## Proposed resolution (priority order)

- [ ] **T1: Paper-exact DMFT simulator** — replace the simplified `χ̄`/`Q_fp`/`G_eff` approximations in the PoC with the paper's exact DMFT self-consistency equations (paper §VIII, Eq. 55 with the full `χ̄_eff` and `Q_fp` from the Gaussian expectation over `tanh`). This is the cleanest fix — it removes the PoC simulator as a confound. Estimated effort: 1–2 days (requires implementing the Gaussian expectation numerically).
- [ ] **T2: chaos_threshold calibration** — if T1 doesn't fully resolve the g=1.0 boundary, sweep `chaos_threshold ∈ [0.8, 1.0]` against the (now-exact) simulator and pick the value that maximizes regime agreement. Document the choice.
- [ ] **T3: hopf_margin calibration** — sweep `hopf_margin ∈ [0.05, 0.3]` to separate `IrregularSwitching` from `GlobalLimitCycle` cleanly. The current 0.1 is too low (calls everything past the Hopf boundary a limit cycle).
- [ ] **T4: Real-game-domain validation** (riir-ai follow-up) — once T1–T3 pass on the simulator, validate on actual NPC crowd data (e.g., the riir-ai crowd MCGS or the PlasmaPath navigation crowd). This is where the Super-GOAT claim would be validated or refuted.

## Non-goals

- **Do NOT remove the `mean_field_regime` feature.** The primitive is mathematically sound, ships opt-in, passes G2/G3/G4/G5, and the G1 PoC is INCONCLUSIVE (not FAILED). It's useful as-is for callers who want the closed-form Hopf discriminant without the full regime taxonomy.
- **Do NOT tune margins silently.** Any margin change must be documented in the plan + research note with the simulator evidence.

## TL;DR

The Plan 371 defend-wrong PoC caught real calibration issues (g=1.0 boundary, switching-vs-limit-cycle margin). The classifier's closed-form Hopf discriminant is mathematically correct, but the PoC's simplified ODE simulator is too crude to validate the full four-regime taxonomy. Keep `mean_field_regime` opt-in; resolve via a paper-exact DMFT simulator (T1) + margin recalibration (T2/T3) + real-game-domain validation (T4).
