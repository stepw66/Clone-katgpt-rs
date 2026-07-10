# Issue 130 — TILR Consumer Wiring: riir-ai reestimation.rs γ-Gated Step Size

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** OPEN (deferred — TILR primitive ships DEFAULT-ON, this wires it)

## Context

Plan 425 shipped the TILR primitive as DEFAULT-ON in `katgpt-core`. The
alignment gate `γ = ‖Πd‖/‖d‖` modulates the step size so that `η = η_base · γ`
— directions that don't align with the invariant subspace get a proportionally
smaller correction.

This issue tracks wiring TILR's γ-gated step size into riir-ai's
`reestimation.rs` path: use the alignment ratio to gate re-estimation step
sizes, so re-estimation only applies meaningfully when the update direction
aligns with the calibrated invariant subspace.

## Why TILR fits this use case

Re-estimation (the process of adjusting latent state estimates based on new
observations) can be destabilized by updates in directions that the model
hasn't calibrated for. TILR's γ-gate naturally suppresses these: if the
re-estimation direction doesn't align with the known invariant subspace, the
step size is scaled down proportionally, preventing destabilization.

This is a more principled alternative to fixed step-size clipping: instead of
clipping the magnitude, TILR gates by *directional alignment* — large updates
in calibrated directions are allowed, while even small updates in uncalibrated
directions are suppressed.

## Proposed integration

1. Identify the re-estimation step-size computation in `reestimation.rs`.
2. At calibration time, build the invariant subspace from historical
   re-estimation deltas (the directions in which re-estimation has historically
   been beneficial).
3. Replace the fixed step size `η` with `η_base · γ` where `γ` is the TILR
   alignment ratio.

**Note:** The calibration (which deltas are "beneficial") may need a consumer-
specific definition. This is a riir-ai design decision, not a katgpt-core
concern.

## Tasks

- [ ] **T1** Locate `reestimation.rs` in riir-engine. Identify the step-size
      computation path and the update direction source.
- [ ] **T2** Define what constitutes a "beneficial re-estimation direction"
      (the contrastive pair for TILR calibration). Document the choice.
- [ ] **T3** Wire `tilr_refine_into` (or extract just the γ-gate logic) into
      the re-estimation path behind a feature flag.
- [ ] **T4** Benchmark: compare fixed-step vs γ-gated re-estimation on a
      representative workload. Gate: γ-gated must not degrade convergence on
      aligned directions, and must reduce instability on misaligned directions.
- [ ] **T5** If the gain is real and modelless → promote to default-on.

## Cross-references

- `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md` — COMPLETE, DEFAULT-ON
- `katgpt-rs/.research/408_*.md` — TILR research note (GOAT verdict)
- `katgpt-rs/.docs/adaptation/tilr_subspace_family.md` — family overview
