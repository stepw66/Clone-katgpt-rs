# Issue 130 — TILR Consumer Wiring: riir-ai reestimation.rs γ-Gated Step Size

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** BLOCKED on design reframe (T1 complete — investigation revealed no step-size η exists in the target path)

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

## T1 Investigation Findings (2026-07-10)

**Blocker discovered:** `reestimation.rs` at
`riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` has **no
step-size computation**. The re-estimation path is closed-form batch
extract-and-replace:

```
f = (1/N) Σ_k (target_k − source_k)   // mean displacement (arithmetic/mod.rs:157)
FunctorEntry::new(direction, ...)       // wholesale replace (reestimation.rs:1281)
```

There is no `s' = s + η·d` additive update — the new direction **replaces**
the old one. TILR's contract (`s' = s + η_base·γ·d_proj`) has no insertion
point because there is no `s` being incrementally updated and no `η`.

`tau_reest` (`DEFAULT_TAU_REEST = 0.4`, table.rs) is a **coherence
threshold for the binary re-estimation trigger**, NOT a step size. It
answers "should we re-estimate?" (binary), not "how big a step?"
(continuous).

### katgpt-core dependency

`katgpt-core` is already a non-optional dep of riir-engine
(`Cargo.toml:26`). TILR is directly callable once
`tilr_invariant_subspace` is forwarded. No `tilr` feature forwarding
exists yet.

### Reframe options (decision needed before T2)

| Option | Description | Effort |
|---|---|---|
| **A** | Use γ to gate the *binary re-estimation decision* (replace `coherence < tau_reest` with `γ > threshold`). Trigger redesign, not step-size gate. Matches Research 408 F2's actual intent. | Medium |
| **B** | Change reestimation to iterative refinement: `direction_new = direction_old + η·γ·(f_extracted)_proj`. Fundamental redesign of extract-and-replace contract. | **High** — touches every FunctorEntry consumer |
| **C** | Redirect to Issue 128 (HLA path). Issue 128's HLA scalar blending already has an additive update with a blend coefficient — a natural η to gate. | Low |

**Recommendation:** Option A or C. Option B is over-engineering — the
extract-and-replace pattern is correct for a ring-buffer batch estimator
(the new mean displacement IS the best estimate). TILR's γ-gate is
designed for *additive corrections to a state that should mostly persist*,
which matches HLA personality blending (Issue 128) far better than batch
functor re-estimation.

## Tasks

- [x] **T1** Locate `reestimation.rs` in riir-engine.
      ✅ `latent_functor/reestimation.rs`. No step-size found — see findings above.
- [-] **T2** Define what constitutes a "beneficial re-estimation direction".
      BLOCKED on reframe decision (Options A/B/C above).
- [-] **T3** Wire `tilr_refine_into` into the re-estimation path.
      BLOCKED on T2.
- [-] **T4** Benchmark.
      BLOCKED on T3.
- [-] **T5** Promote to default if gain is real.
      BLOCKED on T4.

## Cross-references

- `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md` — COMPLETE, DEFAULT-ON
- `katgpt-rs/.research/408_*.md` — TILR research note (GOAT verdict)
- `katgpt-rs/.docs/adaptation/tilr_subspace_family.md` — family overview
