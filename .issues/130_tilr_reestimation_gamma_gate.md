# Issue 130 — TILR Consumer Wiring: riir-ai reestimation.rs γ-Gated Step Size

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** RESOLVED — Option C (redirect to Issue 128). Closed 2026-07-11.

## Context

Plan 425 shipped the TILR primitive as DEFAULT-ON in `katgpt-core`. The
alignment gate `γ = ‖Πd‖/‖d‖` modulates the step size so that `η = η_base · γ`
— directions that don't align with the invariant subspace get a proportionally
smaller correction.

This issue tracked wiring TILR's γ-gated step size into riir-ai's
`reestimation.rs` path: use the alignment ratio to gate re-estimation step
sizes.

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

## Resolution: Option C (2026-07-11)

**Decision: Option C — redirect to Issue 128.**

### Rationale

1. **The reestimation path is architecturally incompatible with TILR's
   additive-correction model.** TILR refines a state that should *mostly
   persist* (`s' = s + η·d_proj`). Batch extract-and-replace wholesale
   overwrites the direction — there is no persistent state to refine. Option B
   (redesign to iterative) was correctly rejected as over-engineering: the
   ring-buffer batch estimator's new mean displacement IS the best estimate.

2. **Option A (gate the binary trigger via γ) is semantic stretching.** The
   reestimation trigger uses *coherence* (alignment of current ρ̂ with
   observations) as its signal — which is already an alignment metric. TILR's
   γ = ‖Πd‖/‖d‖ measures projection of a *direction vector* onto a *basis*.
   Forcing γ into the trigger would require defining what "direction" and
   "basis" mean in the reestimation context, producing a metric that
   duplicates what coherence already provides. No principled gain.

3. **The TILR γ-gate is already correctly applied where there IS an additive
   update.** Plan 438 (Issue 128 Approach B) wired `TilrPersonalityBridge`
   into the committed_blend HLA path: `refine_dz` applies
   `dz += η_base · γ · d_proj` after `tick_committed_blend`, with bit-identical
   no-harm when dz ⊥ basis. This is exactly the use case TILR was designed
   for — graceful refinement of a state that should mostly persist.

4. **The CCE reestimation trigger** (`cce_runtime/reestimation_trigger.rs`)
   already has its own well-designed binary gate (`coherence < tau_reest` +
   cooldown). It composes cleanly with the latent functor's
   `ReestimationScheduler`. No TILR involvement is needed or beneficial here.

### What this means

- Issue 130 is **closed**. No code changes to `reestimation.rs`.
- The TILR consumer-wiring value is captured by Issue 128 (committed_blend HLA
  path, Plan 438) and Issue 129 (neuron-db shard refinement).
- The TILR family doc (`.docs/05_adaptation/tilr_subspace_family.md`) has been
  updated to reflect this resolution.

## Tasks

- [x] **T1** Locate `reestimation.rs` in riir-engine.
      ✅ `latent_functor/reestimation.rs`. No step-size found.
- [x] **T2** Reframe decision: **Option C** — redirect to Issue 128.
      ✅ The TILR γ-gate is already wired into committed_blend (Plan 438).
      The reestimation path has no additive update for TILR to gate.
- [-] **T3** Wire `tilr_refine_into` into the re-estimation path.
      N/A — redirected per Option C. No reestimation wiring.
- [-] **T4** Benchmark.
      N/A — redirected per Option C.
- [-] **T5** Promote to default if gain is real.
      N/A — redirected per Option C.

## Cross-references

- `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md` — COMPLETE, DEFAULT-ON
- `katgpt-rs/.research/408_*.md` — TILR research note (GOAT verdict)
- `katgpt-rs/.docs/05_adaptation/tilr_subspace_family.md` — family overview (updated)
- `katgpt-rs/.issues/128_tilr_hla_personality_refinement.md` — the actual TILR consumer wiring (committed_blend HLA path)
- `riir-ai/.plans/438_tilr_hla_personality_refinement.md` — Plan 438 (committed_blend TILR bridge)
