# Issue 128 — TILR Consumer Wiring: riir-ai HLA No-Harm Personality Refinement

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** OPEN — Plan 438 created (riir-ai), T1 investigation complete, phased implementation planned

## Context

Plan 425 shipped the TILR (Trajectory-Invariant Latent Refinement) primitive as
DEFAULT-ON in `katgpt-core`. The primitive (`tilr_refine_into`) projects a
contrastive direction onto a frozen SVD basis and modulates the step size by
`γ = ‖Πd‖/‖d‖`, with a strict bit-identical no-harm guarantee when `γ→0`.

This issue tracks wiring TILR into riir-ai's HLA (Hierarchical Latent
Attention) personality refinement path: use TILR to refine NPC personality
states along validated "personality" axes (the invariant subspace discovered
from contrastive good/bad epoch pairs), leaving all other latent dimensions
untouched.

## Why TILR fits this use case

The no-harm contract is the key property: if a personality direction doesn't
align with the calibrated invariant subspace, the correction is a bit-identical
no-op. This means TILR can be applied defensively — it will never corrupt
personality states that don't match the calibration data.

## Proposed integration

1. At NPC initialization (or freeze/thaw swap), collect contrastive differences
   from a frozen reference pair (two epoch checkpoints, or two personality
   snapshots).
2. Call `discover_invariant_subspace(&diffs, 0.90)` to produce the basis `U_r`.
3. At each personality update step, call `tilr_refine_into` with the per-instance
   contrastive direction and `eta_base ∈ [0.1, 0.3]`.

## Tasks

- [x] **T1** Identify where HLA personality states are updated in riir-engine.
      ✅ `tick_committed_blend` in `committed_blend/mod.rs:406`. `dz_out` is the
      TILR direction. See Plan 438 for full findings.
- [ ] **T2** Collect or simulate contrastive differences from freeze/thaw
      snapshots. Document the calibration data source.
      Planned in Plan 438 Phase 2 (z-snapshot at re-commit).
- [ ] **T3** Wire `tilr_refine_into` into the update path behind a feature flag.
      Planned in Plan 438 Phase 1+3 (`tilr_personality_refine` feature +
      `TilrPersonalityBridge` module).
- [ ] **T4** Benchmark: verify zero-harm on non-aligned directions, measurable
      refinement on aligned directions.
      Planned in Plan 438 Phase 3 (GOAT gate).
- [ ] **T5** If the gain is real and modelless → promote to default-on.

## Cross-references

- `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md` — COMPLETE, DEFAULT-ON
- `katgpt-rs/.research/408_*.md` — TILR research note (GOAT verdict)
- `katgpt-rs/.docs/adaptation/tilr_subspace_family.md` — family overview
- `riir-ai/.plans/438_tilr_hla_personality_refinement.md` — implementation plan
- `riir-neuron-db/.plans/317_tilr_consolidation_wiring.md` — sibling wiring (Issue 129, COMPLETE)
