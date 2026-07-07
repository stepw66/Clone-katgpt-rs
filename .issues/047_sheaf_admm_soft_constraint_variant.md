# Issue 047 — Sheaf-ADMM soft-constraint variant (paper eq. 25)

**Source:** Extracted from Plan 407 Phase 3 T3.3 (post-promotion feature variant).
**Primitive:** `sheaf_admm` (katgpt-dec, **DEFAULT-ON** since 2026-07-07, G1–G6 ALL PASS).
**Parent research:** [`.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md`](../.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md)
**Source paper:** arXiv:2605.31005 eq. 25.

## Why this is an issue, not a plan task

Per `AGENTS.md`: *optimization/refactor tasks go to `.issues/`, not plans*. This is a feature variant (not a correctness fix) of a shipped primitive.

## The variant

The shipped `sheaf_admm_step` enforces hard consensus `Fz = 0` (primal aligns with the harmonic subspace). Paper eq. 25 generalizes this to a **soft constraint**: replace the hard `Fz = 0` with a quadratic penalty `γ/2 · ‖Fz‖²`. This adds one knob `γ`:

- `γ → ∞`: recovers the hard constraint (current behavior).
- `γ` finite: NPCs retain some individual variation — useful when exact consensus is undesirable (e.g., a faction should preserve some disagreement rather than collapse to a single belief).

This is a modelless feature variant (one extra scalar + one extra term in the z-update proximal solve). No training, no backprop.

## Acceptance

- [ ] Implement `sheaf_admm_step_soft` (or a `consensus: ConsensusMode::{Hard, Soft{ gamma }}` enum on the existing entry point).
- [ ] Bench: hard-vs-soft on a synthetic "faction disagreement" scenario — verify soft-mode preserves measurable residual disagreement `‖Fz‖` proportional to `1/γ`, while hard-mode drives it to ~0.
- [ ] Document the `γ` knob semantics. Promote to default-on only if a consumer (riir-ai Crowd MCGS) demonstrates faction-divergence emergent behavior needs it.

## Notes

- Soft-constraint z-update is still gradient descent on the modified energy `(1/2)z^T L_F z + (γ/2)‖Fz‖²` — modelless.
- The `γ` knob crosses the latent→raw bridge only if committed to chain; locally it's a latent scalar (faction-coherence parameter).
