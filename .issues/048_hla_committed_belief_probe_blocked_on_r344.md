# Issue 048: HLA Committed-Belief Probe (F4) — Speculative, Blocked on R344 Re-validation

**Date:** 2026-07-07
**Source plan:** [katgpt-rs/.plans/406_renoise_ce_self_verifier.md](../.plans/406_renoise_ce_self_verifier.md) (Phase 5, P5.4)
**Type:** Speculative follow-up (do NOT plan yet)
**Status:** BLOCKED — requires R344 re-validation before any implementation

## Summary

The renoise-CE self-verifier (Plan 406) applies a consistency-ensembling perturbation to the input and measures output stability. The F4 fusion extends this pattern to the **committed HLA belief state**: perturb the committed HLA state, re-resolve one cognition step, and measure the drift. If the committed belief is a stable attractor, the drift should be small; if it's near a bifurcation boundary, the drift reveals fragility.

## Why this is blocked (the R344 null-result caveat)

Research 344 (`.research/344_Implicit_Fixed_Point_RNN_Convergence_Halting.md`) established that **modelless fixed-point / attractor approaches fail empirically** in this codebase:

- The `AttractorKernel` (MicroRecurrentBeliefState, Plan 276) benchmarked at **569× more flip-flops** than the leaky integrator (G2.1 coherence failure) and **~273 ns** (G1.4 latency failure).
- The paper's RNN-equivalence theorem (Theorem 1) requires *generic trained weights* — the modelless attractor we shipped does not satisfy that precondition.
- R344's verdict: **"Not Super-GOAT, not GOAT"** — no plan created.

The F4 probe assumes the committed HLA state is a **stable fixed point** worth probing for fragility. If the committed belief is NOT a fixed-point attractor (per R344's null result), the probe measures noise, not signal. The probe is only meaningful if the committed-belief resolution is confirmed to have attractor-like stability properties.

## Unblocking condition

R344 must be re-validated with a **positive** result before this issue can become a plan. Specifically:

1. Confirm that committed HLA state resolution (the `CommittedBlendState` freeze/thaw lifecycle, Plan 336) exhibits **attractor-like stability** — small input perturbations produce small output drift, monotonically, without flip-flopping.
2. If yes → the F4 probe is meaningful; file a plan.
3. If no (the R344 null result holds) → close this issue as "not viable — committed HLA is not a fixed-point attractor, so perturbation-probing measures noise."

## Proposed probe design (if unblocked)

```
fn hla_committed_belief_drift(
    state: &CommittedBlendState,
    perturbation_magnitude: f32,
    rng: &mut impl Rng,
) -> f32 {
    // 1. Snapshot the committed pi weights.
    let pi_baseline = state.blend.pi;
    // 2. Perturb: add noise to pi, re-normalize.
    let pi_perturbed = perturb(&pi_baseline, perturbation_magnitude, rng);
    // 3. Re-resolve one cognition step with the perturbed pi.
    let output_perturbed = resolve_one_step(state, &pi_perturbed);
    let output_baseline = resolve_one_step(state, &pi_baseline);
    // 4. Measure cosine drift.
    cosine_distance(&output_perturbed, &output_baseline)
}
```

Gate (if planned): drift < 0.1 at perturbation_magnitude = 0.01 (stable attractor); drift > 0.5 indicates bifurcation boundary.

## Cross-references

- Plan 406 (renoise-CE self-verifier): `.plans/406_renoise_ce_self_verifier.md`
- Research 344 (the blocking null result): `.research/344_Implicit_Fixed_Point_RNN_Convergence_Halting.md`
- Plan 336 (CommittedBlendState — the probe target): riir-ai side
- Plan 276 (MicroRecurrentBeliefState — the prior-art failure): `.benchmarks/276_micro_belief_goat.md`
