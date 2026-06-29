# Benchmark 342: Latent Trajectory Geometry — Game-Related Gate

> **Plan:** [katgpt-rs/.plans/342_latent_trajectory_geometry.md](../.plans/342_latent_trajectory_geometry.md)
> **Research:** [katgpt-rs/.research/324_Trajectory_Geometry_Transformer_Layers.md](../.research/324_Trajectory_Geometry_Transformer_Layers.md)
> **Source paper:** [arXiv:2606.09287](https://arxiv.org/abs/2606.09287) — Pandey et al., *Trajectory Geometry of Transformer Representations Across Layers*
> **Date:** 2026-06-29
> **Feature gate:** `latent_trajectory_geometry` (opt-in, NOT default)
> **Status:** Phase 1–3 PASS. Promotion candidate — follow-up router-integration plan TBD.

---

## Goal

Prove the curvature signal from Research 324 carries information that the most
natural difficulty proxy (trajectory **length**) does not, on a game-realistic
two-attractor-basin scenario.

## Setup

Three trajectory classes, each consisting of K=20 decisions of fixed magnitude
`step_mag=0.3` (so total length is approximately equal by construction). The
DIRECTION of each decision differs by class:

| Class | Direction pattern | Expected curvature |
|---|---|---|
| **oscillation** | flips ±π each tick (ping-pong, no commitment) | ≈ π (max) |
| **committed** | constant direction (monotonic commitment) | ≈ 0 |
| **drift** | rotates smoothly (0.1 rad/step, exploration without flip) | ≈ 0.1 |

N=50 trajectories per class, seeded xorshift32 + Box-Muller Gaussian noise
(sigma=0.02). Reproducible (seed=42 base).

## Gate output (visible proof)

```
=== Latent Trajectory Geometry — Game-Related Gate (Plan 342 Phase 3) ===

Scenario: agent takes K=20 decisions of fixed magnitude step=0.3,
          direction pattern differs by class. N=50 trajectories per class.
          (noise sigma=0.02, drift angle=0.1 rad/step)

Trajectory class     | mean length | mean curvature (rad)
---------------------|-------------|----------------------
oscillation (flip)   |     6.007   |        3.065
committed (constant) |     6.014   |        0.079
drift (rotate)       |     6.029   |        0.113

Gate G3.1 (curvature separates osc from committed):  PASS
  osc curvature (3.065) - committed curvature (0.079) = +2.986 rad (>= 0.5)
Gate G3.2 (length is blind to the pattern):          PASS
  |osc length (6.007) - committed length (6.014)| / committed = 0.001 (<= 0.15)
Gate G3.3 (drift sits between, control):             PASS
  committed (0.079) < drift (0.113) < oscillation (3.065)

Verdict: curvature signal catches the oscillation pattern that length misses.
         Promotion candidate for router integration (follow-up plan).
```

## GOAT gate summary

| Gate | Threshold | Result | Status |
|---|---|---|---|
| **G1** correctness | Phase 2 formula tests (length scaling, π/2 turn, π reversal, parallel/diverging bifurcation, zero-vector defensive) | 18/18 unit tests pass | ✅ PASS |
| **G2** perf | `from_states` over 100-step × 32-dim trajectory < 5 µs (single-pass fold, no allocation) | (bench not yet landed; primitive is O(L·d) streaming fold by construction) | ⏸ DEFERRED — diagnostic, not hot-path |
| **G3** visible game-related proof | G3.1 curvature gap ≥ 0.5 rad; G3.2 length-diff ratio ≤ 0.15; G3.3 drift ordering | G3.1 +2.986 rad, G3.2 ratio 0.001, G3.3 ordering correct | ✅ PASS |
| **G4** no-regression | `cargo check -p katgpt-core` clean without feature; default tests unaffected | Clean compile without feature; 673 existing tests filtered (untouched) | ✅ PASS |
| **G5** feature isolation | Compiles with and without `latent_trajectory_geometry`; zero overhead when off | `cargo check --all-features` clean; `cargo check` (no feature) clean | ✅ PASS |

## What this proves

**Length is blind (6.007 vs 6.014, ratio 0.001) to the direction-flip pattern;
curvature cleanly separates it (3.065 vs 0.079, gap +2.986 rad ≈ π).**

A difficulty router that uses only trajectory length cannot distinguish an NPC
that ping-pongs between two attractor basins without committing from one that
walks monotonically toward a single basin — both produce the same total
displacement. The curvature signal catches this failure mode directly.

## What this does NOT prove

1. **Does not prove the paper's transformer-layer curvature result transfers
   to HLA emotion trajectories.** The gate uses a synthetic 2-D construction,
   not actual HLA evolution. The realistic damped-oscillation sanity test
   (`t3_realistic_damped_oscillation_sanity`) confirms the signal is present
   on the more realistic "pulled toward basins" model, but does not calibrate
   a threshold for production HLA routing.
2. **Does not prove curvature-augmented routing beats the incumbent signal on
   a real routing-quality benchmark.** That is the follow-up plan's gate. This
   gate only proves the curvature signal carries independent information that
   length lacks.
3. **Does not prove the conformal-naive floor is beaten.** The "Report the
   Floor" rule (Research 322) does NOT apply — the metrics are geometric
   measurements (length, angle), NOT probabilities / confidence scores /
   predictive intervals.

## Promotion decision (Plan 342 T4.3)

**G3 passes.** Primitive is a validated diagnostic. **Stays opt-in** in this
plan. A follow-up plan (TBD) should wire `mean_curvature` as a secondary
signal into a difficulty-aware router (`CollapseAwareAdaptiveThinking` Plan
212, or a new allocator), with its OWN gate: curvature-augmented routing
beats length-only on a routing-quality benchmark.

## Reproduction

```bash
CARGO_TARGET_DIR=/tmp/plan342 cargo test -p katgpt-core \
    --features latent_trajectory_geometry \
    --lib latent_trajectory_geometry::tests::t3_visible_game_related_gate \
    -- --nocapture
```

## TL;DR

Gate PASSES. Curvature signal catches the direction-flip oscillation pattern
that length is structurally blind to (length ratio 0.001, curvature gap +2.986
rad). Primitive ships opt-in as a validated diagnostic; router integration is
a separate follow-up plan with its own routing-quality gate.
