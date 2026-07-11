# Subspace-Projection Family — Alignment-Gated Correction (TILR)

> **Plan 425** (TILR), cross-referencing Plan 412 (`subspace_steering`), Plan 423
> (`spectral_rewire`), Plan 152 (`river_valley`), Plan 301 (`thin_svd_into`).
> Research 408 (arXiv:2606.29164, ICML 2026 Mech Interp Workshop).

## The family

The codebase ships four subspace-projection primitives, each with a different
step-size strategy. They share the core operation — project a direction `d`
onto a frozen orthonormal basis `U_r` — but differ in how the correction is
scaled:

| Member | Feature | Step size | When γ→0 | GOAT |
|---|---|---|---|---|
| **TILR** (alignment-gated) | `tilr_invariant_subspace` | `η_base · γ` where `γ = ‖Πd‖/‖d‖` | **Bit-identical no-harm** (η=0.0 exactly) | ✅ DEFAULT-ON |
| `subspace_steering` (ungated, per-axis) | `subspace_steering` | Fixed `α_k` per basis axis | Fixed correction (no alignment gate) | ✅ DEFAULT-ON |
| `spectral_rewire` (ungated, projection) | `spectral_rewire` | Fixed projection (no step modulation) | Identity (full projection = 0 if orthogonal) | ✅ opt-in |
| `river_valley::subspace_ratios` (diagnostic) | `river_valley` | N/A (diagnostic metric only) | Reports `r_dom` for logging, never gates | ✅ DEFAULT-ON |

**TILR is the alignment-gated member.** The novel integration: using the
alignment ratio `γ` (the same `r_dom` metric that `river_valley` computes as a
diagnostic) to modulate the step size, with a strict bit-identical no-harm
guarantee when the direction is orthogonal to the invariant subspace.

## When to use which

- **TILR** — when you want graceful degradation: the correction should vanish
  for directions that don't align with the known invariant subspace, but apply
  fully for directions that do. Use case: HLA no-harm personality refinement
  (correct only along validated "personality" axes, leave everything else
  untouched).
- **`subspace_steering`** — when you want fixed-strength multi-axis steering:
  each basis axis gets its own `α_k`, applied regardless of alignment. Use case:
  multi-dimensional latent steering (Plan 309 generalization).
- **`spectral_rewire`** — when you want spectral filtering of deltas: project
  out the non-dominant components of a weight delta. Use case: LoRA-scale row
  reshaping via cached SVD index.

## The TILR pipeline

```
Offline (once):                         Online (per step):
┌─────────────────────┐                ┌──────────────────────────┐
│ δ_t = h_good - h_bad │                │ direction d (per-instance)│
│ (N contrastive diffs)│                │ state s                   │
└─────────┬───────────┘                └──────────┬───────────────┘
          │                                       │
          ▼                                       │
┌─────────────────────┐                         │
│ thin_svd_into (P301) │                         │
│ Δ = U_r Σ_r V_rᵀ     │                         │
└─────────┬───────────┘                         │
          │                                       │
          ▼                                       │
┌─────────────────────┐                         │
│ select r by τ=0.90   │ ──basis U_r──▶  ┌───────┴───────────────┐
│ variance retention   │                 │ tilr_refine_into (P425)│
└─────────────────────┘                 │ d_proj = U_r(U_rᵀd)    │
                                        │ γ = ‖d_proj‖/‖d‖        │
                                        │ s' = s + η_base·γ·d_proj│
                                        └────────────────────────┘
```

The online step is `O(d·r)` — two matvecs + one SAXPY. For `d=768, r=12` this
is ~9.2K FMAs, negligible vs `O(d²)` attention.

## The no-harm contract (load-bearing)

When `γ = 0` (direction orthogonal to basis), `η = 0` and `s' = s`
**bit-identically**. This is enforced by clamping `‖d_proj‖² < ε → γ = 0.0`
exactly (not `γ ≈ 1e-38`), so `η` is exactly `0.0`.

This makes TILR safe to apply as a "no-harm refinement" — if the per-instance
direction doesn't align with the calibrated invariant subspace, the correction
is a no-op. This is the property that distinguishes TILR from the ungated
family members.

## Reuse map

| Operation | Source | Notes |
|---|---|---|
| SVD → `U_r` (offline) | `thin_svd_into` (Plan 301) or `discover_invariant_subspace` (Plan 425 Phase 3) | Basis is an INPUT to TILR |
| Alignment ratio `γ` | `subspace_ratios` (Plan 152) | 5-line duplication per Plan 425 DRY decision (B) |
| Subspace projection `Πd` | `spectral_rewire` (Plan 423), `subspace_steering` (Plan 412) | Same projection math |
| SIMD dot products | `simd_dot_f32` (`katgpt-types/simd`) | Projection coefficients + norms |

## Consumer wiring (follow-up issues)

Consumer integration status:

- **Issue 128** — riir-ai HLA no-harm personality refinement. **T1–T5 COMPLETE**
  (two approaches: CGSP priority table via `tilr_hla_refinement`, committed_blend
  HLA vector via `tilr_personality_refine` / Plan 438). **Engine-level
  calibration harness + dispatch COMPLETE for both approaches** (2026-07-11):
  Approach A ships `TilrCalibrationBuffer` + `tick_with_tilr`; Approach B ships
  `TilrPersonalityCalibrationBuffer` (Plan 440 follow-up). **Real-session T5
  validation landed for both** (2026-07-11): Approach A
  `t5_real_session_calibration_buffer_with_runtime` (real `NpcCgspRuntime` +
  buffer + `tick_with_tilr`); Approach B
  `tilr_real_session_calibration_buffer_pipeline` (real `MapInstance` + buffer
  + `set_tilr_bridge` + production dispatch). **Approach B PROMOTED TO
  DEFAULT-ON** (2026-07-11): re-commit event wiring landed
  (`recommit_on_major_emotion` trigger in Phase 2e-cb-recommit, emits
  `SimEvent::PersonalityRecommit` on extreme emotion states). GOAT G1-G5 all
  PASS. Zero-cost no-op when no bridge set. Approach A (`tilr_hla_refinement`)
  remains opt-in — blocked on CGSP runtime production game-tick wiring
  (Plan 299 Phase 4 GOAT gate, separate concern from Issue 128).
- **Issue 129** — riir-neuron-db freeze/thaw shard refinement. **T1–T4 COMPLETE**.
  T5 deferred.
- **Issue 130** — riir-ai `reestimation.rs` γ-gated step size. **RESOLVED —
  Option C (redirect to Issue 128).** The reestimation path is closed-form
  batch extract-and-replace with no additive step-size for TILR to gate. The
  TILR γ-gate is already correctly applied in the committed_blend HLA path
  (Plan 438), where there IS an additive update. Closed 2026-07-11.
