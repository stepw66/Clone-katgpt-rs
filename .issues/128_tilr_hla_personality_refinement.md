# Issue 128 — TILR Consumer Wiring: riir-ai HLA No-Harm Personality Refinement

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** TWO PARALLEL T1 INVESTIGATIONS — both found valid personality-state
> targets. Approach A (`PriorityTableBandit::prios`, CGSP curiosity table) is
> IMPLEMENTED behind `tilr_hla_refinement` (T1–T4 COMPLETE, 9 tests pass).
> Approach B (`CommittedBlendState::z`, committed-blend HLA) is planned in
> Plan 438 (all tasks unchecked). See "Two valid targets" below.

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

## Two valid targets

Two concurrent T1 investigations independently found valid "HLA personality
state" targets — they are **different states** and could coexist or one can be
chosen:

| | Approach A (this issue, COMPLETE) | Approach B (Plan 438, planned) |
|---|---|---|
| **State** | `PriorityTableBandit::prios` (CGSP curiosity table, d=8) | `CommittedBlendState::z` (committed-blend HLA, d=8) |
| **Semantics** | per-axis curiosity-drive weights (which axes does this NPC explore) | committed emotional/personality latent (5 synced scalars are sigmoid projections of it) |
| **Direction** | contrastive priority diff between epoch snapshots | `dz_out` from `tick_committed_blend` (the HLA evolution delta) |
| **Freeze/thaw** | ALREADY wired via `CuriosityPrioritySnapshot.priorities` | needs NEW `z_snapshot` capture at re-commit |
| **Module** | `cgsp_runtime/tilr_refinement.rs` | `committed_blend/tilr_bridge.rs` (planned) |
| **Feature** | `tilr_hla_refinement` (= `cgsp_runtime` + `tilr_invariant_subspace` + `subspace_phase_gate`) | `tilr_personality_refine` (planned, = `tilr_invariant_subspace`) |
| **Status** | T1–T4 COMPLETE, 9 tests pass | all tasks `[ ]` |

**Approach A is further along** and needs no new infrastructure (the priority
table already freeze/thaws via the snapshot). **Approach B targets a state with
richer runtime semantics** (the committed blend produces the synced emotion
scalars). The user should decide whether to (1) keep Approach A only, (2) pursue
Approach B too as a complementary refinement on a different state, or (3)
consolidate. They do not conflict at the code level (different modules, different
features, different states).

## Tasks (Approach A — `tilr_hla_refinement`)

- [x] **T1** Identify where HLA personality states are updated in riir-engine.
      ✅ The personality state is `PriorityTableBandit::prios: Vec<f32>`
      (`crates/riir-engine/src/cgsp_runtime/runtime.rs:84`), one weight per
      `HlaCuriosityDirection` axis (d=8). Updated each cycle via decayed `absorb`
      (`p ← p·decay + reward`, L139) + max-renormalize. Frozen/thawed via
      `CgspLoop::snapshot()`/`restore()` → `CuriosityPrioritySnapshot.priorities`.
      The 64-dim HLA direction *pool* is frozen by design (`restore` does not
      mutate it), so the priority table is the personality state to refine.
- [x] **T2** Collect or simulate contrastive differences from freeze/thaw
      snapshots. Document the calibration data source.
      ✅ Calibration source = `δ_t = snapshot_good.priorities −
      snapshot_bad.priorities` across `(good, bad)` epoch-checkpoint pairs.
      `HlaTilrState::calibrate_from_snapshots` computes the differences and
      calls `discover_invariant_subspace(diffs, τ=0.90)` to build the basis `U_r`.
      `contrastive_direction_from_snapshots` builds a single per-instance
      direction `good − bad` for the online refine step.
- [x] **T3** Wire `tilr_refine_into` into the update path behind a feature flag.
      ✅ New module `crates/riir-engine/src/cgsp_runtime/tilr_refinement.rs`,
      gated on `tilr_hla_refinement` (= `cgsp_runtime` +
      `katgpt-core/tilr_invariant_subspace` + `katgpt-core/subspace_phase_gate`).
      `HlaTilrState::refine_apply` calls `tilr_refine_apply` on the live priority
      table as a post-step after `cycle()`. Module registered in `mod.rs`;
      re-exports `HlaTilrConfig`, `HlaTilrState`, `contrastive_direction_from_snapshots`,
      `DEFAULT_PRIORITY_DIM`.
- [x] **T4** Benchmark: verify zero-harm on non-aligned directions, measurable
      refinement on aligned directions. Gate: no-harm must be bit-identical.
      ✅ GOAT G1–G4 PASS (9 tests, `--no-default-features --features
      tilr_hla_refinement`):
      - G1 no-harm: orthogonal direction → γ=0 → bit-identical priorities
        (`g1_no_harm_when_direction_orthogonal_to_basis`, `g1_no_harm_zero_direction`).
      - G2 alignment gate: in-subspace direction → γ=1, full correction
        (`g2_refine_along_in_subspace_direction`); partial alignment → γ=0.5
        (`g2_gamma_between_zero_and_one_for_partial_alignment`).
      - G3 calibration round-trip: `calibrate_from_snapshots` on synthetic pairs,
        contrastive direction arithmetic, empty-pairs + mismatched-length rejection.
      - G4 perf smoke: `refine_apply` d=8,r=1 under 1000ns/call (lower-level
        primitive benches at 24.7ns for d=768; d=8 is far cheaper).
      `cargo check -p riir-engine --features tilr_hla_refinement --lib` clean.
      Feature isolation confirmed (default check clean, module properly gated).
- [-] **T5** If the gain is real and modelless → promote to default-on.
      DEFERRED: G1–G4 pass and the refinement is pure modelless linear algebra,
      but promotion to DEFAULT-ON needs a real-session personality-divergence
      gain (does TILR refinement measurably improve NPC personality coherence
      / curiosity-axis distribution over a long game session?). The feature is
      safe to ship opt-in — the no-harm contract means enabling it can never
      corrupt personalities that don't match the calibration. A real-game
      benchmark (epoch-snapshot calibration + personality-divergence measurement
      vs no-refinement baseline) is the promotion gate; tracked as a riir-ai
      follow-up.

## Tasks (Approach B — Plan 438, `committed_blend::z`)

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
