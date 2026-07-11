# Issue 128 — TILR Consumer Wiring: riir-ai HLA No-Harm Personality Refinement

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** BOTH APPROACHES IMPLEMENTED. Approach A (`PriorityTableBandit::prios`,
> CGSP curiosity table) COMPLETE behind `tilr_hla_refinement` (T1–T4, 9 tests).
> Approach B (`CommittedBlendState::z`, committed-blend HLA) COMPLETE behind
> `tilr_personality_refine` (T1–T4, 11 tests, Plan 438). Both are opt-in.
> T5: synthetic personality-divergence benchmarks PASS for BOTH approaches
> (Approach A: 2 tests, Approach B: 2 tests, 2026-07-11) — mechanism proven.
> Approach A calibration harness + dispatch function landed (2026-07-11):
> `TilrCalibrationBuffer` (cold-path ring buffer) + `tick_with_tilr` (engine-
> level dispatch). End-to-end pipeline test PASS. Final default-on promotion
> still pending real-session gain validation. See "T5 status" below.

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
| **Status** | T1–T4 COMPLETE, 12 tests pass (9 G1-G4 + 2 T5 + 1 T5-e2e) + 8 calibration buffer tests; calibration harness + dispatch landed | T1–T4 COMPLETE, 13 tests pass (11 G1-G4 + 2 T5) |

**Both approaches are implemented** and independently useful: Approach A refines
the curiosity allocation (which axes the NPC explores), Approach B refines the
emotional dynamics (how the HLA vector evolves). They do not conflict at the
code level (different modules, different features, different states). T5
synthetic personality-divergence benchmarks PASS for both (2026-07-11) —
mechanism proven. Final default-on promotion for both is deferred pending
real-session personality-divergence gain.

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
      **PARTIAL PROGRESS** (2026-07-11): Synthetic personality-divergence
      benchmarks PASS — mechanism proven on the CGSP priority table.
      - `t5_priority_subspace_amplification_gain`: single-NPC 50-cycle
        trajectory showing TILR amplifies axis-0 (calibrated curiosity axis)
        priority over the reward-driven baseline. The baseline rewards axis 2
        (so axis 2 dominates), but TILR refinement toward axis 0 systematically
        biases curiosity allocation upward on axis 0.
      - `t5_two_npc_priority_divergence_gain`: two-NPC divergence showing TILR
        amplifies cross-NPC priority divergence. Two NPCs with different
        contrastive directions (axis 0 vs axis 1) start identical and receive
        the same rewards — without TILR they stay identical (div=0); with TILR
        their priority tables diverge, with NPC A higher on axis 0 and NPC B
        higher on axis 1.
      - `t5e2e_calibration_buffer_pipeline`: end-to-end pipeline test using
        `TilrCalibrationBuffer` to collect snapshot pairs from simulated epoch
        boundaries, calibrate the TILR basis via SVD, and verify the gain over
        the reward-driven baseline. Proves the full pipeline: epoch snapshots →
        buffer.push → try_calibrate → refine_apply.
      **CALIBRATION HARNESS LANDED** (2026-07-11):
      - `TilrCalibrationBuffer` (new module `cgsp_runtime/tilr_calibration.rs`):
        cold-path FIFO ring buffer that accumulates `(good, bad)`
        `CuriosityPrioritySnapshot` pairs at epoch boundaries and calibrates an
        `HlaTilrState` via `calibrate_from_snapshots` when enough pairs are
        collected. 8 unit tests (push, eviction, is_ready, try_calibrate,
        rank-1/rank-2 basis, clear, default).
      - `tick_with_tilr` (new free function in `tilr_refinement.rs`): engine-
        level dispatch wiring — calls `NpcCgspRuntime::tick` (absorb +
        renormalize) then `HlaTilrState::refine_apply` on the live priority
        table. Returns `(TickReport, γ)`. The no-harm contract holds (orthogonal
        direction → γ=0 → bit-identical priorities).
      Both are gated on `tilr_hla_refinement` and re-exported from
      `cgsp_runtime`.
      12/12 tilr_refinement tests pass (9 G1-G4 + 2 T5 + 1 T5-e2e), 8/8
      tilr_calibration tests pass, 0 regressions. Release-mode G4 perf smoke
      passes (debug-mode G4 is a timing flake at 1759ns vs 1000ns budget —
      release passes cleanly). 13/13 Approach B tests pass (no regression).
      **REMAINING for default-on promotion**: real-session validation that the
      calibrated subspace (from contrastive priority-snapshot differences at
      epoch boundaries) captures meaningful curiosity directions in production
      game sessions. The calibration harness + dispatch function are now
      available for game-session-layer wiring — the game tick calls
      `tick_with_tilr` and the epoch-boundary handler pushes to
      `TilrCalibrationBuffer`. The synthetic + end-to-end benchmarks prove the
      mechanism; the real-session benchmark confirms the calibration is
      semantically valid.

## Tasks (Approach B — Plan 438, `committed_blend::z`)

- [x] **T1** Identify where HLA personality states are updated in riir-engine.
      ✅ `tick_committed_blend` in `committed_blend/mod.rs:406`. `dz_out` is the
      TILR direction. See Plan 438 for full findings.
- [x] **T2** Collect or simulate contrastive differences from freeze/thaw
      snapshots. Document the calibration data source.
      ✅ `contrastive_z_difference` helper + `build_difference_refs` +
      `TilrPersonalityBridge::from_differences` (SVD calibration from z-snapshot
      pairs). Documented in the module-level rustdoc.
- [x] **T3** Wire `tilr_refine_into` into the update path behind a feature flag.
      ✅ `tilr_personality_refine` feature + `TilrPersonalityBridge::refine_dz`
      (post-tick in-place dz refinement, additive model). Module at
      `committed_blend/tilr_bridge.rs`.
- [x] **T4** Benchmark: verify zero-harm on non-aligned directions, measurable
      refinement on aligned directions.
      ✅ GOAT G1–G4 PASS (11 tests): G1 no-harm bit-identity (orthogonal + zero
      dz), G2 alignment-gated refinement (full + partial alignment), G3
      construction (from_differences round-trip + from_basis orthonormality
      validation + error rejection), G4 perf smoke (<1µs/call at d=8).
      50/50 committed_blend tests pass (11 new + 39 pre-existing), 0 regressions.
- [-] **T5** If the gain is real and modelless → promote to default-on.
      **PARTIAL PROGRESS** (2026-07-11): Synthetic personality-divergence
      benchmarks PASS — mechanism proven.
      - `t5_personality_subspace_amplification_gain`: single-NPC multi-tick
        trajectory showing TILR amplifies axis-0 (valence) dynamics while axes
        1-7 are bit-identical (no z[0] feedback in default library).
      - `t5_two_npc_personality_divergence_gain`: two-NPC divergence showing
        TILR amplifies personality divergence between NPCs with different
        committed pi vectors (crowd-scale emergent personality gain).
      Both tests pass under `--features tilr_personality_refine`. 13/13
      tilr_bridge tests pass (11 G1-G4 + 2 T5), 0 regressions.
      **REMAINING for default-on promotion**: real-session validation that the
      calibrated subspace (from contrastive z-snapshot differences at re-commit
      events) captures meaningful personality directions in production game
      sessions. The synthetic benchmark proves the amplification mechanism; the
      real-session benchmark confirms the calibration is semantically valid.
      **DISPATCH PATH LANDED** (Plan 439, 2026-07-11): the CommittedBlend
      → riir-games NPC tick dispatch is now wired (`tick_committed_blend`,
      Phase 2e-cb). Real-session validation is unblocked — NPCs can now opt
      into CommittedBlend mode via `ensure_committed_blend_state` and evolve
      via `f_pi(z)` in production game sessions. The TILR bridge can be
      constructed and calibrated from real re-commit event z-snapshots.
      See `riir-ai/.plans/439_committed_blend_riir_games_dispatch.md`.
      **TILR REFINEMENT DISPATCH LANDED** (Plan 440, 2026-07-11): the TILR
      `refine_dz` call is now wired into the production `tick_committed_blend`
      sub-phase (Phase 2e-cb-tilr), gated on `tilr_personality_refine`. NPCs
      with a calibrated `TilrPersonalityBridge` (set via `set_tilr_bridge`)
      now get the additive correction `dz += η_base·γ·d_proj` applied to their
      committed-blend dz BEFORE integration — in the production game tick.
      The no-harm contract holds (orthogonal dz → γ=0 → bit-identical
      pass-through). The only remaining piece for T5 promotion is the
      cold-path calibration harness (z-snapshot ring buffer at re-commit
      events → `from_differences` SVD → `set_tilr_bridge`), which is a
      game-session-layer concern, not an engine concern.
      See `riir-ai/.plans/440_tilr_bridge_committed_blend_dispatch_wiring.md`.

## Cross-references

- `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md` — COMPLETE, DEFAULT-ON
- `katgpt-rs/.research/408_*.md` — TILR research note (GOAT verdict)
- `katgpt-rs/.docs/adaptation/tilr_subspace_family.md` — family overview
- `riir-ai/.plans/438_tilr_hla_personality_refinement.md` — implementation plan
- `riir-neuron-db/.plans/317_tilr_consolidation_wiring.md` — sibling wiring (Issue 129, COMPLETE)
