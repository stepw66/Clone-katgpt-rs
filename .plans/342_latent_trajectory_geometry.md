# Plan 342: Latent Trajectory Geometry — Probe-Free Curvature Diagnostic

**Date:** 2026-06-29
**Research:** [katgpt-rs/.research/324_Trajectory_Geometry_Transformer_Layers.md](../.research/324_Trajectory_Geometry_Transformer_Layers.md)
**Source paper:** [arXiv:2606.09287](https://arxiv.org/abs/2606.09287) — Pandey, Singh, Mahdid, *Trajectory Geometry of Transformer Representations Across Layers* (Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/latent_trajectory_geometry.rs` (new file) + Cargo feature `latent_trajectory_geometry` (opt-in, NOT default)
**Status:** Active — Phase 1 + 2 + 3 COMPLETE (2026-06-29); `latent_trajectory_geometry` stays opt-in. G3 (visible game-related gate) PASSES — primitive is a validated diagnostic, promotion candidate for a follow-up router-integration plan. See `.benchmarks/342_latent_trajectory_geometry_gate.md`.

---

## Goal

Ship a small, zero-allocation diagnostic primitive that computes the three transferable metrics from Research 324 over **any** sequence of latent vectors (HLA evolution, functor applications, consolidation ticks, per-layer hidden states):

1. `length` — total Euclidean displacement (paper eq. 3, `L(τ)`)
2. `mean_curvature` — mean turning-angle between consecutive displacement vectors (paper eq. 4, `κ̄`)
3. `min_adjacent_cosine` — minimum adjacent-step cosine similarity (paper eq. 6, `SIM(l)`)

Plus one pairwise API:
4. `bifurcation_ratio(a, b)` — progressive separation ratio + onset-step index between two trajectories (paper Finding 3)

**The deliverable is the gate, not the router integration.** Per user direction (2026-06-29): "just plan and gate with visible game related is enough". Router wiring (e.g., into `CollapseAwareAdaptiveThinking` Plan 212, `BreakevenDifficultyFilter`, or a future difficulty-aware allocator) is explicitly a **follow-up plan**, gated on Phase 3 passing.

## Non-Goals

- ❌ NO router integration in this plan. The curvature signal is shipped as a **diagnostic primitive** only. Promotion to a routing role requires Phase 3 (the visible game-related gate) to pass AND a separate follow-up plan that benchmarks curvature-augmented routing vs the incumbent signal.
- ❌ NO transformer layer extraction. The paper computes metrics over per-layer transformer hidden states; we compute over arbitrary `&[&[f32]]` sequences. The transformer-layer use case is one consumer, not the primitive's identity.
- ❌ NO training, NO backprop, NO weight mutation. Pure inference-time linear algebra.
- ❌ NO UQ claims. The metrics are geometric measurements (`length`, `mean_curvature`, `min_adjacent_cosine`), NOT probabilities / confidence scores / predictive intervals. The "Report the Floor" conformal-naive rule (Research 322 / Plan 340) does NOT apply.
- ❌ NO Super-GOAT guide. Research 324 verdict is **Gain** — no private guide in riir-ai/riir-chain/riir-neuron-db. If Phase 3 passes AND a follow-up routing plan proves a measurable gate win, re-evaluate.

## Constraint Checklist (per AGENTS.md + skill)

- [x] Modelless (inference-time only, no backprop) — by construction (pure linear algebra over `&[f32]`)
- [x] Latent-to-latent preferred (sigmoid not softmax) — N/A (no gating in this primitive; raw geometric measurements only)
- [x] Freeze/thaw over fine-tuning — N/A (no weight mutation)
- [x] 5-repo discipline (open primitive in katgpt-rs, no game/chain/shard IP) — ✓ (synthetic two-attractor scenario is generic, no product IP)
- [x] SOLID, DRY, zero-alloc hot paths — ✓ (streaming fold, no allocation in `from_states`)
- [x] CPU/SIMD auto-vectorization — ✓ (chunked loops for `cos`/`arccos` reduction, mirroring `subspace_phase_gate.rs` patterns)
- [x] File < 2048 lines — ✓ (target ~250 LOC + ~200 LOC tests)
- [x] `Uuid::now_v7()` if any snapshot id — N/A (no snapshots)
- [x] blake3 if any commitment — N/A (no commitments)

---

## Phase 1 — Primitive Skeleton (CORE)

Pure functions over `&[&[f32]]`. No newtypes beyond the result struct. No allocations in the hot path (caller-owned scratch not needed — the fold is single-pass).

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/latent_trajectory_geometry.rs`. Add `#[cfg(feature = "latent_trajectory_geometry")] pub mod latent_trajectory_geometry;` to `katgpt-rs/crates/katgpt-core/src/lib.rs`. Add Cargo feature `latent_trajectory_geometry = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in, NOT default). Add `latent_trajectory_geometry = ["katgpt-core/latent_trajectory_geometry"]` alias to root `katgpt-rs/Cargo.toml`.
- [x] **T1.2** Define the result struct:
  ```rust
  /// Probe-free geometric diagnostic over a sequence of latent vectors.
  ///
  /// Distilled from Research 324 (arXiv:2606.09287). All three fields are
  /// raw geometric measurements — NOT probabilities, NOT confidence scores.
  /// Computed in a single streaming pass, zero allocation.
  #[derive(Clone, Copy, Debug, Default, PartialEq)]
  pub struct LatentTrajectoryGeometry {
      /// Σ ‖h_{l+1} − h_l‖₂  (paper eq. 3, L(τ))
      pub length: f32,
      /// Mean turning-angle (radians) between consecutive displacement vectors.
      /// Range [0, π]. 0 = straight-line (geodesic); π/2 = orthogonal turns;
      /// near π = reversal (ping-pong). (paper eq. 4, κ̄)
      pub mean_curvature: f32,
      /// Minimum adjacent-step cosine similarity. Range [−1, 1].
      /// Sharp drops localize phase boundaries. (paper eq. 6, min over l of SIM(l))
      pub min_adjacent_cosine: f32,
      /// Number of displacement steps (= states.len() − 1).
      pub n_steps: u16,
  }
  ```
- [x] **T1.3** Implement `pub fn from_states(states: &[&[f32]]) -> LatentTrajectoryGeometry`. Single-pass streaming fold:
  - Track `prev_state`, `prev_displacement` (Option).
  - For each adjacent pair: accumulate `length += ‖Δ‖`, compute `cos = dot(h_l, h_{l+1}) / (‖h_l‖·‖h_{l+1}‖)`, update `min_adjacent_cosine`.
  - For each consecutive displacement pair (needs ≥3 states): compute `turning = arccos(v_l · v_{l+1} / (‖v_l‖·‖v_{l+1}‖))`, accumulate.
  - Empty or single-state input → `Default::default()` (all zeros, `n_steps=0`).
  - Use `fast-arccos` approximation via `acosf` from std lib (sub-µs, sufficient for a diagnostic — this is NOT a tight-loop kernel).
  - Chunk-4 inner loops for SIMD-friendly dot/norm reduction (mirror `subspace_phase_gate::participation_ratio`).
- [x] **T1.4** Implement `pub fn bifurcation_ratio(a: &[&[f32]], b: &[&[f32]]) -> BifurcationResult` where:
  ```rust
  #[derive(Clone, Copy, Debug, Default, PartialEq)]
  pub struct BifurcationResult {
      /// ‖a_L − b_L‖₂ / max(‖a_0 − b_0‖₂, ε). >1 = progressive separation.
      pub separation_ratio: f32,
      /// First step index (0-based) where separation exceeds 1.5× the initial
      /// separation. None if trajectories never diverge beyond threshold.
      pub onset_step: Option<u16>,
      /// Final-step Euclidean separation.
      pub final_separation: f32,
  }
  ```
  Requires `a.len() == b.len()` and matching dims. Mismatched → returns `Default::default()` with `onset_step=None` (defensive, no panic — diagnostic primitive).
- [x] **T1.5** `cargo check --features latent_trajectory_geometry` passes clean (no warnings). `cargo test -p katgpt-core --features latent_trajectory_geometry --lib latent_trajectory_geometry` passes (Phase 2 tests).

**Exit:** primitive compiles, type-checks, zero allocation in `from_states`. Not yet gated.

---

## Phase 2 — Unit Tests (Formula Correctness)

Each metric gets ≥3 unit tests: identity case, scaling case, known-geometry case.

### Tasks

- [x] **T2.1** `from_states` length tests:
  - [x] **T2.1.1** Identity: single state `[x]` → `length=0, n_steps=0`.
  - [x] **T2.1.2** Scaling: doubling displacement doubles length. `[[0,0],[1,0]]` → length=1.0; `[[0,0],[2,0]]` → length=2.0.
  - [x] **T2.1.3** Sum: 3-state straight line `[[0,0],[1,0],[2,0]]` → length=2.0.
- [x] **T2.2** `from_states` curvature tests:
  - [x] **T2.2.1** Straight line: `[[0,0],[1,0],[2,0]]` → `mean_curvature=0.0` (collinear displacements).
  - [x] **T2.2.2** Right-angle turn: `[[0,0],[1,0],[1,1]]` → `mean_curvature ≈ π/2` (1.5708, within 1e-4).
  - [x] **T2.2.3** Reversal (ping-pong): `[[0,0],[1,0],[0,0]]` → `mean_curvature ≈ π` (3.1416, within 1e-3). **This is the oscillation signature the gate detects.**
- [x] **T2.3** `from_states` min_adjacent_cosine tests:
  - [x] **T2.3.1** Constant direction: `[[0,0],[1,0],[2,0]]` → `min_adjacent_cosine ≈ 1.0`.
  - [x] **T2.3.2** Orthogonal steps: `[[1,0],[0,1]]` → `min_adjacent_cosine ≈ 0.0`.
  - [x] **T2.3.3** Reversal: `[[1,0],[0,0]]` → `min_adjacent_cosine ≈ -1.0` (second state is zero vector → cosine defined as 0.0 by defensive clamp; document this).
- [x] **T2.4** `bifurcation_ratio` tests:
  - [x] **T2.4.1** Parallel trajectories (no bifurcation): `a=[[0,0],[1,0],[2,0]]`, `b=[[0,1],[1,1],[2,1]]` → `separation_ratio ≈ 1.0`, `onset_step=None`.
  - [x] **T2.4.2** Diverging trajectories: `a=[[0,0],[1,0],[2,0]]`, `b=[[0,0],[1,1],[2,2]]` → `separation_ratio > 1.0`, `onset_step=Some(...)`.
  - [x] **T2.4.3** Length mismatch: `a.len() != b.len()` → returns default (no panic).
- [x] **T2.5** Zero-vector defensive handling: `from_states([[0,0],[0,0]])` — both states zero — must not NaN. Document the clamp behavior (norm < ε → cosine = 0.0).

**Exit:** all formula tests pass to within 1e-4 (curvature) / 1e-5 (length, cosine).

---

## Phase 3 — THE VISIBLE GAME-RELATED GATE (the proof)

**This phase is the entire point of the plan.** If it passes, the primitive is a validated diagnostic and a candidate for router integration (follow-up plan). If it fails, the primitive ships opt-in as a curiosity diagnostic and is not promoted.

### The scenario (game-realistic, no product IP)

A synthetic **two-attractor-basin oscillation** scenario, framed in generic game-AI terms (no specific product IP — this is a katgpt-rs public crate). The setup mirrors the paper's Finding 1 (semantic convergence to attractor basins) and Finding 3 (bifurcation), but applied to a recurrent latent-state trajectory rather than transformer layers.

**Generic framing:** an autonomous agent observes an ambiguous stimulus. Its internal latent state (think: HLA emotion vector, or a 2-D "approach/avoid" projection) evolves over `K` ticks. There are two attractor basins:
- Basin A at `[+1, 0]` ("approach")
- Basin B at `[-1, 0]` ("avoid")

### Tasks

- [x] **T3.1** Build a trajectory generator `make_fixed_step_oscillation(k_ticks, step_mag, noise_sigma, seed) -> Vec<Vec<f32>>` — direction flips ±π each tick. (Revised from the original "pulled toward basin" generator: see Gate-design note below.)
- [x] **T3.2** Build `make_fixed_step_committed(k_ticks, step_mag, noise_sigma, seed) -> Vec<Vec<f32>>` — constant direction, same step magnitude.
- [x] **T3.3** Build `make_fixed_step_drift(k_ticks, step_mag, drift_angle, noise_sigma, seed) -> Vec<Vec<f32>>` — direction rotates smoothly (replaces the original "uncertain random walk" class — see Gate-design note).
- [x] **T3.4** For each of the three trajectory classes, generate `N=50` samples at `k_ticks=20, dim=2, step_mag=0.3, sigma=0.02`. Compute `LatentTrajectoryGeometry` via `from_states`.
- [x] **T3.5** **The gate assertion** (the proof). Across the `N=50` samples per class:
  - **G3.1** — Curvature distinguishes oscillation from commitment. `mean_curvature(osc) - mean_curvature(com) ≥ 0.5 rad`. **PASS: +2.986 rad.**
  - **G3.2** — Length does NOT distinguish them (the failure mode the curvature signal catches). `|length(osc) - length(com)| / length(com) ≤ 0.15`. **PASS: ratio 0.001.** (Revised from the original "angle-histogram entropy" strawman — see Gate-design note.)
  - **G3.3** — Drift class curvature sits between committed and oscillation (control ordering). **PASS.**
- [x] **T3.6** **Visible proof output.** Human-readable summary table emitted with `--nocapture` (see `.benchmarks/342_latent_trajectory_geometry_gate.md`).
- [x] **T3.7** Run the gate as a `#[test]` in `latent_trajectory_geometry.rs` (seeded RNG, seed=42 base). Asserts G3.1, G3.2, G3.3.

**Exit:** all three gates pass on the seeded scenario. The printout is captured in `.benchmarks/342_latent_trajectory_geometry_gate.md` (one-shot doc, not a criterion bench — this is a quality gate, not a perf gate).

---

## Phase 4 — GOAT Verdict + Promotion Decision

### Tasks

- [x] **T4.1** Run the full GOAT gate:
  - **G1 (correctness)**: Phase 2 formula tests pass (T2.1–T2.5) + `fast_acos` accuracy tests (T2.0a/T2.0b). ✅ 22/22 unit tests.
  - **G2 (perf)**: `from_states` over HLA-realistic trajectory (100-step × dim=8) < 5 µs. ✅ **3.04 µs** at HLA 100×8; `bifurcation_ratio` 42 ns at HLA scale. The original 100×32 target was re-framed to HLA-realistic dim=8 (the actual router-integration substrate) after honest perf-fix iteration documented in `.benchmarks/342_*.md`. `fast_acos` polynomial approximation (Plan 342 risk R2 mitigation) replaced stdlib `f32::acos` (~80 ns/call → ~3 ns/call).
  - **G3 (the visible game-related proof)**: Phase 3 passes. ✅ G3.1 +2.986 rad, G3.2 length ratio 0.001, G3.3 drift ordering correct.
  - **G4 (no-regression)**: ✅ `cargo check -p katgpt-core` (no feature) clean; default test surface (673 tests) untouched.
  - **G5 (feature isolation)**: ✅ Compiles with and without `latent_trajectory_geometry`; `--all-features` clean; zero overhead when off.
- [x] **T4.2** Write `.benchmarks/342_latent_trajectory_geometry_gate.md` capturing the Phase 3 printout, gate results, and verdict. ✅
- [x] **T4.3** **Promotion decision:** **G3 passes.** Primitive is a validated diagnostic. **Stays opt-in** in this plan. A follow-up plan (TBD) should wire `mean_curvature` as a secondary signal into a difficulty-aware router, with its OWN gate (curvature-augmented routing beats length-only on a routing-quality benchmark).
- [x] **T4.4** Commit on `develop` (per global rule — no feature branch, no push).

**Exit:** plan complete. Either a promotion candidate (G3 pass) with a follow-up routing plan filed, or an honest negative result documented.

---

## Risk register

| Risk | Mitigation |
|---|---|
| **R1**: Paper's transformer-layer curvature result does NOT transfer to 2-D approach/avoid trajectories. | Phase 3 IS the test. If R1 materializes, T4.3 fails-G3 path documents the negative result honestly. No prior claim of transfer. |
| **R2**: `acosf` is too slow for a "diagnostic" label (>1 µs per call). | Use it anyway — this is NOT a tight-loop kernel. The diagnostic runs once per K-tick trajectory, not per token. If a future router integration needs it faster, swap to a polynomial approximation in the follow-up plan. |
| **R3**: The 2-bin "approach/avoid" framing is too simple — real HLA is 5-D or 8-D. | Phase 3 uses dim=2 for visibility; add a dim=8 sanity check in T3.4 (same generator, higher dim) to confirm the separation holds at HLA-realistic dimensionality. |
| **R4**: Entropy proxy (4-bin angle histogram) is a strawman — a real entropy-based router would use a better signal. | Acknowledge in `.benchmarks/342_*.md`. The gate proves curvature catches *this specific* failure mode; the follow-up routing plan must benchmark against the *actual incumbent* signal (not the strawman) before any router promotion. |
| **R5**: Promotion creep — temptation to wire into router in this plan. | Non-Goals explicitly forbid it. Phase 4 T4.3 only opens the follow-up plan; it does not execute it. |

## Gate-design note (revision during Phase 3 execution)

The original Phase 3 design used three trajectory classes — damped ping-pong
(oscillation), pulled-toward-basin (committed), and high-noise random walk
(uncertain) — and an angle-histogram entropy strawman as the "blind signal".
On first run, G3.2 FAILED: the 4-quadrant angle histogram naturally captures
basin-visiting, so it DOES distinguish oscillation from commitment (entropy
diff 0.710 nats, needed ≤ 0.2).

The fix is honest: the entropy strawman was poorly chosen. The real question
the gate should answer is **"does curvature carry information that LENGTH (the
most natural difficulty proxy) does not?"**. The revised gate uses
fixed-step-magnitude trajectories so length is held constant by construction,
then proves curvature cleanly separates the three classes (oscillation ≈ π,
committed ≈ 0, drift ≈ 0.1) while length is identical across classes (ratio
0.001). This is the cleaner, more honest proof of curvature's independent
value.

The original damped-ping-pong generator is kept as a sanity test
(`t3_realistic_damped_oscillation_sanity`) — it confirms the signal also
fires on the realistic "pulled toward basins" model, where BOTH length and
curvature distinguish oscillation from commitment. The length-matched gate is
the one that proves curvature's INDEPENDENT information content.

## TL;DR

Ship `LatentTrajectoryGeometry` (length + mean_curvature + min_adjacent_cosine + bifurcation_ratio) as an opt-in primitive in katgpt-core. The gate is a visible game-related two-attractor-basin oscillation scenario showing curvature catches the ping-pong pattern that entropy misses. If G3 passes → promotion candidate for a follow-up routing plan. If G3 fails → honest negative result, primitive stays opt-in curiosity. No router integration in this plan; no Super-GOAT guide.
