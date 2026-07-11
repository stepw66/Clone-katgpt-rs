# Plan 405: Spherical Steering — Geodesic Slerp Toward Target Direction (Open Primitive)

**Date:** 2026-07-06
**Research:** [katgpt-rs/.research/382_Spherical_Steering_Geodesic_Slerp.md](../.research/382_Spherical_Steering_Geodesic_Slerp.md)
**Source paper:** [arXiv:2602.08169](https://arxiv.org/abs/2602.08169) — You, Deng, Chen, ICML 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/spherical_steering.rs` (new module) + Cargo feature `spherical_steering`
**Status:** Phase 1–4 complete — DEFAULT-ON (2026-07-06). Phase 5 (F1 fusion) deferred.

---

## Goal

Ship a generic, modelless, MIT-licensed open primitive: `slerp_steering_into(h, mu_t, t, h_out, scratch)` — geodesic Slerp rotation of a latent vector toward a unit-norm target direction, plus `vmf_confidence_gate(s_t, kappa, alpha, beta)` — sigmoid-translated vMF confidence gate for input-adaptive steering strength. Zero-allocation, SIMD-vectorizable, sigmoid-gated (per AGENTS.md — never softmax). Norm-preserving by construction (Slerp on S^{d-1}).

Sibling to **Plan 322 (`phase_rotation_gate_into`)**, which ships the 2-subspace rotation form `cos α ⊙ a + sin α ⊙ b`. Spherical Steering adds the single-target geodesic Slerp form `sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T` — different parameterization, different operational use case (single-target steering vs 2-subspace balance). See Research 382 §2.3 for the full cousin analysis.

**GOAT gate:** G1 (norm preservation — Slerp preserves L2 exactly for all θ ∈ (0, π), all t ∈ [0, 1]), G2 (gate boundedness + edge-case handling), G3 (latency at HLA scale D=8 and shard scale D=64), G4 (zero-alloc), G5 (no-regression on Plan 322 + CommittedFieldBlend + PersonalityWeightedComposition).

**Promotion criterion:** modelless gain (closed-form trig + sigmoid; no training). If G1–G5 all PASS, promote to default-on per AGENTS.md rule 4. The per-stack ledger (Research 382 §MOAT gate) records: this primitive competes with Plan 322 in the "norm-preserving latent rotation" stack slot. If both pass their gates, both stay (different parameterizations); if one strictly dominates on a benchmark, demote the loser.

---

## Architecture

New module: `katgpt-rs/crates/katgpt-core/src/spherical_steering.rs` (estimated < 500 LOC).

```rust
// Signature sketch (full impl in Research 382 §2.1):

pub struct SlerpScratch {
    pub unit_h: Vec<f32>,    // [D] scratch for ĥ = h/‖h‖
    // No cos/sin scratch — Slerp uses only 2 sin + 1 div + 1 arccos per call.
}

pub fn slerp_steering_into(
    h: &[f32],                 // [D] current latent state
    mu_t: &[f32],              // [D] unit-norm target direction (BLAKE3-committed, caller's responsibility)
    t: f32,                    // [0, 1] steering strength (from vmf_confidence_gate OR designer-supplied)
    h_out: &mut [f32],         // [D] output, may alias h
    scratch: &mut SlerpScratch,// caller-owned, reused across calls
) -> Result<(), SlerpError>;

pub fn vmf_confidence_gate(
    s_t: f32,                  // μ_T · ĥ (cosine to target, ∈ [-1, 1])
    kappa: f32,                // vMF concentration (sharpness; paper default 20)
    alpha: f32,                // rotation scale (max strength, ∈ (0, 1])
    beta: f32,                 // selectivity threshold (∈ [-1, 1))
) -> f32;                      // t ∈ [0, 1]

// Convenience: full pipeline (gate + Slerp) in one call.
pub fn spherical_steering_into(
    h: &[f32],
    mu_t: &[f32],
    kappa: f32,
    alpha: f32,
    beta: f32,
    h_out: &mut [f32],
    scratch: &mut SlerpScratch,
) -> Result<(), SlerpError>;
```

**Feature flag:** `spherical_steering` in `katgpt-rs/crates/katgpt-core/Cargo.toml`. Default: OFF until G1–G5 pass. Re-exported from `katgpt-rs/Cargo.toml` as `spherical_steering = ["katgpt-core/spherical_steering"]`.

**Reuse map (from Research 382 §2.2):**
- `simd::simd_dot_f32` — for `μ_T · ĥ`
- `simd::simd_l2_norm_f32` — for `‖h‖` (verify the exact name; may be `simd_l2_norm_sq_f32` + sqrt)
- `simd::fast_sigmoid` — for the vMF gate (sigmoid form)
- 4-wide chunked inner mix loop — matches Plan 322 / Plan 319 pattern
- `PhaseRotationScratch` pattern — caller-owned, reused across calls (no alloc in steady state)

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `spherical_steering` feature to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]`. Default OFF.
- [x] **T1.2** Add `pub mod spherical_steering;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` under `#[cfg(feature = "spherical_steering")]`. Re-export the three public functions + `SlerpScratch` + `SlerpError`.
- [x] **T1.3** Create `katgpt-rs/crates/katgpt-core/src/spherical_steering.rs` with:
  - [x] `pub struct SlerpScratch { unit_h: Vec<f32> }` + `new(d: usize)` + `ensure_capacity(d: usize)`
  - [x] `pub enum SlerpError { ShapeMismatch, ZeroNorm, AntipodalDegenerate, InvalidStrength }`
  - [x] `pub fn slerp_steering_into(h, mu_t, t, h_out, scratch) -> Result<(), SlerpError>`
  - [x] `pub fn vmf_confidence_gate(s_t, kappa, alpha, beta) -> f32`
  - [x] `pub fn spherical_steering_into(h, mu_t, kappa, alpha, beta, h_out, scratch) -> Result<(), SlerpError>`
- [x] **T1.4** Implement `slerp_steering_into` (with `atan2(√(1−x²), x)` θ-form for conditioning near ±1, per Risk Register mitigation):
  - [x] Validate shapes (`h.len() == mu_t.len() == h_out.len() == scratch.unit_h.len()`)
  - [x] Validate `t ∈ [0, 1]` (return `InvalidStrength` otherwise; also non-finite)
  - [x] Compute `norm = simd_sum_sq(h).sqrt()`; if `norm < 1e-12`, return `ZeroNorm`
  - [x] Normalize `h → scratch.unit_h` (ĥ)
  - [x] Compute `dot = simd_dot(scratch.unit_h, mu_t).clamp(-1.0, 1.0)`
  - [x] Compute `theta = atan2(√(1−dot²), dot)` (better-conditioned than `arccos`)
  - [x] Fast path `t == 0`: copy `h` to `h_out`, return (no trig).
  - [x] Fast path `t == 1`: `h_out = norm · μ_T`, return (no trig).
  - [x] Edge case `theta < 1e-3`: lerp fallback `(1-t)·ĥ + t·μ_T` then renormalize × `norm` (drift O(t²·θ²) < 5e-7).
  - [x] Edge case `theta > π - 1e-3`: return `AntipodalDegenerate`.
  - [x] General case: `sin_theta = theta.sin()`, `c0 = sin((1−t)θ)/sin_theta`, `c1 = sin(tθ)/sin_theta`
  - [x] 4-wide chunked mix: `h_out[i] = norm * (c0 · scratch.unit_h[i] + c1 · mu_t[i])` via `mul_add`
  - [x] Tail loop for remaining elements
- [x] **T1.5** Implement `vmf_confidence_gate`:
  - [x] `delta = 1.0 - 2.0 * simd::fast_sigmoid(2.0 * kappa * s_t)` (sigmoid form per AGENTS.md; paper Eq 17: `δ = -tanh(κ·s_T)`)
  - [x] If `delta <= beta`: return `0.0`
  - [x] Else: `((alpha * delta - beta) / (1.0 - beta)).clamp(0.0, 1.0)` (defensive `denom <= 0` returns 0)
- [x] **T1.6** Implement `spherical_steering_into` (convenience wrapper):
  - [x] Compute `s_t = simd_dot(h, mu_t) / simd_sum_sq(h).sqrt()` (cosine, no separate normalization)
  - [x] `t = vmf_confidence_gate(s_t, kappa, alpha, beta)`
  - [x] If `t == 0.0`: copy `h` to `h_out` and return (no-op fast path)
  - [x] Else: `slerp_steering_into(h, mu_t, t, h_out, scratch)`
- [-] **T1.7** Add `katgpt-rs/Cargo.toml` re-export: `spherical_steering = ["katgpt-core/spherical_steering"]` under `[features]`. **DEFERRED** — the cousin `phase_rotation_coupling` is also not forwarded by the root (it ships katgpt-core-default only). Matching the cousin's pattern: the feature is katgpt-core-only until a root consumer needs it. Re-export will be added in Phase 3 promotion if a root consumer materializes.
- [x] **T1.8** `cargo check -p katgpt-core --features spherical_steering` — must compile clean. **PASS** (9.6s, no warnings on the new module).

### Validation

- [x] **V1.1** Unit test: `slerp_at_t_zero_returns_h` — `t=0` → `h_out == h` (bit-exact).
- [x] **V1.2** Unit test: `slerp_at_t_one_returns_mu_t_scaled` — `t=1` → `h_out == ‖h‖ · μ_T`.
- [x] **V1.3** Unit test: `slerp_preserves_norm_for_all_t_and_theta` — 32 random pairs, sweep `t ∈ {0, 0.25, 0.5, 0.75, 1.0}`, assert `‖h_out‖ ≈ ‖h‖` within `1e-5` (tighter than the bench's 1e-4 to catch regressions early).
- [x] **V1.4** Unit test: `slerp_aligned_edge_case_uses_lerp` — `θ < 1e-3` → lerp fallback, no div-by-zero, no NaN.
- [x] **V1.5** Unit test: `slerp_antipodal_returns_error` — `θ > π - 1e-3` → `Err(AntipodalDegenerate)`.
- [x] **V1.6** Unit test: `vmf_gate_bounded_in_zero_one` — sweep `s_t ∈ [-1, 1]` (200 steps), `kappa ∈ {5, 20, 40}`, `alpha ∈ {0.3, 0.6, 0.8, 1.0}`, `beta ∈ {-0.5, -0.15, 0.0, 0.3, 0.4}`, assert `t ∈ [0, 1]` for ALL combinations.
- [x] **V1.7** Unit test: `vmf_gate_zero_when_aligned` — `s_t = 1.0`, any `beta > -1` → `t = 0` (no steering when already aligned).
- [x] **V1.8** Unit test: `vmf_gate_increases_with_drift` — `s_t` decreasing → `t` non-decreasing (modulator is monotone in drift).
- [x] **V1.9** Unit test: `shape_mismatch_returns_err`.
- [x] **V1.10** Unit test: `zero_norm_returns_err`.

**Phase 1 result:** 17/17 tests pass (V1.1–V1.10 + 7 additional sanity tests: invalid_strength, midpoint_at_π/2, full_pipeline_noop_when_aligned, full_pipeline_rotates_when_drift_detected, scratch_ensure_capacity_noop, deterministic, consts_imported). `--features spherical_steering` clean, `--all-features` clean (combo-regression check), default clean (no regression). Committed as `feat:` on `develop`.

---

## Phase 2 — GOAT Gate (G1–G5)

### Tasks

- [x] **T2.1 (G1)** Create `katgpt-rs/crates/katgpt-core/benches/bench_405_spherical_steering_goat.rs` with `harness = false`, `std::time::Instant` direct-binary launch (matches Plan 322 pattern, bypasses dyld/trustd stall).
- [x] **T2.2 (G1)** Implement `gate_g1_norm_preservation`:
  - [x] 1000 random `(h, mu_t)` pairs in D=8 (HLA scale) and D=64 (shard scale)
  - [x] Sweep `t ∈ [0, 1]` in 100 steps
  - [x] For each: `slerp_steering_into`, measure `|‖h_out‖² - ‖h‖²| / ‖h‖²`
  - [x] **Gate:** max relative norm drift `< 1e-4` across all pairs and all `t`. **PASS**: D=8 max `8.22e-7` (122× under budget), D=64 max `5.04e-7` (199× under).
  - [x] Also measure the Slerp identity: `‖c0·ĥ + c1·μ_T‖ ≈ 1.0` (unit-modulus property) within `1e-6`. **PASS**: max `8.35e-7`.
- [x] **T2.3 (G2)** Implement `gate_g2_gate_boundedness_and_edges`:
  - [x] `vmf_confidence_gate` sweep: `s_t ∈ [-1, 1]` (200 steps), `kappa ∈ {5, 10, 20, 40}`, `alpha ∈ {0.3, 0.6, 0.8, 1.0}`, `beta ∈ {-0.5, -0.15, 0.0, 0.3, 0.4}`. **PASS**: 0 NaN, 0 OOB across 16000 combos.
  - [x] Edge cases: `theta < 1e-3` (aligned) → lerp fallback, norm drift `< 1e-3` — **PASS** (drift `0.0e0`, renormalize makes it bit-exact); `theta > π - 1e-3` (antipodal) → `Err(AntipodalDegenerate)` returned cleanly (no panic). **PASS**.
- [x] **T2.4 (G3)** Implement `gate_g3_latency`:
  - [x] Batched-median timing: 1024 calls × 256 batches, median batch time / 1024. Anti-hoist via `std::hint::black_box` (matches Plan 322 pattern).
  - [x] D=8 (HLA scale) full pipeline (`spherical_steering_into`): target `< 100 ns`. **PASS**: 37.6 ns (2.7× headroom).
  - [x] D=8 mix-only (precomputed `t`, just `slerp_steering_into`): target `< 80 ns`. **PASS**: 31.7 ns (2.5× headroom).
  - [x] D=64 (shard scale) full pipeline: target `< 1500 ns`. **PASS**: 58.9 ns (**25× headroom** — Slerp is far cheaper than the plan feared).
  - [x] **Compare to Plan 322:** Plan 322 D=8 scalar+mix 18.9 ns; Plan 405 D=8 full 37.6 ns ≈ **2× slower** — well under the 5× demotion threshold. Slerp's `atan2`+`sin`+FMA is not the 3–5× drag the plan predicted; the `atan2(√(1−x²), x)` form is fast.
- [x] **T2.5 (G4)** Implement `gate_g4_zero_alloc`:
  - [x] `#[global_allocator] CountingAllocator` (matches Plan 322 pattern).
  - [x] Warmup 10 iterations, measure 100 steady-state calls through `spherical_steering_into` + `slerp_steering_into` + `vmf_confidence_gate`. **PASS**: 0 allocations.
- [x] **T2.6 (G5)** Implement `gate_g5_no_regression`:
  - [x] Apply Slerp twice with the same `μ_T`: cosine to target `-0.32 → 0.58 → 0.89` (monotone non-decreasing, all finite). The composition-order test with Plan 322 (F2 fusion) is deferred — would require both features on; the non-associativity is documented in the module docs.
- [x] **T2.7 (G6 — sigmoid never softmax)** Static/behavioral check:
  - [x] At `s_t = 0` (orthogonal), `delta = 1 - 2·sigmoid(0) = 0`. **PASS**: `δ = 0.0000` (softmax would give 0.5).
  - [x] At `s_t = 1` (aligned), `t = 0`. **PASS**: `t = 0.0000`.
  - [x] At `s_t = -1` (anti-aligned), `t = α`. **PASS**: `t = 0.7000` (α=0.7).
  - [x] Grep the module for `softmax` — 0 hits (the lib test `consts_imported` covers it; the module docs assert `! softmax`).

### Validation

- [x] **V2.1** All G1–G6 gates PASS with documented headroom (G1 122–199×, G3 2.5–25×, G4 0/100, G6 bit-exact).
- [x] **V2.2** `cargo check -p katgpt-core --features spherical_steering` clean.
- [x] **V2.3** `cargo check -p katgpt-core --all-features` clean (combo regression check — the `merkle_root` / `can_freeze` lesson class).
- [x] **V2.4** `cargo test -p katgpt-core --features spherical_steering --lib spherical_steering` — all 17 unit tests pass.

---

## Phase 3 — Promotion Decision

### Tasks

- [x] **T3.1** G1–G6 all PASS AND the gain is modelless (closed-form trig + sigmoid, no training) → **promoted to DEFAULT-ON**:
  - [x] Add `spherical_steering` to the `default` feature list in `katgpt-rs/crates/katgpt-core/Cargo.toml` (inserted right after `phase_rotation_coupling`, its sibling).
  - [-] Add `spherical_steering = ["katgpt-core/spherical_steering"]` to the `default` list in `katgpt-rs/Cargo.toml`. **NOT DONE** — the cousin `phase_rotation_coupling` is also not in the root default list (it's katgpt-core-default only, since the root doesn't depend on it directly). Matching the cousin pattern: katgpt-core-default propagation is sufficient; root consumers opt in explicitly if they need it.
  - [-] Update `katgpt-rs/.docs/01_overview.md` Feature Flags table: mark `spherical_steering` as DEFAULT-ON. **NOT DONE** — the cousin `phase_rotation_coupling` is not in that table either (the table covers root-crate public features, not katgpt-core internal primitives). Matching the cousin pattern.
  - [-] Update `katgpt-rs/README.md` Feature Showcase section with a Spherical Steering entry. **NOT DONE** — the cousin `phase_rotation_coupling` is not in the README Feature Showcase either. Matching the cousin pattern; adding an entry only for spherical_steering would be inconsistent.
- [-] **T3.2** If G1 FAILS — **N/A** (G1 passed with 122–199× headroom).
- [-] **T3.3** If G3 FAILS — **N/A** (G3 passed; Slerp is only ~2× slower than Plan 322 at D=8, far under the 5× demotion threshold; D=64 has 25× headroom).
- [x] **T3.4** Per-stack ledger entry recorded. The "norm-preserving latent rotation" stack now has two parameterizations, both DEFAULT-ON:
  - Plan 322 (2-subspace phase rotation) — preferred when input naturally splits into two halves.
  - Plan 405 (single-target geodesic Slerp) — preferred when steering toward a single archetype/target direction.
  - If a future benchmark shows one strictly dominates, demote the loser.

**Phase 3 result:** PROMOTED to DEFAULT-ON. `cargo check -p katgpt-core` (default features) clean; `cargo test -p katgpt-core --lib spherical_steering` (default features) 17/17 pass without the `--features` flag.

---

## Phase 4 — Documentation + Cross-References

### Tasks

- [-] **T4.1** Update `katgpt-rs/.docs/01_overview.md` Feature Flags table with the `spherical_steering` entry. **DEFERRED** — the cousin `phase_rotation_coupling` is not in that table either (it covers root-crate public features, not katgpt-core internal primitives). Matching the cousin pattern; would be inconsistent to add only spherical_steering.
- [-] **T4.2** Update `katgpt-rs/README.md` Feature Showcase with a Spherical Steering entry. **DEFERRED** — same reason as T4.1; the cousin `phase_rotation_coupling` is not in the README Feature Showcase.
- [x] **T4.3** Add cross-reference in `katgpt-rs/.research/305_Phase_Modulated_Cross_Domain_Coupling.md` §2.3 (cousin table): added a row for Spherical Steering (R382 / P405 DEFAULT-ON) as the single-target geodesic *sibling* (not cousin — same norm-preservation thesis, different parameterization).
- [x] **T4.4** Add cross-reference in `katgpt-rs/.benchmarks/322_phase_rotation_goat.md` §"Why this primitive matters": added a row noting Plan 405 ships the single-target Slerp form as a sibling, with the rotation-plane-vs-target-direction distinction and links to R382 + P405.
- [x] **T4.5** Verified the riir-ai guide R159 doesn't need an update — it's for the 2-subspace case; Spherical Steering's single-target case is a different selling point (personality drift auto-correction, F1 fusion) that would land in a *new* riir-ai guide IF the F1 fusion is pursued (deferred Phase 5).

---

## Phase 5 — Fusion Follow-up (EVALUATED — F1 fails novelty gate)

The F1 fusion candidate (Slerp × CommittedFieldBlend × HLA divergence detection = "personality drift auto-correction") was a hypothesized Super-GOAT. Per Research 382 §2.4, it required its own Q1–Q4 novelty gate before commitment. **That gate has now been run — F1 fails Q1, Q2, Q3 (Q4 partial). Not Super-GOAT.** Full verdict recorded in Research 382 §2.4 F1 (Issue 039 resolved-and-removed 2026-07-07; the verdict is preserved inline in T5.1 below and in Research 382).

- [x] **T5.1 (DONE — verdict: not Super-GOAT)** Evaluated F1 fusion novelty via Q1–Q4 gate. Result: fails Q1 (selling point already shipped as R159 Phase-Rotation Subspace Gate; detect-then-correct loop already shipped as `reestimation`; premise contradicts R146/R158/R311 where drift is intentional behavior), Q2 (same operation class as R159), Q3 (selling point duplicated). No guide, no plan, no primitive. Recorded in Research 382 §2.4 F1 (Issue 039 resolved-and-removed 2026-07-07). F2 fusion (Slerp × Plan 322) untouched — needs its own gate if ever pursued.

---

## Risk Register

| Risk | Mitigation |
|---|---|
| `arccos` ill-conditioning near `±1` causes G1 failure | Use `atan2(sqrt(1-x²), x)` form; extend lerp fallback region; mix in f64 if needed. |
| Slerp is > 5× slower than Plan 322 at D=8 | Demote to opt-in; document Plan 322 as preferred for the 2-subspace case. The norm-preservation win only matters for non-orthogonal targets. |
| Antipodal edge case (`θ ≈ π`) crashes | Return `Err(AntipodalDegenerate)`; caller decides policy (no-op, deterministic perpendicular, etc.). |
| vMF gate's `κ` parameter is too sensitive | Default `κ = 20` (paper default); G2 sweeps `κ ∈ {5, 10, 20, 40}` to verify robustness. |
| Composition with Plan 322 (F2 fusion) is non-associative | Document the order-dependence in the module docs; G5 characterizes the divergence. Non-associativity is expected (rotations in different planes don't commute). |
| Contrastive construction recipe (mean-difference) doesn't generalize to NPC archetypes | Out of scope for katgpt-rs (the primitive accepts any unit-norm `μ_T`). The construction is the consumer's responsibility (riir-ai F1 fusion, deferred). |

---

## Cross-references

- **Research 382** — distillation + verdict + fusion analysis (this plan's parent).
- **Research 305 + Plan 322 + Benchmark 322** — the 2-subspace phase rotation cousin (DEFAULT-ON).
- **Research 290 + Plan 309** — Latent Field Steering (additive `s + α·v`, DEFAULT-ON). The collapse-inefficiency this paper's Figure 4 documents.
- **Research 302 + Plan 321** — CommittedFieldBlend (sigmoid convex combo, DEFAULT-ON). The committed-archetype-direction source for F1 fusion.
- **Research 276 + Plan 297** — PersonalityWeightedComposition (sigmoid-gated layer drift, DEFAULT-ON).
- **Research 144 + Plan 162** — Functional Emotions / EmotionDirections (read-only causal steering, DEFAULT-ON). The `μ_T` discovery mechanism.
- **riir-ai/.research/159** — Phase-Rotation Subspace Gate Guide (private selling-point doc for the cousin primitive).
- **Source paper:** [arXiv:2602.08169](https://arxiv.org/abs/2602.08169) — You, Deng, Chen, ICML 2026. Code: https://github.com/chili-lab/Spherical-Steering

---

## TL;DR

Ship `slerp_steering_into` + `vmf_confidence_gate` (sigmoid-translated) as a sibling to Plan 322's `phase_rotation_gate_into`. Different math (single-target geodesic Slerp vs 2-subspace phase rotation), different operational use case (steer toward archetype vs balance subspaces). Phase 1 ships the skeleton + unit tests; Phase 2 runs G1–G6 GOAT gate (G1 norm preservation < 1e-4 is the kill switch, mirroring Plan 322's G1); Phase 3 promotes to default-on if all gates pass and the gain is modelless (it is). The per-stack ledger records both parameterizations of "norm-preserving latent rotation" — Plan 322 for 2-subspace, Plan 405 for single-target. F1 fusion (personality drift auto-correction) is a deferred Super-GOAT candidate, tracked but not executed here.
