# Plan 405: Spherical Steering — Geodesic Slerp Toward Target Direction (Open Primitive)

**Date:** 2026-07-06
**Research:** [katgpt-rs/.research/382_Spherical_Steering_Geodesic_Slerp.md](../.research/382_Spherical_Steering_Geodesic_Slerp.md)
**Source paper:** [arXiv:2602.08169](https://arxiv.org/abs/2602.08169) — You, Deng, Chen, ICML 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/spherical_steering.rs` (new module) + Cargo feature `spherical_steering`
**Status:** Active — Phase 1 pending

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

- [ ] **T1.1** Add `spherical_steering` feature to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]`. Default OFF.
- [ ] **T1.2** Add `pub mod spherical_steering;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` under `#[cfg(feature = "spherical_steering")]`. Re-export the three public functions + `SlerpScratch` + `SlerpError`.
- [ ] **T1.3** Create `katgpt-rs/crates/katgpt-core/src/spherical_steering.rs` with:
  - [ ] `pub struct SlerpScratch { unit_h: Vec<f32> }` + `new(d: usize)` + `ensure_capacity(d: usize)`
  - [ ] `pub enum SlerpError { ShapeMismatch, ZeroNorm, AntipodalDegenerate, InvalidStrength }`
  - [ ] `pub fn slerp_steering_into(h, mu_t, t, h_out, scratch) -> Result<(), SlerpError>`
  - [ ] `pub fn vmf_confidence_gate(s_t, kappa, alpha, beta) -> f32`
  - [ ] `pub fn spherical_steering_into(h, mu_t, kappa, alpha, beta, h_out, scratch) -> Result<(), SlerpError>`
- [ ] **T1.4** Implement `slerp_steering_into`:
  - [ ] Validate shapes (`h.len() == mu_t.len() == h_out.len() == scratch.unit_h.len()`)
  - [ ] Validate `t ∈ [0, 1]` (return `InvalidStrength` otherwise)
  - [ ] Compute `norm = simd_l2_norm(h)`; if `norm < 1e-12`, return `ZeroNorm`
  - [ ] Normalize `h → scratch.unit_h` (ĥ)
  - [ ] Compute `dot = simd_dot(scratch.unit_h, mu_t).clamp(-1.0, 1.0)`
  - [ ] Compute `theta = dot.acos()`
  - [ ] Edge case `theta < 1e-6`: lerp fallback `(1-t)·ĥ + t·μ_T` then renormalize × `norm` (avoids div-by-zero; drift is O(t²·θ²))
  - [ ] Edge case `theta > π - 1e-6`: return `AntipodalDegenerate` (paper's measure-zero case; caller decides policy)
  - [ ] General case: `sin_theta = theta.sin()`, `c0 = sin((1−t)θ)/sin_theta`, `c1 = sin(tθ)/sin_theta`
  - [ ] 4-wide chunked mix: `h_out[i] = norm * (c0 · scratch.unit_h[i] + c1 · mu_t[i])`
  - [ ] Tail loop for remaining elements
- [ ] **T1.5** Implement `vmf_confidence_gate`:
  - [ ] `delta = 1.0 - 2.0 * simd::fast_sigmoid(2.0 * kappa * s_t)` (sigmoid form per AGENTS.md; paper Eq 17: `δ = -tanh(κ·s_T)`)
  - [ ] If `delta <= beta`: return `0.0`
  - [ ] Else: `((alpha * delta - beta) / (1.0 - beta)).clamp(0.0, 1.0)`
- [ ] **T1.6** Implement `spherical_steering_into` (convenience wrapper):
  - [ ] Compute `s_t = simd_dot(h, mu_t) / simd_l2_norm(h)` (cosine, no separate normalization)
  - [ ] `t = vmf_confidence_gate(s_t, kappa, alpha, beta)`
  - [ ] If `t == 0.0`: copy `h` to `h_out` and return (no-op fast path)
  - [ ] Else: `slerp_steering_into(h, mu_t, t, h_out, scratch)`
- [ ] **T1.7** Add `katgpt-rs/Cargo.toml` re-export: `spherical_steering = ["katgpt-core/spherical_steering"]` under `[features]`.
- [ ] **T1.8** `cargo check -p katgpt-core --features spherical_steering` — must compile clean.

### Validation

- [ ] **V1.1** Unit test: `slerp_at_t_zero_returns_h` — `t=0` → `h_out == h` (bit-exact).
- [ ] **V1.2** Unit test: `slerp_at_t_one_returns_mu_t_scaled` — `t=1` → `h_out == ‖h‖ · μ_T`.
- [ ] **V1.3** Unit test: `slerp_preserves_norm_for_all_t_and_theta` — sweep `t ∈ {0, 0.25, 0.5, 0.75, 1.0}`, `θ ∈ {0.1, π/4, π/2, 2.7}` (avoid 0 and π edges), assert `‖h_out‖ ≈ ‖h‖` within `1e-5`.
- [ ] **V1.4** Unit test: `slerp_aligned_edge_case_uses_lerp` — `θ < 1e-6` → lerp fallback, no div-by-zero, no NaN.
- [ ] **V1.5** Unit test: `slerp_antipodal_returns_error` — `θ > π - 1e-6` → `Err(AntipodalDegenerate)`.
- [ ] **V1.6** Unit test: `vmf_gate_bounded_in_zero_one` — sweep `s_t ∈ [-1, 1]`, `kappa ∈ {5, 20, 40}`, assert `t ∈ [0, 1]`.
- [ ] **V1.7** Unit test: `vmf_gate_zero_when_aligned` — `s_t = 1.0`, any `beta > -1` → `t = 0` (no steering when already aligned).
- [ ] **V1.8** Unit test: `vmf_gate_increases_with_drift` — `s_t` decreasing → `t` non-decreasing (modulator is monotone in drift).
- [ ] **V1.9** Unit test: `shape_mismatch_returns_err`.
- [ ] **V1.10** Unit test: `zero_norm_returns_err`.

---

## Phase 2 — GOAT Gate (G1–G5)

### Tasks

- [ ] **T2.1 (G1)** Create `katgpt-rs/crates/katgpt-core/benches/bench_405_spherical_steering_goat.rs` with `harness = false`, `std::time::Instant` direct-binary launch (matches Plan 322 pattern, bypasses dyld/trustd stall).
- [ ] **T2.2 (G1)** Implement `gate_g1_norm_preservation`:
  - [ ] 1000 random `(h, mu_t)` pairs in D=8 (HLA scale) and D=64 (shard scale)
  - [ ] Sweep `t ∈ [0, 1]` in 100 steps
  - [ ] For each: `slerp_steering_into`, measure `|‖h_out‖² - ‖h‖²| / ‖h‖²`
  - [ ] **Gate:** max relative norm drift `< 1e-4` across all pairs and all `t`.
  - [ ] Also measure the Slerp identity: `‖c0·ĥ + c1·μ_T‖ ≈ 1.0` (unit-modulus property) within `1e-6`.
- [ ] **T2.3 (G2)** Implement `gate_g2_gate_boundedness_and_edges`:
  - [ ] `vmf_confidence_gate` sweep: `s_t ∈ [-1, 1]` (200 steps), `kappa ∈ {5, 10, 20, 40}`, `alpha ∈ {0.3, 0.6, 0.8, 1.0}`, `beta ∈ {-0.5, -0.15, 0.0, 0.3, 0.4}`. Assert `t ∈ [0, 1]` for ALL combinations (no NaN, no out-of-range).
  - [ ] Edge cases: `theta < 1e-6` (aligned) → lerp fallback, norm drift `< 1e-3`; `theta > π - 1e-6` (antipodal) → `Err(AntipodalDegenerate)` returned cleanly (no panic).
- [ ] **T2.4 (G3)** Implement `gate_g3_latency`:
  - [ ] Batched-median timing: 1024 calls × 256 batches, median batch time / 1024. Anti-hoist via `std::hint::black_box` (matches Plan 322 pattern).
  - [ ] D=8 (HLA scale) full pipeline (`spherical_steering_into`): target `< 100 ns` (Slerp requires arccos + 2 sin + div, vs Plan 322's 18.9 ns; expect ~3-5× slower).
  - [ ] D=8 mix-only (precomputed `t`, just `slerp_steering_into`): target `< 80 ns`.
  - [ ] D=64 (shard scale) full pipeline: target `< 1500 ns` (matches Plan 322 D=64 budget).
  - [ ] **Compare to Plan 322:** run `phase_rotation_gate_into` at D=8 and D=64 on the same bench harness; report the latency ratio. If Slerp is > 5× slower at D=8, document and consider demoting to opt-in.
- [ ] **T2.5 (G4)** Implement `gate_g4_zero_alloc`:
  - [ ] `#[global_allocator] CountingAllocator` (matches Plan 322 pattern).
  - [ ] Warmup 10 iterations, measure 100 steady-state calls through `spherical_steering_into` + `slerp_steering_into` + `vmf_confidence_gate`.
  - [ ] **Gate:** 0 allocations in steady state (scratch reused, no `Vec::new` / `vec![]` / `Vec::clone` on hot path).
- [ ] **T2.6 (G5)** Implement `gate_g5_no_regression`:
  - [ ] Run Plan 322's `phase_rotation_gate_into` unit tests under `--features spherical_steering` — all must still pass.
  - [ ] Run Plan 321's `CommittedFieldBlend` unit tests under `--features spherical_steering` — all must still pass.
  - [ ] Run Plan 297's `PersonalityWeightedComposition` unit tests under `--features spherical_steering` — all must still pass.
  - [ ] Composition-order test (F2 fusion): for a test vector `h`, apply `slerp_then_phase` and `phase_then_slerp`; assert the results differ by `< ε` when `t` and `α` are small, and document the divergence when they're large. (Non-associativity is expected; the test characterizes it, not forbids it.)
- [ ] **T2.7 (G6 — sigmoid never softmax)** Static/behavioral check:
  - [ ] At `s_t = 0` (orthogonal), `delta = 1 - 2·sigmoid(0) = 1 - 1 = 0`. Assert `vmf_confidence_gate(0.0, kappa, alpha, beta)` returns `0.0` when `beta ≥ 0` (no steering at zero drift).
  - [ ] Grep the module for `softmax` — must return ZERO hits (per AGENTS.md rule).

### Validation

- [ ] **V2.1** All G1–G6 gates PASS with documented headroom.
- [ ] **V2.2** `cargo check -p katgpt-core --features spherical_steering` clean.
- [ ] **V2.3** `cargo check -p katgpt-core --all-features` clean (combo regression check — the `merkle_root` / `can_freeze` lesson class).
- [ ] **V2.4** `cargo test -p katgpt-core --features spherical_steering --lib spherical_steering` — all unit tests pass.

---

## Phase 3 — Promotion Decision

### Tasks

- [ ] **T3.1** If G1–G6 all PASS AND the gain is modelless (it is — closed-form trig + sigmoid):
  - [ ] Add `spherical_steering` to the `default` feature list in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
  - [ ] Add `spherical_steering = ["katgpt-core/spherical_steering"]` to the `default` list in `katgpt-rs/Cargo.toml`.
  - [ ] Update `katgpt-rs/.docs/01_overview.md` Feature Flags table: mark `spherical_steering` as DEFAULT-ON with the gate summary.
  - [ ] Update `katgpt-rs/README.md` Feature Showcase section with a Spherical Steering entry (sibling to Plan 322's entry).
- [ ] **T3.2** If G1 FAILS (norm drift > 1e-4):
  - [ ] Debug the numerical path. Likely causes: (a) `arccos` near `±1` is ill-conditioned, (b) `sin θ` near 0 amplifies division error, (c) f32 rounding in the mix loop.
  - [ ] Mitigations to try: (a) use `arccos` via `atan2(sqrt(1-x²), x)` for better conditioning near ±1, (b) extend the lerp fallback region from `θ < 1e-6` to `θ < 1e-3`, (c) mix in f64 then round to f32.
  - [ ] If mitigations fail: demote to opt-in, document the norm-drift ceiling, do NOT promote to default.
- [ ] **T3.3** If G3 FAILS (latency > budget):
  - [ ] If Slerp is > 5× slower than Plan 322 at D=8: demote to opt-in. Document Plan 322 as the preferred form for the 2-subspace case; Spherical Steering for the single-target case where the norm-preservation win justifies the cost.
  - [ ] If Slerp is 2-5× slower: promote to default anyway (the norm-preservation for non-orthogonal targets is worth it), but document the latency tradeoff.
- [ ] **T3.4** Per-stack ledger entry (Research 382 §MOAT gate): record the verdict in the plan file. The "norm-preserving latent rotation" stack now has two parameterizations:
  - [ ] Plan 322 (2-subspace phase rotation) — preferred when the input naturally splits into two halves.
  - [ ] Plan 405 (single-target geodesic Slerp) — preferred when steering toward a single archetype/target direction.
  - [ ] If a future benchmark shows one strictly dominates on a unified task, demote the loser.

---

## Phase 4 — Documentation + Cross-References

### Tasks

- [ ] **T4.1** Update `katgpt-rs/.docs/01_overview.md` Feature Flags table with the `spherical_steering` entry (sibling to `phase_rotation_coupling`).
- [ ] **T4.2** Update `katgpt-rs/README.md` Feature Showcase with a Spherical Steering entry. One-paragraph summary: norm-preserving geodesic Slerp toward a target direction + sigmoid-translated vMF confidence gate. Cross-link to Plan 322 as the 2-subspace cousin.
- [ ] **T4.3** Add cross-reference in `katgpt-rs/.research/305_Phase_Modulated_Cross_Domain_Coupling.md` §2.3 (cousin table): add a row for Spherical Steering (R382 / P405) as the single-target geodesic sibling.
- [ ] **T4.4** Add cross-reference in `katgpt-rs/.benchmarks/322_phase_rotation_goat.md` §"Why this primitive matters": note that Plan 405 ships the single-target Slerp form as a sibling.
- [ ] **T4.5** Verify the riir-ai guide R159 (Phase-Rotation Subspace Gate Guide) doesn't need an update — it's for the 2-subspace case; Spherical Steering's single-target case is a different selling point (personality drift auto-correction, F1 fusion) that would land in a *new* riir-ai guide IF the F1 fusion is pursued (not in this plan).

---

## Phase 5 — Fusion Follow-up (DEFERRED — tracked, not executed in this plan)

The F1 fusion candidate (Slerp × CommittedFieldBlend × HLA divergence detection = "personality drift auto-correction") is a potential Super-GOAT. Per Research 382 §2.4, it is **not committed** in this plan — it requires:
1. This primitive (Plan 405) to ship and pass G1–G5.
2. A separate novelty gate (Q1–Q4) for the fusion itself.
3. If Q1–Q4 all pass, a private guide in `riir-ai/.research/` + a plan in `riir-ai/.plans/`.

- [ ] **T5.1 (DEFERRED)** After Plan 405 Phase 3 promotion, evaluate F1 fusion novelty. If promising, open a new research note in `katgpt-rs/.research/` or directly in `riir-ai/.research/` (depending on where the selling point lives).

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
