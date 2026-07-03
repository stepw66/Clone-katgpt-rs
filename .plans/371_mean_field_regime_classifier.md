# Plan 371: Mean-Field Crowd Oscillation Regime Classifier

**Date:** 2026-07-03
**Research:** [katgpt-rs/.research/371_Low_Rank_Adaptation_Oscillation_DMFT.md](../.research/371_Low_Rank_Adaptation_Oscillation_DMFT.md)
**Source paper:** [arXiv:2606.30366](https://arxiv.org/abs/2606.30366) — Zheng, Miller, Fiete (MIT, Jun 2026), "Mean-field theory of rich oscillatory dynamics in low-rank recurrent networks with activity-dependent adaptation"
**Target:** `katgpt-rs/crates/katgpt-core/src/mean_field/` (new module) + Cargo feature `mean_field_regime`
**Status:** Active — Phase 1 not started

---

## Goal

Ship three modelless primitives distilled from arXiv:2606.30366:

1. **`MeanFieldOverlap`** — one-pass crowd-level aggregation of per-NPC HLA states into the paper's `(κ, κ_a, Q)` order parameters (coherent overlap, adaptation overlap, incoherent variance) via dot-product projection onto a frozen direction vector.
2. **`HopfBoundary`** — closed-form 2×2 Jacobian eigenvalue check on `(κ, κ_a)` for oscillatory instability (extends `subspace_phase_gate` from real-eigenvalue to complex-eigenvalue phase transitions).
3. **`RegimeClassifier`** — combine the above + chaos-intensity estimate `g` into a `Regime ∈ {Static, NoiseSustainedOscillation, IrregularSwitching, GlobalLimitCycle}` enum.

These compose the existing per-NPC primitives (`HLA`, `TemporalDerivativeKernel`, `MicroRecurrentBeliefState`, `latent_functor` direction vectors) into crowd-scale emergent oscillations. The paper's wake/sleep/anesthesia biological mapping becomes a runtime knob: β is the per-NPC arousal scalar, and sweeping it across a crowd produces emergent day/night cycles, panic waves, fashion trends.

**GOAT gate (all 5 must pass before promote-to-default):**
- **G1 (correctness):** regime classifier matches the simulated `(κ, κ_a, Q)` ODE trajectory's qualitative behavior on a `(g, β)` grid covering all four regimes. Defend-wrong PoC in `riir-ai/crates/riir-poc/` (§3.6).
- **G2 (perf):** `aggregate_into` over 1000 NPCs (dim=8) ≤ 5µs; `hopf_boundary` ≤ 50ns; `classify` ≤ 100ns.
- **G3 (no-regression):** enabling `mean_field_regime` does not break existing tests.
- **G4 (alloc-free):** `aggregate_into` zero-allocation; `hopf_boundary`/`classify` pure f32 arithmetic.
- **G5 (determinism):** bit-identical across platforms (anti-cheat — regime enum crosses sync boundary).

**Promotion rule:** if G1–G5 pass AND the PoC confirms regime classification on a toy domain → promote `mean_field_regime` to default. If the PoC refutes → keep opt-in, record §PoC Addendum in Research 371, create `.issues/` follow-up.

---

## Phase 1 — Skeleton: `MeanFieldOverlap` aggregator (CORE)

The simplest primitive — dot-product projection of K HLA vectors onto a direction vector `n`, plus the incoherent variance `Q`. This is the population analog of `ict::BranchingDetector::last_population_mean` but over NPCs (not trajectories) and onto a learned direction (not action probabilities).

### Tasks

- [ ] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/mean_field/mod.rs` with feature gate `#[cfg(feature = "mean_field_regime")]`. Add the feature to `katgpt-core/Cargo.toml` as opt-in (`mean_field_regime = []`).
- [ ] **T1.2** Define `pub struct MeanFieldOverlap { kappa, kappa_a, q, scratch_dot, scratch_sq }` — three `f32` outputs + two pre-allocated `Vec<f32>` scratch buffers (one for the dot-product accumulation, one for the squared-firing-rate accumulation). Use `Vec::with_capacity(D)` once at construction; `clear()` + reuse on each `aggregate_into` call.
- [ ] **T1.3** Implement `pub fn aggregate_into<D: usize>(&mut self, hlas: &[&[f32; D]], adapt: &[&[f32; D]], n: &[f32; D])` — one pass over K NPCs:
  - `kappa = (1/K) · Σ_i dot(n, tanh(h_i))` (use `tanh` from `katgpt_types` or inline fast-tanh)
  - `kappa_a = (1/K) · Σ_i dot(n, a_i)` (adaptation currents, no tanh)
  - `q = (1/K) · Σ_i dot(tanh(h_i), tanh(h_i))` (incoherent variance)
  - **Chunk-4 loop** for SIMD auto-vectorization on the dot-product (per AGENTS.md optimization rules).
  - **Zero allocation** in the hot path — write into pre-allocated scratch.
- [ ] **T1.4** Implement `pub fn new<D: usize>(dim: usize) -> Self` constructor with `Vec::with_capacity(dim)` for scratch.
- [ ] **T1.5** Add accessors: `pub fn kappa(&self) -> f32`, `pub fn kappa_a(&self) -> f32`, `pub fn q(&self) -> f32`.
- [ ] **T1.6** Unit tests:
  - Zero HLA → `kappa = kappa_a = q = 0`.
  - All HLA equal to direction vector → `kappa ≈ tanh(1)`, `q ≈ tanh(1)²`.
  - Orthogonal HLA → `kappa ≈ 0`, `q > 0`.
  - Determinism: same inputs → bit-identical outputs across two calls.
- [ ] **T1.7** Add `pub fn estimate_chaos_intensity(&self) -> f32` — `g ≈ sqrt(q / (1 - q))` heuristic (the paper's `Q` grows with `g` above the chaos threshold; this is a rough estimator, refined in Phase 3).

---

## Phase 2 — `HopfBoundary` detector (extends `subspace_phase_gate`)

The closed-form 2×2 eigenvalue check. Paper Eq. 56 characteristic polynomial of the `(κ, κ_a)` planar subsystem:

```
(s·τ_m + 1 − λ_eff·G_eff)·(s·τ_a + 1) + β·G_eff = 0
```

Expanding: `τ_m·τ_a·s² + (τ_m + τ_a − λ_eff·τ_a·G_eff·τ_m/τ_m)·s + (1 − λ_eff·G_eff + β·G_eff·τ_a/τ_a)`. Wait — let me restate cleanly. The planar Jacobian at the fixed point is:

```
J = | ∂κ̇/∂κ    ∂κ̇/∂κ_a |   =   | (−1 + λ_eff·G_eff)/τ_m    −G_eff/τ_m |
    | ∂κ̇_a/∂κ  ∂κ̇_a/∂κ_a |       | β/τ_a                     −1/τ_a    |
```

Eigenvalues `s` satisfy `det(J − sI) = 0`. Hopf boundary = complex conjugate pair with `Re(s) > 0`.

### Tasks

- [ ] **T2.1** Define `pub struct HopfParams { tau_m: f32, tau_a: f32, beta: f32, lambda_eff: f32, g_eff: f32 }`. Defaults: `tau_m = 1.0`, `tau_a = 30.0`, `g_eff = 1.0` (refined in Phase 3 from the `MeanFieldOverlap` fixed-point stats).
- [ ] **T2.2** Implement `pub fn hopf_boundary(&self, params: &HopfParams) -> Option<f32>` on `MeanFieldOverlap`:
  - Compute the 2×2 Jacobian trace `T = (−1 + λ_eff·G_eff)/τ_m + (−1/τ_a)` and determinant `D = ((−1 + λ_eff·G_eff)/τ_m)·(−1/τ_a) − (−G_eff/τ_m)·(β/τ_a)`.
  - Discriminant `Δ = T² − 4·D`. If `Δ < 0` AND `T > 0` → complex eigenvalues with positive real part → **Hopf instability**. Return `Some(sqrt(|Δ|)/2)` as the Hopf frequency `ω_hopf`.
  - Else → `None` (stable, no oscillatory instability).
- [ ] **T2.3** Implement `pub fn static_boundary(&self, params: &HopfParams) -> bool` — the real-eigenvalue crossing (returns `true` if any real eigenvalue `s > 0`, i.e., `D < 0` or (`T > 0` and `Δ ≥ 0`)). This is the paper's chaos-onset-from-coherent-mode boundary (distinct from the random-bulk chaos boundary).
- [ ] **T2.4** Unit tests:
  - β = 0 → `hopf_boundary` returns `None` (adaptation-free, real eigenvalues).
  - Large β with `τ_a ≫ τ_m` → `hopf_boundary` returns `Some(ω)` with `ω ≈ 1/sqrt(τ_a)` (paper Eq. A9).
  - `T < 0` always → `None` (stable focus, not Hopf).
  - Determinism: bit-identical across calls.
- [ ] **T2.5** Add a doc comment cross-referencing `subspace_phase_gate` (Plan 301) — this primitive extends it from *real-eigenvalue* phase transitions to *complex-eigenvalue* (Hopf) phase transitions.

---

## Phase 3 — `RegimeClassifier` (paper's four-way taxonomy)

Combine `MeanFieldOverlap` + `HopfBoundary` + chaos-intensity `g` into the paper's four regimes.

### Tasks

- [ ] **T3.1** Define `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Regime { Static, NoiseSustainedOscillation, IrregularSwitching, GlobalLimitCycle }`. Add `pub fn as_u8(self) -> u8` for sync-boundary serialization (raw, deterministic).
- [ ] **T3.2** Define `pub struct RegimeClassifier { hopf_margin: f32, switching_margin: f32, chaos_threshold: f32 }` — three tunable margins (defaults: `hopf_margin = 0.1`, `switching_margin = 0.05`, `chaos_threshold = 1.0`).
- [ ] **T3.3** Implement `pub fn classify(&self, overlap: &MeanFieldOverlap, params: &HopfParams) -> Regime`:
  1. Estimate `g` from `overlap.estimate_chaos_intensity()` (Phase 1 T1.7).
  2. Check `overlap.hopf_boundary(params)`:
     - `Some(ω)` with `T > hopf_margin` → `Regime::GlobalLimitCycle` (Hopf bifurcation occurred).
     - `Some(ω)` with `0 < T < hopf_margin` AND `g > chaos_threshold` → `Regime::IrregularSwitching` (near-Hopf, noise kicks across separatrix).
     - `None` (stable) AND `g > chaos_threshold` → `Regime::NoiseSustainedOscillation` (stable focus driven by chaotic bulk).
     - `None` AND `g ≤ chaos_threshold` → `Regime::Static` (stable node, no chaos).
  3. Return the classified `Regime`.
- [ ] **T3.4** Unit tests covering each regime on synthetic `(κ, κ_a, Q, g, β)` inputs.
- [ ] **T3.5** Add `pub fn classify_with_g(&self, overlap: &MeanFieldOverlap, params: &HopfParams, g_override: f32) -> Regime` — allow the caller to inject a calibrated `g` (e.g., from `cgsp_runtime` curiosity exploration intensity) instead of the heuristic estimate.

---

## Phase 4 — Wire into `lib.rs` + feature gate

### Tasks

- [ ] **T4.1** Add `#[cfg(feature = "mean_field_regime")] pub mod mean_field;` to `katgpt-rs/crates/katgpt-core/src/lib.rs`.
- [ ] **T4.2** Add `mean_field_regime = []` to `[features]` in `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in, NOT default).
- [ ] **T4.3** Run `cargo check -p katgpt-core --features mean_field_regime` — must pass clean.
- [ ] **T4.4** Run `cargo check -p katgpt-core` (default features) — must still pass (feature is opt-in).
- [ ] **T4.5** Run `cargo test -p katgpt-core --features mean_field_regime --lib` — all unit tests pass.

---

## Phase 5 — GOAT gate (defend-wrong PoC + benchmarks)

**Mandatory defend-wrong PoC** per §3.6 — the verdict asserts the classifier *works* (matches the paper's phase diagram), not just that it *exists*.

### Tasks

- [ ] **T5.1 (PoC — `riir-ai/crates/riir-poc/`)** Implement `benches/mean_field_regime_poc.rs`:
  - Implement the paper's reduced 3D ODE (Eq. 55) as a modelless simulator with `(g, β)` knobs. Use `tau_m = 1.0`, `tau_a = 30.0`, `sigma_m = sigma_n = 2.0`, `gamma = 0.7` (paper defaults). Integrate with simple Euler at `dt = 0.1` for `T = 600` time units.
  - For each `(g, β)` in a grid (e.g., `g ∈ {1.0, 1.2, 1.4, 1.6, 1.8, 2.0}`, `β ∈ {0.0, 0.35, 0.55, 0.85, 1.4}` — paper Fig. 1):
    - Simulate the ODE trajectory.
    - Classify the trajectory's qualitative regime from the trajectory itself (amplitude of `κ(t)` oscillation, switching events, limit-cycle detection).
    - Run `RegimeClassifier::classify` on the simulated `(κ, κ_a, Q)` + `(g, β)`.
    - Record whether they agree.
  - Print a verdict table: `(g, β) | simulated regime | classified regime | match?`
  - **Defend OR refute**: if the classifier misclassifies ≥1 regime (e.g., calls Regime II "Static"), record honestly as a §PoC Addendum in Research 371 and create `.issues/` follow-up. Do NOT silently tune the margins to make it pass.
  - Use `CARGO_TARGET_DIR=/tmp/mean_field_poc` per AGENTS.md rule; clean up when done.
- [ ] **T5.2 (G2 perf bench)** `benches/mean_field_regime_bench.rs`:
  - `aggregate_into` over 1000 NPCs (dim=8) — target ≤ 5µs.
  - `hopf_boundary` — target ≤ 50ns.
  - `classify` — target ≤ 100ns.
  - Use criterion; sample_size ≥ 500 for the micro-benches.
- [ ] **T5.3 (G3 no-regression)** Run `cargo test -p katgpt-core --lib` (default features) — must pass. Run `cargo test --workspace` if feasible.
- [ ] **T5.4 (G4 alloc-free)** Add a `#[test]` that wraps `aggregate_into` in a custom allocator shim and asserts zero allocations in the hot path (mirrors the pattern in `delta_mem` tests).
- [ ] **T5.5 (G5 determinism)** Add a `#[test]` that runs `aggregate_into` + `classify` twice on identical inputs and asserts bit-identical outputs (assert_eq on the `Regime` enum + raw f32 bits via `f32::to_bits`).

---

## Phase 6 — Promote (or defer) decision

### Tasks

- [ ] **T6.1** If G1–G5 all PASS and PoC confirms regime classification (≥4/5 regimes correctly identified on the grid): promote `mean_field_regime` to `default` in `katgpt-core/Cargo.toml`. Update Research 371 with a §"GOAT gate passed" addendum linking the PoC verdict table + bench numbers.
- [ ] **T6.2** If PoC refutes (classifier misclassifies): keep `mean_field_regime` opt-in. Record §PoC Addendum in Research 371 with the raw numbers + which regimes misclassified. Create `katgpt-rs/.issues/NNN_mean_field_regime_poc_refutation.md` tracking the follow-up (margin tuning, better `g` estimator, or fundamental limitation).
- [ ] **T6.3** If promoted: grep for downstream consumers that should adopt it (`ict_runtime`, `cgsp_runtime`, `crowd_attention`, `latent_functor`). Open issues for each consumer to evaluate adoption — do NOT force-wire in this plan.

---

## Out of scope (tracked separately)

- **riir-ai runtime wiring** (per-archetype β via `ArchetypeBlendShard`, surprise-driven regime transition via `temporal_deriv`, crowd oscillation as emergent day/night cycle) — **follow-up issue pending GOAT-gate pass**. If the gate passes and the crowd-scale emergent behavior proves compelling, this becomes a Super-GOAT candidate for riir-ai (the combination with HLA + Committed Personality + cgsp curiosity is where the moat actually lives).
- **DEC continuity-equation fusion** (Fusion A in Research 371 §2.3 — `dec::belief_mass_divergence` on the κ-transport cochain) — separate plan if the GOAT gate passes and the DEC fusion proves load-bearing.
- **UQ extension N/A** — this is not a UQ-bearing primitive (no probability distribution, interval, or coverage claim). The "Report the Floor" rule does not apply.

---

## TL;DR

Ship `MeanFieldOverlap` (crowd `(κ, κ_a, Q)` aggregator) + `HopfBoundary` (closed-form 2×2 eigenvalue check) + `RegimeClassifier` (four-way enum) behind feature flag `mean_field_regime`. The paper's algorithmic content is ~80% covered by shipped primitives (LinOSS, `subspace_phase_gate`, `temporal_deriv`, `MicroRecurrentBeliefState`, `ict::BranchingDetector`); this plan ships the missing 20% — the crowd-scale mean-field order-parameter view + oscillatory-instability detector + regime taxonomy. **Mandatory defend-wrong PoC** (Phase 5 T5.1) verifies the classifier reproduces the paper's phase diagram before any promote-to-default. The wake/sleep/anesthesia biological mapping becomes a runtime knob: β is the per-NPC arousal scalar.
