# Plan 314: Stokes Calculus Wrappers — Fokker-Planck Validator + Boundary-Flux Mass + Line Integral

**Date:** 2026-06-24
**Research:** [katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md](../.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md)
**Source papers:**
- [arxiv 2202.11322](https://arxiv.org/abs/2202.11322) — *Efficient CDF Approximations for Normalizing Flows* (TMLR 2022) — "leverage the divergence theorem to estimate the CDF over a closed region in target space"
- [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/cbf8710b43df3f2c1553e649403426df-Abstract.html) — *Neural Manifold Ordinary Differential Equations* (Lou et al.) — `d/dt log p = -div(f)` instantaneous change-of-variables
**Target:** `katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs` (new file) + Cargo feature `stokes_calculus` (under `dec_operators`)
**Status:** Active — Phase 1 unblocked

---

## Goal

Three thin wrapper primitives (<50 LOC each) over the shipped DEC operators (Plan 251), exposing the Generalized Stokes' Theorem as named modelless tools:

1. **`belief_mass_divergence()`** — Fokker-Planck belief-mass validator. Closes Research 271 §2's known gap ("Fokker-Planck / continuity equation → not yet a runtime invariant validator"). Headline primitive — feeds ICT BranchingDetector (Plan 294) a modelless invariant.
2. **`boundary_flux_mass()`** — divergence theorem for low-dim manifolds. `∫_∂M ω` instead of `∫_M dω`. O(boundary) vs O(volume). Bounded error from `hodge_decompose`'s harmonic component.
3. **`line_integral()`** — discrete line integral of a rank-1 cochain along a path. Path energy / geodesic cost / work. Composes with Plan 312's `manifold_geodesic` path output.

**GOAT gate:** each primitive has its own A/B benchmark (see Phase 3). Promote the winners; demote the losers. Per Research 296 verdict, this is **GOAT** (not Super-GOAT — the mechanism ships, only the wrappers + framing are new).

## Non-Goals

- ❌ NO new DEC operators (Plan 251 shipped them). This plan only wraps.
- ❌ NO training, NO flow-matching policy learning (those → riir-train). The papers' training machinery is explicitly out of scope per Research 296 §1.
- ❌ NO high-dim shard compression via boundary commitment (curse of dimensionality — see Research 296 §3.5).
- ❌ NO Super-GOAT guide / riir-ai guide (verdict is GOAT; guide is Super-GOAT-only per skill §1.5). HLA wiring lands as a Phase 2 task, not a separate guide doc.

## Constraint Checklist (per AGENTS.md + skill)

- [ ] Modelless (inference-time only, no backprop) — ✓ by construction (DEC ops are linear algebra)
- [ ] Latent-to-latent preferred (sigmoid not softmax) — ✓ (no softmax in this primitive)
- [ ] Freeze/thaw over fine-tuning — ✓ (no weight mutation)
- [ ] 5-repo discipline (open primitive in katgpt-rs) — ✓
- [ ] SOLID, DRY, zero-alloc hot paths — ✓ (wrappers reuse DEC scratch buffers)
- [ ] CPU/SIMD/GPU auto-route inherited from DEC `backend.rs` — ✓
- [ ] File < 2048 lines — ✓ (single file, ~300 LOC + tests)

---

## Phase 1 — Unblocking Skeleton (CORE)

Three pure functions over shipped DEC types. No new structs. No allocations in the wrapper layer (delegate to `codifferential` / `exterior_derivative` which already reuse scratch buffers).

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs`. Add `mod stokes_calculus;` to `dec/mod.rs` behind `dec_operators` feature. Add Cargo feature `stokes_calculus = ["katgpt-core/dec_operators"]` to `katgpt-rs/Cargo.toml` (opt-in, NOT default).
- [ ] **T1.2** Implement `pub fn belief_mass_divergence(cx: &CellComplex, belief_flow: &CochainField) -> f32`. Returns `|Σ_v (codifferential ∘ belief_flow)[v]|` over all vertices (rank-1 → rank-0 divergence, then L1 sum). Pure wrapper over `codifferential` (shipped). ~15 LOC.
- [ ] **T1.3** Implement `pub fn boundary_flux_mass(cx: &CellComplex, region_cells: &[u32], field: &CochainField) -> (mass: f32, error_bound: f32)`. Compute `Σ_{e ∈ ∂region} (exterior_derivative ∘ field)[e]` as `mass`, and `‖harmonic(field)‖₁` (from `hodge_decompose`) as `error_bound`. Pure wrappers. ~25 LOC.
- [ ] **T1.4** Implement `pub fn line_integral(cx: &CellComplex, edge_field: &CochainField, path: &[u32]) -> f32`. Sum `edge_field[e]` for each edge `e` on the path (path is a vertex-index slice from `manifold_geodesic` or `manifold_random_walk`). Look up each consecutive `(path[i], path[i+1])` pair's edge index via the cell complex's boundary entries. ~30 LOC.
- [ ] **T1.5** `cargo check --features stokes_calculus` passes. `cargo test -p katgpt-core --features stokes_calculus --lib` passes (unit tests in Phase 2).

**Exit:** three functions compile and type-check. Zero allocations in the wrapper layer.

---

## Phase 2 — Unit Tests (Correctness, not perf)

Each primitive gets ≥3 unit tests: identity case, scaling case, edge case.

### Tasks

- [ ] **T2.1** `belief_mass_divergence` tests:
  - [ ] **T2.1.1** Identity: divergence of a constant flow on a uniform grid = 0 (mass conserved).
  - [ ] **T2.1.2** Scaling: divergence scales linearly with flow magnitude.
  - [ ] **T2.1.3** Anomaly injection: artificially inflate one edge's flow, verify divergence spikes (the "anomaly" case).
- [ ] **T2.2** `boundary_flux_mass` tests:
  - [ ] **T2.2.1** Stokes identity: for a purely exact field (curl-free gradient), `boundary_flux_mass(mass) == full_volume_integral(mass)` to within error_bound. This is the Generalized Stokes' Theorem test.
  - [ ] **T2.2.2** Harmonic bound: for a field with a known harmonic component, `|boundary_mass - volume_mass| ≤ error_bound`.
  - [ ] **T2.2.3** Empty region: `region_cells = []` returns `(0.0, 0.0)` without panicking.
- [ ] **T2.3** `line_integral` tests:
  - [ ] **T2.3.1** Straight path: line integral of a constant field over a path of length L = field_value × L.
  - [ ] **T2.3.2** Reversal antisymmetry: `line_integral(path A→B) == -line_integral(path B→A)` (work reverses sign).
  - [ ] **T2.3.3** Closed loop of an exact field = 0 (gradient theorem / fundamental theorem of calculus for line integrals).
- [ ] **T2.4** Run full test suite: `cargo test -p katgpt-core --features stokes_calculus --lib`. All pass.

**Exit:** three primitives verified correct against their Stokes-theorem identities.

---

## Phase 3 — GOAT Gate (Benchmarks)

Each primitive gets an A/B benchmark vs the naive alternative. **Promote the winner if gain ≥ threshold; demote the loser.** Per Research 296 §4.

### Tasks

- [ ] **T3.1** **G-A (Fokker-Planck validator GOAT):** A/B in riir-ai — does flagging `belief_mass_divergence > τ` catch ICT `BranchingDetector` (Plan 294) branching events earlier/cheaper than the existing JS-divergence-to-mean detector? **This gate runs in riir-ai (needs live HLA), not katgpt-core.** Target: ≥1.5× earlier detection OR ≥2× cheaper per-tick on the same events. **Deferred to riir-ai follow-up plan** (file when HLA wiring starts). Until then, `stokes_calculus` stays opt-in.
- [ ] **T3.2** **G-B (Boundary-flux mass GOAT):** A/B in katgpt-core — on a 256×256 game map (Bomber arena scale), compute "zone threat total" via `boundary_flux_mass` vs full-volume summation. Target: **≥3× faster** with `error_bound / mass < 5%`. Bench in `katgpt-rs/.benchmarks/314_stokes_calculus_goat.md`. If the threshold misses because the field has large harmonic component → the win is conditional (only for near-exact fields); document the condition, keep opt-in.
- [ ] **T3.3** **G-C (Line integral GOAT):** A/B in katgpt-core — on a `SafeManifoldGraph` from Plan 312, run `manifold_geodesic` (unweighted A*) vs `manifold_geodesic` + `line_integral`-weighted reranking of equal-length paths. Metric: number of direction reversals per unit path length. Target: **≥20% fewer reversals** at equal path length. If misses → keep opt-in; line integral still useful as a pure cost function even without the navigation-smoothness win.
- [ ] **T3.4** Write benchmark summary in `katgpt-rs/.benchmarks/314_stokes_calculus_goat.md`. Honest results — if a gate fails, document WHY (e.g. "harmonic component too large for boundary-flux win on this map class").
- [ ] **T3.5** Promotion decision:
  - [ ] If G-B AND G-C pass → promote `stokes_calculus` to default-on in `katgpt-rs/Cargo.toml`.
  - [ ] If only one passes → split: promote the winning primitive's sub-feature, keep the other opt-in.
  - [ ] If both fail → keep `stokes_calculus` opt-in, document as "modelless Stokes-theorem toolkit for callers who want it; not a default-on win".
  - [ ] G-A is independent (runs in riir-ai later); its result feeds back into the promotion decision when available.

**Exit:** GOAT gate run, results documented, promotion decided.

---

## Phase 4 — riir-ai Wiring (Deferred, runs in riir-ai)

This phase lands in `riir-ai/crates/riir-engine/`, NOT katgpt-rs. It is the G-A gate (T3.1) plus the actual HLA integration. **File as a riir-ai plan when this Phase 1–3 work is done.** Cross-reference this plan.

### Tasks (sketch — to be elaborated in riir-ai plan)

- [ ] **T4.1** Construct a `CellComplex` on the HLA belief manifold (8-dim). Options: (a) lattice discretization of the 5 synced scalars' range, (b) reuse `SafeManifoldGraph` from Plan 312 to build a discrete approximation. Decide and document.
- [ ] **T4.2** Per HLA tick, compute the belief-flow cochain from the HLA state delta (`evolve_hla`'s update vector). Feed to `belief_mass_divergence()`.
- [ ] **T4.3** Run G-A gate (T3.1): does `belief_mass_divergence > τ` catch ICT branching events earlier/cheaper than JS-divergence?
- [ ] **T4.4** Wire into `cgsp_runtime/pulse_bridge.rs` as a curiosity signal (divergence > 0 = expanding belief = curiosity). Cross-reference Plan 277 (Temporal Derivative Kernel).
- [ ] **T4.5** Feed to LatCal commitment: the 5 synced scalars should be the boundary flux of a near-zero-divergence field. If `belief_mass_divergence > τ` on the committed slice → flag for `mape_k.rs` self-healing (riir-neuron-db).

---

## Architecture

```
katgpt-rs/crates/katgpt-core/src/dec/
├── mod.rs              — add `mod stokes_calculus;` behind `dec_operators`
├── operators.rs        — EXISTING (d, δ, Δ) — wrapped, not modified
├── hodge.rs            — EXISTING (hodge_decompose) — wrapped, not modified
├── flow.rs             — EXISTING (DecFlowField) — orthogonal
└── stokes_calculus.rs  — NEW (~300 LOC + tests)
    pub fn belief_mass_divergence(cx, belief_flow) -> f32
    pub fn boundary_flux_mass(cx, region_cells, field) -> (mass, error_bound)
    pub fn line_integral(cx, edge_field, path) -> f32
    mod tests  // ~12 unit tests (Phase 2)
```

## Feature Gate

```toml
# katgpt-rs/Cargo.toml
[features]
stokes_calculus = ["katgpt-core/dec_operators"]  # opt-in, NOT default until G-B + G-C pass
```

Promotion to default requires G-B AND G-C passing (T3.5). G-A (riir-ai) feeds back later.

## Validation

- [ ] All 12 Phase-2 unit tests pass (Stokes identities hold by construction).
- [ ] G-B benchmark: boundary-flux mass ≥3× faster than full-volume on 256×256 map, error < 5%.
- [ ] G-C benchmark: line-integral-weighted geodesic ≥20% fewer reversals.
- [ ] G-A (riir-ai, deferred): Fokker-Planck validator catches ICT branching ≥1.5× earlier or ≥2× cheaper.
- [ ] Zero allocations in the wrapper layer (delegate to DEC scratch buffers).
- [ ] Files < 2048 lines.

## Honest Risk Notes

- **G-B may fail** if real game-map fields have large harmonic components (topologically-constrained flow). In that case the boundary-flux win is conditional on "near-exact fields only". This is honest — document it, don't hide it. Plan 251's `DecFlowField` already separates the three components, so callers can check `‖harmonic‖ / ‖total‖` before using `boundary_flux_mass`.
- **G-C may fail** if `manifold_geodesic` paths are already near-optimal in reversal count. The line integral is still useful as a pure cost function (e.g. for NPC path comparison) even if it doesn't improve geodesic smoothness.
- **G-A is the highest-value gate** but runs in riir-ai, not katgpt-core. If it passes, the Fokker-Planck validator becomes the headline application; if it fails, the primitive is still a clean Stokes-theorem tool for callers who want mass-conservation checking.
- **Super-GOAT follow-up (out of scope here):** if G-A reveals that projecting the belief flow onto its divergence-free component (enforce mass conservation by construction, not just validate) gives a steering signal, THAT is Super-GOAT-tier. Track in `katgpt-rs/.issues/` after Phase 3 lands. See Research 296 §5 "selling-point honesty".
