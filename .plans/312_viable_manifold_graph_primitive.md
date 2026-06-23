# Plan 312: Viable Manifold Graph — Open Primitive

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/294_Viable_Manifold_Graph_Primitive.md](../.research/294_Viable_Manifold_Graph_Primitive.md)
**Private Super-GOAT guide:** [riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md](../../../riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md)
**Source paper:** [arxiv 2206.00106](https://arxiv.org/abs/2206.00106) — González-Duque et al., *Mario Plays on a Manifold*, 2022
**Target:** `katgpt-rs/crates/katgpt-core/src/viable_manifold_graph.rs` (new module) + Cargo feature `viable_manifold_graph`
**Status:** Active — Phase 0 complete (research + guide + this plan created in same session per Super-GOAT mandatory-output rule)

---

## Goal

Ship the generic, modelless, MIT-licensed open half of the Viable Manifold Graph Super-GOAT (R294 / riir-ai R154). Three composable primitives:

1. **`pullback_volume`** — given a smooth map `f: R^n → R^m` (closure) and a point `z`, return `log det(J_f(z)^T J_f(z))` via the existing `jacobian_svd_at` (Plan 301). This is the "cost-to-traverse" scalar field. Zero new SVD code — pure reduction over `SvdResult::singular_values`.
2. **`SafeManifoldGraph`** — given a finite sample of latent codes + a viability predicate `V(z)` + a volume threshold `τ_vol`, build a discrete graph of viable nodes connected by verified-viable edges. The graph is the discrete approximation of the safe manifold.
3. **`manifold_geodesic` + `manifold_random_walk`** — A* shortest path on the safe subgraph; uniform-over-neighbors (or weight-driven) random walk. Both stay inside the viable set by construction.

**No game semantics, no chain semantics, no shard semantics.** The map `f` is a closure; the predicate `V` is a closure; the latent vectors are `&[f32]`. The NPC-affect-specific wiring lives in `riir-ai` (R154 / future plan).

**GOAT gate rule (per AGENTS.md):** G1–G7 must PASS before promotion to default-on. Promotion decision deferred until G1–G7 are measured.

---

## Design constraints (non-negotiable, per AGENTS.md + R294)

- **Modelless** — inference-time only. No training, no gradients. The map `f` is provided by the caller; the primitive never backprops through it.
- **Zero-allocation hot path** — `manifold_random_walk` reuses caller-provided scratch; capacity-stable across 1000 steps (G6).
- **Generic over `f` and `V`** — both are closures. No HLA / functor / shard types leak into the open primitive.
- **Sigmoid not softmax** — any blending uses sigmoid (e.g., the calibrated-decoder α(z) semaphore if we expose it). Per AGENTS.md.
- **Latent-vs-raw boundary respected** — operates only on `&[f32]` + closures. Never touches sync. No boundary crossing by construction (G7).
- **Reuse existing infra** — `jacobian_svd_at` (Plan 301) for the SVD; we add only the `det(J^T J)` reduction + graph + navigation.
- **DRY** — graph storage reuses the pattern from `dense_mesh` (Vec-based, no graph crate dependency); A* reuses the pattern from `pruners/pathfinder.rs`.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Add feature gate `viable_manifold_graph = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` and `viable_manifold_graph = ["katgpt-core/viable_manifold_graph"]` to root `katgpt-rs/Cargo.toml`. **Depends on `subspace_phase_gate`** (for `jacobian_svd_at`).
- [ ] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/viable_manifold_graph.rs` with module doc referencing R294 + paper arxiv 2206.00106.
- [ ] **T1.3** Define `pub struct VolumeFieldConfig { pub log_eps: f32 }` — just ε for the `log(Π σᵢ² + ε)` reduction. Default `log_eps = 1e-12`.
- [ ] **T1.4** Implement `pub fn pullback_volume<F>(f: F, z: &[f32], scratch: &mut JacobianSvdScratch, cfg: &VolumeFieldConfig) -> f32 where F: Fn(&[f32], &mut [f32])`:
  - Call `jacobian_svd_at(f, z, eps, scratch)` (eps from scratch's config; reuse Plan 301's default 1e-4).
  - Return `Σ_i log(σᵢ² + cfg.log_eps)` over `result.singular_values`. (Numerically stable equivalent of `log(Π σᵢ² + ε)`.)
  - Zero new allocations beyond what `jacobian_svd_at` already does.
- [ ] **T1.5** Add `pub use viable_manifold_graph::{...}` to `katgpt-core/src/lib.rs` behind `#[cfg(feature = "viable_manifold_graph")]`.

**Exit:** `cargo check -p katgpt-core --features viable_manifold_graph` clean.

---

## Phase 2 — SafeManifoldGraph Construction

### Tasks

- [ ] **T2.1** Define `pub struct SafeManifoldGraph { pub dim: usize, nodes: Vec<f32>, edges: Vec<(u32, u32)> }` — `nodes` is flat `[n_nodes × dim]`, row-major. Edges are bidirectional, deduplicated, sorted.
- [ ] **T2.2** Define `pub struct GraphBuildConfig { pub volume_threshold: f32, pub edge_midpoint_check: bool, pub k_nearest: usize }`:
  - `volume_threshold` — keep nodes where `pullback_volume ≤ threshold`. Caller responsibility to pick (paper uses mean volume).
  - `edge_midpoint_check` — if true, verify the midpoint of each candidate edge is also viable (slower, more correct).
  - `k_nearest` — connect each viable node to its `k` nearest viable neighbors (paper uses grid adjacency; we generalize).
- [ ] **T2.3** Define `pub trait ViabilityPredicate { fn is_viable(&self, z: &[f32]) -> bool; }` for closure-agnostic predicate passing. Provide `pub struct ClosurePredicate<F>(pub F) where F: Fn(&[f32]) -> bool` impl.
- [ ] **T2.4** Implement `pub fn build_safe_manifold_graph<F, V>(f: F, samples: &[f32], dim: usize, predicate: &V, volume_cfg: &VolumeFieldConfig, build_cfg: &GraphBuildConfig, scratch: &mut JacobianSvdScratch) -> SafeManifoldGraph where F: Fn(&[f32], &mut [f32]), V: ViabilityPredicate`:
  - For each sample `z_i`: compute `vol_i = pullback_volume(&f, z_i, scratch, volume_cfg)`. Keep `i` iff `vol_i ≤ build_cfg.volume_threshold AND predicate.is_viable(z_i)`.
  - Build edges: for each kept node, find `k_nearest` kept neighbors (Euclidean in latent space). Optionally verify midpoint viability.
  - Return graph.
- [ ] **T2.5** Implement `SafeManifoldGraph::node_latent(&self, idx: u32) -> &[f32]` — O(1) slice into `nodes`.
- [ ] **T2.6** Implement `SafeManifoldGraph::neighbors(&self, idx: u32) -> Vec<u32>` (or iterator) — O(degree) scan over edges.

**Exit:** unit tests for connected-graph construction (predicate = always true → expected node count), disconnected-graph construction (predicate = "x > 0" → bipartite split).

---

## Phase 3 — Manifold Navigation

### Tasks

- [ ] **T3.1** Implement `pub fn manifold_geodesic(g: &SafeManifoldGraph, src: u32, dst: u32) -> Option<Vec<u32>>` — A* on the graph with Euclidean latent-distance heuristic. Reuse the priority-queue pattern from `pruners/pathfinder.rs::find_path`. Returns node-index path or `None` if unreachable.
- [ ] **T3.2** Implement `pub fn manifold_random_walk<R: Rng>(g: &SafeManifoldGraph, src: u32, m: usize, rng: &mut R) -> Vec<u32>` — uniform-over-neighbors walk for `m` steps. Returns the visited node-index sequence (length `m + 1`, including `src`). Caller owns the returned `Vec`; the walk itself uses a small stack buffer.
- [ ] **T3.3** (Optional, P2) Implement `pub fn manifold_curiosity_walk<R, W>(g: &SafeManifoldGraph, src: u32, m: usize, weights: &W, rng: &mut R) -> Vec<u32> where W: Fn(u32, u32) -> f32` — weighted-over-neighbors walk. The `weights` closure lets the caller (riir-ai) inject curiosity-driven neighbor preference without leaking cgsp types into katgpt-rs. **This is the riir-ai integration hook** — designed so `weights` can wrap `cgsp_runtime::curiosity_step` without that type being visible here.

**Exit:** unit tests for shortest path correctness (matches BFS on a small graph), random walk length, weighted walk converges to high-weight neighbor.

---

## Phase 4 — GOAT Gate Proofs (G1–G7)

### Tasks

- [ ] **T4.1** **G1** — `test_pullback_volume_identity_is_zero`: `pullback_volume(|x| *x, z, ...)` ≈ 0 for `z = [0.0; 4]` and `z = [1.0; 4]`. Tolerance 1e-6.
- [ ] **T4.2** **G2** — `test_pullback_volume_scaling_is_2n_log_c`: `pullback_volume(|x| x * 2.0, z, ...)` ≈ `2 * n * log(2)` for `n = 4`. Tolerance 1e-4.
- [ ] **T4.3** **G3** — `test_safe_graph_build_connected_when_predicate_true`: 100 samples on a 4D grid, predicate = always true, threshold = +∞ → all 100 nodes kept, graph is connected via k-nearest with k=4.
- [ ] **T4.4** **G3b** — `test_safe_graph_build_disconnected_when_predicate_splits`: same 100 samples, predicate = "z[0] > 0" → two disconnected components.
- [ ] **T4.5** **G4** — `test_manifold_geodesic_validity`: build a graph, pick two nodes, run `manifold_geodesic`, verify every node on the path satisfies the predicate. Verify path length ≤ free-space grid A* length on the same nodes.
- [ ] **T4.6** **G5** — `test_manifold_random_walk_validity`: build a graph, run `manifold_random_walk(src, m=25)`, verify every node on the walk satisfies the predicate. **Playability = 1.0 by construction.**
- [ ] **T4.7** **G6** — `test_manifold_random_walk_zero_alloc_across_1000_steps`: walk for 1000 steps, verify no `Vec` capacity growth in the scratch path. Per AGENTS.md hot-loop rule.
- [ ] **T4.8** **G7** — `test_primitive_never_touches_sync`: a static check (or a doc-test) that the primitive signature accepts only `&[f32]` + closures. The lint: the module must not import anything from `riir-chain`, `riir-neuron-db`, or any sync module. (Compile-pass test.)
- [ ] **T4.9** Add `benches/viable_manifold_graph_bench.rs`:
  - `pullback_volume` latency on `R^4 → R^4` (target: < 5µs, since it's one SVD call).
  - `manifold_random_walk` per-step latency (target: < 100ns/step for k=4 neighbors).
  - `build_safe_manifold_graph` on 1000 samples (target: < 10ms, dominated by 1000 SVD calls).
- [ ] **T4.10** Create `examples/viable_manifold_graph_01_basic.rs` — a small synthetic demo: build a safe graph on a 2D toy manifold (matching the paper's setup), run a geodesic and a random walk, print the playability comparison (manifold-constrained vs free Gaussian).

**Exit:** all G1–G7 green; bench numbers documented in `katgpt-rs/.benchmarks/312_viable_manifold_graph_goat.md` (new file).

---

## Phase 5 — Promotion Decision

### Tasks

- [ ] **T5.1** If G1–G7 all PASS: add `viable_manifold_graph` to `default = [...]` in `katgpt-rs/crates/katgpt-core/Cargo.toml` and root `katgpt-rs/Cargo.toml`.
- [ ] **T5.2** Add showcase section to `katgpt-rs/README.md` under "Feature Showcase" with the GOAT table (G1–G7) + the paper citation + a note that game-side wiring lives in riir-ai.
- [ ] **T5.3** Update `katgpt-rs/.docs/01_overview.md` Module Structure + Feature Flags tables.
- [ ] **T5.4** Update `katgpt-rs/.research/294_*.md` with a "Phase 5 result" footer: PROMOTED / DEMOTED + measured numbers + demotion rationale if applicable.

**Exit:** either promoted to default (G1–G7 PASS) or documented as opt-in with the failing gate(s) called out honestly.

---

## Phase 6 — riir-ai Wiring (DEFERRED to separate plan in riir-ai)

This plan ships the open primitive only. The riir-ai-side wiring (G8–G12 from the private guide) is a separate plan in `riir-ai/.plans/` (TBD after Phase 5). It will:

- Use `evolve_hla` as the map `f`.
- Use `latent_functor/quality_gate.rs` coherence as the predicate `V`.
- Store the per-NPC graph in the Entity Cognition Stack (Plan 327).
- Wire `manifold_curiosity_walk`'s `weights` closure to `cgsp_runtime::curiosity_step`.
- Add the designer-facing `schedule_persona_transition` API.

**Not in scope for Plan 312.**

---

## Risk Register

| Risk | Mitigation |
|---|---|
| `pullback_volume` is a 1-liner over `SvdResult`; overclaiming novelty | Honest framing in R294 §2.1: the SVD infra exists (Plan 301); the *reduction* + the *graph* + the *navigation* are the novel pieces, not the volume computation alone. |
| 8D HLA is too high for the discrete-graph approximation (paper used 2D) | Use `participation_ratio` (Plan 301) to estimate intrinsic affect dimensionality; build the graph in the intrinsic subspace. Documented in riir-ai R154 failure modes. |
| DenseMesh precedent (Plan 266 Gate 2 failed) | Different use case (navigation vs composition). Run G8/G9 honestly; demote if no empirical win. Documented in riir-ai R154 failure modes. |
| `jacobian_svd_at` is opt-in (`subspace_phase_gate`); forcing the dep chain | Acceptable — `viable_manifold_graph` depends on `subspace_phase_gate`. Both stay opt-in until各自 GOAT gates pass. Documented in feature-flag table. |
| Free exploration may already keep NPCs coherent (HLA is well-behaved) | G8 is the go/no-go gate. If free Gaussian walks in `R^8` HLA already produce ~0% incoherent states, VMG is Gain not Super-GOAT — demote honestly. |
| Hyperparameter sensitivity (volume threshold, k_nearest, edge_midpoint_check) | Document calibration recipe in Phase 4 bench notes. Paper's "mean volume" rule as default. |

---

## Cross-Refs

- [katgpt-rs/.research/294_Viable_Manifold_Graph_Primitive.md](../.research/294_Viable_Manifold_Graph_Primitive.md) — research note (this plan's source)
- [riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md](../../../riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md) — private Super-GOAT guide
- [katgpt-rs/.plans/301_runtime_subspace_phase_gate_primitive.md](301_runtime_subspace_phase_gate_primitive.md) — ships `jacobian_svd_at` (substrate)
- [katgpt-rs/.plans/309_latent_field_steering_primitive.md](309_latent_field_steering_primitive.md) — complement (top-down injection)
- [katgpt-rs/.plans/252_cubical_category_interval_topology.md](252_cubical_category_interval_topology.md) — raw-space geodesic cousin
- [katgpt-rs/.plans/266_densemesh_latent_node_network.md](266_densemesh_latent_node_network.md) — composition-graph precedent (cautionary)
- [katgpt-rs/.plans/297_personality_weighted_composition.md](297_personality_weighted_composition.md) — sigmoid-blend kernel shape (α(z) semaphore analogue)
