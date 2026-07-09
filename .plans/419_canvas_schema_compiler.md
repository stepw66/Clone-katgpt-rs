# Plan 419: Canvas Schema Compiler — Declared Causal Topology for Attention Masks

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md)
**Source paper:** [canvas-engineering.pdf](http://commandagi.com/research/canvas-engineering.pdf) — Valdez (CommandAGI), July 2026
**Target:** `crates/katgpt-core/src/canvas/` (new module) + Cargo feature `canvas_schema`
**Status:** Active — Phase 0 (planning)

---

## Goal

Ship the **modelless half** of canvas engineering: a typed `CanvasSchema` compiler that lowers a declared region layout + directed topology into (a) an `AttentionMaskSpec` consumable by the existing sparse-attention paths (AC-Prefix, VortexFlow), (b) a `LossWeightMask` for training-time callers, and (c) a `reachability_horizon` / `can_reach` primitive proving the exact-marginal-independence guarantee for binary masks. Plus a `transfer_distance` semantic-type compatibility scalar.

**What this plan does NOT ship (training-dependent → riir-train follow-up):**
- Training a DiT within the declared topology (the 1.73× parameter-efficiency path).
- Looped-attention zero-init learned embeddings (covered by `LoopMode::WeightShared` Plan 108 / `LoopMode::TrainingFree` Plan 136).
- Representation-stability validation across seeds/backbones.

**GOAT gate (the contract):** the compiler + reachability primitives ship on structural/correctness merits — the reachability guarantee is provable by construction (absent edge ⟹ exact marginal independence for binary masks). The behavioral gain is NOT claimed at the modelless level (paper §5 shows modelless application is a 19% loss on untrained backbones) and is tracked separately in `.issues/043` as a fusion PoC. Promote-to-default requires the GOAT gate G1–G6 below; the gate measures *compiler correctness + reachability soundness + perf*, NOT behavioral parity with the paper's training-dependent results.

---

## Constraints (per AGENTS.md + research skill)

| Constraint | How this plan satisfies it |
|---|---|
| Modelless / inference-time | All primitives are pure functions over index sets + graphs. Zero backprop, zero weight mutation. |
| Latent-to-latent preferred | The mask acts on latent positions; `transfer_distance` is a latent cosine. No token decoding. |
| Sigmoid, not softmax | No softmax in the compiler itself. The compiled mask is consumed by existing attention paths (which already follow the sigmoid-where-applicable rule). |
| Zero-alloc hot path | `compile_schema` allocates once at schema-load time. `can_reach` / `reachability_horizon` are pure queries over the compiled artifact (no per-call alloc). |
| CPU/SIMD/GPU auto-route | Compiler runs once at load (CPU). Mask consumption routes per the existing attention path's discipline. |
| Feature flag isolation | `canvas_schema` is opt-in (NOT default-on) until GOAT gate passes. |
| 5-repo discipline | Ships in katgpt-core (generic math, no game/chain/shard semantics). Game-runtime fusion (typed NPC cognitive stack) is a riir-ai follow-up gated on `.issues/043` PoC. |
| Files < 2048 lines | Module split: `mod.rs` (types + compiler), `reachability.rs` (graph queries), `transfer.rs` (semantic distance), `mask.rs` (mask builder). |
| `Uuid::now_v7()` | N/A — no Uuids in this primitive. BLAKE3 commitment is a riir-neuron-db consumer concern (schema-mediated exchange), not this primitive. |

---

## Phase 1 — Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/canvas/` module behind `canvas_schema` feature gate. Wire into `crates/katgpt-core/src/lib.rs` (`#[cfg(feature = "canvas_schema")] pub mod canvas;`).
- [ ] **T1.2** Define core types in `canvas/mod.rs`:
  - `CanvasBounds { t0, t1, h0, h1, w0, w1: u32 }` (region bounds).
  - `RegionId(usize)` newtype.
  - `SemanticType { name: &'static str, frozen_embedding: [f32; D] }` (D fixed, e.g. 64).
  - `AttentionFnFamily` enum: `Cross`, `Linear`, `Sigmoid`, `Gated`, `Perceiver`, `Pooling`, `Copy`, `Mamba`, `Rwkv`, `Hyena`, `Local`, `Sparse`, `None`, `RandomFixed`, `Mixture`. (Mirror paper §2.5; consumers dispatch on this.)
  - `RegionSpec { name, bounds, period, is_output, loss_weight, semantic_type: Option<SemanticType>, default_attn: AttentionFnFamily }`.
  - `Connection { src: RegionId, dst: RegionId, weight: f32, t_src: Option<i32>, t_dst: Option<i32>, fn_family: Option<AttentionFnFamily> }`.
  - `CanvasLayout { t, h, w, d_model: u32, regions: Vec<RegionSpec> }`.
  - `CanvasTopology { connections: Vec<Connection> }`.
  - `CanvasSchema { layout: CanvasLayout, topology: CanvasTopology }`.
  - `CompiledCanvas { region_indices: Vec<Range<usize>>, mask: AttentionMaskSpec, loss_mask: LossWeightMask }`.
- [ ] **T1.3** Implement `region_indices(spec: &RegionSpec, layout: &CanvasLayout) -> Range<usize>` — the struct-offset arithmetic from paper §2.3 (I_r index set as a contiguous range, since regions are axis-aligned boxes). Zero alloc.
- [ ] **T1.4** Implement convenience topology constructors (paper §2.2):
  - `dense(regions: &[RegionId]) -> CanvasTopology` (fully connected).
  - `isolated(regions: &[RegionId]) -> CanvasTopology` (block-diagonal self-attention only).
  - `hub_spoke(hub: RegionId, spokes: &[RegionId]) -> CanvasTopology`.
  - `causal_chain(chain: &[RegionId]) -> CanvasTopology` (A→B→C).
  - `causal_temporal(regions: &[RegionId]) -> CanvasTopology` (same-frame self + previous-frame cross, no future leakage — the default for temporal canvases).
- [ ] **T1.5** Unit tests (T1.1–T1.4): region_indices matches hand-computed offsets; constructors produce expected edge sets; `CanvasSchema` round-trips through serialization (serde behind feature).

**Phase 1 exit:** types + constructors compile; 10+ unit tests pass. No mask building yet.

---

## Phase 2 — The Compiler (mask + loss weight)

### Tasks

- [ ] **T2.1** Create `canvas/mask.rs`. Define `AttentionMaskSpec { n_positions: usize, edges: Vec<(usize, usize, f32)> }` — a sparse representation of `M ∈ R^{N×N}_{≥0}` from paper §2.3. This is NOT a dense matrix; consumers lower it to whatever dense/blocked form their attention kernel needs.
- [ ] **T2.2** Implement `temporal_aligns(t_src: Option<i32>, t_dst: Option<i32>, t_i: u32, t_j: u32) -> bool` — the `A_τ` temporal alignment predicate from paper §2.3. Unset offset = unconstrained. Set offsets: `∃ ref: t_i = ref + t_src ∧ t_j = ref + t_dst`.
- [ ] **T2.3** Implement `build_attention_mask(topology: &CanvasTopology, region_indices: &[Range<usize>]) -> AttentionMaskSpec`:
  - For each `Connection { src, dst, weight, t_src, t_dst, .. }`:
    - For each `(i, j)` with `i ∈ region_indices[src]`, `j ∈ region_indices[dst]`:
      - If `temporal_aligns(t_src, t_dst, t(i), t(j))`: emit edge `(i, j, weight)`.
  - Binary mask case (`weight ∈ {0.0, 1.0}`) is the load-bearing case for the reachability guarantee.
  - **Allocation discipline:** `edges.reserve_exact(total_pairs)` computed in a pre-scan (sum of `|src|·|dst|` over connections), then fill. One alloc.
- [ ] **T2.4** Define `LossWeightMask { weights: Vec<f32> }` (length N). Implement `build_loss_weight_mask(layout: &CanvasLayout, region_indices: &[Range<usize>]) -> LossWeightMask`:
  - `ω_i = Σ_r 1[i ∈ I_r] · loss_weight_r · 1[is_output_r]` (paper §2.3). Non-output regions get 0.
- [ ] **T2.5** Implement `compile_schema(schema: &CanvasSchema) -> CompiledCanvas` — the top-level compiler. Pure structure, zero gradient descent. Pre-allocates `CompiledCanvas` from layout dimensions.
- [ ] **T2.6** Unit tests: `build_attention_mask` on the paper's §2.1 example layout (visual 6×6×5, action 1×5, reward 1×1) produces the expected edge count; `causal_temporal` constructor produces same-frame self + previous-frame cross only; `loss_weight_mask` zeroes non-output regions.

**Phase 2 exit:** compiler produces correct masks. The reachability guarantee is not yet queryable but the mask substrate is sound.

---

## Phase 3 — Reachability Semantics (the provable guarantee)

### Tasks

- [ ] **T3.1** Create `canvas/reachability.rs`. Build the **information-flow graph** `G` from `CanvasTopology`: for each `Connection { src, dst, .. }`, add arc `dst → src` in `G` (paper §2.3 convention: content moves `s → r` because `r` queries `s`). Use CSR adjacency (reuse `katgpt-core`'s existing CSR pattern from `viable_manifold_graph` if compatible, else a small inline CSR).
- [ ] **T3.2** Implement `reachability_horizon(n_blocks: usize, n_steps: usize) -> usize` — returns `n_blocks * n_steps` (paper §2.3: one denoiser pass with L blocks moves info along paths ≤ L; K sampling steps compose to horizon K·L). Trivial but explicit — documents the causal-horizon invariant.
- [ ] **T3.3** Implement `can_reach(topology: &CanvasTopology, from: RegionId, to: RegionId, horizon: usize) -> bool` — BFS on `G` from `from`, bounded by `horizon` hops. Returns true iff a directed path of length ≤ horizon exists.
- [ ] **T3.4** Implement `transitive_closure_bounded(topology: &CanvasTopology, horizon: usize) -> ReachabilityMatrix` — precomputed `(n_regions × n_regions)` boolean matrix of reachability within horizon. Used by `can_reach` after a one-time precompute. Allocation at schema-load, not per-query.
- [ ] **T3.5** **THE SOUNDNESS TEST (G1):** for a binary-mask topology, construct two regions `a, b` with NO directed path `a → b` in `G`. Assert `can_reach(..., a, b, horizon) == false` for all `horizon`. This is the **exact marginal independence** claim (paper §2.3, load-bearing direction) — it holds *by construction* for binary masks. If this test fails, the compiler is broken.
- [ ] **T3.6** **THE HORIZON TEST (G2):** for a `causal_chain(A→B→C)` topology with horizon=1, assert `can_reach(A, C, 1) == false` (path length 2 > horizon 1) but `can_reach(A, C, 2) == true`. The converse direction is bounded, not exact (paper notes this).

**Phase 3 exit:** reachability soundness proven by construction + tests. This is the load-bearing correctness property of the whole primitive.

---

## Phase 4 — transfer_distance (semantic type compatibility)

### Tasks

- [ ] **T4.1** Create `canvas/transfer.rs`. Define `transfer_distance(a: &SemanticType, b: &SemanticType) -> f32` — `1.0 - cosine(a.frozen_embedding, b.frozen_embedding)` (paper §2.4). Zero alloc (operates on slices).
- [ ] **T4.2** Implement `compatible_regions(schema: &CanvasSchema, max_distance: f32) -> Vec<(RegionId, RegionId)>` — returns region pairs whose `transfer_distance` is below threshold (schema ABI compatibility check, paper §2.4 Table 1).
- [ ] **T4.3** Unit tests: two regions with identical embeddings → distance 0; orthogonal embeddings → distance 1; the paper's example (camera vs joint-angles) is not reproducible without their frozen embeddings, so use synthetic embeddings with known cosine.

**Phase 4 exit:** semantic-type routing scalar ships. This is a small primitive but it's genuinely missing from the corpus (grep confirmed zero hits for `transfer_distance`).

---

## Phase 5 — GOAT Gate (G1–G6)

### Tasks

- [ ] **T5.1 (G1 — correctness)** Reachability soundness test (T3.5) passes: absent edge ⟹ `can_reach == false` for all horizons. **This is the load-bearing gate.**
- [ ] **T5.2 (G2 — soundness)** Horizon test (T3.6) passes: `can_reach` respects the K·L horizon bound.
- [ ] **T5.3 (G3 — no regression)** `cargo check --all-features` clean. Feature isolation test: `cargo check --no-default-features` does not pull in `canvas_schema`.
- [ ] **T5.4 (G4 — alloc-free hot path)** `compile_schema` allocates exactly once (verified via `#[track_caller]` alloc counter or manual inspection). `can_reach` / `reachability_horizon` allocate zero per call (use precomputed `ReachabilityMatrix`).
- [ ] **T5.5 (G5 — perf)** `compile_schema` on the paper's 199-region ICU schema (§4) completes in < 10ms. `can_reach` query on a 200-region topology: < 100ns p50.
- [ ] **T5.6 (G6 — feature isolation)** `canvas_schema` feature gate does not leak symbols into default build. Binary size delta when disabled: 0 bytes.

**Promotion decision:** if G1–G6 all pass → promote `canvas_schema` to default-on in `katgpt-core` and root `katgpt-rs`. The primitive is opt-in structure; once the soundness + perf gates are green, it carries no runtime cost when unused.

**What the GOAT gate does NOT measure (the honesty):** behavioral parity with the paper's training-dependent results (1.73× parameter efficiency, cortical R²=0.825). Those are riir-train's job. The modelless primitive ships the *compilation* and the *guarantee*, not the *behavioral gain*.

---

## Phase 6 — Documentation + consumer wiring sketch

### Tasks

- [ ] **T6.1** Add `canvas_schema` to the feature-flag table in `katgpt-rs/.docs/01_overview.md` with a one-line summary + GOAT-gate status.
- [ ] **T6.2** Add a doc example showing the paper's §2.1 layout compiled end-to-end (visual/action/reward on an 8×8×5 canvas).
- [ ] **T6.3** Document the consumer contract: how `AttentionMaskSpec` lowers into (a) AC-Prefix's `AcPrefixMask`, (b) VortexFlow's sparse routing, (c) a generic dense `add log M to logits` path. This is a doc-only task — actual wiring into AC-Prefix/VortexFlow is a separate follow-up plan, not this one.
- [ ] **T6.4** Note the `.issues/043` fusion PoC as the tracked follow-up for the game-runtime Super-GOAT re-evaluation.

---

## Out of scope (tracked elsewhere)

- **Game-runtime fusion (typed NPC cognitive stack):** `.issues/043` + future riir-ai plan if PoC passes. NOT this plan.
- **Training a DiT within declared topology:** riir-train follow-up. NOT this plan.
- **Looped attention zero-init embeddings:** covered by `LoopMode::WeightShared` (Plan 108) / `LoopMode::TrainingFree` (Plan 136). NOT re-shipped here.
- **Schema-mediated latent exchange (freeze/thaw):** the substrate ships (`MerkleFrozenEnvelope`, `CommittedFieldBlend`). A schema-keyed exchange wrapper is a riir-neuron-db follow-up, NOT this plan.
- **Learned topology (propose/prune edges):** paper §6 open problem. Future research, NOT this plan.

---

## Notes

- **Why this is GOAT, not Super-GOAT:** the compiler is novel and modelless, but (a) constituent primitives ship, (b) the headline empirical value is training-dependent, (c) the reachability semantics is a reframing of sparse-attention-as-causal-graph. See Research 398 §3.1 for the full Q1–Q4 novelty-gate reasoning.
- **Why ship at all if the behavioral gain is training-dependent:** the compiler + reachability guarantee is a *correctness* primitive (absent edge = exact marginal independence by construction). Correctness primitives ship on their structural merits, like the DEC `d∘d=0` identity (Plan 251). The behavioral gain is a separate, tracked question.
- **Representation stability (paper §6 linchpin):** out of scope. The primitive does not claim latent geometry aligns across seeds; it only claims the *mask structure* is what the schema declares. Representation stability is an empirical property of trained models, validated in riir-train.
