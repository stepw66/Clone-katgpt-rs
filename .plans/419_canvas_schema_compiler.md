# Plan 419: Canvas Schema Compiler — Declared Causal Topology for Attention Masks

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md)
**Source paper:** [canvas-engineering.pdf](http://commandagi.com/research/canvas-engineering.pdf) — Valdez (CommandAGI), July 2026
**Target:** `crates/katgpt-core/src/canvas/` (new module) + Cargo feature `canvas_schema`
**Status:** ✅ DONE — Phase 1–6 complete, G1–G6 PASS (opt-in; `.issues/043` fusion PoC resolved inconclusively, promotion deferred — see Research 398 §8)

---

## Goal

Ship the **modelless half** of canvas engineering: a typed `CanvasSchema` compiler that lowers a declared region layout + directed topology into (a) an `AttentionMaskSpec` consumable by the existing sparse-attention paths (AC-Prefix, VortexFlow), (b) a `LossWeightMask` for training-time callers, and (c) a `reachability_horizon` / `can_reach` primitive proving the exact-marginal-independence guarantee for binary masks. Plus a `transfer_distance` semantic-type compatibility scalar.

**What this plan does NOT ship (training-dependent → riir-train follow-up):**
- Training a DiT within the declared topology (the 1.73× parameter-efficiency path).
- Looped-attention zero-init learned embeddings (covered by `LoopMode::WeightShared` Plan 108 / `LoopMode::TrainingFree` Plan 136).
- Representation-stability validation across seeds/backbones.

**GOAT gate (the contract):** the compiler + reachability primitives ship on structural/correctness merits — the reachability guarantee is provable by construction (absent edge ⟹ exact marginal independence for binary masks). The behavioral gain is NOT claimed at the modelless level (paper §5 shows modelless application is a 19% loss on untrained backbones); the fusion PoC (`.issues/043`, resolved-and-removed 2026-07-09, inconclusive) is documented in Research 398 §7–8. Promote-to-default requires the GOAT gate G1–G6 below; the gate measures *compiler correctness + reachability soundness + perf*, NOT behavioral parity with the paper's training-dependent results.

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
| 5-repo discipline | Ships in katgpt-core (generic math, no game/chain/shard semantics). Game-runtime fusion (typed NPC cognitive stack) is a riir-ai follow-up; `.issues/043` fusion PoC resolved inconclusively (see Research 398 §7–8). |
| Files < 2048 lines | Module split: `mod.rs` (types + compiler), `reachability.rs` (graph queries), `transfer.rs` (semantic distance), `mask.rs` (mask builder). |
| `Uuid::now_v7()` | N/A — no Uuids in this primitive. BLAKE3 commitment is a riir-neuron-db consumer concern (schema-mediated exchange), not this primitive. |

---

## Phase 1 — Skeleton (CORE) ✅ DONE

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/canvas/` module behind `canvas_schema` feature gate. Wired into `lib.rs`; feature registered in `Cargo.toml` (opt-in, NOT in `default`).
- [x] **T1.2** Core types in `canvas/types.rs` (AGENTS.md `types.rs` convention): `CanvasBounds`, `RegionId`, `SemanticType`, `AttentionFnFamily` (15 families), `RegionSpec`, `Connection`, `CanvasLayout`, `CanvasTopology`, `CanvasSchema`, `CompiledCanvas`, `AttentionMaskSpec`, `LossWeightMask`.
- [x] **T1.3** `region_indices(spec, layout) -> Range<usize>` (struct-offset arithmetic, contiguous-slab convention).
- [x] **T1.4** Topology constructors: `dense`, `isolated`, `hub_spoke`, `causal_chain`, `causal_temporal`.
- [x] **T1.5** Unit tests: region_indices, constructors, `CanvasSchema` structure.

**Phase 1 exit:** types + constructors compile; unit tests pass. ✅

---

## Phase 2 — The Compiler (mask + loss weight) ✅ DONE

### Tasks

- [x] **T2.1** `AttentionMaskSpec { n_positions, edges: Vec<(usize, usize, f32)> }` (sparse `M ∈ R^{N×N}_{≥0}`).
- [x] **T2.2** `temporal_aligns(t_src, t_dst, t_i, t_j) -> bool` (paper §2.3 `A_τ`; `t_i − t_src == t_j − t_dst` when both set; `None` = unconstrained).
- [x] **T2.3** `build_attention_mask(topology, region_indices, layout)` — pre-scan + one alloc; **paper convention** (query=src, key=dst; see Phase 5 record for the direction derivation).
- [x] **T2.4** `LossWeightMask` + `build_loss_weight_mask(layout, region_indices)` (`ω_i = Σ_r 1[i∈I_r]·loss_weight_r·1[is_output_r]`).
- [x] **T2.5** `compile_schema(schema) -> CompiledCanvas`.
- [x] **T2.6** Unit tests: causal-chain directed edges, isolated block-diagonal, loss-mask zeroing of non-output regions.

**Phase 2 exit:** compiler produces correct masks. ✅

---

## Phase 3 — Reachability Semantics (the provable guarantee) ✅ DONE

### Tasks

- [x] **T3.1** `canvas/reachability.rs` — information-flow graph `G` as CSR adjacency (`FlowGraph`); arc `dst → src` per connection (info flow). Reuses the CSR pattern from `viable_manifold_graph`.
- [x] **T3.2** `reachability_horizon(n_blocks, n_steps) -> n_blocks·n_steps`.
- [x] **T3.3** `can_reach(g, from, to, horizon)` — bounded BFS (convenience API; allocates a visited set per call).
- [x] **T3.4** `TransitiveClosure::build(g, horizon)` + `reaches(from, to)` — precomputed `(n×n)` bitset, **zero-alloc** O(1) hot path.
- [x] **T3.5** **THE SOUNDNESS TEST (G1):** `can_reach_absent_edge_means_no_reach` — isolated topology, region 0 cannot reach region 1 at any horizon. PASS.
- [x] **T3.6** **THE HORIZON TEST (G2):** `can_reach_respects_horizon_on_causal_chain` — `can_reach(A,C,1)=false`, `can_reach(A,C,2)=true`. PASS.

**Phase 3 exit:** reachability soundness proven by construction + tests. ✅ The load-bearing correctness property holds.

**Note — direction convention (recorded for future editors):** `Connection(src, dst)` licenses `src` to query `dst` (paper §2.2); info flows `dst → src`; `G` arc is `dst → src` (= info-flow direction); `can_reach(from,to)` reads as "`from` influences `to`". `causal_chain([A,B,C])` emits each region querying its predecessor → info arcs `A→B→C` → T3.6 holds. (Two earlier WIP conventions inverted this and broke T3.6; corrected to the paper convention.)

---

## Phase 4 — transfer_distance (semantic type compatibility) ✅ DONE

### Tasks

- [x] **T4.1** `transfer_distance(a, b) -> 1.0 − cosine` (zero-alloc; **f64 accumulation** to be overflow-safe for large-magnitude embeddings).
- [x] **T4.2** `compatible_regions(schema, max_distance)` + `compatible_regions_in_layout` (upper-triangle pairs below threshold).
- [x] **T4.3** Unit tests: identical → 0, orthogonal → 1, antiparallel → 2, zero-vector → 1 (conservative), parallel-representable → 0, parallel-overflow → 0 (via f64).

**Phase 4 exit:** semantic-type routing scalar ships. ✅

---

## Phase 5 — GOAT Gate (G1–G6) ✅ DONE — all PASS

Bench: `katgpt-core/benches/bench_419_canvas_schema_goat.rs`. Record: [`.benchmarks/419_canvas_schema_goat.md`](../.benchmarks/419_canvas_schema_goat.md).

### Tasks

- [x] **T5.1 (G1 — correctness)** Reachability soundness: isolated topology, region 0 cannot reach region 1 at horizons {0,1,2,10,100,1000,10000}. **PASS** (exact marginal independence by construction).
- [x] **T5.2 (G2 — soundness)** `can_reach(A,C,1)=false`, `can_reach(A,C,2)=true`, `reachability_horizon=n_blocks·n_steps`. **PASS**.
- [x] **T5.3 (G3 — no regression)** `cargo check --all-features` clean; `cargo check --no-default-features` does not pull `canvas`. **PASS**.
- [x] **T5.4 (G4 — alloc-free hot path)** `TransitiveClosure::reaches` + `reachability_horizon` = 0 allocs/1000 calls (CountingAllocator). `compile_schema` allocates only at load (3 `Vec`s). **PASS**.
- [x] **T5.5 (G5 — perf)** `compile_schema(199-region ICU schema)` = **1515 ns** (budget 10 ms, 6600× under); `reaches` p50 = **0 ns** (budget 100 ns). **PASS**.
- [x] **T5.6 (G6 — feature isolation)** `canvas_schema` gates all symbols; `--no-default-features` does not compile canvas. **PASS**.

**Promotion decision:** G1–G6 all pass, but **promotion to default-on is DEFERRED** (and the deferral is now indefinite, not just pending a PoC). The `.issues/043` fusion PoC resolved 2026-07-09 **inconclusively** — it could not isolate a canvas-attributable behavioral gain from the tuned classifier (see Research 398 §7). A faithfulness-probe re-PoC is the correct path if ever needed, but the primitive's constituents are **already DEFAULT-ON with runtime consumers and a measured showcase** (Plan 426 Steering × Geometry Cookbook; `region_subspace_steering` Plan 416 is default-on, wired via `region_subspace_bridge` in riir-engine). canvas_schema parks as an opt-in correctness-class primitive (reachability-by-construction, like DEC `d∘d=0`) until a real runtime path wants declared-topology-as-causal-graph. See Research 398 §8 for the full cross-repo context (default-on status, consumers, showcase) — captured inline to avoid re-grepping.

**What the GOAT gate does NOT measure (the honesty):** behavioral parity with the paper's training-dependent results (1.73× parameter efficiency, cortical R²=0.825). Those are riir-train's job. The modelless primitive ships the *compilation* and the *guarantee*, not the *behavioral gain*.

---

## Phase 6 — Documentation + consumer wiring sketch ✅ DONE

### Tasks

- [x] **T6.1** Added `canvas_schema` to the feature-flag catalog: [`.docs/09_feature_catalog/opt_in_features.md`](../.docs/09_feature_catalog/opt_in_features.md) §12 (one-line summary + GOAT table + honesty note). (The plan referenced `01_overview.md`; that file does not exist — the opt-in catalog is the canonical home for opt-in features.)
- [x] **T6.2** Doc example: the `canvas/mod.rs` module doc carries a compile-tested quick-start (`compile_schema` end-to-end on a 2-region canvas) + a reachability-guarantee quick-start (doctested). See also the G5 199-region ICU fixture in the bench.
- [x] **T6.3** Consumer contract documented in `mask.rs` (`build_attention_mask` doc: the `AttentionMaskSpec` is a sparse `(query, key, weight)` list that consumers lower to whatever dense/blocked form their kernel needs — generic `add log M to logits`, AC-Prefix, or VortexFlow). Actual wiring into AC-Prefix/VortexFlow is a separate follow-up, not this plan.
- [x] **T6.4** `.issues/043` cross-referenced as the tracked follow-up for the game-runtime Super-GOAT re-evaluation (linked from the benchmark record, the catalog entry, and the promotion-decision note above).

---

## Out of scope (tracked elsewhere)

- **Game-runtime fusion (typed NPC cognitive stack):** `.issues/043` fusion PoC resolved inconclusively (see Research 398 §7–8); future riir-ai plan only if a new angle emerges. NOT this plan.
- **Training a DiT within declared topology:** riir-train follow-up. NOT this plan.
- **Looped attention zero-init embeddings:** covered by `LoopMode::WeightShared` (Plan 108) / `LoopMode::TrainingFree` (Plan 136). NOT re-shipped here.
- **Schema-mediated latent exchange (freeze/thaw):** the substrate ships (`MerkleFrozenEnvelope`, `CommittedFieldBlend`). A schema-keyed exchange wrapper is a riir-neuron-db follow-up, NOT this plan.
- **Learned topology (propose/prune edges):** paper §6 open problem. Future research, NOT this plan.

---

## Notes

- **Why this is GOAT, not Super-GOAT:** the compiler is novel and modelless, but (a) constituent primitives ship, (b) the headline empirical value is training-dependent, (c) the reachability semantics is a reframing of sparse-attention-as-causal-graph. See Research 398 §3.1 for the full Q1–Q4 novelty-gate reasoning.
- **Why ship at all if the behavioral gain is training-dependent:** the compiler + reachability guarantee is a *correctness* primitive (absent edge = exact marginal independence by construction). Correctness primitives ship on their structural merits, like the DEC `d∘d=0` identity (Plan 251). The behavioral gain is a separate, tracked question.
- **Representation stability (paper §6 linchpin):** out of scope. The primitive does not claim latent geometry aligns across seeds; it only claims the *mask structure* is what the schema declares. Representation stability is an empirical property of trained models, validated in riir-train.
