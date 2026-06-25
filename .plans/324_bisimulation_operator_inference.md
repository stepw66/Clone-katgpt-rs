# Plan 324: Bisimulation Operator Inference — Open Primitive

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md](../.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md)
**Source paper:** [arXiv:2602.19260](https://arxiv.org/pdf/2602.19260) — Duggan, Lorang, Lu, Scheutz (Tufts), Feb 2026 ("The Price Is Not Right")
**Underlying NSM method:** [arXiv:2508.21501](https://arxiv.org/abs/2508.21501) — Lorang et al., Aug 2025 (few-shot neuro-symbolic imitation learning)
**Target:** `katgpt-rs/crates/katgpt-core/src/bisimulation/` (new module) + Cargo feature `bisimulation_operator_inference`
**Status:** Active — Phase 0 (planning)

---

## Goal

Ship a generic, modelless, **bisimulation-based operator-abstraction primitive** that closes the Research 264 (Closure-Expansion Instrument) gap on the Primitive Transition Graph (PTG) + motif-mining loop. Given a stream of observed state transitions `(s, a, s′, label)`, the primitive produces:

1. A **minimal bisimulation quotient** — a partition of the observed states into equivalence classes such that two states are equivalent iff their outgoing labeled transitions lead to equivalent successor classes.
2. An **inferred operator schema** — one abstract operator per edge-label in the quotient graph, with preconditions (src class membership) and effects (dst class membership).
3. A **chain-committable canonical form** — BLAKE3 hash of the quotient graph `(classes, edges, operator_labels)`, suitable for LatCal-style commitment and anti-cheat replay.

This is the **PDDL-side counterpart** to the existing CWM primitive (Plan 296): where CWM induces *executable code* from trajectories via an LLM refinement loop, this primitive induces a *symbolic operator schema* via a deterministic graph algorithm. The runtime can pick per task: code induction for rich domains, operator-schema induction for structured/combinatorial domains.

**Why GOAT (not Super-GOAT):** the capability class ("system observes structured task, induces verifiable rules, plans via search") is already shipped as the CWM Super-GOAT (R275/Plan 296). This primitive is a *lighter-weight induction path* that closes a flagged gap (R264 §2.2 gap #1 — PTG data structure missing) and provides the missing half of the motif-mining loop (R264 §2.2 gap #2 — recurring sub-path consolidation).

**GOAT gate (open primitive):**

| Gate | Target | How measured |
|---|---|---|
| **G1** Bisimulation correctness | Known graph → known minimal quotient, bit-identical across re-runs | Property tests vs hand-computed canonical partitions |
| **G2** Operator inference soundness | Every observed transition covered; no spurious operators | Coverage check on the source trajectory set |
| **G3** Plan validity | Planner on inferred schema produces executable plans | Replay plans against original transition graph; assert no precondition violations |
| **G4** Latency | Partition refinement ≤ 1 ms for N=1024 nodes | `benches/bisimulation_bench.rs`, release build, Apple Silicon arm64 |
| **G5** Zero-alloc hot path | `class_id(state) -> u32` is O(1), no heap alloc across 10⁶ queries | Allocator-tracking test (mirrors Plan 320 G4 discipline) |

**Constraints:** modelless (no training, no backprop); latent-to-latent preferred (class embeddings are sigmoid projections on direction vectors, never softmax); zero-alloc hot path; deterministic audit record crosses sync boundary as raw `(StateClassId, OperatorLabel, Edge)` triples.

---

## Phase 1 — Types Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create module `crates/katgpt-core/src/bisimulation/mod.rs` with feature gate `#[cfg(feature = "bisimulation_operator_inference")]`. Re-export public API.
- [ ] **T1.2** Define `StateId` (newtype around `u32`, `#[repr(transparent)]`, `Copy`, `Eq`, `Hash`, `Ord`) in `crates/katgpt-core/src/bisimulation/types.rs`.
- [ ] **T1.3** Define `OperatorLabel` (`#[repr(u8)]` enum, `Copy`, `Eq`, `Hash`) — abstract operator tag. `Other(u8)` escape hatch for domain extension.
- [ ] **T1.4** Define `Transition { from: StateId, to: StateId, op: OperatorLabel }` (`#[repr(C)], Copy`). Fixed-size; no heap in the transition record itself.
- [ ] **T1.5** Define `TransitionGraph` in `crates/katgpt-core/src/bisimulation/graph.rs`:
  ```rust
  pub struct TransitionGraph {
      states: Vec<StateId>,                  // dense, sorted
      edges: Vec<Transition>,                // sorted by (from, op)
      edge_index: Vec<(StateId, usize)>,     // (state, offset into edges) for O(log) adjacency
  }
  ```
  Builder: `TransitionGraphBuilder::push_transition(from, to, op)`, `build()` sorts + indexes in-place.
- [ ] **T1.6** Define `StateClassId` (newtype around `u32`, `#[repr(transparent)]`).
- [ ] **T1.7** Define `QuotientEdge { from: StateClassId, to: StateClassId, op: OperatorLabel }` (`#[repr(C)], Copy`).
- [ ] **T1.8** Define `BisimulationQuotient`:
  ```rust
  pub struct BisimulationQuotient {
      pub n_classes: u32,
      pub state_to_class: Vec<StateClassId>,   // indexed by StateId
      pub quotient_edges: Vec<QuotientEdge>,   // sorted, deduped
      pub blake3: [u8; 32],                    // canonical commitment
  }
  ```
- [ ] **T1.9** Unit tests: builder produces sorted edges; `edge_index` lookup is correct; empty graph → empty quotient.

### Acceptance

`cargo check -p katgpt-core --features bisimulation_operator_inference` compiles. Unit tests pass. No heap allocations in `TransitionGraph::adjacency(state)` (verify via `#[track_caller]` allocator hook).

---

## Phase 2 — Partition Refinement (the bisimulation algorithm)

### Tasks

- [ ] **T2.1** Implement `partition_refine(graph: &TransitionGraph) -> BisimulationQuotient` in `crates/katgpt-core/src/bisimulation/refine.rs`. Algorithm: Paige-Tarjan 1987 (or Kanellakis-Smolka hopcroft-style), O((|S| + |E|) log |S|) worst case.
  - Initial partition: one block per distinct `(sorted edge-label multiset)` signature (the "signature" of a state = multiset of its outgoing operator labels).
  - Refine iteratively: split any block whose members have edges going to ≥2 distinct current-block targets under the same operator label. Repeat until stable.
- [ ] **T2.2** Implement `canonicalize(quotient: &mut BisimulationQuotient)`: renumber classes by the smallest `StateId` in each block (deterministic); sort `quotient_edges` lexicographically; dedup.
- [ ] **T2.3** Implement `blake3_commit(quotient: &BisimulationQuotient) -> [u8; 32]`: hash the canonical byte serialization `n_classes_LE || state_to_class_LE || quotient_edges_LE`.
- [ ] **T2.4** Unit tests:
  - Two isomorphic graphs produce bit-identical `blake3`.
  - A 3-state chain `A -op1-> B -op2-> C` quotients to 3 classes.
  - A 4-state graph `A -op1-> B`, `C -op1-> D` where `B,D` are "sink" states quotients `A,C` together and `B,D` together → 2 classes.
  - Re-running `partition_refine` on the same graph produces identical `state_to_class`.
- [ ] **T2.5** Property test (proptest): for any random graph, `partition_refine(partition_refine(g)).blake3 == partition_refine(g).blake3` (idempotence).

### Acceptance

All G1 tests pass. Latency on N=1024 random-transition graph ≤ 1 ms (G4 — measured in Phase 4 bench, threshold-checked here on a smoke graph).

---

## Phase 3 — Operator Schema Inference

### Tasks

- [ ] **T3.1** Define `OperatorSchema` in `crates/katgpt-core/src/bisimulation/operator.rs`:
  ```rust
  pub struct OperatorSchema {
      pub operators: Vec<OperatorDef>,
      pub blake3: [u8; 32],
  }

  pub struct OperatorDef {
      pub label: OperatorLabel,
      pub preconditions: Vec<StateClassId>,   // src classes that can invoke this op
      pub effects: Vec<(StateClassId, StateClassId)>,  // (src, dst) class transitions
  }
  ```
- [ ] **T3.2** Implement `infer_operators(quotient: &BisimulationQuotient) -> OperatorSchema`:
  - One `OperatorDef` per distinct `OperatorLabel` in `quotient_edges`.
  - `preconditions` = sorted unique `from` classes for that label.
  - `effects` = sorted unique `(from, to)` pairs for that label.
  - `blake3` over the canonical serialization.
- [ ] **T3.3** Unit tests (G2):
  - Every edge in the quotient is covered by exactly one `(label, from, to)` tuple in `effects`.
  - No spurious operators: every `OperatorDef.label` is exercised by ≥1 edge.
  - Coverage check: feeding the source trajectory set back through the schema, every transition's `(state, op)` is admitted (state's class ∈ `preconditions` for `op`).
- [ ] **T3.4** Integration test: a hand-crafted Towers-of-Hanoi-style graph (3 pegs, 3 disks, ~27 reachable states) produces a schema with operators `{PickTop, PlaceOn, PlaceOnEmpty}` (or equivalent label set), and the schema admits all transitions in the source trajectory set.

### Acceptance

G2 passes. Schema covers the source trajectory set with no spurious operators.

---

## Phase 4 — Plan Validity (G3) + Latency Bench (G4)

### Tasks

- [ ] **T4.1** Implement a minimal classical planner `plan(schema: &OperatorSchema, start: StateClassId, goal: StateClassId) -> Option<Vec<OperatorLabel>>` in `crates/katgpt-core/src/bisimulation/planner.rs`. Breadth-first search over the quotient graph's operator-labeled edges (sufficient for G3; MetricFF-grade planning is out of scope for the open primitive).
- [ ] **T4.2** G3 validity test:
  - For each `(start, goal)` pair in the source trajectory set where a path exists, `plan()` returns `Some(sequence)`.
  - Replaying the sequence against the original `TransitionGraph` (mapping class IDs back to representative states) never violates operator preconditions and reaches a state in the goal class.
- [ ] **T4.3** G3 negative test: for unreachable `(start, goal)` pairs, `plan()` returns `None`.
- [ ] **T4.4** Create `benches/bisimulation_bench.rs` (mirrors `salience_tri_gate_bench.rs` convention — no Criterion dev-dep). Measure:
  - `partition_refine` on N ∈ {64, 256, 1024, 4096} random-transition graphs.
  - `class_id` lookup throughput (target ≥ 100M lookups/sec).
  - `infer_operators` on the resulting quotients.
- [ ] **T4.5** G4 gate: `partition_refine` ≤ 1 ms for N=1024 on Apple Silicon arm64 release build. Document the actual number in the bench output.

### Acceptance

G3 + G4 pass. Bench output committed to `.benchmarks/324_bisimulation_goat.md` (create on Phase 5 completion).

---

## Phase 5 — Zero-Alloc Hot Path + GOAT Gate Doc

### Tasks

- [ ] **T5.1** Implement `BisimulationQuotient::class_of(&self, state: StateId) -> StateClassId` — direct index into `state_to_class`, O(1), no alloc.
- [ ] **T5.2** G5 test: `class_of` called 10⁶ times on a fixed quotient; verify via allocator hook that `Vec::capacity` of the test's scratch buffer is unchanged before/after (mirrors Plan 320 G4 / Plan 292 G5 discipline).
- [ ] **T5.3** Create `.benchmarks/324_bisimulation_goat.md` documenting G1–G5 results. Format mirrors `296_induced_cwm_primitive_goat.md`.
- [ ] **T5.4** Promotion decision:
  - All G1–G5 PASS → keep `bisimulation_operator_inference` **opt-in** by design (it's a primitive, not a default-on capability — downstream pipelines opt in by enabling the feature). Mark ready for downstream consumption.
  - Any FAIL → stay opt-in, file `.issues/NNN_*` follow-up, do NOT promote.

### Acceptance

GOAT gate doc committed. Feature remains opt-in by design (same policy as Induced CWM, Plan 296).

---

## Phase 6 — Docs + Cross-References

### Tasks

- [ ] **T6.1** Add module-level rustdoc to `crates/katgpt-core/src/bisimulation/mod.rs` covering: purpose, relationship to CWM (complementary induction path), relationship to R264 PTG gap, latent-vs-raw boundary.
- [ ] **T6.2** Update `katgpt-rs/crates/katgpt-core/src/lib.rs` feature gate comment block to list `bisimulation_operator_inference`.
- [ ] **T6.3** Cross-reference from `katgpt-rs/.research/275_*.md` (CWM) §fusion: add a row noting bisimulation as the lighter-weight PDDL-side counterpart.
- [ ] **T6.4** Cross-reference from `katgpt-rs/.research/264_*.md` (CEI) §2.2 gap list: mark gap #1 (PTG data structure) as concretely instantiable via this primitive.
- [ ] **T6.5** Add example `examples/bisimulation_demo.rs`: build a small Towers-of-Hanoi transition graph, refine, infer operators, plan, print the quotient + schema + BLAKE3.

---

## Out of Scope

- **Diffusion policy training** for skills `π_{i,j}`. Training-side → riir-train.
- **MetricFF-grade classical planner.** BFS over quotient edges suffices for G3; full PDDL planning is downstream consumer's problem (could be `mcts_search` over the induced CWM, or a real PDDL solver wired by the consumer).
- **LLM-based symbolic abstraction** (the ASP solver step from the paper). Out of scope for the open primitive — that's the heavier-weight CWM path. This primitive ships the *deterministic* half: graph → bisimulation → operator schema, no LLM.
- **Per-NPC heterogeneous quotients.** Already foreshadowed in R275 fusion; runtime integration is riir-ai's responsibility.
- **LatCal chain commitment wiring.** The `blake3` field is the bridge artifact; actually wiring it into a chain block is riir-chain's responsibility.
- **Real-world robot / game integration.** Engine primitive only.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **R1 — Partition refinement blows up on degenerate inputs** (e.g., fully connected graph → no compaction) | Property tests with adversarial inputs; document the worst-case O((S+E) log S) bound in rustdoc; cap `n_classes` at `states.len()` (no class can be smaller than 1 state). |
| **R2 — BLAKE3 commitment is not stable across schema evolutions** | `canonicalize()` renumbers classes deterministically by smallest `StateId`; serialization is fixed byte layout. Property test: isomorphic graphs → identical BLAKE3. |
| **R3 — Bisimulation is too fine-grained** (preserves *too much*, no compaction) | This is correct behavior, not a bug. The paper applies bisimulation to *abstract* states (post-feature-selector `φ`), not raw states. The consumer is responsible for picking the right state abstraction; this primitive quotients whatever graph it's given. |
| **R4 — Bisimulation is too coarse** (collapses states that should be distinct for planning) | This is consumer's responsibility — if the operator labels don't distinguish two states, they are planning-equivalent by definition. If the consumer needs finer granularity, they provide more operator labels (richer `OperatorLabel` enum). |
| **R5 — Duplication with Induced CWM (Plan 296)** | Different induction paths: CWM induces *code* (LLM refinement); this induces *operator schemas* (deterministic graph algorithm). Documented as complementary in rustdoc + cross-refs. Consumer picks per task. |
| **R6 — Premature riir-train deferral for skill-training half** | Per §3.5 modelless unblock protocol: the skill-training half genuinely requires gradient descent (diffusion policy fitting); the modelless half (this primitive) is what we ship here. The deferral is honest, not premature — checked against the three modelless paths: (1) freeze/thaw doesn't train diffusion policies; (2) deterministically-constructed LoRA can't approximate a diffusion denoiser; (3) latent-space projection can't replace a learned continuous-control policy. All three fail → riir-train deferral is correct. |

---

## References

- **Source paper:** [arXiv:2602.19260](https://arxiv.org/pdf/2602.19260) — Duggan et al., "The Price Is Not Right", Feb 2026.
- **Underlying NSM method:** [arXiv:2508.21501](https://arxiv.org/abs/2508.21501) — Lorang et al., Aug 2025.
- **Research note:** [katgpt-rs/.research/308_*.md](../.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md)
- **Closest cousin (the Super-GOAT this validates):** [katgpt-rs/.research/275_Code_World_Model_Induced_Forward_Model.md](../.research/275_Code_World_Model_Induced_Forward_Model.md), [katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md](296_induced_cwm_kernel_primitive.md)
- **Gap this closes:** [katgpt-rs/.research/264_Compositional_Open_Ended_Intelligence_Framework.md](../.research/264_Compositional_Open_Ended_Intelligence_Framework.md) §2.2 gap #1 (PTG) + #2 (motif mining)
- **Algorithmic prior art:**
  - Paige, Tarjan 1987 — "Three partition refinement algorithms" (the O((S+E) log S) bisimulation algorithm)
  - Kanellakis, Smolka 1990 — "CCS expressions, finite state processes, and three problems of equivalence"
  - Hopcroft 1971 — "An n log n algorithm for minimizing states in a finite automaton"
- **Options framework (skill decomposition with termination):** Sutton, Precup, Singh 1999 — maps to Plan 303 Salience Tri-Gate
- **ASP-based PDDL inference (the heavier-weight path we don't ship here):** Bonet, Geffner 2020; Rodriguez, Bonet, Romero, Geffner 2021
