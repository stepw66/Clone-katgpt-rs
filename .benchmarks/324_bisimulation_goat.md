# Plan 324 — Bisimulation Operator Inference GOAT Gate Results

**Date:** 2026-06-25
**Plan:** [324_bisimulation_operator_inference.md](../.plans/324_bisimulation_operator_inference.md)
**Research:** [308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md](../.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md)
**Source paper:** [arxiv 2602.19260](https://arxiv.org/pdf/2602.19260) — Duggan, Lorang, Lu, Scheutz (Tufts), Feb 2026 ("The Price Is Not Right")
**Underlying NSM method:** [arxiv 2508.21501](https://arxiv.org/abs/2508.21501) — Lorang et al., Aug 2025
**Feature:** `bisimulation_operator_inference` (opt-in)

---

## TL;DR (read this first)

| Gate | Target | Result | Decision |
|------|--------|--------|----------|
| **G1** — Bisimulation correctness | Known graph → known minimal quotient; bit-identical across re-runs; idempotent | ✅ **PASS** (11 `refine` tests + 8 `types` tests) | `partition_refine` + `canonicalize_labels` + `blake3_commit` ship. |
| **G2** — Operator inference soundness | Every observed transition covered; no spurious operators | ✅ **PASS** (6 `operator` tests incl. `schema_covers_all_edges_g2`, `admits_checks_preconditions`, `hanoi_3disk_smoke`) | `infer_operators` ships. |
| **G3** — Plan validity | Planner on inferred schema produces executable plans; unreachable pairs return `None` | ✅ **PASS** (5 `planner` tests incl. `replay_detects_precondition_violation`, `unreachable_goal_returns_none`) | BFS `plan` ships. |
| **G4** — Latency | `partition_refine` ≤ 1 ms for N=1024 on Apple Silicon arm64 release | ✅ **PASS** — **715.7 µs** @ N=1024 (28% headroom) | No optimization needed. |
| **G5** — Zero-alloc hot path | `class_of(state) -> StateClassId` is O(1), ≥ 100M lookups/sec | ✅ **PASS** — **1635.99M lookups/sec** @ N=1024 (16× margin) | `class_of` ships as direct-index O(1). |

**All 5 gates PASS.** Per Plan 324 §Promotion rule: the open primitive is **ready for downstream consumption**. `bisimulation_operator_inference` stays **opt-in by design** — it is a primitive, not a default-on capability (same policy as Induced CWM, Plan 296). Downstream pipelines (riir-ai NPC runtime, riir-chain LatCal consumer) opt in by enabling the feature.

**Shippable output of Plan 324:**

1. `StateId` / `StateClassId` newtypes (`#[repr(transparent)]` u32) — type-safe raw tags (Phase 1).
2. `OperatorLabel` (`#[repr(u8)]`, 2 bytes with `Other(u8)` escape) — abstract operator tag (Phase 1).
3. `Transition` / `QuotientEdge` (`#[repr(C)]`, 12 bytes) — sorted-edge records (Phase 1).
4. `TransitionGraph` + `TransitionGraphBuilder` — sorted, deduped, O(log) adjacency-indexed observed-transition store (Phase 1).
5. `BisimulationQuotient` + `partition_refine` — Paige-Tarjan-style signature-based partition refinement, O((S+E) log S); canonical BLAKE3 commitment (Phase 2).
6. `OperatorSchema` + `OperatorDef` + `infer_operators` — one operator per quotient edge label, preconditions + effects, BLAKE3 commitment (Phase 3).
7. `plan` (BFS over quotient) — minimum-length operator-label sequence start→goal (Phase 4).
8. `bench_324_bisimulation_goat` — G4 + G5 latency/throughput gate (Phase 4).

---

## G1 — Bisimulation correctness

**Contract (Plan 324 §G1):** Given a known graph with a known minimal bisimulation, the partition-refinement algorithm produces the canonical quotient. Re-running on the same graph yields bit-identical class assignments. Idempotent: `partition_refine(partition_refine(g)).blake3 == partition_refine(g).blake3`.

**Fixture:** hand-crafted small graphs with hand-computed canonical quotients, plus a label-oscillation counterexample that distinguishes naive single-pass signature refinement from the iterative fixpoint algorithm.

| Test | Verifies | Result |
|------|----------|--------|
| `empty_graph_yields_empty_quotient` | Empty graph → 0 classes, empty edges. | ✅ PASS |
| `chain_a_to_b_to_c_yields_three_classes` | `A -op1-> B -op2-> C` → 3 classes (each state has a distinct signature). | ✅ PASS |
| `parallel_chains_collapse_into_two_classes` | `A -op1-> B`, `C -op1-> D` with `B,D` sinks → 2 classes (`{A,C}`, `{B,D}`). | ✅ PASS |
| `rerun_is_bit_identical` | Two calls on the same graph produce identical `state_to_class` + `blake3`. | ✅ PASS |
| `identical_inputs_produce_identical_blake3` | Isomorphic inputs → identical BLAKE3 (deterministic canonicalization). | ✅ PASS |
| `quotient_edges_are_sorted_and_deduped` | Output edges obey `(from, op.discriminant(), to)` canonical order; no duplicates. | ✅ PASS |
| `label_oscillation_counterexample_converges` | A graph requiring ≥2 refinement passes (signature depends on a successor's signature) converges to the fixpoint. | ✅ PASS |
| `class_of_is_o1_for_in_range` | `class_of(state)` returns the correct `StateClassId` via direct index. | ✅ PASS |
| `canonicalize_labels_renumbers_by_smallest_member` | Class 0 = the block containing `StateId(0)`; class ids are dense `0..n`. | ✅ PASS |
| `canonicalize_labels_is_idempotent` | Re-canonicalizing a canonical quotient is a no-op. | ✅ PASS |
| `types::*` (8 tests) | `StateId`/`StateClassId` are `#[repr(transparent)]` u32; `OperatorLabel` is 2 bytes; `Transition` is 12 bytes; lex order is total. | ✅ PASS |

**Verdict: G1 PASS.** The signature-based partition-refinement algorithm produces correct canonical quotients on known-answer fixtures, is deterministic across re-runs, and is idempotent. The label-oscillation counterexample specifically guards against the naive single-pass failure mode (where a state's signature depends on a successor that hasn't been refined yet).

---

## G2 — Operator inference soundness

**Contract (Plan 324 §G2):** Given a quotient graph with labeled edges, the inferred operator schema covers every observed transition (no missing operators) and admits no spurious operators (every operator is exercised by ≥1 edge). Feeding the source trajectory set back through the schema, every transition's `(state, op)` is admitted.

| Test | Verifies | Result |
|------|----------|--------|
| `empty_quotient_yields_empty_schema` | Empty quotient → empty operator list, zero BLAKE3. | ✅ PASS |
| `single_edge_yields_one_operator` | One quotient edge → one `OperatorDef` with one precondition + one effect. | ✅ PASS |
| `operators_sorted_by_label_discriminant` | Operators emitted in `discriminant()` order (deterministic). | ✅ PASS |
| `schema_covers_all_edges_g2` | **G2 gate.** Every quotient edge is covered by exactly one `(label, from, to)` tuple in `effects`. | ✅ PASS |
| `admits_checks_preconditions` | `admits(state, op)` returns true iff `state`'s class ∈ operator's preconditions. | ✅ PASS |
| `rerun_is_bit_identical` | Same quotient → identical schema BLAKE3. | ✅ PASS |
| `hanoi_3disk_smoke` | Hand-crafted Towers-of-Hanoi-style graph quotients to a small class set; schema admits all source transitions. | ✅ PASS |

**Verdict: G2 PASS.** Operator inference is a deterministic group-by-label aggregation over quotient edges — by construction every operator is exercised and every edge is covered. The Hanoi smoke test confirms the pipeline works end-to-end on a structured domain.

---

## G3 — Plan validity

**Contract (Plan 324 §G3):** For each `(start, goal)` pair where a path exists, `plan()` returns `Some(sequence)`. Replaying the sequence against the original `TransitionGraph` never violates operator preconditions and reaches a state in the goal class. For unreachable pairs, `plan()` returns `None`.

| Test | Verifies | Result |
|------|----------|--------|
| `empty_plan_when_start_equals_goal` | `start == goal` → `Some(vec![])`. | ✅ PASS |
| `single_step_plan` | Adjacent classes → 1-operator plan. | ✅ PASS |
| `multi_step_plan` | Multi-hop path → minimum-length operator sequence. | ✅ PASS |
| `unreachable_goal_returns_none` | No path in quotient → `None`. | ✅ PASS |
| `replay_detects_precondition_violation` | **G3 gate.** Replaying an invalid operator sequence against the original graph is detected as a precondition violation. | ✅ PASS |

**Verdict: G3 PASS.** BFS over the quotient graph finds minimum-length plans; replay validation catches precondition violations. MetricFF-grade classical planning is explicitly out of scope (Plan 324 §Out of Scope) — downstream consumers needing richer planning wire up `crate::induced_cwm::ismcts` or an external PDDL solver.

---

## G4 — Latency

**Contract (Plan 324 §G4):** `partition_refine` on a graph of N nodes completes in O((S+E) log S) worst case; target ≤ 1 ms for N=1024 on plasma-tier CPU SIMD (Apple Silicon arm64 release build).

**Fixture:** deterministic LCG-seeded (seed=42) random-transition graphs with average out-degree 3, swept over N ∈ {64, 256, 1024, 4096}. Median of 20 timed runs after 10 warmup iterations.

**Bench:** `cargo bench -p katgpt-core --bench bench_324_bisimulation_goat --features bisimulation_operator_inference`

| N_states | refine_time | infer_time | n_classes | n_operators |
|----------|-------------|------------|-----------|-------------|
| 64       | 48.5 µs     | 21.8 µs    | 64        | 3           |
| 256      | 231.6 µs    | 55.2 µs    | 255       | 3           |
| **1024** | **715.7 µs** | 229.4 µs  | 1020      | 3           |
| 4096     | 3.87 ms     | 1.01 ms    | 4075      | 3           |

**Verdict: G4 PASS.** `partition_refine` @ N=1024 = **715.7 µs**, comfortably under the 1 ms target (28% headroom). Scaling is near-linear in N for sparse graphs, consistent with the O((S+E) log S) bound.

**Note on compaction ratio:** random graphs show near-minimal compaction (1020 classes from 1024 states) because they have little exploitable structure — this is correct behavior, not a bug (Plan 324 R3: "bisimulation acts on whatever graph it's given"). Real structured domains (Towers of Hanoi, game state spaces) exhibit heavy compaction; the `hanoi_3disk_smoke` unit test confirms this on a structured fixture.

---

## G5 — Zero-alloc hot path

**Contract (Plan 324 §G5):** `BisimulationQuotient::class_of(state) -> StateClassId` is O(1) direct index, no heap allocation across 10⁶ queries. Target ≥ 100M lookups/sec.

**Fixture:** the N=1024 quotient from the G4 bench. 100,000 lookups per batch × 50 batches, median batch throughput.

| Metric | Value |
|--------|-------|
| `class_of` throughput @ N=1024 | **1635.99M lookups/sec** |
| Target | ≥ 100M lookups/sec |
| Margin | **16× over target** |

**Verdict: G5 PASS.** `class_of` is a single `self.state_to_class[state.0 as usize]` direct index — structurally incapable of allocating. The 16× margin over target confirms the plasma-tier (sub-µs CPU) design.

---

## Reproducibility

```bash
# Compile check (default features off, this feature on)
cargo check -p katgpt-core --features bisimulation_operator_inference

# Full test suite for the feature
cargo test -p katgpt-core --features bisimulation_operator_inference --lib bisimulation

# G4 + G5 latency/throughput gate
cargo bench -p katgpt-core --bench bench_324_bisimulation_goat --features bisimulation_operator_inference
```

Environment: macOS arm64 (Apple Silicon), release profile (`cargo bench` default).

---

## Promotion Decision

**All 5 gates PASS.** Per Plan 324 §Phase 5 T5.4:
- The feature stays **opt-in by design** (`bisimulation_operator_inference = []`). It is a primitive, not a default-on capability — same policy as Induced CWM (Plan 296). Downstream pipelines opt in by enabling the feature.
- No `.issues/` follow-up required.

---

## Deviations from Plan 324

None. All six phases (Types Skeleton, Partition Refinement, Operator Schema, Plan Validity + Latency, Zero-Alloc Hot Path, Docs/Cross-References) ship as specified.

---

## Cross-references

- **Plan:** [324_bisimulation_operator_inference.md](../.plans/324_bisimulation_operator_inference.md)
- **Research:** [308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md](../.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md)
- **Closest cousin (the Super-GOAT this primitive complements):** [296_induced_cwm_kernel_primitive.md](../.plans/296_induced_cwm_kernel_primitive.md) + [296_induced_cwm_primitive_goat.md](296_induced_cwm_primitive_goat.md)
- **Gap this closes:** [264_Compositional_Open_Ended_Intelligence_Framework.md](../.research/264_Compositional_Open_Ended_Intelligence_Framework.md) §2.2 gap #1 (PTG data structure) + #2 (motif mining)
- **Source paper:** [arXiv:2602.19260](https://arxiv.org/pdf/2602.19260) — Duggan et al., "The Price Is Not Right", Feb 2026
- **Underlying NSM method:** [arXiv:2508.21501](https://arxiv.org/abs/2508.21501) — Lorang et al., Aug 2025
