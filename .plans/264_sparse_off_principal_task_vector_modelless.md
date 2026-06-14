# Plan 264: Sparse Off-Principal Task Vector (SOPTV) — Modelless Implementation

**Research:** [231_Sparse_Off_Principal_Task_Vector_OPD.md](../.research/231_Sparse_Off_Principal_Task_Vector_OPD.md)
**Paper:** arXiv 2606.13657 — Dense Supervision, Sparse Updates (OPD parameter geometry)
**Date:** 2026-06-14
**Status:** 🟢 Phases 1–6 complete (all G1–G10 GOAT tests pass; features promoted to default-ON). Phase 7 (docs) pending.
**Feature Gates:** `sparse_task_vector` (Fusion A), `off_principal_retrieval` (Fusion B), `spectral_rank` (Fusion C), `module_energy_route` (Fusion D). All opt-in until GOAT-proven.
**Constraints:**
- Modelless only — no LLM training.
- Sigmoid not softmax.
- Plasma/Hot/Warm/Cold/Freeze tier aware.
- CPU/SIMD/GPU/ANE auto-route via threshold.
- SOLID, DRY, files <2048 lines.
- Tests/examples with before/after expected gains.

---

## Task

### Phase 1 — SparseTaskVector storage (Fusion A, foundation) ✅ DONE

- [x] T1.1 Create `src/sparse_task_vector.rs` module skeleton + `SparseTaskVector` struct.
- [x] T1.2 Implement `SparseTaskVector::from_dense(weight, threshold)` — extract sparse mask from dense delta.
- [x] T1.3 Implement `SparseTaskVector::apply_to(&mut base)` — scatter-add mask into base weight buffer.
- [x] T1.4 Implement `SparseTaskVector::apply_to_scratch(&base, scratch)` — zero-alloc variant for hot path.
- [x] T1.5 Implement `SparseTaskVector::density()` and `relative_norm_vs(&base)` — paper §4.1 metrics.
- [x] T1.6 Add feature gate `sparse_task_vector` in `Cargo.toml` (opt-in).
- [x] T1.7 Wire module into `src/lib.rs` under feature gate.
- [x] T1.8 GOAT test G1: 2.9–5.7× storage reduction vs dense LoRA at paper densities (17.5%, 10.5%).
- [x] T1.9 GOAT test G2: apply roundtrip (`from_dense` → `apply_to`) recovers base+delta within 1e-4 rel.
- [x] T1.10 Example: doc-test in module showing before/after memory footprint (12 unit tests + 1 doc-test pass).

**Phase 1 unblocks Plan 296 Phase 0** — riir-ai can now consume `SparseTaskVector` format.

### Phase 2 — Off-Principal Retrieval (Fusion B)

- [x] T2.1 Implement `off_principal_project(q, u_k, k, scratch)` — `q_off = q − U_k(U_k^T q)`.
- [x] T2.2 Implement `OffPrincipalIndex::new(base_weight, k_frac)` — computes SVD via existing `newton_schulz` and caches `U_k`.
- [x] T2.3 Implement `OffPrincipalIndex::score(query_emb, adapter_emb) -> f32` — dot product in off-principal space.
- [x] T2.4 Sigmoid-bounded score variant `score_bounded(...) -> f32` ∈ [0,1] (no softmax).
- [x] T2.5 Add feature gate `off_principal_retrieval` (depends on `newton_schulz`).
- [x] T2.6 GOAT test G3: off-principal projection removes ≥99% of `W_src` principal component energy from query (paper finding 2).
- [x] T2.7 GOAT test G4: synthetic 8-adapter retrieval, off-principal top-1 accuracy > raw cosine top-1 accuracy.
- [x] T2.8 Example: `examples/off_principal_retrieval.rs` showing before/after retrieval accuracy.

### Phase 3 — Spectral-Concentration Adaptive Rank (Fusion C)

- [x] T3.1 Implement `spectral_concentration(eigenvalues, k) -> f32` — top-k energy ratio (paper eq. 18).
- [x] T3.2 Implement `adaptive_rank(concentration, min_rank, max_rank) -> usize` — sigmoid mapping.
- [x] T3.3 Implement `cot_budget_from_concentration(c, base, max_extra) -> usize` — adaptive CoT linkage.
- [x] T3.4 Add feature gate `spectral_rank`.
- [x] T3.5 GOAT test G5: rank-16 captures 18–33% energy on synthetic OPD-shaped spectrum (paper finding 3).
- [x] T3.6 GOAT test G6: adaptive rank reduces avg rank ≥30% vs fixed max-rank on synthetic 100-query workload.
- [x] T3.7 Example: `examples/adaptive_rank_vs_fixed.rs` showing before/after LoRA compute.

### Phase 4 — Module-Energy Compute Routing (Fusion D)

- [x] T4.1 Add `ComputeTarget` enum (Plasma, Simd, Gpu, Ane) to `src/inference_router.rs`.
- [x] T4.2 Implement `route_by_module_energy(ffn_frac, attn_frac, qps) -> ComputeTarget` — threshold-based.
- [x] T4.3 Extend `TriggerGate` to accept `ModuleEnergyProfile { ffn, attn, embed, other }`. *(Note: `ModuleEnergyProfile` added as standalone struct in `inference_router.rs` behind `#[cfg(feature = "module_energy_route")]` — `trigger_gate.rs` is outside the disjoint write scope, so the profile is composed at call sites rather than stored as a `TriggerGate` field.)*
- [x] T4.4 Add plasma/hot/warm/cold/freeze tier mapping docstring (constraint 8).
- [x] T4.5 Add feature gate `module_energy_route`.
- [x] T4.6 GOAT test G7: route decision matches paper finding 4 (FFN >70% & qps<1000 → Plasma).
- [x] T4.7 GOAT test G8: route transitions are monotone in QPS (no flapping).
- [x] T4.8 Example: `examples/module_aware_routing.rs` showing before/after routing decisions.

### Phase 5 — Adapter Composition via Mask Intersection (paper finding 5)

- [x] T5.1 Implement `compose_intersect(a, b) -> SparseTaskVector` — mask intersection + eta superposition.
- [x] T5.2 Implement `compose_union(a, b) -> SparseTaskVector` — mask union for additive composition.
- [x] T5.3 GOAT test G9: composition is associative (paper §4.3 overlap is symmetric).
- [x] T5.4 GOAT test G10: intersected composition preserves ≥2.21× the random baseline overlap (paper finding 5 floor).

### Phase 6 — GOAT Gate & Promotion

- [x] T6.1 Run full benchmark suite with all four features on. (`cargo test --lib --features sparse_task_vector,off_principal_retrieval,spectral_rank,module_energy_route` → 3430 passed, 4 pre-existing failures unrelated.)
- [x] T6.2 Confirm G1–G10 all pass. (G1–G2 sparse_task_vector 12/12, G3–G4 off_principal 8/8, G5–G6 spectral_concentration 18/18, G7–G8 module_energy_route 11/11, G9–G10 sparse_compose 17/17 = 66 tests, 0 fail.)
- [x] T6.3 If all pass → promote features to `default` (paper-grounded GOAT). (All 4 features added to `default` in Cargo.toml; `cargo build` clean; `cargo test --lib` → 3430 passed, 4 pre-existing failures, zero regression.)
- [x] T6.4 Update README with showcase entry under "GOAT-Proved Additions". (Added 4 rows to the GOAT-Proved Additions table in README.md, updated section header to Plans 225–264.)
- [x] T6.5 Demote any prior dense-only LoRA storage path if SparseTaskVector strictly wins (per user rules). **N/A** — SparseTaskVector complements dense storage (better for density <50%); dense is still superior for high-density deltas. No strict winner across all cases, so no demotion.

### Phase 7 — Documentation

- [ ] T7.1 Add doc comments linking each function to the specific paper section it implements.
- [ ] T7.2 Cross-link Research 231 from README research index.
- [ ] T7.3 Cross-link Plan 264 from README plans index.

---

## GOAT Gate Definitions

| Gate | Metric | Pass Threshold | Paper Source |
|------|--------|----------------|--------------|
| G1 | Storage reduction at 17.5% density | ≥2.5× | §4.1 sparsity table |
| G2 | Apply roundtrip relative error | < 1e-4 (f32 vectorization noise) | correctness |
| G3 | Principal energy removed | ≥99% | §5.2 principal projection ≤1% |
| G4 | Retrieval top-1 accuracy gain | ≥+5pp vs cosine | §5.2 off-principal |
| G5 | Rank-16 energy capture | 18–33% | §5.1 Table 9 |
| G6 | Avg rank reduction | ≥30% | §5.1 stable rank 7–20 |
| G7 | Route matches paper FFN profile | exact | §4.2 Table 3 |
| G8 | Route monotone in QPS | no flapping | constraint 9 |
| G9 | Composition associativity | exact | §4.3 mask symmetry |
| G10 | Composition overlap | ≥2.21× random | §4.3 Table 4 |

---

## File Layout

```
src/
├── sparse_task_vector.rs       # Fusion A: SparseTaskVector struct + ops (Phase 1)
├── off_principal.rs            # Fusion B: OffPrincipalIndex + project (Phase 2)
├── spectral_concentration.rs   # Fusion C: concentration + adaptive_rank (Phase 3)
├── inference_router.rs         # Fusion D: route_by_module_energy (Phase 4, extend existing)
└── sparse_compose.rs           # Phase 5: compose_intersect / compose_union

examples/
├── sparse_vs_dense_storage.rs
├── off_principal_retrieval.rs
├── adaptive_rank_vs_fixed.rs
└── module_aware_routing.rs
```

All files kept <512 lines (well under 2048 limit).

---

## Dependencies

- Existing: `newton_schulz` (R152), `spectral_budget`, `trigger_gate`, `inference_router`, `types::LoraAdapter`, `pruners::freeze`.
- New: none (pure Rust, no new crates).

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Off-principal SVD too slow at load | Newton-Schulz is 5 iterations, O(d²k); cache result |
| Sparse apply slower than dense GEMM for low sparsity | Density threshold check: if density > 50%, fall back to dense |
| Module-energy profile not measured per-game | Default to paper average (FFN=0.78); config override later |
| Composition intersection drops too much signal | Provide `compose_union` as alternative |

---

## TL;DR

Implement the four modelless fusions from Research 231:
1. SparseTaskVector storage (Phase 1).
2. Off-principal retrieval (Phase 2).
3. Spectral adaptive rank (Phase 3).
4. Module-aware routing (Phase 4).

Plus adapter composition (Phase 5). Gate everything behind feature flags until GOAT proofs G1–G10 pass. Promote to default if GOAT, demote prior dense-only paths if strictly worse.
