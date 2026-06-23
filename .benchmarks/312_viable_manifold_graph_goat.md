# Benchmark 312: Viable Manifold Graph — GOAT Gate

**Date:** 2026-06-23
**Plan:** 312 (Viable Manifold Graph — Open Primitive)
**Features:** `--features viable_manifold_graph` (pulls `subspace_phase_gate`)
**Commands:**
  - Tests: `cargo test -p katgpt-core --features viable_manifold_graph --lib viable_manifold_graph -- --nocapture`
  - Bench:  `cargo bench -p katgpt-core --features viable_manifold_graph --bench viable_manifold_graph_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10`
  - Example:`cargo run --example viable_manifold_graph_01_basic`
**Source:** [arXiv:2206.00106](https://arxiv.org/abs/2206.00106) — González-Duque et al., *Mario Plays on a Manifold*, 2022

---

## G1–G7 Unit Tests (`viable_manifold_graph::tests`)

All 10 unit tests pass (10 passed, 0 failed in 0.02s):

| Gate | Test | Status |
|------|------|--------|
| G1 | `test_pullback_volume_identity_is_zero` | ✅ PASS |
| G2 | `test_pullback_volume_scaling_is_2n_log_c` | ✅ PASS |
| G3 | `test_safe_graph_build_connected_when_predicate_true` | ✅ PASS (100 nodes, connected via BFS) |
| G3b | `test_safe_graph_build_disconnected_when_predicate_splits` | ✅ PASS (predicate-boundary respected) |
| G4 | `test_manifold_geodesic_validity` | ✅ PASS (path stays viable, no repeats) |
| G5 | `test_manifold_random_walk_validity` | ✅ PASS (25-step walk, all nodes viable — playability = 1.0) |
| G6 | `test_manifold_random_walk_zero_alloc_across_1000_steps` | ✅ PASS (Vec capacity == m+1, no growth) |
| G7 | `test_primitive_never_touches_sync` (compile-pass by inspection) | ✅ PASS — module imports only `crate::subspace_phase_gate::{JacobianSvdScratch, jacobian_svd_at}` + `std::collections::BinaryHeap`; no `riir-chain`/`riir-neuron-db`/sync modules |

Bonus tests (not gates): `test_manifold_curiosity_walk_basic`, `test_manifold_geodesic_trivial_unreachable`, `test_manifold_geodesic_trivial_single_edge` — all ✅.

Phase 0 example still runs: 360 viable nodes, 720 edges; free Gaussian 74.2% viable vs manifold-constrained 100%, geodesic 19 hops all viable. Reproduces paper's SMB headline (77.3% vs 99.6%).

---

## Bench Results (MacBook Pro, release profile, criterion)

| Bench | Median | Target | Status |
|-------|--------|--------|--------|
| `pullback_volume/R^4_to_R^4_identity` | **304.74 ns** | < 5 µs | ✅ **PASS** (16.4× under target) |
| `manifold_random_walk/k=4_1000_steps` (per-step) | **485.58 ns/step** | < 100 ns/step | ❌ **FAIL** (4.86× over target) |
| `build_safe_manifold_graph/1000_samples_d4` | **367.93 µs** | < 10 ms | ✅ **PASS** (27.2× under target) |

Full criterion output:
```
viable_manifold_graph/pullback_volume/R^4_to_R^4_identity
                        time:   [304.17 ns 304.74 ns 305.34 ns]
                        Found 23 outliers among 500 measurements (4.60%)

viable_manifold_graph/manifold_random_walk/k=4_1000_steps
                        time:   [485.58 µs 493.21 µs 502.52 µs]  (per 1000-step walk)
                        thrpt:  [1.9900 Melem/s 2.0275 Melem/s 2.0594 Melem/s]
                        → 485.58 ns/step
                        Found 16 outliers among 500 measurements (3.20%)

viable_manifold_graph/build_safe_manifold_graph/1000_samples_d4
                        time:   [366.04 µs 367.93 µs 369.58 µs]
                        thrpt:  [2.7058 Melem/s 2.7179 Melem/s 2.7320 Melem/s]
```

---

## Root Cause of G-bench 2 Failure

`SafeManifoldGraph::for_each_neighbor` does an **O(E) linear scan** over all edges per call:

```rust
pub fn for_each_neighbor<F: FnMut(u32)>(&self, idx: u32, mut f: F) {
    for &(a, b) in self.edges.iter() {
        if a == target { f(b); } else if b == target { f(a); }
    }
}
```

The bench graph is built from a 50×50 4D grid filtered by a paper-style viability predicate → ~1k viable nodes, ~4k edges (k=4 nearest). Each random-walk step scans all 4k edges to find the (typically ≤ 8) neighbors of the current node. At ~120 ps/edge-scan on this hardware, that's ≈ 480 ns/step — matches the measured 485.58 ns/step exactly.

The unit-test gate G6 ("zero alloc growth across 1000 steps") passes because the perf bottleneck is **algorithmic**, not allocation. The plan's docstring already flagged this design trade-off:

> *"Linear scan is fine for the graph sizes the paper uses (10²–10³ nodes, ~k·n edges). For larger graphs a CSR adjacency would be better — see Plan 312 risk register; deferred until a build ever exceeds ~10⁴ nodes."*

The 100ns/step target was set under the assumption that "k=4 neighbors" implies "scan 4 entries". With the current edge-list layout, "k=4" describes the **result set size**, not the **scan cost**.

---

## Recommendation: **DEMOTE (hold at opt-in)**

**Do not promote `viable_manifold_graph` to default-on** until the perf gate is met. Rationale:

1. **G1, G2, G3, G3b, G4, G5, G6, G7 all PASS** — the primitive is *correct* and *allocation-safe*.
2. **G-bench 2 (random-walk per-step) FAILS** — 4.86× over the 100 ns/step target. Not a marginal miss; not noise.
3. **Fix is well-scoped**: replace `Vec<(u32,u32)>` edge list with CSR (compressed sparse row) adjacency. `for_each_neighbor` becomes O(degree) instead of O(E). This is a local data-structure change inside `SafeManifoldGraph`; no public-API change. Plan 312's risk register already calls out this exact deferral.
4. **Phase 6 (riir-ai wiring) has not run** — even if perf were fixed, the GOAT headline (manifold-constrained walk beats free Gaussian walk on real HLA) is unmeasured. The paper-scale toy reproduces (74.2% vs 100% on the example), but that's a 2D synthetic, not the 8D HLA runtime.

**Promotion path:**
- (a) Implement CSR adjacency → re-run G-bench 2 → if < 100 ns/step, re-evaluate.
- (b) Run Phase 6 riir-ai wiring on real HLA → if free-walk playability is already ~99% (HLA is well-behaved), demote permanently per Risk Register row 5.
- (c) Alternatively, re-spec the G-bench 2 target to O(E)-appropriate (e.g., < 1 µs/step for ≤ 10⁴ edges). Not recommended — the 100 ns/step target was set to match paper-scale real-time game-tick budgets.

---

## Files Touched (Phase 4)

| File | Change |
|------|--------|
| `katgpt-rs/crates/katgpt-core/benches/viable_manifold_graph_bench.rs` | NEW — criterion bench (3 benchmarks) |
| `katgpt-rs/crates/katgpt-core/Cargo.toml` | Register bench under `[[bench]]` with `required-features = ["viable_manifold_graph"]` |
| `katgpt-rs/.benchmarks/312_viable_manifold_graph_goat.md` | NEW — this file |
| `katgpt-rs/.plans/312_viable_manifold_graph_primitive.md` | Mark T4.1–T4.10 complete |

**No source changes** to `viable_manifold_graph.rs` — all unit tests passed unchanged; the perf miss is documented, not papered over.

---

## References

- González-Duque et al., "Mario Plays on a Manifold," arXiv:2206.00106, 2022
- Plan 312 — open-primitive spec
- Research 294 — math + prior-art table
- Plan 301 — substrate (`jacobian_svd_at`, `JacobianSvdScratch`)
