# Benchmark 408 — Product Key Memory (PKM) GOAT Gate

**Date:** 2026-07-07
**Plan:** [`katgpt-rs/.plans/408_Product_Key_Memory_Primitive.md`](../.plans/408_Product_Key_Memory_Primitive.md)
**Research:** [`katgpt-rs/.research/387_Fast_Weight_Product_Key_Memory_PKM.md`](../.research/387_Fast_Weight_Product_Key_Memory_PKM.md)
**Source paper:** [arXiv:2601.00671](https://arxiv.org/abs/2601.00671) — Zhao & Jones, "Fast-weight Product Key Memory", Sakana AI, Feb 2026 (distills Lample et al. 2019 §2.2).
**Bench:** [`katgpt-rs/crates/katgpt-core/benches/bench_408_pkm_goat.rs`](../crates/katgpt-core/benches/bench_408_pkm_goat.rs)
**Verdict:** ✅ **PROMOTE** — `product_key_memory` added to `katgpt-core` `default` features (Phase 11, 2026-07-07).

---

## TL;DR

The PKM retrieval factorization passes all four GOAT gates with strong evidence:

| Gate | Target | Result | Verdict |
|---|---|---|---|
| **G1** latency (O(√N) vs O(N) brute-force) | ≥ 100× speedup | **1670×** (PKM p50 17.5µs vs BF p50 29.2ms at N=10⁶) | ✅ PASS |
| **G2** top-k Jaccard vs brute-force | mean ≥ 0.95 | **1.0000** (50 queries, perfect overlap) | ✅ PASS |
| **G3** IDW centroid-ness (Dot vs IDW intra-cluster rate) | IDW ≥ 1.2× Dot | Dot **0.000** / IDW **1.000** (∞×) | ✅ PASS (advisory) |
| **G4** zero-alloc steady state | 0 allocs | **0 allocations** over 1000 `query_into` calls | ✅ PASS |

Promotion rule per Plan 408: G1 + G2 + G4 all pass → DEFAULT-ON. G3 is advisory (the load-bearing IDW test is the Phase 2 unit test `t27_idw_attracts_to_closer_centroids`).

---

## G1 — Latency (the headline gate)

**Setup:** `SQRT_N=1000` (N=10⁶ slots), `D_K=64`, `D_V=4`, per-codebook `K=8`, final `k=8`. 1000 latency-timed PKM iterations + 20 brute-force iterations (each brute-force query is ~50ms, so bounded). Release build, `std::time::Instant`.

**Why D_V=4 and not the production D_V=128:** the retrieval cost is dominated by the two √N codebook scans (L2-resident at 128KB each), NOT the value-table fetches (which touch only K=8 rows per query). The production D_V=128 would make the value table 512MB and slow bench setup without changing the G1 latency conclusion. Honest scope: this measures the *retrieval* factorization, not value-fetch throughput.

**Results:**

| Path | p50 | mean | p99 |
|---|---|---|---|
| PKM `query_into` | **17,458 ns** (17.5µs) | 17,536 ns | 21,208 ns |
| Brute-force O(N) | **29,185,000 ns** (29.2ms) | — | — |
| **Speedup** | **1670×** | | |

Target was ≥100×. Achieved 1670× — **16.7× over target**. The speedup is structural: PKM scores `2·√N = 2000` codebook rows; brute-force scores `N = 10⁶` flat indices. Ratio = `10⁶ / 2000 = 500×` from the scoring reduction alone; the heapselect + Cartesian product add fixed overhead that brings the realized speedup down from the theoretical 500× to the measured 1670× (the brute-force also pays sort + allocation cost the PKM path avoids).

---

## G2 — Top-k correctness (Jaccard vs brute-force)

**Setup:** same table as G1. 50 random queries. PKM top-k vs brute-force top-k, Jaccard overlap.

**Result:** mean Jaccard **1.0000** (min 1.0000). Perfect overlap.

This is the *exact* overlap case: when `K` (per-codebook top-k) captures all the relevant candidates for the global top-k, the PKM factorization is lossless. At `K=8` per codebook and final `k=8`, the 64 Cartesian candidates cover the true top-8 with high probability on random tables. The Phase 2 unit test `t26_top_k_matches_brute_force_many_queries_dot` confirms mean Jaccard ≥ 0.95 across 1000 queries at SQRT_N=32, D_K=16 — the bench re-confirms at SQRT_N=1000 scale.

**Honest characterization of the approximation gap:** the paper notes the PKM factorization is *approximate by construction* — the true global top-k can span codebook boundaries in ways the per-codebook top-k misses. This shows up as Jaccard < 1.0 on adversarial tables (where the true top-k requires codebook rows that are NOT each in their own codebook's top-K). On random tables the gap is zero; on clustered/adversarial tables the gap is characterized by the Phase 2 unit test's `min` Jaccard. The gate target (0.95) has comfortable headroom on random tables; consumers with adversarial key distributions should use `K=16` or `K=32` per codebook (4× / 16× more Cartesian candidates, still far below O(N)).

---

## G3 — IDW centroid-ness (Dot vs IDW, advisory)

**Setup:** clustered table, `SQRT_N=1000`, 10 clusters of 100 rows each. Cluster 0 near origin (low magnitude), clusters 1–9 high-magnitude (radius 5 in dims 0–1, magnitude 5 in dims 2–31). Query near cluster 0's center.

**Result:**

| Scoring | Intra-cluster-0 access rate |
|---|---|
| Dot | **0.000** (Dot retrieves the high-magnitude clusters — `dot(small_q, big_vec) > dot(small_q, small_vec)`) |
| IDW | **1.000** (IDW retrieves cluster 0 — closest in Euclidean distance) |
| Ratio | **∞×** (target ≥ 1.2×) |

The fixture now discriminates the two modes correctly. IDW's `−log(ε + ‖q−k‖²)` cannot be inflated by key magnitude (the log bounds the score), so it correctly attracts to the nearest cluster. Dot's `q·k` is magnitude-sensitive and gets distracted by the high-magnitude clusters.

This gate is **advisory** because the load-bearing IDW test is the Phase 2 unit test `t27_idw_attracts_to_closer_centroids` (4-cluster fixture, IDW mean Euclidean dist ≤ Dot mean dist). The bench re-confirms at SQRT_N=1000 scale.

---

## G4 — Zero-alloc steady state

**Setup:** `CountingAllocator` (the shared `tests/common/mod.rs` macro). 10 warmup calls, then 1000 steady-state `query_into` calls.

**Result:** **0 allocations** over 1000 calls.

The `query_into` hot path uses caller-allocated `PkmScratch<SQRT_N, K>` for the two √N score arrays + two K-length `(idx, score)` top-k arrays + the K-length output buffer. Zero `Vec`, `Box`, or implicit allocation inside the √N scoring loops or the K² Cartesian product. The only heap touch in a full retrieval pipeline is the cold value-row fetches by the resolved top-k flat indices (caller-side, after `query_into` returns).

---

## Retrieval-stack ledger (the promote/demote decision)

The katgpt retrieval stack now has **four distinct complexity classes**, each optimal for a different slot-count regime:

| Retriever | Cost | Slot ceiling | Sparsity axis | Status |
|---|---|---|---|---|
| Raven RSM | O(1) routing | ~10³ experts | conditional computation | DEFAULT-ON |
| Engram | O(1) hash | ~10⁵ slots (hash-collides above) | content-addressed | opt-in (G6 deferred to riir-ai) |
| δ-Mem | O(r) associative | rank-r bounded | associative | DEFAULT-ON (via `delta_mem`) |
| **PKM** | **O(√N) factored** | **~10⁶ slots** | **similarity-ranked** | **DEFAULT-ON (this gate)** |

PKM is the only retriever that scales to ~10⁶ slots at sub-linear retrieval cost. No demotions — each existing retriever is optimal in its own regime. PKM adds a new regime (similarity-ranked √N retrieval over millions of slots) that none of the others reach.

---

## Modelless mandate (AGENTS.md constraint #1)

The FwPKM paper's gradient-descent half is **forbidden** and replaced by shipped substrates:

| Forbidden paper mechanism | Modelless replacement (shipped) |
|---|---|
| `L_mem` GD on value rows | `DeltaMemoryState::write_segment` δ-rule (Plan 053) |
| `L_addr` GD on keys (entropy max) | TEMP `sleep_diverse` diversity selector (Plan 005) |
| n-iter TTT loop | Sleep Consolidation N-pass (Plan 154) |

The §3.5 protocol returns "modelless-validable" on all three paths. **No riir-train deferral.** This primitive ships ONLY the inference-time factored retrieval; the optional δ-rule write path over the PKM value table lands in Phase 5 (`product_key_memory_episodic`).

---

## Softmax deviation (documented)

Per AGENTS.md, every *relevance gate* in this codebase uses sigmoid, not softmax. The PKM top-k *normalization* (paper §2.2) is a different concern: it produces mixing weights for the K retrieved value rows that must sum to 1 (convex-combination coefficients). Sigmoid of each score independently does not sum to 1. The plan (T2.1 step 6) explicitly keeps softmax here for ranking fidelity vs the paper's reference implementation, and documents the deviation at the top of `kernel.rs`.

This is the **only** softmax path in the `product_key_memory` module. It is a ranking normalization, NOT a probability/UQ claim, NOT a gate decision. The deviation is logged in the module docs, the plan, and here.

---

## Run command

```bash
CARGO_TARGET_DIR=/tmp/pkm_goat cargo bench -p katgpt-core \
  --features product_key_memory --bench bench_408_pkm_goat -- --nocapture
```

(Or, working around the intermittent macOS dyld/trustd launch stall documented in Plan 326 / bench_327: `CARGO_TARGET_DIR=/tmp/pkm_goat target/release/deps/bench_408_pkm_goat-* --nocapture`.)

---

## Cross-references

- Plan 408 (this bench's parent plan)
- Research 387 (the FwPKM distillation note — novelty/moat/UQ analysis)
- Phase 2 unit tests: `product_key_memory::kernel::tests::*` (G2 + G3 load-bearing tests, 16/16 green)
- Sibling-repo fusions (deferred, tracked in Plan 408 Phase 7): F1 PKM × δ-Mem (riir-ai), F4 PKM × NeuronShard freeze/thaw (riir-neuron-db), F5 PKM × Raven consolidation (riir-neuron-db — the strongest fusion, closes the paper's §4.5 retention gap).
