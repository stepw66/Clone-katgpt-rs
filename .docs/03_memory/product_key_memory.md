# Product Key Memory (PKM): O(√N) Factored Retrieval Memory

> **Status: DEFAULT-ON** in `katgpt-core` (2026-07-07, Plan 408 Phase 3 GOAT gate
> G1+G2+G4 ALL PASS). Zero runtime cost unless a caller constructs
> `ProductKeyMemory`.

## What It Is

Const-generic `ProductKeyMemory<SQRT_N, D_K, D_V>` — a fixed-size key-value
table that retrieves the top-k value rows for a query in `O(√N)` instead of
`O(N)`. Splits the `D_K`-dim query into two halves, scores two √N-row codebooks,
and takes the top-k of the `k×k` Cartesian product. This is the **retrieval
factorization** half of the FwPKM paper (Lample et al. 2019 §2.2; Zhao & Jones
2026 distillation); the gradient-descent half is forbidden by the modelless
mandate and replaced by the shipped δ-rule (Plan 053).

## Why It Exists — The Complexity-Class Gap

The katgpt retrieval stack now has four distinct complexity classes, each
optimal for a different slot-count regime:

| Retriever | Cost | Slot ceiling | Sparsity axis | Feature |
|---|---|---|---|---|
| Raven RSM | O(1) routing | ~10³ experts | conditional computation | always compiled |
| Engram | O(1) hash | ~10⁵ slots (hash-collides above) | content-addressed | `engram` (opt-in) |
| δ-Mem | O(r) associative | rank-r bounded | associative | `delta_mem` |
| **PKM (this)** | **O(√N) factored** | **~10⁶ slots** | **similarity-ranked** | `product_key_memory` (default) |

PKM is the **only** retriever that scales to ~10⁶ slots at sub-linear cost.

## Architecture

```text
ProductKeyMemory<SQRT_N, D_K, D_V>
├── keys_1: Box<[f32]>    // codebook 1: SQRT_N rows × (D_K/2) dims
├── keys_2: Box<[f32]>    // codebook 2: SQRT_N rows × (D_K/2) dims
├── values:  Box<[f32]>   // value table: SQRT_N × SQRT_N rows × D_V dims
├── from_random(seed)              // deterministic splitmix64 init (tests/benches)
├── from_centroids(c1, c2, vals)   // modelless IDW init (caller-supplied centroids)
└── query_into(q, score_fn, k, out, scratch)  // O(√N) factored top-k
```

The query kernel (`query_into`):

1. Split `q` into two `D_K/2`-dim halves.
2. Score + heapselect top-`K` from codebook 1 → `scratch.top_1`. O(√N).
3. Score + heapselect top-`K` from codebook 2 → `scratch.top_2`. O(√N).
4. Cartesian product `top_1 × top_2` (K² candidates, additive scores), top-`k` into `out`. O(K²).
5. Softmax-normalize the k selected scores → weights. *(Deviation from the
   global sigmoid rule — these are convex-combination coefficients over the
   k²-restricted candidate set, not a probability/UQ claim. See module docs.)*

Caller-allocated `PkmScratch<SQRT_N, K>` holds the two √N score arrays + two
K-length top-k buffers; reused across queries → **zero allocation** in the hot
path (G4 gate: 0 allocs / 1000 calls).

## Scoring Functions

| `ScoreFn` | Formula | Use when |
|---|---|---|
| `Dot` (default) | `q_half · key_half` | keys are normalized / magnitude carries signal |
| `Idw { epsilon }` | `−log(ε + ‖q_half − key_half‖²)` | keys are centroids; magnitude should NOT inflate score |

IDW is magnitude-invariant — a key cannot inflate its score by growing its
norm. The log bounds the best achievable score at `−log ε`, so all keys compete
on *nearness*, not *magnitude*. `epsilon` MUST be > 0; the constructor clamps.

## Evidence (Plan 408 GOAT gate)

| Gate | Target | Result | Verdict |
|---|---|---|---|
| **G1** latency (O(√N) vs O(N) brute-force, N=10⁶) | ≥ 100× speedup | **1670×** (PKM p50 17.5µs vs BF p50 29.2ms) | ✅ PASS |
| **G2** top-k Jaccard vs brute-force | mean ≥ 0.95 | **1.0000** (50 queries, perfect overlap) | ✅ PASS |
| **G3** IDW centroid-ness (advisory) | IDW ≥ 1.2× Dot | Dot 0.000 / IDW 1.000 intra-cluster rate | ✅ PASS (advisory) |
| **G4** zero-alloc steady state | 0 allocs | **0 allocations** / 1000 calls | ✅ PASS |

Promotion rule: G1 + G2 + G4 all pass → DEFAULT-ON. G3 is advisory (the
load-bearing IDW test is the Phase 2 unit test `t27_idw_attracts_to_closer_centroids`).

## Code Locations

| File | Content |
|---|---|
| `crates/katgpt-core/src/product_key_memory/types.rs` | `ProductKeyMemory`, `ScoreFn`, `PkQuery`, `PkEntry`, constructors |
| `crates/katgpt-core/src/product_key_memory/kernel.rs` | `query_into`, `score_dot`, `score_idw`, `PkmScratch`, heapselect + Cartesian top-k |
| `crates/katgpt-core/src/product_key_memory/freeze.rs` | `FrozenProductKeyMemory` (Phase 4, gated `product_key_memory_freeze`) — `Arc<RwLock<Arc<...>>>` + BLAKE3 commitment + atomic swap |
| `crates/katgpt-core/src/product_key_memory/episodic.rs` | `PkmEpisodicStore` (Phase 5, gated `product_key_memory_episodic`) — δ-rule write gate (F1 fusion: PKM × δ-Mem) |
| `examples/product_key_memory_demo.rs` | Three-part demo: basic retrieval, latency cliff, IDW vs Dot |
| `.benchmarks/408_pkm_goat.md` | Full GOAT gate results |

## Modelless Mandate (the FwPKM deviation)

The FwPKM paper's three training mechanisms are ALL forbidden per
`katgpt-rs/AGENTS.md` constraint #1, and replaced by shipped substrates:

| Forbidden paper mechanism | Modelless replacement (shipped) |
|---|---|
| `L_mem` GD on V | `DeltaMemoryState::write_segment` δ-rule (Plan 053) |
| `L_addr` GD on K | TEMP `sleep_diverse` diversity selector (Plan 005) |
| n-iter TTT loop | Sleep Consolidation N-pass (Plan 154) |

The Phase 5 δ-rule write path (`PkmEpisodicStore`) is bit-identical to one GD
step at η=1 — but is **not iterated**, so it is modelless.

## Latent vs Raw Boundary

- Slot keys + values (latent patterns) → latent, frozen, BLAKE3-committed (Phase 4).
- PKM table commitment root → raw, syncable audit artifact.
- Top-k weights → latent scalar; bridge at the sync boundary.

## Honest Approximation Gap

The PKM factorization is *approximate by construction* — the true global top-k
can span codebook boundaries in ways the per-codebook top-k misses. On random
tables the gap is zero (G2 = 1.0000); on adversarial/clustered tables the gap
shows up as Jaccard < 1.0. Consumers with adversarial key distributions should
use `K=16` or `K=32` per codebook (4× / 16× more Cartesian candidates, still
far below O(N)).

## Related

- [`.docs/03_memory/raven_rsm.md`](raven_rsm.md) — the O(1) routing retriever (different complexity class)
- [`.docs/03_memory/engram.md`](engram.md) — the O(1) hash retriever (different complexity class)
- Plan: [`.plans/408_Product_Key_Memory_Primitive.md`](../../.plans/408_Product_Key_Memory_Primitive.md)
- Research: [`.research/387_Fast_Weight_Product_Key_Memory_PKM.md`](../../.research/387_Fast_Weight_Product_Key_Memory_PKM.md)
- Source paper: [arXiv:2601.00671](https://arxiv.org/abs/2601.00671) — Zhao & Jones, "Fast-weight Product Key Memory", Sakana AI, Feb 2026
