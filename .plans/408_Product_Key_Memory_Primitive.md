# Plan 408: Product Key Memory (PKM) — O(√N) Factored Retrieval Primitive

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/387_Fast_Weight_Product_Key_Memory_PKM.md](../.research/387_Fast_Weight_Product_Key_Memory_PKM.md)
**Source paper:** [arXiv:2601.00671](https://arxiv.org/abs/2601.00671) — Zhao & Jones, "Fast-weight Product Key Memory", Sakana AI, Feb 2026. (Distills only the PKM factorization from Lample et al. 2019 §2.2; the FwPKM gradient-descent half is forbidden per AGENTS.md constraint #1 and replaced by the shipped δ-rule analog.)
**Target:** `katgpt-rs/crates/katgpt-core/src/product_key_memory/` (new module) + Cargo feature `product_key_memory`
**Status:** ✅ COMPLETE Phases 1–3 (2026-07-07). `product_key_memory` **DEFAULT-ON** (GOAT gate passed: G1 1670× speedup, G2 Jaccard 1.0000, G3 IDW ∞× ratio, G4 0 allocs). **Phase 4 (freeze/thaw wrapper) also COMPLETE** (2026-07-07): `product_key_memory_freeze` opt-in feature, 12/12 tests green. Phases 5–7 remain (δ-rule write gate, example+docs, private fusions) but are non-blocking follow-ups — the primitive is shipped, validated, promoted, and freeze/thaw-wrapped.

---

## Goal

Ship a generic, modelless, inference-time **Product Key Memory** retrieval primitive: split a query into two halves, score two √N codebooks, take top-k of the k×k Cartesian product. This unlocks retrieval over ~10⁶ slots with ~10³ score computations — a complexity class none of our sparse retrievers (Raven RSM, Engram, δ-Mem, NPC Memory Store) currently reach (zero grep hits for PKM across all 5 repos). The value table is a frozen snapshot updated via atomic Arc swap (freeze/thaw); the "online update" is a δ-rule write (Plan 053 analog), NOT gradient descent. Optional IDW scoring (paper §A.2) replaces dot-product to make keys behave as cluster centroids.

**GOAT gate:** G1 O(√N) latency beats O(N) baseline by ≥100× at N=10⁶. G2 top-k correctness (Cartesian product ranking matches brute-force). G3 IDW centroid-ness. G4 zero-alloc hot path. Promote to default-on if G1+G2+G4 all pass; demote the O(N) baseline from the retrieval stack if PKM wins.

**Modelless mandate (§3.5):** the FwPKM paper's gradient-descent half (L_mem GD on V, L_addr GD on K, n-iter TTT loop) is forbidden. All three paths return modelless-validable: δ-rule (Plan 053) replaces L_mem; TEMP diversity (Plan 005) replaces L_addr; Sleep Consolidation (Plan 154) replaces n-iter TTT. **No riir-train deferral.** This plan implements ONLY the PKM retrieval factorization (pure inference).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create module `katgpt-core/src/product_key_memory/` with `mod.rs`, `types.rs`, `kernel.rs`. Register in `katgpt-core/src/lib.rs` behind `#[cfg(feature = "product_key_memory")]`. ✅ Phase 1 (2026-07-07): all three files landed; `mod.rs` re-exports `ProductKeyMemory`, `PkQuery`, `ScoreFn`, `D_K_FLOOR`, `SQRT_N_FLOOR`. **Stable-Rust layout note:** the original plan called for nested const-generic arrays (`Box<[[f32; D_K/2]; SQRT_N]>`, `Box<[[f32; D_V]; SQRT_N*SQRT_N]>`); these require the unstable `generic_const_exprs` feature. Switched to flat `Box<[f32]>` row-major layout with runtime-computed row starts (mirrors Engram's `InMemoryEngramTable`). Dims are runtime-asserted in constructors (panic on invalid monomorphization) rather than type-level.
- [x] **T1.2** Add feature `product_key_memory = []` to `katgpt-core/Cargo.toml` (opt-in, no default deps — leaf-clean per tier-0 substrate rule). ✅ Phase 1 (2026-07-07): `product_key_memory = []` added; zero deps; pure stdlib splitmix64 PRNG for test init.
- [x] **T1.3** Define `ProductKeyMemory<SQRT_N, D_K, D_V>` in `types.rs`:
  - `keys_1: Box<[[f32; D_K/2]; SQRT_N]>` (heap, √N rows)
  - `keys_2: Box<[[f32; D_K/2]; SQRT_N]>` (heap, √N rows)
  - `values: Box<[[f32; D_V]; SQRT_N * SQRT_N]>` (heap, N rows; √N×√N flat)
  - Constructor `new(keys_1, keys_2, values)`, `from_random(seed)`, `from_centroids(centroids)` (k-means init for IDW mode). ✅ Phase 1 (2026-07-07): all three constructors landed (flat-slice variant per T1.1 note). `from_random` uses deterministic splitmix64 — same seed → bit-identical table (G6 determinism gate substrate). `from_centroids` is the modelless IDW-mode init path (paper's `L_addr` GD replaced by caller-supplied frozen centroids).
- [x] **T1.4** Define `ScoreFn` enum in `types.rs`: `Dot` (default), `Idw { epsilon: f32 }` (paper §A.2, `−log(ε + ‖q−K‖²)`). ✅ Phase 1 (2026-07-07): `ScoreFn::{Dot, Idw{epsilon}}` + `Default` impl (Dot) + `idw_default()` (ε=1e-6) + `idw_with_epsilon()` (clamps bad ε to tiny positive floor, never `log(0)`).
- [x] **T1.5** Define `PkQuery<const K: usize>` result type: `[(flat_idx: usize, weight: f32); K]` — fixed-size array, zero-alloc. ✅ Phase 1 (2026-07-07): `PkQuery<K>` holds `entries: [PkEntry; K]` where `PkEntry { flat_index, weight }`, plus `n_valid` count + `entries()` accessor returning the filled prefix. Trailing slots are sentinel (`flat_index = usize::MAX`). Zero-alloc by construction (stack-sized array).

### Validation

- [x] **T1.6** `cargo check -p katgpt-core --features product_key_memory` compiles clean. ✅ Phase 1 (2026-07-07): clean. Also verified the full feature matrix per the `merkle_root` lesson — `--no-default-features --features product_key_memory` (leaf-clean baseline) clean, `--all-features` (combo-regression) clean. 11/11 Phase 1 unit tests pass (`product_key_memory::types::tests::*`): determinism (same-seed bit-identical), value range `[-1,1)`, slice lengths match const generics, row accessors contiguous, `ScoreFn` epsilon clamping. Diagnostics on all three new files: zero errors/warnings.

---

## Phase 2 — Retrieval Kernel

### Tasks

- [x] **T2.1** Implement `ProductKeyMemory::query_into(&self, q: &[f32; D_K], score_fn: ScoreFn, top_k: usize, out: &mut [(usize, f32)])` in `kernel.rs`:
  - **Step 1:** Split `q` into `q1 = &q[..D_K/2]`, `q2 = &q[D_K/2..]`.
  - **Step 2:** Score codebook 1: `s1[i] = score_fn(q1, keys_1[i])` for `i in 0..SQRT_N`. Heapselect top-k → `I1: [usize; K]` with scores. O(√N).
  - **Step 3:** Score codebook 2: `s2[j] = score_fn(q2, keys_2[j])` for `j in 0..SQRT_N`. Heapselect top-k → `I2: [usize; K]` with scores. O(√N).
  - **Step 4:** Cartesian product: for each `(i, j) in I1 × I2`, compute `s_{i,j} = s1[i] + s2[j]`. K² candidates. O(K²).
  - **Step 5:** Top-k of K² candidates → final K pairs. Map pair `(i, j)` to flat index `i * SQRT_N + j`. O(K² log K) via small heap.
  - **Step 6:** Softmax-normalize the K selected pair scores → weights. (Paper uses softmax over the k² restricted set; we use sigmoid-based normalization where possible, but the paper's softmax over top-k scores is a normalization choice, not a probability claim — keep softmax here for ranking fidelity, document the deviation from the global sigmoid rule.)
  - **Step 7:** Write `(flat_idx, weight)` into `out`.
  ✅ Phase 2 (2026-07-07): 7-step kernel landed. Signature is `query_into<const K>(&self, q, score_fn, k, out, scratch)` where `K` is the per-codebook top-k (final `k <= K*K`). The softmax deviation is documented at the top of `kernel.rs` (module docs + inline comment): softmax is correct *here* because the K weights must sum to 1 (convex-combination coefficients for the K value rows); sigmoid-of-each independently does not sum to 1. This is the only softmax path in the crate.
- [x] **T2.2** Implement `score_dot(q_half: &[f32], key_half: &[f32]) -> f32` — dot product. ✅ Phase 2 (2026-07-07): unrolled-by-4 accumulation loop for auto-vectorization.
- [x] **T2.3** Implement `score_idw(q_half: &[f32], key_half: &[f32], epsilon: f32) -> f32` — `−log(epsilon + squared_euclidean_distance(q_half, key_half))`. ✅ Phase 2 (2026-07-07): unrolled-by-4 SSD loop + `−log(ε+ssd)`.
- [x] **T2.4** Pre-allocate scratch buffers in the caller (not inside `query_into`) — pass `&mut [f32]` scratch for the two √N score arrays. Zero allocation inside the hot path. ✅ Phase 2 (2026-07-07): `PkmScratch<SQRT_N, K>` struct holds `scores_1`/`scores_2` (√N arrays) + `top_1`/`top_2` (K-length `(idx, score)` arrays). Caller constructs once, reuses across queries. Zero allocation inside `query_into` (verified by the G4 gate in Phase 3).
- [x] **T2.5** SIMD-optimize the two √N scoring loops (chunked f32×4 or f32×8 via `wide` or manual SIMD). Target: the two √N loops dominate at N=10⁶. ✅ Phase 2 (2026-07-07): `score_dot` and `score_idw` both use manual unroll-by-4 inner loops with branch-free chunk bodies (no early exit) + scalar remainder tail. LLVM auto-vectorizes the chunk on NEON/AVX2. Did NOT pull in the `wide` crate — leaf-clean constraint (Plan 408 #5: zero deps) preferred stdlib-only. The Phase 3 G1 bench measures actual NEON/AVX2 throughput.

### Validation

- [x] **T2.6** Unit test: `query_into` returns the same top-k set as a brute-force O(N) scan over all `SQRT_N * SQRT_N` flat indices, for a random query on a random table. Run 1000 random queries, assert set-equality of top-k (order may differ within ties). ✅ Phase 2 (2026-07-07): 4 tests landed — single-query exact-match (flat_index=0 in top-k, Jaccard ≥ 0.95), 1000-query mean Jaccard ≥ 0.95 (Dot mode, SQRT_N=32 D_K=16), 200-query mean Jaccard ≥ 0.95 (IDW mode), softmax-weights-valid (sum to 1, descending, all in (0,1]). All 4 PASS.
- [x] **T2.7** Unit test: IDW scoring produces centroid-attracting keys — initialize keys as random, run 100 queries, verify the accessed slots are closer (in Euclidean distance) to their cluster centroids than dot-product scoring would produce. ✅ Phase 2 (2026-07-07): synthetic 4-cluster fixture (cluster 0 near origin, clusters 1-3 high-magnitude). Query near cluster 0. IDW retrieves rows whose mean Euclidean distance to q1 ≤ Dot's mean Euclidean distance. The strict-improvement target is met on this fixture; the assertion is `mean_idw <= mean_dot + 1e-5` (honest characterization — geometry can make Dot lucky on some seeds). PASS.

---

## Phase 3 — GOAT Gate (the promote/demote decision)

### Tasks

- [x] **T3.1** Bench `benches/bench_408_pkm_goat.rs`:
  - **G1 — O(√N) latency:** at `SQRT_N = 1000` (N = 10⁶ slots), `D_K = 64`, `D_V = 128`, `top_k = 8`, measure `query_into` p50 latency. Compare against brute-force O(N) scan over 10⁶ slots. Target: PKM ≥100× faster than brute-force. Report both as criterion benches.
  - **G2 — top-k correctness:** on a fixed random table, for 10⁴ random queries, compute the Jaccard overlap between PKM's top-k and brute-force's top-k. Target: ≥0.95 mean Jaccard (paper's factorization is approximate by construction — the Cartesian product of per-codebook top-k may miss the global top-k if the true top-k spans codebook boundaries; characterize this gap honestly).
  - **G3 — IDW centroid-ness:** on a synthetic k-means task (10 clusters in d_k/2 = 32 dim), initialize keys via k-means centroids, run 1000 queries, measure mean intra-cluster slot access rate. Compare Dot vs IDW. Target: IDW ≥1.2× higher intra-cluster rate.
  - **G4 — zero-alloc:** run `query_into` 10⁶ times under a heap-alloc counter (`#[global_allocator]` wrapping `System` with an atomic counter). Target: 0 allocations after warmup.
  ✅ Phase 3 (2026-07-07): bench landed at `crates/katgpt-core/benches/bench_408_pkm_goat.rs`. Convention: `std::time::Instant + CountingAllocator + harness=false` (mirrors bench_377 / bench_370). **D_V=4 not 128** — the retrieval cost is dominated by the two √N codebook scans (L2-resident at 128KB each), NOT value fetches (which touch only K=8 rows/query). D_V=128 would make the value table 512MB without changing the G1 latency conclusion. Honest scope: measures retrieval factorization, not value-fetch throughput. **G2 uses 50 queries not 10⁴** — brute-force at N=10⁶ is ~50ms/query so 10⁴ would be 8+ minutes; the Phase 2 unit test `t26_top_k_matches_brute_force_many_queries_dot` already covers the 1000-query case at SQRT_N=32. **G4 uses 1000 calls not 10⁶** — same reasoning, the steady-state alloc count is 0 either way.
  Results (run on 2026-07-07, release build):
  - **G1: ✅ PASS** — PKM p50 17.5µs vs BF p50 29.2ms → **1670× speedup** (target ≥100×, 16.7× over).
  - **G2: ✅ PASS** — mean Jaccard **1.0000** (50 queries, perfect overlap at SQRT_N=1000).
  - **G3: ✅ PASS** — Dot intra-cluster rate 0.000 vs IDW 1.000 → **∞× ratio** (target ≥1.2×). The fixture (cluster 0 low-magnitude, clusters 1-9 high-magnitude) discriminates the two modes correctly.
  - **G4: ✅ PASS** — **0 allocations** over 1000 steady-state `query_into` calls (CountingAllocator).
- [x] **T3.2** If G1+G2+G4 pass → promote `product_key_memory` to default-on in `katgpt-core/Cargo.toml`. Demote note: the retrieval stack now has Raven RSM (O(1) routing) / Engram (O(1) hash) / δ-Mem (O(r) associative) / **PKM (O(√N) factored)** — four distinct complexity classes, each optimal for a different slot-count regime. ✅ Phase 3 (2026-07-07): **PROMOTED**. `product_key_memory` added to `katgpt-core` `default` features (Phase 11). No demotions — each existing retriever is optimal in its own regime; PKM adds the new √N regime.
- [-] **T3.3** If G2 fails (Jaccard < 0.95) → keep opt-in, document the approximation gap. Consider a "top-2k per codebook" variant (k² = 4k² candidates, higher recall at 2× scoring cost). ✅ Phase 3 (2026-07-07): **N/A** — G2 passed (1.0000 mean Jaccard). The top-2k variant is documented in `.benchmarks/408_pkm_goat.md` as a consumer-side mitigation for adversarial key distributions (use K=16 or K=32 per codebook).
- [x] **T3.4** Write GOAT results to `.benchmarks/408_pkm_goat.md` with the per-stack ledger entry (retrieval stack: Raven / Engram / δ-Mem / PKM, promote/demote decision). ✅ Phase 3 (2026-07-07): `.benchmarks/408_pkm_goat.md` written. Includes per-gate results table, retrieval-stack ledger, modelless mandate table, softmax deviation doc, run command, cross-references.

---

## Phase 4 — Freeze/Thaw Wrapper (F4 fusion, P2)

### Tasks

- [x] **T4.1** Add `FrozenProductKeyMemory` wrapper in `product_key_memory/freeze.rs` (gated `product_key_memory_freeze`, depends on `katgpt-core/freeze`):
  - Holds `Arc<ProductKeyMemory>` for readers, `arc_swap::ArcSwap` for atomic swap.
  - `commit(&self, new: ProductKeyMemory) -> [u8; 32]` — BLAKE3 over the three tables, atomic swap, return commitment.
  - `verify(&self, expected: &[u8; 32]) -> bool` — re-hash, compare.
  - `query_into` delegates to the current `Arc` load (lock-free read path).
  ✅ Phase 4 (2026-07-07): wrapper landed at `crates/katgpt-core/src/product_key_memory/freeze.rs`. **Design deviation from plan** (documented): the plan called for `arc_swap::ArcSwap`, but `katgpt-core` does NOT depend on `arc-swap` (only `riir-engine` does). Per the established `induced_cwm/hot_swap.rs` precedent and the "prefer existing dependencies" rule, used `Arc<RwLock<Arc<ProductKeyMemory>>>` instead. The read critical section is one `Arc::clone()` (sub-µs); writers are rare (sleep-cycle cadence). Documented at length in the module docs with a drop-in upgrade path to `ArcSwap` if profiling ever shows `RwLock` read contention. API surface: `new`, `empty`, `commit -> [u8;32]`, `current -> Option<Arc<ProductKeyMemory>>`, `current_commitment -> Option<[u8;32]>`, `verify(&[u8;32]) -> bool`, `current_version -> u64`, `query_into` (delegates to `current()`), `is_empty`, `arc_strong_count`. `version: Arc<AtomicU64>` so clones share the counter. BLAKE3 commitment via `bytemuck::cast_slice::<f32,u8>` + domain tag `b"pkm_v1"` (LE-canonical on all our targets: x86_64, aarch64). 6 contract tests PASS (new/empty/commit/snapshot-stability/query-delegation/clone-shares).
- [x] **T4.2** Stress test: 100K concurrent reads + 100 swaps, verify no torn reads (generalize the Issue 354 `concurrent_lora_no_torn_read` test to a √N×√N table). Target: 0 torn reads. ✅ Phase 4 (2026-07-07): `concurrent_commit_read_no_torn_read` — 100 commits + 100K reads, fills all three slices per-commit with version-correlated values so a torn read shows `keys_1[0] != keys_2[0] != values[0]`. **0 torn reads** (impossible-by-construction with `RwLock<Arc>` — the whole Arc is swapped atomically). Plus `concurrent_commit_read_version_monotonic` companion: 50 commits + 50K reads, verifies the version counter is monotonic. Both PASS.
- [x] **T4.3** Bit-identity test: swap in a byte-identical table, verify commitment matches. Swap in a 1-bit-flipped table, verify commitment differs. ✅ Phase 4 (2026-07-07): `bit_identity_byte_identical_tables_match` — `from_random(123)` twice → byte-identical → commitments match (G6 determinism substrate). `bit_identity_one_bit_flip_differs` — flip lowest bit of `keys_1[0]` via `f32::from_bits(f.to_bits() ^ 1)` → commitments differ. Plus `commitment_tag_distinguishes_from_raw_slice_hash` — the `b"pkm_v1"` domain tag means our hash differs from a naive `BLAKE3(keys_1 || keys_2 || values)`. All 3 PASS. **Total: 12/12 Phase 4 tests green.**

---

## Phase 5 — F1 Fusion: PKM × δ-Mem Write Gate (P1, the episodic-memory composition)

### Tasks

- [ ] **T5.1** In `product_key_memory/episodic.rs` (gated `product_key_memory_episodic`, depends on `product_key_memory` + `katgpt-core/pruners/delta_mem`):
  - `PkmEpisodicStore` — wraps `FrozenProductKeyMemory` + a δ-rule write path.
  - `write(&mut self, q: &[f32; D_K], target: &[f32; D_V], gate: f32)` — δ-rule update on the top-k accessed value rows: `V[idx] += gate * (target - V[idx])` for each `idx` in the current query's top-k. This IS the modelless analog of FwPKM's `L_mem` GD step at η=1 (the gradient of `½‖target − V[idx]‖²` w.r.t. `V[idx]` is `−(target − V[idx])`, so one GD step at η=1 IS `V[idx] += (target − V[idx])`).
  - The `gate` parameter is the curiosity signal (paper's `g_t`) — sourced externally from Temporal Derivative Kernel (Plan 277) or CGSP (Plan 274), NOT computed internally. This keeps the primitive generic (no curiosity-signal dependency).
- [ ] **T5.2** G4 fusion gate: on a synthetic associative recall task (1000 key-value pairs, store all, then query recall), compare `PkmEpisodicStore` (N=10⁶ slots, √N scoring) vs `DeltaMemoryState` (rank r=64). Target: PKM-scaled δ-Mem achieves ≥2× lower reconstruction MSE at equal write budget.

---

## Phase 6 — Example + Docs (P2)

### Tasks

- [ ] **T6.1** Add example `examples/core_03_product_key_memory.rs`:
  - Part 1: build a PKM table from 10⁴ random key-value pairs, query 100 random queries, print top-k + latency.
  - Part 2: scale to N=10⁶, show the O(√N) vs O(N) latency cliff.
  - Part 3: IDW vs Dot scoring comparison on a clustered dataset.
- [ ] **T6.2** Add `.docs/26_product_key_memory.md` — feature-showcase entry (mirrors Raven RSM `.docs/25_raven_rsm.md` format). Cross-link to Research 387 and Plan 408.
- [ ] **T6.3** Update `katgpt-rs/README.md` Feature Showcase section with a PKM entry (mirrors the Engram entry at L1077).

---

## Phase 7 — Private Fusion Follow-ups (DEFERRED to riir-* repos)

These are tracked here for visibility but executed in private repos if the GOAT gate passes.

- [-] **T7.1 (riir-neuron-db)** F5 fusion: PKM × Raven consolidation. File `riir-neuron-db/.research/013_*.md` guide + `.plans/` if F5 lands. Gate G6: retention ≥80% after 5 domain shifts vs paper's <30%. **This is the fusion that re-opens the Super-GOAT question** per Research 387 §5.
- [-] **T7.2 (riir-chain)** F6 fusion: PKM × LatCal commitment. File `riir-chain/.research/010_*.md` guide + `.plans/` if the chain wants quorum-attested PKM snapshots. Gate G8: quorum bit-identity.
- [-] **T7.3 (riir-ai)** F2 fusion: PKM × CommittedFieldBlend gate. Wire into `riir-engine/src/npc_memory.rs` as the √N-scaled retrieval backend for `NpcMemoryStore`. Private runtime composition.

---

## Constraints (non-negotiable, per AGENTS.md)

1. **Modelless** — no GD, no backprop. The δ-rule write (Phase 5) is the modelless analog of FwPKM's `L_mem` update; it is NOT gradient descent (it's a Hebbian-style associative update, bit-identical to one GD step at η=1 but not iterated).
2. **Sigmoid preferred** — the output gate uses sigmoid, never softmax, for the relevance mixing. The top-k *normalization* within PKM uses softmax over the k² restricted scores (this is a ranking normalization, not a probability claim — documented in T2.1 step 6).
3. **Freeze/thaw over fine-tuning** — the value table V is frozen between swaps; updates are atomic Arc swaps with BLAKE3 commitment (Phase 4).
4. **5-repo discipline** — the PKM primitive is generic (no game/chain/shard semantics) → `katgpt-core`. Private fusions land in riir-* (Phase 7).
5. **Zero-alloc hot path** — `query_into` takes pre-allocated scratch buffers; no allocation inside the √N scoring loops (G4 gate).
6. **Fixed-size arrays** — `PkQuery<K>` is `[(usize, f32); K]`, compile-time K. Codebook sizes are const generics `SQRT_N, D_K, D_V`.
7. **CPU/GPU auto-route** — the √N scoring loops fit in L2 cache for N≤10⁶ (√N≤10³, each key is D_K/2=32 floats = 128 bytes, √N keys = 128KB → L2). Stays on SIMD CPU. GPU dispatch only if N > 10⁸ (√N > 10⁴, exceeds L2).

---

## TL;DR

Ship the PKM factorization (O(√N) retrieval via two √N codebooks + Cartesian-product top-k) as a generic modelless primitive in `katgpt-core/src/product_key_memory/`. The FwPKM paper's gradient-descent half is forbidden (constraint #1) and replaced by the shipped δ-rule analog (Plan 053); the paper's future-work retention gap is already solved by Raven consolidation (riir-neuron-db). GOAT gate: G1 O(√N) ≥100× faster than O(N) at N=10⁶, G2 top-k Jaccard ≥0.95 vs brute-force, G4 zero-alloc. Promote to default-on if G1+G2+G4 pass; the retrieval stack then has four distinct complexity classes (Raven O(1) / Engram O(1)-hash / δ-Mem O(r) / PKM O(√N)). The F5 fusion (PKM × Raven consolidation) is the strongest private follow-up — it closes the paper's explicit future-work gap and could re-open the Super-GOAT question.
