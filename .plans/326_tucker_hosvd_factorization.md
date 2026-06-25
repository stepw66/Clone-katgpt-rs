# Plan 326: Tucker / HOSVD Tensor Factorization

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md](../.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md) (§3 candidate plan #3)
**Source paper:** [arXiv:2511.05963](https://arxiv.org/abs/2511.05963) — Duruisseaux, Kossaffi, Anandkumar, *Fourier Neural Operators: A Practical Perspective* (Caltech + NVIDIA, Nov 2025)
**Target:** `katgpt-rs/crates/katgpt-core/src/linalg/tucker.rs` (open primitive) + `riir-neuron-db/src/shard_compactor.rs` (integration), Cargo feature `tucker_factorization`
**Status:** Active — Phase 1 implementation

---

## Goal

Ship the **N-mode generalization of `thin_svd_into`** — HOSVD (Higher-Order SVD) / Tucker decomposition — as a generic open primitive in `katgpt-rs`, then wire it into the `riir-neuron-db` `ShardCompactor` as an alternative cold-tier compaction path that compactly stores a *batch* of `NeuronShard`s via a low-rank core tensor + factor matrices.

This is the third and last Gain-tier candidate plan from Research 307 (FNO practical-perspective survey). The headline FNO inference primitive (resolution-invariant spectral transport) already ships as our Super-GOAT `cross_resolution_transport` (Research 291 / Plan 310, DEFAULT-ON); this plan closes the only remaining FNO-derived modelless gap.

**Why Tucker for shards:** the existing `ShardCompactor::compact` uses Attention-Matching (AM, Plan 233), which produces `M < N` *output shards* (row-subsampled + fit). Tucker produces a *factorized representation* of the whole batch `(N, 8, 8)` — a small core `(r_N, r_R, r_C)` plus three factor matrices. Compression wins when `N` is large:

| Path | Output | Compression | Use case |
|------|--------|-------------|----------|
| `compact` (AM) | `M` shards × 64 floats | N/M ratio | Live Hot/Warm tier — produce fewer, representative shards |
| `compact_tucker` (new) | core `(r_N·r_R·r_C)` + 3 factors | depends on ranks | Cold-tier archival — store a whole zone's shard set as one factored tensor |

For a zone with N=64 shards reshaped `(64, 8, 8)` and ranks `(8, 4, 4)`: original = 4096 floats, Tucker = 128 (core) + 3·(64·8 + 8·4 + 8·4) = 128 + 1664 = 1792 floats → **2.3× compression**, lossy within the truncation budget. The AM path cannot achieve this because it always produces `Vec<NeuronShard>` — one Pod per output — paying the 64-float overhead per shard.

**Modelless:** pure closed-form linear algebra — mode-n unfoldings + thin SVD + tensor-times-matrix contractions. No gradient descent, no learned weights, no training. Promotable to default-on if the GOAT gate passes.

---

## Phase 1 — Open Primitive (katgpt-rs)

### Goal

Generic HOSVD on flat `&[f32]` + shape descriptor. Pure numeric, no shard/chain/game semantics. Reuses `thin_svd_into` from `subspace_phase_gate` — does NOT reimplement SVD.

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/linalg/tucker.rs` with module doc + feature gating
- [ ] **T1.2** Implement `TuckerConfig` — inline-array shape/ranks (`[usize; MAX_MODES]` + len, no `smallvec` dep) + validation (ranks ≤ shape per mode, ≥1 mode, ≤4 modes)
- [ ] **T1.3** Implement `TuckerScratch` — pre-allocated buffers for: per-mode unfolding matrix (sized for max `(I_n, prod(I_others))`), `SvdScratch` + `SvdResultScratch` (reuse from subspace_phase_gate), core assembly workspace
- [ ] **T1.4** Implement mode-n unfolding: `unfold_into(tensor, shape, mode, out: &mut [f32])` — row-major `I_n × prod(other)` matrix
- [ ] **T1.5** Implement `tucker_decompose_into(tensor, cfg, scratch) -> &TuckerResultScratch`: per-mode unfold → `thin_svd_into` → top-`r_n` cols of U^(n) → store factor; core = `X ×_1 U^(1)^T ×_2 U^(2)^T × … ×_N U^(N)^T`
- [ ] **T1.6** Implement `tucker_reconstruct_into(core, factors, shape, out: &mut [f32])` — inverse `X̃ = S ×_1 U^(1) ×_2 U^(2) × … ×_N U^(N)`
- [ ] **T1.7** Implement `TuckerResultScratch` (SOA, hot-path) + `TuckerResult::clone_from_scratch` convenience (owned)
- [ ] **T1.8** Add `tucker_factorization = ["subspace_phase_gate"]` to `Cargo.toml` features (initially OFF)
- [ ] **T1.9** Register `pub mod tucker;` in `linalg/mod.rs` + re-exports in `lib.rs`
- [ ] **T1.10** Unit tests: known-rank tensor recovery, orthogonality of factors, core energy bound, reconstruction error monotonic in ranks

### Phase 1 GOAT gate

- [ ] **T1.G1** Reconstruction quality: synthetic rank-`(r,r,r)` tensor, HOSVD with ranks=r → reconstruction rel error `< 1e-4`
- [ ] **T1.G2** Perf: `(64, 8, 8)` decompose + reconstruct mean latency ≤ 200µs (cold-tier budget)
- [ ] **T1.G3** No-regression: full-rank decomposition (`r_n = I_n` ∀n) → reconstruction max abs error `< 1e-4`
- [ ] **T1.G4** Alloc-free hot path: `tucker_decompose_into` with pre-warmed `TuckerScratch` → 0 allocations / 100 calls (CountingAllocator)

### Phase 1 promotion

- [ ] **T1.P** If all 4 gates pass + gain is modelless → promote `tucker_factorization` to `default`

---

## Phase 2 — Integration (riir-neuron-db)

### Goal

Wire the open primitive into `ShardCompactor` as an alternative cold-tier archival path.

### Tasks

- [ ] **T2.1** Add `tucker_factorization` feature to `riir-neuron-db/Cargo.toml` (no chain alias needed initially — `ShardCompactor` is leaf-crate-local; the chain alias `tucker_factorization` can be added when riir-chain re-exports)
- [ ] **T2.2** Add `TuckerCompactionConfig { shard_ranks: [usize; 3], target_shape: [usize; 3] }` (default `(64, 8, 8)` → ranks `(8, 4, 4)`)
- [ ] **T2.3** Add `pub struct TuckerCompaction { factors: Vec<f32>, core: Vec<f32>, source_cycle_root: [u8; 32], report: CompactionReport }` (the cold-tier archival envelope)
- [ ] **T2.4** Add `ShardCompactor::compact_tucker(&self, shards) -> Result<TuckerCompaction, CompactShardError>`:
  - Validate (wallets drained, commitments verify — same as `compact`)
  - Flatten to `(N, 8, 8)` (shard × row × col of `style_weights`)
  - Call `tucker_decompose_into`
  - Bind provenance via `compute_cycle_root`
- [ ] **T2.5** Add `TuckerCompaction::reconstruct_into(&self, out: &mut Vec<NeuronShard>)` — inverse path for Hot-tier reload (reconstructs approximate shards; zone_hash derived from cycle_root + slot as in `compact`)
- [ ] **T2.6** Tests: round-trip preserves dominant style direction, provenance root deterministic, distinct inputs → distinct roots
- [ ] **T2.7** Comparison bench: AM compaction vs Tucker compaction on N ∈ {32, 64, 256} — Tucker should win on storage size at N ≥ 64

### Phase 2 GOAT gate

- [ ] **T2.G1** Reconstruction quality: Tucker reconstructs shards whose leading semantic axis (top `semantic_axes` direction) has cosine similarity `≥ 0.95` with the source
- [ ] **T2.G2** Storage compression: Tucker envelope byte size `≤ 0.5 ×` AM-compacted shard set byte size at N=64
- [ ] **T2.G3** No-regression: full-rank Tucker (`r_N=N, r_R=8, r_C=8`) → reconstruction cos sim `≥ 0.999`
- [ ] **T2.G4** Alloc-free reconstruction hot path

### Phase 2 promotion

- [ ] **T2.P** Keep `tucker_factorization` opt-in in riir-neuron-db even if katgpt-rs promotes it to default — the integration is a new cold-tier code path that needs soak time

---

## Phase 3 — Commit + Summary

- [ ] **T3.1** Verify `cargo check --all-features` clean
- [ ] **T3.2** Verify `cargo check -p katgpt-core` (default features, post-Phase-1-promotion) clean
- [ ] **T3.3** Verify `cargo test -p katgpt-core --lib linalg::tucker` all pass
- [ ] **T3.4** Run GOAT bench, record in `.benchmarks/326_tucker_hosvd_goat.md`
- [ ] **T3.5** Commit on `develop` with `feat:` prefix

---

## Risks

- **N-mode generality vs hot-path simplicity.** True N-mode Tucker with arbitrary shape is more code than the 3-mode `(N, 8, 8)` special case we actually need. Mitigation: implement generic N-mode (≤4 modes) upfront — the per-mode unfolding + SVD + contraction pattern is uniform and the generic code is barely longer than a 3-mode special case. The 3-mode `(N,8,8)` is then just one config.
- **SVD sign ambiguity.** `thin_svd_into` documents arbitrary sign conventions. HOSVD factor matrices inherit this; reconstruction is sign-invariant (factors appear in conjugate pairs `U × U^T`), so this is safe. Document it.
- **Cold-tier commitment semantics.** Tucker envelope is not a `Vec<NeuronShard>` — it cannot be committed via the existing per-shard BLAKE3 path. Phase 2 will need its own envelope commitment (BLAKE3 over `core || factors || source_cycle_root`). Out of Phase 1 scope.
- **SVD determinism across quorum nodes.** `thin_svd_into` is documented platform-independent (no SIMD inside). Tucker inherits this. Safe for sync-boundary commitment.
