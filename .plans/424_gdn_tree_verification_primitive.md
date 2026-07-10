# Plan 424: GDN Rollback-Free Tree Verification — Masked Triangular Solve for Delta-Rule Speculative Trees

**Date:** 2026-07-10
**Research:** [katgpt-rs/.research/407_Trees_from_Marginals_GDN_Tree_Verify.md](../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md)
**Source paper:** [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) — Oda et al., "Trees from Marginals", §3.4
**Target:** `katgpt-rs/crates/katgpt-core/src/gdn_tree_verify/` (new module) + Cargo feature `gdn_tree_verify`
**Status:** Active — Phase 1 <state>

---

## Goal

Ship a modelless primitive that verifies speculative draft trees against GDN (Gated DeltaNet) recurrent layers **without rolling back the recurrent state**. The algorithm (paper §3.4) extends the chunked delta-rule recurrence to tree-structured drafts via a partial order (ancestor relation), reducing verification to a masked triangular solve `(I + X)U = βV` followed by an ancestor-masked output read. The committed state is never speculatively written — a single commit pass replays the delta-rule along the accepted path after Traversal verification picks the leaf.

This fills a confirmed gap: katgpt-rs ships GDN2 (Plan 105, default-on) for the main forward path and KV-cache snapshot/rollback tree verification for attention models (Plan 012), but has **no tree verification for GDN/delta-rule recurrent layers**. The paper explicitly frames this as an open problem (STree only handles diagonal recurrences; GDN's non-commutative `I − βkkᵀ` admits no cumulative-product form).

**GOAT gate:** G1 (bit-exact correctness vs per-branch sequential verify), G2 (perf: ≥2× faster than per-branch at T=32, ≥4× at T=64), G3 (no-regression on default tests), G4 (alloc-free hot path — pre-allocated scratch, reused via `clear()`).

**Promotion rule:** if G1–G4 pass → promote `gdn_tree_verify` to opt-in (NOT default — it only activates on `QwenDeltaNet` / GDN-layer configs, which are themselves opt-in via `deltanet_inference`). The feature is a complement to Plan 012's attention verify, not a replacement.

---

## Phase 1 — Skeleton: `GdnTreeVerifier` + ancestor metadata (CORE)

### Tasks

- [ ] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/gdn_tree_verify/mod.rs` with feature gate `#[cfg(feature = "gdn_tree_verify")]`. Add the feature to `katgpt-core/Cargo.toml` as opt-in (`gdn_tree_verify = []`).
- [ ] **T1.2** Define `pub struct TreeTopology { parent: Vec<usize>, ancestor_bits: Vec<u64>, cumulative_log_decay: Vec<f64>, topo_order: Vec<usize> }` — the tree metadata computed once per decode step from parent pointers.
  - `ancestor_bits[i]`: bitmask of proper ancestors of node i, packed into `ceil(T/64)` u64 words. Node j is an ancestor of i iff bit j is set in `ancestor_bits[i]`.
  - `cumulative_log_decay[i]`: `Σ_{j ⪯ i} ln(αⱼ)` (log-space cumulative product of decay factors along the branch from root to i).
  - `topo_order`: nodes sorted topologically (parent before child).
- [ ] **T1.3** Implement `pub fn build_topology(parents: &[usize], alphas: &[f32]) -> TreeTopology`:
  - Compute `ancestor_bits` by propagating parent's ancestor bits + parent bit (BFS/DFS from root).
  - Compute `cumulative_log_decay` by accumulating `ln(α)` along branches.
  - Topological sort (Kahn's algorithm or DFS post-order reverse).
- [ ] **T1.4** Define `pub struct GdnTreeVerifier { scratch_x: Vec<f32>, scratch_u: Vec<f32>, scratch_rhs: Vec<f32>, block_buf: [f32; 1024] }` — pre-allocated scratch buffers (sized for max tree T and head dim d_k). Use `Vec::with_capacity()` once at construction; `clear()` + reuse on each `verify()` call.
- [ ] **T1.5** Unit test: `build_topology` on a small tree (root → 2 children → 3 grandchildren) produces correct ancestor bitmasks, cumulative decays, and topo order.

---

## Phase 2 — Masked triangular solve (CORE ALGORITHM)

### Tasks

- [ ] **T2.1** Implement `fn build_interaction_matrix(&mut self, topo: &TreeTopology, keys: &[f32], betas: &[f32], d_k: usize) -> &[f32]`:
  - Build `X ∈ ℝ^{T×T}` lower-triangular, ancestor-masked: `X[i][j] = 𝟙[j ≺ i] · exp(log_a[i] - log_a[j]) · β[i] · k[i]ᵀk[j]`.
  - The dot product `k[i]ᵀk[j]` uses `simd_dot_f32` (Plan: katgpt-core simd module).
  - The ancestor mask zeros out non-ancestor entries.
  - Store in `self.scratch_x` (row-major, T×T).
- [ ] **T2.2** Implement `fn forward_substitution_tiled(&mut self, x: &[f32], rhs: &[f32], t: usize, block_size: usize) -> &[f32]`:
  - Solve `(I + X)U = rhs` via tiled forward substitution (paper Eq. 13):
    - For each diagonal block `(I + X_bb)` of size `block_size`: invert by repeated squaring (matrix is small, ≤ 32×32, in `self.block_buf`).
    - Off-diagonal: cascade over sub-blocks `U_b = (I + X_bb)⁻¹(rhs_b − Σ_{c<b} X_bc · U_c)`.
  - `block_size = 32` (paper default Bc). On CPU SIMD, 32×32 fits in L1 and maps to 8-wide SIMD chunks.
  - Return `&self.scratch_u` (the solution U, T×d_v).
- [ ] **T2.3** Implement `fn compute_outputs(&mut self, topo: &TreeTopology, u: &[f32], s0: &[f32], queries: &[f32], keys: &[f32], d_k: usize, d_v: usize) -> Vec<f32>`:
  - Compute `O = (1/√d_k)(aQS₀ + Y(U − WS₀))` (paper Eq. 11).
  - The `WS₀` term is folded into the RHS of the solve (paper §3.4.2: "the −wⱼᵀS₀ term is folded into the right-hand side, so W is never formed"). Implement this folding in T2.2's RHS construction.
  - `Y[i][j] = 𝟙[j ⪯ i] · exp(log_a[i] - log_a[j]) · q[i]ᵀk[j]` (query-key, ancestor-or-self masked).
  - Return per-node outputs O (T×d_v).
- [ ] **T2.4** Unit test: for a linear chain tree (each node has exactly 1 child = standard sequential decode), `verify()` produces bit-identical outputs to a sequential GDN2 forward pass on the same tokens. This is the correctness anchor — a chain is a degenerate tree.
- [ ] **T2.5** Unit test: for a branching tree, `verify()` produces outputs matching a per-branch sequential verify (the baseline the paper beats). Both should produce the same per-node outputs (the algorithm is exact, not approximate).

---

## Phase 3 — Commit-on-accept + integration API

### Tasks

- [ ] **T3.1** Implement `pub fn commit_path(&mut self, topo: &TreeTopology, accepted_path: &[usize], keys: &[f32], values: &[f32], alphas: &[f32], betas: &[f32], s0: &mut [f32], d_k: usize, d_v: usize)`:
  - Replay the delta-rule recurrence `Sₜ = αₜ(I − βₜkₜkₜᵀ)Sₜ₋₁ + βₜkₜvₜᵀ` along the accepted path only (paper Eq. 7).
  - This is the ONLY state write — updates `s0` in place.
  - Uses the existing GDN2 recurrence math (factor out from `gdn2/` if not already a free function).
- [ ] **T3.2** Define the top-level API:
  ```rust
  pub fn verify_gdn_tree(
      verifier: &mut GdnTreeVerifier,
      topo: &TreeTopology,
      layer: &GdnLayerParams,  // K, V, Q, α, β for all T nodes
      s0: &[f32],              // committed prefix state (read-only)
      d_k: usize,
      d_v: usize,
  ) -> Vec<f32>               // per-node outputs O
  ```
  And:
  ```rust
  pub fn commit_accepted(
      verifier: &mut GdnTreeVerifier,
      topo: &TreeTopology,
      accepted_leaf: usize,
      layer: &GdnLayerParams,
      s0: &mut [f32],          // updated in place
      d_k: usize,
      d_v: usize,
  )
  ```
- [ ] **T3.3** Wire into `lib.rs`: add `#[cfg(feature = "gdn_tree_verify")] pub mod gdn_tree_verify;` to `katgpt-rs/crates/katgpt-core/src/lib.rs`.
- [ ] **T3.4** Run `cargo check -p katgpt-core --features gdn_tree_verify` — must pass clean.
- [ ] **T3.5** Run `cargo check -p katgpt-core` (default features) — must still pass (feature is opt-in).

---

## Phase 4 — Multi-head batching + QwenDeltaNet integration

### Tasks

- [ ] **T4.1** Extend `verify_gdn_tree` to handle multiple key heads (H_k) and value heads (H_v). The interaction matrix X is per-key-head; the solve is batched across heads. State layout: `[H_v, d_v, d_k]` (matching GDN2's existing layout per Research 070).
- [ ] **T4.2** Add a trait or adapter that bridges `GdnTreeVerifier` to the existing `Gdn2State` (Plan 105). The verifier reads S₀ from `Gdn2State` and writes back via `commit_accepted`. No modification to GDN2 itself.
- [ ] **T4.3** Add an integration path in the speculative step (`speculative/step_paged.rs` or `katgpt-forward::step`): when `Config::architecture == QwenDeltaNet` AND `gdn_tree_verify` is enabled, route GDN layers through `verify_gdn_tree` instead of the KV-rollback path. Attention layers continue to use KV-rollback.
- [ ] **T4.4** Integration test: speculative decode on `Config::qwen_deltanet()` with `gdn_tree_verify` produces the same accepted tokens as without the feature (correctness — the verify is exact, just faster).

---

## Phase 5 — GOAT gate (benchmarks + promote decision)

### Tasks

- [ ] **T5.1 (G1 — correctness)** Test: `verify_gdn_tree` on random trees (T=16,32,64,128) produces outputs within `1e-5` of a per-branch sequential verify reference (f64). The algorithm is exact; deviation is only floating-point rounding. Must pass on all tree sizes.
- [ ] **T5.2 (G2 — perf)** Benchmark `benches/bench_424_gdn_tree_verify.rs`:
  - Compare `verify_gdn_tree` vs per-branch sequential verify at T={16,32,64,128}.
  - Target: ≥2× speedup at T=32, ≥4× at T=64 (paper achieves 2.7× at T=32, 4.6× at T=64 on B200; CPU numbers will differ but the scaling trend should hold — the solve cost grows as ⌈T/Bc⌉ blocks vs T sequential steps).
  - Use `CARGO_TARGET_DIR=/tmp/424_gdn_tree_verify` per AGENTS.md rule.
- [ ] **T5.3 (G3 — no-regression)** Run `cargo test -p katgpt-core --lib` (default features) — all existing tests pass (feature is opt-in, no default impact).
- [ ] **T5.4 (G4 — alloc-free)** Verify `GdnTreeVerifier::verify()` does zero heap allocations after construction (all scratch pre-allocated, reused via `clear()`). Use a debug-allocator assertion or manual inspection.
- [ ] **T5.5** Clean up: `rm -rf /tmp/424_gdn_tree_verify`.

### Promote decision

- [ ] **T5.6** If G1–G4 all pass: keep `gdn_tree_verify` as opt-in (NOT default — only relevant for `QwenDeltaNet` configs which are themselves opt-in). Document in `.docs/inference/speculative_decoding.md` as the GDN-layer complement to Plan 012's attention verify. Record the per-stack outcome in the feature catalog.
- [ ] **T5.7** If G1 fails: the algorithm is wrong — debug against the per-branch reference. If G2 fails (no speedup on CPU): the SIMD tiling needs tuning, or CPU is bottlenecked differently than GPU (the solve is O(T²·d_k) vs sequential O(T·d_k·d_v) — the crossover depends on d_k vs d_v ratio). Record honestly and keep opt-in.

---

## Phase 6 — DDTree argmax-of-marginal tuning (Gain, optional)

### Tasks

- [ ] **T6.1** In `dd_tree.rs`, add a config flag `deep_argmax_threshold: Option<usize>` (default `None`). When set, at tree depth > threshold, use argmax-of-marginal instead of sampling from the full marginal. Based on paper §3.5 / Figure 6 (crossover at draft length 2–4).
- [ ] **T6.2** Benchmark: does `deep_argmax_threshold = Some(4)` improve mean acceptance length on the existing DDTree benchmark? If yes → document; if no → revert, note as config-dependent.

---

## Key Design Decisions

1. **Read-only verify, single-write commit.** The verify pass never touches S₀. Only `commit_accepted` writes S₀, and only along the one accepted path. This is the paper's key design choice — it eliminates rollback entirely.

2. **The `WS₀` folding trick.** The paper folds the `−wⱼᵀS₀` term into the RHS of the forward substitution, so W is never materialized. This saves a full T×T matrix and a second solve. Implement this in T2.2.

3. **CPU vs GPU.** The paper targets B200 with a fused CUDA kernel. katgpt-rs is CPU-first (SIMD). The algorithm maps to blocked SIMD matmul (`simd_matmul_rows`), but the absolute speedup will differ — the paper's 7.1× at T=128 is GPU-specific. CPU speedup depends on whether the ⌈T/32⌉-block solve beats T sequential steps for our head dimensions. G2 target is conservative (≥2× at T=32).

4. **Not a replacement for Plan 012.** Plan 012's KV-rollback verify handles attention layers. This plan handles GDN layers. They coexist — `QwenDeltaNet` configs route each layer type to its respective verifier.

5. **No Traversal verification.** The paper uses Traversal verification [10] for acceptance coupling. Our DDTree has its own acceptance logic. Integrating Traversal is a separate follow-up; this plan only ships the verify primitive (produces per-node outputs), not the acceptance policy.

---

## References

- **Paper:** [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) §3.4 — Oda et al., Jul 2026
- **Research note:** [katgpt-rs/.research/407_*.md](../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md)
- **Internal deps:** Plan 105 (GDN2 — `Gdn2State`), Plan 012 (DDTree — `TreeBuilder`, KV-rollback verify), Plan 182 (QwenDeltaNet — `Config::qwen_deltanet()`)
- **GDN math:** Gated Delta Networks (arXiv:2412.06464), chunked delta rule (arXiv:2406.06499)
- **Prior art (diagonal):** STree (arXiv:2505.14969) — handles Mamba diagonal recurrences only
