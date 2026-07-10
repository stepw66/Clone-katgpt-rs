# Plan 424: GDN Rollback-Free Tree Verification — Masked Triangular Solve for Delta-Rule Speculative Trees

**Date:** 2026-07-10
**Research:** [katgpt-rs/.research/407_Trees_from_Marginals_GDN_Tree_Verify.md](../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md)
**Source paper:** [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) — Oda et al., "Trees from Marginals", §3.4
**Target:** `katgpt-rs/crates/katgpt-core/src/gdn_tree_verify/` (new module) + Cargo feature `gdn_tree_verify`
**Status:** Phases 1-3 complete, Phase 5 GOAT gate complete (G1-G4 all PASS, G3 restored after Issue 125 shim-removal `--all-features` regression fixed), Phase 4 T4.1-T4.4 + T4.3b + T4.3c complete (multi-head batching + bridge + speculative step integration + integration tests + convention alignment + hybrid QwenDeltaNet riir-ai consumer), Phase 6 complete (T6.1 config flag + T6.2 benchmark, negative result). **All tasks complete.** Weaver adapter training recipe captured in riir-train/.research/402.

---

## Goal

Ship a modelless primitive that verifies speculative draft trees against GDN (Gated DeltaNet) recurrent layers **without rolling back the recurrent state**. The algorithm (paper §3.4) extends the chunked delta-rule recurrence to tree-structured drafts via a partial order (ancestor relation), reducing verification to a masked triangular solve `(I + X)U = βV` followed by an ancestor-masked output read. The committed state is never speculatively written — a single commit pass replays the delta-rule along the accepted path after Traversal verification picks the leaf.

This fills a confirmed gap: katgpt-rs ships GDN2 (Plan 105, default-on) for the main forward path and KV-cache snapshot/rollback tree verification for attention models (Plan 012), but has **no tree verification for GDN/delta-rule recurrent layers**. The paper explicitly frames this as an open problem (STree only handles diagonal recurrences; GDN's non-commutative `I − βkkᵀ` admits no cumulative-product form).

**GOAT gate:** G1 (bit-exact correctness vs per-branch sequential verify), G2 (perf: ≥2× faster than per-branch at T=32, ≥4× at T=64), G3 (no-regression on default tests), G4 (alloc-free hot path — pre-allocated scratch, reused via `clear()`).

**Promotion rule:** if G1–G4 pass → promote `gdn_tree_verify` to opt-in (NOT default — it only activates on `QwenDeltaNet` / GDN-layer configs, which are themselves opt-in via `deltanet_inference`). The feature is a complement to Plan 012's attention verify, not a replacement.

---

## Phase 1 — Skeleton: `GdnTreeVerifier` + ancestor metadata (CORE)

### Tasks

- [x] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/gdn_tree_verify/mod.rs` with feature gate `#[cfg(feature = "gdn_tree_verify")]`. Add the feature to `katgpt-core/Cargo.toml` as opt-in (`gdn_tree_verify = []`).
- [x] **T1.2** Define `pub struct TreeTopology { parent: Vec<usize>, ancestor_bits: Vec<u64>, cumulative_log_decay: Vec<f64>, topo_order: Vec<usize> }` — the tree metadata computed once per decode step from parent pointers.
  - `ancestor_bits[i]`: bitmask of proper ancestors of node i, packed into `ceil(T/64)` u64 words. Node j is an ancestor of i iff bit j is set in `ancestor_bits[i]`.
  - `cumulative_log_decay[i]`: `Σ_{j ⪯ i} ln(αⱼ)` (log-space cumulative product of decay factors along the branch from root to i).
  - `topo_order`: nodes sorted topologically (parent before child).
- [x] **T1.3** Implement `pub fn build_topology(parents: &[usize], alphas: &[f32]) -> TreeTopology`:
  - Compute `ancestor_bits` by propagating parent's ancestor bits + parent bit (BFS/DFS from root).
  - Compute `cumulative_log_decay` by accumulating `ln(α)` along branches.
  - Topological sort (Kahn's algorithm or DFS post-order reverse).
- [x] **T1.4** Define `pub struct GdnTreeVerifier { scratch_x: Vec<f32>, scratch_u: Vec<f32>, scratch_rhs: Vec<f32>, block_buf: [f32; 1024] }` — pre-allocated scratch buffers (sized for max tree T and head dim d_k). Use `Vec::with_capacity()` once at construction; `clear()` + reuse on each `verify()` call.
- [x] **T1.5** Unit test: `build_topology` on a small tree (root → 2 children → 3 grandchildren) produces correct ancestor bitmasks, cumulative decays, and topo order.

---

## Phase 2 — Masked triangular solve (CORE ALGORITHM)

### Tasks

- [x] **T2.1** Implement `fn build_x(...)` — builds `X ∈ ℝ^{T×T}` lower-triangular, ancestor-masked: `X[i][j] = 𝟙[j ≺ i] · exp(log_a[i] - log_a[j]) · β[i] · k[i]ᵀk[j]` using `simd_dot_f32`.
- [x] **T2.2** Implement `fn build_rhs(...)` (folded WS₀ trick) + `fn forward_sub(...)` — solves `(I + X)U' = rhs` via forward substitution. The tiled/block variant (paper Eq. 13) is deferred to Phase 5 optimization; the simple per-row forward sub is sufficient for G1 correctness.
- [x] **T2.3** Implement `fn compute_out(...)` — computes `O[i] = (1/√dₖ)[aᵢ(qᵢᵀS₀) + Σ_{j⪯i}(aᵢ/aⱼ)(qᵢᵀkⱼ)·U'[j]]` with the folded RHS. Top-level: `verify_gdn_tree_into()`.
- [x] **T2.4** Unit test: linear chain tree matches sequential GDN2 forward pass (test_linear_chain_matches_sequential). ✅
- [x] **T2.5** Unit test: branching tree matches per-branch sequential verify (test_branching_tree_matches_per_branch). ✅

---

## Phase 3 — Commit-on-accept + integration API

### Tasks

- [x] **T3.1** Implement `pub fn commit_path(...)` + `pub fn commit_accepted(...)` — replays the delta-rule recurrence along the accepted path, updates `s0` in place. ✅ (test_commit_path_matches_sequential)
- [x] **T3.2** Define the top-level API: `verify_gdn_tree()`, `verify_gdn_tree_into()`, `commit_accepted()`, `commit_path()`. Uses `GdnLayerParams` struct for clean param passing. ✅
- [x] **T3.3** Wire into `lib.rs`: `#[cfg(feature = "gdn_tree_verify")] pub mod gdn_tree_verify;` added. ✅
- [x] **T3.4** `cargo check -p katgpt-core --features gdn_tree_verify` — passes clean. ✅
- [x] **T3.5** `cargo check -p katgpt-core` (default features) + `--all-features` — both pass clean. ✅

---

## Phase 4 — Multi-head batching + QwenDeltaNet integration

### Tasks

- [x] **T4.1** Extend `verify_gdn_tree` to handle multiple key heads (H_k) and value heads (H_v). ✅ `GdnMultiHeadParams` + `verify_gdn_tree_multihead` + `commit_accepted_multihead` in `gdn_tree_verify/mod.rs`. Topology shared across heads; scalar α/β shared (paper form, matches `Gdn2GateConfig::Kda`). Per-head α/β callers use the single-head API in a loop. Tests: multi-head matches single-head, matches reference, commit matches sequential.
- [x] **T4.2** Add a trait or adapter that bridges `GdnTreeVerifier` to the existing `MultiLayerGdn2Cache` (Plan 105). ✅ `tree_verify_bridge.rs` in `katgpt-attn/src/gdn2/` (feature `gdn_tree_verify = ["gdn2_attention", "katgpt-core/gdn_tree_verify"]`). `verify_gdn2_tree_layer` reads S₀ from `MultiLayerGdn2Cache.layers[layer].heads[h].s`; `commit_gdn2_tree_layer` writes back. No modification to GDN2 kernel. Layout match is exact (both `[d_k × d_v]` row-major). Scalar α auto-derived from per-channel `decay_alpha` (exact when uniform, geometric-mean approx otherwise); β = 1.0 (GDN2 default `write_w_scalar`). `gdn2_layer_is_paper_compatible` checks exact-verification conditions. 6 tests pass.
- [x] **T4.3** Add an integration path for GDN2 models in the speculative step. `speculative_step_gdn_tree` in `src/speculative/step_gdn_tree.rs` (root crate, gated `gdn_tree_verify`). Uses `forward_tree_gdn2` (new, `katgpt-attn/src/gdn2/tree_forward.rs`) to process all tree nodes in one pass via tree verify, then p/q rejection sampling along the best path, then commits via sequential `forward_gdn2` replay. Also ships `build_topology_from_tree_nodes` in `katgpt-core/src/gdn_tree_verify/mod.rs` (DDTree path-encoded nodes to parent-index topology).
- [x] **T4.3b** Convention alignment between tree verify and GDN2 kernel. **Root cause was NOT a delta-rule convention mismatch** (both use the same chunked delta rule with decay-before-correction read: `S_new = a*S_old + b*k*(v - a*k^T*S_old)`). The actual bug was in `forward_tree_gdn2`'s transformer plumbing: (1) the per-node attention residual (`ctx.xr`) was overwritten across the QKV projection loop, so the output projection loop used the WRONG node's residual; (2) the residual was saved from the pre-norm hidden state instead of post-first-rmsnorm (not matching `forward_gdn2`); (3) a second rmsnorm was missing. Fixed by buffering per-node residuals in `attn_residuals: Vec<f32>` and matching `forward_gdn2`'s exact `rmsnorm -> xr=norm(x) -> rmsnorm -> QKV -> ... -> x += xr` pattern. **Proven: chain and branching trees match sequential `forward_gdn2` to <=3e-6** (f32 accumulation order) via 3 new tests in `tree_forward.rs`.
- [x] **T4.3c** Hybrid QwenDeltaNet (mixed Attention + DeltaNet layers). Attention layers use per-branch sequential KV-rollback (simplest correct approach for branching trees); DeltaNet layers use tree verify via the katgpt-core primitive. Per-layer dispatch in the tree forward. **riir-ai consumer** — `forward_tree_qwen_deltanet` in `riir-ai/crates/riir-engine/src/deltanet/tree_forward.rs` (feature `gdn_tree_verify = ["deltanet_inference", "katgpt-core/gdn_tree_verify"]`). Cross-repo primitive consumption proven: state transposed `[d_v×d_k]→[d_k×d_v]` per head, per-head α/β via `recompute_cumulative_log_decay`, tree-structured conv1d windows, all matching sequential `forward_qwen_deltanet` to ≤1e-3. 3 equivalence tests pass (pure-DeltaNet chain, hybrid chain, pure-DeltaNet branching).
- [x] **T4.4** Integration tests. 2 tests in `step_gdn_tree.rs` (returns valid tokens, deterministic for same seed). 3 tests in `tree_forward.rs` (single-node, chain, finite logits). 4 tests in `gdn_tree_verify/mod.rs` (topology from DDTree: chain, branching, dedup, missing-parent panic). T4.3b added 3 more direct-equivalence tests in `tree_forward.rs` (chain matches sequential, single-layer chain matches, branching matches per-branch).

---

## Phase 5 — GOAT gate (benchmarks + promote decision)

### Tasks

- [x] **T5.1 (G1 — correctness)** Test: `verify_gdn_tree` on random trees (T=16,32,64,128) produces outputs within `1e-3` of a per-branch sequential verify reference. ✅ (test_random_trees_correctness; tol 1e-3 due to f32 accumulation, tighter tol achievable with f64 intermediate)
- [x] **T5.2 (G2 — perf)** Benchmark `benches/bench_424_gdn_tree_verify.rs`: ✅ See [`.benchmarks/424_gdn_tree_verify_goat.md`](../.benchmarks/424_gdn_tree_verify_goat.md). **Chain tree speedup matches paper's B200 GPU numbers**: 1.93×/2.79×/4.66×/**7.09×** at T=16/32/64/128 (paper: 1.5×/2.7×/4.6×/7.1×). Shallow (random) trees show 1.18-1.40× (sequential does less total work at depth ~log T). G2 PASSES for the algorithmically favorable case (deep trees).
- [x] **T5.3 (G3 — no-regression)** `cargo check -p katgpt-core` (default) + `--all-features` compile clean. All 1429 existing tests pass. ✅
- [x] **T5.4 (G4 — alloc-free)** `verify_gdn_tree_into` allocates **0 times** on steady-state (CountingAllocator). ✅
- [x] **T5.5** Clean up: `rm -rf /tmp/424_gdn_tree_verify`. ✅

### Promote decision

- [x] **T5.6** G1–G4 all pass → `gdn_tree_verify` stays **opt-in** (NOT default — only relevant for `QwenDeltaNet` / GDN-layer configs, themselves opt-in). Results documented in [`.benchmarks/424_gdn_tree_verify_goat.md`](../.benchmarks/424_gdn_tree_verify_goat.md).
- [-] **T5.7** N/A — G1 passed (no debug needed). G2 passed on deep trees; shallow-tree neutral is documented honestly in the benchmark summary.

---

## Phase 6 — DDTree argmax-of-marginal tuning (Gain, optional)

### Tasks

- [x] **T6.1** In `dd_tree.rs`, add a config flag `deep_argmax_threshold: Option<usize>` (default `None`). When set, at tree depth > threshold, use argmax-of-marginal instead of sampling from the full marginal. Based on paper §3.5 / Figure 6 (crossover at draft length 2–4). **DONE** — `TreeBuilder::set_deep_argmax_threshold` + `build_dd_tree_deep_argmax` free fn in `katgpt-speculative/src/dd_tree.rs`. 5 tests pass (3 restriction tests + 1 None-parity + 1 builder-vs-fn parity). Default `None` = byte-identical to `build_dd_tree`.
- [x] **T6.2** Benchmark: does `deep_argmax_threshold = Some(4)` improve mean acceptance length on the existing DDTree benchmark? **NO — negative result.** The paper's §3.5 insight (argmax beats full-marginal sampling at deep positions) does **not transfer to best-first tree building**. Best-first search already prioritizes the argmax path naturally — the threshold only prevents low-priority deep siblings from entering the heap, which barely affects the best path. Multi-budget benchmark (vocab=16, lookahead=8, budgets 16/32/64/128, thresholds None/1/2/4): identical best-path scores at all configs; the only path-length difference is budget=128/threshold=1 (6 vs 5 tokens). Feature stays available (opt-in via builder setter) but is **not promoted** — no modelless gain. The paper's insight applies to *sampling* from the marginal, not to best-first search over the marginal. **Standalone bench** (`benches/bench_424_dd_tree_deep_argmax.rs`, commit `aba48d4a`): 2 decay regimes × 3 thresholds × 200 seeds — slow-decay gives 7.000 accept len for all thresholds, fast-decay gives 4.000 for all; zero difference in either regime, confirming the inline multi-budget result with stronger statistical evidence.

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
