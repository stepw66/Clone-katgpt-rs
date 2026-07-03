# Plan 370: Manifold Bandits — Latent Task Tree + Hierarchical Thompson Sampler (Open Primitive)

**Date:** 2026-07-03
**Research:** [katgpt-rs/.research/370_manifold_bandits_latent_task_tree_hierarchical_thompson.md](../.research/370_manifold_bandits_latent_task_tree_hierarchical_thompson.md)
**Source paper:** [arXiv:2606.19750](https://arxiv.org/abs/2606.19750) — McKenzie, Hansen, Wang, *Manifold Bandits: Bayesian Curriculum Learning over the Latent Geometry of Large Language Models*, UCSD, 2026
**Code:** [github.com/DarrienMcKenzie/manifold-bandits](https://github.com/DarrienMcKenzie/manifold-bandits) (MIT)
**Target:** `katgpt-rs/crates/katgpt-core/src/manifold_bandit.rs` (new module) + Cargo feature `manifold_bandit`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship a generic, modelless, structure-aware **manifold-structured contextual bandit** that closes the explicitly-documented gap in Plans 030 / 032 / riir-ai 025 ("Contextual bandits — Out of Scope"). The primitive has three composable parts distilled from arXiv:2606.19750:

1. **`LatentTaskTree`** — a frozen, offline-built hierarchical clustering of an arm/query/item space by latent similarity. Construction = extract embeddings → PCA → UMAP (fixed seed) → Chart Test → HDBSCAN → recurse. All modelless, BLAKE3-committable.
2. **`HierarchicalThompsonSampler`** — top-down Beta(α, β) posterior descent through the tree to a leaf. Structure-aware: a reward on one arm updates beliefs on its siblings and ancestors (via Empirical Bayes bottom-up aggregation).
3. **`BayesianFilterArm`** — per-arm non-stationary belief tracking via a predict-update Bayesian Filter (small transition model for drift). Handles the documented Plan 030 gap ("Non-stationary environments — Out of Scope"). Complementary to Dual-Pool CGSP (Plan 312, default-on), which handles non-stationarity via dual-pool mechanism — they compose, not compete.

The **BMC training curriculum** (sampling problems during GSPO/GRPO RL training of Qwen3-8B) is **training-only** and routes to `riir-train` — not in scope for this plan. This plan ships the **inference-time routing primitive** that any runtime caller (NPC curiosity router, shard retrieval, item retrieval, quest selection, meta-policy selection) can consume.

**Routing:** open primitive in `katgpt-core` (generic math, no game/shard/chain semantics). No private guide created this session — the §2.5 DEC-cochain fusion is a TBD Super-GOAT candidate, not committed (per Research §1.5, calling it a "candidate" would trigger mandatory outputs; we are explicitly NOT doing that until a PoC proves the cochain reframing beats a naive tree on compute).

### GOAT gate (G1–G5)

- **G1 — Structural advantage on clustered arms.** On a synthetic structured-domain bandit (e.g., 64 arms arranged in 8 latent clusters of 8, reward = `cluster_mean + per-arm_noise`), hierarchical Thompson reaches ≥90% optimal-arm selection in strictly fewer steps than flat `BanditStrategy::ThompsonSampling` (Plan 030). This is the productivity frontier from Research §1.5.
- **G2 — Diversity preservation.** On the same structured domain, after T=2000 steps, hierarchical Thompson visits ≥1.5× more distinct clusters than flat Thompson at matched cumulative reward. This is the diversity frontier from Research §1.5. (Both G1 and G2 together = the structural advantage.)
- **G3 — Non-stationarity recovery.** On a non-stationary bandit (optimal arm shifts at step T/2), `BayesianFilterArm` recovers to ≥80% optimal-arm selection within K steps of the shift. Compare against flat Thompson (no filter) and Dual-Pool CGSP (Plan 312, dual-pool mechanism) as baselines. **This gate is the modelless-unblock check — if the filter doesn't recover, the non-stationarity handling is broken.**
- **G4 — Latency.** `sample(rng) -> leaf_id` ≤ 500 ns at tree depth ≤ 6 (plasma-tier budget). `observe(leaf_id, reward)` ≤ 300 ns (amortized, includes bottom-up Empirical Bayes update). Zero allocations after tree construction.
- **G5 — Bit-reproducibility.** Two `HierarchicalThompsonSampler` instances with identical `(tree, seed, observations)` produce byte-identical leaf-selection sequences. Required for deterministic-replay / quorum-commitment downstream.

**Demote-on-fail:** if G1 fails (hierarchical Thompson doesn't beat flat Thompson on clustered arms), the structural advantage claim is wrong — downgrade to opt-in Gain-tier, file issue, do not promote. If G3 fails (filter doesn't recover from shift), the non-stationarity claim is broken — keep the tree + flat Thompson variant, drop the filter, file issue. If G4 > 5µs at depth 6, demote (plasma budget blown). If G5 fails, the deterministic-replay story is dead — block promotion.

**No quality-parity claim against the paper** (per Research §3.6 defend-wrong check): we are not claiming "matches BMC's downstream eval gains on DAPO-Math-17k" — those are training-loop results that belong in riir-train. This plan's GOAT claim is *architectural + latency + structural-advantage-on-synthetic* only. If a future plan claims inference-time parity with BMC's productivity × diversity frontier, a PoC at `riir-ai/crates/riir-poc/` becomes mandatory.

---

## Architecture

```
                ┌──────────────────────────────────────────────────┐
   embeddings ─▶│  LatentTaskTree (frozen, BLAKE3-committable)      │
   (Vec<[f32]>) │                                                  │
                │   root: TreeNode                                 │
                │   TreeNode::Internal { children, beta }           │
                │   TreeNode::Leaf { arm_id, filter }               │
                │                                                  │
                │   build(embeddings, config) -> Self              │
                │     - PCA (top-d eigenvectors)                   │
                │     - UMAP (2D, fixed seed)                      │
                │     - Chart Test (manifold locality)             │
                │     - HDBSCAN (density clustering, no k)         │
                │     - recurse on each cluster                    │
                └──────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
   │ sample(rng)        │  │ observe(arm, r)  │  │ blake3_root()       │
   │  -> arm_id         │  │  -> ()           │  │  -> [u8; 32]        │
   │                    │  │                  │  │                     │
   │ Thompson descent:  │  │ Per-arm filter   │  │ Commit the frozen   │
   │ root → sample child│  │ update (predict- │  │ tree topology +     │
   │ via Beta(α,β) →    │  │ update), then    │  │ per-node Beta pri   │
   │ descend → repeat   │  │ Empirical Bayes  │  │ ors for freeze/thaw │
   │ → leaf             │  │ bottom-up agg    │  │ integrity envelope  │
   └────────────────────┘  └──────────────────┘  └─────────────────────┘
```

### Trait / type sketch

```rust
/// A node in the Latent Task Tree.
#[derive(Clone, Debug)]
pub enum TreeNode {
    Internal {
        /// Child subtrees (ordered by cluster id from HDBSCAN).
        children: Vec<TreeNode>,
        /// Beta(α, β) posterior for "which child to descend into".
        /// Updated bottom-up via Empirical Bayes from children.
        beta_alpha: f32,
        beta_beta: f32,
        /// Observed reward count (for Empirical Bayes aggregation).
        n_obs: u32,
    },
    Leaf {
        /// The arm id (index into the original embedding list).
        arm_id: usize,
        /// Non-stationary belief filter for this arm.
        filter: BayesianFilterArm,
    },
}

/// Per-arm non-stationary belief via a predict-update Bayesian Filter.
/// The "predict" step applies a drift model (belief → belief * (1 - drift_rate));
/// the "update" step is a Beta conjugate update on observation.
/// Composes with Dual-Pool CGSP (P312) — they handle different non-stationarity
/// axes (Dual-Pool: pool-level strategy switch; this: per-arm belief drift).
#[derive(Clone, Debug)]
pub struct BayesianFilterArm {
    pub alpha: f32,
    pub beta: f32,
    pub drift_rate: f32,   // λ in (0, 1); 0 = stationary (degrades to flat Thompson)
    pub last_obs_step: u64,
}

impl BayesianFilterArm {
    /// Predict step: decay belief toward uniform (drift).
    /// alpha' = alpha * (1 - λ) + λ       (pulls toward Beta(1,1))
    /// beta'  = beta  * (1 - λ) + λ
    pub fn predict(&mut self, current_step: u64);

    /// Update step: Beta conjugate update on Bernoulli reward.
    /// alpha += r, beta += (1 - r)
    pub fn update(&mut self, reward: f32, current_step: u64);

    /// Thompson sample: draw from Beta(alpha, beta) via Jöhnk's algorithm
    /// (reuse the existing impl from pruners/bandit.rs).
    pub fn thompson_sample(&self, rng: &mut Rng) -> f32;
}

/// Configuration for tree construction.
#[derive(Clone, Debug)]
pub struct LatentTaskTreeConfig {
    pub pca_dim: usize,           // reduce to this dim before UMAP (default: 16)
    pub umap_seed: u64,           // fixed seed for determinism (default: 42)
    pub chart_test_threshold: f32,// manifold locality threshold (default: 0.85)
    pub hdbscan_min_cluster: usize, // min cluster size (default: 4)
    pub max_depth: usize,         // tree depth cap (default: 6)
    pub filter_drift_rate: f32,   // default λ for new BayesianFilterArm (default: 0.01)
}

/// The frozen, BLAKE3-committable Latent Task Tree + its sampler state.
pub struct LatentTaskTree {
    root: TreeNode,
    config: LatentTaskTreeConfig,
    blake3: [u8; 32],  // commitment of (topology + per-node Beta priors)
}

impl LatentTaskTree {
    /// Build the tree offline from a list of embeddings.
    /// Deterministic given (embeddings, config). Modelless — no training.
    pub fn build(embeddings: &[Vec<f32>], config: LatentTaskTreeConfig) -> Self;

    /// Thompson-sample a leaf (arm_id) by descending the tree.
    /// O(depth) — typically 4-6 Beta draws.
    pub fn sample(&self, rng: &mut Rng) -> usize;

    /// Observe a reward on an arm. Updates the leaf's BayesianFilterArm,
    /// then propagates bottom-up via Empirical Bayes (parent Beta = aggregation
    /// of children's observed rewards).
    pub fn observe(&mut self, arm_id: usize, reward: f32, current_step: u64);

    /// BLAKE3 commitment of the frozen tree (topology + Beta priors at build time).
    /// Used for freeze/thaw integrity envelope (composes with riir-neuron-db
    /// MerkleFrozenEnvelope downstream).
    pub fn blake3_root(&self) -> [u8; 32];
}
```

### Why this design

- **Frozen tree, mutable sampler state.** The tree topology (PCA/UMAP/HDBSCAN output) is immutable at inference time — only the Beta posteriors drift. This split makes the tree BLAKE3-committable (topology + initial priors) while the runtime state (current Beta values) is the mutable part. Composes with freeze/thaw: snapshot the current Beta state, thaw into a fresh sampler on the same frozen tree.
- **Beta(α, β) everywhere, sigmoid never softmax.** Per AGENTS.md constraint #2: use sigmoid not softmax. The Beta posterior is conjugate to Bernoulli reward; the Thompson sample is a single Beta draw (no normalization needed — no softmax in sight).
- **Empirical Bayes = deterministic aggregation.** Parent Beta(α_p, β_p) = (Σ α_c, Σ β_c) over children — pure arithmetic, no learned parameters. Gate by R279 N≥d phase transition in Phase 4 (don't aggregate from under-sampled subtrees).
- **Composable with existing bandit stack.** The `HierarchicalThompsonSampler` is a *drop-in alternative* to `BanditPruner<BetaThompson>` for callers whose arm space has latent structure. `MetaRouter` (Plan 196), `GZeroPlayer`, `BanditFrameSampler` (riir-ai combat) can swap in the manifold bandit when a LatentTaskTree is available; flat Thompson stays the default when no tree exists.

---

## Phase 1 — Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/manifold_bandit.rs` with the type sketch above (`TreeNode`, `BayesianFilterArm`, `LatentTaskTreeConfig`, `LatentTaskTree`). No PCA/UMAP/HDBSCAN yet — `build()` accepts a pre-computed tree topology (for testing) and just stamps the Beta priors.
- [ ] **T1.2** Implement `BayesianFilterArm::{predict, update, thompson_sample}`. Reuse Jöhnk's Beta sampler from `pruners/bandit.rs` (DRY — no duplicate RNG code). `predict` with `drift_rate=0` must degenerate to stationary Beta (compatibility with flat Thompson).
- [ ] **T1.3** Implement `LatentTaskTree::sample` (top-down Thompson descent) and `observe` (per-arm filter update + bottom-up Empirical Bayes propagation). O(depth) per sample, O(depth) per observe.
- [ ] **T1.4** Add the `manifold_bandit` feature flag to `katgpt-core/Cargo.toml` (opt-in, `[]` deps — no new external crates yet; PCA/UMAP/HDBSCAN land in Phase 3).
- [ ] **T1.5** Unit tests: (a) `sample` on a hand-built 3-level tree returns valid leaf ids; (b) `observe` updates the correct leaf's filter and propagates to parent Beta via Empirical Bayes; (c) `drift_rate=0` matches flat Thompson sample distribution (statistical test over 10K samples); (d) `blake3_root` is stable across rebuilds with identical input.
- [ ] **T1.6** `cargo test -p katgpt-core --features manifold_bandit --lib` passes. Zero regressions on default features (`cargo test -p katgpt-core --lib` unchanged).

---

## Phase 2 — GOAT Gate Benchmark (G1, G2, G3, G4, G5)

### Tasks

- [ ] **T2.1** **G1 — Structural advantage benchmark.** Synthetic domain: 64 arms in 8 latent clusters of 8. Reward = `cluster_mean[k] + arm_noise[i]`, where `cluster_mean ~ Uniform(0.2, 0.8)` and `arm_noise ~ Normal(0, 0.05)`. Run hierarchical Thompson vs flat Thompson (Plan 030) for T=5000 steps, 200 trials. Metric: steps-to-90%-optimal-arm-selection. **PASS gate: hierarchical reaches 90% in ≤0.8× the steps of flat.** Record in `.benchmarks/370_manifold_bandit_goat.md`.
- [ ] **T2.2** **G2 — Diversity benchmark.** Same domain. After T=2000 steps, count distinct clusters visited (max 8). **PASS gate: hierarchical visits ≥1.5× the clusters flat visits at matched cumulative reward (±5%).**
- [ ] **T2.3** **G3 — Non-stationarity recovery benchmark.** 16-arm bandit, optimal arm shifts from arm 0 to arm 5 at step T/2=1000. Compare: (a) flat Thompson (no filter), (b) hierarchical Thompson with `BayesianFilterArm` (drift_rate=0.05), (c) Dual-Pool CGSP (Plan 312) as baseline. Metric: steps-to-80%-optimal-after-shift. **PASS gate: hierarchical-with-filter recovers in ≤0.5× the steps of flat Thompson.** (No requirement to beat Dual-Pool — they're complementary; record the comparison honestly.)
- [ ] **T2.4** **G4 — Latency benchmark.** `criterion` bench: `sample` at depth 6 (p50 ≤ 500 ns), `observe` at depth 6 (p50 ≤ 300 ns). Zero allocations after `build` (verify with a custom allocator counter in the bench). Use `CARGO_TARGET_DIR=/tmp/manifold_bandit_370` per AGENTS.md rule; clean up when done.
- [ ] **T2.5** **G5 — Bit-reproducibility test.** Two `LatentTaskTree` instances from identical `(embeddings, config, seed)`, run identical `(arm_id, reward, step)` observation sequences → byte-identical leaf-selection sequences over 10K samples. Required for deterministic-replay / quorum-commitment downstream.
- [ ] **T2.6** **GOAT gate verdict.** If G1+G2+G3+G4+G5 all PASS → promote `manifold_bandit` to default-on in `katgpt-core` for structure-aware callers (document that flat Thompson stays default for unstructured callers — no forced demote). If G1 fails → demote to opt-in Gain-tier, file `.issues/` for the structural-advantage claim. If G3 fails → keep tree + flat Thompson variant, drop the filter, file issue. Record verdict in `.benchmarks/370_manifold_bandit_goat.md`.

---

## Phase 3 — Real Tree Construction (PCA + UMAP + Chart Test + HDBSCAN)

### Tasks

- [ ] **T3.1** Implement PCA (top-d eigenvectors via power iteration or Jacobi — no external LAPACK dep; we already have SIMD matmul in `katgpt-core/src/simd.rs`). Deterministic given fixed seed.
- [ ] **T3.2** Implement UMAP (2D embedding, fixed seed). This is the heaviest piece — evaluate: (a) full UMAP port (significant code), (b) a simpler deterministic alternative (e.g., t-SNE-style gradient descent with fixed iterations, or a spectral embedding). **Decision deferred to Phase 3 — the sampler doesn't care which 2D embedding is used, only that it's deterministic and preserves local neighborhoods.**
- [ ] **T3.3** Implement Chart Test (manifold locality check). For each point, check if its k-nearest neighbors lie approximately on a linear subspace (PCA on the local neighborhood, check residual / total variance ratio < threshold). Points failing are marked as noise.
- [ ] **T3.4** Implement HDBSCAN (density-based hierarchical clustering, no k). This is non-trivial — evaluate: (a) full HDBSCAN port, (b) a simpler density-based alternative (DBSCAN with adaptive ε, or a mutual-reachability-graph MST cut). **Decision deferred to Phase 3 — the sampler needs *some* hierarchical clustering, not specifically HDBSCAN.**
- [ ] **T3.5** Wire `build()` to use the real construction pipeline. Add integration test: build a tree from 128 synthetic embeddings (8 Gaussian clusters in 16-dim space), verify the tree has 8 top-level clusters. BLAKE3 commitment stable across rebuilds.
- [ ] **T3.6** Re-run G1–G5 with real-constructed trees (not hand-built). The structural advantage should be *stronger* with real manifold structure.

---

## Phase 4 — Fusion Exploration (TBD Super-GOAT candidate, NOT committed)

This phase is **exploratory only**. Per Research §2.5, the DEC-cochain fusion (tree ≡ cell complex, Thompson ≡ pushforward, Empirical Bayes ≡ pullback) is a TBD Super-GOAT candidate. It only becomes a committed plan if a PoC proves the cochain reframing beats a naive tree on compute.

### Tasks

- [ ] **T4.1** PoC at `riir-ai/crates/riir-poc/benches/manifold_thompson_cochain.rs`: implement the same `HierarchicalThompsonSampler` using DEC operators (`katgpt-core/src/dec/`) — belief as a `CochainField` on the tree cell complex, sample = pushforward, observe = pullback. Compare latency + correctness against the naive-tree impl from Phase 1.
- [ ] **T4.2** Add the R279 N≥d phase gate to Empirical Bayes: a subtree with `n_obs < intrinsic_dim` should NOT propagate belief (its posterior is below the phase transition — it's noise). Benchmark whether this improves G1 (structural advantage) or hurts it (under-aggregation).
- [ ] **T4.3** **Verdict.** If the cochain impl matches naive-tree latency AND the N≥d gate improves G1 → this is a Super-GOAT candidate. Create the riir-ai private guide (`riir-ai/.research/NNN_manifold_thompson_cochain_router_guide.md`) per Research §1.5 mandatory outputs, re-gate as Super-GOAT. If cochain is slower or N≥d hurts → drop Phase 4, the GOAT from Phase 2 stands.

---

## Out of Scope

- **BMC training curriculum** (GSPO/GRPO RL on Qwen3-8B) — training-only, routes to `riir-train`. Not in this plan.
- **BMC-T target-driven utility** (train/target overlap precomputation) — useful but secondary; defer to a follow-up plan if a runtime caller needs eval-relevance steering.
- **Integration with specific runtime callers** (MetaRouter, GZeroPlayer, BanditFrameSampler, ShardIndex retrieval, ItemEmbedIndex retrieval) — these are consumer plans in riir-ai / riir-neuron-db, not this plan. This plan ships the *primitive*; consumers wire it.
- **Full UMAP / HDBSCAN ports** if a simpler deterministic alternative suffices (Phase 3 decision).
- **LatCal commitment of the Beta posteriors across the sync boundary** — that's a riir-chain bridge plan, not this plan. The `blake3_root()` here is the local commitment; the cross-quorum story is downstream.

---

## TL;DR

Ship a modelless, structure-aware **manifold-structured contextual bandit** (`LatentTaskTree` + `HierarchicalThompsonSampler` + `BayesianFilterArm`) in `katgpt-core` behind `manifold_bandit` feature flag. Closes the explicitly-documented contextual/non-stationary bandit gap (Plans 030/032/025). GOAT gate: G1 structural advantage on clustered arms (hierarchical beats flat Thompson), G2 diversity preservation, G3 non-stationarity recovery (Bayesian filter), G4 sub-µs latency, G5 bit-reproducibility. Phase 1 skeleton (hand-built tree), Phase 2 GOAT gate, Phase 3 real PCA/UMAP/HDBSCAN construction, Phase 4 exploratory DEC-cochain fusion (TBD Super-GOAT — not committed, needs PoC). Training curriculum → riir-train (one-line redirect, no files created). No quality-parity claim against the paper's training-loop results — this plan's claim is architectural + latency + structural-advantage-on-synthetic only, per Research §3.6 defend-wrong check.
