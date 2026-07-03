//! manifold_bandit — Latent Task Tree + Hierarchical Thompson Sampler.
//!
//! Distilled from McKenzie, Hansen, Wang, *Manifold Bandits: Bayesian Curriculum
//! Learning over the Latent Geometry of Large Language Models* (arXiv:2606.19750,
//! UCSD 2026). The paper's headline (BMC training curriculum for GSPO/GRPO RL on
//! Qwen3-8B) is training-only → `riir-train`. This module ships the **modelless
//! inference-time routing primitive**: a frozen, BLAKE3-committable hierarchical
//! clustering of an arm space + top-down Beta posterior descent + per-arm
//! non-stationary Bayesian filtering.
//!
//! # Three composable parts
//!
//! 1. **[`LatentTaskTree`]** — frozen hierarchical clustering of arms by latent
//!    similarity. Phase 1 accepts a pre-computed topology ([`TreeNode`]); the real
//!    PCA → UMAP → Chart Test → HDBSCAN construction pipeline ships in Phase 3.
//! 2. **Top-down Thompson descent** — [`LatentTaskTree::sample`] descends root →
//!    leaf by Thompson-sampling each child's Beta and picking the max. O(depth)
//!    Beta draws. Structure-aware: reward on one arm updates siblings + ancestors
//!    via bottom-up Empirical Bayes.
//! 3. **[`BayesianFilterArm`]** — per-arm non-stationary belief via a predict-
//!    update filter (drift toward uniform between observations). Handles the
//!    documented Plan 030 gap ("Non-stationary environments — Out of Scope").
//!    Complementary to Dual-Pool CGSP (Plan 312) — they handle different axes of
//!    non-stationarity.
//!
//! # Design notes
//!
//! - **Beta(α, β) everywhere, sigmoid never softmax** (AGENTS.md §2). The Beta
//!   posterior is conjugate to Bernoulli reward; the Thompson sample is a single
//!   Beta draw — no normalization, no softmax.
//! - **Empirical Bayes = evidence pooling.** Parent (α, β) = (1 + Σ(α_c − 1),
//!   1 + Σ(β_c − 1)) over children — pools observed successes/failures without
//!   pseudo-count dilution. Starts at Beta(1, 1) when all children are uniform
//!   (high variance → explores); concentrates correctly with data (sharper
//!   than SUM which dilutes signal with per-child pseudo-counts). See Plan 370
//!   Phase 2 GOAT gate.
//! - **Frozen tree, mutable sampler state.** Topology is immutable at inference
//!   time; only Beta posteriors drift. The tree is BLAKE3-committable at build
//!   time (topology + initial priors); runtime mutations are tracked separately.
//! - **Zero allocations after construction.** [`LatentTaskTree::sample`] and
//!   [`LatentTaskTree::observe`] are allocation-free. The arm→path lookup uses a
//!   stack-allocated [`ArmPath`] (Copy) to avoid the borrow-checker conflict
//!   between reading the path and mutating the tree.
//!
//! # DRY note
//!
//! The Beta sampler ([`sample_beta`]) is a private helper using the Gamma-ratio
//! method (Marsaglia-Tsang gamma + Box-Muller normal). The same algorithm is used
//! in `katgpt-rs/src/dense_mesh/edge_bandit.rs` (with a rough normal
//! approximation). Other copies (`katgpt-pruners/src/bandit.rs`,
//! `katgpt-rs/src/fold/fold_bandit.rs`, `katgpt-rs/src/speculative/
//! thinking_controller.rs`, `katgpt-rs/tests/partial_scoring_goat.rs`) use
//! Jöhnk's algorithm, which is correct but has catastrophically low acceptance
//! rates for large α, β (the posteriors after many observations). This module
//! uses the Gamma-ratio method for correctness across the full parameter range.
//! Consolidating all copies into a shared `katgpt-core::rng_util` is a worthwhile
//! follow-up refactor (tracked separately).
//!
//! See: `katgpt-rs/.research/370_manifold_bandits_latent_task_tree_hierarchical_thompson.md`
//! See: `katgpt-rs/.plans/370_manifold_bandit_latent_task_tree.md`

// ──────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────

/// Maximum supported tree depth. Bounds the stack-allocated path buffer so that
/// [`LatentTaskTree::observe`] can copy the path without allocating.
///
/// 16 is generous — the Phase 3 construction pipeline caps at `max_depth = 6`,
/// and deeper trees blow the ≤500 ns latency budget (G4) anyway.
const MAX_DEPTH: usize = 16;

// ──────────────────────────────────────────────────────────────────────────
// ArmPath — zero-alloc path lookup for observe
// ──────────────────────────────────────────────────────────────────────────

/// Pre-computed root→leaf path as a stack-allocated Copy struct.
///
/// Indexed by arm_id at construction time. [`LatentTaskTree::observe`] copies the
/// path out of the lookup table (no borrow held), then descends the tree with
/// `&mut` access — this is the zero-alloc solution to the borrow-checker conflict
/// between reading `&self.arm_paths[arm_id]` and mutating `&mut self.root`.
#[derive(Clone, Copy, Debug)]
struct ArmPath {
    /// Child indices from root toward the leaf.
    indices: [usize; MAX_DEPTH],
    /// Number of valid entries in `indices` (= tree depth to this leaf).
    len: usize,
}

impl ArmPath {
    fn empty() -> Self {
        Self {
            indices: [0; MAX_DEPTH],
            len: 0,
        }
    }

    fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len]
    }
}

// ──────────────────────────────────────────────────────────────────────────
// TreeNode
// ──────────────────────────────────────────────────────────────────────────

/// A node in the Latent Task Tree.
///
/// Internal nodes carry the aggregate Empirical Bayes Beta(α, β) posterior for
/// their subtree; leaves carry the per-arm non-stationary [`BayesianFilterArm`].
#[derive(Clone, Debug)]
pub enum TreeNode {
    /// A branching node. `beta_alpha` / `beta_beta` are the Empirical Bayes
    /// aggregate of all children's Beta parameters (mean of α_c, mean of β_c),
    /// recomputed bottom-up after each observation.
    Internal {
        /// Child subtrees (ordered by cluster id from the construction pipeline).
        children: Vec<TreeNode>,
        /// Beta(α) for "how good is this subtree" — the aggregate of children.
        beta_alpha: f32,
        /// Beta(β) companion.
        beta_beta: f32,
        /// Total observations that have passed through this node.
        n_obs: u32,
    },
    /// A terminal node — one arm in the original arm space.
    Leaf {
        /// The arm id (index into the original embedding list).
        arm_id: usize,
        /// Non-stationary belief filter for this arm.
        filter: BayesianFilterArm,
    },
}

impl TreeNode {
    /// Returns the (alpha, beta) Beta parameters for this node.
    ///
    /// - Internal: the aggregate Empirical Bayes posterior.
    /// - Leaf: the per-arm [`BayesianFilterArm`] posterior.
    fn beta_params(&self) -> (f32, f32) {
        match self {
            TreeNode::Internal {
                beta_alpha,
                beta_beta,
                ..
            } => (*beta_alpha, *beta_beta),
            TreeNode::Leaf { filter, .. } => (filter.alpha, filter.beta),
        }
    }

    /// Convenience: construct a leaf with a fresh uniform-prior filter.
    pub fn leaf(arm_id: usize, drift_rate: f32) -> Self {
        TreeNode::Leaf {
            arm_id,
            filter: BayesianFilterArm::new(drift_rate),
        }
    }

    /// Convenience: construct an internal node whose Empirical Bayes aggregate
    /// pools the **evidence** (observed successes/failures) from children:
    /// `parent_α = 1 + Σ(child_α - 1)`, `parent_β = 1 + Σ(child_β - 1)`.
    ///
    /// This is the standard Beta-Bernoulli evidence pooling: each child's
    /// pseudocount (the +1 prior) is subtracted before summing, then a single
    /// +1 is added back. The result starts at Beta(1, 1) when all children are
    /// uniform (high variance → explores), and concentrates faster than either
    /// SUM (Beta(N, N) — over-concentrated initially, signal diluted by
    /// pseudo-counts) or MEAN (never concentrates — too diffuse). See Plan 370
    /// Phase 2 GOAT gate.
    pub fn internal(children: Vec<TreeNode>) -> Self {
        let n = children.len() as f32;
        let (a, b) = children
            .iter()
            .fold((0.0_f32, 0.0_f32), |(sa, sb), c| {
                let (ca, cb) = c.beta_params();
                (sa + ca, sb + cb)
            });
        // Evidence pooling: subtract per-child pseudocount, add back one.
        TreeNode::Internal {
            children,
            beta_alpha: (a - n + 1.0).max(1.0),
            beta_beta: (b - n + 1.0).max(1.0),
            n_obs: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// BayesianFilterArm
// ──────────────────────────────────────────────────────────────────────────

/// Per-arm non-stationary belief via a predict-update Bayesian Filter.
///
/// The "predict" step decays the Beta posterior toward uniform Beta(1, 1)
/// proportional to elapsed steps since the last observation — modeling belief
/// drift in a non-stationary environment. The "update" step applies the standard
/// Beta-Bernoulli conjugate update on the observed reward.
///
/// With `drift_rate = 0.0`, `predict` is a no-op and the arm degenerates to a
/// stationary Beta posterior (identical to flat Thompson from Plan 030).
///
/// Composes with Dual-Pool CGSP (Plan 312): Dual-Pool handles pool-level strategy
/// switches; this handles per-arm belief drift. They operate on different axes
/// of non-stationarity.
#[derive(Clone, Debug)]
pub struct BayesianFilterArm {
    /// Beta(α) — successes + pseudocount. Starts at 1.0 (uniform prior).
    pub alpha: f32,
    /// Beta(β) — failures + pseudocount. Starts at 1.0 (uniform prior).
    pub beta: f32,
    /// Drift rate λ ∈ [0, 1). 0 = stationary (no drift). Higher = faster forgetting.
    pub drift_rate: f32,
    /// The step at which the filter was last touched (predict or update).
    pub last_obs_step: u64,
}

impl BayesianFilterArm {
    /// Create a fresh arm with uniform Beta(1, 1) prior and the given drift rate.
    pub fn new(drift_rate: f32) -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            drift_rate: drift_rate.clamp(0.0, 0.999),
            last_obs_step: 0,
        }
    }

    /// Predict step: decay belief toward uniform Beta(1, 1).
    ///
    /// Applies `elapsed` steps of geometric decay:
    /// ```text
    /// decay = (1 - λ)^elapsed
    /// α' = α · decay + (1 - decay)     ← pulls toward 1
    /// β' = β · decay + (1 - decay)
    /// ```
    /// With `drift_rate = 0` or `elapsed = 0`, this is a no-op.
    pub fn predict(&mut self, current_step: u64) {
        let elapsed = current_step.saturating_sub(self.last_obs_step);
        if elapsed == 0 || self.drift_rate <= 0.0 {
            return;
        }
        let decay = (1.0 - self.drift_rate).powi(elapsed as i32);
        self.alpha = self.alpha * decay + (1.0 - decay);
        self.beta = self.beta * decay + (1.0 - decay);
        self.last_obs_step = current_step;
    }

    /// Update step: Beta-Bernoulli conjugate update.
    ///
    /// `reward` is clamped to `[0, 1]`: `α += r`, `β += (1 - r)`.
    /// Updates `last_obs_step` so the next `predict` measures drift from here.
    pub fn update(&mut self, reward: f32, current_step: u64) {
        let r = reward.clamp(0.0, 1.0);
        self.alpha += r;
        self.beta += 1.0 - r;
        self.last_obs_step = current_step;
    }

    /// Thompson sample: draw from Beta(α, β) via Jöhnk's algorithm.
    pub fn thompson_sample(&self, rng: &mut fastrand::Rng) -> f32 {
        sample_beta(self.alpha, self.beta, rng)
    }

    /// Posterior mean `α / (α + β)` — useful for diagnostics, not the hot path.
    pub fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }
}

impl Default for BayesianFilterArm {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// LatentTaskTreeConfig
// ──────────────────────────────────────────────────────────────────────────

/// Configuration for [`LatentTaskTree`] construction.
///
/// Fields `pca_dim` through `hdbscan_min_cluster` configure the Phase 3
/// construction pipeline (PCA → UMAP → Chart Test → HDBSCAN). In Phase 1 they are
/// stored but unused — the tree topology is supplied directly via
/// [`LatentTaskTree::from_root`].
#[derive(Clone, Debug)]
pub struct LatentTaskTreeConfig {
    /// Reduce embeddings to this dimensionality before UMAP (Phase 3). Default: 16.
    pub pca_dim: usize,
    /// Fixed seed for UMAP determinism (Phase 3). Default: 42.
    pub umap_seed: u64,
    /// Manifold locality threshold for the Chart Test (Phase 3). Default: 0.85.
    pub chart_test_threshold: f32,
    /// Minimum cluster size for HDBSCAN (Phase 3). Default: 4.
    pub hdbscan_min_cluster: usize,
    /// Tree depth cap. Default: 6 (keeps sample/observe within the G4 latency
    /// budget; deeper trees blow ≤500 ns per sample).
    pub max_depth: usize,
    /// Default drift rate λ for new [`BayesianFilterArm`] instances. Default: 0.01.
    pub filter_drift_rate: f32,
}

impl Default for LatentTaskTreeConfig {
    fn default() -> Self {
        Self {
            pca_dim: 16,
            umap_seed: 42,
            chart_test_threshold: 0.85,
            hdbscan_min_cluster: 4,
            max_depth: 6,
            filter_drift_rate: 0.01,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// LatentTaskTree
// ──────────────────────────────────────────────────────────────────────────

/// The frozen, BLAKE3-committable Latent Task Tree + its sampler state.
///
/// The tree topology (the [`TreeNode`] structure) is immutable at inference time
/// — only the Beta posteriors drift. The [`blake3_root`](Self::blake3_root)
/// commits the **initial** topology + priors (computed once at construction);
/// runtime mutations are the mutable sampler state, tracked separately for
/// freeze/thaw snapshots.
pub struct LatentTaskTree {
    root: TreeNode,
    config: LatentTaskTreeConfig,
    blake3: [u8; 32],
    /// arm_id → root-to-leaf path. Built once at construction; zero-alloc reads.
    arm_paths: Vec<ArmPath>,
}

impl LatentTaskTree {
    /// Construct from a pre-built tree topology (Phase 1 entry point).
    ///
    /// Stamps uniform Beta(1, 1) priors (via [`BayesianFilterArm::new`] on each
    /// leaf), builds the arm→path lookup, and computes the BLAKE3 commitment of
    /// the initial tree state.
    ///
    /// The full `build(embeddings, config)` pipeline (PCA → UMAP → Chart Test →
    /// HDBSCAN) ships in Phase 3.
    pub fn from_root(root: TreeNode, config: LatentTaskTreeConfig) -> Self {
        // Collect arm paths for O(1) observe lookup.
        let mut arm_paths: Vec<ArmPath> = Vec::new();
        let mut current_path: Vec<usize> = Vec::with_capacity(MAX_DEPTH);
        Self::collect_arm_paths(&root, &mut current_path, &mut arm_paths);

        // Compute BLAKE3 of the initial tree (topology + initial Beta priors).
        let mut hasher = blake3::Hasher::new();
        Self::compute_blake3(&root, &mut hasher);
        let blake3 = *hasher.finalize().as_bytes();

        Self {
            root,
            config,
            blake3,
            arm_paths,
        }
    }

    /// Thompson-sample a leaf (arm_id) by descending the tree.
    ///
    /// At each internal node, draws a Beta sample from each child's posterior and
    /// descends into the child with the highest sample. O(branching × depth) Beta
    /// draws — typically 4–6 levels × 4–8 children.
    ///
    /// Zero allocations.
    pub fn sample(&self, rng: &mut fastrand::Rng) -> usize {
        let mut node = &self.root;
        loop {
            match node {
                TreeNode::Leaf { arm_id, .. } => return *arm_id,
                TreeNode::Internal { children, .. } => {
                    if children.is_empty() {
                        // Degenerate internal node with no children — should not
                        // happen in a well-formed tree, but avoid an infinite loop.
                        panic!("manifold_bandit: Internal node with no children");
                    }
                    let mut best_idx = 0usize;
                    let mut best_sample = f32::NEG_INFINITY;
                    for (i, child) in children.iter().enumerate() {
                        let (a, b) = child.beta_params();
                        let s = sample_beta(a, b, rng);
                        if s > best_sample {
                            best_sample = s;
                            best_idx = i;
                        }
                    }
                    node = &children[best_idx];
                }
            }
        }
    }

    /// Observe a reward on an arm.
    ///
    /// Descends to the leaf for `arm_id`, runs the predict-update Bayesian filter,
    /// then propagates bottom-up via Empirical Bayes (parent α, β = mean of
    /// children's α, β). O(branching × depth) — the Empirical Bayes recompute
    /// iterates all children at each level.
    ///
    /// Zero allocations (the path is copied from the pre-computed lookup as a
    /// stack-local [`ArmPath`]).
    ///
    /// # Panics
    /// Panics if `arm_id` is not present in the tree.
    pub fn observe(&mut self, arm_id: usize, reward: f32, current_step: u64) {
        assert!(
            arm_id < self.arm_paths.len(),
            "manifold_bandit: arm_id {arm_id} not in tree ({} arms)",
            self.arm_paths.len()
        );
        // Copy the path (no borrow held on self after this line).
        let path = self.arm_paths[arm_id];
        Self::observe_recursive(&mut self.root, path.as_slice(), reward, current_step);
    }

    /// BLAKE3 commitment of the frozen tree (topology + initial Beta priors at
    /// construction time).
    ///
    /// This is the **initial** commitment — it does NOT reflect runtime Beta
    /// mutations. Use it for freeze/thaw integrity envelopes: snapshot the current
    /// Beta state separately, and verify the frozen topology matches on thaw.
    pub fn blake3_root(&self) -> [u8; 32] {
        self.blake3
    }

    /// Number of arms (leaves) in the tree.
    pub fn num_arms(&self) -> usize {
        self.arm_paths.len()
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &LatentTaskTreeConfig {
        &self.config
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Recursively collect root→leaf paths for every arm_id.
    fn collect_arm_paths(
        node: &TreeNode,
        current_path: &mut Vec<usize>,
        paths: &mut Vec<ArmPath>,
    ) {
        match node {
            TreeNode::Leaf { arm_id, .. } => {
                while paths.len() <= *arm_id {
                    paths.push(ArmPath::empty());
                }
                let mut buf = [0usize; MAX_DEPTH];
                for (i, &idx) in current_path.iter().enumerate() {
                    buf[i] = idx;
                }
                paths[*arm_id] = ArmPath {
                    indices: buf,
                    len: current_path.len(),
                };
            }
            TreeNode::Internal { children, .. } => {
                for (i, child) in children.iter().enumerate() {
                    current_path.push(i);
                    Self::collect_arm_paths(child, current_path, paths);
                    current_path.pop();
                }
            }
        }
    }

    /// Recursively descend following `path`, update the leaf filter, then
    /// recompute each ancestor's Empirical Bayes aggregate on the way back up.
    fn observe_recursive(node: &mut TreeNode, path: &[usize], reward: f32, step: u64) {
        match node {
            TreeNode::Leaf { filter, .. } => {
                filter.predict(step);
                filter.update(reward, step);
            }
            TreeNode::Internal {
                children,
                beta_alpha,
                beta_beta,
                n_obs,
            } => {
                let child_idx = path[0];
                Self::observe_recursive(&mut children[child_idx], &path[1..], reward, step);
                // Recompute this node's aggregate via EVIDENCE pooling: subtract
                // each child's pseudocount before summing, add back one. This
                // gives the parent the total observed evidence (successes /
                // failures) without dilution from unobserved children.
                let n = children.len() as f32;
                let (a, b) = children
                    .iter()
                    .fold((0.0_f32, 0.0_f32), |(sa, sb), c| {
                        let (ca, cb) = c.beta_params();
                        (sa + ca, sb + cb)
                    });
                *beta_alpha = (a - n + 1.0).max(1.0);
                *beta_beta = (b - n + 1.0).max(1.0);
                *n_obs += 1;
            }
        }
    }

    /// Recursively hash the tree topology + initial Beta priors into `hasher`.
    ///
    /// Canonical pre-order traversal: Internal nodes tagged `0x01`, leaves tagged
    /// `0x02`. All numeric fields in little-endian bytes for cross-platform
    /// determinism.
    fn compute_blake3(node: &TreeNode, hasher: &mut blake3::Hasher) {
        match node {
            TreeNode::Leaf { arm_id, filter } => {
                hasher.update(&[0x02]);
                hasher.update(&arm_id.to_le_bytes());
                hasher.update(&filter.alpha.to_le_bytes());
                hasher.update(&filter.beta.to_le_bytes());
                hasher.update(&filter.drift_rate.to_le_bytes());
            }
            TreeNode::Internal {
                children,
                beta_alpha,
                beta_beta,
                n_obs,
            } => {
                hasher.update(&[0x01]);
                hasher.update(&beta_alpha.to_le_bytes());
                hasher.update(&beta_beta.to_le_bytes());
                hasher.update(&n_obs.to_le_bytes());
                hasher.update(&(children.len() as u32).to_le_bytes());
                for child in children.iter() {
                    Self::compute_blake3(child, hasher);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Beta sampling — Gamma-ratio method (Marsaglia-Tsang + Box-Muller)
// ─────────────────────────────────────────────────────────────────────

/// Sample from Beta(α, β) via the Gamma-ratio identity:
/// `Beta(α,β) = Gamma(α,1) / (Gamma(α,1) + Gamma(β,1))`.
///
/// Correct for all α, β > 0. Our regime is α, β ≥ 1 (the +1 pseudocount
/// guarantees this), so we use Marsaglia-Tsang for shape ≥ 1.
///
/// **Why not Jöhnk's:** the other copies in this codebase use Jöhnk's algorithm,
/// which has acceptance rate Γ(α+1)Γ(β+1)/Γ(α+β+1) — this drops to ~0.001% for
/// the posteriors that arise after 10+ observations (e.g. Beta(16, 6)). The
/// Gamma-ratio method has >90% acceptance per iteration regardless of α, β.
///
/// See the module-level DRY note — this is a private copy because `katgpt-core`
/// cannot import from `katgpt-pruners` (wrong dependency direction).
fn sample_beta(alpha: f32, beta: f32, rng: &mut fastrand::Rng) -> f32 {
    // Uniform prior Beta(1, 1): just return a uniform draw.
    if (alpha - 1.0).abs() < f32::EPSILON && (beta - 1.0).abs() < f32::EPSILON {
        return rng.f32();
    }
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    x / (x + y)
}

/// Sample from Gamma(shape, scale=1) via the Marsaglia-Tsang squeeze method.
///
/// For `shape ≥ 1` the acceptance rate is >90% per iteration (typically 1–2
/// iterations). For `shape < 1` (should not arise with our +1 pseudocount, but
/// handled for correctness), the boost trick `gamma(s) = gamma(s+1) · U^(1/s)`
/// is applied.
fn sample_gamma(shape: f32, rng: &mut fastrand::Rng) -> f32 {
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, rng);
        let u = rng.f32().max(f32::EPSILON);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.f32().max(f32::EPSILON);
        // Squeeze (fast acceptance).
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v3;
        }
        // Full acceptance check.
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

/// Standard normal variate via Box-Muller transform.
///
/// Returns `N(0, 1)` using two uniform draws. Uses `cos` (discards the `sin`
/// component — a future micro-optimization could cache the second variate).
#[inline]
fn standard_normal(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(f32::EPSILON);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * std::f32::consts::PI * u2).cos()
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small 3-level test tree:
    /// ```text
    ///                 root
    ///                /    \
    ///           [0,1]      [2,3]
    ///          /   \      /   \
    ///        L0    L1   L2    L3
    /// ```
    fn build_test_tree(drift_rate: f32) -> LatentTaskTree {
        let left = TreeNode::internal(vec![
            TreeNode::leaf(0, drift_rate),
            TreeNode::leaf(1, drift_rate),
        ]);
        let right = TreeNode::internal(vec![
            TreeNode::leaf(2, drift_rate),
            TreeNode::leaf(3, drift_rate),
        ]);
        let root = TreeNode::internal(vec![left, right]);
        LatentTaskTree::from_root(root, LatentTaskTreeConfig::default())
    }

    // ── T1.5(a): sample returns valid leaf ids ──────────────────────────

    #[test]
    fn test_sample_returns_valid_leaf_ids() {
        let tree = build_test_tree(0.0);
        let mut rng = fastrand::Rng::with_seed(42);
        let valid_arms: std::collections::HashSet<usize> = (0..4).collect();

        for _ in 0..1000 {
            let arm = tree.sample(&mut rng);
            assert!(
                valid_arms.contains(&arm),
                "sample returned invalid arm_id {arm}"
            );
        }
    }

    #[test]
    fn test_sample_single_leaf_tree() {
        // Degenerate tree: root is a single leaf.
        let root = TreeNode::leaf(7, 0.0);
        let tree = LatentTaskTree::from_root(root, LatentTaskTreeConfig::default());
        let mut rng = fastrand::Rng::with_seed(0);
        for _ in 0..100 {
            assert_eq!(tree.sample(&mut rng), 7);
        }
    }

    #[test]
    fn test_sample_visits_all_arms_with_uniform_prior() {
        // With all-uniform Beta(1,1) priors, every arm should be visited.
        let tree = build_test_tree(0.0);
        let mut rng = fastrand::Rng::with_seed(123);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            seen.insert(tree.sample(&mut rng));
        }
        assert_eq!(seen.len(), 4, "uniform prior should visit all 4 arms");
    }

    // ── T1.5(b): observe updates leaf filter + Empirical Bayes ──────────

    #[test]
    fn test_observe_updates_leaf_filter() {
        let mut tree = build_test_tree(0.0);

        // Observe 10 successes on arm 2.
        for step in 0..10u64 {
            tree.observe(2, 1.0, step);
        }

        // Find arm 2's leaf and check its filter.
        let leaf = find_leaf(&tree.root, 2).expect("arm 2 should exist");
        let filter = match leaf {
            TreeNode::Leaf { filter, .. } => filter,
            _ => panic!("expected leaf"),
        };
        // 10 successes: alpha = 1 + 10 = 11, beta = 1 + 0 = 1.
        assert!(
            (filter.alpha - 11.0).abs() < 1e-5,
            "alpha should be 11 after 10 successes, got {}",
            filter.alpha
        );
        assert!(
            (filter.beta - 1.0).abs() < 1e-5,
            "beta should be 1 after 10 successes, got {}",
            filter.beta
        );
    }

    #[test]
    fn test_observe_propagates_empirical_bayes() {
        let mut tree = build_test_tree(0.0);

        // Before any observation: root has Beta(1, 1) — evidence pooling of 4
        // children × Beta(1,1): 1 + Σ(1-1) = 1 for both α and β.
        let (ra, rb) = tree.root.beta_params();
        assert!((ra - 1.0).abs() < 1e-5, "root alpha should start at 1, got {ra}");
        assert!((rb - 1.0).abs() < 1e-5, "root beta should start at 1, got {rb}");

        // Observe 1 success on arm 0.
        tree.observe(0, 1.0, 0);

        // Arm 0's leaf: Beta(2, 1). Its sibling (arm 1): Beta(1, 1).
        // Left internal: 1+Σ(α-1)=1+(1+0)=2, 1+Σ(β-1)=1+(0+0)=1 → Beta(2, 1).
        // Right internal: 1+Σ(α-1)=1+(0+0)=1, 1+Σ(β-1)=1+(0+0)=1 → Beta(1, 1).
        // Root: 1+Σ(α-1)=1+(1+0)=2, 1+Σ(β-1)=1+(0+0)=1 → Beta(2, 1).
        let (ra, rb) = tree.root.beta_params();
        assert!(
            (ra - 2.0).abs() < 1e-5,
            "root alpha should be 2 after 1 success, got {ra}"
        );
        assert!(
            (rb - 1.0).abs() < 1e-5,
            "root beta should be 1 after 1 success, got {rb}"
        );
    }

    #[test]
    fn test_observe_mixed_rewards() {
        let mut tree = build_test_tree(0.0);
        tree.observe(1, 0.7, 0);
        tree.observe(1, 0.3, 1);

        let leaf = find_leaf(&tree.root, 1).expect("arm 1 should exist");
        if let TreeNode::Leaf { filter, .. } = leaf {
            // alpha = 1 + 0.7 + 0.3 = 2.0
            assert!((filter.alpha - 2.0).abs() < 1e-5, "alpha={}", filter.alpha);
            // beta = 1 + 0.3 + 0.7 = 2.0
            assert!((filter.beta - 2.0).abs() < 1e-5, "beta={}", filter.beta);
        }
    }

    // ── T1.5(c): drift_rate=0 degenerates to stationary Beta ────────────

    #[test]
    fn test_predict_zero_drift_is_noop() {
        let mut arm = BayesianFilterArm::new(0.0);
        arm.alpha = 11.0;
        arm.beta = 6.0;
        arm.last_obs_step = 0;

        // Predict at step 100 — should be a complete no-op.
        arm.predict(100);

        assert!((arm.alpha - 11.0).abs() < 1e-5, "alpha unchanged");
        assert!((arm.beta - 6.0).abs() < 1e-5, "beta unchanged");
    }

    #[test]
    fn test_zero_drift_sample_mean_matches_beta_mean() {
        // With drift_rate=0, after observing evidence, the sample distribution
        // should match Beta(alpha, beta). Verify the empirical mean over 10K draws.
        let mut arm = BayesianFilterArm::new(0.0);
        // 15 successes, 5 failures → Beta(16, 6), mean = 16/22 ≈ 0.7273.
        for _ in 0..15 {
            arm.update(1.0, 0);
        }
        for _ in 0..5 {
            arm.update(0.0, 0);
        }

        let expected_mean = 16.0_f32 / 22.0;
        let mut rng = fastrand::Rng::with_seed(42);
        let n = 10_000_usize;
        let sum: f32 = (0..n).map(|_| arm.thompson_sample(&mut rng)).sum();
        let empirical_mean = sum / n as f32;

        assert!(
            (empirical_mean - expected_mean).abs() < 0.02,
            "empirical mean {empirical_mean:.4} too far from Beta mean {expected_mean:.4}"
        );
    }

    #[test]
    fn test_predict_nonzero_drift_pulls_toward_uniform() {
        let mut arm = BayesianFilterArm::new(0.5);
        arm.alpha = 21.0; // strong evidence
        arm.beta = 1.0;
        arm.last_obs_step = 0;

        // After 1 step with λ=0.5:
        // decay = 0.5, alpha' = 21*0.5 + 0.5 = 11.0, beta' = 1*0.5 + 0.5 = 1.0.
        arm.predict(1);
        assert!((arm.alpha - 11.0).abs() < 1e-4, "alpha after 1 drift step: {}", arm.alpha);
        assert!((arm.beta - 1.0).abs() < 1e-4, "beta after 1 drift step: {}", arm.beta);

        // After another step (total elapsed = 1 from last_obs_step=1):
        // decay = 0.5, alpha' = 11*0.5 + 0.5 = 6.0.
        arm.predict(2);
        assert!((arm.alpha - 6.0).abs() < 1e-4, "alpha after 2 drift steps: {}", arm.alpha);
    }

    // ── T1.5(d): blake3 stability ───────────────────────────────────────

    #[test]
    fn test_blake3_stable_across_rebuilds() {
        let config = LatentTaskTreeConfig::default();

        // Build two identical trees.
        let tree1 = build_test_tree(0.01);
        let tree2 = {
            let left = TreeNode::internal(vec![
                TreeNode::leaf(0, 0.01),
                TreeNode::leaf(1, 0.01),
            ]);
            let right = TreeNode::internal(vec![
                TreeNode::leaf(2, 0.01),
                TreeNode::leaf(3, 0.01),
            ]);
            let root = TreeNode::internal(vec![left, right]);
            LatentTaskTree::from_root(root, config)
        };

        assert_eq!(
            tree1.blake3_root(),
            tree2.blake3_root(),
            "identical trees must have identical BLAKE3 commitments"
        );
    }

    #[test]
    fn test_blake3_changes_on_different_trees() {
        let tree_a = build_test_tree(0.01);

        // Different topology.
        let different_root = TreeNode::internal(vec![
            TreeNode::leaf(0, 0.01),
            TreeNode::leaf(1, 0.01),
            TreeNode::leaf(2, 0.01),
        ]);
        let tree_b = LatentTaskTree::from_root(different_root, LatentTaskTreeConfig::default());

        assert_ne!(
            tree_a.blake3_root(),
            tree_b.blake3_root(),
            "different trees must have different BLAKE3 commitments"
        );
    }

    #[test]
    fn test_blake3_changes_on_different_drift_rate() {
        // Different drift_rate → different initial filter state → different hash.
        let tree_a = build_test_tree(0.01);
        let tree_b = build_test_tree(0.05);
        assert_ne!(tree_a.blake3_root(), tree_b.blake3_root());
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_observe_increments_n_obs() {
        let mut tree = build_test_tree(0.0);
        tree.observe(0, 1.0, 0);
        tree.observe(0, 0.0, 1);
        tree.observe(1, 1.0, 2);

        // Root should have n_obs = 3.
        if let TreeNode::Internal { n_obs, .. } = &tree.root {
            assert_eq!(*n_obs, 3, "root n_obs should be 3");
        } else {
            panic!("root should be Internal");
        }
    }

    #[test]
    #[should_panic(expected = "not in tree")]
    fn test_observe_invalid_arm_panics() {
        let mut tree = build_test_tree(0.0);
        tree.observe(99, 1.0, 0);
    }

    #[test]
    fn test_arm_path_depth_correct() {
        let tree = build_test_tree(0.0);
        // All leaves are at depth 2 in the test tree.
        for arm in 0..4 {
            assert_eq!(
                tree.arm_paths[arm].len, 2,
                "arm {arm} path should have depth 2"
            );
        }
    }

    #[test]
    fn test_num_arms() {
        let tree = build_test_tree(0.0);
        assert_eq!(tree.num_arms(), 4);
    }

    // ── Helper: find a leaf by arm_id ───────────────────────────────────

    fn find_leaf<'a>(node: &'a TreeNode, arm_id: usize) -> Option<&'a TreeNode> {
        match node {
            TreeNode::Leaf {
                arm_id: id, ..
            } if *id == arm_id => Some(node),
            TreeNode::Leaf { .. } => None,
            TreeNode::Internal { children, .. } => {
                for child in children {
                    if let Some(found) = find_leaf(child, arm_id) {
                        return Some(found);
                    }
                }
                None
            }
        }
    }
}
