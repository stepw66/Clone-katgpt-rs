//! Reconstructive Memory Navigation — OctreeCTC (Plan 248).
//!
//! Multi-step active reconstruction over KG-Latent-Octree sense modules.
//! Distilled from MRAgent (ICML 2026): Cue–Tag–Content graph with iterative
//! HLA-state-aware navigation. Modelless: entropy bandit + dot-product + sigmoid.
//!
//! Key insight: single-shot `NpcBrain::project_all()` is passive retrieval.
//! This module adds active reconstruction — the HLA state evolves based on
//! accumulated evidence, producing strictly more expressive retrieval
//! (Theorem 4.1, arXiv:2606.06036).

use crate::types::SenseModule;

/// Morton-code identifier for an octree node.
/// Encodes spatial position in the KG latent embedding space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct OctreeNodeId(pub u32);

impl OctreeNodeId {
    /// Root node ID.
    pub const ROOT: Self = Self(0);

    /// Child index (0..8) at given depth.
    #[inline]
    pub fn child(&self, child_idx: u8) -> Self {
        // Octree: node n at depth d has children at indices 8*n+1 .. 8*n+8
        Self(self.0 * 8 + 1 + child_idx as u32)
    }

    /// Depth in the octree.
    #[inline]
    pub fn depth(&self) -> u8 {
        if self.0 == 0 {
            return 0;
        }
        // For octree: node 0=depth 0, nodes 1-8=depth 1, nodes 9-72=depth 2, etc.
        // Use iterative approach for correctness
        let mut n = self.0;
        let mut d = 0u8;
        while n > 0 {
            n = (n - 1) / 8;
            d += 1;
        }
        d
    }

    /// Parent node ID, or None for root.
    #[inline]
    pub fn parent(&self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(Self((self.0 - 1) / 8))
        }
    }
}

/// Traversal action during reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraversalAction {
    /// Expand forward: cue → tag → content.
    Forward { tag_idx: u8 },
    /// Reverse: content → cue + tag (backtrack with new evidence).
    Reverse { content_idx: u8 },
    /// Stop reconstruction — sufficient evidence accumulated.
    Halt,
}

/// Configuration for reconstruction behavior.
#[derive(Clone, Debug)]
pub struct ReconstructionConfig {
    /// Maximum reconstruction steps (default: 3, MRAgent shows diminishing returns after 3-4).
    pub max_steps: u8,
    /// Learning rate for HLA state evolution (default: 0.1).
    pub hla_learning_rate: f32,
    /// Entropy threshold for early stopping (default: 0.05).
    /// Below this entropy, evidence is considered sufficient.
    pub entropy_threshold: f32,
    /// Enable LOD-adaptive pruning (default: true).
    /// Reduces octree depth when activation spread is narrow.
    pub lod_adaptive: bool,
    /// Maximum activation delta per step (default: 0.3).
    /// Prevents HLA state from jumping too far in one step.
    pub max_hla_delta: f32,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            max_steps: 3,
            hla_learning_rate: 0.1,
            entropy_threshold: 0.05,
            lod_adaptive: true,
            max_hla_delta: 0.3,
        }
    }
}

/// Accumulated KG triple evidence from reconstruction.
/// Fixed-size for zero-allocation hot path.
#[derive(Clone, Debug, Default)]
pub struct TripleEvidence {
    /// Number of triples recovered.
    pub count: u8,
    /// Sum of confidence scores for recovered triples.
    pub confidence_sum: f32,
    /// Per-kind activation strengths (indexed by SenseKind discriminant).
    pub kind_activations: [f32; 6],
}

impl TripleEvidence {
    /// Mean confidence of recovered evidence.
    #[inline]
    pub fn mean_confidence(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.confidence_sum / self.count as f32
    }

    /// Entropy of kind activations — low entropy means focused evidence.
    /// Uses fast approximation: 1.0 - max_activation (0 = focused, 1 = uniform).
    #[inline]
    pub fn activation_entropy(&self) -> f32 {
        let total: f32 = self.kind_activations.iter().copied().sum();
        if total < 1e-8 {
            return 1.0;
        }
        let max_val = self.kind_activations.iter().copied().fold(0.0f32, f32::max);
        1.0 - max_val / total
    }

    /// Merge evidence from another source.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.count = self.count.saturating_add(other.count);
        self.confidence_sum += other.confidence_sum;
        for i in 0..6 {
            self.kind_activations[i] += other.kind_activations[i];
        }
    }
}

/// Reconstruction state: tracks active traversal across the sense octree.
///
/// This is the core of active reconstruction — the HLA state evolves based on
/// accumulated evidence, producing adaptive multi-step retrieval without LLM calls.
pub struct ReconstructionState {
    /// Evolving HLA state (cue). Updated by `evolve_hla()` after each step.
    hla: [f32; 8],
    /// Active octree nodes being explored (Z(t)).
    /// Used in Phase 2 for full octree traversal.
    #[allow(dead_code)]
    active_nodes: [Option<OctreeNodeId>; 8],
    /// Number of active nodes.
    /// Used in Phase 2 for full octree traversal.
    #[allow(dead_code)]
    n_active: u8,
    /// Accumulated evidence (H(t)).
    evidence: TripleEvidence,
    /// Current reconstruction step.
    step: u8,
    /// Configuration.
    config: ReconstructionConfig,
}

impl ReconstructionState {
    /// Initialize reconstruction with a starting HLA state.
    #[inline]
    pub fn new(hla: [f32; 8]) -> Self {
        let mut active_nodes = [None; 8];
        active_nodes[0] = Some(OctreeNodeId::ROOT);
        Self {
            hla,
            active_nodes,
            n_active: 1,
            evidence: TripleEvidence::default(),
            step: 0,
            config: ReconstructionConfig::default(),
        }
    }

    /// Initialize with custom config.
    #[inline]
    pub fn with_config(hla: [f32; 8], config: ReconstructionConfig) -> Self {
        let mut active_nodes = [None; 8];
        active_nodes[0] = Some(OctreeNodeId::ROOT);
        Self {
            hla,
            active_nodes,
            n_active: 1,
            evidence: TripleEvidence::default(),
            step: 0,
            config,
        }
    }

    /// Current HLA state (cue).
    #[inline]
    pub fn hla(&self) -> &[f32; 8] {
        &self.hla
    }

    /// Current accumulated evidence.
    #[inline]
    pub fn evidence(&self) -> &TripleEvidence {
        &self.evidence
    }

    /// Current step number.
    #[inline]
    pub fn step(&self) -> u8 {
        self.step
    }

    /// Whether reconstruction should stop.
    #[inline]
    pub fn sufficient(&self) -> bool {
        self.step >= self.config.max_steps
            || (self.step > 0 && self.evidence.activation_entropy() < self.config.entropy_threshold)
    }

    /// Expand active nodes: project each module with current HLA, rank results.
    /// Returns activation scores per module, sorted descending.
    ///
    /// This is the `expand` step from MRAgent Algorithm 1:
    /// `Z'(t+1) = union(Π_a(Z(t)) for a in A(t))`
    pub fn expand(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        let mut activations = [0.0f32; 6];

        // Project all modules with current (evolved) HLA state
        for module in &brain.modules {
            let kind_idx = module.kind as usize;
            if kind_idx < 6 {
                activations[kind_idx] = module.project(&self.hla);
            }
        }

        activations
    }

    /// Route: select which modules to follow based on activation strength.
    /// Uses entropy-gated selection — keep modules above mean + threshold.
    ///
    /// This is the `route` step from MRAgent: `Z(t+1) = f_route(x, H(t), Z'(t+1))`
    pub fn route(&self, activations: &[f32; 6]) -> [bool; 6] {
        let total: f32 = activations.iter().copied().sum();
        if total < 1e-8 {
            return [false; 6];
        }

        let mean = total / 6.0;
        let mut selected = [false; 6];

        for i in 0..6 {
            // Select modules above mean activation (entropy-gated threshold)
            selected[i] = activations[i] > mean;
        }

        // Ensure at least one module selected (pick max)
        if !selected.iter().any(|&s| s) {
            let max_idx = activations
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            selected[max_idx] = true;
        }

        selected
    }

    /// Accumulate evidence from selected modules.
    ///
    /// This is the accumulate step: `H(t+1) = H(t) ∪ Z(t+1)`
    pub fn accumulate(&mut self, selected: &[bool; 6], activations: &[f32; 6]) {
        for (i, &sel) in selected.iter().enumerate() {
            if !sel {
                continue;
            }
            if activations[i] > 0.0 {
                self.evidence.count = self.evidence.count.saturating_add(1);
                self.evidence.confidence_sum += activations[i];
                self.evidence.kind_activations[i] += activations[i];
            }
        }
    }

    /// Evolve HLA state based on accumulated evidence.
    ///
    /// Bridge function per AGENTS.md: raw KG triples → latent HLA update.
    /// Uses dot-product projection + sigmoid. No softmax.
    /// Zero-allocation. Clamp to valid range [-1, 1].
    pub fn evolve_hla(&mut self) {
        let lr = self.config.hla_learning_rate;
        let max_delta = self.config.max_hla_delta;
        let total_activation: f32 = self.evidence.kind_activations.iter().copied().sum();

        if total_activation < 1e-8 {
            return; // No evidence — don't evolve
        }

        // Project accumulated activations back onto HLA dimensions.
        // Each HLA dimension gets a delta proportional to its contribution
        // to the total activation, scaled by learning rate.
        for i in 0..8 {
            // Cross-couple: use activation pattern to shift HLA
            // Simple bridge: HLA[i] += lr * (activation[i % 6] / total - 0.5) * activation_sum
            let kind_idx = i % 6;
            let normalized = self.evidence.kind_activations[kind_idx] / total_activation;
            let delta = lr * (normalized - 0.5) * total_activation.min(1.0);

            // Clamp delta to prevent large jumps
            let clamped_delta = delta.clamp(-max_delta, max_delta);

            // Update HLA with clamped delta
            self.hla[i] = (self.hla[i] + clamped_delta).clamp(-1.0, 1.0);
        }
    }

    /// Run full reconstruction loop.
    ///
    /// Combines expand → route → accumulate → evolve_hla → sufficient check
    /// into a single call. Returns final activations.
    ///
    /// Equivalent to MRAgent Algorithm 1, but modelless:
    /// - select: entropy-gated threshold (not LLM)
    /// - expand: SenseModule::project (not graph traversal)
    /// - route: activation ranking (not LLM routing)
    /// - accumulate: TripleEvidence merge (not LLM summarization)
    /// - evolve_hla: bridge function (not LLM reasoning)
    pub fn reconstruct(&mut self, brain: &crate::sense::brain::NpcBrain) -> [f32; 6] {
        loop {
            let activations = self.expand(brain);
            let selected = self.route(&activations);
            self.accumulate(&selected, &activations);
            self.evolve_hla();

            self.step += 1;

            if self.sufficient() {
                return activations;
            }
        }
    }
}

impl SenseModule {
    /// Project with reconstruction awareness — same as `project()` but exposed
    /// for reconstruction loop to call explicitly.
    #[inline]
    pub fn project_reconstruction(&self, hla_state: &[f32; 8]) -> f32 {
        self.project(hla_state)
    }

    /// Get octree children that are occupied at the given level.
    /// Returns bitmask of occupied children (bit i = child i is occupied).
    #[inline]
    pub fn occupied_children(&self, parent_depth: u8) -> u8 {
        let level = parent_depth as usize;
        if level >= 4 {
            return 0;
        }
        // Extract 8 bits from octree_bits for children at this depth
        let bit_offset = level * 8;
        let word = bit_offset / 64;
        let shift = bit_offset % 64;
        if word >= 4 {
            return 0;
        }
        ((self.octree_bits[word] >> shift) & 0xFF) as u8
    }
}

/// Reconstruction result with before/after comparison for GOAT proof.
#[derive(Clone, Debug)]
pub struct ReconstructionResult {
    /// Passive single-shot activations (baseline).
    pub passive: [f32; 6],
    /// Active multi-step activations (reconstructed).
    pub active: [f32; 6],
    /// Number of reconstruction steps taken.
    pub steps: u8,
    /// Final evidence state.
    pub evidence: TripleEvidence,
    /// HLA state delta (active - passive HLA).
    pub hla_delta: [f32; 8],
}

/// Run side-by-side comparison: passive vs active reconstruction.
/// Used for GOAT proof tests and benchmarks.
pub fn compare_reconstruction(
    brain: &crate::sense::brain::NpcBrain,
    hla: [f32; 8],
) -> ReconstructionResult {
    // Passive: single-shot projection
    let passive = {
        let mut acts = [0.0f32; 6];
        for module in &brain.modules {
            let idx = module.kind as usize;
            if idx < 6 {
                acts[idx] = module.project(&hla);
            }
        }
        acts
    };

    // Active: multi-step reconstruction
    let mut state = ReconstructionState::new(hla);
    let active = state.reconstruct(brain);

    // Compute HLA delta
    let mut hla_delta = [0.0f32; 8];
    for i in 0..8 {
        hla_delta[i] = state.hla[i] - hla[i];
    }

    ReconstructionResult {
        passive,
        active,
        steps: state.step,
        evidence: state.evidence.clone(),
        hla_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octree_node_id_depth() {
        assert_eq!(OctreeNodeId::ROOT.depth(), 0);
        assert_eq!(OctreeNodeId(1).depth(), 1); // first child of root
        assert_eq!(OctreeNodeId(8).depth(), 1); // last child of root
        assert_eq!(OctreeNodeId(9).depth(), 2); // first grandchild
        assert_eq!(OctreeNodeId(72).depth(), 2); // last grandchild
    }

    #[test]
    fn octree_node_id_parent() {
        assert!(OctreeNodeId::ROOT.parent().is_none());
        assert_eq!(OctreeNodeId(1).parent(), Some(OctreeNodeId::ROOT));
        assert_eq!(OctreeNodeId(8).parent(), Some(OctreeNodeId::ROOT));
        assert_eq!(OctreeNodeId(9).parent(), Some(OctreeNodeId(1)));
        assert_eq!(OctreeNodeId(72).parent(), Some(OctreeNodeId(8)));
    }

    #[test]
    fn octree_node_id_child() {
        assert_eq!(OctreeNodeId::ROOT.child(0), OctreeNodeId(1));
        assert_eq!(OctreeNodeId::ROOT.child(7), OctreeNodeId(8));
        assert_eq!(OctreeNodeId(1).child(0), OctreeNodeId(9));
    }

    #[test]
    fn reconstruction_config_default() {
        let config = ReconstructionConfig::default();
        assert_eq!(config.max_steps, 3);
        assert!((config.hla_learning_rate - 0.1).abs() < 1e-6);
        assert!((config.entropy_threshold - 0.05).abs() < 1e-6);
        assert!(config.lod_adaptive);
    }

    #[test]
    fn triple_evidence_entropy_focused() {
        let mut ev = TripleEvidence::default();
        ev.kind_activations[0] = 1.0;
        ev.kind_activations[1] = 0.0;
        ev.kind_activations[2] = 0.0;
        ev.kind_activations[3] = 0.0;
        ev.kind_activations[4] = 0.0;
        ev.kind_activations[5] = 0.0;
        let entropy = ev.activation_entropy();
        assert!(
            entropy < 0.1,
            "Focused evidence should have low entropy, got {entropy}"
        );
    }

    #[test]
    fn triple_evidence_entropy_uniform() {
        let mut ev = TripleEvidence::default();
        ev.kind_activations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let entropy = ev.activation_entropy();
        assert!(
            entropy > 0.5,
            "Uniform evidence should have high entropy, got {entropy}"
        );
    }

    #[test]
    fn reconstruction_state_sufficient_at_max_steps() {
        let config = ReconstructionConfig {
            max_steps: 0,
            ..Default::default()
        };
        let state = ReconstructionState::with_config([0.0; 8], config);
        assert!(state.sufficient());
    }

    #[test]
    fn reconstruction_state_not_sufficient_initially() {
        let state = ReconstructionState::new([0.5; 8]);
        assert!(!state.sufficient()); // step 0, max_steps 3
    }
}
