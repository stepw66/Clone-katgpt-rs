//! `MuxDdTree` — superposition DD-tree with BFS frontier mode.
//!
//! Each node carries K tokens as a weighted span (superposition).
//! `hypothesis_coverage()` = `leaf_count() * K^depth`.
//!
//! BFS frontier mode reads logit distributions at each depth, detects the
//! effective width (number of valid superposition peaks), and expands all
//! peaks simultaneously.

use crate::mux::span_pruner::MuxSpanPruner;
use crate::mux::top_k::extract_top_k_peaks;

/// Default superposition width (number of tokens per node).
pub const DEFAULT_K: usize = 4;

/// A node in the DD-tree that carries K tokens as a weighted span.
#[derive(Debug, Clone)]
pub struct MuxNode {
    /// Token IDs held in superposition at this node.
    pub tokens: Vec<u32>,
    /// Corresponding weights (logit values) for each token.
    pub weights: Vec<f32>,
    /// Child nodes (branching factor = width at this depth).
    pub children: Vec<MuxNode>,
}

impl MuxNode {
    pub fn new(tokens: Vec<u32>, weights: Vec<f32>) -> Self {
        assert_eq!(tokens.len(), weights.len());
        Self {
            tokens,
            weights,
            children: Vec::new(),
        }
    }

    /// Number of tokens in the superposition span.
    pub fn span_size(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if this node is a leaf (no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// DD-tree wrapper that manages superposition expansion.
#[derive(Debug, Clone)]
pub struct MuxDdTree {
    /// Root node.
    pub root: MuxNode,
    /// Maximum superposition width per node.
    pub k: usize,
    /// Current depth of the tree.
    pub depth: usize,
    /// Pruner for validating superposition spans.
    pub pruner: MuxSpanPruner,
}

impl MuxDdTree {
    pub fn new(k: usize) -> Self {
        let pruner = MuxSpanPruner::new(k, 0.5);
        Self {
            root: MuxNode::new(Vec::new(), Vec::new()),
            k,
            depth: 0,
            pruner,
        }
    }

    /// Initialize the root with an initial superposition from logit distribution.
    pub fn init_root(&mut self, logits: &[f32]) {
        let peaks = extract_top_k_peaks(logits, self.k);
        let tokens: Vec<u32> = (0..peaks.len() as u32).collect();
        self.root = MuxNode::new(tokens, peaks);
        self.depth = 0;
    }

    /// Count leaf nodes in the tree.
    pub fn leaf_count(&self) -> usize {
        Self::count_leaves(&self.root)
    }

    fn count_leaves(node: &MuxNode) -> usize {
        if node.is_leaf() {
            1
        } else {
            node.children.iter().map(Self::count_leaves).sum()
        }
    }

    /// Total hypothesis coverage: `leaf_count * K^depth`.
    pub fn hypothesis_coverage(&self) -> usize {
        let k_pow = self.k.pow(self.depth as u32);
        self.leaf_count() * k_pow
    }

    /// Expand a leaf node at the given path using logit distribution.
    /// Creates `width` children, each with top-K tokens from `logits`.
    pub fn expand_node(&mut self, path: &[usize], logits: &[f32], width: usize) {
        let node = Self::get_node_mut(&mut self.root, path);
        let peaks = extract_top_k_peaks(logits, self.k);
        let effective_width = width.min(peaks.len()).max(1);

        for i in 0..effective_width {
            // Distribute peaks across children: each child gets a shifted view
            let offset = (i * self.k / effective_width).min(peaks.len());
            let child_tokens: Vec<u32> =
                (offset as u32..(offset + peaks.len().min(self.k)) as u32).collect();
            let child_weights: Vec<f32> = peaks.iter().take(self.k).copied().collect();
            node.children
                .push(MuxNode::new(child_tokens, child_weights));
        }

        // Track maximum depth
        let new_depth = path.len() + 1;
        if new_depth > self.depth {
            self.depth = new_depth;
        }
    }

    /// **BFS frontier mode**: expand all current leaves simultaneously using
    /// per-depth logit distributions and dynamic width detection.
    ///
    /// For each leaf, reads the logit distribution, determines the effective
    /// width via `detect_width`, validates with the pruner, and expands.
    pub fn expand_bfs_frontier<F>(&mut self, depth: usize, logits_by_leaf: &[F])
    where
        F: AsRef<[f32]>,
    {
        let leaves = self.collect_leaf_paths();
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "must provide logits for every leaf"
        );

        for (path, logits) in leaves.into_iter().zip(logits_by_leaf.iter()) {
            let logits = logits.as_ref();
            let width = self.detect_width(logits);
            if width > 0 && self.pruner.is_valid(logits, depth) {
                self.expand_node(&path, logits, width);
            }
        }
    }

    /// Detect the effective branching width from a logit distribution.
    /// Returns 1 for peaked (single dominant token) distributions,
    /// or K for multi-peak (valid superposition) distributions.
    pub fn detect_width(&self, logits: &[f32]) -> usize {
        let peaks = extract_top_k_peaks(logits, self.k);
        if peaks.len() < 2 {
            return 1;
        }
        // Check if the distribution is peaked: top value dominates
        let total: f32 = peaks.iter().sum();
        if total <= 0.0 {
            return 1;
        }
        let top_ratio = peaks[0] / total;
        if top_ratio > 0.8 {
            // Peaked distribution — single-token expansion
            1
        } else {
            // Multi-peak distribution — full superposition width
            peaks.len().min(self.k)
        }
    }

    /// Collect paths to all leaf nodes (BFS order).
    pub fn collect_leaf_paths(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut queue: Vec<(Vec<usize>, &MuxNode)> = vec![(Vec::new(), &self.root)];
        while let Some((path, node)) = queue.pop() {
            if node.is_leaf() {
                result.push(path);
            } else {
                for (i, child) in node.children.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(i);
                    queue.push((child_path, child));
                }
            }
        }
        result
    }

    fn get_node_mut<'a>(node: &'a mut MuxNode, path: &[usize]) -> &'a mut MuxNode {
        let mut current = node;
        for &idx in path {
            current = &mut current.children[idx];
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_root_and_leaf_count() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        tree.init_root(&logits);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.root.span_size(), 4); // top-4 peaks
        assert_eq!(tree.depth, 0);
    }

    #[test]
    fn hypothesis_coverage_formula() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        tree.init_root(&logits);
        // 1 leaf * 4^0 = 1
        assert_eq!(tree.hypothesis_coverage(), 1);
    }

    #[test]
    fn expand_node_increases_leaves() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        tree.init_root(&logits);
        tree.expand_node(&[], &logits, 2);
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.depth, 1);
        // 2 leaves * 4^1 = 8
        assert_eq!(tree.hypothesis_coverage(), 8);
    }

    #[test]
    fn detect_width_peaked() {
        let tree = MuxDdTree::new(4);
        // Single dominant peak (1.0 is > 80% of total)
        let logits = vec![1.0, 0.05, 0.03, 0.02, 0.01];
        assert_eq!(tree.detect_width(&logits), 1);
    }

    #[test]
    fn detect_width_multi_peak() {
        let tree = MuxDdTree::new(4);
        // Spread across multiple peaks
        let logits = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        assert_eq!(tree.detect_width(&logits), 4);
    }

    #[test]
    fn bfs_frontier_expansion() {
        let mut tree = MuxDdTree::new(4);
        let logits = vec![0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01];
        tree.init_root(&logits);
        assert_eq!(tree.leaf_count(), 1);

        // Expand frontier: multi-peak distribution should expand to width 4
        let leaf_logits: Vec<Vec<f32>> = vec![logits.clone()];
        tree.expand_bfs_frontier(1, &leaf_logits);
        assert!(tree.leaf_count() > 1);
        assert_eq!(tree.depth, 1);
    }
}
