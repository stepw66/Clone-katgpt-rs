//! WorkflowLattice — sigmoid-graded relevance scoring (Plan 223, Phase 4).
//!
//! Builds a lattice of predicates with implication ordering.
//! Sigmoid-graded satisfaction incrementally propagates across DDTree levels.

/// A node in the workflow lattice with sigmoid-graded satisfaction.
#[derive(Clone, Debug)]
pub struct PredicateNode {
    /// Unique identifier for this predicate.
    pub id: u32,
    /// Current satisfaction score [0, 1] via sigmoid.
    pub satisfaction: f32,
    /// Implication ordering: predicates that must be satisfied before this one.
    pub implies: Vec<u32>,
}

impl PredicateNode {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            satisfaction: 0.0,
            implies: Vec::new(),
        }
    }

    /// Update satisfaction using sigmoid grading.
    /// raw ∈ (-∞, +∞) → sigmoid → [0, 1]
    pub fn update_satisfaction(&mut self, raw_score: f32) {
        self.satisfaction = 1.0 / (1.0 + (-raw_score).exp());
    }
}

/// Workflow lattice for incremental satisfaction tracking.
#[derive(Clone, Debug, Default)]
pub struct WorkflowLattice {
    /// Predicate nodes indexed by id.
    pub nodes: Vec<PredicateNode>,
}

impl WorkflowLattice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate node to the lattice.
    pub fn add_node(&mut self, node: PredicateNode) {
        let id = node.id as usize;
        if id >= self.nodes.len() {
            self.nodes.resize(id + 1, PredicateNode::new(0));
        }
        self.nodes[id] = node;
    }

    /// Compute overall lattice satisfaction.
    /// Uses geometric mean of all node satisfactions (not softmax).
    pub fn overall_satisfaction(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let log_sum: f32 = self
            .nodes
            .iter()
            .filter(|n| n.satisfaction > 1e-10)
            .map(|n| n.satisfaction.ln())
            .sum();
        let n = self.nodes.iter().filter(|n| n.satisfaction > 1e-10).count();
        if n == 0 {
            return 0.0;
        }
        (log_sum / n as f32).exp()
    }

    /// Propagate satisfaction from parent to child predicates.
    /// If parent is unsatisfied, child satisfaction is reduced.
    pub fn propagate(&mut self) {
        let mut new_sats: Vec<f32> = self.nodes.iter().map(|n| n.satisfaction).collect();
        for node in &self.nodes {
            for &implies_id in &node.implies {
                let implies_id = implies_id as usize;
                if implies_id < new_sats.len() {
                    // Child satisfaction is bounded by parent satisfaction
                    let parent_sat = self.nodes[implies_id].satisfaction;
                    let child_idx = node.id as usize;
                    if child_idx < new_sats.len() {
                        new_sats[child_idx] = new_sats[child_idx].min(parent_sat);
                    }
                }
            }
        }
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.satisfaction = new_sats[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_grading() {
        let mut node = PredicateNode::new(0);
        node.update_satisfaction(0.0);
        assert!((node.satisfaction - 0.5).abs() < 0.01);
        node.update_satisfaction(5.0);
        assert!(node.satisfaction > 0.9);
        node.update_satisfaction(-5.0);
        assert!(node.satisfaction < 0.1);
    }

    #[test]
    fn test_overall_satisfaction() {
        let mut lattice = WorkflowLattice::new();
        let mut n1 = PredicateNode::new(0);
        n1.satisfaction = 0.8;
        let mut n2 = PredicateNode::new(1);
        n2.satisfaction = 0.6;
        lattice.add_node(n1);
        lattice.add_node(n2);
        let overall = lattice.overall_satisfaction();
        assert!(overall > 0.0 && overall <= 1.0);
    }

    #[test]
    fn test_propagation() {
        let mut lattice = WorkflowLattice::new();
        let mut parent = PredicateNode::new(0);
        parent.satisfaction = 0.3;
        let mut child = PredicateNode::new(1);
        child.satisfaction = 0.9;
        child.implies = vec![0]; // depends on parent
        lattice.add_node(parent);
        lattice.add_node(child);
        lattice.propagate();
        // Child satisfaction should be bounded by parent
        assert!(lattice.nodes[1].satisfaction <= 0.3 + 0.01);
    }
}
