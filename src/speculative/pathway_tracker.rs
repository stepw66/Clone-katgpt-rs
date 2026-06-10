//! Deep Manifold §4.2-4.3 — Intrinsic Pathway Stability Detection (Research 205, Plan 231)
//!
//! Inference traverses intrinsic pathways through stacked manifolds.
//! Stable pathway = converged fixed point. Unstable = keep searching.

/// Tracks branch selection patterns across consecutive inference steps.
pub struct PathwayTracker {
    history: Vec<Vec<usize>>,
    max_depth: usize,
    cursor: usize,
    steps: usize,
}

impl PathwayTracker {
    pub fn new(max_depth: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_depth),
            max_depth,
            cursor: 0,
            steps: 0,
        }
    }

    /// Record branch selection for current step.
    pub fn update(&mut self, branches: &[usize]) {
        let mut sorted = branches.to_vec();
        sorted.sort_unstable();
        if self.history.len() < self.max_depth {
            self.history.push(sorted);
        } else {
            self.history[self.cursor] = sorted;
        }
        self.cursor = (self.cursor + 1) % self.max_depth;
        self.steps += 1;
    }

    /// Compute pathway stability: sigmoid of consecutive-match ratio.
    /// Near 1.0 = very stable, near 0.0 = unstable.
    pub fn stability(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.5;
        }
        let mut matches = 0usize;
        let mut comparisons = 0usize;
        for i in 1..self.history.len() {
            let prev = &self.history[i - 1];
            let curr = &self.history[i];
            comparisons += 1;
            let overlap = prev
                .iter()
                .filter(|b| curr.binary_search(b).is_ok())
                .count();
            let max_len = prev.len().max(curr.len()).max(1);
            if overlap >= max_len / 2 {
                matches += 1;
            }
        }
        if comparisons == 0 {
            return 0.5;
        }
        let ratio = matches as f32 / comparisons as f32;
        1.0 / (1.0 + (-(ratio - 0.5) * 4.0).exp())
    }

    /// Check if pathway has converged.
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.steps >= 3 && self.stability() > threshold
    }

    /// Reset for new inference session.
    pub fn reset(&mut self) {
        self.history.clear();
        self.cursor = 0;
        self.steps = 0;
    }

    /// Number of steps recorded.
    pub fn steps(&self) -> usize {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_returns_mid_stability() {
        let tracker = PathwayTracker::new(10);
        assert!((tracker.stability() - 0.5).abs() < 1e-6);
        assert!(!tracker.is_converged(0.8));
    }

    #[test]
    fn single_entry_returns_mid_stability() {
        let mut tracker = PathwayTracker::new(10);
        tracker.update(&[1, 2, 3]);
        assert!((tracker.stability() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn stable_pathway_converges() {
        let mut tracker = PathwayTracker::new(10);
        // Same pattern repeated
        for _ in 0..5 {
            tracker.update(&[1, 3, 5]);
        }
        assert!(
            tracker.stability() > 0.8,
            "stability should be high, got {}",
            tracker.stability()
        );
        assert!(tracker.is_converged(0.7));
    }

    #[test]
    fn unstable_pathway_does_not_converge() {
        let mut tracker = PathwayTracker::new(10);
        for i in 0..5 {
            tracker.update(&[i, i + 10, i + 20]);
        }
        assert!(
            !tracker.is_converged(0.8),
            "should not converge with shifting branches"
        );
    }

    #[test]
    fn convergence_requires_min_steps() {
        let mut tracker = PathwayTracker::new(10);
        tracker.update(&[1, 2]);
        tracker.update(&[1, 2]);
        // Only 2 steps, need >= 3
        assert!(!tracker.is_converged(0.1));
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = PathwayTracker::new(10);
        for _ in 0..5 {
            tracker.update(&[1, 2, 3]);
        }
        assert!(tracker.steps() > 0);
        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        assert!(!tracker.is_converged(0.1));
    }

    #[test]
    fn ring_buffer_wraps() {
        let mut tracker = PathwayTracker::new(3);
        tracker.update(&[1]);
        tracker.update(&[2]);
        tracker.update(&[3]);
        tracker.update(&[4]); // wraps, overwrites [1]
        assert_eq!(tracker.steps(), 4);
    }

    #[test]
    fn partial_overlap_mid_stability() {
        let mut tracker = PathwayTracker::new(10);
        tracker.update(&[1, 2, 3, 4, 5]);
        tracker.update(&[1, 2, 3, 6, 7]); // 3/5 overlap → matches
        tracker.update(&[8, 9, 10, 11, 12]); // 0/5 overlap → no match
        // 1 match out of 2 comparisons = 0.5 ratio
        let stab = tracker.stability();
        assert!(
            stab > 0.4 && stab < 0.8,
            "partial overlap stability should be mid-range, got {}",
            stab
        );
    }
}
