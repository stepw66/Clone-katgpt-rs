//! Decomposition quality reviewer. Prevents search collapse on dead-end branches.
//! Inspired by LEAP's LLM reviewer that rejects unproductive decompositions.
//!
//! Uses ProofGoalCache hit rate as the progress signal:
//! - High cache miss rate → exploring new territory → productive → keep
//! - High cache hit rate → revisiting known states → unproductive → prune
//!
//! The "review" is a statistical check, not a model inference.

use std::sync::atomic::{AtomicU64, Ordering};

/// Decomposition quality reviewer. Prevents search collapse on dead-end branches.
pub struct DecompositionReviewer {
    /// Minimum novelty (cache miss rate) to consider a branch productive.
    /// Range [0.0, 1.0]. Default: 0.3 (30% of goals must be novel).
    min_novelty: f32,
    /// Per-branch cache hit counter.
    branch_hits: AtomicU64,
    /// Per-branch cache miss counter.
    branch_misses: AtomicU64,
}

impl DecompositionReviewer {
    /// Create a reviewer with a minimum novelty threshold.
    pub fn new(min_novelty: f32) -> Self {
        Self {
            min_novelty: min_novelty.clamp(0.0, 1.0),
            branch_hits: AtomicU64::new(0),
            branch_misses: AtomicU64::new(0),
        }
    }

    /// Create with default threshold (0.3).
    pub fn default() -> Self {
        Self::new(0.3)
    }

    /// Record a cache hit for the current branch.
    pub fn record_hit(&self) {
        self.branch_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss for the current branch.
    pub fn record_miss(&self) {
        self.branch_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if a decomposition branch is productive.
    ///
    /// Productivity = novelty = cache miss rate.
    /// - `novelty >= min_novelty` → productive (exploring new territory)
    /// - `novelty < min_novelty` → unproductive (stuck in known states)
    ///
    /// Returns true if the branch should be kept.
    pub fn is_productive(&self) -> bool {
        let hits = self.branch_hits.load(Ordering::Relaxed);
        let misses = self.branch_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        match total {
            0 => true, // no data yet, assume productive
            _ => {
                let novelty = misses as f32 / total as f32;
                novelty >= self.min_novelty
            }
        }
    }

    /// Get the novelty score (cache miss rate) for the current branch.
    pub fn novelty(&self) -> f32 {
        let hits = self.branch_hits.load(Ordering::Relaxed);
        let misses = self.branch_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        match total {
            0 => 1.0,
            _ => misses as f32 / total as f32,
        }
    }

    /// Reset counters for a new branch evaluation.
    pub fn reset_branch(&self) {
        self.branch_hits.store(0, Ordering::Relaxed);
        self.branch_misses.store(0, Ordering::Relaxed);
    }

    /// Get the minimum novelty threshold.
    pub fn min_novelty(&self) -> f32 {
        self.min_novelty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn productive_branch_high_miss_rate() {
        let reviewer = DecompositionReviewer::new(0.3);
        // 7 misses, 3 hits → novelty = 0.7 ≥ 0.3 → productive
        for _ in 0..7 {
            reviewer.record_miss();
        }
        for _ in 0..3 {
            reviewer.record_hit();
        }
        assert!(reviewer.is_productive());
        assert!((reviewer.novelty() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn unproductive_branch_high_hit_rate() {
        let reviewer = DecompositionReviewer::new(0.3);
        // 1 miss, 9 hits → novelty = 0.1 < 0.3 → unproductive
        reviewer.record_miss();
        for _ in 0..9 {
            reviewer.record_hit();
        }
        assert!(!reviewer.is_productive());
        assert!((reviewer.novelty() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn no_data_assumes_productive() {
        let reviewer = DecompositionReviewer::new(0.3);
        assert!(reviewer.is_productive());
        assert!((reviewer.novelty() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reset_clears_counters() {
        let reviewer = DecompositionReviewer::new(0.3);
        for _ in 0..5 {
            reviewer.record_hit();
        }
        for _ in 0..5 {
            reviewer.record_miss();
        }
        assert!((reviewer.novelty() - 0.5).abs() < f32::EPSILON);

        reviewer.reset_branch();
        assert!(reviewer.is_productive()); // back to "no data" state
        assert!((reviewer.novelty() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn novelty_score_computation() {
        let reviewer = DecompositionReviewer::new(0.5);
        // Exact boundary: 5 misses, 5 hits → novelty = 0.5
        for _ in 0..5 {
            reviewer.record_miss();
        }
        for _ in 0..5 {
            reviewer.record_hit();
        }
        assert!((reviewer.novelty() - 0.5).abs() < f32::EPSILON);
        assert!(reviewer.is_productive()); // 0.5 >= 0.5

        reviewer.reset_branch();
        // 4 misses, 6 hits → novelty = 0.4 < 0.5
        for _ in 0..4 {
            reviewer.record_miss();
        }
        for _ in 0..6 {
            reviewer.record_hit();
        }
        assert!(!reviewer.is_productive());
    }

    #[test]
    fn threshold_clamping() {
        // Below 0.0 → clamped to 0.0 → everything is productive
        let reviewer = DecompositionReviewer::new(-0.5);
        assert!((reviewer.min_novelty() - 0.0).abs() < f32::EPSILON);
        for _ in 0..10 {
            reviewer.record_hit();
        }
        assert!(reviewer.is_productive());

        // Above 1.0 → clamped to 1.0 → only 100% miss rate is productive
        let strict = DecompositionReviewer::new(1.5);
        assert!((strict.min_novelty() - 1.0).abs() < f32::EPSILON);
        strict.record_hit();
        strict.record_miss();
        assert!(!strict.is_productive()); // 0.5 < 1.0

        strict.reset_branch();
        strict.record_miss(); // 100% miss rate
        assert!(strict.is_productive());
    }
}
