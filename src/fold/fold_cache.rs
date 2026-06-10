//! FoldCache — KV cache rollback support — Plan 195 T4.
//!
//! Provides the interface for truncating and replaying KV cache entries
//! based on fold decisions. This is a lightweight wrapper since actual
//! KV cache access is managed by the transformer forward pass.

use super::types::{FoldDecision, StepBoundary};

/// KV cache management for chain folding.
///
/// Tracks which steps should be truncated or replayed after a fold operation.
/// The actual KV cache manipulation is delegated to the transformer runtime;
/// this struct provides the planning layer.
#[derive(Debug, Clone)]
pub struct FoldCache {
    /// Step indices to keep after folding (sorted ascending).
    essential_steps: Vec<usize>,
    /// Total number of steps before folding.
    total_steps: usize,
    /// Token position to truncate to (after folding).
    truncate_pos: Option<usize>,
}

impl FoldCache {
    /// Create a new fold cache for the given step count.
    pub fn new(total_steps: usize) -> Self {
        Self {
            essential_steps: Vec::new(),
            total_steps,
            truncate_pos: None,
        }
    }

    /// Truncate the KV cache to the given step boundary.
    ///
    /// Records the truncation position. The actual truncation happens
    /// when the transformer runtime calls `truncate_pos()`.
    pub fn truncate_to_step(&mut self, step: usize, boundaries: &[StepBoundary]) {
        if step >= boundaries.len() {
            return;
        }
        self.truncate_pos = Some(boundaries[step].token_pos);
    }

    /// Mark essential steps for replay after folding.
    ///
    /// Only non-folded steps are replayed. Anchor steps are always included.
    pub fn replay_essential(&mut self, decisions: &[FoldDecision], _boundaries: &[StepBoundary]) {
        self.essential_steps.clear();
        for (i, decision) in decisions.iter().enumerate() {
            match decision {
                FoldDecision::Keep | FoldDecision::Anchor => {
                    self.essential_steps.push(i);
                }
                FoldDecision::Fold => continue,
            }
        }
    }

    /// Get the truncation position (if truncation was requested).
    #[inline]
    pub fn truncate_pos(&self) -> Option<usize> {
        self.truncate_pos
    }

    /// Get the essential steps to replay.
    #[inline]
    pub fn essential_steps(&self) -> &[usize] {
        &self.essential_steps
    }

    /// Total steps before folding.
    #[inline]
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Number of essential steps (kept after folding).
    #[inline]
    pub fn essential_count(&self) -> usize {
        self.essential_steps.len()
    }

    /// Compute token positions for essential steps.
    ///
    /// Returns the token positions that need to be replayed.
    pub fn essential_token_positions(&self, boundaries: &[StepBoundary]) -> Vec<usize> {
        self.essential_steps
            .iter()
            .filter_map(|&step| boundaries.get(step).map(|b| b.token_pos))
            .collect()
    }

    /// Reset the cache for a new fold operation.
    pub fn reset(&mut self) {
        self.essential_steps.clear();
        self.truncate_pos = None;
    }
}

impl Default for FoldCache {
    fn default() -> Self {
        Self::new(0)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_boundaries() -> Vec<StepBoundary> {
        vec![
            StepBoundary::new(0, 0, true),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
            StepBoundary::new(30, 3, true),
            StepBoundary::new(40, 4, false),
        ]
    }

    #[test]
    fn test_fold_cache_new() {
        let cache = FoldCache::new(5);
        assert_eq!(cache.total_steps(), 5);
        assert!(cache.essential_steps().is_empty());
        assert!(cache.truncate_pos().is_none());
    }

    #[test]
    fn test_truncate_to_step() {
        let mut cache = FoldCache::new(5);
        let boundaries = make_boundaries();

        cache.truncate_to_step(2, &boundaries);
        assert_eq!(cache.truncate_pos(), Some(20));
    }

    #[test]
    fn test_truncate_to_step_out_of_bounds() {
        let mut cache = FoldCache::new(5);
        let boundaries = make_boundaries();

        cache.truncate_to_step(10, &boundaries);
        assert!(cache.truncate_pos().is_none());
    }

    #[test]
    fn test_replay_essential() {
        let mut cache = FoldCache::new(5);
        let boundaries = make_boundaries();

        let decisions = vec![
            FoldDecision::Anchor, // step 0
            FoldDecision::Fold,   // step 1
            FoldDecision::Fold,   // step 2
            FoldDecision::Anchor, // step 3
            FoldDecision::Keep,   // step 4
        ];

        cache.replay_essential(&decisions, &boundaries);
        assert_eq!(cache.essential_steps(), &[0, 3, 4]);
        assert_eq!(cache.essential_count(), 3);
    }

    #[test]
    fn test_essential_token_positions() {
        let mut cache = FoldCache::new(5);
        let boundaries = make_boundaries();

        let decisions = vec![
            FoldDecision::Anchor,
            FoldDecision::Fold,
            FoldDecision::Keep,
            FoldDecision::Fold,
            FoldDecision::Keep,
        ];

        cache.replay_essential(&decisions, &boundaries);
        let positions = cache.essential_token_positions(&boundaries);
        assert_eq!(positions, vec![0, 20, 40]);
    }

    #[test]
    fn test_reset() {
        let mut cache = FoldCache::new(5);
        let boundaries = make_boundaries();

        cache.truncate_to_step(1, &boundaries);
        cache.replay_essential(&[FoldDecision::Keep, FoldDecision::Keep], &boundaries[..2]);

        cache.reset();
        assert!(cache.essential_steps().is_empty());
        assert!(cache.truncate_pos().is_none());
    }

    #[test]
    fn test_default() {
        let cache = FoldCache::default();
        assert_eq!(cache.total_steps(), 0);
    }
}
