//! ChainFolder — main folding logic implementing ScreeningPruner — Plan 195 T3.
//!
//! The `ChainFolder` ranks reasoning steps by attention importance, then
//! uses binary search to find the minimal set of steps that preserves
//! verification. It implements `ScreeningPruner` so it plugs directly
//! into the DDTree screening pipeline.

use super::attention_importance::AttentionImportance;
use super::types::{FoldContext, FoldDecision, FoldResult, StepBoundary};

/// Main chain folder — prunes redundant reasoning steps.
///
/// Implements [`ScreeningPruner`] for DDTree integration. The relevance
/// score is derived from per-step attention importance: essential and
/// anchor steps get `1.0`, foldable steps get `0.0`.
#[derive(Debug, Clone)]
pub struct ChainFolder {
    /// Fraction of steps to keep (0.0–1.0). Higher = less aggressive folding.
    fold_budget: f32,
    /// Attention importance scorer.
    importance_scorer: AttentionImportance,
    /// Cached fold decisions from last `binary_search_fold`.
    decisions: Vec<FoldDecision>,
    /// Cached boundaries from last fold.
    boundaries: Vec<StepBoundary>,
}

impl ChainFolder {
    /// Create a new chain folder with the given fold budget.
    ///
    /// `fold_budget` = fraction of steps to keep (0.0 = fold everything,
    /// 1.0 = keep everything). Default is 0.7.
    pub fn new(fold_budget: f32) -> Self {
        Self {
            fold_budget: fold_budget.clamp(0.0, 1.0),
            importance_scorer: AttentionImportance::new(),
            decisions: Vec::new(),
            boundaries: Vec::new(),
        }
    }

    /// Current fold budget (fraction of steps to keep).
    #[inline]
    pub fn fold_budget(&self) -> f32 {
        self.fold_budget
    }

    /// Set the fold budget.
    pub fn set_fold_budget(&mut self, budget: f32) {
        self.fold_budget = budget.clamp(0.0, 1.0);
    }

    /// Get cached fold decisions.
    pub fn decisions(&self) -> &[FoldDecision] {
        &self.decisions
    }

    /// Binary search fold — find the minimal set of steps that preserves correctness.
    ///
    /// Given attention scores and step boundaries, binary searches on the retention
    /// ratio to find the most aggressive fold that still passes verification.
    ///
    /// Returns a `FoldResult` with verification status and token savings estimate.
    pub fn binary_search_fold(&mut self, context: &FoldContext) -> FoldResult {
        let total_steps = context.step_count();
        if total_steps == 0 {
            return FoldResult::no_fold(0);
        }

        // Compute importance scores per step.
        let importance = self
            .importance_scorer
            .score_steps(&context.importance_scores, &context.boundaries);

        if importance.is_empty() {
            return FoldResult::no_fold(total_steps);
        }

        let target_keep = (self.fold_budget * total_steps as f32).ceil() as usize;
        let mut best_keep = total_steps;

        // Rank steps by importance (ascending) to find the least important ones.
        let mut indexed: Vec<(usize, f32)> = importance.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Binary search: how many of the least-important steps can we fold?
        let mut lo = 0_usize;
        let mut hi = total_steps.saturating_sub(target_keep);

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let kept = total_steps - mid;

            // Verify: anchors must be kept, we must keep at least target_keep steps.
            if verify_fold(&context.boundaries, &indexed, mid) && kept >= target_keep {
                best_keep = kept;
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Apply the best fold.
        let fold_count = total_steps - best_keep;
        let decisions = build_decisions(&context.boundaries, &indexed, fold_count);

        let kept = decisions
            .iter()
            .filter(|d| **d != FoldDecision::Fold)
            .count();
        let folded = total_steps - kept;

        // Estimate token savings from boundaries.
        let tokens_saved = estimate_tokens_saved(&context.boundaries, &decisions);

        self.decisions = decisions;
        self.boundaries = context.boundaries.clone();

        let retention_ratio = if total_steps > 0 {
            kept as f32 / total_steps as f32
        } else {
            1.0
        };

        FoldResult {
            total_steps,
            kept_steps: kept,
            folded_steps: folded,
            tokens_saved,
            retention_ratio,
            verification_passed: true,
        }
    }

    /// Map a token depth to a step index using cached boundaries.
    fn step_index_for_depth(&self, depth: usize) -> Option<usize> {
        match self
            .boundaries
            .binary_search_by(|b| b.token_pos.cmp(&depth))
        {
            Ok(idx) => Some(idx),
            Err(idx) => {
                if idx == 0 {
                    None
                } else {
                    Some(idx - 1)
                }
            }
        }
    }
}

/// Implement ScreeningPruner for DDTree integration.
///
/// Returns relevance based on cached fold decisions.
/// - Steps marked `Keep` or `Anchor` → 1.0
/// - Steps marked `Fold` → 0.0
/// - If no cached decisions → 1.0 (passthrough, safe default)
impl crate::speculative::types::ScreeningPruner for ChainFolder {
    fn relevance(&self, depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        let step_idx = self.step_index_for_depth(depth);
        match step_idx {
            Some(idx) => match self.decisions.get(idx) {
                Some(FoldDecision::Fold) => 0.0,
                Some(FoldDecision::Keep | FoldDecision::Anchor) => 1.0,
                None => 1.0,
            },
            None => 1.0,
        }
    }
}

impl Default for ChainFolder {
    fn default() -> Self {
        Self::new(0.7)
    }
}

// ── Helper functions ────────────────────────────────────────────

/// Verify that a fold is safe: anchors are never folded.
///
/// Returns `true` if folding `fold_count` least-important steps preserves
/// all anchor steps.
fn verify_fold(boundaries: &[StepBoundary], ranked: &[(usize, f32)], fold_count: usize) -> bool {
    if fold_count == 0 {
        return true;
    }

    // Steps to fold are the first `fold_count` entries in `ranked` (lowest importance).
    for (step_idx, _) in ranked.iter().take(fold_count) {
        match boundaries.get(*step_idx) {
            Some(b) if b.is_anchor => return false,
            _ => continue,
        }
    }

    true
}

/// Build fold decisions for each step.
fn build_decisions(
    boundaries: &[StepBoundary],
    ranked: &[(usize, f32)],
    fold_count: usize,
) -> Vec<FoldDecision> {
    let total = boundaries.len();
    let mut decisions = vec![FoldDecision::Keep; total];

    // Mark anchors.
    for (i, b) in boundaries.iter().enumerate() {
        if b.is_anchor {
            decisions[i] = FoldDecision::Anchor;
        }
    }

    // Fold the least-important non-anchor steps.
    for (step_idx, _) in ranked.iter().take(fold_count) {
        match boundaries.get(*step_idx) {
            Some(b) if !b.is_anchor => decisions[*step_idx] = FoldDecision::Fold,
            _ => continue,
        }
    }

    decisions
}

/// Estimate token savings from fold decisions.
fn estimate_tokens_saved(boundaries: &[StepBoundary], decisions: &[FoldDecision]) -> usize {
    let total_tokens = boundaries.last().map(|b| b.token_pos).unwrap_or(0);

    let mut saved = 0_usize;
    for (i, decision) in decisions.iter().enumerate() {
        if *decision == FoldDecision::Fold {
            let start = boundaries[i].token_pos;
            let end = boundaries
                .get(i + 1)
                .map(|b| b.token_pos)
                .unwrap_or(total_tokens);
            saved += end.saturating_sub(start);
        }
    }

    saved
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(
        scores: &[f32],
        positions: &[usize],
        anchors: &[usize],
        budget: f32,
    ) -> FoldContext {
        let boundaries: Vec<StepBoundary> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| StepBoundary::new(pos, i, anchors.contains(&i)))
            .collect();
        FoldContext {
            importance_scores: scores.to_vec(),
            boundaries,
            fold_budget: budget,
        }
    }

    #[test]
    fn test_chain_folder_default() {
        let cf = ChainFolder::default();
        assert!((cf.fold_budget() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chain_folder_new_clamps_budget() {
        let cf = ChainFolder::new(-0.5);
        assert!((cf.fold_budget()).abs() < f32::EPSILON);

        let cf = ChainFolder::new(2.0);
        assert!((cf.fold_budget() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_binary_search_fold_empty() {
        let mut cf = ChainFolder::new(0.7);
        let ctx = FoldContext {
            importance_scores: vec![],
            boundaries: vec![],
            fold_budget: 0.7,
        };
        let result = cf.binary_search_fold(&ctx);
        assert_eq!(result.total_steps, 0);
    }

    #[test]
    fn test_binary_search_fold_keeps_anchors() {
        // 5 steps, step 0 and 3 are anchors, low importance on step 2.
        let scores: Vec<f32> = vec![0.9; 10]
            .into_iter()
            .chain(vec![0.9; 10])
            .chain(vec![0.1; 10]) // step 2: low importance
            .chain(vec![0.9; 10])
            .chain(vec![0.9; 10])
            .collect();

        let ctx = make_context(&scores, &[0, 10, 20, 30, 40], &[0, 3], 0.6);
        let mut cf = ChainFolder::new(0.6);
        let result = cf.binary_search_fold(&ctx);

        // Anchors (step 0, 3) must not be folded.
        let decisions = cf.decisions();
        assert_eq!(decisions[0], FoldDecision::Anchor);
        assert_eq!(decisions[3], FoldDecision::Anchor);

        // Step 2 (low importance) should be folded.
        assert_eq!(decisions[2], FoldDecision::Fold);

        assert!(result.verification_passed);
    }

    #[test]
    fn test_binary_search_fold_budget_one() {
        // Budget 1.0 = keep everything.
        let scores = vec![0.5; 20];
        let ctx = make_context(&scores, &[0, 5, 10, 15], &[], 1.0);
        let mut cf = ChainFolder::new(1.0);
        let result = cf.binary_search_fold(&ctx);

        assert_eq!(result.folded_steps, 0);
        assert!((result.retention_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_fold_budget() {
        let mut cf = ChainFolder::new(0.5);
        cf.set_fold_budget(0.9);
        assert!((cf.fold_budget() - 0.9).abs() < f32::EPSILON);
        cf.set_fold_budget(-1.0);
        assert!((cf.fold_budget()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_verify_fold_no_fold() {
        let boundaries = vec![
            StepBoundary::new(0, 0, true),
            StepBoundary::new(10, 1, false),
        ];
        let ranked = vec![(0, 0.5), (1, 0.3)];
        assert!(verify_fold(&boundaries, &ranked, 0));
    }

    #[test]
    fn test_verify_fold_anchor_protection() {
        let boundaries = vec![
            StepBoundary::new(0, 0, true), // anchor
            StepBoundary::new(10, 1, false),
        ];
        let ranked = vec![(0, 0.1), (1, 0.9)]; // step 0 is lowest importance
        // Trying to fold 1 step would hit the anchor → should fail.
        assert!(!verify_fold(&boundaries, &ranked, 1));
    }

    #[test]
    fn test_estimate_tokens_saved() {
        let boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        let decisions = vec![FoldDecision::Keep, FoldDecision::Fold, FoldDecision::Keep];
        let saved = estimate_tokens_saved(&boundaries, &decisions);
        assert_eq!(saved, 10); // Step 1 spans tokens 10..20
    }

    #[test]
    fn test_step_index_for_depth() {
        let mut cf = ChainFolder::new(0.7);
        cf.boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        cf.decisions = vec![FoldDecision::Keep, FoldDecision::Fold, FoldDecision::Keep];

        assert_eq!(cf.step_index_for_depth(0), Some(0));
        assert_eq!(cf.step_index_for_depth(5), Some(0));
        assert_eq!(cf.step_index_for_depth(15), Some(1));
        assert_eq!(cf.step_index_for_depth(25), Some(2));
    }
}
