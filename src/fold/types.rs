//! ThoughtFold core types — Plan 195.
//!
//! Defines `FoldDecision`, `FoldResult`, `FoldContext`, `FoldStats`, and
//! `StepBoundary` used across all chain-folding submodules.
//!
//! _Root-resident by design (Issue 033 §C, Option C)._ Defines fold types
//! consumed by root-only `crate::speculative::types::ScreeningPruner` and
//! `ThinkingController`.

// ── StepBoundary ────────────────────────────────────────────────

/// A reasoning step boundary detected in the CoT text.
///
/// Maps a token position to a step index, optionally marking it as an
/// anchor (must-keep reasoning boundary like think-tag transitions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBoundary {
    /// Token position where this step begins.
    pub token_pos: usize,
    /// Step index (0-based, sequential).
    pub step_index: usize,
    /// Whether this step is an anchor (must keep).
    pub is_anchor: bool,
}

impl StepBoundary {
    /// Create a new step boundary.
    #[inline]
    pub const fn new(token_pos: usize, step_index: usize, is_anchor: bool) -> Self {
        Self {
            token_pos,
            step_index,
            is_anchor,
        }
    }
}

// ── FoldDecision ────────────────────────────────────────────────

/// Decision for a reasoning step during chain folding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FoldDecision {
    /// Keep this step — essential for correctness.
    Keep = 0,
    /// Fold (remove) this step — redundant or low-importance.
    Fold = 1,
    /// Anchor step — must keep, marks a reasoning boundary.
    Anchor = 2,
}

impl FoldDecision {
    /// Relevance score for [`ScreeningPruner`] integration.
    ///
    /// Returns `1.0` for kept/anchor steps, `0.0` for folded steps.
    #[inline]
    pub const fn relevance(&self) -> f32 {
        match self {
            Self::Keep | Self::Anchor => 1.0,
            Self::Fold => 0.0,
        }
    }
}

// ── FoldResult ──────────────────────────────────────────────────

/// Result of a chain fold operation.
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Total steps in original chain.
    pub total_steps: usize,
    /// Steps that were kept (essential + anchors).
    pub kept_steps: usize,
    /// Steps that were folded (removed).
    pub folded_steps: usize,
    /// Estimated token savings.
    pub tokens_saved: usize,
    /// Retention ratio (kept / total).
    pub retention_ratio: f32,
    /// Whether verification passed.
    pub verification_passed: bool,
}

impl FoldResult {
    /// Create a no-op fold result (everything kept, nothing folded).
    pub fn no_fold(total_steps: usize) -> Self {
        Self {
            total_steps,
            kept_steps: total_steps,
            folded_steps: 0,
            tokens_saved: 0,
            retention_ratio: 1.0,
            verification_passed: true,
        }
    }
}

// ── FoldContext ─────────────────────────────────────────────────

/// Context provided to the chain folder for making fold decisions.
#[derive(Debug, Clone)]
pub struct FoldContext {
    /// Per-step attention importance scores.
    pub importance_scores: Vec<f32>,
    /// Step boundaries (token positions).
    pub boundaries: Vec<StepBoundary>,
    /// Current fold budget (0.0–1.0, fraction of steps to keep).
    pub fold_budget: f32,
}

impl FoldContext {
    /// Number of steps in this context.
    pub fn step_count(&self) -> usize {
        self.boundaries.len()
    }
}

// ── FoldStats ───────────────────────────────────────────────────

/// Statistics about fold operations for feedback integration.
#[derive(Debug, Clone, Default)]
pub struct FoldStats {
    /// Total tokens saved across all queries.
    pub total_tokens_saved: usize,
    /// Total steps folded.
    pub total_steps_folded: usize,
    /// Number of queries where folding was applied.
    pub queries_folded: usize,
    /// Verification pass rate (0.0–1.0).
    pub verification_pass_rate: f32,
}

impl FoldStats {
    /// Record a single fold result into running stats.
    pub fn record(&mut self, result: &FoldResult) {
        self.total_tokens_saved += result.tokens_saved;
        self.total_steps_folded += result.folded_steps;
        self.queries_folded += 1;

        // Running average verification pass rate (exponential moving average).
        let alpha = 1.0 / self.queries_folded as f32;
        let passed = if result.verification_passed { 1.0 } else { 0.0 };
        self.verification_pass_rate += alpha * (passed - self.verification_pass_rate);
    }
}

// ── ThinkingFoldFeedback ────────────────────────────────────────

/// Fold feedback for integration with ThinkingController.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkingFoldFeedback {
    /// Tokens saved in this fold operation.
    pub tokens_saved: usize,
    /// Steps folded in this fold operation.
    pub steps_folded: usize,
    /// Fold budget used (0.0–1.0).
    pub fold_budget: f32,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_decision_relevance() {
        assert_eq!(FoldDecision::Keep.relevance(), 1.0);
        assert_eq!(FoldDecision::Fold.relevance(), 0.0);
        assert_eq!(FoldDecision::Anchor.relevance(), 1.0);
    }

    #[test]
    fn test_step_boundary_new() {
        let b = StepBoundary::new(42, 3, true);
        assert_eq!(b.token_pos, 42);
        assert_eq!(b.step_index, 3);
        assert!(b.is_anchor);
    }

    #[test]
    fn test_fold_result_no_fold() {
        let r = FoldResult::no_fold(10);
        assert_eq!(r.total_steps, 10);
        assert_eq!(r.kept_steps, 10);
        assert_eq!(r.folded_steps, 0);
        assert_eq!(r.tokens_saved, 0);
        assert!((r.retention_ratio - 1.0).abs() < f32::EPSILON);
        assert!(r.verification_passed);
    }

    #[test]
    fn test_fold_context_step_count() {
        let ctx = FoldContext {
            importance_scores: vec![0.5; 5],
            boundaries: vec![
                StepBoundary::new(0, 0, false),
                StepBoundary::new(10, 1, true),
                StepBoundary::new(20, 2, false),
                StepBoundary::new(30, 3, true),
                StepBoundary::new(40, 4, false),
            ],
            fold_budget: 0.7,
        };
        assert_eq!(ctx.step_count(), 5);
    }

    #[test]
    fn test_fold_stats_record() {
        let mut stats = FoldStats::default();
        let r1 = FoldResult {
            total_steps: 10,
            kept_steps: 7,
            folded_steps: 3,
            tokens_saved: 100,
            retention_ratio: 0.7,
            verification_passed: true,
        };
        stats.record(&r1);
        assert_eq!(stats.total_tokens_saved, 100);
        assert_eq!(stats.total_steps_folded, 3);
        assert_eq!(stats.queries_folded, 1);
        assert!((stats.verification_pass_rate - 1.0).abs() < f32::EPSILON);

        // Record a failed verification.
        let r2 = FoldResult {
            total_steps: 8,
            kept_steps: 8,
            folded_steps: 0,
            tokens_saved: 0,
            retention_ratio: 1.0,
            verification_passed: false,
        };
        stats.record(&r2);
        assert_eq!(stats.queries_folded, 2);
        assert!((stats.verification_pass_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fold_decision_repr_u8() {
        assert_eq!(FoldDecision::Keep as u8, 0);
        assert_eq!(FoldDecision::Fold as u8, 1);
        assert_eq!(FoldDecision::Anchor as u8, 2);
    }
}
