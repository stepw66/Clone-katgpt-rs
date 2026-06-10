//! ThinkingController integration — Plan 195 T5.
//!
//! Extension functions to integrate chain folding with the ThinkingController
//! feedback loop. Converts fold statistics into thinking feedback.

use super::types::{FoldResult, FoldStats, ThinkingFoldFeedback};

/// Convert a fold result into thinking feedback for the controller.
pub fn fold_thinking_feedback(result: &FoldResult, fold_budget: f32) -> ThinkingFoldFeedback {
    ThinkingFoldFeedback {
        tokens_saved: result.tokens_saved,
        steps_folded: result.folded_steps,
        fold_budget,
    }
}

/// Convert accumulated fold stats into a summary feedback.
pub fn fold_stats_feedback(stats: &FoldStats) -> ThinkingFoldFeedback {
    ThinkingFoldFeedback {
        tokens_saved: stats.total_tokens_saved,
        steps_folded: stats.total_steps_folded,
        fold_budget: 0.0, // Stats don't track per-query budget
    }
}

/// Calculate the effective token reduction ratio from fold stats.
///
/// Returns 0.0 if no queries have been folded.
pub fn token_reduction_ratio(stats: &FoldStats) -> f32 {
    if stats.queries_folded == 0 {
        return 0.0;
    }
    stats.total_tokens_saved as f32 / stats.queries_folded as f32
}

/// Calculate the effective step reduction ratio from fold stats.
///
/// Returns 0.0 if no queries have been folded.
pub fn step_reduction_ratio(stats: &FoldStats) -> f32 {
    if stats.queries_folded == 0 {
        return 0.0;
    }
    stats.total_steps_folded as f32 / stats.queries_folded as f32
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_thinking_feedback() {
        let result = FoldResult {
            total_steps: 10,
            kept_steps: 7,
            folded_steps: 3,
            tokens_saved: 50,
            retention_ratio: 0.7,
            verification_passed: true,
        };

        let feedback = fold_thinking_feedback(&result, 0.7);
        assert_eq!(feedback.tokens_saved, 50);
        assert_eq!(feedback.steps_folded, 3);
        assert!((feedback.fold_budget - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fold_stats_feedback() {
        let stats = FoldStats {
            total_tokens_saved: 500,
            total_steps_folded: 30,
            queries_folded: 10,
            verification_pass_rate: 0.9,
        };

        let feedback = fold_stats_feedback(&stats);
        assert_eq!(feedback.tokens_saved, 500);
        assert_eq!(feedback.steps_folded, 30);
    }

    #[test]
    fn test_token_reduction_ratio_empty() {
        let stats = FoldStats::default();
        assert_eq!(token_reduction_ratio(&stats), 0.0);
    }

    #[test]
    fn test_token_reduction_ratio() {
        let stats = FoldStats {
            total_tokens_saved: 100,
            total_steps_folded: 10,
            queries_folded: 5,
            verification_pass_rate: 1.0,
        };
        let ratio = token_reduction_ratio(&stats);
        assert!((ratio - 20.0).abs() < f32::EPSILON); // 100 / 5
    }

    #[test]
    fn test_step_reduction_ratio_empty() {
        let stats = FoldStats::default();
        assert_eq!(step_reduction_ratio(&stats), 0.0);
    }

    #[test]
    fn test_step_reduction_ratio() {
        let stats = FoldStats {
            total_tokens_saved: 100,
            total_steps_folded: 15,
            queries_folded: 3,
            verification_pass_rate: 0.8,
        };
        let ratio = step_reduction_ratio(&stats);
        assert!((ratio - 5.0).abs() < f32::EPSILON); // 15 / 3
    }
}
