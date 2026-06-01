//! Compression-adaptive decode budget — PFlash complexity signal for DDTree budget scaling.
//!
//! Uses the prompt compression ratio (a free byproduct of prefill scoring) to dynamically
//! scale DDTree budget per-prompt. Simple prompts → less search. Complex → more.
//!
//! # Feature flag
//! `budget_adaptation` — Plan 167, Research R050

use crate::speculative::types::BudgetAdaptation;

/// Derive per-prompt tree_budget from base + complexity signal.
///
/// Returns budget clamped to [base/2, base*2].
///
/// # Arguments
/// * `base_budget` — default tree budget from domain config
/// * `compression_ratio` — r ∈ (0, 1]: fraction of blocks/selected that matter
/// * `mode` — adaptation strategy
///
/// # Scaling curve (Compression mode)
/// ```text
/// r=0.0 → scale=0.5  (budget halved, simple prompt)
/// r=0.5 → scale=1.25 (budget slightly above base)
/// r=1.0 → scale=2.0  (budget doubled, complex prompt)
/// ```
pub fn adaptive_tree_budget(
    base_budget: usize,
    compression_ratio: f32,
    mode: BudgetAdaptation,
) -> usize {
    match mode {
        BudgetAdaptation::Off => base_budget,
        BudgetAdaptation::Compression => {
            let r = compression_ratio.clamp(0.0, 1.0);
            // Linear scale: f(0)=0.5, f(0.5)=1.25, f(1)=2.0
            let scale = 0.5 + 1.5 * r;
            let adapted = (base_budget as f32 * scale) as usize;
            adapted.max(base_budget / 2).min(base_budget * 2)
        }
        BudgetAdaptation::Entropy => {
            // TODO: derive from first-marginal entropy (future work)
            base_budget
        }
    }
}

/// Derive compression ratio from block selection results.
///
/// Given the total number of blocks and the number selected by `block_select`,
/// returns the fraction r ∈ (0, 1] that passed the importance filter.
///
/// This is a zero-alloc computation — just a division.
#[inline]
pub fn compression_ratio(selected_count: usize, total_count: usize) -> f32 {
    if total_count == 0 {
        return 1.0; // no blocks = nothing to compress, treat as complex
    }
    (selected_count as f32) / (total_count as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_budget_off_returns_base() {
        assert_eq!(adaptive_tree_budget(100, 0.5, BudgetAdaptation::Off), 100);
    }

    #[test]
    fn test_adaptive_budget_compression_midpoint() {
        let budget = adaptive_tree_budget(100, 0.5, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*0.5 = 1.25 → 125
        assert_eq!(budget, 125);
    }

    #[test]
    fn test_adaptive_budget_compression_low() {
        let budget = adaptive_tree_budget(100, 0.0, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*0.0 = 0.5 → clamped to max(50, 50) = 50
        assert_eq!(budget, 50);
    }

    #[test]
    fn test_adaptive_budget_compression_high() {
        let budget = adaptive_tree_budget(100, 1.0, BudgetAdaptation::Compression);
        // scale = 0.5 + 1.5*1.0 = 2.0 → 200
        assert_eq!(budget, 200);
    }

    #[test]
    fn test_adaptive_budget_clamped_lower() {
        // Even with r=0.01, budget shouldn't go below base/2
        let budget = adaptive_tree_budget(100, 0.01, BudgetAdaptation::Compression);
        assert!(budget >= 50, "budget {} < base/2 = 50", budget);
    }

    #[test]
    fn test_adaptive_budget_clamped_upper() {
        // Even with r=1.0, budget shouldn't exceed base*2
        let budget = adaptive_tree_budget(100, 1.0, BudgetAdaptation::Compression);
        assert!(budget <= 200, "budget {} > base*2 = 200", budget);
    }

    #[test]
    fn test_adaptive_budget_entropy_returns_base() {
        assert_eq!(adaptive_tree_budget(100, 0.5, BudgetAdaptation::Entropy), 100);
    }

    #[test]
    fn test_compression_ratio_normal() {
        assert!((compression_ratio(5, 10) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compression_ratio_zero_total() {
        assert_eq!(compression_ratio(0, 0), 1.0);
    }

    #[test]
    fn test_compression_ratio_all_selected() {
        assert!((compression_ratio(10, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaling_curve_monotonic() {
        let budgets: Vec<usize> = (0..=10)
            .map(|r| adaptive_tree_budget(1000, r as f32 / 10.0, BudgetAdaptation::Compression))
            .collect();
        for w in budgets.windows(2) {
            assert!(w[0] <= w[1], "not monotonic: {} > {}", w[0], w[1]);
        }
    }
}
