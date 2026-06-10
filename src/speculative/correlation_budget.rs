//! Correlation Budget Allocation — Data-Driven Speculative Depth (Plan 200).
//!
//! Replaces heuristic `PositionWeightedBudget` gamma with EMA-tracked agreement rates:
//! - High agreement → allocate more budget (high-confidence positions)
//! - Low agreement → allocate less budget (uncertain positions)
//!
//! Feature flag: `corr_budget`

// ── CorrelationBudgetAllocator ─────────────────────────────────

/// Data-driven budget allocation via empirical draft↔target agreement.
///
/// Tracks acceptance rates per speculative depth using exponential moving average.
/// After each decode step, call `update(depth, accepted)` with the verification result.
/// The allocator then distributes `max_budget` across depths proportional to agreement.
///
/// # Convergence
/// After ~100 steps (with α=0.1), EMA converges to true acceptance rate.
/// First 100 steps use boosted α (0.3) for faster adaptation.
///
/// # Feature flag
/// `corr_budget` — Plan 200, Research R178
#[derive(Debug, Clone)]
pub struct CorrelationBudgetAllocator {
    /// Per-depth agreement rate (EMA)
    depth_agreement_rate: Vec<f32>,
    /// EMA smoothing factor (standard)
    ema_alpha: f32,
    /// EMA smoothing factor (warmup — first `warmup_steps` updates)
    ema_alpha_warmup: f32,
    /// Number of updates so far (for warmup detection)
    update_count: usize,
    /// Warmup period: use `ema_alpha_warmup` for first N updates
    warmup_steps: usize,
    /// Minimum budget per depth (floor)
    min_budget_per_depth: usize,
    /// Default agreement rate for unseen depths
    default_rate: f32,
}

impl Default for CorrelationBudgetAllocator {
    fn default() -> Self {
        Self {
            depth_agreement_rate: Vec::new(),
            ema_alpha: 0.1,
            ema_alpha_warmup: 0.3,
            update_count: 0,
            warmup_steps: 100,
            min_budget_per_depth: 1,
            default_rate: 0.5,
        }
    }
}

impl CorrelationBudgetAllocator {
    /// Create a new allocator with custom EMA alpha.
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            ema_alpha,
            ..Self::default()
        }
    }

    /// Get the current EMA alpha (warmup-aware).
    fn current_alpha(&self) -> f32 {
        match self.update_count < self.warmup_steps {
            true => self.ema_alpha_warmup,
            false => self.ema_alpha,
        }
    }

    /// Get the agreement rate for a specific depth.
    /// Returns `default_rate` for unseen depths.
    pub fn agreement_rate(&self, depth: usize) -> f32 {
        match self.depth_agreement_rate.get(depth) {
            Some(&rate) => rate,
            None => self.default_rate,
        }
    }

    /// Update agreement rates from latest speculative decode results.
    /// Called after each decode step with acceptance/rejection data.
    pub fn update(&mut self, depth: usize, accepted: bool) {
        // Extend if needed
        while self.depth_agreement_rate.len() <= depth {
            self.depth_agreement_rate.push(self.default_rate);
        }

        let alpha = self.current_alpha();
        let old = self.depth_agreement_rate[depth];
        let new_val = match accepted {
            true => 1.0_f32,
            false => 0.0_f32,
        };
        self.depth_agreement_rate[depth] = old * (1.0 - alpha) + new_val * alpha;
        self.update_count += 1;
    }

    /// Batch update: feed multiple depth/accepted pairs from one decode step.
    pub fn update_batch(&mut self, results: &[(usize, bool)]) {
        for &(depth, accepted) in results {
            self.update(depth, accepted);
        }
    }

    /// Allocate speculative depth budget across positions.
    /// Higher agreement → more budget.
    ///
    /// Returns Vec of length `max_depth` where each element is the
    /// number of tree nodes allocated to that depth. Sum ≤ `max_budget`.
    pub fn allocate(&self, max_budget: usize, max_depth: usize) -> Vec<usize> {
        if max_depth == 0 || max_budget == 0 {
            return vec![];
        }

        let rates: Vec<f32> = (0..max_depth)
            .map(|d| self.agreement_rate(d).max(0.01)) // floor to avoid zero
            .collect();
        let total: f32 = rates.iter().sum();

        let mut allocation: Vec<usize> = rates
            .iter()
            .map(|&r| ((r / total) * max_budget as f32) as usize)
            .collect();

        // Enforce minimum
        for a in &mut allocation {
            *a = (*a).max(self.min_budget_per_depth);
        }

        // Adjust to match max_budget
        let current: usize = allocation.iter().sum();
        match current.cmp(&max_budget) {
            std::cmp::Ordering::Less => {
                let mut remaining = max_budget - current;
                // Distribute to highest-agreement depths first
                let mut indexed: Vec<(usize, f32)> = rates.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (i, _) in indexed {
                    if remaining == 0 {
                        break;
                    }
                    allocation[i] += 1;
                    remaining -= 1;
                }
            }
            std::cmp::Ordering::Greater => {
                let mut excess = current - max_budget;
                // Trim from lowest-agreement depths first
                let mut indexed: Vec<(usize, f32)> = rates.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (i, _) in indexed {
                    if excess == 0 {
                        break;
                    }
                    let trim = excess.min(allocation[i].saturating_sub(self.min_budget_per_depth));
                    allocation[i] -= trim;
                    excess -= trim;
                }
            }
            std::cmp::Ordering::Equal => {}
        }

        allocation
    }

    /// Reset all agreement rates to default.
    pub fn reset(&mut self) {
        self.depth_agreement_rate.clear();
        self.update_count = 0;
    }

    /// Number of depths tracked.
    pub fn tracked_depths(&self) -> usize {
        self.depth_agreement_rate.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_allocator_empty() {
        let alloc = CorrelationBudgetAllocator::default();
        assert_eq!(alloc.agreement_rate(0), 0.5);
        assert_eq!(alloc.agreement_rate(5), 0.5);
    }

    #[test]
    fn test_update_converges_to_accepted() {
        let mut alloc = CorrelationBudgetAllocator::new(0.5);
        // Always accepted at depth 0
        for _ in 0..100 {
            alloc.update(0, true);
        }
        // EMA should converge close to 1.0
        assert!(
            alloc.agreement_rate(0) > 0.95,
            "rate should converge to ~1.0, got {}",
            alloc.agreement_rate(0)
        );
    }

    #[test]
    fn test_update_converges_to_rejected() {
        let mut alloc = CorrelationBudgetAllocator::new(0.5);
        for _ in 0..100 {
            alloc.update(0, false);
        }
        assert!(
            alloc.agreement_rate(0) < 0.05,
            "rate should converge to ~0.0, got {}",
            alloc.agreement_rate(0)
        );
    }

    #[test]
    fn test_allocate_sums_to_budget() {
        let mut alloc = CorrelationBudgetAllocator::default();
        // Train: depth 0 always accepted, depth 1 always rejected
        for _ in 0..200 {
            alloc.update(0, true);
            alloc.update(1, false);
        }
        let budget = 100;
        let allocation = alloc.allocate(budget, 2);
        let total: usize = allocation.iter().sum();
        assert_eq!(allocation.len(), 2);
        assert!(
            total <= budget + 2,
            "total {} should be close to budget {}",
            total,
            budget
        );
        // Depth 0 should get MORE budget than depth 1
        assert!(
            allocation[0] > allocation[1],
            "depth 0 ({}) should get more than depth 1 ({})",
            allocation[0],
            allocation[1]
        );
    }

    #[test]
    fn test_allocate_empty_depth() {
        let alloc = CorrelationBudgetAllocator::default();
        let allocation = alloc.allocate(100, 0);
        assert!(allocation.is_empty());
    }

    #[test]
    fn test_allocate_respects_minimum() {
        let mut alloc = CorrelationBudgetAllocator::default();
        for _ in 0..200 {
            alloc.update(0, true);
            alloc.update(1, false);
        }
        let allocation = alloc.allocate(100, 2);
        for &a in &allocation {
            assert!(
                a >= 1,
                "each depth should get at least min_budget_per_depth, got {}",
                a
            );
        }
    }

    #[test]
    fn test_warmup_uses_higher_alpha() {
        let mut alloc = CorrelationBudgetAllocator::default();
        // Warmup alpha = 0.3, standard = 0.1
        // First update: rate = 0.5 * 0.7 + 1.0 * 0.3 = 0.65
        alloc.update(0, true);
        let rate_after_1 = alloc.agreement_rate(0);
        assert!(
            (rate_after_1 - 0.65).abs() < 0.01,
            "warmup rate should be 0.65, got {}",
            rate_after_1
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut alloc = CorrelationBudgetAllocator::default();
        alloc.update(0, true);
        alloc.update(1, false);
        assert_eq!(alloc.tracked_depths(), 2);
        alloc.reset();
        assert_eq!(alloc.tracked_depths(), 0);
        assert_eq!(alloc.agreement_rate(0), 0.5); // back to default
    }

    #[test]
    fn test_budget_converges_after_n_steps() {
        let mut alloc = CorrelationBudgetAllocator::new(0.3);
        // Simulate 3 depths: 0=90% acceptance, 1=50%, 2=10%
        for _ in 0..500 {
            alloc.update(0, true);
            alloc.update(1, fastrand::bool()); // ~50%
            alloc.update(2, false);
        }
        let allocation = alloc.allocate(300, 3);
        assert!(
            allocation[0] > allocation[1],
            "depth 0 should get more budget"
        );
        assert!(
            allocation[1] > allocation[2],
            "depth 1 should get more budget than depth 2"
        );
    }

    #[test]
    fn test_batch_update() {
        let mut alloc = CorrelationBudgetAllocator::new(0.5);
        alloc.update_batch(&[(0, true), (1, false), (2, true)]);
        assert_eq!(alloc.tracked_depths(), 3);
        assert!(alloc.agreement_rate(0) > 0.5);
        assert!(alloc.agreement_rate(1) < 0.5);
        assert!(alloc.agreement_rate(2) > 0.5);
    }
}
