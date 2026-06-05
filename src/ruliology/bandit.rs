//! Ruliology Bandit — FSM strategies as UCB1 bandit arms.
//!
//! Pipeline: enumerate FSMs → tournament → Pareto filter → UCB1 selection.
//! Plan 188 Phase 2.

use crate::ruliology::fsm::{FsmEnumerator, FsmStrategy};
use crate::ruliology::types::SimpleProgram;

// ── RuliologyArm ──────────────────────────────────────────────

/// Bandit arm backed by an enumerated FSM strategy.
pub struct RuliologyArm {
    /// The FSM strategy for this arm.
    pub strategy: FsmStrategy,
    /// Running payoff estimate (updated via incremental mean).
    payoff_estimate: f64,
    /// Number of times this arm has been pulled.
    pull_count: u32,
}

impl RuliologyArm {
    pub fn new(strategy: FsmStrategy) -> Self {
        Self {
            strategy,
            payoff_estimate: 0.0,
            pull_count: 0,
        }
    }

    /// Update payoff with incremental mean.
    pub fn update(&mut self, reward: f64) {
        self.pull_count += 1;
        let n = self.pull_count as f64;
        self.payoff_estimate += (reward - self.payoff_estimate) / n;
    }

    /// Get the current payoff estimate.
    #[inline]
    pub fn payoff(&self) -> f64 {
        self.payoff_estimate
    }

    /// Number of pulls.
    #[inline]
    pub fn pulls(&self) -> u32 {
        self.pull_count
    }

    /// UCB1 score for this arm.
    pub fn ucb1_score(&self, total_pulls: u32) -> f64 {
        if self.pull_count == 0 {
            return f64::INFINITY;
        }
        let exploration = (2.0 * (total_pulls as f64).ln() / self.pull_count as f64).sqrt();
        self.payoff_estimate + exploration
    }
}

// ── RuliologyBandit ───────────────────────────────────────────

/// Bandit that selects from Pareto-filtered FSM strategies.
///
/// Workflow:
/// 1. Enumerate all N-state FSMs via FsmEnumerator
/// 2. Run round-robin tournament to rank them
/// 3. Filter to Pareto-optimal arms via WinMatrix::pareto_front
/// 4. Use UCB1 to select best arm per round
/// 5. Update payoffs from game results
pub struct RuliologyBandit {
    /// Pre-filtered arms from Pareto front.
    arms: Vec<RuliologyArm>,
    /// Total pulls across all arms.
    total_pulls: u32,
    /// Minimum payoff threshold for filtering.
    payoff_threshold: f64,
    /// Maximum complexity for filtering.
    complexity_threshold: f32,
}

impl RuliologyBandit {
    /// Create from pre-enumerated FSM strategies.
    ///
    /// Runs tournament, extracts Pareto front, creates arms from optimal strategies.
    pub fn from_strategies(
        strategies: &[FsmStrategy],
        rounds: u32,
        payoff_fn: &dyn Fn(u8, u8) -> f64,
        payoff_threshold: f64,
        complexity_threshold: f32,
    ) -> Self {
        // Run tournament
        let win_matrix = FsmEnumerator::tournament(strategies, rounds, payoff_fn);

        // Compute complexities for Pareto front
        let complexities: Vec<f32> = strategies.iter().map(|s| s.complexity()).collect();

        // Extract Pareto-optimal strategies
        let pareto = win_matrix.pareto_front(&complexities);

        // Create arms from Pareto-optimal strategies
        let arms: Vec<RuliologyArm> = pareto
            .iter()
            .filter(|(_, payoff, _)| *payoff >= payoff_threshold)
            .filter_map(|(id, _, _)| strategies.iter().find(|s| s.id() == *id))
            .map(|s| RuliologyArm::new(s.clone()))
            .collect();

        Self {
            arms,
            total_pulls: 0,
            payoff_threshold,
            complexity_threshold,
        }
    }

    /// Select best arm using UCB1.
    pub fn select_arm(&self) -> usize {
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let score = arm.ucb1_score(self.total_pulls);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// Get the FSM strategy for an arm.
    pub fn strategy(&self, arm: usize) -> &FsmStrategy {
        &self.arms[arm].strategy
    }

    /// Update an arm's payoff after observing a reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.arms[arm].update(reward);
        self.total_pulls += 1;
    }

    /// Number of arms (Pareto-optimal strategies).
    #[inline]
    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    /// Best arm by payoff estimate.
    pub fn best_arm(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.payoff().partial_cmp(&b.payoff()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliology::payoff::matching_pennies;

    #[test]
    fn test_ruliology_arm_ucb1_unvisited_max() {
        let transitions = [[0u8; 2]; crate::ruliology::fsm::MAX_STATES];
        let outputs = [0u8; crate::ruliology::fsm::MAX_STATES];
        let strategy = FsmStrategy::new(transitions, outputs, 1, 0);
        let arm = RuliologyArm::new(strategy);
        // Unvisited arm should have infinite UCB1 score.
        assert!(arm.ucb1_score(100).is_infinite());
        assert!(arm.ucb1_score(0).is_infinite());
    }

    #[test]
    fn test_ruliology_arm_update_incremental_mean() {
        let transitions = [[0u8; 2]; crate::ruliology::fsm::MAX_STATES];
        let outputs = [0u8; crate::ruliology::fsm::MAX_STATES];
        let strategy = FsmStrategy::new(transitions, outputs, 1, 0);
        let mut arm = RuliologyArm::new(strategy);

        // Feed rewards 1.0, 0.0, 1.0, 0.0 → should converge to 0.5
        let rewards = [1.0, 0.0, 1.0, 0.0];
        for r in rewards {
            arm.update(r);
        }
        assert!(
            (arm.payoff() - 0.5).abs() < 1e-9,
            "expected 0.5, got {}",
            arm.payoff()
        );
        assert_eq!(arm.pulls(), 4);
    }

    #[test]
    fn test_ruliology_bandit_from_strategies_mp() {
        let strategies = FsmEnumerator::enumerate(2);
        // Expect ~22 distinct 2-state FSMs (Wolfram result, may vary with dedup).
        assert!(
            strategies.len() >= 18 && strategies.len() <= 30,
            "expected ~22 distinct 2-state FSMs, got {}",
            strategies.len()
        );

        let bandit = RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY, // accept all payoffs
            f32::MAX,          // accept all complexities
        );

        // Should have at least some Pareto-optimal arms.
        assert!(
            bandit.num_arms() >= 1,
            "expected >= 1 arm, got {}",
            bandit.num_arms()
        );
    }

    #[test]
    fn test_ruliology_bandit_selects_best_arm() {
        let strategies = FsmEnumerator::enumerate(2);

        let mut bandit = RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY,
            f32::MAX,
        );

        assert!(bandit.num_arms() > 0);

        // Pull arms many times, giving arm 0 a consistently higher reward.
        for _ in 0..200 {
            let arm = bandit.select_arm();
            // Give arm 0 a higher reward than others to make it converge.
            let reward = if arm == 0 { 1.0 } else { -1.0 };
            bandit.update(arm, reward);
        }

        // After many updates with biased rewards, arm 0 should be best.
        let best = bandit.best_arm();
        assert_eq!(best, 0, "arm 0 should be best after biased updates");
    }

    #[test]
    fn test_ruliology_bandit_num_arms_positive() {
        let strategies = FsmEnumerator::enumerate(2);

        let bandit = RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY,
            f32::MAX,
        );

        // Pareto front from 22 strategies should yield at least 1 arm.
        assert!(bandit.num_arms() >= 1);
    }
}

// TL;DR: RuliologyArm (FSM-backed UCB1 arm) + RuliologyBandit (Pareto-filtered strategy selection pipeline). 5 tests behind ruliology feature gate.
