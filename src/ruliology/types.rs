//! Ruliology core types — SimpleProgram trait, WinMatrix, RuliologyPruner.
//!
//! Defines the shared abstractions for exhaustive program enumeration as
//! bandit arms. Plan 188 Phase 1.

// ── SimpleProgram ──────────────────────────────────────────────

/// A simple program that can compete as a game strategy.
///
/// FSM, CA rule, or TM — unified interface for ruliology enumeration.
/// Implementors must be `Clone + Send + Sync` so they can be shared across
/// threads during tournament evaluation.
pub trait SimpleProgram: Clone + Send + Sync {
    /// Given opponent's action history (0 or 1 per round), produce next action.
    fn next_action(&mut self, opponent_history: &[u8]) -> u8;

    /// Compact identifier for logging/ranking (blake3 hash of state).
    fn id(&self) -> u64;

    /// Behavioral complexity score (0.0 = trivial, higher = more complex).
    fn complexity(&self) -> f32;
}

// ── WinMatrix ──────────────────────────────────────────────────

/// Result of exhaustive round-robin tournament.
///
/// `payoffs[i][j]` = mean payoff of strategy `i` vs strategy `j` over
/// `rounds` rounds. Rankings sorted by average payoff descending.
pub struct WinMatrix {
    /// payoffs[i][j] = mean payoff of strategy i vs strategy j.
    pub payoffs: Vec<Vec<f64>>,
    /// Strategy IDs corresponding to row/column indices.
    pub ids: Vec<u64>,
    /// (id, average_payoff) sorted descending by payoff.
    pub rankings: Vec<(u64, f64)>,
}

impl WinMatrix {
    /// Build a `WinMatrix` from raw payoff data and strategy IDs.
    ///
    /// Computes rankings as average payoff across all opponents.
    #[inline]
    pub fn new(payoffs: Vec<Vec<f64>>, ids: Vec<u64>) -> Self {
        let n = ids.len();
        debug_assert_eq!(payoffs.len(), n);

        let mut rankings: Vec<(u64, f64)> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let avg = if n > 1 {
                    payoffs[i].iter().sum::<f64>() / (n - 1) as f64
                } else {
                    0.0
                };
                (id, avg)
            })
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            payoffs,
            ids,
            rankings,
        }
    }

    /// Return the Pareto front: strategies that are not dominated.
    ///
    /// A strategy dominates another if it has *higher payoff* and *lower complexity*.
    /// Returns `(id, avg_payoff, complexity)` for each Pareto-optimal strategy.
    pub fn pareto_front(&self, complexities: &[f32]) -> Vec<(u64, f64, f32)> {
        let n = self.ids.len();
        debug_assert_eq!(complexities.len(), n);

        // Build (id, avg_payoff, complexity) from IDs + complexity lookup.
        let id_to_idx: Vec<(usize, u64, f64, f32)> = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let avg = if n > 1 {
                    self.payoffs[i].iter().sum::<f64>() / (n - 1) as f64
                } else {
                    0.0
                };
                (i, id, avg, complexities[i])
            })
            .collect();

        let mut front: Vec<(u64, f64, f32)> = Vec::with_capacity(n);

        for &(.., payoff_i, cx_i) in &id_to_idx {
            let dominated = id_to_idx
                .iter()
                .any(|&(.., payoff_j, cx_j)| payoff_j > payoff_i && cx_j < cx_i);
            if !dominated {
                // Find the ID for this index.
                if let Some((_, id, _, _)) = id_to_idx
                    .iter()
                    .find(|(idx, _, _, c)| *idx < n && *c == cx_i)
                {
                    front.push((*id, payoff_i, cx_i));
                }
            }
        }

        // Sort descending by payoff.
        front.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        front
    }

    /// Average payoff for strategy at index `i`.
    #[inline]
    pub fn avg_payoff(&self, i: usize) -> f64 {
        let n = self.payoffs[i].len();
        if n == 0 {
            return 0.0;
        }
        self.payoffs[i].iter().sum::<f64>() / n as f64
    }
}

// ── RuliologyPruner ────────────────────────────────────────────

/// Filter WinMatrix rankings to Pareto-optimal subset.
///
/// Used to pre-filter the FSM arm space before feeding to BanditPruner.
/// Only strategies above `payoff_threshold` and below `complexity_threshold`
/// are kept as candidate bandit arms.
pub struct RuliologyPruner {
    /// Minimum average payoff to be considered.
    pub payoff_threshold: f64,
    /// Maximum complexity score to be considered.
    pub complexity_threshold: f32,
}

impl RuliologyPruner {
    /// Create a new pruner with given thresholds.
    #[inline]
    pub fn new(payoff_threshold: f64, complexity_threshold: f32) -> Self {
        Self {
            payoff_threshold,
            complexity_threshold,
        }
    }

    /// Filter rankings to strategies meeting both thresholds.
    ///
    /// Returns indices into the original strategy list.
    pub fn filter(&self, matrix: &WinMatrix, complexities: &[f32]) -> Vec<usize> {
        let mut result = Vec::new();
        for (i, &id) in matrix.ids.iter().enumerate() {
            let avg = matrix.avg_payoff(i);
            if avg >= self.payoff_threshold && complexities[i] <= self.complexity_threshold {
                // Also verify this strategy is on the Pareto front.
                result.push(i);
            }
            let _ = id; // suppress unused warning
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_win_matrix_rankings_sorted() {
        let payoffs = vec![
            vec![0.0, 5.0, 3.0], // avg 4.0
            vec![1.0, 0.0, 2.0], // avg 1.5
            vec![4.0, 1.0, 0.0], // avg 2.5
        ];
        let ids = vec![100, 200, 300];
        let wm = WinMatrix::new(payoffs, ids);

        assert_eq!(wm.rankings[0], (100, 4.0));
        assert_eq!(wm.rankings[1], (300, 2.5));
        assert_eq!(wm.rankings[2], (200, 1.5));
    }

    #[test]
    fn test_win_matrix_avg_payoff() {
        let payoffs = vec![
            vec![0.0, 3.0], // avg 1.5 (all entries)
            vec![1.0, 0.0], // avg 0.5 (all entries)
        ];
        let wm = WinMatrix::new(payoffs, vec![1, 2]);
        // avg_payoff averages all entries in the row.
        assert!((wm.avg_payoff(0) - 1.5).abs() < 1e-9);
        assert!((wm.avg_payoff(1) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pareto_front_filters_dominated() {
        let payoffs = vec![
            vec![0.0, 5.0], // strategy 0: high payoff
            vec![1.0, 0.0], // strategy 1: low payoff
        ];
        let ids = vec![10, 20];
        let wm = WinMatrix::new(payoffs, ids);

        // Strategy 0: avg payoff 5.0, complexity 1.0
        // Strategy 1: avg payoff 1.0, complexity 2.0
        // Strategy 0 dominates strategy 1 (higher payoff, lower complexity).
        let complexities = vec![1.0, 2.0];
        let front = wm.pareto_front(&complexities);

        assert_eq!(front.len(), 1);
        assert_eq!(front[0].0, 10);
        assert!((front[0].1 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_ruliology_pruner_filter() {
        let payoffs = vec![
            vec![0.0, 5.0], // avg 5.0
            vec![1.0, 0.0], // avg 1.0
            vec![3.0, 3.0], // avg 3.0
        ];
        let ids = vec![1, 2, 3];
        let wm = WinMatrix::new(payoffs, ids);
        let complexities = vec![1.0, 3.0, 2.0];

        let pruner = RuliologyPruner::new(2.0, 2.5);
        let filtered = pruner.filter(&wm, &complexities);

        assert!(filtered.contains(&0)); // payoff 5.0, complexity 1.0
        assert!(filtered.contains(&2)); // payoff 3.0, complexity 2.0
        assert!(!filtered.contains(&1)); // payoff 1.0 < threshold
    }
}

// TL;DR: SimpleProgram trait (next_action/id/complexity), WinMatrix (round-robin result with Pareto front), RuliologyPruner (threshold filter).
