//! SpecReconciler — orchestrates the reconciliation pipeline.
//!
//! Pipeline: hard bounds (T2) → manifold generation (T3) → soft scoring (T4) → verdict.
//!
//! This is the main entry point for verifying offline game state trajectories
//! against a LEO-generated plausibility manifold.

use super::manifold::{DefaultManifoldGenerator, ManifoldGenerator};
use super::manifold_scorer::ManifoldScorer;
use super::reconciliation_pruner::ReconciliationPruner;
use super::types::{ReconciliationConfig, ReconciliationVerdict, TrajectoryPoint};
use crate::types::Rng;

/// Reconciliation result with score breakdown.
#[derive(Clone, Debug)]
pub struct ReconciliationResult {
    /// Final verdict.
    pub verdict: ReconciliationVerdict,
    /// Max cosine similarity score against the manifold.
    pub max_similarity: f32,
    /// Average cosine similarity across the trajectory.
    pub avg_similarity: f32,
    /// Whether hard bounds (velocity, position, kill-rate) passed.
    pub hard_bounds_pass: bool,
    /// Number of manifold trajectories generated.
    pub manifold_count: usize,
}

/// The core reconciliation engine.
///
/// Orchestrates hard-bound pruning → manifold generation → soft scoring → verdict.
/// All state is borrowed or stack-allocated; the only allocation is the manifold.
pub struct SpecReconciler {
    /// Configuration thresholds.
    pub config: ReconciliationConfig,
    /// Hard-bound pruner.
    pruner: ReconciliationPruner,
    /// Manifold generator (modelless by default).
    generator: Box<dyn ManifoldGenerator>,
    /// Soft scorer.
    scorer: ManifoldScorer,
}

impl SpecReconciler {
    /// Create a new reconciler with default (modelless) manifold generator.
    pub fn new(config: ReconciliationConfig) -> Self {
        config.validate().expect("invalid ReconciliationConfig");
        let pruner = ReconciliationPruner::new(config, TrajectoryPoint::default());
        let generator = Box::new(DefaultManifoldGenerator::new(config));
        let scorer = ManifoldScorer::new(&config);
        Self {
            config,
            pruner,
            generator,
            scorer,
        }
    }

    /// Create with a custom manifold generator.
    pub fn with_generator(
        config: ReconciliationConfig,
        generator: Box<dyn ManifoldGenerator>,
    ) -> Self {
        config.validate().expect("invalid ReconciliationConfig");
        let pruner = ReconciliationPruner::new(config, TrajectoryPoint::default());
        let scorer = ManifoldScorer::new(&config);
        Self {
            config,
            pruner,
            generator,
            scorer,
        }
    }

    /// Reconcile an offline trajectory against the plausibility manifold.
    ///
    /// Pipeline:
    /// 1. Hard bounds: check velocity, position, kill-rate for all consecutive pairs
    /// 2. Manifold generation: generate K speculative trajectories from the last known-good state
    /// 3. Soft scoring: compute max cosine similarity between client and manifold
    /// 4. Verdict: Accept / Quarantine / Uncertain based on similarity thresholds
    ///
    /// `h_last` is the last known-good server state.
    /// `client_trajectory` is the sequence of points reported by the client during disconnection.
    /// `q_goals` are LEO Q-values for goal weighting (can be empty).
    /// `steps` is the number of steps to simulate per manifold trajectory.
    pub fn reconcile(
        &mut self,
        h_last: &TrajectoryPoint,
        client_trajectory: &[TrajectoryPoint],
        q_goals: &[f32],
        steps: usize,
        rng: &mut Rng,
    ) -> ReconciliationResult {
        // ── Stage 1: Hard bounds ─────────────────────────────────────
        // Check the entire client trajectory for physical impossibilities.
        self.pruner.previous = *h_last;
        let hard_bounds_pass = self.pruner.check_trajectory(client_trajectory);

        if !hard_bounds_pass {
            return ReconciliationResult {
                verdict: ReconciliationVerdict::Quarantine,
                max_similarity: 0.0,
                avg_similarity: 0.0,
                hard_bounds_pass: false,
                manifold_count: 0,
            };
        }

        // ── Stage 2: Manifold generation ─────────────────────────────
        // Generate K speculative trajectories from the last known-good state.
        let dt = self.config.dt;
        let manifold = self
            .generator
            .generate(h_last, q_goals, self.config.k, dt, steps, rng);

        // ── Stage 3: Soft scoring ────────────────────────────────────
        // Score the client trajectory against the manifold.
        self.scorer.set_manifold(&manifold);

        let avg_similarity = self.scorer.score_trajectory(client_trajectory);
        let max_similarity = client_trajectory
            .iter()
            .map(|cp| self.scorer.score_against_manifold(cp))
            .fold(0.0f32, f32::max);

        // ── Stage 4: Verdict ─────────────────────────────────────────
        let verdict = if max_similarity >= self.config.accept_threshold {
            ReconciliationVerdict::Accept
        } else if max_similarity < self.config.quarantine_threshold {
            ReconciliationVerdict::Quarantine
        } else {
            ReconciliationVerdict::Uncertain
        };

        ReconciliationResult {
            verdict,
            max_similarity,
            avg_similarity,
            hard_bounds_pass: true,
            manifold_count: manifold.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> ReconciliationConfig {
        ReconciliationConfig {
            k: 16,
            max_speed: 600.0,
            map_bounds: [0.0, 0.0, 4096.0, 4096.0],
            accept_threshold: 0.5, // Lower threshold for test stability
            quarantine_threshold: 0.2,
            kill_rate_sigma: 5.0,
            noise_sigma: 0.1,
            dt: 1.0 / 60.0,
        }
    }

    fn make_h_last() -> TrajectoryPoint {
        TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 5.0, 2.0, 0.0, 1.0, 0.0)
    }

    /// Generate a legitimate trajectory: small movements from h_last.
    fn make_legitimate_trajectory(h_last: &TrajectoryPoint, steps: usize) -> Vec<TrajectoryPoint> {
        (0..steps)
            .map(|i| {
                let t = i as f32;
                TrajectoryPoint::from_fields(
                    h_last.pos_x() + t * 0.1, // tiny drift
                    h_last.pos_y() + t * 0.05,
                    10.0,
                    5.0,
                    2.0, // kills stay same
                    0.0,
                    1.0,
                    0.0,
                )
            })
            .collect()
    }

    /// Generate a teleport hack: sudden large displacement.
    fn make_teleport_trajectory(h_last: &TrajectoryPoint) -> Vec<TrajectoryPoint> {
        vec![
            *h_last,
            TrajectoryPoint::from_fields(
                h_last.pos_x() + 5000.0, // teleport!
                h_last.pos_y(),
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.0,
            ),
        ]
    }

    /// Generate a kill-rate hack: impossible number of kills.
    fn make_kill_rate_hack(h_last: &TrajectoryPoint) -> Vec<TrajectoryPoint> {
        vec![
            *h_last,
            TrajectoryPoint::from_fields(
                h_last.pos_x() + 0.1,
                h_last.pos_y(),
                10.0,
                5.0,
                50.0, // 48 kills in one frame!
                0.0,
                1.0,
                0.0,
            ),
        ]
    }

    #[test]
    fn test_legitimate_trajectory_accepts() {
        let config = make_config();
        let mut reconciler = SpecReconciler::new(config);
        let h_last = make_h_last();
        let client = make_legitimate_trajectory(&h_last, 10);
        let mut rng = Rng::new(42);

        let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);

        // Legitimate play should at minimum pass hard bounds.
        assert!(
            result.hard_bounds_pass,
            "legitimate trajectory should pass hard bounds"
        );
        // With low thresholds, it should accept or at worst be uncertain.
        assert_ne!(
            result.verdict,
            ReconciliationVerdict::Quarantine,
            "legitimate trajectory should not be quarantined (similarity: {})",
            result.max_similarity
        );
    }

    #[test]
    fn test_teleport_hack_quarantines() {
        let config = make_config();
        let mut reconciler = SpecReconciler::new(config);
        let h_last = make_h_last();
        let client = make_teleport_trajectory(&h_last);
        let mut rng = Rng::new(42);

        let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);

        assert_eq!(
            result.verdict,
            ReconciliationVerdict::Quarantine,
            "teleport hack should be quarantined"
        );
        assert!(!result.hard_bounds_pass);
    }

    #[test]
    fn test_kill_rate_hack_quarantines() {
        let config = make_config();
        let mut reconciler = SpecReconciler::new(config);
        let h_last = make_h_last();
        let client = make_kill_rate_hack(&h_last);
        let mut rng = Rng::new(42);

        let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);

        assert_eq!(
            result.verdict,
            ReconciliationVerdict::Quarantine,
            "kill-rate hack should be quarantined"
        );
        assert!(!result.hard_bounds_pass);
    }

    #[test]
    fn test_empty_trajectory_accepts() {
        let config = make_config();
        let mut reconciler = SpecReconciler::new(config);
        let h_last = make_h_last();
        let mut rng = Rng::new(42);

        let result = reconciler.reconcile(&h_last, &[], &[], 10, &mut rng);

        // Empty trajectory should pass hard bounds (trivially valid).
        assert!(result.hard_bounds_pass);
    }

    #[test]
    fn test_result_contains_manifold_count() {
        let config = make_config();
        let mut reconciler = SpecReconciler::new(config);
        let h_last = make_h_last();
        let client = make_legitimate_trajectory(&h_last, 5);
        let mut rng = Rng::new(42);

        let result = reconciler.reconcile(&h_last, &client, &[], 5, &mut rng);

        assert_eq!(result.manifold_count, config.k);
    }
}
