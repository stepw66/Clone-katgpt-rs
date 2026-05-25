//! SpecHop continuous pipeline loop (Algorithm 1 from paper).
//!
//! Orchestrates speculative prediction, window management, verification,
//! and rollback for multi-hop tool-use trajectories.
//!
//! **Algorithm 1** (simplified):
//! 1. Extend window — speculator predicts observation for next action
//! 2. Verify earliest pending thread against returned target
//! 3. Commit if match, rollback if mismatch
//! 4. Repeat until trajectory complete or early termination

use crate::spechop::speculator::HopSpeculator;
use crate::spechop::types::{HopObservation, SpecHopConfig, SpecOutcome};
use crate::spechop::verifier::ObservationVerifier;
use crate::spechop::window::SpecWindow;

// ── Trajectory Types ──────────────────────────────────────────

/// A single hop in a tool-use trajectory.
///
/// Each hop represents one (action → observation) pair. The pipeline
/// speculates on observations ahead of time and verifies when the
/// target tool returns.
#[derive(Clone, Debug)]
pub struct TrajectoryHop {
    /// The tool-call action (e.g., `"search:rust language"`).
    pub action: String,
    /// The actual observation returned by the target tool.
    pub o_target: String,
    /// Whether this hop produces the final answer (triggers early termination check).
    pub is_final: bool,
}

impl TrajectoryHop {
    /// Create a non-final hop.
    pub fn new(action: impl Into<String>, o_target: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            o_target: o_target.into(),
            is_final: false,
        }
    }

    /// Create a final hop (triggers early termination check).
    pub fn final_hop(action: impl Into<String>, o_target: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            o_target: o_target.into(),
            is_final: true,
        }
    }
}

// ── Pipeline Result ───────────────────────────────────────────

/// Result of executing the speculative pipeline on a trajectory.
#[derive(Clone, Debug)]
#[derive(Default)]
pub struct PipelineResult {
    /// All committed observations in trajectory order.
    pub committed: Vec<HopObservation>,
    /// Number of speculative predictions that matched the target.
    pub speculation_hits: usize,
    /// Number of speculative predictions that didn't match (rolled back).
    pub speculation_misses: usize,
    /// Number of hops committed directly (speculator had no prediction).
    pub direct_commits: usize,
    /// Total hops in the trajectory.
    pub total_hops: usize,
    /// Whether the pipeline terminated early (final answer committed).
    pub early_terminated: bool,
}


impl PipelineResult {
    /// Effective speculator accuracy: `hits / (hits + misses)`.
    ///
    /// Returns 0.0 when no speculations were attempted.
    pub fn accuracy(&self) -> f64 {
        let total = self.speculation_hits + self.speculation_misses;
        if total == 0 {
            return 0.0;
        }
        self.speculation_hits as f64 / total as f64
    }

    /// Speculation coverage: `(hits + misses) / total_hops`.
    ///
    /// Fraction of hops where the speculator had a prediction available.
    pub fn coverage(&self) -> f64 {
        if self.total_hops == 0 {
            return 0.0;
        }
        (self.speculation_hits + self.speculation_misses) as f64 / self.total_hops as f64
    }

    /// Total successfully committed hops (hits + misses + direct).
    ///
    /// Misses are still committed — the real observation replaces the
    /// speculative prediction after rollback.
    pub fn total_committed(&self) -> usize {
        self.speculation_hits + self.speculation_misses + self.direct_commits
    }
}

// ── Pipeline ──────────────────────────────────────────────────

/// Continuous multi-hop speculation pipeline (T20).
///
/// Generic over speculator `S` and verifier `V` for testability.
/// The pipeline runs Algorithm 1 from the SpecHop paper:
///
/// 1. **Extend window**: speculator predicts observation for next action
/// 2. **Verify earliest**: compare prediction against returned target
/// 3. **Commit/Rollback**: accept match, reject mismatch
/// 4. **Repeat** until trajectory complete or early termination
pub struct SpecHopPipeline<S: HopSpeculator, V: ObservationVerifier> {
    /// Pipeline configuration (α, β, p, k).
    config: SpecHopConfig,
    /// Speculator for predicting tool observations.
    speculator: S,
    /// Verifier for comparing predictions against targets.
    verifier: V,
    /// Window managing speculative threads.
    window: SpecWindow,
    /// Optional early-stop pattern (T23).
    /// If a committed observation contains this substring, terminate immediately.
    early_stop_pattern: Option<String>,
}

impl<S: HopSpeculator, V: ObservationVerifier> SpecHopPipeline<S, V> {
    /// Create a new pipeline with the given config, speculator, and verifier.
    ///
    /// The window capacity is set to `config.effective_k()`.
    pub fn new(config: SpecHopConfig, speculator: S, verifier: V) -> Self {
        let k = config.effective_k();
        Self {
            config,
            speculator,
            verifier,
            window: SpecWindow::new(k),
            early_stop_pattern: None,
        }
    }

    /// Set an early-stop pattern for T23 early termination.
    ///
    /// If a committed observation for a final hop contains this substring,
    /// the pipeline terminates immediately without processing remaining hops.
    pub fn with_early_stop(mut self, pattern: impl Into<String>) -> Self {
        self.early_stop_pattern = Some(pattern.into());
        self
    }

    /// Reference to the pipeline configuration.
    pub fn config(&self) -> &SpecHopConfig {
        &self.config
    }

    /// Reference to the internal window.
    pub fn window(&self) -> &SpecWindow {
        &self.window
    }

    /// Reset the pipeline state for a new trajectory.
    pub fn reset(&mut self) {
        self.window.reset();
    }

    /// Execute the pipeline on a trajectory of hops (T21).
    ///
    /// Main loop (Algorithm 1):
    /// 1. **Extend window**: speculator predicts observation for next action
    /// 2. **Verify earliest**: when target returns, check if prediction matches
    /// 3. **Commit or rollback** based on verification result
    /// 4. **Early termination**: if final answer committed, stop immediately (T23)
    ///
    /// Hop-level state machine (T22):
    /// - `Speculating` → `Committed` (verify passes)
    /// - `Speculating` → `RolledBack` (verify fails) → rollback all
    /// - No speculation → `Committed` directly (direct commit path)
    pub fn execute(&mut self, trajectory: &[TrajectoryHop]) -> PipelineResult {
        let mut result = PipelineResult {
            committed: Vec::with_capacity(trajectory.len()),
            total_hops: trajectory.len(),
            ..Default::default()
        };

        for hop in trajectory {
            let speculated = self.extend_window(&hop.action);

            if speculated {
                // Verify the earliest speculative thread
                self.verify_earliest(&hop.action, &hop.o_target, &mut result);
            } else {
                // No prediction available → direct commit
                self.direct_commit(&hop.action, &hop.o_target, &mut result);
            }

            // Drain committed threads from window
            result.committed.extend(self.window.drain_committed());

            // T23: Early termination on final hop
            if hop.is_final && self.check_early_stop(&hop.o_target) {
                result.early_terminated = true;
                return result;
            }
        }

        // Drain any remaining committed threads
        result.committed.extend(self.window.drain_committed());
        result
    }

    /// Extend the window by speculating on the given action.
    ///
    /// Returns `true` if speculation was available and pushed to window,
    /// `false` if speculator had no prediction (cache miss).
    fn extend_window(&mut self, action: &str) -> bool {
        match self.speculator.speculate(action) {
            Ok(o_spec) => {
                let obs = HopObservation::speculating(action, &o_spec);
                if !self.window.is_full() {
                    self.window.push_thread(obs);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Verify the earliest pending speculative thread (T22 state machine).
    ///
    /// Handles:
    /// - **Commit** (`Speculating` → `Committed`): prediction matched target
    /// - **Rollback** (`Speculating` → `RolledBack`): prediction mismatched,
    ///   all downstream speculation discarded
    fn verify_earliest(&mut self, action: &str, o_target: &str, result: &mut PipelineResult) {
        match self.window.verify_earliest(&self.verifier, o_target) {
            Some(SpecOutcome::Commit) => {
                result.speculation_hits += 1;
                // Feed correct observation back to speculator
                self.speculator.observe(action, o_target);
            }
            Some(SpecOutcome::Rollback) => {
                result.speculation_misses += 1;
                // Discard all downstream speculation
                self.window.rollback_all();
                // Commit the correct observation (speculation missed, but hop still produces output)
                let mut obs = HopObservation::awaiting(action);
                obs.commit(o_target);
                result.committed.push(obs);
                // Feed back for future predictions
                self.speculator.observe(action, o_target);
            }
            None => {
                // No pending speculative threads — shouldn't happen after extend_window
            }
        }
    }

    /// Direct commit for hops where speculator had no prediction.
    ///
    /// State machine: creates observation in `Committed` state directly
    /// (no speculation → no verification needed).
    fn direct_commit(&mut self, action: &str, o_target: &str, result: &mut PipelineResult) {
        let mut obs = HopObservation::awaiting(action);
        obs.commit(o_target);
        result.direct_commits += 1;
        result.committed.push(obs);
        // Feed back to speculator for future predictions
        self.speculator.observe(action, o_target);
    }

    /// Check if the observation matches the early-stop pattern (T23).
    fn check_early_stop(&self, o_target: &str) -> bool {
        match &self.early_stop_pattern {
            Some(pattern) => o_target.contains(pattern.as_str()),
            None => false, // No pattern → never early-stop (process all hops naturally)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spechop::speculator::CacheSpeculator;
    use crate::spechop::verifier::RuleBasedVerifier;

    /// Helper: create a default pipeline with cache speculator.
    fn make_pipeline(
        config: SpecHopConfig,
        cache_entries: Vec<(&str, &str)>,
    ) -> SpecHopPipeline<CacheSpeculator, RuleBasedVerifier> {
        let speculator = CacheSpeculator::with_entries(cache_entries);
        let verifier = RuleBasedVerifier::default();
        SpecHopPipeline::new(config, speculator, verifier)
    }

    /// Helper: default config with k=4.
    fn default_config() -> SpecHopConfig {
        SpecHopConfig {
            alpha: 0.2,
            beta: 0.15,
            p: 0.7,
            k: Some(4),
            volatility: 0.4,
        }
    }

    // ── T20: SpecHopPipeline struct ────────────────────────────

    #[test]
    fn test_pipeline_creation() {
        let pipeline = make_pipeline(default_config(), vec![]);
        assert_eq!(pipeline.config().effective_k(), 4);
        assert_eq!(pipeline.window().capacity(), 4);
    }

    #[test]
    fn test_pipeline_with_early_stop() {
        let pipeline = make_pipeline(default_config(), vec![]);
        let _pipeline = pipeline.with_early_stop("DONE");
        // Pipeline was consumed, early_stop_pattern set
    }

    // ── T21: execute() main loop ───────────────────────────────

    #[test]
    fn test_execute_empty_trajectory() {
        let mut pipeline = make_pipeline(default_config(), vec![]);
        let result = pipeline.execute(&[]);
        assert_eq!(result.total_hops, 0);
        assert_eq!(result.total_committed(), 0);
        assert!(!result.early_terminated);
    }

    #[test]
    fn test_execute_all_cache_hits() {
        let mut pipeline = make_pipeline(
            default_config(),
            vec![
                ("search_a", "result alpha"),
                ("search_b", "result beta"),
                ("search_c", "result gamma"),
            ],
        );

        let trajectory = vec![
            TrajectoryHop::new("search_a", "result alpha"),
            TrajectoryHop::new("search_b", "result beta"),
            TrajectoryHop::new("search_c", "result gamma"),
        ];

        let result = pipeline.execute(&trajectory);

        assert_eq!(result.total_hops, 3);
        assert_eq!(result.speculation_hits, 3);
        assert_eq!(result.speculation_misses, 0);
        assert_eq!(result.direct_commits, 0);
        assert_eq!(result.total_committed(), 3);
        assert!((result.accuracy() - 1.0).abs() < f64::EPSILON);
        assert!((result.coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_execute_all_cache_misses() {
        let mut pipeline = make_pipeline(default_config(), vec![]);

        let trajectory = vec![
            TrajectoryHop::new("search_a", "result alpha"),
            TrajectoryHop::new("search_b", "result beta"),
        ];

        let result = pipeline.execute(&trajectory);

        assert_eq!(result.total_hops, 2);
        assert_eq!(result.speculation_hits, 0);
        assert_eq!(result.speculation_misses, 0);
        assert_eq!(result.direct_commits, 2);
        assert_eq!(result.total_committed(), 2);
        assert!((result.accuracy()).abs() < f64::EPSILON); // no speculations
        assert!((result.coverage()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_execute_mixed_hits_and_misses() {
        let mut pipeline = make_pipeline(
            default_config(),
            vec![
                ("search_a", "result alpha"), // hit
                // search_b: miss (no cache entry)
                ("search_c", "result gamma"), // hit
            ],
        );

        let trajectory = vec![
            TrajectoryHop::new("search_a", "result alpha"),
            TrajectoryHop::new("search_b", "result beta"),
            TrajectoryHop::new("search_c", "result gamma"),
        ];

        let result = pipeline.execute(&trajectory);

        assert_eq!(result.total_hops, 3);
        assert_eq!(result.speculation_hits, 2);
        assert_eq!(result.direct_commits, 1);
        assert_eq!(result.total_committed(), 3);
    }

    // ── T22: Hop-level state machine ───────────────────────────

    #[test]
    fn test_state_machine_speculating_to_committed() {
        let mut pipeline = make_pipeline(default_config(), vec![("action", "expected result")]);

        let trajectory = vec![TrajectoryHop::new("action", "expected result")];
        let result = pipeline.execute(&trajectory);

        assert_eq!(result.speculation_hits, 1);
        // The committed observation should be in Committed state
        assert_eq!(result.committed.len(), 1);
        assert_eq!(
            result.committed[0].state,
            crate::spechop::types::HopState::Committed
        );
    }

    #[test]
    fn test_state_machine_speculating_to_rolledback() {
        let mut pipeline = make_pipeline(default_config(), vec![("action", "wrong prediction")]);

        let trajectory = vec![TrajectoryHop::new("action", "completely different target")];
        let result = pipeline.execute(&trajectory);

        assert_eq!(result.speculation_misses, 1);
        // After rollback, the committed list should be empty (rolled back)
        // But the speculator should have learned the real answer
    }

    #[test]
    fn test_state_machine_awaiting_to_committed() {
        let mut pipeline = make_pipeline(default_config(), vec![]);

        let trajectory = vec![TrajectoryHop::new("unknown_action", "real result")];
        let result = pipeline.execute(&trajectory);

        assert_eq!(result.direct_commits, 1);
        assert_eq!(result.committed.len(), 1);
        assert_eq!(
            result.committed[0].state,
            crate::spechop::types::HopState::Committed
        );
        assert_eq!(result.committed[0].o_target.as_deref(), Some("real result"));
    }

    // ── T23: Early termination ─────────────────────────────────

    #[test]
    fn test_early_termination_on_final_hop() {
        let pipeline = make_pipeline(
            default_config(),
            vec![
                ("hop1", "result1"),
                ("hop2", "result2"),
                ("final", "DONE: answer is 42"),
            ],
        );
        let mut pipeline = pipeline.with_early_stop("DONE:");

        let trajectory = vec![
            TrajectoryHop::new("hop1", "result1"),
            TrajectoryHop::new("hop2", "result2"),
            TrajectoryHop::final_hop("final", "DONE: answer is 42"),
            TrajectoryHop::new("hop4", "should not reach"),
        ];

        let result = pipeline.execute(&trajectory);

        assert!(result.early_terminated);
        assert_eq!(result.total_hops, 4); // total in trajectory
        assert!(result.committed.len() < 4); // but not all processed
        assert!(result.committed.len() >= 3); // at least 3 committed
    }

    #[test]
    fn test_no_early_termination_without_final_flag() {
        let mut pipeline = make_pipeline(
            default_config(),
            vec![("hop1", "result1"), ("hop2", "result2")],
        );

        let trajectory = vec![
            TrajectoryHop::new("hop1", "result1"),
            TrajectoryHop::new("hop2", "result2"), // not marked as final
        ];

        let result = pipeline.execute(&trajectory);

        assert!(!result.early_terminated);
        assert_eq!(result.committed.len(), 2);
    }

    #[test]
    fn test_early_stop_pattern_mismatch_continues() {
        let pipeline = make_pipeline(default_config(), vec![]);
        let mut pipeline = pipeline.with_early_stop("DONE:");

        let trajectory = vec![
            TrajectoryHop::final_hop("final", "no match here"),
            TrajectoryHop::new("extra", "extra result"),
        ];

        let result = pipeline.execute(&trajectory);

        // Pattern didn't match, but is_final triggered without pattern check
        // Actually, check_early_stop returns true when pattern is Some but doesn't match
        // Wait, let me re-check the logic...
        // check_early_stop returns `o_target.contains(pattern)` — if pattern doesn't match, returns false
        // So early termination doesn't trigger
        assert!(!result.early_terminated);
    }

    // ── T24: Integration test — 4-hop trajectory ──────────────

    #[test]
    fn test_t24_four_hop_trajectory_matches_sequential() {
        // Pre-populate cache: hops 1 and 3 have predictions, hops 2 and 4 don't
        let mut pipeline = make_pipeline(
            default_config(),
            vec![
                ("search:rust", "Rust is a systems programming language"),
                ("search:go", "Go is a compiled language"), // wrong prediction!
                                                            // "search:python" and "compute:sum" not in cache
            ],
        );

        let trajectory = vec![
            TrajectoryHop::new("search:rust", "Rust is a systems programming language"),
            TrajectoryHop::new("search:python", "Python is an interpreted language"),
            TrajectoryHop::new(
                "search:go",
                "Go is a statically typed compiled language designed at Google", // different from cache
            ),
            TrajectoryHop::final_hop("compute:sum", "The sum is 42"),
        ];

        let result = pipeline.execute(&trajectory);

        // Verify counts
        assert_eq!(result.total_hops, 4);
        // Hop 1: cache hit, prediction matches → speculation_hit
        // Hop 2: cache miss → direct_commit
        // Hop 3: cache hit, prediction DIFFERS → speculation_miss (rollback)
        // Hop 4: cache miss → direct_commit
        assert_eq!(result.speculation_hits, 1);
        assert_eq!(result.speculation_misses, 1);
        assert_eq!(result.direct_commits, 2);
        assert_eq!(result.total_committed(), 4); // all hops eventually committed

        // Verify all observations are correct (matches sequential execution)
        assert_eq!(result.committed.len(), 4);

        // Hop 1: committed with correct target
        assert_eq!(result.committed[0].action, "search:rust");
        assert_eq!(
            result.committed[0].o_target.as_deref(),
            Some("Rust is a systems programming language")
        );

        // Hop 2: direct commit
        assert_eq!(result.committed[1].action, "search:python");
        assert_eq!(
            result.committed[1].o_target.as_deref(),
            Some("Python is an interpreted language")
        );

        // Hop 3: after rollback, direct commit with real target
        // (rollback clears speculative state, real observation is committed)
        assert_eq!(result.committed[2].action, "search:go");
        assert_eq!(
            result.committed[2].o_target.as_deref(),
            Some("Go is a statically typed compiled language designed at Google")
        );

        // Hop 4: final hop, direct commit
        assert_eq!(result.committed[3].action, "compute:sum");
        assert_eq!(
            result.committed[3].o_target.as_deref(),
            Some("The sum is 42")
        );

        // Pipeline should have learned from observations
        assert!(!result.early_terminated);
    }

    #[test]
    fn test_t24_perfect_speculator_all_correct() {
        // All 4 hops have correct predictions in cache
        let mut pipeline = make_pipeline(
            default_config(),
            vec![
                ("hop1", "answer1"),
                ("hop2", "answer2"),
                ("hop3", "answer3"),
                ("hop4", "answer4"),
            ],
        );

        let trajectory = vec![
            TrajectoryHop::new("hop1", "answer1"),
            TrajectoryHop::new("hop2", "answer2"),
            TrajectoryHop::new("hop3", "answer3"),
            TrajectoryHop::new("hop4", "answer4"),
        ];

        let result = pipeline.execute(&trajectory);

        // Perfect speculator: all hits, no misses, no direct commits
        assert_eq!(result.speculation_hits, 4);
        assert_eq!(result.speculation_misses, 0);
        assert_eq!(result.direct_commits, 0);
        assert!((result.accuracy() - 1.0).abs() < f64::EPSILON);
        assert!((result.coverage() - 1.0).abs() < f64::EPSILON);

        // All committed observations match sequential execution
        assert_eq!(result.committed.len(), 4);
        for (i, obs) in result.committed.iter().enumerate() {
            let expected = format!("answer{}", i + 1);
            assert_eq!(obs.o_target.as_deref(), Some(expected.as_str()));
        }
    }

    #[test]
    fn test_pipeline_resets_between_trajectories() {
        let mut pipeline = make_pipeline(default_config(), vec![("a", "r1")]);

        let trajectory1 = vec![TrajectoryHop::new("a", "r1")];
        let result1 = pipeline.execute(&trajectory1);
        assert_eq!(result1.speculation_hits, 1);

        // Reset and run again — speculator should still have the cache entry
        pipeline.reset();
        let result2 = pipeline.execute(&trajectory1);
        assert_eq!(result2.speculation_hits, 1);
    }

    // ── PipelineResult metrics ─────────────────────────────────

    #[test]
    fn test_pipeline_result_metrics() {
        let result = PipelineResult {
            speculation_hits: 3,
            speculation_misses: 1,
            direct_commits: 2,
            total_hops: 6,
            ..Default::default()
        };

        assert!((result.accuracy() - 0.75).abs() < f64::EPSILON);
        assert!((result.coverage() - (4.0 / 6.0)).abs() < 1e-10);
        assert_eq!(result.total_committed(), 6); // 3 hits + 1 miss + 2 direct
    }

    #[test]
    fn test_pipeline_result_default() {
        let result = PipelineResult::default();
        assert!(result.committed.is_empty());
        assert_eq!(result.speculation_hits, 0);
        assert_eq!(result.speculation_misses, 0);
        assert_eq!(result.direct_commits, 0);
        assert_eq!(result.total_hops, 0);
        assert!(!result.early_terminated);
    }

    // ── TrajectoryHop helpers ──────────────────────────────────

    #[test]
    fn test_trajectory_hop_new() {
        let hop = TrajectoryHop::new("action", "result");
        assert_eq!(hop.action, "action");
        assert_eq!(hop.o_target, "result");
        assert!(!hop.is_final);
    }

    #[test]
    fn test_trajectory_hop_final() {
        let hop = TrajectoryHop::final_hop("action", "result");
        assert_eq!(hop.action, "action");
        assert_eq!(hop.o_target, "result");
        assert!(hop.is_final);
    }
}
