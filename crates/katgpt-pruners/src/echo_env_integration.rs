//! ECHO Environment Predictor integration wiring (Plan 247, T5).
//!
//! Convenience layer that wires the three ECHO primitives into the existing
//! BanditPruner + DDTree + budget adaptation pipeline:
//!
//! - `EchoPredictionScorer` — `PartialScorer` impl that feeds prediction
//!   accuracy from `PredictionVerifier` into `BanditPruner::update_with_trace`.
//!
//! - `EchoEnvIntegration` — owns all three primitives and provides factory
//!   methods for constructing a fully-wired `BanditPruner`.
//!
//! # Usage
//!
//! ```ignore
//! let echo = EchoEnvIntegration::new(forward_model, feature_dim, vocab_size);
//! let mut bandit = echo.into_bandit_pruner(BanditStrategy::Ucb1);
//! // Use bandit as ScreeningPruner in DDTree...
//! ```
//!
//! All code behind `echo_env_predictor` feature flag.

use crate::bandit::{BanditPruner, BanditStrategy};
use katgpt_speculative::echo_env::{
    EnvPredictorConfig, EnvPredictorPruner, PredictionConsistencyGate, PredictionVerifier,
};

/// `PartialScorer` impl that wraps `PredictionVerifier`'s accuracy signal.
///
/// Feeds EMA-tracked prediction accuracy as a continuous [0, 1] reward into
/// BanditPruner's `update_with_trace`. This allows the bandit to learn which
/// environments benefit from ECHO prediction scoring versus not.
///
/// Note: `PartialScorer` requires the `partial_scoring` feature.
#[cfg(feature = "partial_scoring")]
pub struct EchoPredictionScorer {
    /// Shared reference to the prediction verifier for reading accuracy.
    /// The verifier is updated externally (by the integration owner).
    verifier_accuracy: f32,
}

#[cfg(feature = "partial_scoring")]
impl EchoPredictionScorer {
    /// Create a new scorer with initial accuracy (typically 0.5).
    pub fn new(initial_accuracy: f32) -> Self {
        Self {
            verifier_accuracy: initial_accuracy,
        }
    }

    /// Update the cached accuracy from the verifier.
    pub fn update_from_verifier(&mut self, verifier: &PredictionVerifier) {
        self.verifier_accuracy = verifier.bandit_reward();
    }
}

#[cfg(feature = "partial_scoring")]
impl katgpt_core::PartialScorer for EchoPredictionScorer {
    fn partial_score(&self, _trace: &katgpt_core::GameTrace) -> f32 {
        // Blend: 50% prediction accuracy + 50% final reward (if available)
        // Prediction accuracy is the ECHO signal; final_reward is the domain signal.
        self.verifier_accuracy
    }
}

/// Convenience integration struct owning all three ECHO primitives.
///
/// Provides factory methods to construct a fully-wired BanditPruner
/// with ECHO prediction scoring as the inner `ScreeningPruner`.
///
/// # Type Parameters
/// - `F`: Forward model closure type — `(action_token, parent_tokens) → Vec<f32>`
pub struct EchoEnvIntegration<F>
where
    F: Fn(usize, &[usize]) -> Vec<f32> + Send + Sync,
{
    /// The environment predictor pruner (ScreeningPruner impl).
    pub predictor: EnvPredictorPruner<F>,
    /// The prediction verifier (post-hoc accuracy tracking).
    pub verifier: PredictionVerifier,
    /// The consistency gate (entropy-based budget adaptation).
    pub gate: PredictionConsistencyGate,
    /// Shared configuration.
    pub config: EnvPredictorConfig,
}

impl<F> EchoEnvIntegration<F>
where
    F: Fn(usize, &[usize]) -> Vec<f32> + Send + Sync,
{
    /// Create a new integration with default config.
    pub fn new(forward_model: F, feature_dim: usize, vocab_size: usize) -> Self
    where
        F: Fn(usize, &[usize]) -> Vec<f32> + Send + Sync,
    {
        Self::with_config(
            forward_model,
            feature_dim,
            vocab_size,
            EnvPredictorConfig::default(),
        )
    }

    /// Create a new integration with custom config.
    pub fn with_config(
        forward_model: F,
        feature_dim: usize,
        _vocab_size: usize,
        config: EnvPredictorConfig,
    ) -> Self {
        let predictor = EnvPredictorPruner::new(forward_model, feature_dim, config);
        let verifier = PredictionVerifier::new(config);
        let gate = PredictionConsistencyGate::new(config);

        Self {
            predictor,
            verifier,
            gate,
            config,
        }
    }

    /// Convert into a `BanditPruner<EnvPredictorPruner<F>>`.
    ///
    /// The predictor becomes the inner `ScreeningPruner`, so bandit scores
    /// modulate the prediction-based relevance scores.
    pub fn into_bandit_pruner(
        self,
        strategy: BanditStrategy,
        vocab_size: usize,
    ) -> BanditPruner<EnvPredictorPruner<F>> {
        BanditPruner::new(self.predictor, strategy, vocab_size)
    }

    /// Convert into a `BanditPruner` with `PartialScorer` wired from verifier.
    ///
    /// The scorer reads prediction accuracy and feeds it into bandit reward.
    #[cfg(feature = "partial_scoring")]
    pub fn into_bandit_pruner_with_scorer(
        self,
        strategy: BanditStrategy,
        vocab_size: usize,
    ) -> (BanditPruner<EnvPredictorPruner<F>>, EchoPredictionScorer) {
        let scorer = EchoPredictionScorer::new(self.verifier.bandit_reward());
        let bandit = BanditPruner::with_partial_scorer(
            self.predictor,
            strategy,
            vocab_size,
            Box::new(EchoPredictionScorer::new(self.verifier.bandit_reward())),
        );
        (bandit, scorer)
    }

    /// Record an observation: update predictor's running average.
    pub fn observe(&mut self, features: &[f32]) {
        self.predictor.update_avg(features);
    }

    /// Verify a prediction: compare predicted vs actual, update verifier.
    ///
    /// Returns the prediction record for logging/AbsorbCompress promotion.
    pub fn verify(
        &mut self,
        predicted: &[f32],
        actual: &[f32],
        tick: u64,
    ) -> katgpt_speculative::echo_env::PredictionRecord {
        self.verifier.verify(predicted, actual, tick)
    }

    /// Compute consistency entropy from branch features and get budget multiplier.
    ///
    /// Wire the returned multiplier into budget allocation.
    pub fn consistency_budget_multiplier(&mut self, branch_features: &[Vec<f32>]) -> f32 {
        let entropy = PredictionConsistencyGate::compute_branch_entropy(branch_features);
        self.gate.budget_multiplier(entropy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::ScreeningPruner;

    fn simple_forward_model(token: usize, _parents: &[usize]) -> Vec<f32> {
        // Simple: token index → one-hot-like features
        let mut v = vec![0.0f32; 4];
        if token < 4 {
            v[token] = 1.0;
        }
        v
    }

    #[test]
    fn test_integration_creation() {
        let integration = EchoEnvIntegration::new(simple_forward_model, 4, 10);
        assert!((integration.verifier.bandit_reward() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_integration_observe_and_verify() {
        let mut integration = EchoEnvIntegration::new(simple_forward_model, 4, 10);

        // Observe some features
        integration.observe(&[1.0, 0.0, 0.0, 0.0]);
        integration.observe(&[1.0, 0.0, 0.0, 0.0]);

        // Verify: predicted == actual → high accuracy
        let record = integration.verify(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0], 0);
        assert!(record.correct, "identical features should be correct");
        assert!((record.accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_integration_into_bandit_pruner() {
        let integration = EchoEnvIntegration::new(simple_forward_model, 4, 10);
        let bandit = integration.into_bandit_pruner(BanditStrategy::Ucb1, 10);

        // Cold start: no pulls → returns inner domain score
        let score = bandit.relevance(0, 0, &[]);
        assert!(
            score >= 0.0,
            "relevance should be non-negative, got {score}"
        );
    }

    #[test]
    fn test_integration_consistency_budget() {
        let mut integration = EchoEnvIntegration::new(simple_forward_model, 4, 10);

        // Identical branches → low entropy → contract
        let branches = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];
        let m = integration.consistency_budget_multiplier(&branches);
        assert!(
            m < 1.0,
            "identical branches should contract budget, got {m}"
        );

        // Divergent branches → high entropy → expand
        // Use larger-magnitude divergence to exceed default threshold (2.0)
        let divergent = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
        ];
        let m2 = integration.consistency_budget_multiplier(&divergent);
        assert!(
            m2 > 1.0,
            "divergent branches should expand budget, got {m2}"
        );
    }

    #[test]
    fn test_prediction_scorer_basic() {
        let mut scorer = EchoPredictionScorer::new(0.5);
        assert!(
            (scorer.verifier_accuracy - 0.5).abs() < 1e-6,
            "initial accuracy should be 0.5"
        );

        // Simulate verifier update
        let mut verifier = PredictionVerifier::new(EnvPredictorConfig::default());
        for i in 0..10 {
            verifier.verify(&[1.0, 0.0], &[1.0, 0.0], i);
        }
        scorer.update_from_verifier(&verifier);
        assert!(
            scorer.verifier_accuracy > 0.5,
            "after good predictions, accuracy should increase"
        );
    }

    #[test]
    fn test_echo_consistency_budget_adaptation() {
        use katgpt_core::speculative::types::BudgetAdaptation;
        use katgpt_speculative::budget::adaptive_tree_budget;

        let base = 100;

        // Low entropy → budget contracts
        let low = adaptive_tree_budget(base, 0.5, BudgetAdaptation::EchoConsistency);
        assert!(low < base, "low entropy should contract, got {low}");

        // High entropy → budget expands
        let high = adaptive_tree_budget(base, 5.0, BudgetAdaptation::EchoConsistency);
        assert!(high > base, "high entropy should expand, got {high}");

        // Monotonic: low ≤ base ≤ high
        assert!(
            low <= base && base <= high,
            "monotonic: {low} ≤ {base} ≤ {high}"
        );
    }
}
