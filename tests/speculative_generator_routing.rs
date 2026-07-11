//! Plan 193 T14: Integration test for SpeculativeGenerator routing via InferenceRouter.
//!
//! Validates that InferenceRouter can dispatch SpeculativeGenerator instances
//! and prune results through GenerativeConstraintPruner.

#![cfg(feature = "speculative_generator")]

use katgpt_core::GenerativeConstraintPruner;
use katgpt_rs::inference_router::InferenceRouter;
use katgpt_rs::speculative::{MarginalTokenGenerator, TokenCondition, TokenOutput};
use katgpt_core::trigger_gate::TriggerGateConfig;
use katgpt_rs::types::Config;

/// Simple pruner that accepts everything — for testing routing only.
struct AcceptAllPruner;

impl GenerativeConstraintPruner<TokenOutput> for AcceptAllPruner {
    fn is_valid(&self, _output: &TokenOutput) -> bool {
        true
    }
}

/// Pruner that rejects tokens above index 5 — for testing pruning.
struct LowTokenPruner;

impl GenerativeConstraintPruner<TokenOutput> for LowTokenPruner {
    fn is_valid(&self, output: &TokenOutput) -> bool {
        output.token_idx <= 5
    }
}

#[test]
fn test_generate_validated_accepts_all() {
    let mut generator = MarginalTokenGenerator { top_k: 4 };
    let pruner = AcceptAllPruner;

    let condition = TokenCondition {
        parent_tokens: vec![],
        depth: 0,
        marginals: vec![0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    };

    let mut rng = fastrand::Rng::new();
    let mut router = create_test_router();

    let results = router.generate_validated(&mut generator, &pruner, &condition, &mut rng);
    assert_eq!(results.len(), 4, "All 4 candidates should pass AcceptAll");
}

#[test]
fn test_generate_validated_prunes_high_tokens() {
    let mut generator = MarginalTokenGenerator { top_k: 8 };
    let pruner = LowTokenPruner;

    let condition = TokenCondition {
        parent_tokens: vec![],
        depth: 0,
        marginals: vec![0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04], // tokens 0-7
    };

    let mut rng = fastrand::Rng::new();
    let mut router = create_test_router();

    let results = router.generate_validated(&mut generator, &pruner, &condition, &mut rng);
    // Only tokens 0-5 should survive (indices <= 5)
    for output in &results {
        assert!(
            output.token_idx <= 5,
            "Token {} should have been pruned",
            output.token_idx
        );
    }
    assert!(results.len() <= 6, "Should have at most 6 valid tokens");
}

#[test]
fn test_generate_validated_empty_candidates() {
    let mut generator = MarginalTokenGenerator { top_k: 4 };
    let pruner = AcceptAllPruner;

    let condition = TokenCondition {
        parent_tokens: vec![],
        depth: 0,
        marginals: vec![], // empty → no candidates
    };

    let mut rng = fastrand::Rng::new();
    let mut router = create_test_router();

    let results = router.generate_validated(&mut generator, &pruner, &condition, &mut rng);
    assert!(
        results.is_empty(),
        "Empty marginals should produce no candidates"
    );
}

fn create_test_router() -> InferenceRouter {
    InferenceRouter::new(TriggerGateConfig::default(), Config::micro(), false, false)
}
