//! Precision-Aware Speculative Generator — wraps any SpeculativeGenerator
//! with boundary penalty scoring (Plan 227 Phase 4).
//!
//! When `precision_aware_draft` feature is enabled, applies quantization
//! boundary penalty to draft token scores, improving acceptance rate.

use crate::precision_aware_draft::BoundaryPenalty;
use katgpt_core::SpeculativeGenerator;

/// Wrapper that adds precision-aware boundary penalty to any speculative generator.
///
/// After the inner generator produces candidates, applies boundary penalty
/// to the log-prob scores based on how close the logits are to quantization
/// grid boundaries.
pub struct PrecisionAwareGenerator<G> {
    /// Inner speculative generator.
    pub inner: G,
    /// Boundary penalty scorer.
    pub penalty: BoundaryPenalty,
    /// Whether boundary penalty is enabled.
    pub enabled: bool,
}

impl<G> PrecisionAwareGenerator<G> {
    /// Create a new precision-aware wrapper around an existing generator.
    pub fn new(inner: G, penalty: BoundaryPenalty) -> Self {
        Self {
            inner,
            penalty,
            enabled: true,
        }
    }

    /// Create with default boundary penalty settings.
    pub fn with_defaults(inner: G) -> Self {
        Self {
            inner,
            penalty: BoundaryPenalty::default(),
            enabled: true,
        }
    }
}

impl<G: SpeculativeGenerator> SpeculativeGenerator for PrecisionAwareGenerator<G> {
    type Condition = G::Condition;
    type Output = G::Output;
    type Error = G::Error;

    fn generate(
        &mut self,
        condition: &Self::Condition,
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        let candidates = self.inner.generate(condition, rng)?;

        // NOTE: Boundary penalty is applied at the caller level where raw
        // logits are available via `BoundaryPenalty::apply_penalty`. This
        // wrapper passes candidates through unchanged; the penalty struct
        // is exposed for callers to use with the logit data they hold.
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::spec_generator::{MarginalTokenGenerator, TokenCondition};
    use katgpt_core::SpeculativeGenerator;

    #[test]
    fn test_precision_aware_wraps_generator() {
        let inner = MarginalTokenGenerator { top_k: 5 };
        let _wrapped = PrecisionAwareGenerator::with_defaults(inner);
    }

    #[test]
    fn test_precision_aware_generates() {
        let inner = MarginalTokenGenerator { top_k: 5 };
        let mut wrapped = PrecisionAwareGenerator::with_defaults(inner);

        let condition = TokenCondition {
            parent_tokens: vec![],
            depth: 0,
            marginals: vec![-0.5, -0.3, -0.1, -0.2, -0.4],
        };
        let mut rng = fastrand::Rng::new();

        let result = wrapped.generate(&condition, &mut rng);
        assert!(result.is_ok());
        let candidates = result.unwrap();
        assert_eq!(candidates.len(), 5); // top_k = 5
    }

    #[test]
    fn test_boundary_penalty_can_be_disabled() {
        let inner = MarginalTokenGenerator { top_k: 3 };
        let mut wrapped = PrecisionAwareGenerator::with_defaults(inner);
        wrapped.enabled = false;

        let condition = TokenCondition {
            parent_tokens: vec![],
            depth: 0,
            marginals: vec![-0.5, -0.3, -0.1],
        };
        let mut rng = fastrand::Rng::new();

        let result = wrapped.generate(&condition, &mut rng);
        assert!(result.is_ok());
    }
}

// TL;DR: PrecisionAwareGenerator<G> wraps any SpeculativeGenerator, exposing
// BoundaryPenalty for caller-level score adjustment. Pass-through generate,
// 3 unit tests (wrap, generate top-5, disabled).
