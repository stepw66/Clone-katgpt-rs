//! SpeculativeGenerator token-domain implementation (Plan 193 Phase 1).
//!
//! Provides `MarginalTokenGenerator` — a simple top-K marginal sampler
//! implementing `SpeculativeGenerator<Condition=TokenCondition, Output=TokenOutput>`.
//!
//! Also provides `TokenConstraintPruner<P>` — adapter from `ConstraintPruner`
//! to `GenerativeConstraintPruner<TokenOutput>`.

use katgpt_core::{ConstraintPruner, GenerativeConstraintPruner, SpeculativeGenerator};

// ── Types ──────────────────────────────────────────────────────

/// Token generation condition: current context + marginal probabilities.
#[derive(Clone, Debug)]
pub struct TokenCondition {
    /// Token indices placed at earlier depths in the current path.
    pub parent_tokens: Vec<usize>,
    /// Current tree depth.
    pub depth: usize,
    /// Marginal log-probabilities per token index.
    pub marginals: Vec<f32>,
}

/// Token generator output: a logit-indexed candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenOutput {
    /// Token index in the vocabulary.
    pub token_idx: usize,
    /// Log-probability of this token.
    pub log_prob: f32,
}

/// Token generator error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenGenError {
    /// No marginals provided.
    NoMarginals,
}

// ── MarginalTokenGenerator ─────────────────────────────────────

/// Simple top-K marginal sampler for token candidates.
///
/// Sorts marginals by log-prob descending and returns the top-K.
/// The `rng` parameter is accepted for trait compatibility but unused
/// since selection is deterministic top-K.
pub struct MarginalTokenGenerator {
    /// Maximum number of candidates to return.
    pub top_k: usize,
}

impl SpeculativeGenerator for MarginalTokenGenerator {
    type Condition = TokenCondition;
    type Output = TokenOutput;
    type Error = TokenGenError;

    #[inline]
    fn generate(
        &mut self,
        condition: &Self::Condition,
        _rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        if condition.marginals.is_empty() {
            return Err(TokenGenError::NoMarginals);
        }

        let mut indexed: Vec<(usize, f32)> =
            condition.marginals.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(indexed
            .into_iter()
            .take(self.top_k)
            .map(|(idx, lp)| TokenOutput {
                token_idx: idx,
                log_prob: lp,
            })
            .collect())
    }
}

// ── TokenConstraintPruner ──────────────────────────────────────

/// Adapter: wraps an existing `ConstraintPruner` as a typed
/// `GenerativeConstraintPruner<TokenOutput>`.
///
/// Maps `TokenOutput` fields to the `ConstraintPruner::is_valid` signature:
/// `is_valid(depth, token_idx, parent_tokens)`.
pub struct TokenConstraintPruner<P> {
    inner: P,
}

impl<P> TokenConstraintPruner<P> {
    #[inline]
    pub fn new(pruner: P) -> Self {
        Self { inner: pruner }
    }
}

impl<P: ConstraintPruner> GenerativeConstraintPruner<TokenOutput> for TokenConstraintPruner<P> {
    #[inline]
    fn is_valid(&self, output: &TokenOutput) -> bool {
        // TokenCondition carries parent context, but the pruner interface
        // needs it at construction. For the adapter, we pass empty parents
        // at depth 0 — the caller should construct a depth-aware wrapper
        // when needed.
        self.inner.is_valid(0, output.token_idx, &[])
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::NoPruner;

    #[test]
    fn test_marginal_token_generator_top_k() {
        let mut generator = MarginalTokenGenerator { top_k: 3 };
        let condition = TokenCondition {
            parent_tokens: vec![],
            depth: 0,
            marginals: vec![0.1, 0.5, 0.3, 0.05, 0.9],
        };
        let mut rng = fastrand::Rng::new();
        let result = generator.generate(&condition, &mut rng).unwrap();

        assert_eq!(result.len(), 3);
        // Top-3 by descending log-prob: index 4 (0.9), 1 (0.5), 2 (0.3)
        assert_eq!(result[0].token_idx, 4);
        assert_eq!(result[0].log_prob, 0.9);
        assert_eq!(result[1].token_idx, 1);
        assert_eq!(result[1].log_prob, 0.5);
        assert_eq!(result[2].token_idx, 2);
        assert_eq!(result[2].log_prob, 0.3);
    }

    #[test]
    fn test_marginal_token_generator_empty_marginals() {
        let mut generator = MarginalTokenGenerator { top_k: 3 };
        let condition = TokenCondition {
            parent_tokens: vec![],
            depth: 0,
            marginals: vec![],
        };
        let mut rng = fastrand::Rng::new();
        assert_eq!(
            generator.generate(&condition, &mut rng),
            Err(TokenGenError::NoMarginals)
        );
    }

    #[test]
    fn test_token_constraint_pruner_no_pruner() {
        let pruner = TokenConstraintPruner::new(NoPruner);
        let output = TokenOutput {
            token_idx: 42,
            log_prob: -1.0,
        };
        assert!(pruner.is_valid(&output));
    }

    #[test]
    fn test_batch_generate() {
        let mut generator = MarginalTokenGenerator { top_k: 2 };
        let conditions = vec![
            TokenCondition {
                parent_tokens: vec![],
                depth: 0,
                marginals: vec![0.1, 0.9],
            },
            TokenCondition {
                parent_tokens: vec![1],
                depth: 1,
                marginals: vec![0.5, 0.2, 0.8],
            },
        ];
        let mut rng = fastrand::Rng::new();
        let results = generator.generate_batch(&conditions, &mut rng).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[0][0].token_idx, 1); // 0.9
        assert_eq!(results[1].len(), 2);
        assert_eq!(results[1][0].token_idx, 2); // 0.8
    }

    #[test]
    fn test_batch_is_valid() {
        let pruner = TokenConstraintPruner::new(NoPruner);
        let outputs = vec![
            TokenOutput {
                token_idx: 0,
                log_prob: -0.1,
            },
            TokenOutput {
                token_idx: 1,
                log_prob: -0.5,
            },
            TokenOutput {
                token_idx: 2,
                log_prob: -1.0,
            },
        ];
        let results = pruner.batch_is_valid(&outputs);
        assert_eq!(results, vec![true, true, true]);
    }
}

// TL;DR: SpeculativeGenerator + GenerativeConstraintPruner token-domain impl.
// MarginalTokenGenerator does top-K from marginals; TokenConstraintPruner wraps
// ConstraintPruner for typed output validation. 6 tests covering top-K, empty,
// NoPruner adapter, batch generate, and batch validation.
