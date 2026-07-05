//! FlashAR Strided Anchor-Then-Fill D2F Decoding — root re-export shim.
//!
//! Plan 400 (2026-07-05): the production code moved to
//! `crates/katgpt-forward/src/flashar_anchor.rs`. This file is now a thin
//! re-export shim that preserves the historical
//! `crate::speculative::flashar_anchor::*` import path, plus the 2
//! training-coupled tests that depend on `crate::dllm::{train_mini_dllm,
//! generate_pattern_dataset}` (root-only training code).

#![allow(clippy::too_many_arguments)]

pub use katgpt_forward::flashar_anchor::{AnchorConfig, AnchorFillResult, anchor_then_fill};

// ---------------------------------------------------------------------------
// Tests — 2 training-coupled tests that cannot move to katgpt-forward.
//
// These tests call `crate::dllm::{generate_pattern_dataset, train_mini_dllm}`
// which is root-only training code. The 6 inference-only tests moved with
// the production file to `katgpt-forward/src/flashar_anchor.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::dllm::D2fContext;
    use crate::speculative::d2f::D2fDecodeConfig;
    use crate::speculative::flashar_anchor::{AnchorConfig, anchor_then_fill};
    use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
    use crate::types::{Config, Rng};

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    /// Train a mini D2F model on pattern data (same recipe as
    /// `speculative::d2f::tests::test_decode_with_trained_model`).
    ///
    /// The anchor-then-fill tests assert properties that only hold for a
    /// trained model — a random `TransformerWeights::new` produces a
    /// degenerate all-mask baseline that trivially converges in 1 step by
    /// emitting the same token at every position, which inverts the
    /// step-reduction comparison the tests are trying to verify.
    fn make_trained_weights() -> (Config, TransformerWeights) {
        use crate::dllm::{generate_pattern_dataset, train_mini_dllm};
        let config = make_config();
        let mut train_rng = Rng::new(123);
        let train_data =
            generate_pattern_dataset(&mut train_rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut train_rng, 5, config.block_size, config.vocab_size - 1);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);
        (config, weights)
    }

    #[test]
    fn test_anchor_then_fill_produces_valid_output() {
        // Uses a trained mini D2F model — random weights produce a degenerate
        // all-same-token output that doesn't exercise the fill path.
        let (config, weights) = make_trained_weights();
        let mut rng = Rng::new(42);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0, // seed_token
            0, // start_pos
            &mut rng,
        );

        // Should produce block_size tokens
        assert_eq!(result.tokens.len(), block_size);
        // All tokens should be valid (non-mask)
        let mask = config.mask_token;
        for (i, &t) in result.tokens.iter().enumerate() {
            assert_ne!(t, mask, "token at position {i} should not be mask");
        }
        // Should have anchors
        assert!(result.n_anchors > 0, "should have at least 1 anchor");
    }

    #[test]
    fn test_anchor_then_fill_reduces_steps() {
        // The "anchors reduce denoising steps" property only holds for a
        // trained model — with random weights, the all-mask baseline
        // degenerately converges in 1 step (same token at every position),
        // inverting the comparison. See `make_trained_weights` doc.
        let (config, weights) = make_trained_weights();
        let mut rng = Rng::new(42);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng,
        );

        // With anchors pre-filled, fill steps should be ≤ baseline steps
        // (anchors reduce the denoising search space)
        assert!(
            result.fill_steps_used <= result.baseline_steps_used,
            "anchor fill ({}) should use ≤ baseline ({}) steps",
            result.fill_steps_used,
            result.baseline_steps_used,
        );

        println!(
            "  anchors={}, fill_steps={}, baseline_steps={}, reduction={}",
            result.n_anchors,
            result.fill_steps_used,
            result.baseline_steps_used,
            result.step_reduction,
        );
    }
}
