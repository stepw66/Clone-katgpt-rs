//! Speculative prefill: root-only complement (Plan 394, 2026-07-05).
//!
//! The forward-coupled scorers (`AttentionScorer`, `BlockAttentionScorer`) and
//! the pure substrate re-export shim moved to `katgpt_forward::prefill` in
//! Plan 394. This file is the slim root-side complement that retains:
//!
//! - `block_select_entmax` (Plan 106 T20) — gated `dash_attn`. Stays root
//!   because `crate::dash_attn::{entmax_1p5, entmax_support}` is a re-export
//!   from katgpt-attn's heavy `dash_attn` feature chain (would pull
//!   katgpt-forward, katgpt-pruners/bandit, katgpt-kv, katgpt-transformer,
//!   serde). Adding katgpt-attn as a dep of katgpt-forward would create a
//!   cycle (katgpt-attn already depends on katgpt-forward for forward glue).
//! - Re-exports of the moved symbols so historical
//!   `katgpt_rs::speculative::prefill::{AttentionScorer, BlockAttentionScorer}`
//!   paths keep resolving (the root mod.rs's `pub use prefill::{...}` lines
//!   expect these names to be visible at `crate::speculative::prefill::*`).
//! - Bridge test (gated `rest`) — depends on `SimulatedVerifier` (now in
//!   katgpt-forward, re-exported through `crate::speculative::SimulatedVerifier`)
//!   and `crate::transformer::forward` (re-export of `katgpt_forward::forward`).

// Plan 394 (2026-07-05): scorers + substrate re-export shim moved to
// katgpt-forward. Re-export them here so historical paths resolve.
pub use katgpt_forward::prefill::{
    AttentionScorer, BlockAttentionScorer, PrefillScorer, RandomScorer, UniformScorer,
    block_compression_ratio, block_select, block_select_grid, compress_prompt,
    compress_prompt_blocks, should_compress, speculative_prefill, speculative_prefill_adaptive,
    speculative_prefill_block,
};
#[cfg(feature = "maxsim")]
pub use katgpt_forward::prefill::block_score_maxsim;

// `FlashPrefillConfig` is only used by the `dash_attn`-gated `block_select_entmax`.
// Gate the import to match so it doesn't read as unused under default features.
#[cfg(feature = "dash_attn")]
use crate::speculative::types::FlashPrefillConfig;

// ── Adaptive block selection using α-entmax (Plan 106 T20) ────
//
// Stays in root because `crate::dash_attn::{entmax_1p5, entmax_support}` is a
// re-export from katgpt-attn's heavy `dash_attn` feature chain (pulls
// katgpt-forward, katgpt-pruners/bandit, katgpt-kv, katgpt-transformer, serde).
// Pulling that into katgpt-forward for one function would create a cycle.
//
// See `katgpt_speculative::prefill` for the pure substrate `block_select`.

/// Adaptive block selection using α-entmax (α=1.5) sparse routing.
///
/// Unlike `block_select()` which uses fixed top-k via alpha threshold,
/// this uses entmax to produce a sparse probability distribution over blocks
/// and selects all blocks with non-zero probability — the support size
/// varies per query, adapting to difficulty.
///
/// Rules:
///   - sink:       k_block < attention_sink → always include
///   - window:     |q_block - k_block| < window → include (recent context)
///   - causal:     k_block <= q_block
///   - entmax:     α-entmax produces sparse probs; select all with p > 0
///
/// # Arguments
///
/// * `block_scores` - Per-block importance scores (same as `block_select`)
/// * `cfg` - PFlash config for sink/window rules
///
/// # Returns
///
/// Variable-length `Vec<usize>` of selected block indices. The number of
/// selected blocks varies per query — hard queries select more, easy ones fewer.
#[cfg(feature = "dash_attn")]
pub fn block_select_entmax(block_scores: &[f32], cfg: &FlashPrefillConfig) -> Vec<usize> {
    use crate::dash_attn::{entmax_1p5, entmax_support};

    let num_blocks = block_scores.len();
    if num_blocks == 0 {
        return Vec::new();
    }

    let q_block = num_blocks - 1;

    // Apply α-entmax routing to get sparse probability distribution
    let (probs, _tau) = entmax_1p5(block_scores);
    let entmax_selected = entmax_support(&probs);

    // Fallback to block_select() if entmax produces empty support (e.g. NaN inputs)
    if entmax_selected.is_empty() {
        return block_select(block_scores, cfg);
    }

    let mut selected: Vec<usize> = Vec::with_capacity(num_blocks);

    for (k_block, _) in block_scores.iter().enumerate() {
        if k_block > q_block {
            continue;
        }

        // Sink + window rules are unconditional; entmax replaces the alpha threshold.
        // Linear `Vec::contains` beats `HashSet` for typical block counts (< 64) —
        // avoids hashing overhead and a heap allocation per call.
        let keep = k_block < cfg.attention_sink
            || q_block.abs_diff(k_block) < cfg.window
            || entmax_selected.contains(&k_block);

        if keep {
            selected.push(k_block);
        }
    }

    // Monotonic k_block iteration keeps `selected` sorted & unique.
    selected
}

// ── Tests ──────────────────────────────────────────────────────
//
// Only the root-only tests (entmax + rest-bridge) live here. The forward-coupled
// scorer test moved to katgpt-forward::prefill::tests. The pure substrate tests
// (compress_prompt, block_select, RandomScorer, UniformScorer, NIAH,
// should_compress, etc.) live in katgpt_speculative::prefill::tests (Plan 390).

#[cfg(test)]
mod tests {
    use super::*;
    // Only used by the `rest`-gated bridge test + helper; gate the imports to
    // match so they don't read as unused under default features.
    #[cfg(feature = "rest")]
    use crate::transformer::TransformerWeights;
    #[cfg(feature = "rest")]
    use crate::types::Rng;
    #[cfg(feature = "rest")]
    use crate::types::Config;

    // Only used by the `rest`-gated bridge test; gate the helper to match so it
    // doesn't read as dead code under default features.
    #[cfg(feature = "rest")]
    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    /// Bridge test: prefill compression → KV cache fill → speculative decode.
    /// Validates the data flow from prompt compression through to speculative
    /// step, simulating what speculative_step_rest() would do with a real REST target.
    #[cfg(feature = "rest")]
    #[test]
    fn test_bridge_prefill_to_speculative_decode() {
        use crate::speculative::{SimulatedVerifier, speculative_step_verifier};
        use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};

        // Config with block_size large enough for 32 tokens
        let config = Config {
            block_size: 64,
            ..Config::draft()
        };
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // 1. Create a 32-token prompt (vocab_size=27, so cycle tokens)
        let prompt_tokens: Vec<usize> = (0..32).map(|i| i % config.vocab_size).collect();
        assert_eq!(prompt_tokens.len(), 32);

        // 2. Run speculative prefill to compress
        let compressed_indices =
            speculative_prefill(&UniformScorer, &weights, &config, &prompt_tokens, 0.5, 2, 2);

        // Compression should reduce the prompt
        assert!(!compressed_indices.is_empty(), "should select some tokens");
        let compressed_len = compressed_indices.len();
        assert!(
            compressed_len < prompt_tokens.len(),
            "should compress from 32 to {compressed_len}"
        );

        // Indices should be in ascending order (compress_prompt guarantees this)
        for window in compressed_indices.windows(2) {
            assert!(window[0] < window[1], "indices should be sorted ascending");
        }

        // 3. Build target KV cache and fill with compressed tokens
        let mut target_cache = MultiLayerKVCache::new(&config);
        let mut target_ctx = ForwardContext::new(&config);

        for (pos, &idx) in compressed_indices.iter().enumerate() {
            let token = prompt_tokens[idx];
            let _logits = forward(
                &mut target_ctx,
                &weights,
                &mut target_cache,
                token,
                pos,
                &config,
            );
        }

        // 4. Verify target KV cache is populated (has non-zero values)
        let filled_positions = compressed_indices.len();
        let kv_dim = crate::types::kv_dim(&config);
        for layer in &target_cache.layers {
            let end = filled_positions * kv_dim;
            let key_nonzero = layer.key[..end].iter().any(|&v| v != 0.0);
            assert!(
                key_nonzero,
                "target KV cache key should have non-zero values"
            );
            let val_nonzero = layer.value[..end].iter().any(|&v| v != 0.0);
            assert!(
                val_nonzero,
                "target KV cache value should have non-zero values"
            );
        }

        // 5. Run speculative step with verifier using the filled cache state
        //    (simulates what speculative_step_rest does after prefill)
        let next_pos = filled_positions;
        let last_token = prompt_tokens[*compressed_indices.last().unwrap()];
        let mut verifier = SimulatedVerifier::new(0.75, &config);
        let mut step_rng = Rng::new(123);
        let (accepted, accept_len) = speculative_step_verifier(
            &weights,
            &config,
            last_token,
            next_pos,
            &mut step_rng,
            &mut verifier,
        );

        // Should always accept at least one token
        assert!(
            !accepted.is_empty(),
            "should accept at least 1 token from speculative step"
        );
        assert!(accept_len >= 1, "accept_len should be >= 1");
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of vocab range");
        }
    }

    // ── Plan 106 T20: block_select_entmax tests ──────────────

    #[cfg(feature = "dash_attn")]
    #[test]
    fn test_block_select_entmax_produces_valid_indices() {
        let cfg = FlashPrefillConfig {
            attention_sink: 1,
            window: 1,
            ..Default::default()
        };
        let scores = vec![0.8, 0.2, 0.5, 0.1, 0.9];
        let selected = block_select_entmax(&scores, &cfg);

        // All indices must be within bounds
        for &idx in &selected {
            assert!(
                idx < scores.len(),
                "index {idx} out of bounds (max {})",
                scores.len()
            );
        }

        // Indices should be sorted and unique (dedup contract)
        for w in selected.windows(2) {
            assert!(w[0] < w[1], "indices should be sorted ascending");
        }

        // Should select at least sink + window blocks
        assert!(!selected.is_empty(), "should select at least some blocks");
    }

    #[cfg(feature = "dash_attn")]
    #[test]
    fn test_block_select_entmax_adaptive_support() {
        let cfg = FlashPrefillConfig {
            attention_sink: 0,
            window: 0,
            ..Default::default()
        };

        // Concentrated scores: one dominant block → entmax selects few
        let concentrated = vec![10.0f32, 0.01, 0.01, 0.01, 0.01];
        let selected_concentrated = block_select_entmax(&concentrated, &cfg);

        // Uniform scores: all equal → entmax selects all
        let uniform = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let selected_uniform = block_select_entmax(&uniform, &cfg);

        // Adaptive: different inputs select different numbers of blocks
        assert_ne!(
            selected_concentrated.len(),
            selected_uniform.len(),
            "concentrated scores should select fewer blocks than uniform"
        );
    }

    #[cfg(feature = "dash_attn")]
    #[test]
    fn test_block_select_entmax_fallback_on_empty() {
        let cfg = FlashPrefillConfig {
            attention_sink: 1,
            window: 1,
            ..Default::default()
        };
        // NaN scores cause entmax to produce empty support → triggers fallback
        let scores = vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN];
        let selected = block_select_entmax(&scores, &cfg);
        let fallback = block_select(&scores, &cfg);

        assert_eq!(
            selected, fallback,
            "should fall back to block_select when entmax produces empty support"
        );
        // Fallback should still select sink + window blocks
        assert!(
            !selected.is_empty(),
            "fallback should select sink/window blocks"
        );
    }
}
