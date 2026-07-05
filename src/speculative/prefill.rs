//! Speculative prefill: forward-coupled impls (Plan 390).
//!
//! This file is the **root-resident complement** to
//! `katgpt_speculative::prefill`, which hosts the pure substrate half (the
//! `PrefillScorer` trait + substrate scorers + pure compression/selection
//! functions + orchestrators). It was extracted in Plan 390 (2026-07-05) via
//! the **trait-impl split** technique refined in Plan 389.
//!
//! ## Why this file is root-only
//!
//! The remaining items need either:
//! - `crate::transformer::forward` + `crate::speculative::types::SpeculativeContext`
//!   (forward-cycle blocker — see Proposal 003 Phase 16 DEFER), or
//! - `crate::dash_attn::{entmax_1p5, entmax_support}` — a re-export from
//!   `katgpt-attn`'s heavy `dash_attn` feature chain (would pull
//!   katgpt-forward, katgpt-pruners/bandit, katgpt-kv, katgpt-transformer, serde).
//!
//! ## Contents
//!
//! - Re-export of the substrate from katgpt-speculative (back-compat for
//!   `katgpt_rs::speculative::prefill::*` paths).
//! - `AttentionScorer`, `BlockAttentionScorer` (forward-coupled impls).
//! - `block_select_entmax` (entmax-coupled, gated `dash_attn`).

// Re-export the pure substrate from the leaf crate so historical
// `crate::speculative::prefill::*` paths resolve unchanged.
pub use katgpt_speculative::prefill::{
    PrefillScorer, RandomScorer, UniformScorer, block_compression_ratio, block_select,
    block_select_grid, compress_prompt, compress_prompt_blocks, should_compress,
    speculative_prefill, speculative_prefill_adaptive, speculative_prefill_block,
};
// `block_score_maxsim` is gated `maxsim` in the leaf crate; preserve the same
// gate on the re-export.
#[cfg(feature = "maxsim")]
pub use katgpt_speculative::prefill::block_score_maxsim;

use crate::speculative::types::{FlashPrefillConfig, SpeculativeContext};
use crate::transformer::{TransformerWeights, forward};
use crate::types::Config;

// ── Attention Scorer (forward-coupled) ────────────────────────

/// Attention-based importance scorer (PFlash-inspired).
/// Uses softmax'd self-attention weights from draft model forward pass.
///
/// After each `forward()` call, `ctx.scores[0..=pos]` contains the last
/// attention head's normalized attention weights. The weight at index `pos`
/// (the self-attention weight) serves as a proxy for per-token importance.
pub struct AttentionScorer;

impl AttentionScorer {
    /// Zero-alloc scoring using pre-allocated context.
    /// `scores` must be `>= prompt_tokens.len()`.
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        if prompt_tokens.is_empty() {
            return;
        }

        sctx.cache.reset();

        let len = prompt_tokens.len().min(scores.len());
        scores[..len].fill(0.0f32);

        for (pos, &token) in prompt_tokens.iter().enumerate() {
            if pos >= draft_config.block_size {
                break;
            }
            let _logits = forward(
                &mut sctx.ctx,
                draft_weights,
                &mut sctx.cache,
                token,
                pos,
                draft_config,
            );
            scores[pos] = sctx.ctx.scores[pos];
        }

        // Normalize scores to [0, 1] range
        let max_score = scores[..len].iter().cloned().fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for s in scores[..len].iter_mut() {
                *s /= max_score;
            }
        }
    }
}

impl PrefillScorer for AttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        if prompt_tokens.is_empty() {
            return Vec::new();
        }

        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            prompt_tokens,
            &mut scores,
        );
        scores
    }
}

// ── Block-Attention Scorer (forward-coupled) ─────────────────

/// Block-sparse attention scorer (CPU fallback for PFlash).
///
/// Aggregates token-level attention scores into block-level importance,
/// then upsamples back to per-token scores for compression.
pub struct BlockAttentionScorer {
    pub config: FlashPrefillConfig,
}

impl BlockAttentionScorer {
    /// Zero-alloc scoring using pre-allocated context.
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        let block_size = self.config.block_size;
        let seq_len = prompt_tokens.len();
        let num_blocks = seq_len.div_ceil(block_size);

        if seq_len == 0 {
            return;
        }

        sctx.cache.reset();

        let filled = seq_len.min(draft_config.block_size);
        for (pos, &token) in prompt_tokens.iter().enumerate().take(filled) {
            let _logits = forward(
                &mut sctx.ctx,
                draft_weights,
                &mut sctx.cache,
                token,
                pos,
                draft_config,
            );
        }

        let mut block_scores = vec![0.0f32; num_blocks];
        let mut block_counts = vec![0usize; num_blocks];

        let tail_start = seq_len.saturating_sub(self.config.tail_window * block_size);
        for pos in tail_start..filled {
            let score = sctx.ctx.scores[pos];
            let block_idx = pos / block_size;
            if block_idx < num_blocks {
                block_scores[block_idx] += score;
                block_counts[block_idx] += 1;
            }
        }

        for i in 0..num_blocks {
            if block_counts[i] > 0 {
                block_scores[i] /= block_counts[i] as f32;
            }
        }

        let max_block = block_scores.iter().cloned().fold(0.0f32, f32::max);
        if max_block > 0.0 {
            for s in &mut block_scores {
                *s /= max_block;
            }
        }

        for (pos, slot) in scores.iter_mut().enumerate().take(seq_len) {
            *slot = block_scores[pos / block_size];
        }
    }
}

impl PrefillScorer for BlockAttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            prompt_tokens,
            &mut scores,
        );
        scores
    }
}

// ── Adaptive block selection using α-entmax (Plan 106 T20) ────
//
// Stays in root because `crate::dash_attn::{entmax_1p5, entmax_support}` is a
// re-export from katgpt-attn's heavy `dash_attn` feature chain (pulls
// katgpt-forward, katgpt-pruners/bandit, katgpt-kv, katgpt-transformer, serde).
// Pulling that into katgpt-speculative for one function would be disproportionate.
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
// Only forward-coupled + entmax-coupled tests live here. The pure substrate
// tests (compress_prompt, block_select, RandomScorer, UniformScorer, NIAH,
// should_compress, etc.) moved to katgpt_speculative::prefill::tests in
// Plan 390.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::Rng;

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_attention_scorer_produces_scores() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = (0..config.vocab_size.min(8)).collect();
        let scorer = AttentionScorer;
        let scores = scorer.score(&weights, &config, &tokens);
        assert_eq!(scores.len(), tokens.len());
        // All scores should be finite
        for &s in &scores {
            assert!(s.is_finite(), "score should be finite, got {s}");
        }
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
