//! Plug-in trait that bridges [`AcPrefix`] to any concrete causal Transformer
//! forward pass without the primitive naming the weight type.
//!
//! Mirrors the [`SpeculativeGenerator`](crate::traits::SpeculativeGenerator)
//! pattern (open for extension, closed for modification): callers implement
//! [`ForwardForAcPrefix`] over their own weight/cache types and pass
//! `&mut impl ForwardForAcPrefix` to the convenience methods on `AcPrefix`.

use super::{AcPrefix, AcPrefixMask};

/// Plug-in trait for any causal Transformer forward pass that can apply an
/// AC-GPT-style three-region attention mask.
///
/// `forward_for_ac_prefix` runs the model once over the augmented sequence and
/// returns per-position logprobs for the *actual* token at each slot
/// (i.e. `logprobs[i] = log p(token[i] | attended_context_i)`). The caller is
/// responsible for RoPE rotation using `augmented_positions`, for respecting
/// `mask.attends(i, j)`, and for the LM-head log-softmax (the AGENTS.md "sigmoid
/// not softmax" rule applies to blending/decision gates — the LM-head log-softmax
/// over vocab is standard and exempt).
pub trait ForwardForAcPrefix {
    /// Per-position logprobs over the augmented sequence.
    ///
    /// `augmented_tokens`, `augmented_positions`, `mask`, `loss_mask` are as
    /// described in [`AcPrefix::conditional_logprob`]. Returns a `Vec<f32>` of
    /// length `augmented_tokens.len()` — the model's logprob for the *actual*
    /// token at each position.
    fn forward_for_ac_prefix(
        &mut self,
        augmented_tokens: &[u32],
        augmented_positions: &[usize],
        mask: &AcPrefixMask,
        loss_mask: &[f32],
    ) -> Vec<f32>;

    /// Logits at a single eval slot (used by [`AcPrefix::conditional_sample_via`]).
    ///
    /// Default implementation re-runs [`Self::forward_for_ac_prefix`] and
    /// returns the lm_head distribution at `eval_slot`. Override when the
    /// concrete model can compute a single-slot logit cheaper than a full
    /// augmented forward (e.g. KV-cache hit).
    fn forward_logits_at(
        &mut self,
        augmented_tokens: &[u32],
        augmented_positions: &[usize],
        mask: &AcPrefixMask,
        loss_mask: &[f32],
        eval_slot: usize,
    ) -> Vec<f32>;
}

impl<'a> AcPrefix<'a> {
    /// Convenience wrapper around [`Self::conditional_logprob`] that takes a
    /// `&mut impl ForwardForAcPrefix` instead of a bare closure.
    pub fn conditional_logprob_via<F: ForwardForAcPrefix>(&self, fwd: &mut F) -> f32 {
        let n = self.augmented_len();
        let mut augmented_tokens = vec![0u32; n];
        let mut augmented_positions = vec![0usize; n];
        let mut loss_mask = vec![0.0f32; n];
        self.augmented_tokens_into(&mut augmented_tokens);
        self.original_positions_into(&mut augmented_positions);
        self.loss_mask_into(&mut loss_mask);
        let mask = AcPrefixMask::materialize_from(self);
        let logprobs =
            fwd.forward_for_ac_prefix(&augmented_tokens, &augmented_positions, &mask, &loss_mask);
        debug_assert_eq!(
            logprobs.len(),
            n,
            "forward_for_ac_prefix must return one logprob per augmented slot"
        );
        let mut acc = 0.0f32;
        for (lp, m) in logprobs.iter().zip(loss_mask.iter()) {
            acc += *lp * *m;
        }
        acc
    }

    /// Convenience wrapper around [`Self::conditional_sample`] that takes a
    /// `&mut impl ForwardForAcPrefix` and the RNG.
    pub fn conditional_sample_via<F: ForwardForAcPrefix>(
        &self,
        fwd: &mut F,
        rng: &mut fastrand::Rng,
    ) -> Vec<u32> {
        let n = self.augmented_len();
        let mut augmented_tokens = vec![0u32; n];
        let mut augmented_positions = vec![0usize; n];
        let mut loss_mask = vec![0.0f32; n];
        self.augmented_tokens_into(&mut augmented_tokens);
        self.original_positions_into(&mut augmented_positions);
        self.loss_mask_into(&mut loss_mask);
        let mask = AcPrefixMask::materialize_from(self);

        let mut sampled = Vec::with_capacity(n);
        for eval_slot in 0..n {
            if loss_mask[eval_slot] == 0.0 {
                continue;
            }
            let logits = fwd.forward_logits_at(
                &augmented_tokens,
                &augmented_positions,
                &mask,
                &loss_mask,
                eval_slot,
            );
            let token = crate::ac_prefix::gumbel_max_sample(&logits, rng);
            augmented_tokens[eval_slot] = token;
            sampled.push(token);
        }
        sampled
    }
}
