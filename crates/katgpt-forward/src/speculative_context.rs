//! Speculative-decoding composition context (Plan 393, 2026-07-05).
//!
//! `SpeculativeContext` moved here from root `src/speculative/types.rs` because
//! it composes `ForwardContext` (which lives in this crate since Issue 007
//! Phase F). It cannot live in `katgpt-speculative` (the leaf below us) because
//! that would force `katgpt-forward → katgpt-speculative → katgpt-forward` — a
//! cycle. katgpt-forward is the lowest crate that can see both `ForwardContext`
//! and the speculative substrate types.
//!
//! Root re-exports this via `pub use katgpt_forward::SpeculativeContext;` so all
//! `crate::speculative::types::SpeculativeContext` and
//! `crate::speculative::SpeculativeContext` import paths continue to resolve.

use crate::ForwardContext;
use katgpt_core::speculative::types::SdeConfig;
use katgpt_transformer::{KVSnapshot, MultiLayerKVCache};
use katgpt_types::Config;

/// Pre-allocated buffers for zero-alloc speculative decoding.
///
/// Create once with `SpeculativeContext::new(config)`, reuse across calls.
/// Call `reset()` between decode steps to clear transient state.
///
/// All hot-path operations borrow from this struct instead of allocating:
/// - `dflash_predict_with` reuses `ctx`, `cache`, `marginals_flat`, `probs_buf`
/// - `build_dd_tree` reuses `TreeBuilder` heap/tree buffers
/// - `sample_residual_distribution_into` reuses `residual_buf`
/// - Leviathan rejection sampling reuses `p_distributions_flat`
pub struct SpeculativeContext {
    /// Pre-allocated forward pass buffers (embedding, attention, MLP, logits).
    pub ctx: ForwardContext,
    /// Pre-allocated KV cache for draft model.
    pub cache: MultiLayerKVCache,
    /// Flat marginals buffer: `[draft_lookahead * vocab_size]`.
    /// Each step's marginal occupies `[step * vocab_size..(step+1) * vocab_size]`.
    pub marginals_flat: Vec<f32>,
    /// Temp probs buffer: `[vocab_size]` for logits→softmax in-place.
    pub probs_buf: Vec<f32>,
    /// Pre-allocated sampled tokens: `[draft_lookahead]`.
    pub sampled_tokens: Vec<usize>,
    /// Pre-allocated accepted tokens: `[draft_lookahead + 1]`.
    pub accepted_buf: Vec<usize>,
    /// Pre-allocated path buffer: `[draft_lookahead + 1]`.
    pub path_buf: Vec<usize>,
    /// Residual distribution scratch: `[vocab_size]` for `sample_residual_distribution_into`.
    pub residual_buf: Vec<f32>,
    /// Flat p-distributions buffer for Leviathan: `[(draft_lookahead + 1) * vocab_size]`.
    pub p_distributions_flat: Vec<f32>,
    /// Number of steps populated in last operation (for slicing).
    pub steps_populated: usize,
    /// SDE noise injection config for DDTree expansion (ELF Plan 079).
    pub sde_config: SdeConfig,
    /// Reusable target-cache snapshot scratch for `snapshot_into` (zero-alloc
    /// rollback path in `speculative_step_rollback_with*`). Hoisted out of the
    /// per-step loop so the per-layer `key`/`value` Vecs are allocated once
    /// and reused across all speculation steps in the AR decode loop.
    pub target_snap: KVSnapshot,
}

impl SpeculativeContext {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &Config) -> Self {
        let vocab_size = config.vocab_size;
        let draft_lookahead = config.draft_lookahead;

        Self {
            ctx: ForwardContext::new(config),
            cache: MultiLayerKVCache::new(config),
            marginals_flat: vec![0.0f32; draft_lookahead * vocab_size],
            probs_buf: vec![0.0f32; vocab_size],
            sampled_tokens: vec![0usize; draft_lookahead],
            accepted_buf: vec![0usize; draft_lookahead + 1],
            path_buf: vec![0usize; draft_lookahead + 1],
            residual_buf: vec![0.0f32; vocab_size],
            p_distributions_flat: vec![0.0f32; (draft_lookahead + 1) * vocab_size],
            steps_populated: 0,
            sde_config: SdeConfig::default(),
            target_snap: KVSnapshot::default(),
        }
    }

    /// Reset transient state between decode steps.
    /// Clears lengths to zero; buffers retain capacity for reuse.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.steps_populated = 0;
    }

    /// Get marginal slice for a specific step.
    /// Returns empty slice if step is out of range.
    pub fn marginal_slice(&self, step: usize, vocab_size: usize) -> &[f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.marginals_flat.len() && step < self.steps_populated {
            &self.marginals_flat[start..end]
        } else {
            &[]
        }
    }

    /// Get mutable marginal slice for a specific step.
    pub fn marginal_slice_mut(&mut self, step: usize, vocab_size: usize) -> &mut [f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.marginals_flat.len() {
            &mut self.marginals_flat[start..end]
        } else {
            &mut []
        }
    }

    /// Get populated marginals as slice-of-slices (borrowed view).
    /// Returns a Vec of borrowed slices for compatibility with existing APIs.
    /// Prefer [`marginals_into`] for hot paths (zero-alloc).
    pub fn marginals_view(&self, vocab_size: usize) -> Vec<&[f32]> {
        (0..self.steps_populated)
            .map(|step| self.marginal_slice(step, vocab_size))
            .collect()
    }

    /// Zero-alloc marginals view: writes borrowed slices into caller-provided buffer.
    ///
    /// Returns the populated portion of `buf` as `&[&[f32]]`.
    /// `buf` must be at least `steps_populated` long (bounded by `draft_lookahead`, typically ≤64).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut buf: [&[f32]; 64] = [&[]; 64];
    /// let view = sctx.marginals_into(&mut buf, vocab_size);
    /// tree_builder.build(view, config, &NoPruner, false);
    /// ```
    pub fn marginals_into<'s, 'a>(
        &'s self,
        buf: &'a mut [&'s [f32]],
        vocab_size: usize,
    ) -> &'a [&'s [f32]] {
        let count = self.steps_populated.min(buf.len());
        for (i, slot) in buf.iter_mut().enumerate().take(count) {
            *slot = self.marginal_slice(i, vocab_size);
        }
        &buf[..count]
    }

    /// Get populated sampled tokens.
    pub fn sampled_tokens(&self) -> &[usize] {
        &self.sampled_tokens[..self.steps_populated]
    }

    /// Get p-distribution slice for Leviathan step.
    pub fn p_dist_slice(&self, step: usize, vocab_size: usize) -> &[f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.p_distributions_flat.len() {
            &self.p_distributions_flat[start..end]
        } else {
            &[]
        }
    }

    /// Get mutable p-distribution slice for Leviathan step.
    pub fn p_dist_slice_mut(&mut self, step: usize, vocab_size: usize) -> &mut [f32] {
        let start = step * vocab_size;
        let end = start + vocab_size;
        if end <= self.p_distributions_flat.len() {
            &mut self.p_distributions_flat[start..end]
        } else {
            &mut []
        }
    }
}
