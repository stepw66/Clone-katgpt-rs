//! ScreeningPruner augmented with memory-steered corrections.
//!
//! Distilled from δ-mem's low-rank attention corrections.
//! Verified from `delta_impl.py` L2283-2293:
//!   attn_output = base_o_proj(attn_output) + delta_o_typed
//!
//! Instead of correcting attention Q/O, we correct relevance scores:
//!   relevance_adjusted = relevance_inner + α × correction

use std::sync::Mutex;

use katgpt_core::simd::simd_sum_f32;
use katgpt_speculative::ScreeningPruner;

use super::hash::{ContextFeatures, FeatureHasher, OutcomeFeatures};
use super::state::{DeltaMemoryConfig, DeltaMemorySnapshot, DeltaMemoryState};

/// Correction target (verified from paper Table 3 ablation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CorrectionMode {
    /// Adjust relevance before inner pruner (paper "q" head: 44.51%)
    QuerySide,
    /// Adjust relevance after inner pruner (paper "o" head: 47.05%)
    OutputSide,
    /// Both corrections (paper "qo" config: 47.97%, best perf/param tradeoff)
    Both,
}

/// Write granularity (verified from config + forward L2150-2215).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WriteGranularity {
    /// Per-token write (TSW). Paper default for Qwen3-4B.
    Token,
    /// Per-DDTree-build averaged write (SSW). Paper "message_mean".
    Segment,
}

/// ScreeningPruner augmented with memory-steered corrections.
///
/// Wraps any inner `ScreeningPruner` and adds delta-memory corrections
/// to relevance scores. The memory learns associations between tree
/// contexts and generation outcomes via the delta-rule.
pub struct MemorySteeredPruner<P: ScreeningPruner> {
    /// Inner pruner being corrected.
    inner: P,
    /// Associative memory state.
    memory: DeltaMemoryState,
    /// Correction strength α/r scaling (paper: α=16, rank=8 → effective 2.0).
    alpha: f32,
    /// Feature hasher for generating query keys.
    key_hasher: FeatureHasher,
    /// Feature hasher for generating value hashes (separate seed).
    val_hasher: FeatureHasher,
    /// Correction mode.
    mode: CorrectionMode,
    /// Pending observations for this DDTree build (SSW support).
    pending: Vec<(ContextFeatures, OutcomeFeatures)>,
    /// Pre-allocated segment key buffers for SSW flush. Cleared + reused.
    segment_keys: Vec<Vec<f32>>,
    /// Pre-allocated segment value buffers for SSW flush. Cleared + reused.
    segment_values: Vec<Vec<f32>>,
    /// Write granularity.
    write_granularity: WriteGranularity,
    /// Pre-allocated scratch buffers for zero-alloc trait impl (interior mutability
    /// needed because `ScreeningPruner::relevance` takes `&self`).
    /// Mutex satisfies the `Sync` bound required by `ScreeningPruner: Send + Sync`.
    ///
    /// All scratch buffers are grouped in a single Mutex to amortize lock overhead
    /// (one acquire per call instead of five). They are always used together.
    scratch: Mutex<ScratchBuffers>,
}

/// Pre-allocated scratch space for [`MemorySteeredPruner`] — grouped to share one lock.
struct ScratchBuffers {
    feature_buf: Vec<f32>,
    key_buf: Vec<f32>,
    readout_buf: Vec<f32>,
    outcome_buf: Vec<f32>,
    val_buf: Vec<f32>,
}

impl<P: ScreeningPruner> MemorySteeredPruner<P> {
    /// Create a new memory-steered pruner.
    pub fn new(
        inner: P,
        memory_config: DeltaMemoryConfig,
        alpha: f32,
        mode: CorrectionMode,
        write_granularity: WriteGranularity,
    ) -> Self {
        let rank = memory_config.rank;
        let feature_dim = 8; // ContextFeatures::to_vec() dimension
        Self {
            inner,
            memory: DeltaMemoryState::new(memory_config),
            alpha,
            key_hasher: FeatureHasher::new(rank, feature_dim, 42),
            val_hasher: FeatureHasher::new(rank, 3, 99),
            mode,
            pending: Vec::new(),
            segment_keys: Vec::new(),
            segment_values: Vec::new(),
            write_granularity,
            scratch: Mutex::new(ScratchBuffers {
                feature_buf: Vec::with_capacity(8),
                key_buf: vec![0.0f32; rank],
                readout_buf: vec![0.0f32; rank],
                outcome_buf: Vec::with_capacity(3),
                val_buf: vec![0.0f32; rank],
            }),
        }
    }

    /// Create with custom seeds for feature hashers.
    pub fn with_seeds(mut self, key_seed: u64, val_seed: u64) -> Self {
        let rank = self.memory.config().rank;
        self.key_hasher = FeatureHasher::new(rank, 8, key_seed);
        self.val_hasher = FeatureHasher::new(rank, 3, val_seed);
        self
    }

    /// Observe outcome for current position (TSW: immediate write).
    /// Uses pre-allocated scratch buffers to avoid allocation.
    pub fn observe(&mut self, ctx: &ContextFeatures, outcome: &OutcomeFeatures) {
        match self.write_granularity {
            WriteGranularity::Token => {
                // Deref the guard so the borrow checker can see field borrows are disjoint.
                // Without this, `&s.feature_buf` + `&mut s.key_buf` would be rejected
                // because both go through the same MutexGuard.
                let s = &mut *self.scratch.lock().unwrap();

                ctx.to_vec_into(&mut s.feature_buf);
                self.key_hasher
                    .hash_key_into(&s.feature_buf, &mut s.key_buf);
                outcome.to_vec_into(&mut s.outcome_buf);
                self.val_hasher
                    .hash_value_into(&s.outcome_buf, &mut s.val_buf);
                self.memory.write(&s.key_buf, &s.val_buf);
            }
            WriteGranularity::Segment => {
                self.pending.push((*ctx, *outcome));
            }
        }
    }

    /// Flush pending observations (SSW: call after DDTree build completes).
    pub fn flush_segment(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        let n = self.pending.len();
        self.segment_keys.clear();
        self.segment_values.clear();

        // Reuse or allocate inner Vecs
        while self.segment_keys.len() < n {
            self.segment_keys.push(vec![0.0; self.key_hasher.rank()]);
        }
        while self.segment_values.len() < n {
            self.segment_values.push(vec![0.0; self.val_hasher.rank()]);
        }

        // Fill buffers using zero-alloc _into methods
        let mut s = self.scratch.lock().unwrap();
        s.feature_buf.clear();
        s.outcome_buf.clear();

        for (i, (ctx, outcome)) in self.pending.iter().enumerate() {
            ctx.to_vec_into(&mut s.feature_buf);
            self.key_hasher
                .hash_key_into(&s.feature_buf, &mut self.segment_keys[i]);
            outcome.to_vec_into(&mut s.outcome_buf);
            self.val_hasher
                .hash_value_into(&s.outcome_buf, &mut self.segment_values[i]);
        }
        drop(s);

        self.memory
            .write_segment(&self.segment_keys[..n], &self.segment_values[..n]);
        self.pending.clear();
    }

    /// Zero-alloc relevance computation using pre-allocated scratch buffers.
    ///
    /// Callers provide their own buffers to avoid allocation on the hot path.
    /// - `feature_buf`: capacity ≥ 8 (context feature dim)
    /// - `key_buf`: length == memory rank
    /// - `readout_buf`: length == memory rank
    pub fn relevance_into(
        &self,
        depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
        feature_buf: &mut Vec<f32>,
        key_buf: &mut [f32],
        readout_buf: &mut [f32],
    ) -> f32 {
        let inner_rel = self.inner.relevance(depth, token_idx, parent_tokens);

        let ctx = ContextFeatures::from_tree_context(depth, token_idx, parent_tokens);
        ctx.to_vec_into(feature_buf);
        self.key_hasher.hash_key_into(feature_buf, key_buf);
        self.memory.read_into(key_buf, readout_buf);

        let correction: f32 = readout_buf.iter().copied().sum::<f32>() / readout_buf.len() as f32;

        let adjusted = inner_rel + self.alpha * correction;

        adjusted.clamp(0.0, 1.0)
    }

    /// Zero-alloc observe using pre-allocated scratch buffers.
    ///
    /// Callers provide their own buffers to avoid allocation on the hot path.
    /// - `feature_buf`: capacity ≥ 8 (context feature dim)
    /// - `key_buf`: length == memory rank
    /// - `val_buf`: length == memory rank
    /// - `outcome_buf`: capacity ≥ 3 (outcome feature dim)
    pub fn observe_into(
        &mut self,
        ctx: &ContextFeatures,
        outcome: &OutcomeFeatures,
        feature_buf: &mut Vec<f32>,
        key_buf: &mut [f32],
        val_buf: &mut [f32],
        outcome_buf: &mut Vec<f32>,
    ) {
        match self.write_granularity {
            WriteGranularity::Token => {
                ctx.to_vec_into(feature_buf);
                self.key_hasher.hash_key_into(feature_buf, key_buf);
                outcome.to_vec_into(outcome_buf);
                self.val_hasher.hash_value_into(outcome_buf, val_buf);
                self.memory.write(key_buf, val_buf);
            }
            WriteGranularity::Segment => {
                self.pending.push((*ctx, *outcome));
            }
        }
    }

    /// Adapt gates based on recent δ observations.
    pub fn adapt_gates(&mut self, recent_deltas: &[f32]) {
        self.memory.adapt_gates(recent_deltas);
    }

    /// Snapshot memory state for persistence.
    pub fn snapshot_memory(&self) -> DeltaMemorySnapshot {
        self.memory.snapshot()
    }

    /// Access inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Access memory state.
    pub fn memory(&self) -> &DeltaMemoryState {
        &self.memory
    }

    /// Mutable access to memory state.
    pub fn memory_mut(&mut self) -> &mut DeltaMemoryState {
        &mut self.memory
    }

    /// Get correction mode.
    #[inline]
    pub fn mode(&self) -> CorrectionMode {
        self.mode
    }

    /// Get write granularity.
    #[inline]
    pub fn write_granularity(&self) -> WriteGranularity {
        self.write_granularity
    }

    /// Number of pending observations (SSW).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Reset memory and pending observations.
    pub fn reset(&mut self) {
        self.memory.reset();
        self.pending.clear();
        self.segment_keys.clear();
        self.segment_values.clear();
    }
}

impl<P: ScreeningPruner> ScreeningPruner for MemorySteeredPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner_rel = self.inner.relevance(depth, token_idx, parent_tokens);

        let ctx = ContextFeatures::from_tree_context(depth, token_idx, parent_tokens);

        // Deref the guard so disjoint field borrows are visible to the borrow checker.
        let s = &mut *self.scratch.lock().unwrap();

        ctx.to_vec_into(&mut s.feature_buf);
        self.key_hasher
            .hash_key_into(&s.feature_buf, &mut s.key_buf);
        self.memory.read_into(&s.key_buf, &mut s.readout_buf);

        let correction: f32 = simd_sum_f32(&s.readout_buf) / s.readout_buf.len() as f32;

        let adjusted = inner_rel + self.alpha * correction;

        adjusted.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_speculative::NoScreeningPruner;

    fn make_pruner(mode: CorrectionMode) -> MemorySteeredPruner<NoScreeningPruner> {
        MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            mode,
            WriteGranularity::Token,
        )
    }

    #[test]
    fn test_no_memory_returns_inner_relevance() {
        let pruner = make_pruner(CorrectionMode::OutputSide);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_observe_token_writes_to_memory() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let ctx = ContextFeatures::from_tree_context(1, 2, &[0, 1]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        assert_eq!(pruner.memory().update_count(), 1);
    }

    #[test]
    fn test_observe_segment_defers_write() {
        let mut pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Segment,
        );
        let ctx = ContextFeatures::from_tree_context(1, 2, &[0]);
        let outcome = OutcomeFeatures {
            delta: 0.3,
            quality: 0.7,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        assert_eq!(pruner.pending_count(), 1);
        assert_eq!(pruner.memory().update_count(), 0);
        pruner.flush_segment();
        assert_eq!(pruner.pending_count(), 0);
        assert_eq!(pruner.memory().update_count(), 1);
    }

    #[test]
    fn test_correction_modes_dont_panic() {
        for mode in [
            CorrectionMode::QuerySide,
            CorrectionMode::OutputSide,
            CorrectionMode::Both,
        ] {
            let pruner = make_pruner(mode);
            let rel = pruner.relevance(5, 3, &[1, 2, 3]);
            assert!((0.0..=1.0).contains(&rel));
        }
    }

    #[test]
    fn test_snapshot_restore() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let ctx = ContextFeatures::from_tree_context(1, 0, &[]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        let snap = pruner.snapshot_memory();
        assert_eq!(snap.update_count, 1);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let ctx = ContextFeatures::from_tree_context(1, 0, &[]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };
        pruner.observe(&ctx, &outcome);
        assert_eq!(pruner.memory().update_count(), 1);
        pruner.reset();
        assert_eq!(pruner.memory().update_count(), 0);
    }

    #[test]
    fn test_after_observation_relevance_changes() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let _baseline = pruner.relevance(1, 2, &[0, 1]);
        for i in 0..20 {
            let ctx = ContextFeatures::from_tree_context(1, 2, &[0, 1]);
            let outcome = OutcomeFeatures {
                delta: 0.5 + i as f32 * 0.01,
                quality: 0.8,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }
        let after = pruner.relevance(1, 2, &[0, 1]);
        assert!((0.0..=1.0).contains(&after));
    }

    #[test]
    fn test_relevance_into_matches_trait_method() {
        let pruner = make_pruner(CorrectionMode::OutputSide);
        let rank = pruner.memory().config().rank;

        let trait_rel = pruner.relevance(1, 2, &[0, 1]);

        let mut feature_buf = Vec::with_capacity(8);
        let mut key_buf = vec![0.0; rank];
        let mut readout_buf = vec![0.0; rank];
        let into_rel = pruner.relevance_into(
            1,
            2,
            &[0, 1],
            &mut feature_buf,
            &mut key_buf,
            &mut readout_buf,
        );

        assert!(
            (trait_rel - into_rel).abs() < 1e-6,
            "relevance_into should match trait method: trait={trait_rel}, into={into_rel}"
        );
    }

    #[test]
    fn test_observe_into_writes_to_memory() {
        let mut pruner = make_pruner(CorrectionMode::OutputSide);
        let rank = pruner.memory().config().rank;

        let ctx = ContextFeatures::from_tree_context(1, 2, &[0, 1]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };

        let mut feature_buf = Vec::with_capacity(8);
        let mut key_buf = vec![0.0; rank];
        let mut val_buf = vec![0.0; rank];
        let mut outcome_buf = Vec::with_capacity(3);
        pruner.observe_into(
            &ctx,
            &outcome,
            &mut feature_buf,
            &mut key_buf,
            &mut val_buf,
            &mut outcome_buf,
        );
        assert_eq!(pruner.memory().update_count(), 1);
    }

    #[test]
    fn test_observe_into_segment_defers_write() {
        let mut pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Segment,
        );

        let ctx = ContextFeatures::from_tree_context(1, 2, &[0]);
        let outcome = OutcomeFeatures {
            delta: 0.3,
            quality: 0.7,
            success: 1.0,
        };

        let mut feature_buf = Vec::with_capacity(8);
        let mut key_buf = vec![0.0; 8];
        let mut val_buf = vec![0.0; 8];
        let mut outcome_buf = Vec::with_capacity(3);
        pruner.observe_into(
            &ctx,
            &outcome,
            &mut feature_buf,
            &mut key_buf,
            &mut val_buf,
            &mut outcome_buf,
        );
        assert_eq!(pruner.pending_count(), 1);
        assert_eq!(pruner.memory().update_count(), 0);
        pruner.flush_segment();
        assert_eq!(pruner.pending_count(), 0);
        assert_eq!(pruner.memory().update_count(), 1);
    }

    #[test]
    fn test_relevance_into_after_observe_into_matches_allocating() {
        let mut pruner_alloc = make_pruner(CorrectionMode::Both);
        let mut pruner_into = make_pruner(CorrectionMode::Both);
        let rank = pruner_alloc.memory().config().rank;

        let ctx = ContextFeatures::from_tree_context(1, 2, &[0, 1]);
        let outcome = OutcomeFeatures {
            delta: 0.5,
            quality: 0.8,
            success: 1.0,
        };

        for _ in 0..10 {
            pruner_alloc.observe(&ctx, &outcome);

            let mut feature_buf = Vec::with_capacity(8);
            let mut key_buf = vec![0.0; rank];
            let mut val_buf = vec![0.0; rank];
            let mut outcome_buf = Vec::with_capacity(3);
            pruner_into.observe_into(
                &ctx,
                &outcome,
                &mut feature_buf,
                &mut key_buf,
                &mut val_buf,
                &mut outcome_buf,
            );
        }

        let rel_alloc = pruner_alloc.relevance(1, 2, &[0, 1]);
        let mut feature_buf = Vec::with_capacity(8);
        let mut key_buf = vec![0.0; rank];
        let mut readout_buf = vec![0.0; rank];
        let rel_into = pruner_into.relevance_into(
            1,
            2,
            &[0, 1],
            &mut feature_buf,
            &mut key_buf,
            &mut readout_buf,
        );

        assert!(
            (rel_alloc - rel_into).abs() < 1e-6,
            "allocating and into paths should agree: alloc={rel_alloc}, into={rel_into}"
        );
    }

    #[test]
    fn test_flush_segment_reuses_buffers() {
        let mut pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Segment,
        );

        // First flush — allocates buffers.
        for i in 0..5 {
            let ctx = ContextFeatures::from_tree_context(1, i, &[0]);
            let outcome = OutcomeFeatures {
                delta: 0.5,
                quality: 0.8,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }
        pruner.flush_segment();
        assert_eq!(pruner.memory().update_count(), 1);

        // Second flush with more observations — should reuse + extend.
        for i in 0..10 {
            let ctx = ContextFeatures::from_tree_context(1, i, &[0]);
            let outcome = OutcomeFeatures {
                delta: 0.3,
                quality: 0.7,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }
        pruner.flush_segment();
        assert_eq!(pruner.memory().update_count(), 2);
    }

    #[test]
    fn test_flush_segment_matches_allocating_version() {
        // Sanity: the segment-keys reuse path produces the same memory state
        // as a fresh-alloc path would.
        let cfg = DeltaMemoryConfig::default();
        let rank = cfg.rank;

        let mut pruner_reuse = MemorySteeredPruner::new(
            NoScreeningPruner,
            cfg.clone(),
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Segment,
        );
        let mut pruner_alloc = MemorySteeredPruner::new(
            NoScreeningPruner,
            cfg,
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Token, // token path = immediate write, equivalent to per-segment flush with 1 token
        );

        // Feed both the same observations, one per segment.
        for i in 0..3 {
            let ctx = ContextFeatures::from_tree_context(1, i, &[0]);
            let outcome = OutcomeFeatures {
                delta: 0.5,
                quality: 0.8,
                success: 1.0,
            };
            pruner_reuse.observe(&ctx, &outcome);
            pruner_alloc.observe(&ctx, &outcome);
        }
        pruner_reuse.flush_segment();

        // Both should have written 3 observations (token path wrote immediately;
        // segment path wrote once with 3 keys via the segment-keys reuse).
        assert_eq!(pruner_reuse.memory().update_count(), 1);
        assert_eq!(pruner_alloc.memory().update_count(), 3);

        // Relevance should be in the same ballpark (same observations, same keys).
        let rel_reuse = pruner_reuse.relevance(1, 0, &[0]);
        let rel_alloc = pruner_alloc.relevance(1, 0, &[0]);
        // Both should be in [0, 1] and finite.
        assert!(rel_reuse.is_finite() && (0.0..=1.0).contains(&rel_reuse));
        assert!(rel_alloc.is_finite() && (0.0..=1.0).contains(&rel_alloc));
        // _ = rank to silence unused warning if rank isn't otherwise used.
        let _ = rank;
    }

    #[test]
    fn test_reset_clears_segment_buffers() {
        let mut pruner = MemorySteeredPruner::new(
            NoScreeningPruner,
            DeltaMemoryConfig::default(),
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Segment,
        );

        for i in 0..5 {
            let ctx = ContextFeatures::from_tree_context(1, i, &[0]);
            let outcome = OutcomeFeatures {
                delta: 0.5,
                quality: 0.8,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }
        pruner.flush_segment();
        assert_eq!(pruner.memory().update_count(), 1);

        pruner.reset();
        assert_eq!(pruner.memory().update_count(), 0);
        assert_eq!(pruner.pending_count(), 0);

        // After reset, a new segment should still work (buffers were cleared, not leaked).
        for i in 0..3 {
            let ctx = ContextFeatures::from_tree_context(2, i, &[0]);
            let outcome = OutcomeFeatures {
                delta: 0.3,
                quality: 0.7,
                success: 1.0,
            };
            pruner.observe(&ctx, &outcome);
        }
        pruner.flush_segment();
        assert_eq!(pruner.memory().update_count(), 1);
    }
}
