//! GOAT proofs for Epiplexity Structural Information Scoring (Plan 130).
//!
//! **Source:** Epiplexity paper (arXiv:2601.03220): Structural information
//! extractable by computationally bounded observers, measured as area under
//! loss curve above final loss.
//!
//! **GOAT targets (Plan 130):**
//! - P1: EpiplexityEstimator — constant→S≈0, random→S≈0, structured→S>0
//! - P2: ScreeningPruner — α=0 preserves inner, α=1 uses epiplexity, blend correct
//! - P3: LossCurveTracker — batch/epoch tracking, prequential estimate, ring buffer
//! - P4: FactorizationScorer — forward/reverse scoring, gap direction, adaptive
//!
//! These unit tests verify the core algorithms are correct and the GOAT
//! performance targets are met.

#[cfg(feature = "epiplexity_scoring")]
mod tests {
    use katgpt_rs::pruners::epiplexity::{
        EpiplexityEstimator, EpiplexityScreeningPruner, EpiplexityWeight, FactorizationOrder,
        FactorizationScorer, LossCurveTracker, PerPositionLossTracker, TimeBoundedEntropy,
    };
    use katgpt_rs::speculative::types::ScreeningPruner;

    // ── Helpers ─────────────────────────────────────────────────

    /// Trivial pruner: always returns 1.0 relevance.
    struct UnitPruner;

    impl ScreeningPruner for UnitPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    /// Trivial pruner: always returns a fixed value.
    struct FixedPruner {
        value: f32,
    }

    impl ScreeningPruner for FixedPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.value
        }
    }

    /// Generate a decreasing loss curve (structured data pattern).
    fn structured_losses(n: usize, start: f32, end: f32) -> Vec<f32> {
        (0..n)
            .map(|i| start - (start - end) * (i as f32) / ((n - 1).max(1) as f32))
            .collect()
    }

    /// Generate a constant loss curve.
    fn constant_losses(n: usize, value: f32) -> Vec<f32> {
        vec![value; n]
    }

    /// Generate random-ish loss curve using simple hash (no external rand dep).
    fn pseudo_random_losses(n: usize, seed: u64, center: f32, spread: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let hash = (seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((i as u64).wrapping_mul(1442695040888963407)))
                    >> 33;
                let normalized = (hash as f32) / (u32::MAX as f32); // [0, 1)
                center + (normalized - 0.5) * 2.0 * spread
            })
            .collect()
    }

    // ════════════════════════════════════════════════════════════
    // P1: EpiplexityEstimator Core
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_p1_constant_data_epiplexity_near_zero() {
        let mut est = EpiplexityEstimator::new(100);
        let final_loss = 2.5;
        for _ in 0..50 {
            est.record_step(final_loss);
        }
        let s = est.compute_epiplexity(final_loss);
        assert!(s < 0.01, "P1: constant data → S≈0, got {s}");
    }

    #[test]
    fn test_p1_random_data_epiplexity_bounded() {
        let mut est = EpiplexityEstimator::new(1000);
        let final_loss = 3.0;
        let losses = pseudo_random_losses(500, 42, final_loss, 1.0);
        for loss in &losses {
            est.record_step(*loss);
        }
        let s = est.compute_epiplexity(final_loss);
        // Per-step contribution should be small (< 1.0)
        let s_per_step = s / (est.len() as f32);
        assert!(
            s_per_step < 1.0,
            "P1: random per-step epiplexity should be small, got {s_per_step}"
        );
        // Total should be bounded — not orders of magnitude larger than structured
        assert!(s < 500.0, "P1: random total should be bounded, got {s}");
    }

    #[test]
    fn test_p1_structured_data_epiplexity_positive() {
        let mut est = EpiplexityEstimator::new(100);
        let final_loss = 1.0;
        let losses = structured_losses(50, 5.0, 1.1);
        for loss in &losses {
            est.record_step(*loss);
        }
        let s = est.compute_epiplexity(final_loss);
        assert!(s > 1.0, "P1: structured data → S>1.0, got {s}");
    }

    #[test]
    fn test_p1_structured_greater_than_random() {
        // Structured data should have higher S_T than random data
        let mut est_structured = EpiplexityEstimator::new(100);
        let mut est_random = EpiplexityEstimator::new(100);

        let final_loss = 2.0;

        // Structured: large initial loss, smooth decrease
        for loss in structured_losses(50, 6.0, 2.1) {
            est_structured.record_step(loss);
        }

        // Random: centered around final_loss
        for loss in pseudo_random_losses(50, 99, final_loss, 0.5) {
            est_random.record_step(loss);
        }

        let s_structured = est_structured.compute_epiplexity(final_loss);
        let s_random = est_random.compute_epiplexity(final_loss);
        assert!(
            s_structured > s_random,
            "P1: structured S ({s_structured}) should be > random S ({s_random})"
        );
    }

    #[test]
    fn test_p1_more_structure_higher_epiplexity() {
        let final_loss = 1.0;

        let mut est_low = EpiplexityEstimator::new(30);
        let mut est_high = EpiplexityEstimator::new(30);

        for loss in structured_losses(30, 3.0, 1.1) {
            est_low.record_step(loss);
        }
        for loss in structured_losses(30, 8.0, 1.1) {
            est_high.record_step(loss);
        }

        let s_low = est_low.compute_epiplexity(final_loss);
        let s_high = est_high.compute_epiplexity(final_loss);
        assert!(
            s_high > s_low,
            "P1: more structure → higher S: {s_high} should be > {s_low}"
        );
    }

    #[test]
    fn test_p1_ring_buffer_bounded() {
        let mut est = EpiplexityEstimator::new(5);
        for i in 0..20 {
            est.record_step(i as f32);
        }
        assert_eq!(est.len(), 5, "ring buffer should cap at capacity");
        // Only last 5: [15, 16, 17, 18, 19]
        let s = est.compute_epiplexity(0.0);
        let expected: f32 = 15.0 + 16.0 + 17.0 + 18.0 + 19.0;
        assert!((s - expected).abs() < 0.01, "expected {expected}, got {s}");
    }

    #[test]
    fn test_p1_per_sample_epiplexity() {
        let mut est = EpiplexityEstimator::new(10);
        for loss in structured_losses(10, 4.0, 1.5) {
            est.record_step(loss);
        }
        let final_losses = vec![1.0, 2.0, 5.0];
        let per_sample = est.compute_per_sample(&final_losses);

        // Lower final loss → more excess → higher epiplexity
        assert!(
            per_sample[0] > per_sample[1],
            "P1: lower final → higher S: {} should be > {}",
            per_sample[0],
            per_sample[1]
        );
        // final=5.0 exceeds all step losses (max ~4.0) → S≈0
        assert!(
            per_sample[2] < 0.01,
            "P1: final above all steps → S≈0, got {}",
            per_sample[2]
        );
    }

    #[test]
    fn test_p1_time_bounded_entropy() {
        let tbe = TimeBoundedEntropy::new(10);
        let h = tbe.compute_entropy(2.5, 100);
        assert!(
            (h - 250.0).abs() < 0.01,
            "P1: entropy = final_loss × n_tokens = 2.5×100 = 250.0, got {h}"
        );
    }

    #[test]
    fn test_p1_structural_fraction_bounded() {
        let mut tbe = TimeBoundedEntropy::new(100);
        for loss in structured_losses(50, 5.0, 1.0) {
            tbe.record_step(loss);
        }
        let frac = tbe.structural_fraction(1.0, 100);
        assert!(
            (0.0..=1.0).contains(&frac),
            "P1: structural fraction ∈ [0,1], got {frac}"
        );
    }

    #[test]
    fn test_p1_clear_resets() {
        let mut est = EpiplexityEstimator::new(10);
        est.record_step(5.0);
        est.record_step(3.0);
        assert_eq!(est.len(), 2);
        est.clear();
        assert!(est.is_empty());
        assert_eq!(est.compute_epiplexity(0.0), 0.0);
    }

    // ════════════════════════════════════════════════════════════
    // P2: EpiplexityScreeningPruner
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_p2_alpha_zero_preserves_inner_unit() {
        let pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 0.0, EpiplexityWeight::Uniform, 100);
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 1.0).abs() < 1e-6,
            "P2: α=0 should preserve UnitPruner (1.0), got {rel}"
        );
    }

    #[test]
    fn test_p2_alpha_zero_preserves_inner_fixed() {
        let pruner = EpiplexityScreeningPruner::new(
            FixedPruner { value: 0.3 },
            0.0,
            EpiplexityWeight::Uniform,
            100,
        );
        let rel = pruner.relevance(5, 10, &[1, 2, 3, 4, 5]);
        assert!(
            (rel - 0.3).abs() < 1e-6,
            "P2: α=0 should preserve FixedPruner (0.3), got {rel}"
        );
    }

    #[test]
    fn test_p2_alpha_one_uses_epiplexity_uniform() {
        let pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::Uniform, 100);
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 1.0).abs() < 1e-6,
            "P2: α=1 Uniform returns 1.0, got {rel}"
        );
    }

    #[test]
    fn test_p2_alpha_one_loss_drop_mode() {
        let mut pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::LossDrop, 100);
        pruner.set_position_drops(vec![0.0, 5.0]);

        let rel0 = pruner.relevance(0, 0, &[]);
        let rel1 = pruner.relevance(1, 0, &[]);

        // sigmoid(0) = 0.5, sigmoid(5) ≈ 0.993
        assert!(
            (rel0 - 0.5).abs() < 0.01,
            "P2: drop=0 → sigmoid(0)≈0.5, got {rel0}"
        );
        assert!(rel1 > 0.99, "P2: drop=5 → sigmoid(5)>0.99, got {rel1}");
    }

    #[test]
    fn test_p2_alpha_one_cumulative_area_empty() {
        let pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::CumulativeArea, 100);
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 0.0).abs() < 1e-6,
            "P2: empty history → signal=0.0, got {rel}"
        );
    }

    #[test]
    fn test_p2_alpha_one_cumulative_area_with_structure() {
        let mut pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::CumulativeArea, 100);
        for loss in structured_losses(20, 8.0, 1.0) {
            pruner.record_step(loss);
        }
        pruner.set_final_loss(1.0);
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            rel > 0.5,
            "P2: cumulative area with structure → signal>0.5, got {rel}"
        );
    }

    #[test]
    fn test_p2_blending_interpolation() {
        let inner_val = 0.4;
        let pruner = EpiplexityScreeningPruner::new(
            FixedPruner { value: inner_val },
            0.3,
            EpiplexityWeight::Uniform,
            100,
        );
        let rel = pruner.relevance(0, 0, &[]);
        let expected = inner_val * 0.7 + 1.0 * 0.3; // 0.28 + 0.30 = 0.58
        assert!(
            (rel - expected).abs() < 1e-6,
            "P2: blend = {inner_val}×0.7 + 1.0×0.3 = {expected}, got {rel}"
        );
    }

    #[test]
    fn test_p2_alpha_setter_clamps() {
        let mut pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 0.5, EpiplexityWeight::Uniform, 10);
        pruner.set_alpha(-5.0);
        assert!((pruner.alpha() - 0.0).abs() < 1e-6, "clamped to 0.0");
        pruner.set_alpha(100.0);
        assert!((pruner.alpha() - 1.0).abs() < 1e-6, "clamped to 1.0");
    }

    #[test]
    fn test_p2_inner_accessor() {
        let pruner = EpiplexityScreeningPruner::new(UnitPruner, 0.5, EpiplexityWeight::Uniform, 10);
        // Just verify it compiles and returns reference
        let _inner: &UnitPruner = pruner.inner();
    }

    // ════════════════════════════════════════════════════════════
    // P3: LossCurveTracker
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_p3_batch_tracking_counts() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        tracker.on_batch_end(1, 4.0);
        tracker.on_batch_end(2, 3.0);
        assert_eq!(tracker.batch_count(), 3);
        assert!((tracker.latest_batch_loss().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_p3_epoch_tracking_counts() {
        let mut tracker = LossCurveTracker::new(100, 5);
        tracker.on_epoch_end(0, 4.5);
        tracker.on_epoch_end(1, 3.5);
        assert_eq!(tracker.epoch_count(), 2);
        assert!((tracker.latest_epoch_loss().unwrap() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_p3_prequential_estimate_structured() {
        let mut tracker = LossCurveTracker::new(100, 10);
        for (i, loss) in structured_losses(20, 6.0, 1.0).into_iter().enumerate() {
            tracker.on_batch_end(i, loss);
        }
        let s = tracker.epiplexity_estimate();
        assert!(s > 0.0, "P3: structured batches → prequential S>0, got {s}");
    }

    #[test]
    fn test_p3_prequential_estimate_constant() {
        let mut tracker = LossCurveTracker::new(100, 10);
        for (i, loss) in constant_losses(20, 3.0).into_iter().enumerate() {
            tracker.on_batch_end(i, loss);
        }
        let s = tracker.epiplexity_estimate();
        assert!(s < 0.01, "P3: constant batches → prequential S≈0, got {s}");
    }

    #[test]
    fn test_p3_running_min_updates() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        assert!((tracker.running_min() - 5.0).abs() < 1e-6);
        tracker.on_batch_end(1, 3.0);
        assert!((tracker.running_min() - 3.0).abs() < 1e-6);
        tracker.on_batch_end(2, 4.0); // goes up
        assert!((tracker.running_min() - 3.0).abs() < 1e-6, "min unchanged");
    }

    #[test]
    fn test_p3_epoch_epiplexity() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_epoch_end(0, 5.0);
        tracker.on_epoch_end(1, 3.0);
        tracker.on_epoch_end(2, 2.0);
        let s = tracker.epoch_epiplexity();
        // S = (5-2) + (3-2) + (2-2) = 3 + 1 + 0 = 4.0
        assert!(
            (s - 4.0).abs() < 1e-6,
            "P3: epoch epiplexity = 4.0, got {s}"
        );
    }

    #[test]
    fn test_p3_epoch_epiplexity_empty() {
        let tracker = LossCurveTracker::new(100, 10);
        assert_eq!(tracker.epoch_epiplexity(), 0.0);
    }

    #[test]
    fn test_p3_total_loss_drop() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        tracker.on_batch_end(1, 3.0);
        tracker.on_batch_end(2, 2.0);
        assert!(
            (tracker.total_loss_drop() - 3.0).abs() < 1e-6,
            "P3: drop = 5.0-2.0 = 3.0"
        );
    }

    #[test]
    fn test_p3_batch_ring_buffer_overflow() {
        let mut tracker = LossCurveTracker::new(3, 10);
        for i in 0..10 {
            tracker.on_batch_end(i, i as f32);
        }
        assert_eq!(tracker.batch_count(), 3);
    }

    #[test]
    fn test_p3_epoch_ring_buffer_overflow() {
        let mut tracker = LossCurveTracker::new(100, 3);
        for i in 0..10 {
            tracker.on_epoch_end(i, i as f32);
        }
        assert_eq!(tracker.epoch_count(), 3);
    }

    #[test]
    fn test_p3_clear_resets() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        tracker.on_epoch_end(0, 4.0);
        tracker.clear();
        assert_eq!(tracker.batch_count(), 0);
        assert_eq!(tracker.epoch_count(), 0);
        assert_eq!(tracker.running_min(), 0.0);
    }

    #[test]
    fn test_p3_per_position_basic() {
        let mut tracker = PerPositionLossTracker::new(3, 10);
        tracker.record_step(&[5.0, 4.0, 3.0]);
        tracker.record_step(&[4.0, 3.0, 2.0]);
        tracker.record_step(&[3.0, 2.0, 1.0]);

        let epi = tracker.per_position_epiplexity();
        assert_eq!(epi.len(), 3);
        for (i, &s) in epi.iter().enumerate() {
            assert!(s > 0.0, "P3: position {i} should have S>0, got {s}");
        }
    }

    #[test]
    fn test_p3_per_position_with_final() {
        let mut tracker = PerPositionLossTracker::new(2, 10);
        tracker.record_step(&[3.0, 2.0]);
        tracker.record_step(&[2.0, 1.0]);

        let final_losses = vec![1.0, 0.5];
        let epi = tracker.per_position_epiplexity_with_final(&final_losses);
        assert!(
            (epi[0] - 3.0).abs() < 1e-5,
            "P3: pos0 S=3.0, got {}",
            epi[0]
        );
        assert!(
            (epi[1] - 2.0).abs() < 1e-5,
            "P3: pos1 S=2.0, got {}",
            epi[1]
        );
    }

    #[test]
    fn test_p3_per_position_top_k() {
        let mut tracker = PerPositionLossTracker::new(4, 10);
        // Position 0: high structure (large drop), position 2: constant
        tracker.record_step(&[8.0, 5.0, 3.0, 0.0]);
        tracker.record_step(&[6.0, 4.0, 3.0, 0.0]);
        tracker.record_step(&[2.0, 3.0, 3.0, 0.0]);

        let top2 = tracker.top_k_structural(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0, "P3: position 0 should be most structural");
    }

    #[test]
    fn test_p3_per_position_total() {
        let mut tracker = PerPositionLossTracker::new(2, 10);
        tracker.record_step(&[4.0, 3.0]);
        tracker.record_step(&[2.0, 1.0]);
        let total = tracker.total_epiplexity();
        assert!(total > 0.0, "P3: total should be > 0, got {total}");
    }

    // ════════════════════════════════════════════════════════════
    // P4: FactorizationScorer
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_p4_forward_decreasing_positive() {
        let scorer = FactorizationScorer::new(100);
        let trace = structured_losses(20, 6.0, 1.0);
        let s = scorer.score_forward(&trace);
        assert!(s > 0.0, "P4: forward decreasing → S>0, got {s}");
    }

    #[test]
    fn test_p4_forward_constant_near_zero() {
        let scorer = FactorizationScorer::new(100);
        let trace = constant_losses(20, 3.0);
        let s = scorer.score_forward(&trace);
        assert!(s < 0.01, "P4: forward constant → S≈0, got {s}");
    }

    #[test]
    fn test_p4_forward_empty() {
        let scorer = FactorizationScorer::new(100);
        assert_eq!(scorer.score_forward(&[]), 0.0);
    }

    #[test]
    fn test_p4_reverse_reverses_trace() {
        let scorer = FactorizationScorer::new(100);
        // Increasing trace: forward S≈0, reversed (decreasing) S>0
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.5).collect();
        let fwd = scorer.score_forward(&trace);
        let rev = scorer.score_reverse(&trace);
        assert!(
            rev > fwd,
            "P4: reversed increasing → higher S: rev={rev}, fwd={fwd}"
        );
    }

    #[test]
    fn test_p4_preferred_order_decreasing() {
        let scorer = FactorizationScorer::new(100);
        let trace = structured_losses(20, 5.0, 1.0);
        let order = scorer.preferred_order(&trace);
        assert_eq!(
            order,
            FactorizationOrder::Forward,
            "P4: decreasing trace → forward preferred"
        );
    }

    #[test]
    fn test_p4_preferred_order_increasing() {
        let scorer = FactorizationScorer::new(100);
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.5).collect();
        let order = scorer.preferred_order(&trace);
        assert_eq!(
            order,
            FactorizationOrder::Reverse,
            "P4: increasing trace → reverse preferred"
        );
    }

    #[test]
    fn test_p4_epiplexity_gap_positive_for_increasing() {
        let scorer = FactorizationScorer::new(100);
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.5).collect();
        let gap = scorer.epiplexity_gap(&trace);
        assert!(
            gap > 0.0,
            "P4: increasing trace → gap>0 (reverse better), got {gap}"
        );
    }

    #[test]
    fn test_p4_epiplexity_gap_negative_for_decreasing() {
        let scorer = FactorizationScorer::new(100);
        let trace = structured_losses(10, 5.0, 1.0);
        let gap = scorer.epiplexity_gap(&trace);
        assert!(
            gap < 0.0,
            "P4: decreasing trace → gap<0 (forward better), got {gap}"
        );
    }

    #[test]
    fn test_p4_epiplexity_gap_constant_near_zero() {
        let scorer = FactorizationScorer::new(100);
        let trace = constant_losses(20, 3.0);
        let gap = scorer.epiplexity_gap(&trace);
        assert!(gap.abs() < 0.01, "P4: constant trace → gap≈0, got {gap}");
    }

    #[test]
    fn test_p4_adaptive_takes_max() {
        let scorer = FactorizationScorer::new(100);
        // Increasing: reverse > forward
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.5).collect();
        let adaptive = scorer.score(&trace, FactorizationOrder::Adaptive);
        let reverse = scorer.score_reverse(&trace);
        assert!(
            (adaptive - reverse).abs() < 1e-6,
            "P4: adaptive should pick max (reverse), got adaptive={adaptive}, reverse={reverse}"
        );
    }

    #[test]
    fn test_p4_rank_traces_ordering() {
        let scorer = FactorizationScorer::new(100);
        let high = structured_losses(10, 8.0, 1.0);
        let low = structured_losses(10, 3.0, 1.5);
        let constant = constant_losses(10, 2.0);

        let traces: &[&[f32]] = &[&constant, &low, &high];
        let ranked = scorer.rank_traces(traces, FactorizationOrder::Forward);

        // High structure > low structure > constant
        assert_eq!(ranked[0].0, 2, "P4: highest structure first");
        assert!(ranked[0].1 > ranked[1].1, "P4: score descending");
    }

    #[test]
    fn test_p4_order_preference_counts() {
        let decreasing = structured_losses(10, 5.0, 1.0);
        let increasing: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.5).collect();
        let constant = constant_losses(10, 3.0);

        let traces: &[&[f32]] = &[&decreasing, &increasing, &constant];
        let (fwd, rev) = FactorizationScorer::order_preference_counts(traces, 100);

        // decreasing → forward, increasing → reverse, constant → forward (tie goes forward)
        assert!(fwd >= 1, "P4: at least 1 forward preference, got {fwd}");
        assert!(rev >= 1, "P4: at least 1 reverse preference, got {rev}");
        assert_eq!(fwd + rev, 3, "P4: total = n_traces");
    }

    #[test]
    fn test_p4_factorization_order_default() {
        assert_eq!(FactorizationOrder::default(), FactorizationOrder::Adaptive);
    }

    #[test]
    fn test_p4_factorization_order_display() {
        assert_eq!(format!("{}", FactorizationOrder::Forward), "Forward");
        assert_eq!(format!("{}", FactorizationOrder::Reverse), "Reverse");
        assert_eq!(format!("{}", FactorizationOrder::Adaptive), "Adaptive");
    }
}

// ── T4: SR²AM Context Extension (Plan 130 T4) ──────────────────
//
// GOAT proofs for epiplexity-aware bandit arm selection.
// Feature-gated on `epiplexity_bandit` = epiplexity_scoring + sr2am_configurator.

#[cfg(feature = "epiplexity_bandit")]
mod t4_bandit {
    use katgpt_core::{ConfiguratorContext, PlanningDecision};
    use katgpt_rs::pruners::configurator_bandit::{ConfiguratorBandit, EpiplexityArmHeuristic};

    // ── ConfiguratorContext epiplexity_bin ───────────────────────

    #[test]
    fn test_context_from_entropy_epiplexity() {
        let ctx = ConfiguratorContext::from_entropy_epiplexity(0, 0.35, 0.72);
        assert_eq!(ctx.domain, 0);
        assert_eq!(ctx.entropy_bin, 3); // floor(0.35 * 10) = 3
        assert_eq!(ctx.epiplexity_bin, 7); // floor(0.72 * 10) = 7
        assert_eq!(ctx.desperation_bin, 0); // default
    }

    #[test]
    fn test_context_with_epiplexity_builder() {
        let ctx = ConfiguratorContext::new(1, 5).with_epiplexity(0.83);
        assert_eq!(ctx.domain, 1);
        assert_eq!(ctx.entropy_bin, 5);
        assert_eq!(ctx.epiplexity_bin, 8);
        assert_eq!(ctx.desperation_bin, 0);
    }

    #[test]
    fn test_context_with_epiplexity_clamp_high() {
        let ctx = ConfiguratorContext::new(0, 0).with_epiplexity(5.0);
        assert_eq!(ctx.epiplexity_bin, 9, "should clamp to 9");
    }

    #[test]
    fn test_context_with_epiplexity_clamp_zero() {
        let ctx = ConfiguratorContext::new(0, 0).with_epiplexity(0.0);
        assert_eq!(ctx.epiplexity_bin, 0);
    }

    #[test]
    fn test_context_with_epiplexity_negative() {
        // Negative epiplexity = no structure, should clamp to 0
        let ctx = ConfiguratorContext::new(0, 0).with_epiplexity(-1.0);
        assert_eq!(ctx.epiplexity_bin, 0);
    }

    #[test]
    fn test_epiplexity_bin_static_helper() {
        assert_eq!(ConfiguratorContext::epiplexity_bin(0.0), 0);
        assert_eq!(ConfiguratorContext::epiplexity_bin(0.55), 5);
        assert_eq!(ConfiguratorContext::epiplexity_bin(0.99), 9);
        assert_eq!(ConfiguratorContext::epiplexity_bin(10.0), 9);
    }

    #[test]
    fn test_context_backward_compat_new_defaults_epiplexity_zero() {
        let ctx = ConfiguratorContext::new(2, 7);
        assert_eq!(
            ctx.epiplexity_bin, 0,
            "new() should default epiplexity_bin to 0"
        );
    }

    #[test]
    fn test_context_equality_includes_epiplexity() {
        let ctx_a = ConfiguratorContext::new(0, 5).with_epiplexity(0.5);
        let ctx_b = ConfiguratorContext::new(0, 5).with_epiplexity(0.5);
        let ctx_c = ConfiguratorContext::new(0, 5).with_epiplexity(0.6);

        assert_eq!(ctx_a, ctx_b, "same epiplexity_bin should be equal");
        assert_ne!(ctx_a, ctx_c, "different epiplexity_bin should not be equal");
    }

    // ── EpiplexityArmHeuristic ─────────────────────────────────

    #[test]
    fn test_heuristic_high_structure_low_entropy_plan_extend() {
        let decision = EpiplexityArmHeuristic::suggest(7, 2); // high S_T, low H_T
        assert_eq!(
            decision,
            PlanningDecision::PlanExtend,
            "high S_T + low H_T should suggest PlanExtend"
        );
    }

    #[test]
    fn test_heuristic_low_structure_high_entropy_plan_skip() {
        let decision = EpiplexityArmHeuristic::suggest(2, 8); // low S_T, high H_T
        assert_eq!(
            decision,
            PlanningDecision::PlanSkip,
            "low S_T + high H_T should suggest PlanSkip"
        );
    }

    #[test]
    fn test_heuristic_high_structure_high_entropy_plan_new() {
        let decision = EpiplexityArmHeuristic::suggest(8, 8); // high S_T, high H_T
        assert_eq!(
            decision,
            PlanningDecision::PlanNew,
            "high S_T + high H_T should suggest PlanNew"
        );
    }

    #[test]
    fn test_heuristic_low_structure_low_entropy_plan_skip() {
        let decision = EpiplexityArmHeuristic::suggest(2, 2); // low S_T, low H_T
        assert_eq!(
            decision,
            PlanningDecision::PlanSkip,
            "low S_T + low H_T should suggest PlanSkip"
        );
    }

    #[test]
    fn test_consistency_bonus_match() {
        let bonus = EpiplexityArmHeuristic::consistency_bonus(
            7,                            // high S_T
            2,                            // low H_T
            PlanningDecision::PlanExtend, // matches heuristic
        );
        assert!(
            (bonus - 0.1).abs() < 1e-6,
            "matching arm should get 0.1 bonus, got {bonus}"
        );
    }

    #[test]
    fn test_consistency_bonus_mismatch() {
        let bonus = EpiplexityArmHeuristic::consistency_bonus(
            7,                          // high S_T
            2,                          // low H_T
            PlanningDecision::PlanSkip, // does NOT match heuristic (expects PlanExtend)
        );
        assert!(
            (bonus - 0.0).abs() < 1e-6,
            "mismatching arm should get 0.0 bonus, got {bonus}"
        );
    }

    // ── select_with_epiplexity ──────────────────────────────────

    #[test]
    fn test_select_with_epiplexity_uses_heuristic_for_unvisited() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::from_entropy_epiplexity(0, 0.2, 0.8);
        // high S_T (bin 8) + low H_T (bin 2) → PlanExtend

        let decision = bandit.select_with_epiplexity(ctx);
        assert_eq!(
            decision,
            PlanningDecision::PlanExtend,
            "unvisited context should use heuristic"
        );
    }

    #[test]
    fn test_select_with_epiplexity_uses_ucb1_after_visit() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::from_entropy_epiplexity(0, 0.2, 0.8);

        // First select uses heuristic → PlanExtend
        let d1 = bandit.select_with_epiplexity(ctx);
        assert_eq!(d1, PlanningDecision::PlanExtend);

        // Give PlanSkip a high reward → should eventually converge to PlanSkip via UCB1
        for _ in 0..200 {
            let d = bandit.select_with_epiplexity(ctx);
            let reward = match d {
                PlanningDecision::PlanSkip => 1.0,
                _ => 0.0,
            };
            bandit.update_with_epiplexity(ctx, d, reward);
        }

        let d_final = bandit.select_with_epiplexity(ctx);
        assert_eq!(
            d_final,
            PlanningDecision::PlanSkip,
            "after training, UCB1 should override heuristic"
        );
    }

    #[test]
    fn test_select_with_epiplexity_context_isolation() {
        let mut bandit = ConfiguratorBandit::new();

        // Two different epiplexity bins → different contexts
        let ctx_high_s = ConfiguratorContext::from_entropy_epiplexity(0, 0.2, 0.8);
        let ctx_low_s = ConfiguratorContext::from_entropy_epiplexity(0, 0.2, 0.1);

        // Train high S_T to prefer PlanExtend
        for _ in 0..200 {
            let d = bandit.select_with_epiplexity(ctx_high_s);
            let reward = match d {
                PlanningDecision::PlanExtend => 1.0,
                _ => 0.0,
            };
            bandit.update_with_epiplexity(ctx_high_s, d, reward);
        }

        // Train low S_T to prefer PlanSkip
        for _ in 0..200 {
            let d = bandit.select_with_epiplexity(ctx_low_s);
            let reward = match d {
                PlanningDecision::PlanSkip => 1.0,
                _ => 0.0,
            };
            bandit.update_with_epiplexity(ctx_low_s, d, reward);
        }

        let d_high = bandit.select_with_epiplexity(ctx_high_s);
        let d_low = bandit.select_with_epiplexity(ctx_low_s);

        assert_eq!(
            d_high,
            PlanningDecision::PlanExtend,
            "high S_T should prefer PlanExtend"
        );
        assert_eq!(
            d_low,
            PlanningDecision::PlanSkip,
            "low S_T should prefer PlanSkip"
        );
    }

    #[test]
    fn test_update_with_epiplexity_adds_bonus() {
        let mut bandit = ConfiguratorBandit::new();
        let ctx = ConfiguratorContext::from_entropy_epiplexity(0, 0.2, 0.8);

        // PlanExtend matches heuristic → should get bonus on top of base reward
        bandit.update_with_epiplexity(ctx, PlanningDecision::PlanExtend, 0.5);
        let q_extend = bandit.q_value(ctx, PlanningDecision::PlanExtend).unwrap();
        assert_eq!(q_extend, 0.6, "0.5 reward + 0.1 consistency bonus = 0.6");

        // PlanSkip does NOT match → no bonus
        bandit.update_with_epiplexity(ctx, PlanningDecision::PlanSkip, 0.5);
        let q_skip = bandit.q_value(ctx, PlanningDecision::PlanSkip).unwrap();
        assert_eq!(q_skip, 0.5, "0.5 reward + 0.0 no bonus = 0.5");
    }

    #[test]
    fn test_epiplexity_context_separate_from_entropy_only() {
        // Same entropy bin but different epiplexity bins → separate Q-values
        let mut bandit = ConfiguratorBandit::new();

        let ctx_no_ep = ConfiguratorContext::new(0, 5); // epiplexity_bin = 0
        let ctx_high_ep = ConfiguratorContext::new(0, 5).with_epiplexity(0.8); // epiplexity_bin = 8

        // Train no-epiplexity to prefer PlanSkip
        for _ in 0..100 {
            let d = bandit.select(ctx_no_ep);
            let reward = match d {
                PlanningDecision::PlanSkip => 1.0,
                _ => 0.0,
            };
            bandit.update(ctx_no_ep, d, reward);
        }

        // Train high-epiplexity to prefer PlanNew
        for _ in 0..100 {
            let d = bandit.select(ctx_high_ep);
            let reward = match d {
                PlanningDecision::PlanNew => 1.0,
                _ => 0.0,
            };
            bandit.update(ctx_high_ep, d, reward);
        }

        assert_eq!(
            bandit.select(ctx_no_ep),
            PlanningDecision::PlanSkip,
            "no epiplexity context should prefer PlanSkip"
        );
        assert_eq!(
            bandit.select(ctx_high_ep),
            PlanningDecision::PlanNew,
            "high epiplexity context should prefer PlanNew"
        );
    }
}
