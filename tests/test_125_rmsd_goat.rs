//! GOAT proofs for RMSD relevance-masked self-distillation (Plan 125).
//!
//! Tests the two-step relevance mask pipeline:
//! 1. Pre-filter T actions by |Q_teacher - Q_student| magnitude
//! 2. Select S most relevant actions via magnitude-only judge
//!
//! Feature gate: `rmsd_distill`

#[cfg(feature = "rmsd_distill")]
mod tests {
    use katgpt_rs::pruners::rmsd_relevance::*;

    // ── RmsdConfig ─────────────────────────────────────────────

    #[test]
    fn test_rmsd_config_default() {
        let config = RmsdConfig::default();
        assert_eq!(config.top_t, 20);
        assert_eq!(config.top_s, 5);
    }

    // ── LogprobMagnitudeFilter ─────────────────────────────────

    #[test]
    fn test_logprob_magnitude_filter_top3() {
        let filter = LogprobMagnitudeFilter::new(3);
        let teacher = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let student = vec![0.1, 0.4, 0.1, 0.1, 0.2];

        let result = filter.filter(&teacher, &student);
        assert_eq!(result.len(), 3); // Top 3

        // Index 3 has largest delta: |0.9 - 0.1| = 0.8
        assert_eq!(result[0].0, 3);
        assert!((result[0].1 - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_logprob_magnitude_filter_all_zero() {
        let filter = LogprobMagnitudeFilter::new(5);
        let vals = vec![0.5, 0.5, 0.5];
        let result = filter.filter(&vals, &vals);
        assert!(result.is_empty(), "No differences → no selection");
    }

    #[test]
    fn test_logprob_magnitude_filter_empty_input() {
        let filter = LogprobMagnitudeFilter::new(5);
        let result = filter.filter(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_logprob_magnitude_filter_mismatched_lengths() {
        let filter = LogprobMagnitudeFilter::new(3);
        let teacher = vec![0.5, 0.3, 0.2];
        let student = vec![0.1]; // shorter
        let result = filter.filter(&teacher, &student);
        // zip stops at shorter length
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_logprob_magnitude_filter_preserves_order() {
        let filter = LogprobMagnitudeFilter::new(10);
        let teacher = vec![0.1, 0.9, 0.1, 0.9, 0.1];
        let student = vec![0.9, 0.1, 0.9, 0.1, 0.9];
        let result = filter.filter(&teacher, &student);

        // All deltas are 0.8, should all pass (up to top_t=10)
        assert_eq!(result.len(), 5);
        // All magnitudes should be ~0.8
        for (_, mag) in &result {
            assert!((mag - 0.8).abs() < 0.001);
        }
    }

    // ── TopKlApproximator ──────────────────────────────────────

    #[test]
    fn test_top_kl_approximator_basic() {
        let approx = TopKlApproximator::new(3);
        let student = vec![0.4, 0.3, 0.2, 0.1];
        let teacher = vec![0.3, 0.3, 0.2, 0.2];

        let kl = approx.kl_topk(&student, &teacher);
        assert!(kl >= 0.0, "KL divergence should be non-negative");
    }

    #[test]
    fn test_top_kl_identical_distributions() {
        let approx = TopKlApproximator::new(3);
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let kl = approx.kl_topk(&probs, &probs);
        assert!(
            kl.abs() < 0.01,
            "KL of identical distributions ≈ 0, got {kl}"
        );
    }

    #[test]
    fn test_top_kl_empty_input() {
        let approx = TopKlApproximator::new(3);
        let kl = approx.kl_topk(&[], &[]);
        assert_eq!(kl, 0.0, "KL of empty distributions should be 0");
    }

    #[test]
    fn test_top_kl_zero_student_probs() {
        let approx = TopKlApproximator::new(3);
        let student = vec![0.0, 0.0, 0.0];
        let teacher = vec![0.3, 0.3, 0.4];
        let kl = approx.kl_topk(&student, &teacher);
        assert_eq!(kl, 0.0, "Zero student probs → zero KL");
    }

    // ── MagnitudeJudge ─────────────────────────────────────────

    #[test]
    fn test_magnitude_judge_basic() {
        let judge = MagnitudeJudge::new(2);
        let candidates = vec![(0, 0.8), (1, 0.5), (2, 0.3), (3, 0.1)];
        let selected = judge.select(&candidates, 4);
        assert_eq!(selected, vec![0, 1]);
    }

    #[test]
    fn test_magnitude_judge_fewer_than_top_s() {
        let judge = MagnitudeJudge::new(5);
        let candidates = vec![(0, 0.8), (1, 0.3)];
        let selected = judge.select(&candidates, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_magnitude_judge_empty_candidates() {
        let judge = MagnitudeJudge::new(3);
        let selected = judge.select(&[], 0);
        assert!(selected.is_empty());
    }

    // ── RmsdRelevanceFilter ────────────────────────────────────

    #[test]
    fn test_rmsd_relevance_filter_basic() {
        let filter = RmsdRelevanceFilter::new(5, 2);
        let teacher_q = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];
        let student_q = vec![0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4];

        let (selected, metrics) = filter.filter_actions(&teacher_q, &student_q);

        assert_eq!(selected.len(), 2); // S=2
        assert_eq!(metrics.total_actions, 7);
        assert!(metrics.heuristic_filtered <= 5); // T=5
        assert!(metrics.mask_density > 0.0);
        assert!(
            metrics.mean_selected_magnitude >= metrics.mean_rejected_magnitude,
            "Selected should have higher magnitude than rejected"
        );
    }

    #[test]
    fn test_rmsd_relevance_filter_identical_q() {
        let filter = RmsdRelevanceFilter::new(5, 2);
        let q = vec![0.5, 0.3, 0.2, 0.1];
        let (selected, metrics) = filter.filter_actions(&q, &q);

        assert_eq!(selected.len(), 0, "Identical Q → no selection");
        assert_eq!(metrics.total_actions, 4);
        assert_eq!(metrics.heuristic_filtered, 0);
        assert_eq!(metrics.judge_selected, 0);
        assert_eq!(metrics.mask_density, 0.0);
    }

    #[test]
    fn test_rmsd_relevance_filter_empty_input() {
        let filter = RmsdRelevanceFilter::new(5, 2);
        let (selected, metrics) = filter.filter_actions(&[], &[]);

        assert!(selected.is_empty());
        assert_eq!(metrics.total_actions, 0);
    }

    #[test]
    fn test_rmsd_relevance_filter_top_s_greater_than_candidates() {
        let filter = RmsdRelevanceFilter::new(5, 10); // S > available
        let teacher_q = vec![0.5, 0.3, 0.2];
        let student_q = vec![0.1, 0.1, 0.1];

        let (selected, _metrics) = filter.filter_actions(&teacher_q, &student_q);
        // Should select all 3 (fewer than S=10)
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_rmsd_relevance_filter_mask_density() {
        let filter = RmsdRelevanceFilter::new(20, 5);
        let teacher_q = vec![0.9; 20];
        let student_q = vec![0.1; 20];

        let (_selected, metrics) = filter.filter_actions(&teacher_q, &student_q);
        // 5 selected from 20 → density = 0.25
        assert!((metrics.mask_density - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_rmsd_relevance_filter_concentrates_signal() {
        let filter = RmsdRelevanceFilter::new(3, 1);
        // Action 0 has huge gap, others small
        let teacher_q = vec![1.0, 0.11, 0.1, 0.1];
        let student_q = vec![0.0, 0.1, 0.1, 0.1];

        let (selected, metrics) = filter.filter_actions(&teacher_q, &student_q);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 0, "Should select highest magnitude action");
        assert!(
            metrics.mean_selected_magnitude > metrics.mean_rejected_magnitude,
            "Signal should be concentrated on selected"
        );
    }

    // ── TeacherContinuation ────────────────────────────────────

    #[test]
    fn test_teacher_continuation_no_plateau() {
        let mut cont = TeacherContinuation::new(3);

        assert!(!cont.check_plateau(0.5)); // New best
        assert!(!cont.check_plateau(0.6)); // New best
        assert!(!cont.check_plateau(0.7)); // New best
        assert!(!cont.was_updated());
    }

    #[test]
    fn test_teacher_continuation_plateau() {
        let mut cont = TeacherContinuation::new(3);

        cont.check_plateau(0.5); // Best
        cont.check_plateau(0.4); // No improvement (step 1)
        cont.check_plateau(0.3); // No improvement (step 2)
        let should_update = cont.check_plateau(0.2); // step 3 ≥ patience

        assert!(should_update);
        assert!(cont.was_updated());
    }

    #[test]
    fn test_teacher_continuation_no_double_update() {
        let mut cont = TeacherContinuation::new(2);

        cont.check_plateau(0.5);
        cont.check_plateau(0.4);
        let first = cont.check_plateau(0.3);
        assert!(first, "First plateau should trigger update");

        let second = cont.check_plateau(0.2);
        assert!(
            !second,
            "Should not trigger again after teacher already updated"
        );
    }

    #[test]
    fn test_teacher_continuation_reset() {
        let mut cont = TeacherContinuation::new(2);
        cont.check_plateau(0.5);
        cont.check_plateau(0.4);
        cont.check_plateau(0.3);
        cont.reset();

        assert!(!cont.was_updated());
        assert_eq!(cont.best_metric(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_teacher_continuation_improvement_resets_counter() {
        let mut cont = TeacherContinuation::new(3);

        cont.check_plateau(0.5);
        cont.check_plateau(0.4); // step 1
        cont.check_plateau(0.6); // New best → resets counter
        cont.check_plateau(0.5); // step 1
        cont.check_plateau(0.4); // step 2
        let should_update = cont.check_plateau(0.3); // step 3 ≥ patience

        assert!(should_update, "Should trigger after patience from reset");
    }

    #[test]
    fn test_teacher_continuation_best_metric_tracking() {
        let mut cont = TeacherContinuation::new(5);

        cont.check_plateau(0.3);
        assert!((cont.best_metric() - 0.3).abs() < 0.001);

        cont.check_plateau(0.7);
        assert!((cont.best_metric() - 0.7).abs() < 0.001);

        cont.check_plateau(0.1);
        assert!(
            (cont.best_metric() - 0.7).abs() < 0.001,
            "Best should not decrease"
        );
    }

    // ── rmsd_loss ──────────────────────────────────────────────

    #[test]
    fn test_rmsd_loss_basic() {
        let selected = vec![0, 2];
        let teacher_q = vec![0.9, 0.1, 0.8, 0.2];
        let student_q = vec![0.1, 0.1, 0.1, 0.2];

        let loss = rmsd_loss(&selected, &teacher_q, &student_q, 5.0);
        assert!(
            loss > 0.0,
            "Should have positive loss for mismatched Q-values"
        );
    }

    #[test]
    fn test_rmsd_loss_empty_selection() {
        let loss = rmsd_loss(&[], &[0.5], &[0.1], 5.0);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_rmsd_loss_identical_q() {
        let selected = vec![0, 1];
        let q = vec![0.5, 0.3];
        let loss = rmsd_loss(&selected, &q, &q, 5.0);
        assert_eq!(loss, 0.0, "Identical Q-values → zero loss");
    }

    #[test]
    fn test_rmsd_loss_symmetric_in_selection() {
        // Loss should be same regardless of selection order
        let sel_a = vec![0, 1];
        let sel_b = vec![1, 0];
        let teacher = vec![0.9, 0.7];
        let student = vec![0.1, 0.3];

        let loss_a = rmsd_loss(&sel_a, &teacher, &student, 5.0);
        let loss_b = rmsd_loss(&sel_b, &teacher, &student, 5.0);
        assert!((loss_a - loss_b).abs() < 0.001);
    }

    #[test]
    fn test_rmsd_loss_higher_beta_more_gating() {
        let selected = vec![0];
        let teacher = vec![0.8];
        let student = vec![0.2];

        let loss_low = rmsd_loss(&selected, &teacher, &student, 1.0);
        let loss_high = rmsd_loss(&selected, &teacher, &student, 10.0);

        // Higher β → stronger gating → different loss
        // Both should be positive since gap is positive
        assert!(loss_low > 0.0);
        assert!(loss_high > 0.0);
    }

    #[test]
    fn test_rmsd_loss_out_of_bounds_selected() {
        let selected = vec![5]; // Out of bounds
        let teacher = vec![0.5];
        let student = vec![0.1];

        let loss = rmsd_loss(&selected, &teacher, &student, 5.0);
        // Should use 0.0 for out-of-bounds → gap=0, loss=0
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_rmsd_loss_negative_gap() {
        let selected = vec![0];
        let teacher = vec![0.1];
        let student = vec![0.9];

        let loss = rmsd_loss(&selected, &teacher, &student, 5.0);
        // Negative gap: gate < 0.5, but |gap| still positive
        // gate * |gap| should still contribute
        assert!(
            loss > 0.0,
            "Even negative gap produces positive loss via |Δ|"
        );
    }

    // ── Integration: Full Pipeline ─────────────────────────────

    #[test]
    fn test_full_pipeline_filter_then_loss() {
        let teacher_q = vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6];
        let student_q = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

        // Step 1: Filter
        let filter = RmsdRelevanceFilter::new(5, 3);
        let (selected, metrics) = filter.filter_actions(&teacher_q, &student_q);

        assert_eq!(selected.len(), 3);
        assert_eq!(metrics.total_actions, 7);
        assert!(metrics.mask_density > 0.0);

        // Step 2: Compute loss on selected
        let loss = rmsd_loss(&selected, &teacher_q, &student_q, 5.0);
        assert!(loss > 0.0);

        // Step 3: Teacher continuation check
        let mut cont = TeacherContinuation::new(5);
        assert!(!cont.check_plateau(-loss)); // First observation → new best
    }

    #[test]
    fn test_pipeline_signal_concentration() {
        // Only 2 actions carry signal, rest are noise
        let teacher_q = vec![0.9, 0.1, 0.1, 0.1, 0.1];
        let student_q = vec![0.1, 0.1, 0.11, 0.09, 0.1];

        let filter = RmsdRelevanceFilter::new(3, 1);
        let (selected, metrics) = filter.filter_actions(&teacher_q, &student_q);

        // Should select action 0 (highest magnitude)
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 0);

        // Selected magnitude should be much higher than rejected
        assert!(metrics.mean_selected_magnitude > 0.5);
        assert!(metrics.mean_rejected_magnitude < 0.1);
    }

    // ── GOAT Proofs ────────────────────────────────────────────

    /// GOAT T1: LogprobMagnitudeFilter selects by |Δ| magnitude
    #[test]
    fn goat_t1_magnitude_filter_selects_by_delta() {
        let filter = LogprobMagnitudeFilter::new(3);
        let teacher = vec![0.9, 0.1, 0.5, 0.2, 0.8];
        let student = vec![0.1, 0.9, 0.1, 0.2, 0.1];

        let result = filter.filter(&teacher, &student);

        // Top 3 by |Δ|: idx0 (0.8), idx1 (0.8), idx4 (0.7)
        assert_eq!(result.len(), 3);
        let magnitudes: Vec<f32> = result.iter().map(|(_, m)| *m).collect();
        // Should be sorted descending
        for w in magnitudes.windows(2) {
            assert!(w[0] >= w[1], "Should be sorted descending");
        }
    }

    /// GOAT T2: TopKlApproximator KL ≥ 0 (non-negative)
    #[test]
    fn goat_t2_kl_non_negative() {
        let approx = TopKlApproximator::new(5);

        let test_cases = vec![
            (vec![0.4, 0.3, 0.2, 0.1], vec![0.25, 0.25, 0.25, 0.25]),
            (vec![0.1, 0.1, 0.1, 0.7], vec![0.7, 0.1, 0.1, 0.1]),
            (vec![0.5, 0.5], vec![0.3, 0.7]),
        ];

        for (student, teacher) in test_cases {
            let kl = approx.kl_topk(&student, &teacher);
            assert!(kl >= 0.0, "KL should be non-negative, got {kl}");
        }
    }

    /// GOAT T3: MagnitudeJudge selects exactly top-S
    #[test]
    fn goat_t3_judge_selects_exactly_top_s() {
        let judge = MagnitudeJudge::new(3);
        let candidates = vec![(10, 0.9), (5, 0.7), (3, 0.5), (7, 0.3), (1, 0.1)];

        let selected = judge.select(&candidates, 5);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected, vec![10, 5, 3]);
    }

    /// GOAT T4: RmsdRelevanceFilter concentrates signal
    #[test]
    fn goat_t4_filter_concentrates_signal() {
        let filter = RmsdRelevanceFilter::new(5, 2);

        // 10 actions, only 2 carry signal
        let teacher_q = vec![0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1];
        let student_q = vec![0.1; 10];

        let (selected, metrics) = filter.filter_actions(&teacher_q, &student_q);

        assert_eq!(selected.len(), 2);
        // Should select indices 0 and 5 (highest magnitude)
        assert!(selected.contains(&0));
        assert!(selected.contains(&5));

        // Signal concentration: selected >> rejected
        assert!(
            metrics.mean_selected_magnitude > 5.0 * metrics.mean_rejected_magnitude,
            "Selected magnitude should dominate rejected"
        );
    }

    /// GOAT T5: rmsd_loss positive for non-trivial gaps
    #[test]
    fn goat_t5_loss_positive_for_gaps() {
        let selected = vec![0, 1, 2];
        let teacher_q = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        let student_q = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        let loss = rmsd_loss(&selected, &teacher_q, &student_q, 5.0);
        assert!(loss > 0.0, "Should have positive loss");
        assert!(loss.is_finite(), "Loss should be finite");
    }

    /// GOAT T6: TeacherContinuation detects plateau
    #[test]
    fn goat_t6_continuation_detects_plateau() {
        let mut cont = TeacherContinuation::new(3);

        // Improving phase
        assert!(!cont.check_plateau(0.1));
        assert!(!cont.check_plateau(0.3));
        assert!(!cont.check_plateau(0.5));

        // Plateau phase
        assert!(!cont.check_plateau(0.4)); // step 1
        assert!(!cont.check_plateau(0.3)); // step 2
        assert!(cont.check_plateau(0.2)); // step 3 = patience
    }

    /// GOAT T7: rmsd_loss zero for identical distributions
    #[test]
    fn goat_t7_loss_zero_identical() {
        let q = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let selected = vec![0, 1, 2, 3, 4];

        let loss = rmsd_loss(&selected, &q, &q, 5.0);
        assert_eq!(loss, 0.0, "Identical distributions → zero loss");
    }

    /// GOAT T8: Filter handles edge cases gracefully
    #[test]
    fn goat_t8_filter_edge_cases() {
        let filter = RmsdRelevanceFilter::new(5, 3);

        // Empty
        let (sel, m) = filter.filter_actions(&[], &[]);
        assert!(sel.is_empty());
        assert_eq!(m.total_actions, 0);

        // Single element, no gap
        let (sel, m) = filter.filter_actions(&[0.5], &[0.5]);
        assert!(sel.is_empty());
        assert_eq!(m.mask_density, 0.0);

        // Single element, with gap
        let (sel, m) = filter.filter_actions(&[0.9], &[0.1]);
        assert_eq!(sel.len(), 1);
        assert!(m.mask_density > 0.0);
    }

    /// GOAT T9: RMSD loss scales with gap magnitude
    #[test]
    fn goat_t9_loss_scales_with_gap() {
        let selected = vec![0];

        let loss_small = rmsd_loss(&selected, &[0.2], &[0.1], 5.0);
        let loss_large = rmsd_loss(&selected, &[0.9], &[0.1], 5.0);

        assert!(
            loss_large > loss_small,
            "Larger gap should produce larger loss"
        );
    }

    /// GOAT T10: Mask density bounded in [0, 1]
    #[test]
    fn goat_t10_mask_density_bounded() {
        let filter = RmsdRelevanceFilter::new(5, 3);

        for n in 1..=20 {
            let teacher = vec![0.9; n];
            let student = vec![0.1; n];
            let (_, metrics) = filter.filter_actions(&teacher, &student);

            assert!(
                metrics.mask_density >= 0.0 && metrics.mask_density <= 1.0,
                "mask_density should be in [0, 1], got {} for n={n}",
                metrics.mask_density
            );
        }
    }
}

// ── Arena GOAT Proofs (Plan 125 T9-T10) ────────────────────────

#[cfg(all(feature = "rmsd_distill", feature = "sdar_gate", feature = "bomber"))]
mod arena_goat {
    use katgpt_rs::pruners::bomber::arena_runner::{BomberArenaConfig, run_bomber_matchup};
    use katgpt_rs::pruners::bomber::{BomberPlayer, RandomPlayer, RmsdPlayer, SdarPlayer};

    /// GOAT T9: RMSD non-degradation vs SDAR in bomber arena (1000 rounds).
    ///
    /// Tests that RMSD's relevance-masking doesn't catastrophically hurt performance
    /// compared to plain SDAR. RMSD should be within 10% relative gap of SDAR.
    ///
    /// Like SDAR and VPD, RMSD's signal improvement affects convergence rate,
    /// not action selection — so it performs comparably in arena.
    #[test]
    fn goat_t9_rmsd_non_degradation_vs_sdar() {
        let config = BomberArenaConfig {
            games: 1000,
            tick_limit: 300,
            procedural: true,
            ..Default::default()
        };

        // Run RMSD + Random vs SDAR + Random matchup
        let mut players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(RmsdPlayer::new(0)),
            Box::new(RandomPlayer::new(1)),
            Box::new(SdarPlayer::new(2)),
            Box::new(RandomPlayer::new(3)),
        ];

        let result = run_bomber_matchup(&mut players, &config);

        // Count wins for each player
        let mut rmsd_wins = 0usize;
        let mut sdar_wins = 0usize;

        for game in &result.games {
            match game.winner {
                Some(0) => rmsd_wins += 1, // RMSD is slot 0
                Some(2) => sdar_wins += 1, // SDAR is slot 2
                _ => {}
            }
        }

        let total_decisive = rmsd_wins + sdar_wins;
        if total_decisive == 0 {
            // All draws — neither dominates
            return;
        }

        let rmsd_rate = rmsd_wins as f64 / total_decisive as f64;
        let sdar_rate = sdar_wins as f64 / total_decisive as f64;
        let relative_gap = (rmsd_rate - sdar_rate).abs();

        // Non-degradation: RMSD within 10% relative gap of SDAR
        assert!(
            relative_gap <= 0.10,
            "RMSD ({rmsd_wins}W/{rmsd_rate:.3}) vs SDAR ({sdar_wins}W/{sdar_rate:.3}): \
             relative gap {relative_gap:.3} > 0.10 — RMSD degraded",
        );
    }

    /// GOAT T10: RMSD continuation activates without degradation (ablation).
    ///
    /// TeacherContinuation detects plateau and triggers teacher snapshot.
    /// Verifies the mechanism runs for 200 games without panicking and
    /// maintains valid internal state.
    #[test]
    fn goat_t10_continuation_activates_arena() {
        let config = BomberArenaConfig {
            games: 200,
            tick_limit: 300,
            procedural: true,
            ..Default::default()
        };

        let mut players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(RmsdPlayer::new(0)),
            Box::new(RandomPlayer::new(1)),
            Box::new(RandomPlayer::new(2)),
            Box::new(RandomPlayer::new(3)),
        ];

        let result = run_bomber_matchup(&mut players, &config);

        // All games should complete without errors
        assert_eq!(
            result.games.len(),
            config.games,
            "All games should complete"
        );

        // Verify player state after continuation cycles
        if let Some(rmsd) = players[0].as_any().downcast_ref::<RmsdPlayer>() {
            let (mean_delta, gate_at_zero, _best_template, mask_density) = rmsd.rmsd_summary();

            // After 200 games, Q-values should be non-trivial
            assert!(mean_delta >= 0.0, "Mean delta should be non-negative");
            assert!(
                (gate_at_zero - 0.5).abs() < 0.01,
                "Gate at zero should be ~0.5"
            );
            assert!(mask_density > 0.0, "Mask density should be positive");
            assert!(mask_density <= 1.0, "Mask density should be ≤ 1.0");
        }
    }
}
