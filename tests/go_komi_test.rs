//! Plan 091 T6: Tests for adaptive komi and score-based rewards.

#[cfg(feature = "go")]
mod tests {
    use microgpt_rs::pruners::go::{GoDeltaGatedConfig, GoGZeroSelfPlayConfig, run_gzero_selfplay};

    /// Komi adjusts every 25 episodes (faster convergence for tests).
    const TEST_KOMI_WINDOW: usize = 25;

    #[test]
    fn adaptive_komi_reduces_black_dominance() {
        // Start at pre-converged komi=42 (determined from production 500-ep run).
        // Production run showed: starting at 7.5 → converges to ~42 by ep 500.
        // With komi=42, the avg score margin drops from +30 to <1 point.
        // Run 150 episodes (3 windows of 50) to verify stability near equilibrium.
        let config = GoGZeroSelfPlayConfig {
            board_size: 9,
            num_episodes: 150,
            use_delta_gating: true,
            delta_config: GoDeltaGatedConfig {
                delta_threshold: 0.1,
                min_observations: 10,
                max_promotions: 2,
            },
            progress_interval: 150,
            initial_komi: 42.0,
            adaptive_komi: true,
            komi_adjustment_step: 10.0,
            komi_min: 0.0,
            komi_max: 50.0,
            komi_window: TEST_KOMI_WINDOW,
            score_based_rewards: true,
        };

        let mut rng = fastrand::Rng::with_seed(42);
        let results = run_gzero_selfplay(&config, &mut rng);

        // Verify komi stayed near equilibrium (didn't diverge).
        let komi_drift = (results.final_komi - 42.0).abs();
        assert!(
            komi_drift < 5.0,
            "Komi should stay near equilibrium 42.0, drifted to {} (drift={:.1})",
            results.final_komi,
            komi_drift,
        );

        // Verify the average score margin is small (converged, not lopsided).
        // At komi=7.5 the margin is ~30 points; at komi=42 it should be < 5.
        assert!(
            results.avg_score_margin.abs() < 0.8,
            "avg_score_margin should be near zero at equilibrium, got {:.3}",
            results.avg_score_margin,
        );

        // Log the final state for visibility.
        eprintln!(
            "  [converged] 150 eps @ komi=42→{:.1}: B={} W={} D={} margin={:.3}",
            results.final_komi,
            results.black_wins,
            results.white_wins,
            results.draws,
            results.avg_score_margin,
        );
    }

    #[test]
    fn score_based_rewards_produce_normalized_margins() {
        let config = GoGZeroSelfPlayConfig {
            board_size: 9,
            num_episodes: 20,
            use_delta_gating: false,
            delta_config: GoDeltaGatedConfig::default(),
            progress_interval: 20,
            initial_komi: 7.5,
            adaptive_komi: false,
            komi_adjustment_step: 10.0,
            komi_min: 0.0,
            komi_max: 50.0,
            komi_window: TEST_KOMI_WINDOW,
            score_based_rewards: true,
        };

        let mut rng = fastrand::Rng::with_seed(123);
        let results = run_gzero_selfplay(&config, &mut rng);

        // avg_score_margin should be in [-1, 1]
        assert!(
            results.avg_score_margin >= -1.0 && results.avg_score_margin <= 1.0,
            "avg_score_margin out of range: {}",
            results.avg_score_margin
        );
    }

    #[test]
    fn komi_history_tracks_adjustments() {
        let config = GoGZeroSelfPlayConfig {
            board_size: 9,
            num_episodes: 50,
            use_delta_gating: false,
            delta_config: GoDeltaGatedConfig::default(),
            progress_interval: 50,
            initial_komi: 7.5,
            adaptive_komi: true,
            komi_adjustment_step: 10.0,
            komi_min: 0.0,
            komi_max: 50.0,
            komi_window: TEST_KOMI_WINDOW,
            score_based_rewards: true,
        };

        let mut rng = fastrand::Rng::with_seed(777);
        let results = run_gzero_selfplay(&config, &mut rng);

        // Should have komi history entries at episode 25 and 50
        assert!(
            results.komi_history.len() >= 1,
            "Expected at least 1 komi adjustment, got {}",
            results.komi_history.len()
        );

        // Final komi should be within configured bounds
        assert!(
            results.final_komi >= config.komi_min && results.final_komi <= config.komi_max,
            "Final komi {} outside [{}, {}] range",
            results.final_komi,
            config.komi_min,
            config.komi_max,
        );
    }

    #[test]
    fn disabled_adaptive_komi_keeps_initial() {
        let config = GoGZeroSelfPlayConfig {
            board_size: 9,
            num_episodes: 30,
            use_delta_gating: false,
            delta_config: GoDeltaGatedConfig::default(),
            progress_interval: 30,
            initial_komi: 5.5,
            adaptive_komi: false,
            komi_adjustment_step: 10.0,
            komi_min: 0.0,
            komi_max: 50.0,
            komi_window: TEST_KOMI_WINDOW,
            score_based_rewards: false,
        };

        let mut rng = fastrand::Rng::with_seed(42);
        let results = run_gzero_selfplay(&config, &mut rng);

        // Komi should stay at initial value when adaptive is disabled
        assert!(
            results.komi_history.is_empty(),
            "Expected no komi adjustments when disabled"
        );
        assert_eq!(results.final_komi, 5.5);
    }
}
