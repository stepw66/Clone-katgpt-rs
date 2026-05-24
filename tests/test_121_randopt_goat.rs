//! GOAT proofs for RandOpt weight-space perturbation (Plan 121).

#[cfg(feature = "randopt_weight")]
mod tests {
    use katgpt_rs::pruners::bandit::{solution_density, spectral_discordance};
    use katgpt_rs::pruners::randopt::*;

    #[test]
    fn test_randopt_config_default() {
        let config = RandOptConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.ensemble_size, 10);
        assert_eq!(config.sigma_set.len(), 3);
        assert_eq!(config.base_seed, 42);
    }

    #[test]
    fn test_perturb_deterministic() {
        let config = RandOptConfig::default();
        let sampler = RandOptWeightSampler::new(config);
        let base = vec![0.5; 10];
        let p1 = sampler.perturb(&base, 0);
        let p2 = sampler.perturb(&base, 0);
        assert_eq!(p1, p2, "Same arm should produce identical perturbation");
    }

    #[test]
    fn test_perturb_different_arms_differ() {
        let config = RandOptConfig::default();
        let sampler = RandOptWeightSampler::new(config);
        let base = vec![0.5; 10];
        let p1 = sampler.perturb(&base, 0);
        let p2 = sampler.perturb(&base, 1);
        assert_ne!(
            p1, p2,
            "Different arms should produce different perturbations"
        );
    }

    #[test]
    fn test_ensemble_improves_over_base() {
        let target: Vec<f32> = vec![1.0; 50];
        let base: Vec<f32> = (0..50).map(|i| (i as f32 * 0.1).sin()).collect();
        let scorer = AccuracyScorer {
            expected: &target,
            threshold: 0.5,
        };

        let config = RandOptConfig {
            population_size: 100,
            ..Default::default()
        };
        let session = RandOptSession::new(config);
        let result = session.run(&base, &scorer);

        assert!(
            result.ensemble_score >= result.base_score * 0.9,
            "Ensemble should not be much worse than base"
        );
    }

    #[test]
    fn test_solution_density_bounds() {
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let d = solution_density(&scores, 0.5, 0.0);
        assert!((0.0..=1.0).contains(&d));
        assert!((d - 0.6).abs() < 0.01); // 3/5 ≥ 0.5
    }

    #[test]
    fn test_spectral_discordance_bounds() {
        let matrix = vec![vec![0.5; 4]; 10]; // All generalists
        let d = spectral_discordance(&matrix);
        assert!((0.0..=1.0).contains(&d));
        assert!(d < 0.01, "Uniform matrix should have low discordance");
    }

    #[test]
    fn test_spectral_discordance_specialists() {
        let matrix: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut row = vec![0.0; 5];
                row[i % 5] = 1.0;
                row
            })
            .collect();
        let d = spectral_discordance(&matrix);
        assert!(d > 0.1, "Specialists should have higher discordance");
    }

    #[test]
    fn test_sigma_round_robin() {
        let config = RandOptConfig {
            sigma_set: vec![0.01, 0.02, 0.03],
            ..Default::default()
        };
        let sampler = RandOptWeightSampler::new(config);
        assert!((sampler.sigma_for_arm(0) - 0.01).abs() < f32::EPSILON);
        assert!((sampler.sigma_for_arm(1) - 0.02).abs() < f32::EPSILON);
        assert!((sampler.sigma_for_arm(2) - 0.03).abs() < f32::EPSILON);
        assert!((sampler.sigma_for_arm(3) - 0.01).abs() < f32::EPSILON); // wraps
    }

    #[test]
    fn test_ensemble_discrete_majority_vote() {
        let ensemble = RandOptEnsemble::new(5);
        let predictions = vec![1, 1, 1, 2, 3];
        assert_eq!(ensemble.aggregate_discrete(&predictions), 1);
    }

    #[test]
    fn test_ensemble_discrete_empty() {
        let ensemble = RandOptEnsemble::new(5);
        assert_eq!(ensemble.aggregate_discrete(&[]), 0);
    }

    #[test]
    fn test_ensemble_continuous_mean() {
        let ensemble = RandOptEnsemble::new(3);
        let predictions = vec![0.2, 0.4, 0.6];
        let mean = ensemble.aggregate_continuous(&predictions);
        assert!((mean - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_ensemble_continuous_empty() {
        let ensemble = RandOptEnsemble::new(3);
        assert_eq!(ensemble.aggregate_continuous(&[]), 0.0);
    }

    #[test]
    fn test_session_result_fields_populated() {
        let base = vec![0.5; 20];
        let target = vec![1.0; 20];
        let scorer = AccuracyScorer {
            expected: &target,
            threshold: 0.6,
        };
        let config = RandOptConfig {
            population_size: 10,
            ensemble_size: 3,
            ..Default::default()
        };
        let session = RandOptSession::new(config);
        let result = session.run(&base, &scorer);

        assert_eq!(result.scores.len(), 10, "should score all 10 arms");
        assert_eq!(result.top_k_indices.len(), 3, "should select top-3");
        assert_eq!(result.best_seeds.len(), 3);
        assert_eq!(result.best_sigmas.len(), 3);
        assert!((0.0..=1.0).contains(&result.solution_density));
    }

    #[test]
    fn test_solution_density_empty() {
        let d = solution_density(&[], 0.5, 0.0);
        assert_eq!(d, 0.0, "empty scores should return 0.0 density");
    }

    #[test]
    fn test_solution_density_with_margin() {
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        // base=0.5, margin=0.2 → threshold=0.7 → 2/5 = 0.4
        let d = solution_density(&scores, 0.5, 0.2);
        assert!((d - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_spectral_discordance_empty() {
        assert_eq!(spectral_discordance(&[]), 0.0);
    }

    #[test]
    fn test_spectral_discordance_single_task() {
        let matrix = vec![vec![0.5]; 5];
        assert_eq!(
            spectral_discordance(&matrix),
            0.0,
            "single task should have 0 discordance"
        );
    }

    #[test]
    fn test_perturb_preserves_shape() {
        let config = RandOptConfig::default();
        let sampler = RandOptWeightSampler::new(config);
        let base = vec![0.0; 42];
        let perturbed = sampler.perturb(&base, 7);
        assert_eq!(
            perturbed.len(),
            42,
            "perturbed weights should have same length"
        );
    }

    #[test]
    fn test_perturb_applies_noise() {
        let config = RandOptConfig {
            sigma_set: vec![1.0],
            ..Default::default()
        };
        let sampler = RandOptWeightSampler::new(config);
        let base = vec![0.0; 20];
        let perturbed = sampler.perturb(&base, 0);
        // With sigma=1.0, perturbed should differ from zero base
        let any_nonzero = perturbed.iter().any(|&v| v.abs() > 0.01);
        assert!(
            any_nonzero,
            "perturbation with sigma=1.0 should produce non-zero values"
        );
    }

    #[test]
    fn test_accuracy_scorer_perfect() {
        let expected = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 2.0, 3.0];
        let scorer = AccuracyScorer {
            expected: &expected,
            threshold: 0.01,
        };
        assert!((scorer.score(&weights) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_accuracy_scorer_zero() {
        let expected = vec![1.0, 1.0, 1.0];
        let weights = vec![0.0, 0.0, 0.0];
        let scorer = AccuracyScorer {
            expected: &expected,
            threshold: 0.5,
        };
        assert!((scorer.score(&weights) - 0.0).abs() < 0.001);
    }
}
