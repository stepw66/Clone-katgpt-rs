//! RandOpt basic example — synthetic weight perturbation demo.
//!
//! Demonstrates:
//! 1. Generate synthetic base weights
//! 2. Run RandOpt with N=100 perturbations, K=10 ensemble
//! 3. Show base → top-1 → ensemble improvement
//! 4. Print solution density and spectral discordance

use katgpt_rs::pruners::bandit::spectral_discordance;
use katgpt_rs::pruners::randopt::*;

fn main() {
    println!("=== RandOpt Weight Perturbation Demo (Plan 121) ===\n");

    // Synthetic "model" weights — 100 params
    let base_weights: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();

    // Target: we want weights close to all-ones
    let target: Vec<f32> = vec![1.0; 100];

    let scorer = AccuracyScorer {
        expected: &target,
        threshold: 0.5,
    };

    // Base score
    let base_score = scorer.score(&base_weights);
    println!("Base weights score: {base_score:.4}");

    // Run RandOpt
    let config = RandOptConfig::default();
    let session = RandOptSession::new(config);
    let result = session.run(&base_weights, &scorer);

    println!("Population: {}", result.scores.len());
    println!("Top-K ensemble: {}", result.top_k_indices.len());
    println!(
        "Best single score: {}",
        result
            .scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!("Ensemble score: {:.4}", result.ensemble_score);
    println!("Base score: {:.4}", result.base_score);
    println!(
        "Solution density (margin=0): {:.4}",
        result.solution_density
    );

    // Spectral discordance demo
    let perf_matrix: Vec<Vec<f32>> = (0..20)
        .map(|i| (0..5).map(|j| ((i + j * 3) as f32 * 0.1).sin()).collect())
        .collect();
    let discordance = spectral_discordance(&perf_matrix);
    println!("Spectral discordance: {discordance:.4}");

    println!("\n✅ RandOpt demo complete");
}
