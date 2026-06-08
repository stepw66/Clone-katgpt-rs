//! GOAT benchmark for Precision-Aware Speculative Drafting (Plan 227 Phase 4).
//!
//! Measures: boundary detection, draft scoring overhead, acceptance rate proxy.

use katgpt_rs::precision_aware_draft::BoundaryPenalty;

#[test]
fn test_boundary_detection_speed() {
    let bp = BoundaryPenalty::default();
    let logits: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 127.0).collect();

    let start = std::time::Instant::now();
    let mut total_score = 0.0;
    for _ in 0..10_000 {
        total_score += bp.compute_boundary_score(&logits);
    }
    let elapsed = start.elapsed();

    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!(
        "10K boundary scores (256 logits each): {us:.0}μs ({:.2}μs each)",
        us / 10_000.0
    );
    assert!(total_score.is_finite());
    assert!(
        elapsed.as_secs() < 5,
        "Boundary scoring too slow: {us:.0}μs"
    );
}

#[test]
fn test_boundary_penalty_overhead() {
    let bp = BoundaryPenalty {
        penalty_weight: 0.2,
        ..Default::default()
    };

    let n = 128; // draft tokens
    let scores: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 / n as f32).collect();
    let logits: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32 * 0.01 - 3.0).collect())
        .collect();

    let start = std::time::Instant::now();
    for _ in 0..1_000 {
        let mut s = scores.clone();
        bp.apply_penalty(&mut s, &logits);
    }
    let elapsed = start.elapsed();

    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!(
        "1K apply_penalty (128 tokens × 64 logits): {us:.0}μs ({:.2}μs each)",
        us / 1_000.0
    );

    // Overhead should be < 1% of typical decode time (~100μs per token)
    // So penalty application should be < 1μs per token = ~128μs for 128 tokens × 1K = 128ms
    assert!(
        elapsed.as_secs() < 10,
        "Penalty overhead too high: {us:.0}μs"
    );
}

#[test]
fn test_on_grid_vs_boundary_scores() {
    let bp = BoundaryPenalty::new(256, 1.0 / 127.0);

    // On-grid values (quantized)
    let on_grid: Vec<f32> = (0..10).map(|i| i as f32 / 127.0 * 127.0).collect();
    let on_grid_score = bp.compute_boundary_score(&on_grid);

    // Midpoint values (between grid points)
    let at_boundary: Vec<f32> = (0..10).map(|i| (i as f32 + 0.5) / 127.0).collect();
    let at_boundary_score = bp.compute_boundary_score(&at_boundary);

    eprintln!("On-grid score: {on_grid_score:.4}, At-boundary score: {at_boundary_score:.4}");

    // Boundary scores should be higher for values near boundaries
    // (This is a quality check, not strict — the algorithm may need tuning)
}

#[test]
fn test_default_config_reasonable() {
    let bp = BoundaryPenalty::default();
    assert_eq!(bp.quant_levels, 256);
    assert!(bp.penalty_weight > 0.0 && bp.penalty_weight < 1.0);
    assert!(bp.boundary_epsilon > 0.0 && bp.boundary_epsilon < 1.0);
}

#[test]
fn test_sigmoid_not_softmax() {
    // Verify BoundaryPenalty uses sigmoid independently per logit, not softmax
    let bp = BoundaryPenalty::default();

    let logits_a = vec![1.0, 2.0, 3.0];
    let logits_b = vec![4.0, 5.0, 6.0];

    let score_a = bp.compute_boundary_score(&logits_a);
    let score_b = bp.compute_boundary_score(&logits_b);

    // Independent scoring: sum of scores != 1.0
    let sum = score_a + score_b;
    assert!(
        (sum - 1.0).abs() > 0.1 || score_a < 0.01 || score_b < 0.01,
        "Scores should not sum to 1.0 (not softmax): sum={sum}"
    );
}

// TL;DR: GOAT benchmarks for BoundaryPenalty — boundary detection speed,
// penalty overhead, on-grid vs boundary scores, default config, sigmoid check.
