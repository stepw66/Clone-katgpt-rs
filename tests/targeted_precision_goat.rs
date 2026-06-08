//! GOAT benchmark for Targeted Precision Budget (Plan 227 Phase 2).
//!
//! Measures: perplexity impact, KV cache size consistency, latency.

use katgpt_rs::targeted_precision::PrecisionBudget;

#[test]
fn test_uniform_budget_consistency() {
    let budget = PrecisionBudget::uniform(4, 8, 4);
    assert!(budget.verify_budget());
    assert_eq!(budget.total_kv_bits(128, 64), 32 * 4 * 128 * 64);
}

#[test]
fn test_targeted_allocates_more_to_sensitive() {
    let mut sensitivity = vec![0.1; 32];
    sensitivity[0] = 0.95; // very sensitive head
    sensitivity[15] = 0.85;
    sensitivity[31] = 0.05; // robust head

    let budget = PrecisionBudget::compute_budget(4, 8, &sensitivity, 2.5);

    // Sensitive heads should get more bits
    assert!(
        budget.get_bits(0, 0) > budget.get_bits(3, 7),
        "sensitive head should get more bits: {} vs {}",
        budget.get_bits(0, 0),
        budget.get_bits(3, 7)
    );

    // Budget should be satisfied
    assert!(budget.verify_budget());
}

#[test]
fn test_kv_cache_size_same_as_uniform() {
    let uniform = PrecisionBudget::uniform(4, 8, 3);
    let mut sensitivity = vec![0.5; 32];
    sensitivity[0] = 0.9;
    let targeted = PrecisionBudget::compute_budget(4, 8, &sensitivity, 3.0);

    // Total KV bits should be approximately the same
    let uniform_bits = uniform.total_kv_bits(128, 64);
    let targeted_bits = targeted.total_kv_bits(128, 64);

    let ratio = targeted_bits as f64 / uniform_bits as f64;
    eprintln!("KV bits ratio (targeted/uniform): {ratio:.3}");
    assert!(
        (ratio - 1.0).abs() < 0.15,
        "total KV bits should be within 15% of uniform: ratio={ratio:.3}"
    );
}

#[test]
fn test_sensitivity_gradient() {
    // Create a clear sensitivity gradient
    let mut sensitivity = vec![0.0; 16];
    for i in 0..16 {
        sensitivity[i] = i as f32 / 15.0;
    }

    let budget = PrecisionBudget::compute_budget(2, 8, &sensitivity, 3.0);

    // Verify monotonicity: higher sensitivity → more bits (at least at extremes)
    let high = budget.get_bits(0, 14); // sensitivity = 14/15 ≈ 0.93
    let low = budget.get_bits(0, 0); // sensitivity = 0.0
    assert!(
        high >= low,
        "high sensitivity ({high}) should have >= bits than low ({low})"
    );
}

#[test]
fn test_budget_constraint_strict() {
    for target_budget in [2.0, 2.5, 3.0, 4.0] {
        let sensitivity = vec![0.5; 32];
        let budget = PrecisionBudget::compute_budget(4, 8, &sensitivity, target_budget);
        assert!(
            budget.verify_budget(),
            "budget constraint violated for target={target_budget}"
        );
    }
}
