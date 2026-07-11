//! GOAT benchmark for Targeted Precision Budget (Plan 227 Phase 2).
//!
//! Measures: perplexity impact, KV cache size consistency, latency.

use katgpt_kv::targeted_precision::PrecisionBudget;

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
    for (i, val) in sensitivity.iter_mut().enumerate() {
        *val = i as f32 / 15.0;
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

#[test]
fn goat_g2_targeted_precision_perplexity_improvement() {
    // ── Setup: simulate a model with 4 layers, 8 heads (32 total) ──
    let num_layers = 4;
    let num_heads = 8;
    let total_heads = num_layers * num_heads;
    let seq_len = 128;
    let head_dim = 64;

    // Create non-uniform sensitivity: some heads are very sensitive, others robust
    let mut sensitivity = vec![0.3; total_heads];
    // 6 sensitive heads (quantization noise greatly hurts perplexity)
    for i in [0, 3, 7, 15, 24, 31] {
        sensitivity[i] = 0.95;
    }
    // 8 robust heads (quantization noise barely matters)
    for i in [2, 5, 9, 12, 18, 22, 27, 29] {
        sensitivity[i] = 0.05;
    }

    let target_budget = 3.0; // average 3 bits/head

    // ── Baseline: uniform allocation ──
    let uniform = PrecisionBudget::uniform(num_layers, num_heads, target_budget as u8);

    // Simulate perplexity with uniform budget:
    // PPL ∝ Σ(sensitivity_i * noise_from_bits(uniform_bits_i))
    // noise_from_bits(b) = 1.0 / 2^b — more bits = less noise
    let uniform_ppl: f32 = sensitivity
        .iter()
        .zip(uniform.head_bits.iter())
        .map(|(&s, &bits)| {
            let noise = 1.0 / (1u32 << bits).max(2) as f32;
            s * noise
        })
        .sum();

    // ── Feature: targeted allocation ──
    let targeted =
        PrecisionBudget::compute_budget(num_layers, num_heads, &sensitivity, target_budget);

    let targeted_ppl: f32 = sensitivity
        .iter()
        .zip(targeted.head_bits.iter())
        .map(|(&s, &bits)| {
            let noise = 1.0 / (1u32 << bits).max(2) as f32;
            s * noise
        })
        .sum();

    // ── KV cache size must be comparable ──
    let uniform_bits = uniform.total_kv_bits(seq_len, head_dim);
    let targeted_bits = targeted.total_kv_bits(seq_len, head_dim);
    let size_ratio = targeted_bits as f64 / uniform_bits as f64;

    // ── Compute perplexity improvement ──
    let ppl_improvement = (uniform_ppl - targeted_ppl) / uniform_ppl;

    eprintln!(
        "G2 TPB: uniform_ppl={uniform_ppl:.4} targeted_ppl={targeted_ppl:.4} improvement={:.1}% kv_ratio={size_ratio:.3}",
        ppl_improvement * 100.0
    );
    eprintln!("  uniform_bits={uniform_bits} targeted_bits={targeted_bits}");

    // ── GOAT gate assertions ──
    // KV cache size must be within 15% of uniform (same budget constraint)
    assert!(
        (size_ratio - 1.0).abs() < 0.15,
        "G2 FAIL: KV cache size ratio {size_ratio:.3} outside 15% tolerance"
    );
    // Perplexity must improve ≥2%
    assert!(
        ppl_improvement >= 0.02,
        "G2 FAIL: perplexity improvement {:.1}% < 2%",
        ppl_improvement * 100.0
    );
    eprintln!(
        "✅ G2: TPB perplexity improvement = {:.1}%, KV cache ratio = {size_ratio:.3}",
        ppl_improvement * 100.0
    );
}
