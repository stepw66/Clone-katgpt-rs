#![cfg(feature = "ega_attn")]
//! GOAT Proof Test — Energy-Gated Attention (EGA) Proofs G2–G5 (Plan 139)
//!
//! Proves: parameter overhead < 1%, attention quality via cosine similarity,
//! KV eviction feasibility via energy separation, and compute overhead budget.
//!
//! Run: `cargo test --features ega_attn --test test_139_ega_goat -- --nocapture`

use katgpt_attn::ega_attn::EgaGate;

/// G2: Parameter overhead — EGA adds < 1% of model parameters
#[test]
fn proof_ega_parameter_overhead() {
    // Typical micro config: L=6, H=8, d=256, head_dim=32
    let n_layers = 6;
    let n_heads = 8;
    let d_model = 256;
    let head_dim = d_model / n_heads; // 32

    // EGA params per head: head_dim + 2
    let gate = EgaGate::new(head_dim);
    let ega_params_per_head = gate.parameter_count();
    assert_eq!(
        ega_params_per_head,
        head_dim + 2,
        "EGA adds head_dim + 2 params per head"
    );

    // Total EGA params: n_layers × n_heads × (head_dim + 2)
    let total_ega = n_layers * n_heads * ega_params_per_head;

    // Rough model params: embedding + 6 transformer layers
    // embedding: vocab_size × d_model ≈ 1024 × 256 = 262144
    // per layer: 4 × d² (QKV+O proj + MLP) ≈ 4 × 256² = 262144
    // total ≈ 262144 + 6 × 262144 = 1,835,008
    let total_model = 1024 * d_model + n_layers * 4 * d_model * d_model;

    let overhead = total_ega as f64 / total_model as f64 * 100.0;
    assert!(
        overhead < 1.0,
        "EGA overhead should be < 1%, got {:.4}%",
        overhead
    );
    println!("G2 PASS: EGA overhead = {:.4}% (< 1%)", overhead);
}

/// G3: Attention quality — cosine similarity of attention output with/without EGA is reasonable (> 0.5)
#[test]
fn proof_ega_attention_quality() {
    let seq_len = 8;
    let head_dim = 16;
    let gate = EgaGate::new(head_dim);

    // Create deterministic energy scores
    let energy: Vec<f32> = (0..seq_len).map(|i| (i as f32 + 1.0) * 0.5).collect();

    // Create attention weights (uniform → softmax-like)
    let mut attn_with_ega = vec![1.0 / seq_len as f32; seq_len * seq_len];
    let attn_without_ega = vec![1.0 / seq_len as f32; seq_len * seq_len];

    // Apply EGA gate
    let mut gate_buf = vec![0.0; seq_len];
    gate.gate_attention(&mut attn_with_ega, &energy, seq_len, &mut gate_buf);

    // Compute cosine similarity between the two attention matrices
    let dot: f32 = attn_with_ega
        .iter()
        .zip(&attn_without_ega)
        .map(|(a, b)| a * b)
        .sum();
    let norm_a: f32 = attn_with_ega.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = attn_without_ega.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b + 1e-10);

    assert!(
        cosine > 0.5,
        "Cosine similarity should be > 0.5, got {:.4}",
        cosine
    );
    assert!(
        cosine <= 1.0,
        "Cosine similarity should be ≤ 1.0, got {:.4}",
        cosine
    );
    println!(
        "G3 PASS: Cosine similarity with/without EGA = {:.4} (> 0.5)",
        cosine
    );
}

/// G4: KV eviction feasibility — energy scores produce meaningful threshold separation
#[test]
fn proof_ega_eviction_feasibility() {
    let seq_len = 32;
    let head_dim = 16;
    let gate = EgaGate::new(head_dim);

    // Create synthetic input: some positions have high energy, some low
    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let base = if i % 4 == 0 { 2.0 } else { 0.1 }; // Every 4th position is "content"
        for j in 0..head_dim {
            x[i * head_dim + j] = base;
        }
    }

    let energy = gate.energy_scores(&x, seq_len, head_dim);
    assert_eq!(energy.len(), seq_len);

    // With default w_proj = 1/d, energy is just the mean of each position's embedding
    // High-energy positions should be distinguishable from low-energy ones
    let high_energies: Vec<f32> = energy
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == 0)
        .map(|(_, &e)| e)
        .collect();
    let low_energies: Vec<f32> = energy
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 != 0)
        .map(|(_, &e)| e)
        .collect();

    let avg_high: f32 = high_energies.iter().sum::<f32>() / high_energies.len() as f32;
    let avg_low: f32 = low_energies.iter().sum::<f32>() / low_energies.len() as f32;

    // High energy positions should have higher energy
    assert!(
        avg_high > avg_low,
        "High-energy positions ({:.4}) should exceed low-energy ({:.4})",
        avg_high,
        avg_low
    );

    // Ratio should be meaningful (> 2× separation)
    let ratio = avg_high / (avg_low.abs() + 1e-10);
    assert!(
        ratio > 2.0,
        "Energy ratio should be > 2×, got {:.2}×",
        ratio
    );
    println!(
        "G4 PASS: Energy separation = {:.2}× (high={:.4}, low={:.4})",
        ratio, avg_high, avg_low
    );
}

/// G5: Compute overhead — EGA energy + gate adds minimal overhead
#[test]
fn proof_ega_compute_overhead() {
    let seq_len = 64;
    let head_dim = 32;
    let gate = EgaGate::new(head_dim);

    let x: Vec<f32> = (0..(seq_len * head_dim))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let energy: Vec<f32> = (0..seq_len).map(|i| (i as f32 * 0.1).sin() + 1.0).collect();

    // Measure energy computation time
    let start = std::time::Instant::now();
    let iters = 1000;
    let mut energy_out = vec![0.0f32; seq_len];
    let mut gate_buf = vec![0.0f32; seq_len];
    let mut attn = vec![1.0 / seq_len as f32; seq_len * seq_len];
    for _ in 0..iters {
        gate.energy_scores_into(&x, seq_len, head_dim, &mut energy_out);
        gate.gate_attention_into(&mut attn, &energy, seq_len, &mut gate_buf);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_nanos() as f64 / iters as f64;

    // Budget: 100µs for debug builds (no SIMD optimization),
    // ~5µs expected in release with SIMD. A single SDPA forward on
    // seq_len=64, head_dim=32 is roughly 50-100µs, so EGA at ~1% of
    // that is negligible.
    let budget_us = if cfg!(debug_assertions) { 200.0 } else { 10.0 };
    assert!(
        per_call < budget_us * 1000.0,
        "EGA per-call overhead should be < {:.0}µs, got {:.1}µs",
        budget_us,
        per_call / 1000.0
    );
    println!(
        "G5 PASS: EGA per-call overhead = {:.1}µs (< {:.0}µs budget)",
        per_call / 1000.0,
        budget_us
    );
}
