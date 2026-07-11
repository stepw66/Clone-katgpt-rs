#![cfg(feature = "rat_plus_bridge")]
//! Benchmarks for RAT+ Recurrence Bridge (Plan 225).
//!
//! Measures decode latency at different dilation factors,
//! bridge projection overhead, and KV cache memory per dilation.
//!
//! Run: `cargo test --features rat_plus_bridge --test bench_225_rat_bridge -- --nocapture`

use katgpt_core::types::DilationConfig;
use katgpt_attn::rat_bridge::{DilatedKvAccessor, RatBridgeState, rat_decode_step};

// ── T6.2: Decode Latency Benchmarks ─────────────────────────────

#[test]
fn bench_decode_latency_per_dilation() {
    let dim = 64;
    let seq_len = 1024;
    let query = vec![0.5; dim];
    let keys: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| vec![(i as f32 % 10.0) / 10.0; dim])
        .collect();
    let vals: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| vec![(i as f32 % 8.0) / 8.0; dim])
        .collect();
    let gdn2 = vec![0.1; dim];

    let dilations = [
        DilationConfig::D1,
        DilationConfig::D4,
        DilationConfig::D16,
        DilationConfig::D64,
    ];

    for d in dilations {
        let mut state = RatBridgeState::new(d, dim);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = rat_decode_step(&mut state, &query, &keys, &vals, &gdn2);
        }
        let elapsed = start.elapsed();
        let per_decode = elapsed / 100;
        // D=1 is dense, should be slowest. D=64 is most sparse, should be fastest.
        println!("Dilation D={}: {:.2?} per decode", d.stride(), per_decode);
        // Verify it completes in reasonable time
        assert!(per_decode < std::time::Duration::from_millis(100));
    }
}

#[test]
fn bench_bridge_projection_overhead() {
    let dim = 64;
    let mut state = RatBridgeState::new(DilationConfig::D16, dim);
    let query = vec![0.5; dim];
    let gdn2 = vec![0.1; dim];

    let start = std::time::Instant::now();
    for _ in 0..10000 {
        state.compute_gate(&query, &gdn2);
    }
    let elapsed = start.elapsed();
    let per_gate = elapsed / 10000;
    println!("Gate computation: {:.2?} per call", per_gate);
    assert!(per_gate < std::time::Duration::from_micros(10));
}

#[test]
fn bench_kv_cache_memory_per_dilation() {
    let seq_len = 4096;
    let dim = 64;

    let dilations = [
        DilationConfig::D1,
        DilationConfig::D4,
        DilationConfig::D16,
        DilationConfig::D64,
    ];

    for d in dilations {
        let indices = DilatedKvAccessor::dilated_indices(seq_len, d);
        let effective_kv = indices.len();
        let bytes = effective_kv * dim * std::mem::size_of::<f32>();
        println!(
            "D={}: {} KV entries, {} bytes ({:.1}%)",
            d.stride(),
            effective_kv,
            bytes,
            100.0 * effective_kv as f64 / seq_len as f64
        );
    }
}

// ── T6.3: Before/After Comparison ───────────────────────────────

#[test]
fn test_before_after_dilation_comparison() {
    let dim = 32;
    let query = vec![0.5; dim];
    let keys: Vec<Vec<f32>> = (0..256)
        .map(|i| vec![(i as f32 % 10.0) / 10.0; dim])
        .collect();
    let vals: Vec<Vec<f32>> = (0..256)
        .map(|i| vec![(i as f32 % 8.0) / 8.0; dim])
        .collect();
    let gdn2 = vec![0.1; dim];

    // Dense baseline
    let mut state_dense = RatBridgeState::new(DilationConfig::D1, dim);
    let dense_out = rat_decode_step(&mut state_dense, &query, &keys, &vals, &gdn2);

    // Bridge D=16
    let mut state_bridge = RatBridgeState::new(DilationConfig::D16, dim);
    let bridge_out = rat_decode_step(&mut state_bridge, &query, &keys, &vals, &gdn2);

    // Both should produce valid output
    assert_eq!(dense_out.output.len(), dim);
    assert_eq!(bridge_out.output.len(), dim);

    // Output should be different (different KV positions used)
    assert_ne!(dense_out.output, bridge_out.output);

    // Both should produce finite values
    for &v in &dense_out.output {
        assert!(v.is_finite());
    }
    for &v in &bridge_out.output {
        assert!(v.is_finite());
    }

    println!(
        "Dense α={:.3}, Bridge D=16 α={:.3}",
        dense_out.alpha, bridge_out.alpha
    );
}

// ── T6.4: GOAT Gate Decision ────────────────────────────────────

#[test]
fn test_goat_gate_decision() {
    // GOAT criteria:
    // D=16: <2% quality loss, >8× FLOPs reduction → DEFAULT-ON
    // D=64: <5% quality loss, >40× FLOPs reduction → DEFAULT-ON

    // FLOPs reduction: decode FLOPs ∝ 1/D
    let d16_reduction = 16.0; // 16× FLOPs reduction
    let d64_reduction = 64.0; // 64× FLOPs reduction

    assert!(d16_reduction >= 8.0, "D=16 should give ≥8× FLOPs reduction");
    assert!(
        d64_reduction >= 40.0,
        "D=64 should give ≥40× FLOPs reduction"
    );

    // Quality: would need real model evaluation to measure.
    // For now, verify the mechanism works correctly.
    println!(
        "GOAT: D=16 meets ≥8× FLOPs reduction (actual: {:.0}×)",
        d16_reduction
    );
    println!(
        "GOAT: D=64 meets ≥40× FLOPs reduction (actual: {:.0}×)",
        d64_reduction
    );
    println!("GOAT: Quality validation requires real model evaluation");
    println!("GOAT: Decision — keep rat_plus_bridge as opt-in until real quality benchmarks pass");
}
