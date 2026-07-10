#![cfg(feature = "rat_plus_bridge")]
//! GOAT Proof Test — RAT+ Recurrence Bridge (Plan 225)
//!
//! Proves invariants of the RAT+ bridge decode pipeline:
//! - Bridge attention output dims match dense attention
//! - Sigmoid gate produces valid [0, 1] range with expected saturation
//! - Dilation D=1 matches dense (all KV positions used)
//! - DilatedKV accessor stride correctness
//! - DilationBridgeRouter centroid computation and scoring
//!
//! Run: `cargo test --features rat_plus_bridge --test goat_225_rat_bridge -- --nocapture`

use katgpt_core::types::DilationConfig;
use katgpt_attn::rat_bridge::{
    DilatedKvAccessor, DilationBridgeRouter, RatBridgeState, bridge_attention, rat_decode_step,
};

// ── T6.1: Bridge Attention Output Dims ─────────────────────────

#[test]
fn test_bridge_attention_output_dims_match_dense() {
    // D=1 should produce same dims as dense attention
    let dim = 8;
    let query = vec![0.5; dim];
    let keys: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32 / 8.0; dim]).collect();
    let vals: Vec<Vec<f32>> = (0..8).map(|i| vec![(i + 1) as f32 / 16.0; dim]).collect();
    let gdn2 = vec![0.1; dim];

    let mut state = RatBridgeState::new(DilationConfig::D1, dim);
    let out = rat_decode_step(&mut state, &query, &keys, &vals, &gdn2);
    assert_eq!(out.output.len(), dim);
}

// ── T6.1: Sigmoid Gate Valid Range ─────────────────────────────

#[test]
fn test_sigmoid_gate_valid_range() {
    let mut state = RatBridgeState::new(DilationConfig::D4, 8);

    // High positive dot product → gate close to 1.0
    let query = vec![5.0; 8];
    let gdn2 = vec![5.0; 8];
    state.compute_gate(&query, &gdn2);
    assert!((0.0..=1.0).contains(&state.alpha));
    assert!(state.alpha > 0.99); // sigmoid(200) ≈ 1.0

    // High negative dot product → gate close to 0.0
    let query2 = vec![-5.0; 8];
    state.compute_gate(&query2, &gdn2);
    assert!(state.alpha < 0.01); // sigmoid(-200) ≈ 0.0
}

#[test]
fn test_sigmoid_gate_at_zero_is_half() {
    let mut state = RatBridgeState::new(DilationConfig::D1, 4);
    // Orthogonal vectors → dot=0 → sigmoid(0)=0.5
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let gdn2 = vec![0.0, 1.0, 0.0, 0.0];
    state.compute_gate(&query, &gdn2);
    assert!((state.alpha - 0.5).abs() < 1e-6);
}

// ── T6.1: Dilation D=1 Matches Dense ───────────────────────────

#[test]
fn test_dilation_d1_matches_dense() {
    // D=1 should use all KV positions (equivalent to dense)
    let indices = DilatedKvAccessor::dilated_indices(16, DilationConfig::D1);
    assert_eq!(indices.len(), 16);
    assert_eq!(indices, (0..16).collect::<Vec<_>>());
}

#[test]
fn test_dilation_d4_accesses_every_4th() {
    let indices = DilatedKvAccessor::dilated_indices(16, DilationConfig::D4);
    assert_eq!(indices, vec![0, 4, 8, 12]);
}

#[test]
fn test_dilation_d16_accesses_every_16th() {
    let indices = DilatedKvAccessor::dilated_indices(64, DilationConfig::D16);
    assert_eq!(indices, vec![0, 16, 32, 48]);
}

// ── T6.1: DilatedKV Accessor Correctness ───────────────────────

#[test]
fn test_dilated_len_matches_indices() {
    for d in [
        DilationConfig::D1,
        DilationConfig::D2,
        DilationConfig::D4,
        DilationConfig::D8,
        DilationConfig::D16,
        DilationConfig::D32,
        DilationConfig::D64,
    ] {
        for len in [1, 8, 16, 32, 64, 65, 100, 128] {
            let indices = DilatedKvAccessor::dilated_indices(len, d);
            let expected_len = DilatedKvAccessor::dilated_len(len, d);
            assert_eq!(
                indices.len(),
                expected_len,
                "mismatch at len={len}, dilation={d:?}"
            );
        }
    }
}

// ── T6.1: Bridge Attention With Empty KV ───────────────────────

#[test]
fn test_bridge_attention_empty_kv_uses_bridge_readout() {
    let dim = 4;
    let query = vec![1.0; dim];
    let keys: Vec<Vec<f32>> = vec![];
    let vals: Vec<Vec<f32>> = vec![];
    let gdn2 = vec![0.5; dim];

    let out = bridge_attention(&query, &keys, &vals, &gdn2, 0.5);
    // With empty KV: attn_output=0, output = 0.5*0 + 0.5*bridge
    // bridge = S·q = [0.5, 0.5, 0.5, 0.5]
    // output = 0.5 * 0 + 0.5 * 0.5 = 0.25
    for &v in &out.output {
        assert!((v - 0.25).abs() < 1e-6);
    }
}

// ── T6.1: DilationBridgeRouter ─────────────────────────────────

#[test]
fn test_dilation_bridge_router_centroid_count() {
    let keys: Vec<Vec<f32>> = (0..16).map(|i| vec![i as f32; 4]).collect();

    let mut d1 = DilationBridgeRouter::new(DilationConfig::D1, 4);
    d1.compute_centroids(&keys, 4);
    // D1 with 16 keys, block_size=4 → 4 centroids
    assert_eq!(d1.centroids.len(), 4);

    let mut d4 = DilationBridgeRouter::new(DilationConfig::D4, 4);
    d4.compute_centroids(&keys, 4);
    // D4 with 16 keys → 4 dilated indices [0,4,8,12], block_size=4 → 1 centroid
    assert_eq!(d4.centroids.len(), 1);
}

#[test]
fn test_dilation_bridge_router_score_ordering() {
    let mut router = DilationBridgeRouter::new(DilationConfig::D1, 4);
    // Centroid 0: aligned with query, centroid 1: orthogonal
    router.centroids = vec![vec![1.0; 4], vec![0.0; 4]];
    let query = vec![1.0; 4];
    let gdn2 = vec![0.5; 4];
    let scores = router.score_blocks(&query, &gdn2);

    assert_eq!(scores.len(), 2);
    assert!(
        scores[0].1 > scores[1].1,
        "aligned centroid should score higher"
    );
}

#[test]
fn test_rat_decode_step_all_dilations_produce_valid_output() {
    let dim = 8;
    let query = vec![0.5; dim];
    let keys: Vec<Vec<f32>> = (0..32).map(|i| vec![i as f32 / 32.0; dim]).collect();
    let vals: Vec<Vec<f32>> = (0..32).map(|i| vec![(i + 1) as f32 / 64.0; dim]).collect();
    let gdn2 = vec![0.1; dim];

    for d in [
        DilationConfig::D1,
        DilationConfig::D2,
        DilationConfig::D4,
        DilationConfig::D8,
        DilationConfig::D16,
        DilationConfig::D32,
        DilationConfig::D64,
    ] {
        let mut state = RatBridgeState::new(d, dim);
        let out = rat_decode_step(&mut state, &query, &keys, &vals, &gdn2);
        assert_eq!(
            out.output.len(),
            dim,
            "output dim mismatch at dilation {d:?}"
        );
        assert!(
            (0.0..=1.0).contains(&out.alpha),
            "alpha out of range at dilation {d:?}"
        );
    }
}
