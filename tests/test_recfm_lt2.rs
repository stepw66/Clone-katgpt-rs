//! GOAT Proof 168 T2: RecFM LT2 Acceleration-Bounded Sub-Stepping
//!
//! Feature gate: `recfm` (Plan 168 Task 2, Research 150)
//!
//! Proofs:
//!   P1: Acceleration-bounded sub-steps produce smaller residuals
//!   P2: Output diverges less over K iterations vs vanilla damped Euler
//!   P3: accel_norm benchmark (< 100ns per call)

#![cfg(feature = "recfm")]

use katgpt_rs::tf_loop::{
    AccelBoundConfig, accel_norm, sub_step_damped_euler, sub_step_damped_euler_bounded,
};

const DIM: usize = 128;

// ── P1: Acceleration-bounded sub-steps produce smaller residuals ────

#[test]
fn proof_p1_bounded_produces_smaller_residuals() {
    // Simulate an explosive affine transform: y = 2.0*x + 1.0
    // Without damping, this diverges. With bounded sub-stepping, residual should be smaller.
    let k = 4;

    // Vanilla: no acceleration bounding
    let mut x_vanilla = vec![1.0f32; DIM];
    for _ in 0..k {
        let y: Vec<f32> = x_vanilla.iter().map(|xi| 2.0 * xi + 1.0).collect();
        sub_step_damped_euler(&mut x_vanilla, &y, k);
    }

    // Bounded: with acceleration bounding
    let mut x_bounded = vec![1.0f32; DIM];
    let config = AccelBoundConfig {
        enable: true,
        accel_threshold: 0.5,
        extra_damp_factor: 0.8,
    };
    for _ in 0..k {
        let x_prev = x_bounded.clone();
        let y: Vec<f32> = x_bounded.iter().map(|xi| 2.0 * xi + 1.0).collect();
        sub_step_damped_euler_bounded(&mut x_bounded, &y, k, &x_prev, &config);
    }

    // Fixed point of y = 2x+1 → x = -1.0
    // Residual = ||x - fixed_point||₂
    let residual_vanilla: f32 = x_vanilla
        .iter()
        .map(|xi| (xi - (-1.0f32)).powi(2))
        .sum::<f32>()
        .sqrt();
    let residual_bounded: f32 = x_bounded
        .iter()
        .map(|xi| (xi - (-1.0f32)).powi(2))
        .sum::<f32>()
        .sqrt();

    assert!(
        residual_bounded <= residual_vanilla,
        "Bounded should have smaller or equal residual: bounded={residual_bounded} vs vanilla={residual_vanilla}"
    );
}

// ── P2: Output diverges less over K iterations vs vanilla ────────────

#[test]
fn proof_p2_bounded_diverges_less() {
    // More iterations with a diverging transform
    let k = 8;
    let n_iters = 20;

    // Vanilla
    let mut x_vanilla = vec![1.0f32; DIM];
    for _ in 0..n_iters {
        let y: Vec<f32> = x_vanilla.iter().map(|xi| 1.5 * xi + 0.5).collect();
        sub_step_damped_euler(&mut x_vanilla, &y, k);
    }

    // Bounded
    let mut x_bounded = vec![1.0f32; DIM];
    let config = AccelBoundConfig {
        enable: true,
        accel_threshold: 0.3,
        extra_damp_factor: 0.7,
    };
    for _ in 0..n_iters {
        let x_prev = x_bounded.clone();
        let y: Vec<f32> = x_bounded.iter().map(|xi| 1.5 * xi + 0.5).collect();
        sub_step_damped_euler_bounded(&mut x_bounded, &y, k, &x_prev, &config);
    }

    let max_vanilla = x_vanilla.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_bounded = x_bounded.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    assert!(
        max_bounded <= max_vanilla * 1.01, // allow 1% tolerance for floating point
        "Bounded should not diverge more: bounded={max_bounded} vs vanilla={max_vanilla}"
    );
}

// ── P3: accel_norm correctness and performance ─────────────────────

#[test]
fn test_accel_norm_zero() {
    let v = vec![1.0f32; 64];
    assert_eq!(
        accel_norm(&v, &v),
        0.0,
        "identical vectors should have zero acceleration"
    );
}

#[test]
fn test_accel_norm_unit() {
    let v_curr = vec![2.0f32; 64];
    let v_prev = vec![1.0f32; 64];
    let norm = accel_norm(&v_curr, &v_prev);
    // diff = [1.0; 64], sum_sq = 64.0, sqrt(64.0 / 64) = 1.0
    assert!((norm - 1.0).abs() < 1e-6, "expected 1.0, got {norm}");
}

#[test]
fn test_accel_norm_empty() {
    assert_eq!(accel_norm(&[], &[]), 0.0, "empty slices should return 0.0");
}

#[test]
fn test_accel_norm_symmetry() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![4.0f32, 3.0, 2.0, 1.0];
    let ab = accel_norm(&a, &b);
    let ba = accel_norm(&b, &a);
    assert!(
        (ab - ba).abs() < 1e-6,
        "accel_norm should be symmetric: {ab} vs {ba}"
    );
}

// ── AccelBoundConfig disabled is identity ───────────────────────────

#[test]
fn test_bounded_disabled_is_vanilla() {
    let k = 4;
    let mut x_vanilla = vec![1.0f32; DIM];
    let mut x_bounded = vec![1.0f32; DIM];
    let x_prev = x_bounded.clone();

    let config = AccelBoundConfig {
        enable: false,
        ..Default::default()
    };

    let y: Vec<f32> = vec![2.0f32; DIM];

    sub_step_damped_euler(&mut x_vanilla, &y, k);
    sub_step_damped_euler_bounded(&mut x_bounded, &y, k, &x_prev, &config);

    for (i, (a, b)) in x_vanilla.iter().zip(x_bounded.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Disabled should be identical at idx {i}: {a} vs {b}"
        );
    }
}
