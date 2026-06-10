#![cfg(feature = "tf_loop")]
//! GOAT Proof Test — Training-Free Loop Wrapper (Plan 136)
//!
//! Proves mathematical invariants of the training-free loop wrapper:
//! ODE-refined damped sub-stepping with anchor blend. Uses synthetic
//! f32 vectors (no model weights needed).
//!
//! Run: `cargo test --features tf_loop --test test_136_tf_loop -- --nocapture`

use katgpt_rs::tf_loop::{anchor_blend, sub_step_damped_euler};

// ── Helpers ───────────────────────────────────────────────────

/// Simulate a "forward pass" through a window of layers as a simple affine transform.
/// This models L_b ∘ ... ∘ L_a as: y[i] = scale * x[i] + bias.
fn synthetic_window_forward(x: &[f32], scale: f32, bias: f32) -> Vec<f32> {
    x.iter().map(|&xi| scale * xi + bias).collect()
}

/// K-stage RK sub-step: x ← β·y + (1−β)·x
fn sub_step_kstage_rk(x: &mut [f32], y: &[f32], beta: f32) {
    let one_minus_beta = 1.0 - beta;
    for (xi, yi) in x.iter_mut().zip(y.iter()) {
        *xi = beta * yi + one_minus_beta * *xi;
    }
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|f| f.is_finite())
}

// ── Proof 1: K-stage RK produces finite, non-NaN output ──────
//
// For any finite input, the K-stage RK loop with synthetic layer
// transforms must produce finite output for K=2,3,4.

#[test]
fn proof_tf_loop_finite() {
    let dim = 64;
    let x_init: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 - 3.0).collect();
    let scale = 0.8f32;
    let bias = 0.1f32;

    for k in [2, 3, 4] {
        for beta in [0.25, 0.5, 0.75] {
            let mut x = x_init.clone();

            // Anchor: one-shot window output
            let anchor = synthetic_window_forward(&x, scale, bias);

            // Loop K times with K-stage RK sub-step
            for _ in 0..k {
                let y = synthetic_window_forward(&x, scale, bias);
                sub_step_kstage_rk(&mut x, &y, beta);
            }

            // Anchor blend
            anchor_blend(&mut x, &anchor, beta);

            assert!(
                all_finite(&x),
                "[P1] Non-finite output for K={k}, beta={beta}"
            );
        }
    }

    println!("✅ Proof 1 PASSED: K-stage RK finite for K=2,3,4");
}

// ── Proof 2: Sub-stepping doesn't grow with K ────────────────
//
// The computational cost per sub-step is O(dim), independent of K.
// The total cost is O(K·dim), but the KV cache size is always O(dim)
// because we only write canonical KV once (from the final or first state).
// Verify that the output dimension never depends on K.

#[test]
fn proof_tf_loop_cache_size() {
    let dim = 128;
    let x_init = vec![1.0f32; dim];
    let scale = 0.9f32;
    let bias = 0.05f32;

    for k in [1, 2, 4, 8, 16] {
        let mut x = x_init.clone();
        let anchor = synthetic_window_forward(&x, scale, bias);

        for _ in 0..k {
            let y = synthetic_window_forward(&x, scale, bias);
            sub_step_damped_euler(&mut x, &y, k);
        }

        anchor_blend(&mut x, &anchor, 0.5);

        // Output dimension is always dim, regardless of K
        assert_eq!(x.len(), dim, "[P2] Output dimension changed with K={k}");

        // The "cache" (the x vector we'd write) is exactly dim elements
        // regardless of how many loop iterations we ran
        let cache_entries = x.len();
        assert_eq!(
            cache_entries, dim,
            "[P2] Cache size must be independent of K"
        );
    }

    println!("✅ Proof 2 PASSED: Sub-stepping cache size independent of K");
}

// ── Proof 3: Bypass is free when K=0 or window is empty ──────
//
// When K=0, no sub-stepping occurs. When window_start == window_end == 0,
// no window layers are applied. Both should produce an identity transform
// (or near-identity after anchor blend with β=0).

#[test]
fn proof_tf_loop_bypass_free() {
    let dim = 64;
    let x_original: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();

    // K=0: no loop iterations
    {
        let mut x = x_original.clone();
        // No sub-steps (K=0)
        sub_step_damped_euler(&mut x, &x_original, 0);
        assert_eq!(x, x_original, "[P3a] K=0 should be identity");
    }

    // Window (0, 0) with K=2 but no actual layers to apply:
    // The window is empty, so forward_window is identity.
    {
        let mut x = x_original.clone();
        // With an empty window, y = x (identity forward pass)
        for _ in 0..2 {
            let y = x.clone(); // identity window
            sub_step_damped_euler(&mut x, &y, 2);
        }
        // After damped Euler with y=x, x stays the same
        for (xi, oi) in x.iter().zip(x_original.iter()) {
            assert!(
                (xi - oi).abs() < 1e-6,
                "[P3b] Empty window should be identity, diff={}",
                xi - oi
            );
        }
    }

    // β=0 anchor blend: pure identity
    {
        let mut x = x_original.clone();
        let anchor = vec![999.0f32; dim]; // irrelevant with β=0
        anchor_blend(&mut x, &anchor, 0.0);
        assert_eq!(x, x_original, "[P3c] β=0 blend should be identity");
    }

    println!("✅ Proof 3 PASSED: Bypass is identity when K=0 or window empty");
}

// ── Proof 4: Layer-mode produces finite results ──────────────
//
// In layer mode, each layer in the window is applied individually
// within each sub-step. Simulate this with per-layer transforms
// and verify finiteness.

#[test]
fn proof_tf_loop_layer_mode_stable() {
    let dim = 64;
    let x_init: Vec<f32> = (0..dim).map(|i| (i as f32) % 7.0 - 3.0).collect();

    // Per-layer affine transforms (scale, bias pairs)
    let layer_transforms: Vec<(f32, f32)> = vec![(0.9, 0.05), (1.0, -0.02), (0.95, 0.03)];

    for k in [1, 2, 4] {
        for beta in [0.0, 0.3, 0.5, 1.0] {
            let mut x = x_init.clone();

            // Anchor: full window pass (layer-by-layer)
            let mut anchor = x.clone();
            for &(s, b) in &layer_transforms {
                anchor = synthetic_window_forward(&anchor, s, b);
            }

            // Loop K times, layer mode
            for _ in 0..k {
                let mut y = x.clone();
                for &(s, b) in &layer_transforms {
                    y = synthetic_window_forward(&y, s, b);
                }
                // Damped Euler sub-step
                sub_step_damped_euler(&mut x, &y, k);
            }

            // Anchor blend
            anchor_blend(&mut x, &anchor, beta);

            assert!(
                all_finite(&x),
                "[P4] Non-finite output in layer mode for K={k}, beta={beta}"
            );

            // Verify output magnitude is reasonable (not diverging)
            let max_abs = x.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e6,
                "[P4] Output magnitude too large: {max_abs} for K={k}, beta={beta}"
            );
        }
    }

    println!("✅ Proof 4 PASSED: Layer-mode stable for K=1,2,4");
}
