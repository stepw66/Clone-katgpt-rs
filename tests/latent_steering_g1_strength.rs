//! Latent Field Steering — G1 steering strength gate (Plan 309 T2.1).
//!
//! ## Hypothesis (Research 290 §1.3)
//!
//! Injecting a unit-norm direction aligned with the HLA `fear` axis (index 4)
//! at strength α=0.5 produces a measurable, targeted affect shift: the fear-axis
//! projection increases by ≥30%, while non-target axes are unaffected.
//!
//! ## Setup
//!
//! - Baseline 8-dim state with fear-axis projection = 1.0 (moderate baseline).
//! - Steering vector: one-hot at HLA_FEAR (index 4), unit-norm (trivially).
//! - α = 0.5.
//! - Measure fear-axis projection before and after.
//!
//! ## Gate
//!
//! - **PASS:** post/pre ≥ 1.30 (≥30% relative shift on the target axis) AND
//!   non-target axes unchanged (|delta| < 1e-5).
//! - **KILL:** post/pre < 1.10 — steering is too weak to matter.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features latent_field_steering --release \
//!   --test latent_steering_g1_strength -- --nocapture
//! ```

#![cfg(feature = "latent_field_steering")]

use katgpt_core::latent_steering::{HLA_FEAR, LatentSteeringVector, apply_latent_steering};

const ALPHA: f32 = 0.5;
/// Required relative shift on the target axis: post/pre ≥ 1.30.
const TARGET_SHIFT_RATIO: f32 = 1.30;

#[test]
fn g1_steering_strength() {
    // ── Build the steering vector: one-hot at fear axis ────────────────
    let mut direction = vec![0.0f32; 8];
    direction[HLA_FEAR] = 1.0; // unit-norm by construction
    let steering = LatentSteeringVector::new(direction, ALPHA, 1e-5).unwrap();

    // ── Baseline state: fear = 1.0, other axes = small spread ──────────
    let baseline = vec![0.2f32, 0.3, -0.1, 0.4, 1.0, 0.0, 0.0, 0.0];
    let mut state = baseline.clone();

    // ── Apply steering ──────────────────────────────────────────────────
    apply_latent_steering(&mut state, &steering);

    // ── Gate 1: target-axis shift ≥ 30% ────────────────────────────────
    let fear_pre = baseline[HLA_FEAR];
    let fear_post = state[HLA_FEAR];
    let ratio = fear_post / fear_pre;
    println!(
        "G1 fear-axis: pre={fear_pre:.4} post={fear_post:.4} ratio={ratio:.4} \
         (gate ≥ {TARGET_SHIFT_RATIO})"
    );
    assert!(
        ratio >= TARGET_SHIFT_RATIO,
        "G1 FAIL: fear-axis shift ratio {ratio:.4} < {TARGET_SHIFT_RATIO}. Steering is too weak."
    );

    // ── Gate 2: non-target axes unchanged (|delta| < 1e-5) ─────────────
    // The steering direction is one-hot at fear, so ALL other axes must be
    // untouched. This verifies targeted affect shift — no leakage to other
    // emotional axes.
    for i in 0..8 {
        if i == HLA_FEAR {
            continue;
        }
        let delta = state[i] - baseline[i];
        assert!(
            delta.abs() < 1e-5,
            "G1 FAIL: non-target axis {i} shifted by {delta:.6} — steering leaked"
        );
    }

    // ── Verify the shift magnitude matches α exactly ───────────────────
    // s_fear' = s_fear + α · v_fear = 1.0 + 0.5 · 1.0 = 1.5
    assert!(
        (fear_post - 1.5).abs() < 1e-5,
        "G1 FAIL: fear_post={fear_post} expected 1.5 (baseline 1.0 + α 0.5)"
    );

    println!("G1 PASS: fear-axis shifted {ratio:.2}×, non-target axes unchanged.");
}
