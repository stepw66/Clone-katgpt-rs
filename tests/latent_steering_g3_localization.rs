//! Latent Field Steering — G3 localization gate (Plan 309 T2.3).
//!
//! ## Hypothesis (Research 290 §5.2)
//!
//! A radius-banded field (`sigmoid((bandwidth - distance) · steepness)` kernel)
//! applies near-full strength inside the bandwidth and near-zero outside. The
//! leakage to entities outside the support must be negligible.
//!
//! ## Setup
//!
//! - Radius field at center (0,0), bandwidth=10, steepness=2.0.
//! - 500 NPCs inside at distance 5 (well within bandwidth).
//! - 500 NPCs outside at distance 15 (well outside bandwidth).
//! - Apply field, measure per-NPC shift magnitude.
//!
//! ## Gate
//!
//! - **PASS:** mean_outside_shift / mean_inside_shift < 0.01 (≤1% leakage).
//! - **KILL:** ratio > 0.05 — field propagates uncontrollably.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features latent_field_steering --release \
//!   --test latent_steering_g3_localization -- --nocapture
//! ```

#![cfg(feature = "latent_field_steering")]

use katgpt_core::latent_steering::{
    FieldSupport, LatentField, LatentSteeringVector, apply_field_to_crowd,
};

const D: usize = 8;
const N_INSIDE: usize = 500;
const N_OUTSIDE: usize = 500;
const N_TOTAL: usize = N_INSIDE + N_OUTSIDE;

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_f32(&mut self) -> f32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        let bits = (self.s >> 11) as u32;
        bits as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

#[test]
fn g3_localization() {
    let mut rng = Rng::new(0x10C);

    // ── Random unit steering direction ─────────────────────────────────
    let mut dir: Vec<f32> = (0..D).map(|_| rng.next_f32()).collect();
    let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut dir {
        *x /= norm;
    }
    let steering = LatentSteeringVector::new(dir, 0.5, 1e-4).unwrap();
    let field = LatentField {
        steering,
        support: FieldSupport::Radius {
            center: [0.0, 0.0],
            bandwidth: 10.0,
            steepness: 2.0,
        },
    };

    // ── Build crowd: inside at d≈5, outside at d≈15 ────────────────────
    let mut states = vec![0.0f32; N_TOTAL * D];
    let positions: Vec<Option<[f32; 2]>> = (0..N_TOTAL)
        .map(|i| {
            if i < N_INSIDE {
                // Inside: distance ≈ 5 from origin (spread on a ring).
                let angle = (i as f32) * 0.1;
                Some([5.0 * angle.cos(), 5.0 * angle.sin()])
            } else {
                // Outside: distance ≈ 15 from origin.
                let angle = ((i - N_INSIDE) as f32) * 0.1;
                Some([15.0 * angle.cos(), 15.0 * angle.sin()])
            }
        })
        .collect();
    let zones = vec![None; N_TOTAL];

    // ── Apply ───────────────────────────────────────────────────────────
    apply_field_to_crowd(&mut states, D, &positions, &zones, &field);

    // ── Measure per-NPC shift magnitude ────────────────────────────────
    // Since baseline states are all 0.0, the shift IS the final state.
    let inside_shift_sum: f32 = states[..N_INSIDE * D].iter().map(|x| x.abs()).sum();
    let outside_shift_sum: f32 = states[N_INSIDE * D..].iter().map(|x| x.abs()).sum();
    let mean_inside = inside_shift_sum / (N_INSIDE * D) as f32;
    let mean_outside = outside_shift_sum / (N_OUTSIDE * D) as f32;

    let ratio = if mean_inside > 1e-12 {
        mean_outside / mean_inside
    } else {
        f32::INFINITY
    };

    // sigmoid((10-5)*2)  = sigmoid(10)  ≈ 0.99995  → inside gets ~full strength
    // sigmoid((10-15)*2) = sigmoid(-10) ≈ 4.5e-5   → outside gets ~0
    println!(
        "G3 localization: mean_inside_shift={mean_inside:.6}, mean_outside_shift={mean_outside:.9}, \
         ratio={ratio:.6} (gate < 0.01)"
    );
    println!(
        "  (expected: inside kernel ≈ {:.5}, outside kernel ≈ {:.2e})",
        1.0 / (1.0 + (-10.0f32).exp()),
        1.0 / (1.0 + 10.0f32.exp())
    );

    assert!(
        ratio < 0.01,
        "G3 FAIL: leakage ratio {ratio:.6} ≥ 0.01 — field propagates outside its support"
    );
    assert!(
        mean_inside > 0.1,
        "G3 FAIL: inside shift {mean_inside:.6} too small — field not applying inside bandwidth"
    );
    println!("G3 PASS: leakage ratio {ratio:.6} < 0.01 (negligible outside-support propagation).");
}
