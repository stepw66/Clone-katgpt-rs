//! Latent Field Steering — G2 behavior rank preservation gate (Plan 309 T2.2).
//!
//! **THE headline gate.** If this fails, the primitive is dangerous — steering
//! corrupts NPC decision-making and must be demoted to research-only.
//!
//! ## Hypothesis (Research 290 §5.1)
//!
//! Adding `α · v` to latent state changes action scores by a constant offset
//! `c = α · W^T · v` (same for all entities). If `||c||` is small relative to
//! `||scores||`, the cosine similarity of pre/post score vectors stays high.
//! At α=0.3 with a random unit direction, the shift is moderate enough that
//! mean cos ≥ 0.95 — rankings are preserved.
//!
//! ## Setup
//!
//! - 100 random 8-dim latent states (entries U(-1, 1) — realistic HLA range).
//! - Fixed 8×5 action weight matrix (random, fixed seed).
//! - Action scoring: `scores = W^T · state` (5 candidate actions).
//! - Steering: random unit direction at α ∈ {0.1, 0.3, 0.5, 0.9}.
//! - Measure cosine similarity of pre/post score vectors + argmax stability.
//!
//! ## Gate
//!
//! - **PASS (α=0.3):** mean cos ≥ 0.95, min cos ≥ 0.90.
//! - **KILL:** mean cos < 0.90 — steering corrupts decisions.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features latent_field_steering --release \
//!   --test latent_steering_g2_rank_preservation -- --nocapture
//! ```

#![cfg(feature = "latent_field_steering")]

use katgpt_core::latent_steering::{LatentSteeringVector, apply_latent_steering};
use katgpt_core::simd;

const D: usize = 8;
const N_ACTIONS: usize = 5;
const N_SAMPLES: usize = 100;
/// The headline α for the gate.
const ALPHA_GATE: f32 = 0.3;

// ── Deterministic xorshift64* PRNG ─────────────────────────────────────────

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
    fn next_unit_vec(&mut self, d: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|_| self.next_f32()).collect();
        let norm = simd::simd_dot_f32(&v, &v, d).sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        v.into_iter().map(|x| x * inv).collect()
    }
}

/// `scores = W^T · state` where W is (D, N_ACTIONS) row-major.
/// Returns N_ACTIONS-dim score vector.
fn action_scores(state: &[f32], weights: &[f32], d: usize, n_actions: usize) -> Vec<f32> {
    debug_assert_eq!(state.len(), d);
    debug_assert_eq!(weights.len(), d * n_actions);
    (0..n_actions)
        .map(|a| {
            // Column a of W: weights[r * n_actions + a] for r in 0..d.
            let mut acc = 0.0f32;
            for r in 0..d {
                acc += weights[r * n_actions + a] * state[r];
            }
            acc
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = simd::simd_dot_f32(a, b, a.len());
    let na = simd::simd_dot_f32(a, a, a.len()).sqrt();
    let nb = simd::simd_dot_f32(b, b, b.len()).sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Run the rank-preservation measurement at a given α. Returns (mean_cos, min_cos, argmax_flip_rate).
fn measure_at_alpha(
    alpha: f32,
    states: &[Vec<f32>],
    weights: &[f32],
    direction: &[f32],
) -> (f32, f32, f32) {
    let steering = LatentSteeringVector::new_unchecked(direction.to_vec(), alpha);
    let mut cosines: Vec<f32> = Vec::with_capacity(states.len());
    let mut argmax_flips = 0usize;

    for state_orig in states {
        let scores_pre = action_scores(state_orig, weights, D, N_ACTIONS);
        let mut state = state_orig.clone();
        apply_latent_steering(&mut state, &steering);
        let scores_post = action_scores(&state, weights, D, N_ACTIONS);

        let cos = cosine(&scores_pre, &scores_post);
        cosines.push(cos);

        // Argmax stability: does the top-1 action change?
        let argmax_pre = scores_pre
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let argmax_post = scores_post
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if argmax_pre != argmax_post {
            argmax_flips += 1;
        }
    }

    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min = cosines[0];
    let flip_rate = argmax_flips as f32 / states.len() as f32;
    (mean, min, flip_rate)
}

#[test]
fn g2_behavior_rank_preservation() {
    let mut rng = Rng::new(0xBEEF);

    // ── Fixed action weight matrix (8 × 5), row-major ──────────────────
    let weights: Vec<f32> = (0..D * N_ACTIONS).map(|_| rng.next_f32()).collect();

    // ── 100 random 8-dim states (entries U(-1, 1)) ─────────────────────
    let states: Vec<Vec<f32>> = (0..N_SAMPLES)
        .map(|_| (0..D).map(|_| rng.next_f32()).collect())
        .collect();

    // ── Random unit steering direction ─────────────────────────────────
    let direction = rng.next_unit_vec(D);

    // ── α sweep: characterize rank preservation across steering strengths ─
    println!("G2 rank preservation sweep (d={D}, n_actions={N_ACTIONS}, n_samples={N_SAMPLES}):");
    println!("  {:>6} | {:>10} | {:>10} | {:>12}", "alpha", "mean_cos", "min_cos", "argmax_flip%");
    println!("  -------|------------|------------|--------------");

    let mut gate_result = None;
    for &alpha in &[0.1, ALPHA_GATE, 0.5, 0.9] {
        let (mean, min, flip) = measure_at_alpha(alpha, &states, &weights, &direction);
        println!("  {:>6.1} | {:>10.4} | {:>10.4} | {:>11.1}%", alpha, mean, min, flip * 100.0);
        if (alpha - ALPHA_GATE).abs() < 1e-5 {
            gate_result = Some((mean, min, flip));
        }
    }

    let (mean, min, flip) = gate_result.expect("α=0.3 must be in the sweep");

    // ── Headline gate: mean cos ≥ 0.95, min cos ≥ 0.90 at α=0.3 ────────
    println!(
        "\nG2 gate (α={ALPHA_GATE}): mean_cos={mean:.4} (≥0.95), min_cos={min:.4} (≥0.90), \
         argmax_flip={:.1}%",
        flip * 100.0
    );
    assert!(
        mean >= 0.95,
        "G2 FAIL: mean cos {mean:.4} < 0.95 — steering corrupts action rankings. \
         The primitive must be demoted to research-only."
    );
    assert!(
        min >= 0.90,
        "G2 FAIL: min cos {min:.4} < 0.90 — at least one entity's ranking was heavily corrupted."
    );
    println!("G2 PASS: behavior rank preserved (mean cos {mean:.4} ≥ 0.95).");
}
