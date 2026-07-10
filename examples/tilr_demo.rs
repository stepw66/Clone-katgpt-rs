//! TILR Demo — Trajectory-Invariant Latent Refinement (alignment-gated subspace correction).
//!
//! Demonstrates the full TILR pipeline:
//! 1. Collect synthetic contrastive differences `δ_t = h_good − h_bad`.
//! 2. Discover the invariant subspace `S_r` via truncated SVD (`discover_invariant_subspace`).
//! 3. Apply γ-gated correction (`tilr_refine_into`) and verify three behaviors:
//!    (a) **No-harm** (γ=0): direction orthogonal to basis → state unchanged.
//!    (b) **Full correction** (γ=1): direction in basis span → full step applied.
//!    (c) **Graceful intermediate** (0<γ<1): partial correction, proportional to alignment.
//!
//! Run: `cargo run --example tilr_demo`
//!
//! Plan 425, Research 408, arXiv:2606.29164 (ICML 2026 Mech Interp Workshop).

use katgpt_core::tilr::{
    check_orthonormal, discover_invariant_subspace, tilr_refine_into, TilrScratch,
};

fn main() {
    let d = 6usize;

    // ── Step 1: synthetic contrastive differences ──────────────────────
    // Simulate "good vs bad" activation differences. In a real deployment
    // these come from a frozen reference pair (two epoch checkpoints). Here
    // we generate them deterministically along two "semantic axes" in ℝ⁶.
    println!("=== TILR Demo (d={d}) ===\n");
    println!("Step 1: Collecting 10 synthetic contrastive differences...");

    // Axis 1: (1, 1, 0, 0, 0, 0) / √2 — the "valence" direction.
    // Axis 2: (0, 0, 1, 1, 0, 0) / √2 — the "arousal" direction.
    let a1 = [1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0];
    let a2 = [0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0];
    let scale = std::f32::consts::FRAC_1_SQRT_2;

    // 10 differences: independent linear combinations of a1 and a2 so the
    // difference set spans the full 2-d plane (rank-2). The weights for a1 and
    // a2 are decorrelated across samples — if both tracked the same index they
    // would be collinear and the SVD would collapse to rank-1.
    let mut seed: u32 = 42;
    let lcg = |s: &mut u32| -> f32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*s >> 8) as f32) / 16777216.0 - 0.5 // [-0.5, 0.5)
    };
    let diffs: Vec<Vec<f32>> = (0..10)
        .map(|_| {
            let w1 = 1.0 + lcg(&mut seed) * 2.0;
            let w2 = 1.0 + lcg(&mut seed) * 2.0;
            (0..d)
                .map(|i| scale * (w1 * a1[i] + w2 * a2[i]))
                .collect()
        })
        .collect();
    let diff_refs: Vec<&[f32]> = diffs.iter().map(|v| v.as_slice()).collect();

    // ── Step 2: discover invariant subspace via SVD ────────────────────
    println!("Step 2: Discovering invariant subspace (tau=0.95)...");
    let (basis, r) = discover_invariant_subspace(&diff_refs, 0.95).unwrap();
    println!("  → rank r = {r} (expected 2: valence + arousal axes)");
    println!("  → basis: {} floats ({r} × {d})", basis.len());

    // Validate the basis is orthonormal (setup-time check).
    check_orthonormal(&basis, r, d, 1e-4).unwrap();
    println!("  → orthonormality check: PASS");

    // ── Step 3: apply γ-gated correction ───────────────────────────────
    println!("\nStep 3: Applying γ-gated correction (eta_base=0.5)...\n");
    let eta_base = 0.5f32;
    let mut scratch = TilrScratch::with_capacity(d, r);

    // (a) No-harm: direction orthogonal to basis → γ = 0.
    let state_a = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let direction_orth = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // along e_4, outside span
    let mut out_a = [0.0f32; 6];
    let gamma_a = tilr_refine_into(
        &state_a,
        &direction_orth,
        &basis,
        r,
        eta_base,
        1e-12,
        &mut scratch,
        &mut out_a,
    )
    .unwrap();
    let unchanged = state_a
        .iter()
        .zip(out_a.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("(a) No-harm (direction ⊥ basis):");
    println!("    γ = {gamma_a:.6} (expected 0.0)");
    println!("    state unchanged bit-identically: {unchanged}");
    assert!(unchanged, "no-harm contract violated!");

    // (b) Full correction: direction in span(basis) → γ ≈ 1.
    let state_b = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let direction_in = [1.0f32, 1.0, 0.5, 0.5, 0.0, 0.0]; // in the valence+arousal plane
    let mut out_b = [0.0f32; 6];
    let gamma_b = tilr_refine_into(
        &state_b,
        &direction_in,
        &basis,
        r,
        eta_base,
        1e-12,
        &mut scratch,
        &mut out_b,
    )
    .unwrap();
    println!("\n(b) Full correction (direction ∈ span(basis)):");
    println!("    γ = {gamma_b:.6} (expected ≈ 1.0)");
    println!("    state:   {:?}", state_b);
    println!("    out:     {:?}", out_b);
    println!("    shift:   {:?}", {
        let shift: Vec<f32> = out_b.iter().zip(state_b.iter()).map(|(o, s)| o - s).collect();
        shift
    });
    assert!((gamma_b - 1.0).abs() < 0.01, "expected γ≈1.0 for in-span direction");

    // (c) Graceful intermediate: direction partially aligned → 0 < γ < 1.
    let state_c = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    // Half in-span (dims 0-1), half out-of-span (dims 4-5).
    let direction_mixed = [0.5f32, 0.5, 0.0, 0.0, 0.5, 0.5];
    let mut out_c = [0.0f32; 6];
    let gamma_c = tilr_refine_into(
        &state_c,
        &direction_mixed,
        &basis,
        r,
        eta_base,
        1e-12,
        &mut scratch,
        &mut out_c,
    )
    .unwrap();
    let in_norm: f32 = direction_mixed[..4].iter().map(|x| x * x).sum::<f32>().sqrt();
    let full_norm: f32 = direction_mixed.iter().map(|x| x * x).sum::<f32>().sqrt();
    let expected_gamma = in_norm / full_norm;
    println!("\n(c) Graceful intermediate (direction partially aligned):");
    println!("    γ = {gamma_c:.6} (expected ≈ {expected_gamma:.6})");
    println!("    step size η = eta_base × γ = {:.6}", eta_base * gamma_c);
    println!("    dims 4-5 unchanged (out-of-span): {}",
        (4..6).all(|i| (out_c[i] - state_c[i]).abs() < 1e-6));
    assert!((gamma_c - expected_gamma).abs() < 0.01, "γ should match alignment ratio");
    assert!(gamma_c > 0.0 && gamma_c < 1.0, "γ should be strictly intermediate");

    println!("\n=== All three behaviors verified. TILR no-harm contract holds. ===");
}
