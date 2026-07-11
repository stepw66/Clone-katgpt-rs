//! MANCE Manifold-Aware Concept Erasure demo (Plan 426).
//!
//! Demonstrates the manifold-constrained concept erasure primitive on a
//! synthetic 8-d latent state with 50 natural reference points.
//!
//! Shows: (a) target alignment drops, (b) orthogonal directions preserved,
//! (c) displacement within trust region. Compares MANCE vs unconstrained erasure.
//!
//! # Run
//!
//! ```bash
//! cargo run --example manifold_erasure_demo --features manifold_erasure
//! ```

#![cfg(feature = "manifold_erasure")]

use katgpt_core::{
    ManceConfig, ManceScratch, manifold_erasure_loop_into, manifold_erasure_step_into,
};
use katgpt_core::simd::simd_dot_f32;

fn main() {
    let d = 8;
    let n = 50;

    // Create a natural pool (the manifold) — 50 points in 8-d space.
    let mut pool = vec![0.0f32; n * d];
    let mut s: u64 = 42;
    for i in 0..n {
        for j in 0..d {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((s >> 33) as f32) / (1u64 << 31) as f32;
            pool[i * d + j] = r * 2.0 - 1.0;
        }
    }

    // Input latent state.
    let x = vec![0.5f32; d];

    // Erasure direction (the concept to remove).
    let gradient = vec![1.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2, -0.6];

    // Normalize gradient for alignment measurement.
    let grad_norm = simd_dot_f32(&gradient, &gradient, d).sqrt();
    let u: Vec<f32> = gradient.iter().map(|g| g / grad_norm).collect();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  MANCE Manifold-Aware Concept Erasure Demo (Plan 426)     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Input: x = {:?}", x);
    println!("Erasure direction: u = {:?}", u);
    println!("Natural pool: {} points in {}-d space\n", n, d);

    // (a) Single MANCE step with default config.
    let config = ManceConfig { k: 16, r: 8, alpha: 0.0, ..Default::default() };
    let mut scratch = ManceScratch::with_capacity(d, config.k, config.r);
    let mut out_mance = vec![0.0; d];

    let info = manifold_erasure_step_into(&x, &gradient, &pool, n, &config, &mut scratch, &mut out_mance).unwrap();

    let x_align = simd_dot_f32(&x, &u, d).abs();
    let out_align = simd_dot_f32(&out_mance, &u, d).abs();

    println!("── (a) Single MANCE step (ε={}, α={}) ──", config.epsilon, config.alpha);
    println!("  Target alignment before: {:.6}", x_align);
    println!("  Target alignment after:  {:.6}", out_align);
    println!("  Reduction:               {:.1}%", (1.0 - out_align / x_align) * 100.0);
    println!("  Displacement:            {:.6}", info.displacement);
    println!("  Trust region bound:      {:.6} (ε·r_i)", config.epsilon * info.local_radius);
    println!("  Local radius r_i:        {:.6}", info.local_radius);
    println!("  Gradient-tangent align:  {:.6}", info.alignment);

    // (b) Orthogonal direction preservation.
    println!("\n── (b) Orthogonal direction preservation ──");
    let pool_2d: Vec<f32> = (0..n)
        .flat_map(|i| {
            let v = i as f32 * 0.1 - 1.0;
            vec![v, v * 0.5, 0.0, 0.0]
        })
        .collect();
    let x_2d = vec![0.5, 0.5, 0.7, 0.3];
    let grad_2d = vec![1.0, 1.0, 0.0, 0.0];
    let config_2d = ManceConfig { k: 8, r: 2, alpha: 0.0, ..Default::default() };
    let mut scratch_2d = ManceScratch::with_capacity(4, config_2d.k, config_2d.r);
    let mut out_2d = vec![0.0; 4];

    manifold_erasure_step_into(&x_2d, &grad_2d, &pool_2d, n, &config_2d, &mut scratch_2d, &mut out_2d).unwrap();

    println!("  e3 before: {:.6}, after: {:.6} (Δ = {:.2e})", x_2d[2], out_2d[2], (out_2d[2] - x_2d[2]).abs());
    println!("  e4 before: {:.6}, after: {:.6} (Δ = {:.2e})", x_2d[3], out_2d[3], (out_2d[3] - x_2d[3]).abs());

    // (c) 10-round iterative loop.
    println!("\n── (c) 10-round iterative loop ──");
    let mut out_loop = vec![0.0; d];
    let grad_ref = &gradient;
    let gf = move |_state: &[f32], buf: &mut [f32]| {
        buf.copy_from_slice(grad_ref);
    };
    let infos = manifold_erasure_loop_into(&x, &gf, &pool, n, &config, 10, &mut scratch, &mut out_loop).unwrap();

    let loop_align = simd_dot_f32(&out_loop, &u, d).abs();
    println!("  Rounds executed: {}", infos.len());
    println!("  Target alignment after {} rounds: {:.6}", infos.len(), loop_align);
    println!("  Total reduction: {:.1}%", (1.0 - loop_align / x_align) * 100.0);

    // (d) MANCE vs unconstrained erasure (the AmbCE++ ablation).
    println!("\n── (d) MANCE vs unconstrained erasure ──");
    let grad_norm = simd_dot_f32(&gradient, &gradient, d).sqrt();
    let u_uncon: Vec<f32> = gradient.iter().map(|g| g / grad_norm).collect();
    let x_proj_uncon = simd_dot_f32(&x, &u_uncon, d);
    let lambda = info.lambda;
    let scale_uncon = lambda * x_proj_uncon;
    let mut out_uncon = vec![0.0; d];
    for i in 0..d {
        out_uncon[i] = x[i] - scale_uncon * u_uncon[i];
    }

    let mance_norm = simd_dot_f32(&out_mance, &out_mance, d).sqrt();
    let uncon_norm = simd_dot_f32(&out_uncon, &out_uncon, d).sqrt();
    let mance_proj = simd_dot_f32(&out_mance, &u_uncon, d).abs();
    let uncon_proj = simd_dot_f32(&out_uncon, &u_uncon, d).abs();
    let mance_orth = (mance_norm * mance_norm - mance_proj * mance_proj).max(0.0).sqrt();
    let uncon_orth = (uncon_norm * uncon_norm - uncon_proj * uncon_proj).max(0.0).sqrt();

    println!("  MANCE orthogonal energy:        {:.6}", mance_orth);
    println!("  Unconstrained orthogonal energy: {:.6}", uncon_orth);
    println!("  MANCE preserves more: {}", mance_orth >= uncon_orth);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Demo complete — MANCE erasure works as expected          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
}
