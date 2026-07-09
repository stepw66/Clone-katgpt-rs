//! Issue 043 — Performance benchmark for `ns_inv_sqrt_psd_into` with the
//! blocked `blocked_dot8` kernel (r=32, r=64 — the LoRA-Muon production cases).
//!
//! Run with: `cargo bench --bench issue043_ns_inv_sqrt_perf --features newton_schulz`
//! Or: `cargo test --bench issue043_ns_inv_sqrt_perf --features newton_schulz -- --nocapture --ignored`

use katgpt_rs::newton_schulz::{InvSqrtScratch, ns_inv_sqrt_psd_into};
use std::hint::black_box;
use std::time::Instant;

fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*state >> 33) as u32;
    (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn make_psd(seed: u64, r: usize) -> Vec<f32> {
    let mut state = seed;
    let m = (0..r * r).map(|_| lcg_next(&mut state)).collect::<Vec<f32>>();
    let mut p = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f32;
            for k in 0..r {
                s += m[k * r + i] * m[k * r + j];
            }
            p[i * r + j] = s;
        }
    }
    p
}

fn bench_ns_inv_sqrt(r: usize, n_iters: u8, warmup: usize, rounds: usize) -> f64 {
    let p = make_psd(42, r);
    let mut out = vec![0.0f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);

    // Warmup
    for _ in 0..warmup {
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, n_iters);
    }

    let mut best_us = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..100 {
            ns_inv_sqrt_psd_into(black_box(&p), r, black_box(&mut out), &mut scratch, n_iters);
        }
        let elapsed = start.elapsed();
        let us = elapsed.as_secs_f64() * 1e6 / 100.0;
        if us < best_us {
            best_us = us;
        }
    }
    best_us
}

#[test]
#[ignore]
fn issue043_perf_r32_r64() {
    // r=32 (typical LoRA rank)
    let us32 = bench_ns_inv_sqrt(32, 7, 10, 5);
    let gflops32 = (7.0 * 3.0 * 32.0f64.powi(3) * 2.0 / 1e9) / (us32 / 1e6);
    eprintln!("ns_inv_sqrt_psd_into(r=32, 7 iters): {us32:.1} µs ({gflops32:.1} GFLOP/s)");

    // r=64 (production LoRA-Muon rank)
    let us64 = bench_ns_inv_sqrt(64, 7, 10, 5);
    let gflops64 = (7.0 * 3.0 * 64.0f64.powi(3) * 2.0 / 1e9) / (us64 / 1e6);
    eprintln!("ns_inv_sqrt_psd_into(r=64, 7 iters): {us64:.1} µs ({gflops64:.1} GFLOP/s)");

    // r=16 (minimum for blocked path)
    let us16 = bench_ns_inv_sqrt(16, 7, 10, 5);
    let gflops16 = (7.0 * 3.0 * 16.0f64.powi(3) * 2.0 / 1e9) / (us16 / 1e6);
    eprintln!("ns_inv_sqrt_psd_into(r=16, 7 iters): {us16:.1} µs ({gflops16:.1} GFLOP/s)");

    eprintln!("\nBaseline (Plan 421, simd_dot_f32 only):");
    eprintln!("  r=32: ~51 µs (15.7 GFLOP/s)");
    eprintln!("  r=64: ~297 µs (22.0 GFLOP/s)");
}
