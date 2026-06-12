#![cfg(feature = "vortex_flow")]
//! Benchmark — SIMD Register TopK for k≤16 (Plan 256 Phase 1)
//!
//! Compares SIMD-optimized argtopk path vs scalar fallback across
//! k=4, 8, 16, 32 and n=64, 128, 256, 512, 1024 block counts.
//!
//! Run: `cargo test --features vortex_flow --test bench_256_simd_topk -- --nocapture`

use katgpt_rs::dash_attn::block_topk::{argtopk, argtopk_scalar_heap};

// ── Helpers ───────────────────────────────────────────────────

/// Deterministic pseudo-random score generator (index-based seed).
fn make_scores(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i.wrapping_mul(2654435761)).wrapping_add(seed.wrapping_mul(40503))) as f32;
            (x * 0.0001).sin() * 0.5 + 0.5
        })
        .collect()
}

/// Reference scalar argtopk — full sort + take top-k.
fn argtopk_reference(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

// ── Correctness check ─────────────────────────────────────────

#[test]
fn bench_simd_topk_correctness_and_speed() {
    let k_values = [4, 8, 16];
    let n_values = [64, 128, 256, 512, 1024];
    let seed = 42;
    let iters = 1000;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 256 — SIMD Register TopK Benchmark (k≤16)               ║");
    println!("╠═════════╦════════╦════════════════╦════════════════╦═══════════╣");
    println!("║    k    ║   n    ║  SIMD (ns/call)║ Scalar(ns/call)║ Speedup   ║");
    println!("╠═════════╬════════╬════════════════╬════════════════╬═══════════╣");

    for &k in &k_values {
        for &n in &n_values {
            let scores = make_scores(n, seed);

            // Verify correctness first
            let mut simd_indices = Vec::with_capacity(k);
            argtopk(&scores, k, &mut simd_indices);
            let ref_indices = argtopk_reference(&scores, k);
            assert_eq!(
                simd_indices, ref_indices,
                "SIMD mismatch at k={k}, n={n}: simd={simd_indices:?} != ref={ref_indices:?}"
            );

            // Benchmark SIMD path
            let mut simd_indices = Vec::with_capacity(k);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                simd_indices.clear();
                argtopk(&scores, k, &mut simd_indices);
            }
            let simd_ns = start.elapsed().as_nanos() as f64 / iters as f64;

            // Benchmark scalar path
            let mut scalar_indices = Vec::with_capacity(k);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                scalar_indices.clear();
                argtopk_scalar_heap(&scores, k, &mut scalar_indices);
            }
            let scalar_ns = start.elapsed().as_nanos() as f64 / iters as f64;

            let speedup = scalar_ns / simd_ns;
            println!(
                "║ k={k:<5}║ n={n:<5}║ {simd_ns:>12.1}  ║ {scalar_ns:>12.1}  ║ {speedup:>7.2}x  ║"
            );
        }
    }

    println!("╚═════════╩════════╩════════════════╩════════════════╩═══════════╝");
}

// ── k=32 fallback benchmark (scalar path) ─────────────────────

#[test]
fn bench_simd_topk_k32_scalar_fallback() {
    let n_values = [64, 128, 256, 512, 1024];
    let seed = 99;
    let iters = 1000;
    let k = 32;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  k=32 scalar fallback (selection sort) — n sweep              ║");
    println!("╠════════════════╦════════════════════════════════════════════════╣");
    println!("║       n        ║     ns/call                                    ║");
    println!("╠════════════════╬════════════════════════════════════════════════╣");

    for &n in &n_values {
        let scores = make_scores(n, seed);

        // Verify correctness
        let mut indices = Vec::with_capacity(k);
        argtopk(&scores, k, &mut indices);
        let ref_indices = argtopk_reference(&scores, k);
        assert_eq!(
            indices, ref_indices,
            "Scalar fallback mismatch at k={k}, n={n}"
        );

        // Benchmark
        let mut indices = Vec::with_capacity(k);
        let mut pairs = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            indices.clear();
            pairs.clear();
            katgpt_rs::dash_attn::block_topk::argtopk_with_scratch(
                &scores,
                k,
                &mut indices,
                &mut pairs,
            );
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;

        println!("║ n={n:<12}║ {ns:>12.1} ns                                ║",);
    }

    println!("╚════════════════╩════════════════════════════════════════════════╝");
}

// ── Detailed sweep: fixed n=256, k sweep ──────────────────────

#[test]
fn bench_simd_topk_k_sweep_n256() {
    let k_values = [1, 2, 4, 8, 12, 16];
    let n = 256;
    let seed = 77;
    let iters = 2000;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  n=256 — k sweep (SIMD path k≤16 vs scalar)                   ║");
    println!("╠═════════╦════════════════╦════════════════╦═════════════════════╣");
    println!("║    k    ║  SIMD (ns/call)║ Scalar(ns/call)║ Speedup             ║");
    println!("╠═════════╬════════════════╬════════════════╬═════════════════════╣");

    for &k in &k_values {
        let scores = make_scores(n, seed);

        // Correctness check
        let mut simd_indices = Vec::with_capacity(k);
        argtopk(&scores, k, &mut simd_indices);
        let ref_indices = argtopk_reference(&scores, k);
        assert_eq!(simd_indices, ref_indices, "Mismatch at k={k}");

        // SIMD benchmark
        let mut simd_indices = Vec::with_capacity(k);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            simd_indices.clear();
            argtopk(&scores, k, &mut simd_indices);
        }
        let simd_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        // Scalar benchmark
        let mut scalar_indices = Vec::with_capacity(k);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            scalar_indices.clear();
            argtopk_scalar_heap(&scores, k, &mut scalar_indices);
        }
        let scalar_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = scalar_ns / simd_ns;
        println!(
            "║ k={k:<5}║ {simd_ns:>12.1}  ║ {scalar_ns:>12.1}  ║ {speedup:>7.2}x              ║"
        );
    }

    println!("╚═════════╩════════════════╩════════════════╩═════════════════════╝");
}
