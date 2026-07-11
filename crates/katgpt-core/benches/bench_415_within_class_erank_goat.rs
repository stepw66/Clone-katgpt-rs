//! Plan 415 G4 latency bench — within_class_effective_rank vs effective_rank.
//!
//! Both functions share the Jacobi eigensolver (O(dim³) sweeps). The within-class
//! variant adds an O(n·d) class-mean + residual pass which should be negligible
//! vs the eigendecomposition. This bench confirms they're in the same ballpark.
//!
//! Run: cargo bench --bench bench_415_within_class_erank_goat --features sink_aware_attn

use katgpt_core::data_probe::geometry::{effective_rank, within_class_effective_rank};

fn main() {
    let mut rng = fastrand::Rng::with_seed(42);
    let dim = 64usize;
    let n = 256usize;
    let n_classes = 4usize;
    let per_class = n / n_classes;

    // Build 4 well-separated isotropic classes.
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    let mut labels: Vec<usize> = Vec::with_capacity(n);
    for c in 0..n_classes {
        let centroid: Vec<f32> = (0..dim).map(|_| (c as f32) * 100.0).collect();
        for _ in 0..per_class {
            for &mu in centroid.iter() {
                // CLT gaussian noise.
                let noise: f32 = (0..12).map(|_| rng.f32()).sum::<f32>() - 6.0;
                flat.push(mu + noise);
            }
            labels.push(c);
        }
    }
    // Owned copy for effective_rank's &[Vec<f32>] signature.
    let owned: Vec<Vec<f32>> = (0..n)
        .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // Warmup.
    for _ in 0..10 {
        let _ = within_class_effective_rank(&flat, dim, &labels);
        let _ = effective_rank(&owned);
    }

    let iters = 1000usize;

    // ── within_class_effective_rank ──
    let t0 = std::time::Instant::now();
    let mut acc_w = 0.0f32;
    for _ in 0..iters {
        acc_w += within_class_effective_rank(&flat, dim, &labels);
    }
    let dt_w = t0.elapsed();
    let _ = acc_w; // prevent dead-code elimination

    // ── effective_rank (global) ──
    let t1 = std::time::Instant::now();
    let mut acc_g = 0.0f32;
    for _ in 0..iters {
        acc_g += effective_rank(&owned);
    }
    let dt_g = t1.elapsed();
    let _ = acc_g;

    let ns_w = dt_w.as_nanos() as f64 / iters as f64;
    let ns_g = dt_g.as_nanos() as f64 / iters as f64;
    let ratio = ns_w / ns_g;

    println!("── Plan 415 G4 latency bench (dim={dim}, n={n}, C={n_classes}, iters={iters}) ──");
    println!("    within_class_effective_rank : {ns_w:>10.1} ns/call");
    println!("    effective_rank (global)     : {ns_g:>10.1} ns/call");
    println!("    ratio (within / global)     : {ratio:>10.3}x");
    println!();
    println!("Both dominated by the O(dim³) Jacobi eigensolver; the O(n·d) class-mean");
    println!("pass in the within-class variant is negligible vs the eigendecomposition.");
    println!();

    // GATE: within-class should be within ~2x of global (it does strictly more
    // work: class-mean pass + rank-1 residual updates, but the eigendecomp
    // dominates). Allow generous headroom for the HashMap class-grouping pass.
    let gate_pass = ratio < 3.0;
    println!(
        "G4 GATE (ratio < 3.0x): {}",
        if gate_pass { "PASS" } else { "FAIL" }
    );
    if !gate_pass {
        std::process::exit(1);
    }
}
