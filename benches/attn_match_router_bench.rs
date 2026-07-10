//! Router benchmark (Plan 271 Phase 2, T2.7).
//!
//! Sweeps `t` from 16 to 4096 and measures:
//! - `pick_backend` latency (should be ~nanoseconds, no allocation).
//! - Backend transitions across the configured thresholds.
//!
//! Uses `std::time::Instant` (criterion is not a project dependency). Run with:
//! ```bash
//! cargo bench --bench attn_match_router_bench --features attn_match
//! ```
//!
//! Or as a plain binary (since this isn't a criterion harness):
//! ```bash
//! cargo run --release --bench attn_match_router_bench --features attn_match
//! ```

use katgpt_attn_match::router::{
    SolverBackend, SolverRouter, SolverRouterConfig, pick_backend,
};

fn main() {
    println!("=== Attention Matching Router Benchmark (Plan 271 Phase 2) ===\n");

    // 1. Sweep t across regimes and show backend transitions.
    let cfg = SolverRouterConfig::default();
    println!(
        "Config: cpu_max_t={} simd_max_t={} gpu_min_t={} ane_max_t={} hysteresis_pct={}",
        cfg.cpu_max_t, cfg.simd_max_t, cfg.gpu_min_t, cfg.ane_max_t, cfg.hysteresis_pct
    );
    println!();

    let t_values: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
    println!("{:>6} {:>10} {:>10}", "t", "no_gpu", "with_gpu");
    for &t in t_values {
        let b_no_gpu = pick_backend(t, t * 4, false, &cfg);
        let b_gpu = pick_backend(t, t * 4, true, &cfg);
        println!("{:>6} {:>10} {:>10}", t, b_no_gpu.as_str(), b_gpu.as_str());
    }
    println!();

    // 2. Measure pick_backend latency with hysteresis tracking.
    // We call pick_backend 1M times in a tight loop and report ns/call.
    let iters = 1_000_000;
    let mut router = SolverRouter::new(cfg);

    // Warm up.
    for i in 0..1000 {
        let _ = router.pick_backend(64 + (i % 128), 8192, true);
    }

    let start = std::time::Instant::now();
    let mut last = SolverBackend::CpuScalar;
    for i in 0..iters {
        // Sweep t across regimes to exercise the hysteresis path.
        let t = 16 + (i % 4096);
        last = router.pick_backend(t, 8192, true);
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;

    println!(
        "pick_backend: {} calls in {:?} → {:.2} ns/call (last backend: {:?})",
        iters, elapsed, ns_per_call, last
    );
    println!();

    // 3. Show hysteresis effect: small t fluctuations keep the prior backend.
    println!("Hysteresis demo (small t wiggle around threshold):");
    let mut router = SolverRouter::new(cfg);
    let t0 = 512usize; // well within simd regime
    let b0 = router.pick_backend(t0, 8192, true);
    println!("  t={:>4} → backend={:?}", t0, b0);
    for &dt in &[1, 5, 10, 20, 50] {
        let t = t0 + dt;
        let b = router.pick_backend(t, 8192, true);
        let pct = (dt as f64 / t0 as f64) * 100.0;
        println!(
            "  t={:>4} (+{}, {:.1}%) → backend={:?} {}",
            t,
            dt,
            pct,
            b,
            if b == b0 { "(kept)" } else { "(switched)" }
        );
    }

    println!();
    println!("Done.");
}
