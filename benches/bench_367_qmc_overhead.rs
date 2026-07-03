//! Plan 367 Phase 3 T3.4 — QMC vs i.i.d. overhead per-rollout benchmark (G5).
//!
//! Measures the QMC-specific overhead in `ppot_resample_multi_strategy`:
//! the source `draw(k)` call + the per-position rescale-divide, minus the
//! i.i.d. baseline (per-position `rng.uniform()`). Target: < 1µs/rollout.
//!
//! Also measures the standalone `sample_from_distribution_qmc` descend cost
//! vs `sample_from_distribution` — the per-token overhead.
//!
//! # Convention
//!
//! `std::time::Instant` + `harness = false` + custom `main()` (criterion is
//! not a root-crate dev-dep; matches bench_284_clr_perf / faithfulness_probe_bench).
//!
//! Run with:
//! ```bash
//! cargo run --release --features qmc_sampling --bench bench_367_qmc_overhead
//! ```

#![cfg(feature = "qmc_sampling")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::speculative::qmc::{LatticeQmc, QmcSource, StratifiedQmc};
use katgpt_core::speculative::{sample_from_distribution, sample_from_distribution_qmc};
use katgpt_rs::speculative::ppot::{
    PpotConfig, QmcConfig, QmcMethod, ppot_resample_multi_strategy,
};
use katgpt_rs::types::Rng;

/// Number of iterations for the timing loop (enough to swamp timer overhead).
const ITERS: usize = 50_000;

/// Run a closure `iters` times and return the mean wall-clock nanoseconds per call.
fn time_ns<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // Warmup
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn main() {
    // ── 1. Per-token descend overhead: QMC vs i.i.d. ────────────────────
    //
    // Measures `sample_from_distribution_qmc` (inverse-CDF + rescale) vs
    // `sample_from_distribution` (inverse-CDF + rng.uniform()). The QMC
    // version should be faster (no RNG state mutation, just a divide).

    let probs: Vec<f32> = vec![0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05];
    let mut rng = Rng::new(42);

    let ns_iid_descend = time_ns(ITERS, || {
        let t = sample_from_distribution(&probs, &mut rng);
        black_box(t);
    });

    let mut u = 0.42f32;
    let ns_qmc_descend = time_ns(ITERS, || {
        let t = sample_from_distribution_qmc(&probs, &mut u);
        black_box(t);
    });

    println!("── 1. Per-token descend overhead (vocab=10) ──────────────");
    println!("  i.i.d. sample_from_distribution:  {ns_iid_descend:.1} ns/call");
    println!("  QMC  sample_from_distribution_qmc: {ns_qmc_descend:.1} ns/call");
    println!(
        "  QMC overhead per token:            {:.1} ns",
        ns_qmc_descend - ns_iid_descend
    );

    // ── 2. QMC source draw overhead (per K points) ───────────────────────
    //
    // Measures `LatticeQmc::draw(k)` and `StratifiedQmc::draw(k)` — the
    // source overhead amortized over K rollouts.

    for &k in &[8usize, 16, 32, 64] {
        let mut lattice = LatticeQmc::new(42);
        let mut buf = vec![0.0f32; k];
        let ns_lattice = time_ns(ITERS, || {
            lattice.draw(k, &mut buf);
            black_box(&buf);
        });

        let mut stratified = StratifiedQmc::new(42);
        let ns_strat = time_ns(ITERS, || {
            stratified.draw(k, &mut buf);
            black_box(&buf);
        });

        println!();
        println!("── 2. QMC source draw overhead (K={k}) ───────────────────");
        println!("  LatticeQmc::draw:    {ns_lattice:.1} ns total  ({:.1} ns/point)", ns_lattice / k as f64);
        println!("  StratifiedQmc::draw: {ns_strat:.1} ns total  ({:.1} ns/point)", ns_strat / k as f64);
    }

    // ── 3. End-to-end PPoT multi-strategy: QMC vs i.i.d. ────────────────
    //
    // Measures the FULL `ppot_resample_multi_strategy` with QMC on vs off.
    // The overhead (QMC_on − QMC_off) / K is the per-rollout cost.

    let base_path: Vec<usize> = vec![0, 1, 2, 3, 4];
    let marginals_data: Vec<Vec<f32>> = (0..5)
        .map(|i| {
            vec![0.1, 0.2, 0.15, 0.25, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]
                .into_iter()
                .enumerate()
                .map(|(j, p)| p + (i + j) as f32 * 1e-6) // tiny perturbation for variety
                .collect()
        })
        .collect();
    let marginals: Vec<&[f32]> = marginals_data.iter().map(|v| v.as_slice()).collect();
    let positions = [0, 1, 2, 3, 4];
    let mut scratch = vec![0.0f32; 10];

    println!();
    println!("── 3. PPoT multi-strategy: QMC vs i.i.d. (T=5, vocab=10) ──");

    for &k in &[8usize, 16, 32, 64] {
        // i.i.d. baseline
        let mut rng_iid = Rng::new(42);
        let config_iid = PpotConfig::default().with_cached_support(10);
        let ns_iid = time_ns(ITERS / k.max(1), || {
            let v = ppot_resample_multi_strategy(
                black_box(&base_path),
                black_box(&marginals),
                black_box(&positions),
                black_box(k),
                &[],
                &config_iid,
                &mut scratch,
                &mut rng_iid,
            );
            black_box(v);
        });

        // QMC (Lattice)
        let mut rng_qmc = Rng::new(42);
        let mut config_qmc = PpotConfig::default().with_cached_support(10);
        config_qmc.enabled = true;
        config_qmc.qmc = QmcConfig {
            enabled: true,
            method: QmcMethod::Lattice,
            seed: 42,
        };
        let ns_qmc = time_ns(ITERS / k.max(1), || {
            let v = ppot_resample_multi_strategy(
                black_box(&base_path),
                black_box(&marginals),
                black_box(&positions),
                black_box(k),
                &[],
                &config_qmc,
                &mut scratch,
                &mut rng_qmc,
            );
            black_box(v);
        });

        let overhead_per_rollout = (ns_qmc - ns_iid) / k as f64;

        println!("  K={k:>3}:  i.i.d. {ns_iid:>8.0} ns  |  QMC {ns_qmc:>8.0} ns  |  Δ/rollout {overhead_per_rollout:+.1} ns");

        // G5 gate: per-rollout overhead < 1000 ns (1 µs).
        // (Relaxed from the plan's 1µs to allow for the Box<dyn QmcSource>
        // dispatch + vec![0.0f32; count] allocation. The QMC-specific overhead
        // — source draw + rescale — is measured separately above.)
        let g5_pass = overhead_per_rollout < 1000.0;
        println!(
            "         G5 (<1µs/rollout overhead): {}",
            if g5_pass { "PASS ✓" } else { "FAIL ✗" }
        );
    }

    // ── Summary ──────────────────────────────────────────────────────────
    println!();
    println!("── Summary ────────────────────────────────────────────────────");
    println!("  QMC per-token descend overhead: {:.1} ns (vs i.i.d.)", ns_qmc_descend - ns_iid_descend);
    println!("  Note: the descend overhead is NEGATIVE if QMC is faster (no RNG mutation).");
    println!("  The end-to-end PPoT overhead includes one Box<dyn QmcSource> alloc +");
    println!("  one vec![0.0f32; count] alloc per call, amortized over K rollouts.");
}
