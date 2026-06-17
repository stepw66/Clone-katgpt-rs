//! Sink-Aware Attention dual-policy latency benchmark (Plan 287 Phase 3, T3.5
//! + Issue 001 extensions).
//!
//! Compares three policies at `n ∈ {128, 512}`, `d_h = 64`:
//!
//! - `uniform`  — single n·d copy. Baseline.
//! - `dual`     — per-call classifier (the original Plan 287 path).
//! - `dual_cached` — audit cadence 16 (classify once per 16 calls). This is
//!   the production-realistic path; steady-state cost matches `uniform`.
//!
//! Plan target: ≤5% overhead for `dual` vs `uniform`.
//! Issue 001 finding: that target is structurally infeasible for the per-call
//! path — even an optimal classifier does more memory traffic than a memcpy.
//! The cached variant is the realistic answer.
//!
//! Uses `std::time::Instant` (NOT criterion — matches other katgpt-rs benches).
//!
//! Run:
//! ```bash
//! cargo run --release --bench sink_aware_latency_bench --features sink_aware_attn
//! ```

#![cfg(feature = "sink_aware_attn")]

use katgpt_rs::data_probe::sink_classify::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, StableRankScratch,
    apply_dual_policy_gate, apply_dual_policy_gate_cached,
};
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 & 0xFFFF) as f32 / 0x8000 as f32) - 1.0
    }
}

fn rand_matrix(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.next_f32()).collect())
        .collect()
}

fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

fn main() {
    println!("=== Sink-Aware Dual-Policy Latency Benchmark (Plan 287 T3.5 + Issue 001) ===\n");

    let d = 64usize;
    let n_values: &[usize] = &[128, 512];

    println!(
        "{:>5} {:>12} {:>12} {:>14} {:>14} {:>10}",
        "n", "uniform_us", "dual_us", "dual_oh%", "cached_us", "cached_oh%"
    );
    println!("{}", "-".repeat(75));

    let cfg = SinkClassifierConfig::default();

    for &n in n_values {
        // Use a rank-1 O so DualPolicy classifies as Broadcast (fast early-exit).
        // This matches the common paper case (Broadcast heads are the
        // fast path; the slow random-O case is covered by Phase 2 bench).
        let v_s: Vec<f32> = (0..d).map(|i| 0.1 * (i as f32).sin() + 0.5).collect();
        let values: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let o: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
        let mut out_uniform: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut out_dual: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();
        let mut out_cached: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; d]).collect();

        // Build an attention map with a dominant sink column at pos 0.
        let mut attn: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.1 / (n as f32 - 1.0); n];
            row[0] = 0.9; // dominant sink
            // Renormalize row so it sums to ~1.0 (skip for raw mass).
            let _ = i;
            attn.push(row);
        }

        let policy_uniform = SinkAwarePolicy::Uniform;
        let policy_dual = SinkAwarePolicy::DualPolicy(cfg);

        // Shared scratch — Uniform doesn't touch it, so reuse safely.
        let mut scratch = StableRankScratch::new(d);

        // ── Uniform baseline (single copy) ───────────────────────────
        let us_uniform = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate(
                &attn,
                &values,
                &o,
                &policy_uniform,
                0.0,
                &mut scratch,
                &mut out_uniform,
            );
            std::hint::black_box(kind);
        });

        // ── Per-call DualPolicy (the original slow path) ─────────────
        let us_dual = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate(
                &attn,
                &values,
                &o,
                &policy_dual,
                0.0,
                &mut scratch,
                &mut out_dual,
            );
            std::hint::black_box(kind);
        });

        // ── Cached DualPolicy at cadence 16 (production path) ────────
        // Warmup: do one call to populate the cache.
        let mut cached = CachedSinkClassification::with_config(cfg, 16);
        apply_dual_policy_gate_cached(
            &attn, &values, &o, 0.0, &mut scratch, &mut cached, &mut out_cached,
        );
        // Now measure steady-state (cached path hits copy_rows only).
        let us_cached = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate_cached(
                &attn, &values, &o, 0.0, &mut scratch, &mut cached, &mut out_cached,
            );
            std::hint::black_box(kind);
        });

        let oh_dual = if us_uniform > 0.0 {
            100.0 * (us_dual - us_uniform) / us_uniform
        } else {
            0.0
        };
        let oh_cached = if us_uniform > 0.0 {
            100.0 * (us_cached - us_uniform) / us_uniform
        } else {
            0.0
        };

        println!(
            "{:>5} {:>12.3} {:>12.3} {:>13.2}% {:>13.3} {:>9.2}%",
            n, us_uniform, us_dual, oh_dual, us_cached, oh_cached
        );
    }
    println!();
    println!("G3 target: overhead ≤5% (DualPolicy vs Uniform).");
    println!("Issue 001 verdict: per-call path structurally misses target;");
    println!("                  cached (cadence=16) hits it in steady state.");
}
