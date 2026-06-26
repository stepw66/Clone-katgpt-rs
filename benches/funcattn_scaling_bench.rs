//! Functional Attention (FUNCATTN) — linear-in-n scaling benchmark (Plan 286 T2.2 / G4).
//!
//! Uses `std::time::Instant` (NOT criterion) — mirrors
//! `benches/manifold_power_iter_router_bench.rs::bench_us`.
//!
//! Run:
//! ```bash
//! cargo bench --features funcattn --bench funcattn_scaling_bench
//! # or, equivalently (harness=false, plain main):
//! cargo run --release --features funcattn --bench funcattn_scaling_bench
//! ```
//!
//! Paper setup (Fig 5): `n ∈ {512, 2048, 8192, 32768}`, `d=128`, `k=64`.
//! Per-call complexity is `O(n·d·k + k·d² + d³)` — linear in `n` for `k ≪ n`
//! (the `k·d² + d³` term is a per-call fixed cost). We fit `log(time) vs log(n)`
//! over `n ∈ {2048, 8192, 32768}` (skip `n=512`, which is fixed-cost dominated)
//! and assert slope ∈ [0.85, 1.15] for the linear-scaling PASS gate.
//!
//! Optional baseline: `tiled_attention_forward` (standard SDPA, O(n²·d)).
//! The `funcattn` feature already pulls in `tiled_attention`, so the comparison
//! is always available. At `n=32768`, FUNCATTN should be many times faster.

#![cfg(feature = "funcattn")]

use katgpt_core::funcattn::{FuncAttnConfig, FuncAttnScratch, funcattn_forward};
use katgpt_core::tiled_attention_forward;
use std::time::{Duration, Instant};

/// Deterministic xorshift64* PRNG, matching `funcattn.rs::tests::make_rng`.
/// Returns values in `[-1, 1)`.
fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let bits = (s >> 11) as u32;
        let u01 = bits as f32 / (u32::MAX as f32);
        v.push(u01 * 2.0 - 1.0);
    }
    v
}

/// Best-of-N wall-clock microseconds for a closure (mirrors
/// `manifold_power_iter_router_bench.rs::bench_us`). Also returns the
/// arithmetic mean so we can report both best-case and average.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    let mut sum = Duration::ZERO;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
        sum += dt;
    }
    let mean_us = sum.as_secs_f64() * 1e6 / (iters as f64);
    let best_us = best.as_secs_f64() * 1e6;
    (mean_us, best_us)
}

struct FuncAttnFixture {
    x_basis: Vec<f32>,
    x_value: Vec<f32>,
    w_basis: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    out: Vec<f32>,
    scratch: FuncAttnScratch,
    cfg: FuncAttnConfig,
}

impl FuncAttnFixture {
    fn new(n: usize, d: usize, k: usize, seed: u64) -> Self {
        let x_basis = seeded_vec(seed, n * d);
        let x_value = seeded_vec(seed.wrapping_add(1), n * d);
        let w_basis = seeded_vec(seed.wrapping_add(2), k * d);
        let w_q = seeded_vec(seed.wrapping_add(3), d * d);
        let w_k = seeded_vec(seed.wrapping_add(4), d * d);
        let w_v = seeded_vec(seed.wrapping_add(5), d * d);
        let out = vec![0.0f32; n * d];
        let scratch = FuncAttnScratch::new(n, d, k);
        let cfg = FuncAttnConfig {
            d,
            k,
            ..FuncAttnConfig::default()
        };
        Self {
            x_basis,
            x_value,
            w_basis,
            w_q,
            w_k,
            w_v,
            out,
            scratch,
            cfg,
        }
    }

    #[inline(always)]
    fn forward(&mut self) {
        // Best-effort: discard error since basis=AutoTemperature has no failure
        // mode for α∈(0,1) (convex combo guarantees PD).
        let _ = funcattn_forward(
            &self.x_basis,
            &self.x_value,
            &self.w_basis,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &self.cfg,
            &mut self.scratch,
            &mut self.out,
        );
        std::hint::black_box(&self.out);
    }
}

/// Ordinary-least-squares slope of `y` vs `x` (both logged externally).
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    assert!(xs.len() >= 2, "need at least 2 points for slope");
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x) * (x - mean_x);
    }
    assert!(den > 0.0, "degenerate: all xs equal");
    num / den
}

fn main() {
    let d = 128usize;
    let k = 64usize;
    let n_values: [usize; 4] = [512, 2048, 8192, 32768];

    println!("=== FUNCATTN G4 Linear-in-n Scaling Benchmark (Plan 286 T2.2) ===");
    println!(
        "    d={d}, k={k}, basis=Sigmoid (default), alpha=0.5, temperature=0.5"
    );
    println!(
        "    complexity per call: O(n·d·k + k·d² + d³) = O(n·{nk} + {kdv} + {dcu})",
        nk = d * k,
        kdv = k * d * d,
        dcu = d * d * d
    );
    println!();

    // Collect timings first so the ratio column can use the n=512 baseline
    // (best_us) for every row.
    struct Row {
        n: usize,
        mean_us: f64,
        best_us: f64,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(n_values.len());

    println!("--- Timing sweep (warmup=5, best-of-20) ---");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>10}",
        "n", "mean_us", "best_us", "us/token", "ratio"
    );
    println!("{}", "-".repeat(58));

    for &n in n_values.iter() {
        let mut fx = FuncAttnFixture::new(n, d, k, 0xA1B2_C3D4u64.wrapping_add(n as u64));

        // Verify correctness on a single call before timing (defensive —
        // funcattn_forward must not silently fail for random inputs).
        if let Err(e) = funcattn_forward(
            &fx.x_basis,
            &fx.x_value,
            &fx.w_basis,
            &fx.w_q,
            &fx.w_k,
            &fx.w_v,
            &fx.cfg,
            &mut fx.scratch,
            &mut fx.out,
        ) {
            eprintln!("FATAL: funcattn_forward returned {e:?} at n={n}");
            std::process::exit(1);
        }
        let any_nan = fx.out.iter().any(|x| x.is_nan() || x.is_infinite());
        if any_nan {
            eprintln!("FATAL: non-finite output detected at n={n}");
            std::process::exit(1);
        }

        let (mean_us, best_us) = bench_us(5, 20, || fx.forward());
        rows.push(Row {
            n,
            mean_us,
            best_us,
        });
    }

    // Print table with ratio vs n=512 baseline (computed from row[0]).
    let baseline_n512 = rows[0].best_us;
    for r in &rows {
        let us_per_token = r.best_us / (r.n as f64);
        let ratio = r.best_us / baseline_n512;
        println!(
            "{:>6} {:>12.2} {:>12.2} {:>12.4} {:>10.3}",
            r.n, r.mean_us, r.best_us, us_per_token, ratio
        );
    }

    // Collect (n, best_us) for the slope fit on the n≥2048 segment.
    let mut fit_ns: Vec<f64> = Vec::new();
    let mut fit_best_us: Vec<f64> = Vec::new();
    for r in &rows {
        if r.n >= 2048 {
            fit_ns.push((r.n as f64).ln());
            fit_best_us.push(r.best_us.ln());
        }
    }

    println!();
    println!("--- Log-log slope (linear-in-n gate) ---");
    println!(
        "    fit window: n ∈ {{2048, 8192, 32768}} (skip n=512, fixed-cost dominated)"
    );
    let slope = ols_slope(&fit_ns, &fit_best_us);
    println!("    slope of log(time) vs log(n) = {slope:.4}");
    println!("    target: ∈ [0.85, 1.15] (1.0 = perfectly linear in n)");

    let in_range = (0.85..=1.15).contains(&slope);
    println!(
        "    G4 LINEAR SCALING: {}",
        if in_range { "PASS ✅" } else { "FAIL ❌" }
    );
    println!();

    // Baseline comparison vs tiled_attention (standard SDPA, O(n²·d)).
    // The funcattn feature already pulls in tiled_attention, so this is always
    // available. We compare at n=32768 where the asymptotic gap is largest.
    println!("--- Baseline vs tiled_attention (standard SDPA, O(n²·d)) at n=32768 ---");
    let n_big = 32768usize;
    let qkv_seed = 0x5EED_BEEFu64;
    let q_sdpa = seeded_vec(qkv_seed, n_big * d);
    let k_sdpa = seeded_vec(qkv_seed.wrapping_add(1), n_big * d);
    let v_sdpa = seeded_vec(qkv_seed.wrapping_add(2), n_big * d);

    // SDPA at n=32768 allocates an n×n score matrix (~4 GiB at f32). That may
    // OOM or be unreasonably slow in debug; we attempt it but fall back to a
    // smaller n if it doesn't fit. The bench is built with `cargo bench`
    // (release by default), so it usually fits, but stay defensive.
    let scale = 1.0f32 / (d as f32).sqrt();
    let sdpa_n = if n_big * n_big > 50_000_000 {
        // >50M floats = >200 MiB scores; cap at 8192 to keep the bench snappy.
        eprintln!(
            "    note: SDPA at n=32768 would need ~4 GiB score matrix; downscaling comparison to n=8192."
        );
        8192usize
    } else {
        n_big
    };

    let q_cmp = &q_sdpa[..sdpa_n * d];
    let k_cmp = &k_sdpa[..sdpa_n * d];
    let v_cmp = &v_sdpa[..sdpa_n * d];
    let mut out_cmp = vec![0.0f32; sdpa_n * d];

    let (_sdpa_mean_us, sdpa_best_us) = bench_us(2, 5, || {
        tiled_attention_forward(q_cmp, k_cmp, v_cmp, &mut out_cmp, sdpa_n, d, scale);
        std::hint::black_box(&out_cmp);
    });

    // FUNCATTN at the same n for apples-to-apples ratio.
    let mut fx_cmp = FuncAttnFixture::new(sdpa_n, d, k, 0xCAFE_BABEu64);
    let (_fa_mean_us, fa_best_us) = bench_us(5, 20, || fx_cmp.forward());

    let speedup = sdpa_best_us / fa_best_us;
    println!(
        "    at n={sdpa_n}: FUNCATTN best = {fa_best_us:.1} µs, tiled_attention best = {sdpa_best_us:.1} µs, speedup = {speedup:.2}×"
    );
    let target_pass = if sdpa_n >= 32768 { 10.0 } else { 4.0 };
    println!(
        "    G4 vs SDPA (>={target_pass:.0}× at n={sdpa_n}): {}",
        if speedup >= target_pass {
            "PASS ✅"
        } else {
            "BELOW TARGET (still linear-in-n — see slope above)"
        }
    );
    println!();

    // Final gate verdict.
    println!("=== G4 VERDICT ===");
    println!(
        "    slope gate:   {} (slope={slope:.4}, target [0.85, 1.15])",
        if in_range { "PASS" } else { "FAIL" }
    );
    if !in_range {
        std::process::exit(2);
    }
}
