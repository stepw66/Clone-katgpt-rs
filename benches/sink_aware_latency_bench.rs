//! Sink-Aware Attention dual-policy latency benchmark (Plan 287 Phase 3, T3.5
//! + Issue 001 extensions + Plan 288 flat-layout variants).
//!
//! Compares policies at `n ∈ {128, 512}`, `d_h = 64`, on two O regimes:
//!
//! - `rank1`  — `O = a_s · v_s^T` (Broadcast head). Cosine probe fires;
//!   stable-rank computation is skipped. Common case in trained transformers.
//! - `random` — `O` filled with i.i.d. noise. Cosine probe fails; full power
//!   iteration runs. Worst case for the classifier.
//!
//! Policies measured:
//! - `uniform`       — single n·d copy. Baseline.
//! - `dual`          — per-call classifier, `&[Vec<f32>]` layout (Plan 287).
//! - `dual_flat`     — per-call classifier, flat `&[f32]` layout (Plan 288).
//! - `dual_cached`   — audit cadence 16, `&[Vec<f32>]` layout (Issue 001).
//! - `dual_cached_flat` — audit cadence 16, flat `&[f32]` layout (Plan 288).
//!
//! Plan 287 target: ≤5% overhead for `dual` vs `uniform`.
//! Issue 001 finding: per-call target structurally infeasible (memory-bandwidth
//! bound). Cached variant is the realistic answer.
//! Plan 288 hypothesis: flat variants ≥ Vec<Vec<f32>> due to cache locality,
//! especially on the `random` regime (no cosine-probe short-circuit).
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
    apply_dual_policy_gate, apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat,
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

/// Flatten a Vec<Vec<f32>> row-major into a single contiguous Vec<f32>.
fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(rows.len() * d);
    for r in rows {
        out.extend_from_slice(r);
    }
    out
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

fn pct(measured: f64, baseline: f64) -> f64 {
    if baseline > 0.0 {
        100.0 * (measured - baseline) / baseline
    } else {
        0.0
    }
}

struct CaseData {
    d: usize,
    attn_rows: Vec<Vec<f32>>,
    values_rows: Vec<Vec<f32>>,
    o_rows: Vec<Vec<f32>>,
    attn_flat: Vec<f32>,
    values_flat: Vec<f32>,
    o_flat: Vec<f32>,
}

fn build_case(n: usize, d: usize, regime: &str) -> CaseData {
    // Values: always equal-norm content vectors so Broadcast detection works.
    let v_s: Vec<f32> = (0..d).map(|i| 0.1 * (i as f32).sin() + 0.5).collect();
    let values_rows: Vec<Vec<f32>> = (0..n).map(|_| v_s.clone()).collect();
    let o_rows: Vec<Vec<f32>> = match regime {
        "rank1" => (0..n).map(|_| v_s.clone()).collect(),
        "random" => rand_matrix(n, d, 0xDEAD_BEEF),
        _ => panic!("unknown regime: {regime}"),
    };
    // Attention map with dominant sink at pos 0.
    let attn_rows: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let mut row = vec![0.1 / (n as f32 - 1.0); n];
            row[0] = 0.9;
            row
        })
        .collect();

    let attn_flat = flatten(&attn_rows);
    let values_flat = flatten(&values_rows);
    let o_flat = flatten(&o_rows);

    CaseData {
        d,
        attn_rows,
        values_rows,
        o_rows,
        attn_flat,
        values_flat,
        o_flat,
    }
}

fn run_regime(regime: &str, n_values: &[usize], cfg: SinkClassifierConfig) {
    println!("── regime: {regime} ──────────────────────────────────────────────");
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>8} {:>10} {:>8}",
        "n", "uniform", "dual", "dual_flat", "oh%", "cached", "oh%"
    );
    println!("{}", "-".repeat(75));

    for &n in n_values {
        let cd = build_case(n, 64, regime);
        let policy_uniform = SinkAwarePolicy::Uniform;
        let policy_dual = SinkAwarePolicy::DualPolicy(cfg);

        let mut out_rows: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0; cd.d]).collect();
        let mut out_flat = vec![0.0f32; n * cd.d];
        let mut scratch = StableRankScratch::new(cd.d);

        // ── Uniform baseline (Vec<Vec> copy) ────────────────────────
        let us_uniform = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate(
                &cd.attn_rows,
                &cd.values_rows,
                &cd.o_rows,
                &policy_uniform,
                0.0,
                &mut scratch,
                &mut out_rows,
            );
            std::hint::black_box(kind);
        });

        // ── Per-call DualPolicy, Vec<Vec<f32>> ──────────────────────
        let us_dual = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate(
                &cd.attn_rows,
                &cd.values_rows,
                &cd.o_rows,
                &policy_dual,
                0.0,
                &mut scratch,
                &mut out_rows,
            );
            std::hint::black_box(kind);
        });

        // ── Per-call DualPolicy, flat &[f32] (Plan 288) ─────────────
        let us_dual_flat = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate_flat(
                &cd.attn_flat,
                &cd.values_flat,
                &cd.o_flat,
                n,
                cd.d,
                &policy_dual,
                0.0,
                &mut scratch,
                &mut out_flat,
            );
            std::hint::black_box(kind);
        });

        // ── Cached DualPolicy (cadence 16) — Vec<Vec> vs flat ──────
        // We measure flat-only for the cached path since steady-state is
        // a copy + (conditional) scale in both layouts; the difference is
        // the same as in the per-call case. The cached_us column below is
        // the flat cached variant, which is the production path going
        // forward.
        let mut cached = CachedSinkClassification::with_config(cfg, 16);
        apply_dual_policy_gate_cached_flat(
            &cd.attn_flat,
            &cd.values_flat,
            &cd.o_flat,
            n,
            cd.d,
            0.0,
            &mut scratch,
            &mut cached,
            &mut out_flat,
        );
        let us_cached_flat = bench_us(3, 30, || {
            let kind = apply_dual_policy_gate_cached_flat(
                &cd.attn_flat,
                &cd.values_flat,
                &cd.o_flat,
                n,
                cd.d,
                0.0,
                &mut scratch,
                &mut cached,
                &mut out_flat,
            );
            std::hint::black_box(kind);
        });

        println!(
            "{:>5} {:>10.3} {:>10.3} {:>10.3} {:>7.1}% {:>10.3} {:>7.1}%",
            n,
            us_uniform,
            us_dual,
            us_dual_flat,
            pct(us_dual_flat, us_uniform),
            us_cached_flat,
            pct(us_cached_flat, us_uniform),
        );
    }
    println!();
}

fn main() {
    println!("=== Sink-Aware Dual-Policy Latency Benchmark ===");
    println!("=== (Plan 287 T3.5 + Issue 001 + Plan 288 flat) ===\n");

    let d = 64usize;
    let n_values: &[usize] = &[128, 512];
    let cfg = SinkClassifierConfig::default();
    let _ = d;

    run_regime("rank1", n_values, cfg);
    run_regime("random", n_values, cfg);

    println!("G3 target: overhead ≤5% (DualPolicy vs Uniform).");
    println!("Issue 001: per-call path structurally misses (memory-bandwidth bound).");
    println!("           cached (cadence=16) hits ≤5% in steady state.");
    println!("Plan 288: flat layout should match or beat Vec<Vec<f32>> on both");
    println!("          regimes; largest gain expected on 'random' (no cosine probe).");
}
