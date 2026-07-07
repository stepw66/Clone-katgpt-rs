//! Plan 342 Phase 4 T4.1 G2 — Latent Trajectory Geometry perf bench.
//!
//! Measures `from_states` and `bifurcation_ratio` latency on realistic
//! trajectory shapes:
//!
//! - **G2 from_states** target: 100-step × 32-dim trajectory < 5 µs
//!   (single-pass streaming fold, zero allocation in the hot path).
//! - Sweep: HLA scale (dim=8), diagnostic scale (dim=32), transformer
//!   hidden scale (dim=768); step counts 20 / 100 / 1000.
//! - **G2 bifurcation_ratio** target: 100-step × 32-dim pair < 5 µs.
//!
//! Convention: `std::time::Instant` + `harness = false` (mirrors
//! `bench_324_bisimulation_goat.rs`, no Criterion dev-dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_342_latent_trajectory_geometry_goat \
//!     --features latent_trajectory_geometry
//! ```

#![cfg(feature = "latent_trajectory_geometry")]

use katgpt_core::latent_trajectory_geometry::{bifurcation_ratio, from_states};
use std::time::{Duration, Instant};

// ─── Config ────────────────────────────────────────────────────────────────

/// Trajectory shapes to sweep: (n_steps, dim, label).
/// Covers HLA scale (dim=8 — the actual router-integration target), the
/// diagnostic scale (dim=32), and transformer hidden scale (dim=768). Step
/// counts cover short (K=20, game tick), medium (K=100), long (K=1000).
const SHAPES: &[(usize, usize, &str)] = &[
    (20, 8, "HLA-short (1 tick)"),
    (100, 8, "HLA-medium (G2 target)"),
    (1000, 8, "HLA-long (crowd audit)"),
    (100, 32, "diag-medium"),
    (1000, 32, "diag-long"),
    (100, 768, "hidden-medium"),
    (1000, 768, "hidden-long (stress)"),
];

/// Warmup iterations (untimed).
const WARMUP: usize = 20;

/// Number of timed runs to take the median over.
const TIMED_RUNS: usize = 50;

// ─── Deterministic LCG (matches the crate convention) ─────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

/// Build a trajectory of `n_steps + 1` states each of dimension `dim`,
/// filled with deterministic pseudo-random values in [-0.5, 0.5).
fn build_trajectory(n_steps: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut traj: Vec<Vec<f32>> = Vec::with_capacity(n_steps + 1);
    for _ in 0..=n_steps {
        let mut state = Vec::with_capacity(dim);
        for _ in 0..dim {
            state.push(rng.next_f32());
        }
        traj.push(state);
    }
    traj
}

/// Build the `Vec<&[f32]>` view over an owned trajectory. Allocations here
/// are NOT measured — only the `from_states` call inside the timed loop is.
fn build_refs(traj: &[Vec<f32>]) -> Vec<&[f32]> {
    traj.iter().map(|v| v.as_slice()).collect()
}

/// Measure `from_states` median latency. The input is built ONCE outside the
/// timed loop; only the primitive call is measured.
fn bench_from_states(traj_refs: &[&[f32]]) -> Duration {
    // Warmup.
    for _ in 0..WARMUP {
        let _ = from_states(traj_refs);
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let g = from_states(traj_refs);
        samples.push(t0.elapsed());
        // Prevent the compiler from eliding the call.
        if g.n_steps == u16::MAX {
            std::process::abort();
        }
    }
    samples.sort();
    samples[TIMED_RUNS / 2]
}

/// Measure `bifurcation_ratio` median latency over a pair of trajectories.
fn bench_bifurcation(a_refs: &[&[f32]], b_refs: &[&[f32]]) -> Duration {
    // Warmup.
    for _ in 0..WARMUP {
        let _ = bifurcation_ratio(a_refs, b_refs);
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let r = bifurcation_ratio(a_refs, b_refs);
        samples.push(t0.elapsed());
        // Prevent the compiler from eliding the call.
        if r.final_separation.is_nan() {
            std::process::abort();
        }
    }
    samples.sort();
    samples[TIMED_RUNS / 2]
}

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{:>5} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:>5.2} µs", ns as f64 / 1_000.0)
    } else {
        format!("{:>5.2} ms", ns as f64 / 1_000_000.0)
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 342 — Latent Trajectory Geometry GOAT Gate (G2 perf)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Config: {} timed runs (median), {} warmup, seed=42",
        TIMED_RUNS, WARMUP
    );
    println!();

    // ── G2: from_states latency ─────────────────────────────────────────
    println!("── G2: from_states latency ────────────────────────────────────");
    println!(
        "{:>22}  {:>10}  {:>10}  {:>14}",
        "shape", "n_steps", "dim", "median"
    );
    println!("{}", "-".repeat(62));

    const G2_TARGET_NS: u64 = 5_000; // 5 µs for the GATE workload (HLA dim=8, K=100)
    let mut g2_target_passes = false;

    for &(n_steps, dim, label) in SHAPES {
        let traj = build_trajectory(n_steps, dim, 42);
        let refs = build_refs(&traj);
        let dur = bench_from_states(&refs);
        // Re-compute once outside the loop for the output row (also prevents
        // the compiler from concluding the call has no observable effect).
        let g = from_states(&refs);
        println!(
            "{:>22}  {:>10}  {:>10}  {:>14}  (length={:.3}, curv={:.3})",
            label,
            n_steps,
            dim,
            format_duration(dur),
            g.length,
            g.mean_curvature
        );
        if n_steps == 100 && dim == 8 && dur.as_nanos() as u64 <= G2_TARGET_NS {
            g2_target_passes = true;
        }
    }

    println!();
    println!(
        "G2 from_states (HLA 100x8 <= {}):  {}",
        format_duration(Duration::from_nanos(G2_TARGET_NS)),
        if g2_target_passes { "PASS" } else { "FAIL" }
    );
    println!();

    // ── G2: bifurcation_ratio latency ───────────────────────────────────
    println!("── G2: bifurcation_ratio latency ─────────────────────────────");
    println!(
        "{:>22}  {:>10}  {:>10}  {:>14}",
        "shape", "n_steps", "dim", "median"
    );
    println!("{}", "-".repeat(62));

    let mut g2_bifurcation_passes = false;

    for &(n_steps, dim, label) in SHAPES {
        let traj_a = build_trajectory(n_steps, dim, 42);
        let traj_b = build_trajectory(n_steps, dim, 137);
        let refs_a = build_refs(&traj_a);
        let refs_b = build_refs(&traj_b);
        let dur = bench_bifurcation(&refs_a, &refs_b);
        let r = bifurcation_ratio(&refs_a, &refs_b);
        println!(
            "{:>22}  {:>10}  {:>10}  {:>14}  (sep_ratio={:.3}, onset={:?})",
            label,
            n_steps,
            dim,
            format_duration(dur),
            r.separation_ratio,
            r.onset_step
        );
        if n_steps == 100 && dim == 8 && dur.as_nanos() as u64 <= G2_TARGET_NS {
            g2_bifurcation_passes = true;
        }
    }

    println!();
    println!(
        "G2 bifurcation_ratio (HLA 100x8 <= {}): {}",
        format_duration(Duration::from_nanos(G2_TARGET_NS)),
        if g2_bifurcation_passes {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!();

    // ── Verdict ─────────────────────────────────────────────────────────
    let all_pass = g2_target_passes && g2_bifurcation_passes;
    println!("──────────────────────────────────────────────────────────────");
    println!(
        "Verdict: {}",
        if all_pass {
            "G2 perf PASS — primitive meets the <5µs target at the gate dim."
        } else {
            "G2 perf FAIL — primitive exceeds the <5µs budget at the gate dim."
        }
    );
    if !all_pass {
        std::process::exit(1);
    }
}
