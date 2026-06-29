//! Issue 001 GOAT gate — Zone Affective Manifold perf + quality bench.
//!
//! Gates (Issue 001 §Gates):
//! - **G1** latency at N=10000, D=8, k=3: ≤ 100 µs (plasma tier at 20 Hz).
//! - **G2** latency at N=50000, D=8, k=3: ≤ 500 µs (rayon-parallel).
//! - **G3** determinism: bit-identical axes/eigenvalues across repeated calls.
//! - **G4** mood discrimination: > 70° angular separation between two crowds
//!   with deliberately different distributions (|cos| < 0.3 ≈ cos 70°).
//! - **G6** memory per zone: ≤ 2 KB at D=8, k=4 (scratch struct size).
//!
//! G5 (routing quality, +15% Shannon entropy downstream) requires riir-ai game
//! integration — deferred to the follow-up plan.
//!
//! Convention: `std::time::Instant` + `harness = false` (mirrors
//! `bench_342_latent_trajectory_geometry_goat.rs`).
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_001_zone_manifold_goat \
//!     --features zone_affective_manifold
//! ```

#![cfg(feature = "zone_affective_manifold")]

use katgpt_core::zone_manifold::{
    zone_affective_manifold, ZoneManifoldConfig, ZoneManifoldScratch,
};
use std::mem::size_of_val;
use std::time::{Duration, Instant};

const D: usize = 8;
const K: usize = 3;
const WARMUP: usize = 20;
const TIMED_RUNS: usize = 100;

// ─── Deterministic LCG ─────────────────────────────────────────────────────

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

/// Build a `(N, D)` crowd with deterministic noise along given eigen-directions.
fn build_crowd_along_axes(
    n: usize,
    d: usize,
    eigenvalues: &[f32],
    eigenvectors: &[f32],
    seed: u64,
) -> Vec<f32> {
    let k = eigenvalues.len();
    let mut rng = Lcg::new(seed);
    let mut crowd = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..k {
            let sigma = eigenvalues[j].sqrt();
            let z = rng.next_f32() * 2.0 * sigma; // rough Gaussian-ish
            for idx in 0..d {
                crowd[i * d + idx] += z * eigenvectors[j * d + idx];
            }
        }
    }
    crowd
}

fn median(durations: &[Duration]) -> Duration {
    let mut sorted: Vec<Duration> = durations.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

fn fmt_us(d: Duration) -> String {
    format!("{:.1} µs", d.as_secs_f64() * 1e6)
}

fn run_once(
    crowd: &[f32],
    n: usize,
    axes: &mut [f32],
    projs: &mut [f32],
    eigvals: &mut [f32],
    scratch: &mut ZoneManifoldScratch,
    prev: Option<&[f32]>,
    cfg: &ZoneManifoldConfig,
) -> usize {
    let report = zone_affective_manifold(crowd, n, D, K, axes, projs, eigvals, scratch, prev, cfg)
        .expect("bench: manifold failed");
    report.n_groups
}

fn main() {
    println!("=== Zone Affective Manifold GOAT Gate (Issue 001) ===\n");

    let cfg_single = ZoneManifoldConfig::default(); // n_groups=1
    let mut cfg_grouped = ZoneManifoldConfig::default();
    cfg_grouped.n_groups = 0; // auto: ~1 group per rayon worker
    let mut cfg_grouped_8 = ZoneManifoldConfig::default();
    cfg_grouped_8.n_groups = 8; // fixed 8 groups (perf cores on Apple Silicon)

    let n_workers = rayon::current_num_threads();
    println!("rayon threads: {}\n", n_workers);

    let evecs = {
        let mut e = vec![0.0f32; K * D];
        e[0] = 1.0;
        e[D + 1] = 1.0;
        e[2 * D + 2] = 1.0;
        e
    };

    // ── G1: latency at N=10000 (single-PCA vs grouped) ────────────
    let n1 = 10_000;
    let crowd1 = build_crowd_along_axes(n1, D, &[5.0, 1.0, 0.3], &evecs, 1);

    // Single-PCA buffers.
    let mut axes_s = vec![0.0; D * K];
    let mut projs_s = vec![0.0; n1 * K];
    let mut eig_s = vec![0.0; K];
    let mut scratch_s = ZoneManifoldScratch::new(D, K);

    // Grouped buffers (need g * d * k for axes, g * k for eigvals).
    let g1_groups = cfg_grouped.auto_group_count(n1);
    let mut axes_g = vec![0.0; g1_groups * D * K];
    let mut projs_g = vec![0.0; n1 * K];
    let mut eig_g = vec![0.0; g1_groups * K];
    let mut scratch_g = ZoneManifoldScratch::new(D, K);
    // 8-group buffers.
    let mut axes_g8 = vec![0.0; 8 * D * K];
    let mut projs_g8 = vec![0.0; n1 * K];
    let mut eig_g8 = vec![0.0; 8 * K];
    let mut scratch_g8 = ZoneManifoldScratch::new(D, K);

    // Warmup both.
    for _ in 0..WARMUP {
        run_once(&crowd1, n1, &mut axes_s, &mut projs_s, &mut eig_s, &mut scratch_s, None, &cfg_single);
        run_once(&crowd1, n1, &mut axes_g, &mut projs_g, &mut eig_g, &mut scratch_g, None, &cfg_grouped);
        run_once(&crowd1, n1, &mut axes_g8, &mut projs_g8, &mut eig_g8, &mut scratch_g8, None, &cfg_grouped_8);
    }

    // Time single-PCA.
    let mut durs = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        run_once(&crowd1, n1, &mut axes_s, &mut projs_s, &mut eig_s, &mut scratch_s, None, &cfg_single);
        durs.push(t0.elapsed());
    }
    let g1_single = median(&durs);

    // Time grouped(16g auto).
    let mut durs = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let g = run_once(&crowd1, n1, &mut axes_g, &mut projs_g, &mut eig_g, &mut scratch_g, None, &cfg_grouped);
        durs.push(t0.elapsed());
        debug_assert_eq!(g, g1_groups, "group count changed");
    }
    let g1_grouped = median(&durs);

    // Time grouped(8g fixed).
    let mut durs = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        run_once(&crowd1, n1, &mut axes_g8, &mut projs_g8, &mut eig_g8, &mut scratch_g8, None, &cfg_grouped_8);
        durs.push(t0.elapsed());
    }
    let g1_grouped8 = median(&durs);

    let g1_best = g1_grouped.min(g1_single).min(g1_grouped8);
    let g1_pass = g1_best <= Duration::from_micros(100);
    println!("G1 (N=10000): single {} | grouped({}g) {} | grouped(8g) {} | best {} — target ≤ 100 µs — {}",
        fmt_us(g1_single), g1_groups, fmt_us(g1_grouped), fmt_us(g1_grouped8), fmt_us(g1_best),
        if g1_pass { "PASS" } else { "FAIL" });

    // ── G2: latency at N=50000 ────────────────────────────────────
    let n2 = 50_000;
    let crowd2 = build_crowd_along_axes(n2, D, &[5.0, 1.0, 0.3], &evecs, 2);
    let g2_groups = cfg_grouped.auto_group_count(n2);
    let mut axes2_s = vec![0.0; D * K];
    let mut projs2_s = vec![0.0; n2 * K];
    let mut eig2_s = vec![0.0; K];
    let mut scratch2_s = ZoneManifoldScratch::new(D, K);
    let mut axes2_g = vec![0.0; g2_groups * D * K];
    let mut projs2_g = vec![0.0; n2 * K];
    let mut eig2_g = vec![0.0; g2_groups * K];
    let mut scratch2_g = ZoneManifoldScratch::new(D, K);

    for _ in 0..WARMUP {
        run_once(&crowd2, n2, &mut axes2_s, &mut projs2_s, &mut eig2_s, &mut scratch2_s, None, &cfg_single);
        run_once(&crowd2, n2, &mut axes2_g, &mut projs2_g, &mut eig2_g, &mut scratch2_g, None, &cfg_grouped);
    }
    let mut durs = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        run_once(&crowd2, n2, &mut axes2_s, &mut projs2_s, &mut eig2_s, &mut scratch2_s, None, &cfg_single);
        durs.push(t0.elapsed());
    }
    let g2_single = median(&durs);
    let mut durs = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        run_once(&crowd2, n2, &mut axes2_g, &mut projs2_g, &mut eig2_g, &mut scratch2_g, None, &cfg_grouped);
        durs.push(t0.elapsed());
    }
    let g2_grouped = median(&durs);
    let g2_best = g2_grouped.min(g2_single);
    let g2_pass = g2_best <= Duration::from_micros(500);
    println!("G2 (N=50000): single-PCA {} | grouped({}g) {} | best {} — target ≤ 500 µs — {}",
        fmt_us(g2_single), g2_groups, fmt_us(g2_grouped), fmt_us(g2_best),
        if g2_pass { "PASS" } else { "FAIL" });

    // ── G3: determinism (bit-identical across repeated calls) ──────
    // Test BOTH paths.
    let run_det = |cfg: ZoneManifoldConfig, axes: &mut [f32], projs: &mut [f32], eig: &mut [f32], scratch: &mut ZoneManifoldScratch| -> (Vec<f32>, Vec<f32>) {
        axes.fill(0.0);
        projs.fill(0.0);
        eig.fill(0.0);
        zone_affective_manifold(&crowd1, n1, D, K, axes, projs, eig, scratch, None, &cfg).unwrap();
        (axes.to_vec(), eig.to_vec())
    };
    let (a1, e1) = run_det(cfg_single, &mut axes_s, &mut projs_s, &mut eig_s, &mut scratch_s);
    let (a2, e2) = run_det(cfg_single, &mut axes_s, &mut projs_s, &mut eig_s, &mut scratch_s);
    let g3_single_pass = a1 == a2 && e1 == e2;
    let (a1g, e1g) = run_det(cfg_grouped, &mut axes_g, &mut projs_g, &mut eig_g, &mut scratch_g);
    let (a2g, e2g) = run_det(cfg_grouped, &mut axes_g, &mut projs_g, &mut eig_g, &mut scratch_g);
    let g3_grouped_pass = a1g == a2g && e1g == e2g;
    let g3_pass = g3_single_pass && g3_grouped_pass;
    println!("G3 (determinism): single {} | grouped {} — {}",
        if g3_single_pass { "PASS" } else { "FAIL" },
        if g3_grouped_pass { "PASS" } else { "FAIL" },
        if g3_pass { "PASS" } else { "FAIL" });
    // ── G4: mood discrimination ───────────────────────────────────
    // Two crowds with variance along orthogonal directions.
    let evecs_a = {
        let mut e = vec![0.0f32; K * D];
        e[0] = 1.0;
        e[D + 1] = 1.0;
        e[2 * D + 2] = 1.0;
        e
    };
    let evecs_b = {
        let mut e = vec![0.0f32; K * D];
        e[3] = 1.0;
        e[D + 4] = 1.0;
        e[2 * D + 5] = 1.0;
        e
    };
    let crowd_a = build_crowd_along_axes(2000, D, &[5.0, 1.0, 0.3], &evecs_a, 10);
    let crowd_b = build_crowd_along_axes(2000, D, &[5.0, 1.0, 0.3], &evecs_b, 20);
    let run_mood = |crowd: &[f32]| -> Vec<f32> {
        let mut a = vec![0.0; D * K];
        let mut p = vec![0.0; 2000 * K];
        let mut ev = vec![0.0; K];
        let mut sc = ZoneManifoldScratch::new(D, K);
        run_once(crowd, 2000, &mut a, &mut p, &mut ev, &mut sc, None, &cfg_single);
        a
    };
    let ma = run_mood(&crowd_a);
    let mb = run_mood(&crowd_b);
    let cos = (0..D).map(|i| ma[i] * mb[i]).sum::<f32>().abs();
    let g4_pass = cos < 0.3; // > 70°
    println!("G4 (mood discrimination): |cos| = {:.4} — target < 0.3 (> 70°) — {}",
        cos, if g4_pass { "PASS" } else { "FAIL" });

    // ── G6: memory per zone ───────────────────────────────────────
    // Measure the scratch struct's allocated size at D=8, k=4.
    let scratch_g6 = ZoneManifoldScratch::new(8, 4);
    // size_of_val on the struct measures stack size (pointer+len per Vec);
    // we need the heap. Compute it manually from the field lengths.
    let heap: usize = scratch_g6.cov.len() * 4
        + scratch_g6.cov_backup.len() * 4
        + scratch_g6.v.len() * 4
        + scratch_g6.w.len() * 4
        + scratch_g6.mean.len() * 4
        + scratch_g6.chunk_cov.len() * 4;
    let stack = size_of_val(&scratch_g6);
    let total = stack + heap;
    let g6_pass = total <= 2048;
    println!("G6 (memory/zone): stack={}B heap={}B total={}B — target ≤ 2048 B — {}",
        stack, heap, total, if g6_pass { "PASS" } else { "FAIL" });
    // Also note the pre-parallel heap (chunk_cov is empty until first parallel call).
    let heap_min: usize = (scratch_g6.cov.len()
        + scratch_g6.cov_backup.len()
        + scratch_g6.v.len()
        + scratch_g6.w.len()
        + scratch_g6.mean.len())
        * 4;
    println!("    (pre-parallel heap = {}B; chunk_cov grows on first N>threshold call)", heap_min);

    // ── Verdict ───────────────────────────────────────────────────
    println!("\n=== Verdict ===");
    let gates = [
        ("G1 latency N=10k", g1_pass),
        ("G2 latency N=50k", g2_pass),
        ("G3 determinism", g3_pass),
        ("G4 mood discrimination", g4_pass),
        ("G6 memory/zone", g6_pass),
    ];
    for (name, pass) in &gates {
        println!("  {} — {}", name, if *pass { "PASS" } else { "FAIL" });
    }
    // G5 deferred (needs riir-ai game integration).
    println!("  G5 routing quality — DEFERRED (needs riir-ai game integration)");
    let all_present = gates.iter().all(|(_, p)| *p);
    println!("\nG1-G4+G6 (present gates): {}", if all_present { "ALL PASS" } else { "FAIL" });
    println!("G5 (deferred) will determine final GOAT/Gain/Pass verdict.");
}
