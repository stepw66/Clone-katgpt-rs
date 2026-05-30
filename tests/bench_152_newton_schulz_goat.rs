#![cfg(all(feature = "newton_schulz", feature = "river_valley"))]
//! GOAT Benchmark Test — Newton-Schulz Orthogonalization + Muon Momentum + River-Valley Diagnostics (Plan 152)
//!
//! Validates all 9 GOAT criteria from Plan 152:
//! 1. T1.1 Orthogonality: X@X^T approximately I for random matrices
//! 2. T1.2 Convergence: 5 iterations sufficient for well-conditioned matrices
//! 3. T1.3 Non-square: Correct transpose handling (12×6, 6×12, 64×16)
//! 4. T2.1 Subspace ratios: r_dom² + r_bulk² = 1.0
//! 5. T2.2 Effective rank: Known matrix → correct rank
//! 6. T2.3 Cosine similarity: Known directions → correct values
//! 7. T3.1 Muon output: Approximately orthogonal, no NaN/Inf
//! 8. T3.2 Momentum: Accumulating magnitude over steps
//! 9. T3.3 Throughput: Scratch API parity with allocating API
//!
//! Note on orthogonality thresholds: Newton-Schulz with 5 fixed iterations and
//! coefficients a=3.4445, b=-4.7750, c=2.0315 converges singular values to [0.68, 1.12]
//! (Keller Jordan's Muon blog). This produces *approximate* orthogonalization, not exact.
//! The orthogonality error (max |X@X^T - I|) is typically 0.2–0.5 for 5 iterations,
//! which is sufficient for Muon optimizer use — the key property is that the update
//! direction is well-conditioned, not that it's exactly on the Stiefel manifold.
//!
//! Run: `cargo test --features "newton_schulz,river_valley" --test bench_152_newton_schulz_goat -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::newton_schulz::{
    NewtonSchulzScratch, muon_update, muon_update_into, newton_schulz5, newton_schulz5_into,
};
use katgpt_rs::river_valley;

const WARMUP: usize = 10;
const ITERS: usize = 100;

// ── Helpers ───────────────────────────────────────────────────

/// Generate a simple pseudo-random matrix with a fixed seed.
fn seeded_random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut s = seed;
    let mut mat = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0;
        mat.push(v);
    }
    mat
}

/// Max absolute error between X @ X^T and the identity matrix.
fn orthogonality_error(x: &[f32], m: usize, n: usize) -> f32 {
    let mut max_err = 0.0f32;
    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0f32;
            for k in 0..n {
                dot += x[i * n + k] * x[j * n + k];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            max_err = max_err.max((dot - expected).abs());
        }
    }
    max_err
}

/// Compute X^T @ X and return max error from identity (for non-square m > n).
fn orthogonality_error_transpose(x: &[f32], m: usize, n: usize) -> f32 {
    let mut max_err = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0f32;
            for k in 0..m {
                dot += x[k * n + i] * x[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            max_err = max_err.max((dot - expected).abs());
        }
    }
    max_err
}

// ── T1.1: Orthogonality ──────────────────────────────────────
//
// Newton-Schulz with 5 iterations produces approximately orthogonal output.
// Singular values converge to [0.68, 1.12] — diagonal entries of X@X^T are
// in that range, not exactly 1.0. Off-diagonal entries should be small (< 0.5).

#[test]
fn goat_t1_1_orthogonality_64x64() {
    let g = seeded_random_matrix(42, 64, 64);
    let mut out = vec![0.0f32; 64 * 64];
    newton_schulz5(&g, 64, 64, &mut out);

    let err = orthogonality_error(&out, 64, 64);
    println!("T1.1: 64×64 orthogonality error = {err:.6e}");
    // 5-iteration Newton-Schulz: singular values converge to [0.68, 1.12]
    assert!(
        err < 0.5,
        "X@X^T should be approximately I, max error = {err}"
    );
}

#[test]
fn goat_t1_1_orthogonality_32x32() {
    let g = seeded_random_matrix(123, 32, 32);
    let mut out = vec![0.0f32; 32 * 32];
    newton_schulz5(&g, 32, 32, &mut out);

    let err = orthogonality_error(&out, 32, 32);
    println!("T1.1: 32×32 orthogonality error = {err:.6e}");
    assert!(
        err < 0.5,
        "X@X^T should be approximately I, max error = {err}"
    );
}

#[test]
fn goat_t1_1_orthogonality_8x8() {
    let g = seeded_random_matrix(42, 8, 8);
    let mut out = vec![0.0f32; 64];
    newton_schulz5(&g, 8, 8, &mut out);

    let err = orthogonality_error(&out, 8, 8);
    println!("T1.1: 8×8 orthogonality error = {err:.6e}");
    assert!(
        err < 0.5,
        "X@X^T should be approximately I, max error = {err}"
    );
}

#[test]
fn goat_t1_1_all_outputs_finite() {
    // No NaN/Inf for various matrix sizes and seeds
    for &(rows, cols) in &[(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)] {
        for seed in [42u64, 99, 777] {
            let g = seeded_random_matrix(seed, rows, cols);
            let mut out = vec![0.0f32; rows * cols];
            newton_schulz5(&g, rows, cols, &mut out);
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "NaN/Inf at {rows}×{cols} seed={seed} idx={i}: {v}"
                );
            }
        }
    }
    println!("T1.1: all outputs finite for 5 sizes × 3 seeds ✓");
}

// ── T1.2: Convergence (≤5 iterations) ────────────────────────
//
// The algorithm always runs exactly 5 iterations. The criterion is that 5 iters
// is sufficient — the result should be in the approximate orthogonal regime.

#[test]
fn goat_t1_2_convergence_multiple_seeds() {
    for seed in [42u64, 99, 123, 456, 789] {
        let g = seeded_random_matrix(seed, 16, 16);
        let mut out = vec![0.0f32; 16 * 16];
        newton_schulz5(&g, 16, 16, &mut out);

        let err = orthogonality_error(&out, 16, 16);
        assert!(
            err < 0.5,
            "Seed {seed}: convergence in 5 iters, orthogonality error = {err}"
        );
        println!("T1.2: seed={seed}, orthogonality error = {err:.6e}");
    }
}

// ── T1.3: Non-square handling ─────────────────────────────────

#[test]
fn goat_t1_3_nonsquare_12x6() {
    let g = seeded_random_matrix(99, 12, 6);
    let mut out = vec![0.0f32; 12 * 6];
    newton_schulz5(&g, 12, 6, &mut out);

    let err = orthogonality_error_transpose(&out, 12, 6);
    println!("T1.3: 12×6 (rows > cols) X^T@X error = {err:.6e}");
    assert!(
        err < 0.5,
        "12×6 X^T@X should be approximately I, max error = {err}"
    );
}

#[test]
fn goat_t1_3_nonsquare_6x12() {
    let g = seeded_random_matrix(88, 6, 12);
    let mut out = vec![0.0f32; 6 * 12];
    newton_schulz5(&g, 6, 12, &mut out);

    let err = orthogonality_error(&out, 6, 12);
    println!("T1.3: 6×12 (rows < cols) X@X^T error = {err:.6e}");
    assert!(
        err < 0.5,
        "6×12 X@X^T should be approximately I, max error = {err}"
    );
}

#[test]
fn goat_t1_3_nonsquare_tall_skinny() {
    let g = seeded_random_matrix(55, 64, 16);
    let mut out = vec![0.0f32; 64 * 16];
    newton_schulz5(&g, 64, 16, &mut out);

    let err = orthogonality_error_transpose(&out, 64, 16);
    println!("T1.3: 64×16 tall-skinny X^T@X error = {err:.6e}");
    assert!(
        err < 0.5,
        "64×16 X^T@X should be approximately I, max error = {err}"
    );
}

#[test]
fn goat_t1_3_allocating_vs_scratch_match() {
    // Both APIs must produce identical results
    for &(rows, cols) in &[(8, 8), (12, 6), (6, 12), (16, 8), (64, 16)] {
        let g = seeded_random_matrix(42, rows, cols);
        let mut out_alloc = vec![0.0f32; rows * cols];
        let mut out_scratch = vec![0.0f32; rows * cols];

        newton_schulz5(&g, rows, cols, &mut out_alloc);

        let mut scratch = NewtonSchulzScratch::new(rows, cols);
        newton_schulz5_into(&g, rows, cols, &mut out_scratch, &mut scratch);

        let mut max_diff = 0.0f32;
        for i in 0..(rows * cols) {
            max_diff = max_diff.max((out_alloc[i] - out_scratch[i]).abs());
        }
        assert!(
            max_diff < 1e-6,
            "{rows}×{cols}: both APIs should match, max diff = {max_diff}"
        );
    }
    println!("T1.3: allocating vs scratch match for 5 shapes ✓");
}

// ── T2.1: Subspace ratios ─────────────────────────────────────

#[test]
fn goat_t2_1_subspace_ratios_pythagorean() {
    let gradient = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let e1 = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = vec![0.0, 1.0, 0.0, 0.0, 0.0];
    let e3 = vec![0.0, 0.0, 1.0, 0.0, 0.0];

    let (r_dom, r_bulk) = river_valley::subspace_ratios(&gradient, &[e1, e2, e3]);
    let pythag = r_dom * r_dom + r_bulk * r_bulk;

    println!("T2.1: r_dom={r_dom:.6}, r_bulk={r_bulk:.6}, r_dom²+r_bulk²={pythag:.6}");
    assert!(
        (pythag - 1.0).abs() < 1e-5,
        "r_dom² + r_bulk² should = 1.0, got {pythag}"
    );
}

#[test]
fn goat_t2_1_subspace_ratios_full_projection() {
    // All eigenvectors span the full space → r_dom should be ≈ 1.0
    let gradient = vec![1.0, 2.0, 3.0];
    let e1 = vec![1.0, 0.0, 0.0];
    let e2 = vec![0.0, 1.0, 0.0];
    let e3 = vec![0.0, 0.0, 1.0];

    let (r_dom, r_bulk) = river_valley::subspace_ratios(&gradient, &[e1, e2, e3]);
    println!("T2.1: full projection r_dom={r_dom:.6}, r_bulk={r_bulk:.6}");
    assert!(
        (r_dom - 1.0).abs() < 1e-3,
        "Full projection r_dom should ≈ 1.0, got {r_dom}"
    );
    // r_bulk may not be exactly 0 due to floating-point sqrt rounding
    assert!(
        r_bulk < 1e-3,
        "Full projection r_bulk should ≈ 0.0, got {r_bulk}"
    );
}

#[test]
fn goat_t2_1_subspace_ratios_zero_gradient() {
    let gradient = vec![0.0; 5];
    let e1 = vec![1.0, 0.0, 0.0, 0.0, 0.0];

    let (r_dom, r_bulk) = river_valley::subspace_ratios(&gradient, &[e1]);
    println!("T2.1: zero gradient r_dom={r_dom}, r_bulk={r_bulk}");
    assert_eq!(r_dom, 0.0);
    assert_eq!(r_bulk, 1.0);
}

// ── T2.2: Effective rank ──────────────────────────────────────

#[test]
fn goat_t2_2_effective_rank_identity() {
    let mut identity = vec![0.0f32; 16];
    for i in 0..4 {
        identity[i * 4 + i] = 1.0;
    }
    let erank = river_valley::effective_rank(&identity, 4, 4);
    println!("T2.2: 4×4 identity effective rank = {erank:.4}");
    assert!(
        (erank - 4.0).abs() < 0.1,
        "4×4 identity effective rank should ≈ 4.0, got {erank}"
    );
}

#[test]
fn goat_t2_2_effective_rank_rank_deficient() {
    let mat = vec![
        1.0, 2.0, 3.0, //
        1.0, 2.0, 3.0, //
        1.0, 2.0, 3.0,
    ];
    let erank = river_valley::effective_rank(&mat, 3, 3);
    println!("T2.2: rank-1 matrix effective rank = {erank:.4}");
    assert!(
        (erank - 1.0).abs() < 0.1,
        "Rank-1 effective rank should ≈ 1.0, got {erank}"
    );
}

#[test]
fn goat_t2_2_effective_rank_nonsquare() {
    let mat = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0,
    ];
    let erank = river_valley::effective_rank(&mat, 2, 4);
    println!("T2.2: 2×4 rank-2 matrix effective rank = {erank:.4}");
    assert!(
        (erank - 2.0).abs() < 0.1,
        "2×4 rank-2 effective rank should ≈ 2.0, got {erank}"
    );
}

// ── T2.3: Cosine similarity ───────────────────────────────────

#[test]
fn goat_t2_3_cosine_similarity_constant() {
    let updates = vec![
        vec![1.0, 0.0, 0.0],
        vec![2.0, 0.0, 0.0],
        vec![3.0, 0.0, 0.0],
        vec![5.0, 0.0, 0.0],
    ];
    let cos = river_valley::update_cosine_similarity(&updates);
    println!("T2.3: constant direction cosine = {cos:.6}");
    assert!(
        (cos - 1.0).abs() < 1e-6,
        "Constant direction cosine should = 1.0, got {cos}"
    );
}

#[test]
fn goat_t2_3_cosine_similarity_orthogonal() {
    let updates = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let cos = river_valley::update_cosine_similarity(&updates);
    println!("T2.3: orthogonal direction cosine = {cos:.6}");
    assert!(
        cos.abs() < 1e-6,
        "Orthogonal direction cosine should = 0.0, got {cos}"
    );
}

#[test]
fn goat_t2_3_cosine_similarity_opposite() {
    let updates = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
    let cos = river_valley::update_cosine_similarity(&updates);
    println!("T2.3: opposite direction cosine = {cos:.6}");
    assert!(
        (cos - (-1.0)).abs() < 1e-6,
        "Opposite direction cosine should = -1.0, got {cos}"
    );
}

#[test]
fn goat_t2_3_cosine_similarity_single_update() {
    // Single update → defined as 1.0 (no pairs to compare)
    let updates = vec![vec![1.0, 2.0]];
    let cos = river_valley::update_cosine_similarity(&updates);
    println!("T2.3: single update cosine = {cos:.6}");
    assert!(
        (cos - 1.0).abs() < 1e-6,
        "Single update cosine should = 1.0, got {cos}"
    );
}

// ── T3.1: Muon output approximately orthogonal ────────────────

#[test]
fn goat_t3_1_muon_output_8x8() {
    let grad = seeded_random_matrix(77, 8, 8);
    let mut momentum = vec![0.0f32; 64];
    let mut out = vec![0.0f32; 64];
    muon_update(&grad, &mut momentum, 0.9, 8, 8, &mut out);

    // After momentum accumulation + orthogonalization + scaling,
    // the output should be approximately orthogonal
    let err = orthogonality_error(&out, 8, 8);
    println!("T3.1: Muon 8×8 output orthogonality error = {err:.6e}");
    // Scaling doesn't change the orthogonality ratio, so same threshold applies
    // But momentum accumulation means the input to NS is not the raw gradient
    assert!(
        err < 1.0,
        "Muon output should be approximately orthogonal, max error = {err}"
    );
}

#[test]
fn goat_t3_1_muon_output_all_finite_100_steps() {
    // Apply 100 Muon steps and check for NaN/Inf stability
    let mut momentum = vec![0.0f32; 64];
    let mut out = vec![0.0f32; 64];
    let mut scratch = NewtonSchulzScratch::new(8, 8);

    for step in 0..100 {
        let g = seeded_random_matrix(33 + step as u64, 8, 8);
        muon_update_into(&g, &mut momentum, 0.9, 8, 8, &mut out, &mut scratch);
    }

    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "NaN/Inf at index {i}: {v}");
    }
    println!("T3.1: 100 Muon steps — all outputs finite ✓");
}

#[test]
fn goat_t3_1_muon_into_matches_allocating() {
    // Zero-alloc and allocating API should produce identical Muon updates
    for seed in [42u64, 99, 555] {
        let grad = seeded_random_matrix(seed, 8, 8);
        let mut mom_a = vec![0.0f32; 64];
        let mut mom_b = vec![0.0f32; 64];
        let mut out_a = vec![0.0f32; 64];
        let mut out_b = vec![0.0f32; 64];
        let mut scratch = NewtonSchulzScratch::new(8, 8);

        muon_update(&grad, &mut mom_a, 0.9, 8, 8, &mut out_a);
        muon_update_into(&grad, &mut mom_b, 0.9, 8, 8, &mut out_b, &mut scratch);

        let mut max_diff_out = 0.0f32;
        let mut max_diff_mom = 0.0f32;
        for i in 0..64 {
            max_diff_out = max_diff_out.max((out_a[i] - out_b[i]).abs());
            max_diff_mom = max_diff_mom.max((mom_a[i] - mom_b[i]).abs());
        }
        assert!(
            max_diff_out < 1e-6,
            "Seed {seed}: output mismatch {max_diff_out}"
        );
        assert!(
            max_diff_mom < 1e-6,
            "Seed {seed}: momentum mismatch {max_diff_mom}"
        );
    }
    println!("T3.1: muon_update vs muon_update_into match for 3 seeds ✓");
}

// ── T3.2: Momentum accumulation ───────────────────────────────

#[test]
fn goat_t3_2_momentum_accumulation() {
    let grad = seeded_random_matrix(33, 4, 4);
    let mut momentum = vec![0.0f32; 16];
    let mut out = vec![0.0f32; 16];

    let mut norms = Vec::new();
    for _ in 0..5 {
        muon_update(&grad, &mut momentum, 0.9, 4, 4, &mut out);
        let mom_norm: f32 = momentum.iter().map(|v| v * v).sum::<f32>().sqrt();
        norms.push(mom_norm);
    }

    println!("T3.2: momentum norms over 5 steps = {norms:?}");

    // Strictly increasing momentum magnitude
    for i in 1..norms.len() {
        assert!(
            norms[i] > norms[i - 1],
            "Momentum should accumulate: norms = {norms:?}"
        );
    }
}

// ── T3.3: Throughput ──────────────────────────────────────────

#[test]
fn goat_t3_3_throughput_parity() {
    let g = seeded_random_matrix(42, 32, 32);

    // Warmup
    let mut out = vec![0.0f32; 32 * 32];
    let mut scratch = NewtonSchulzScratch::new(32, 32);
    for _ in 0..WARMUP {
        newton_schulz5(&g, 32, 32, &mut out);
        newton_schulz5_into(&g, 32, 32, &mut out, &mut scratch);
    }

    // Bench allocating API
    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(newton_schulz5(&g, 32, 32, &mut out));
    }
    let alloc_time = start.elapsed();

    // Bench scratch API
    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(newton_schulz5_into(&g, 32, 32, &mut out, &mut scratch));
    }
    let scratch_time = start.elapsed();

    let alloc_us = alloc_time.as_secs_f64() * 1e6 / ITERS as f64;
    let scratch_us = scratch_time.as_secs_f64() * 1e6 / ITERS as f64;

    println!("T3.3: 32×32 allocating API = {alloc_us:.1} µs/call");
    println!("T3.3: 32×32 scratch API    = {scratch_us:.1} µs/call");

    let ratio = scratch_us / alloc_us;
    println!("T3.3: scratch/alloc ratio = {ratio:.3}×");
    assert!(
        ratio < 1.5,
        "Scratch API should not be >1.5× slower, ratio = {ratio}"
    );
}

#[test]
fn goat_t3_3_muon_throughput() {
    let grad = seeded_random_matrix(42, 16, 16);
    let mut momentum = vec![0.0f32; 256];
    let mut out = vec![0.0f32; 256];
    let mut scratch = NewtonSchulzScratch::new(16, 16);

    // Warmup
    for _ in 0..WARMUP {
        muon_update_into(&grad, &mut momentum, 0.9, 16, 16, &mut out, &mut scratch);
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(muon_update_into(
            &grad,
            &mut momentum,
            0.9,
            16,
            16,
            &mut out,
            &mut scratch,
        ));
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / ITERS as f64;

    println!("T3.3: Muon 16×16 update = {us_per_call:.1} µs/call");
    assert!(
        us_per_call < 5000.0,
        "Muon update too slow: {us_per_call} µs"
    );
}
