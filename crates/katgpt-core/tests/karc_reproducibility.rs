//! KARC reproducibility test — GOAT gate G4 (Plan 308).
//!
//! Two `KarcForecaster` instances fit on byte-identical synthetic trajectories
//! must produce byte-identical `Wout`. This is the substrate for downstream
//! quorum commitment (riir-chain LatCal, riir-neuron-db KarcShard freeze).
//!
//! Varied across λ ∈ {1e-8, 1e-6, 1e-4} per Plan T1.9 to confirm stability
//! across regularization strengths.

use katgpt_core::{ChebyshevBasis, FourierBasis, KarcForecaster};
use std::fs;
use std::io::Write;

/// Deterministic synthetic trajectory (no RNG — bit-stable across runs).
/// A genuinely rich multi-frequency 2D signal so the basis-expanded features
/// span the full space (avoids rank-deficient Grams that fail Cholesky at very
/// small λ). Combines incommensurate frequencies + cross-terms.
fn synthetic_trajectory(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 * 0.073; // irrational-ish step to avoid periodicity
        let a = (0.7 * t).sin() + 0.4 * (1.9 * t).cos() + 0.2 * (3.1 * t).sin();
        let b = (1.3 * t).cos() + 0.5 * (0.6 * t).sin() + 0.3 * (2.4 * t).cos();
        out.push(a);
        out.push(b);
    }
    out
}

/// Build training pairs from a trajectory and fit a forecaster. Returns the
/// forecaster (caller inspects `.wout`).
fn fit_on_trajectory<
    B: katgpt_core::KarcBasis<M>,
    const D: usize,
    const M: usize,
    const K: usize,
>(
    basis: B,
    traj: &[f32],
    lambda: f32,
) -> KarcForecaster<B, D, M, K> {
    let n_total = traj.len() / D;
    let mut f = KarcForecaster::with_capacity(basis, n_total);
    let kd = K * D;
    for t in (K - 1)..(n_total - 1) {
        let mut delay = vec![0.0f32; kd];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        let mut target = vec![0.0f32; D];
        for d in 0..D {
            target[d] = traj[(t + 1) * D + d];
        }
        f.accumulate_pair(&delay, target.as_slice().try_into().unwrap());
    }
    f.fit_ridge(lambda).expect("fit_ridge");
    f
}

#[test]
fn g4_wout_byte_identical_across_instances_fourier() {
    const D: usize = 2;
    const K: usize = 3;
    const M: usize = 8;
    let traj = synthetic_trajectory(500);
    for &lambda in &[1e-8f32, 1e-6, 1e-4] {
        let f1 =
            fit_on_trajectory::<FourierBasis<M>, D, M, K>(FourierBasis::new(4.0), &traj, lambda);
        let f2 =
            fit_on_trajectory::<FourierBasis<M>, D, M, K>(FourierBasis::new(4.0), &traj, lambda);
        assert_eq!(
            f1.wout.len(),
            f2.wout.len(),
            "λ={}: wout length mismatch",
            lambda
        );
        // Byte-identical comparison via raw bit patterns (catches NaN payload
        // differences and signed-zero differences that == would miss).
        let bits_a: Vec<u32> = f1.wout.iter().map(|x| x.to_bits()).collect();
        let bits_b: Vec<u32> = f2.wout.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            bits_a, bits_b,
            "λ={}: Wout not byte-identical (bit-pattern mismatch)",
            lambda
        );
    }
}

#[test]
fn g4_wout_byte_identical_across_instances_chebyshev() {
    const D: usize = 2;
    const K: usize = 3;
    const M: usize = 6;
    let traj = synthetic_trajectory(500);
    for &lambda in &[1e-8f32, 1e-6, 1e-4] {
        let f1 =
            fit_on_trajectory::<ChebyshevBasis<M>, D, M, K>(ChebyshevBasis::new(), &traj, lambda);
        let f2 =
            fit_on_trajectory::<ChebyshevBasis<M>, D, M, K>(ChebyshevBasis::new(), &traj, lambda);
        let bits_a: Vec<u32> = f1.wout.iter().map(|x| x.to_bits()).collect();
        let bits_b: Vec<u32> = f2.wout.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            bits_a, bits_b,
            "λ={}: Wout not byte-identical (Chebyshev)",
            lambda
        );
    }
}

#[test]
fn g4_wout_changes_with_lambda() {
    // Sanity: different λ should produce different Wout (otherwise the test
    // above is vacuous — both could be reading uninitialised memory).
    const D: usize = 2;
    const K: usize = 3;
    const M: usize = 8;
    let traj = synthetic_trajectory(500);
    let f_small =
        fit_on_trajectory::<FourierBasis<M>, D, M, K>(FourierBasis::new(4.0), &traj, 1e-8);
    let f_large =
        fit_on_trajectory::<FourierBasis<M>, D, M, K>(FourierBasis::new(4.0), &traj, 1e-2);
    let bits_small: Vec<u32> = f_small.wout.iter().map(|x| x.to_bits()).collect();
    let bits_large: Vec<u32> = f_large.wout.iter().map(|x| x.to_bits()).collect();
    assert_ne!(
        bits_small, bits_large,
        "Wout identical across λ=1e-8 vs λ=1e-2 — test is vacuous"
    );
}

#[test]
fn g4_wout_dump_for_audit() {
    // Optional audit artifact: dump the bit patterns to a file so a second
    // machine can verify byte-identical reproduction. Disabled by default —
    // uncomment the KARC_G4_DUMP env check to enable.
    if std::env::var("KARC_G4_DUMP").is_err() {
        return;
    }
    const D: usize = 2;
    const K: usize = 3;
    const M: usize = 8;
    let traj = synthetic_trajectory(200);
    let f = fit_on_trajectory::<FourierBasis<M>, D, M, K>(FourierBasis::new(4.0), &traj, 1e-6);
    let mut file = fs::File::create("karc_g4_wout_audit.bin").expect("create audit file");
    for &x in &f.wout {
        file.write_all(&x.to_le_bytes()).expect("write audit");
    }
}
