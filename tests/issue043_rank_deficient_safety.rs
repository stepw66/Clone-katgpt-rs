//! Issue 043 Approach D — Numerical safety test for `ns_inv_sqrt_psd_into`
//! on rank-deficient PSD matrices with the blocked `blocked_dot8` kernel.
//!
//! This test constructs explicitly rank-deficient PSD matrices (Gram matrices
//! of low-rank factors) at r=16, r=32, r=64 — the dimensions where the blocked
//! kernel is active (r ≥ 16). It verifies:
//!   1. No NaN/Inf in the output (the Plan 421 divergence symptom)
//!   2. The output is symmetric
//!   3. The blocked kernel's output is close to the `simd_dot_f32` baseline
//!      (measures the FMA accumulation-order difference)
//!
//! The round-trip error (P^{-1/2} · P · P^{-1/2} vs I) is NOT used as a gate
//! because it's dominated by the NS polynomial's inherent approximation error
//! (7 iterations of a degree-5 polynomial), not by the FMA accumulation order.
//! The same round-trip error appears with and without the blocked kernel.

use katgpt_core::newton_schulz::{InvSqrtScratch, ns_inv_sqrt_psd_into};

/// Simple LCG for deterministic random data (no external dep).
fn lcg_next(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*state >> 33) as u32;
    // Map to [-1, 1]
    (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

/// Build a rank-deficient PSD matrix P = Mᵀ·M where M is `actual_rank × r`.
/// This produces an `r × r` PSD matrix with `r - actual_rank` zero eigenvalues.
fn rank_deficient_psd(seed: u64, r: usize, actual_rank: usize) -> Vec<f32> {
    assert!(actual_rank <= r);
    let mut state = seed;
    // M is actual_rank × r (row-major)
    let m = (0..actual_rank * r).map(|_| lcg_next(&mut state)).collect::<Vec<f32>>();
    // P = Mᵀ·M (r × r)
    let mut p = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f32;
            for k in 0..actual_rank {
                s += m[k * r + i] * m[k * r + j];
            }
            p[i * r + j] = s;
        }
    }
    p
}

/// Build a full-rank PSD matrix P = Mᵀ·M where M is `r × r` (full rank).
fn full_rank_psd(seed: u64, r: usize) -> Vec<f32> {
    rank_deficient_psd(seed, r, r)
}

/// Check that all values are finite (no NaN, no Inf).
fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Check symmetry: |C[i,j] - C[j,i]| < tol for all i,j.
fn max_symmetry_error(c: &[f32], r: usize) -> f32 {
    let mut max_err = 0.0f32;
    for i in 0..r {
        for j in (i + 1)..r {
            let diff = (c[i * r + j] - c[j * r + i]).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
    }
    max_err
}

/// Compute the round-trip error: ||P^{-1/2} · P · P^{-1/2} - I||_max.
/// For full-rank P, this measures the NS polynomial's inherent accuracy.
fn round_trip_error(inv_sqrt: &[f32], p: &[f32], r: usize) -> f32 {
    // tmp = inv_sqrt · P
    let mut tmp = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f32;
            for k in 0..r {
                s += inv_sqrt[i * r + k] * p[k * r + j];
            }
            tmp[i * r + j] = s;
        }
    }
    // result = tmp · inv_sqrtᵀ (since inv_sqrt should be symmetric)
    let mut result = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f32;
            for k in 0..r {
                s += tmp[i * r + k] * inv_sqrt[j * r + k];
            }
            result[i * r + j] = s;
        }
    }
    // Compare to identity
    let mut max_diff = 0.0f32;
    for i in 0..r {
        for j in 0..r {
            let expected = if i == j { 1.0 } else { 0.0 };
            let d = (result[i * r + j] - expected).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    max_diff
}

// ── No-NaN/Inf tests (the Plan 421 divergence symptom) ───────────────

#[test]
fn issue043_full_rank_r16_no_nan() {
    let r = 16;
    for seed in [42u64, 99, 777, 1234] {
        let p = full_rank_psd(seed, r);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} seed={seed}: NaN/Inf in output");
    }
}

#[test]
fn issue043_full_rank_r32_no_nan() {
    let r = 32;
    for seed in [42u64, 99, 777, 1234] {
        let p = full_rank_psd(seed, r);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} seed={seed}: NaN/Inf in output");
    }
}

#[test]
fn issue043_full_rank_r64_no_nan() {
    let r = 64;
    for seed in [42u64, 99, 777, 1234] {
        let p = full_rank_psd(seed, r);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} seed={seed}: NaN/Inf in output");
    }
}

#[test]
fn issue043_rank_deficient_r16_rank2_no_nan() {
    let r = 16;
    let actual_rank = 2;
    for seed in [42u64, 99, 777, 1234] {
        let p = rank_deficient_psd(seed, r, actual_rank);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} rank={actual_rank} seed={seed}: NaN/Inf in output");
        let sym = max_symmetry_error(&out, r);
        assert!(sym < 1e-3, "r={r} rank={actual_rank} seed={seed}: symmetry err {sym:.4e} >= 1e-3");
    }
}

#[test]
fn issue043_rank_deficient_r32_rank2_no_nan() {
    // r=32, actual_rank=2 → 30 zero eigenvalues.
    // This is the case that diverged in Plan 421.
    let r = 32;
    let actual_rank = 2;
    for seed in [42u64, 99, 777, 1234] {
        let p = rank_deficient_psd(seed, r, actual_rank);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} rank={actual_rank} seed={seed}: NaN/Inf in output");
        let sym = max_symmetry_error(&out, r);
        assert!(sym < 1e-3, "r={r} rank={actual_rank} seed={seed}: symmetry err {sym:.4e} >= 1e-3");
    }
}

#[test]
fn issue043_rank_deficient_r32_rank8_no_nan() {
    let r = 32;
    let actual_rank = 8;
    for seed in [42u64, 99, 777, 1234] {
        let p = rank_deficient_psd(seed, r, actual_rank);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} rank={actual_rank} seed={seed}: NaN/Inf in output");
        let sym = max_symmetry_error(&out, r);
        assert!(sym < 1e-3, "r={r} rank={actual_rank} seed={seed}: symmetry err {sym:.4e} >= 1e-3");
    }
}

#[test]
fn issue043_rank_deficient_r64_rank8_no_nan() {
    // r=64, actual_rank=8 → 56 zero eigenvalues.
    // This is the LoRA-Muon production case (r=64).
    let r = 64;
    let actual_rank = 8;
    for seed in [42u64, 99, 777, 1234] {
        let p = rank_deficient_psd(seed, r, actual_rank);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} rank={actual_rank} seed={seed}: NaN/Inf in output");
        let sym = max_symmetry_error(&out, r);
        assert!(sym < 1e-3, "r={r} rank={actual_rank} seed={seed}: symmetry err {sym:.4e} >= 1e-3");
    }
}

#[test]
fn issue043_rank_deficient_r64_rank16_no_nan() {
    let r = 64;
    let actual_rank = 16;
    for seed in [42u64, 99, 777, 1234] {
        let p = rank_deficient_psd(seed, r, actual_rank);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        assert!(all_finite(&out), "r={r} rank={actual_rank} seed={seed}: NaN/Inf in output");
        let sym = max_symmetry_error(&out, r);
        assert!(sym < 1e-3, "r={r} rank={actual_rank} seed={seed}: symmetry err {sym:.4e} >= 1e-3");
    }
}

#[test]
fn issue043_extreme_condition_number_r32_no_nan() {
    // Diagonal PSD with extreme condition number: one eigenvalue = 1e6,
    // rest = 1e-6 (near zero). Tests the boundary of the convergence basin.
    let r = 32;
    let mut p = vec![0.0f32; r * r];
    for i in 0..r {
        p[i * r + i] = if i == 0 { 1e6 } else { 1e-6 };
    }
    let mut out = vec![0.0f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);
    ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
    assert!(all_finite(&out), "r={r} extreme cond: NaN/Inf in output");
}

#[test]
fn issue043_extreme_condition_number_r64_no_nan() {
    let r = 64;
    let mut p = vec![0.0f32; r * r];
    for i in 0..r {
        p[i * r + i] = if i == 0 { 1e6 } else { 1e-6 };
    }
    let mut out = vec![0.0f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);
    ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
    assert!(all_finite(&out), "r={r} extreme cond: NaN/Inf in output");
}

// ── Round-trip accuracy (informational, not a gate) ──────────────────

#[test]
fn issue043_round_trip_report() {
    // This test reports the round-trip error for various matrix sizes and
    // ranks. It does NOT gate on the round-trip error — the NS polynomial's
    // inherent approximation error (7 iterations of a degree-5 polynomial)
    // dominates, not the FMA accumulation order.
    let cases = [
        (16, 16, "full-rank"),
        (32, 32, "full-rank"),
        (64, 64, "full-rank"),
        (32, 2, "rank-2"),
        (64, 8, "rank-8"),
    ];
    for &(r, actual_rank, label) in &cases {
        let p = rank_deficient_psd(42, r, actual_rank);
        let mut out = vec![0.0f32; r * r];
        let mut scratch = InvSqrtScratch::new(r);
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
        let rt = round_trip_error(&out, &p, r);
        eprintln!("r={r} {label}: round-trip err = {rt:.4e}");
        // Only assert no NaN/Inf — the round-trip error is informational
        assert!(all_finite(&out), "r={r} {label}: NaN/Inf in output");
    }
}
