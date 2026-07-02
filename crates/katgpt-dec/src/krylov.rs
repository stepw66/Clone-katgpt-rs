//! Krylov subspace matrix-exponential-vector product (Plan 359 Phase 2).
//!
//! Computes `exp(t·A)·b` without forming `A` explicitly and without
//! eigendecomposition. Uses Arnoldi iteration to project `A` onto a
//! `k`-dimensional Krylov subspace, computes the exponential of the small
//! projected Hessenberg matrix `H_k`, and reconstructs the result.
//!
//! # When to use this instead of [`DecEigendecomposition`](crate::DecEigendecomposition)
//!
//! - **Large complexes** where the eigendecomposition is prohibitive
//!   (256×256 = 65k vertices). The Krylov path is `O(k · nnz)` per matvec
//!   with `k ≈ 20–50` iterations, vs `O(n²·k_eig)` for the eigendecomposition.
//! - **Online use** (no offline precompute step). The Krylov path computes
//!   the trajectory from scratch each call.
//! - **General/unstable systems**. The Krylov approximation converges
//!   superlinearly regardless of the spectrum; eigendecomposition-based
//!   reconstruction amplifies spurious projections from approximate
//!   eigenvectors when unstable modes exist.
//!
//! # The math
//!
//! Given a matvec `A` and starting vector `b`, build the Krylov subspace
//! `K_k(A, b) = span{b, A·b, A²·b, ..., A^{k-1}·b}`. Arnoldi iteration
//! produces an orthonormal basis `V_k` (n×k) and upper Hessenberg `H_k`
//! (k×k) with `A·V_k = V_k·H_k + f·e_k^T`. The approximation:
//!
//! `exp(t·A)·b ≈ ‖b‖ · V_k · exp(t·H_k) · e₁`
//!
//! Converges superlinearly in `k` (faster than any geometric rate) for any
//! `t`. For symmetric `A` (e.g. the graph Laplacian), `H_k` is symmetric
//! tridiagonal and convergence is even faster.
//!
//! # References
//!
//! - Saad, "Analysis of Some Krylov Subspace Approximations to the Matrix
//!   Exponential Operator", SIAM J. Numer. Anal. 29 (1992).
//! - Higham, "Functions of Matrices: Theory and Computation", SIAM, 2008
//!   (Chapter 10 — scaling-squaring Padé / Taylor).

use crate::simd::simd_dot_f32;

/// Maximum Krylov subspace dimension supported. Caps the `H_k` small-matrix
/// exponential cost at `O(KRYLOV_K_MAX³) = O(64³) ≈ 260K` FMA per trajectory
/// prediction — negligible against the `O(k · nnz)` matvec cost on large
/// grids. For typical DEC heat-kernel use (rank-0 graph Laplacian), `k = 30`
/// suffices for `t` up to ~100 ticks with stable motor; the cap is generous
/// headroom.
pub const KRYLOV_K_MAX: usize = 64;

/// Arnoldi breakdown tolerance. If the Gram-Schmidt residual drops below
/// this, an invariant Krylov subspace has been found — `exp(t·A)·b` is
/// computed EXACTLY within the `m`-dimensional subspace (the approximation is
/// no longer an approximation; `H_m` captures `A` exactly on `K_m`).
const ARNOLDI_TOL: f32 = 1e-12;

/// Maximum Taylor series terms for [`expm_small`]. With scaling to
/// `‖M‖ ≤ 0.5`, convergence to f32 machine epsilon needs ≤ 15 terms; 30 is
/// generous headroom with an early-exit check.
const EXPM_MAX_TERMS: usize = 30;

/// Taylor series early-exit tolerance. When the infinity-norm of a Taylor
/// term drops below this, the series has converged.
const EXPM_TOL: f32 = 1e-14;

/// Scaling-squaring threshold. `‖M‖` is scaled down to `≤ 0.5` before the
/// Taylor series; the result is then squared `s` times.
const EXPM_SCALING_NORM: f32 = 0.5;

/// Apply `exp(t·A)·b` via Krylov subspace approximation.
///
/// Allocates the Krylov basis `V_k` (n×k floats) and Hessenberg `H_k` (k×k)
/// internally — the ONE allowed allocation for the Krylov path (Plan 359
/// T5.5). Use [`krylov_expmv_into`] to avoid allocating the output.
///
/// # Arguments
///
/// - `a_apply` — Closure computing `v → A·v` (sparse matrix-vector product).
///   Called `k` times. The closure owns whatever scratch it needs.
/// - `h0` — The starting vector `b` (length `n`).
/// - `t` — The time/scaling parameter.
/// - `k_requested` — Krylov subspace dimension. Capped at
///   [`KRYLOV_K_MAX`] and `h0.len()`. Larger `k` → better accuracy at the
///   cost of more matvecs.
///
/// # Returns
///
/// `exp(t·A)·b` as a new `Vec<f32>` of length `h0.len()`.
///
/// # Example
///
/// ```ignore
/// use katgpt_dec::krylov::krylov_expmv;
///
/// // A = identity → exp(t·I)·b = exp(t)·b
/// let b = vec![1.0, 2.0, 3.0];
/// let result = krylov_expmv(|v: &[f32], out: &mut [f32]| {
///     out.copy_from_slice(v);
/// }, &b, 1.0, 3);
/// // result ≈ [e, 2e, 3e]
/// ```
#[inline]
pub fn krylov_expmv<F>(a_apply: &mut F, h0: &[f32], t: f32, k_requested: usize) -> Vec<f32>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let n = h0.len();
    let mut out = vec![0.0f32; n];
    krylov_expmv_into(a_apply, h0, t, k_requested, &mut out);
    out
}

/// Zero-output-alloc variant of [`krylov_expmv`] — writes into `out`.
///
/// `out` must have length `≥ h0.len()`. Only `out[..h0.len()]` is written.
/// The Krylov basis `V_k` is still allocated internally (the one allowed
/// allocation per Plan 359 T5.5).
#[inline]
pub fn krylov_expmv_into<F>(
    a_apply: &mut F,
    h0: &[f32],
    t: f32,
    k_requested: usize,
    out: &mut [f32],
) where
    F: FnMut(&[f32], &mut [f32]),
{
    let n = h0.len();
    debug_assert!(
        out.len() >= n,
        "krylov_expmv_into: out.len() {} < h0.len() {}",
        out.len(),
        n
    );

    if n == 0 {
        return;
    }

    // t = 0 → exp(0)·b = b
    if t == 0.0 {
        out[..n].copy_from_slice(h0);
        return;
    }

    let k = k_requested.min(KRYLOV_K_MAX).min(n);

    // β = ‖h0‖
    let beta = simd_dot_f32(h0, h0, n).sqrt();
    if beta < 1e-30 {
        out[..n].fill(0.0);
        return;
    }

    // ── Arnoldi iteration ──────────────────────────────────────────────
    // V: n×(k+1) column-major (column j = v_j). +1 column for the last w/‖w‖
    //    that we compute but don't orthogonalize against (it's the residual).
    // H: k×k row-major, upper Hessenberg.
    // w: scratch for A·v_j − Σ projections.
    let mut v_basis = vec![0.0f32; n * (k + 1)];
    let mut h = vec![0.0f32; k * k];
    let mut w = vec![0.0f32; n];

    // v_0 = h0 / β
    let inv_beta = 1.0 / beta;
    for i in 0..n {
        v_basis[i] = h0[i] * inv_beta;
    }

    let mut m = k; // actual Krylov dimension (may shrink on breakdown)

    for j in 0..k {
        // w = A · v_j
        let vj_off = j * n;
        a_apply(&v_basis[vj_off..vj_off + n], &mut w);

        // Modified Gram-Schmidt: orthogonalize w against v_0, ..., v_j.
        // MGS (sequential subtract) is more numerically stable than classical
        // GS (compute-all-then-subtract) for ill-conditioned Krylov bases.
        for i in 0..=j {
            let vi_off = i * n;
            let dot = simd_dot_f32(&v_basis[vi_off..vi_off + n], &w, n);
            h[i * k + j] = dot;
            for idx in 0..n {
                w[idx] -= dot * v_basis[vi_off + idx];
            }
        }

        let w_norm = simd_dot_f32(&w, &w, n).sqrt();

        // Breakdown: invariant subspace found. exp(t·A)·b is exact within
        // the (j+1)-dimensional subspace; no more iterations needed.
        if w_norm < ARNOLDI_TOL {
            m = j + 1;
            break;
        }

        // Subdiagonal + next basis vector (unless this is the last column).
        if j + 1 < k {
            h[(j + 1) * k + j] = w_norm;
            let inv_wnorm = 1.0 / w_norm;
            let vj1_off = (j + 1) * n;
            for idx in 0..n {
                v_basis[vj1_off + idx] = w[idx] * inv_wnorm;
            }
        } else {
            // j == k-1: last iteration, m stays at k.
            m = k;
        }
    }

    // ── Small-matrix exponential: exp(t · H_m) ──────────────────────────
    // Extract the m×m leading principal submatrix of H (which is k×k) and
    // scale by t.
    let mut h_sub = vec![0.0f32; m * m];
    for i in 0..m {
        for jj in 0..m {
            h_sub[i * m + jj] = h[i * k + jj] * t;
        }
    }
    let exp_h = expm_small(m, &h_sub);

    // ── Reconstruct: result = β · V_m · exp(t·H_m) · e₁ ────────────────
    // y = exp(t·H_m) · e₁ = first column of exp_h: y[j] = exp_h[j·m].
    // result[i] = β · Σ_{j=0}^{m-1} y[j] · v_j[i]
    //           = β · Σ_{j=0}^{m-1} exp_h[j·m] · v_basis[j·n + i]
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..m {
            sum += exp_h[j * m] * v_basis[j * n + i];
        }
        out[i] = beta * sum;
    }
}

/// Compute `exp(M)` for a small `m×m` matrix `M` (row-major) via
/// scaling-squaring + Taylor series.
///
/// # Algorithm
///
/// 1. Compute `‖M‖_∞` (max absolute row sum).
/// 2. Choose `s` such that `‖M / 2^s‖_∞ ≤ 0.5`.
/// 3. Taylor series on the scaled matrix: `Σ_{j=0}^∞ (M/2^s)^j / j!`.
/// 4. Square the result `s` times: `exp(M) = (exp(M/2^s))^(2^s)`.
///
/// For `m ≤ KRYLOV_K_MAX (64)`, each matmul is `O(m³) ≤ O(260K)` — cheap.
/// The number of squarings is `s ≈ log2(‖M‖/0.5)`, bounded for the DEC
/// operator (graph Laplacian `‖Δ‖ ≈ 8`, motor bounded → `‖A‖ ≈ 10`,
/// `‖t·H_m‖ ≈ 10·t`, `s ≈ log2(20·t)`).
///
/// # Accuracy
///
/// For `‖M_scaled‖ ≤ 0.5`, the Taylor series converges to f32 machine
/// epsilon in ≤ 15 terms. The squaring step compounds rounding at `s`
/// doublings, but for `s ≤ 15` (covers `‖M‖ ≤ 16K`) the error stays well
/// within f32 precision for the DEC use case.
fn expm_small(m: usize, mat: &[f32]) -> Vec<f32> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![mat[0].exp()];
    }

    // 1. ‖M‖_∞
    let norm = inf_norm(m, mat);

    // 2. Scaling: s = ceil(log2(norm / 0.5)) if norm > 0.5, else 0.
    let s = if norm > EXPM_SCALING_NORM {
        let log2_ratio = (norm / EXPM_SCALING_NORM).ln() / 2.0f32.ln();
        log2_ratio.ceil() as i32
    } else {
        0
    };
    let s = s.max(0);
    let scale = 1.0f32 / 2.0f32.powi(s);

    // 3. Scaled matrix M_scaled = scale · M
    let mut m_scaled = vec![0.0f32; m * m];
    for i in 0..m * m {
        m_scaled[i] = mat[i] * scale;
    }

    // 4. Taylor series: result = I + M + M²/2 + M³/6 + ...
    //    term_j = M^j / j!; term_{j+1} = term_j · M / (j+1)
    let mut result = vec![0.0f32; m * m];
    let mut term = vec![0.0f32; m * m];
    for i in 0..m {
        result[i * m + i] = 1.0; // term_0 = I
        term[i * m + i] = 1.0;
    }

    for j in 1..=EXPM_MAX_TERMS {
        // term = term · m_scaled / j
        let new_term = matmul(m, &term, &m_scaled);
        let inv_j = 1.0 / j as f32;
        for i in 0..m * m {
            term[i] = new_term[i] * inv_j;
        }

        // result += term
        for i in 0..m * m {
            result[i] += term[i];
        }

        // Convergence check
        if inf_norm(m, &term) < EXPM_TOL {
            break;
        }
    }

    // 5. Squaring: result = result^(2^s)
    for _ in 0..s {
        let squared = matmul(m, &result, &result);
        result.copy_from_slice(&squared);
    }

    result
}

/// `m×m` matrix multiply: `out = a · b` (row-major throughout).
fn matmul(m: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f32;
            for l in 0..m {
                sum += a[i * m + l] * b[l * m + j];
            }
            out[i * m + j] = sum;
        }
    }
    out
}

/// Infinity norm: max absolute row sum of an `m×m` row-major matrix.
fn inf_norm(m: usize, mat: &[f32]) -> f32 {
    let mut norm = 0.0f32;
    for i in 0..m {
        let mut row_sum = 0.0f32;
        for j in 0..m {
            row_sum += mat[i * m + j].abs();
        }
        if row_sum > norm {
            norm = row_sum;
        }
    }
    norm
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn vec_approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for i in 0..a.len() {
            if !approx_eq(a[i], b[i], tol) {
                return false;
            }
        }
        true
    }

    // ── expm_small tests ──────────────────────────────────────────────

    #[test]
    fn expm_zero_is_identity() {
        for m in [1, 2, 3, 5, 8] {
            let zero_mat = vec![0.0f32; m * m];
            let result = expm_small(m, &zero_mat);
            // exp(0) = I
            for i in 0..m {
                for j in 0..m {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        approx_eq(result[i * m + j], expected, 1e-6),
                        "exp(0)[{i},{j}] = {} != {expected}",
                        result[i * m + j]
                    );
                }
            }
        }
    }

    #[test]
    fn expm_identity_matrix() {
        // exp(t·I) = exp(t)·I
        let m = 4;
        let t = 2.5f32;
        let mut identity = vec![0.0f32; m * m];
        for i in 0..m {
            identity[i * m + i] = t;
        }
        let result = expm_small(m, &identity);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { t.exp() } else { 0.0 };
                assert!(
                    approx_eq(result[i * m + j], expected, 1e-4),
                    "exp({t}·I)[{i},{j}] = {} != {expected}",
                    result[i * m + j]
                );
            }
        }
    }

    #[test]
    fn expm_diagonal() {
        // exp(diag(d_0, d_1, d_2)) = diag(exp(d_0), exp(d_1), exp(d_2))
        let m = 3;
        let diag = [-1.0, 0.5, 2.0];
        let mut mat = vec![0.0f32; m * m];
        for i in 0..m {
            mat[i * m + i] = diag[i];
        }
        let result = expm_small(m, &mat);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { diag[i].exp() } else { 0.0 };
                assert!(
                    approx_eq(result[i * m + j], expected, 1e-4),
                    "exp(diag)[{i},{j}] = {} != {expected}",
                    result[i * m + j]
                );
            }
        }
    }

    #[test]
    fn expm_rotation_matrix() {
        // exp([[0, θ], [-θ, 0]]) = [[cos θ, sin θ], [-sin θ, cos θ]]
        let theta = 1.0f32;
        let mat = vec![0.0, theta, -theta, 0.0];
        let result = expm_small(2, &mat);
        assert!(approx_eq(result[0], theta.cos(), 1e-4), "cos: {}", result[0]);
        assert!(approx_eq(result[1], theta.sin(), 1e-4), "sin: {}", result[1]);
        assert!(approx_eq(result[2], -theta.sin(), 1e-4), "-sin: {}", result[2]);
        assert!(approx_eq(result[3], theta.cos(), 1e-4), "cos: {}", result[3]);
    }

    #[test]
    fn expm_scaling_squaring_large_norm() {
        // Large-norm matrix: scaling-squaring must handle it
        // exp(10·I) = exp(10)·I ≈ 22026·I
        let m = 2;
        let mat = vec![10.0, 0.0, 0.0, 10.0];
        let result = expm_small(m, &mat);
        let expected = 10.0f32.exp();
        assert!(approx_eq(result[0], expected, 1e-3), "exp(10) = {} != {}", result[0], expected);
        assert!(approx_eq(result[3], expected, 1e-3), "exp(10) = {} != {}", result[3], expected);
    }

    // ── krylov_expmv tests ────────────────────────────────────────────

    #[test]
    fn krylov_zero_input() {
        let h0 = vec![0.0f32; 5];
        let result = krylov_expmv(&mut |v: &[f32], out: &mut [f32]| out.copy_from_slice(v), &h0, 5.0, 3);
        assert!(result.iter().all(|&x| x.abs() < 1e-30), "nonzero result for zero input");
    }

    #[test]
    fn krylov_t_zero_identity() {
        // exp(0·A)·b = b
        let h0 = vec![1.0, 2.0, 3.0, 4.0];
        let result = krylov_expmv(&mut |v: &[f32], out: &mut [f32]| {
            // A = diag(1, -1, 2, -2)
            out[0] = v[0];
            out[1] = -v[1];
            out[2] = 2.0 * v[2];
            out[3] = -2.0 * v[3];
        }, &h0, 0.0, 4);
        assert!(vec_approx_eq(&result, &h0, 1e-6), "t=0 should return h0, got {:?}", result);
    }

    #[test]
    fn krylov_identity_operator() {
        // A = I → exp(t·I)·b = exp(t)·b
        let h0 = vec![1.0f32, 2.0, 3.0];
        let t = 2.0f32;
        let result = krylov_expmv(&mut |v: &[f32], out: &mut [f32]| out.copy_from_slice(v), &h0, t, 3);
        let scale = t.exp();
        let expected: Vec<f32> = h0.iter().map(|&x| x * scale).collect();
        assert!(
            vec_approx_eq(&result, &expected, 1e-4),
            "exp({t}·I)·b = {:?} != {:?}",
            result,
            expected
        );
    }

    #[test]
    fn krylov_diagonal_operator_exact_at_full_k() {
        // A = diag(1, -1, 2, -2), full k=4 → exact
        let h0 = vec![1.0f32, 1.0, 1.0, 1.0];
        let t = 1.5f32;
        let result = krylov_expmv(&mut |v: &[f32], out: &mut [f32]| {
            out[0] = 1.0 * v[0];
            out[1] = -1.0 * v[1];
            out[2] = 2.0 * v[2];
            out[3] = -2.0 * v[3];
        }, &h0, t, 4);
        // Exact: [exp(t), exp(-t), exp(2t), exp(-2t)]
        let expected = vec![t.exp(), (-t).exp(), (2.0 * t).exp(), (-2.0 * t).exp()];
        assert!(
            vec_approx_eq(&result, &expected, 1e-3),
            "krylov diag exact: {:?} != {:?}",
            result,
            expected
        );
    }

    #[test]
    fn krylov_diagonal_operator_converges_with_k() {
        // As k increases, the Krylov approximation converges to exact.
        let h0 = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let t = 1.0f32;
        let diag = [1.0, -1.0, 0.5, -0.5, 2.0];
        let exact: Vec<f32> = diag.iter().map(|&d| (d * t).exp()).collect();

        let mut a_apply = |v: &[f32], out: &mut [f32]| {
            for i in 0..5 {
                out[i] = diag[i] * v[i];
            }
        };

        // k=1: poor (captures only the mean)
        let r1 = krylov_expmv(&mut a_apply, &h0, t, 1);
        let err1: f32 = h0.iter().zip(r1.iter().chain(std::iter::repeat(&0.0))).zip(exact.iter())
            .map(|((&_, &rk), &ex)| (rk - ex).abs())
            .sum();

        // k=5: exact (full subspace)
        let r5 = krylov_expmv(&mut a_apply, &h0, t, 5);
        let err5: f32 = r5.iter().zip(exact.iter())
            .map(|(&rk, &ex)| (rk - ex).abs())
            .sum();

        assert!(err5 < err1, "k=5 error {err5} should be < k=1 error {err1}");
        assert!(err5 < 1e-3, "k=5 should be near-exact, err = {err5}");
    }

    #[test]
    fn krylov_into_matches_allocating() {
        let h0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = 1.0f32;
        let mut a_apply = |v: &[f32], out: &mut [f32]| {
            out[0] = -1.0 * v[0];
            out[1] = -2.0 * v[1];
            out[2] = -3.0 * v[2];
            out[3] = -4.0 * v[3];
        };
        let alloc_result = krylov_expmv(&mut a_apply, &h0, t, 4);
        let mut into_result = vec![0.0f32; 4];
        krylov_expmv_into(&mut a_apply, &h0, t, 4, &mut into_result);
        assert!(
            vec_approx_eq(&alloc_result, &into_result, 1e-6),
            "alloc {:?} != into {:?}",
            alloc_result,
            into_result
        );
    }

    #[test]
    fn krylov_breakdown_invariant_subspace() {
        // h0 is an eigenvector of A → Krylov subspace is 1-dimensional.
        // The Arnoldi should break down after j=0 (w becomes 0).
        // A = 2·I, h0 = anything → A·h0 = 2·h0, so v_0 captures everything.
        let h0 = vec![3.0f32, 6.0, 9.0]; // not unit, but A·v = 2v means breakdown at j=0
        let t = 1.0f32;
        let result = krylov_expmv(&mut |v: &[f32], out: &mut [f32]| {
            for i in 0..3 {
                out[i] = 2.0 * v[i];
            }
        }, &h0, t, 5);
        // exp(t·2I)·h0 = exp(2t)·h0
        let scale = (2.0 * t).exp();
        let expected: Vec<f32> = h0.iter().map(|&x| x * scale).collect();
        assert!(
            vec_approx_eq(&result, &expected, 1e-4),
            "breakdown: {:?} != {:?}",
            result,
            expected
        );
    }

    #[test]
    fn krylov_k_capped_at_kmax() {
        // Request k=1000, should silently cap to KRYLOV_K_MAX (64) and n.
        let h0 = vec![1.0f32; 10];
        let result = krylov_expmv(&mut |v: &[f32], out: &mut [f32]| out.copy_from_slice(v), &h0, 1.0, 1000);
        // exp(1·I)·1 = e ≈ 2.718 for each element
        let expected = vec![(1.0f32).exp(); 10];
        assert!(
            vec_approx_eq(&result, &expected, 1e-4),
            "k cap: {:?} != {:?}",
            result,
            expected
        );
    }
}
