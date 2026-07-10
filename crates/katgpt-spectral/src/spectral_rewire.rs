//! Spectral Rewiring — Weight Delta Purification via Base SVD Projection.
//!
//! Plan 423, Research 406. Distilled from Zhang et al., *Spectral Rewiring for
//! Exploration, Purification, and Model Merging* (arXiv:2607.03065, Tsinghua
//! AIR / ByteDance Seed, Jul 2026).
//!
//! Given base weights W₀ and a weight delta ΔW, project ΔW onto W₀'s SVD
//! spectral subspace:
//! ```text
//! W₀ = U Σ Vᵀ           (thin SVD of the base)
//! M  = U_rᵀ ΔW V_r       (rewiring matrix — r×r, cross-skill interactions)
//! ΔW* = U_r M V_rᵀ        (purified delta — on-manifold component)
//! ΔW⊥ = ΔW − ΔW*          (off-manifold residual)
//! ```
//!
//! The rewiring matrix M is compact (r×r). Its off-diagonal elements M[i][j]
//! (i≠j) represent cross-skill "rewiring" — many-to-one logical synthesis.
//! Its diagonal elements M[i][i] represent in-skill strength modulation.
//!
//! # Modelless
//!
//! SVD + matrix multiply. No training, no gradient descent. The paper's
//! headline use case (extract a reasoning core from a trained RL delta W_RL)
//! routes to riir-train; this module ships the modelless kernel that operates
//! on any weight delta — freeze/thaw deltas, LoRA overlays, consolidation
//! deltas.
//!
//! # Sibling relationship
//!
//! This is the **on-principal complement** of [`crate::off_principal`] (Plan 264).
//! `off_principal` projects a query *away* from the base SVD subspace
//! (off-principal). `spectral_rewire` projects a delta *onto* the base SVD
//! subspace (on-principal). Both reuse the same SVD substrate; together they
//! decompose any delta into on-manifold + off-manifold components (Issue 123,
//! Fusion B).
//!
//! # Feature gate
//!
//! Opt-in until the Plan 423 GOAT gate passes. The make-or-break gate is G1:
//! spectral concentration at NPC scale (the paper proves it for 1.5B–32B LLM
//! weights; our 64×64 / 128×128 matrices are unvalidated).

use katgpt_core::simd::simd_dot_f32;
use katgpt_core::{SvdResultScratch, SvdScratch, thin_svd_into};

// ---------------------------------------------------------------------------
// SpectralRewireScratch — pre-allocated reusable buffers
// ---------------------------------------------------------------------------

/// Pre-allocated scratch for [`spectral_rewire_into`]. Reuse across calls to
/// eliminate per-call allocation.
///
/// Sized via [`SpectralRewireScratch::with_capacity`] for the largest expected
/// `(d_out, d_in, rank)` triple. All internal buffers grow on demand if a
/// larger matrix is presented, but the zero-alloc contract holds only when
/// dimensions do not exceed capacity.
pub struct SpectralRewireScratch {
    /// SVD result (SOA scratch) for W₀ decomposition — stores U (column-major)
    /// and V (column-major) plus singular values.
    svd_result: SvdResultScratch,
    /// SVD working buffers (working copy of W₀, V accumulator, column norms).
    svd_work: SvdScratch,
    /// Temp buffer for `A = U_rᵀ · ΔW` (rank × d_in), row-major.
    a_buf: Vec<f32>,
    /// Rewiring matrix `M = A · V_r` (rank × rank), row-major.
    m_buf: Vec<f32>,
    /// Temp buffer for `B = U_r · M` (d_out × rank), row-major.
    b_buf: Vec<f32>,
    /// Purified delta `ΔW* = B · V_rᵀ` (d_out × d_in), row-major.
    delta_star_buf: Vec<f32>,
    /// Off-manifold residual `ΔW⊥ = ΔW − ΔW*` (d_out × d_in), row-major.
    residual_buf: Vec<f32>,
}

impl SpectralRewireScratch {
    /// Allocate scratch sized for factoring a `d_out × d_in` base matrix and
    /// projecting at up to `max_rank` spectral components.
    pub fn with_capacity(d_out: usize, d_in: usize, max_rank: usize) -> Self {
        let total = d_out * d_in;
        let rank = max_rank.min(d_out.min(d_in));
        Self {
            svd_result: SvdResultScratch::with_capacity(d_out, d_in),
            svd_work: SvdScratch::with_capacity(d_in, d_out),
            a_buf: vec![0.0; rank * d_in],
            m_buf: vec![0.0; rank * rank],
            b_buf: vec![0.0; d_out * rank],
            delta_star_buf: vec![0.0; total],
            residual_buf: vec![0.0; total],
        }
    }

    /// Grow all buffers if the requested `(d_out, d_in, rank)` exceeds current
    /// capacity. No-op (no allocation) when dimensions fit.
    fn ensure_capacity(&mut self, d_out: usize, d_in: usize, rank: usize) {
        let total = d_out * d_in;
        let r = rank.min(d_out.min(d_in));
        if r * d_in > self.a_buf.len() {
            self.a_buf.resize(r * d_in, 0.0);
        }
        if r * r > self.m_buf.len() {
            self.m_buf.resize(r * r, 0.0);
        }
        if d_out * r > self.b_buf.len() {
            self.b_buf.resize(d_out * r, 0.0);
        }
        if total > self.delta_star_buf.len() {
            self.delta_star_buf.resize(total, 0.0);
        }
        if total > self.residual_buf.len() {
            self.residual_buf.resize(total, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralRewireOutput — borrows into scratch
// ---------------------------------------------------------------------------

/// Output view of [`spectral_rewire_into`]. All slices borrow from the caller's
/// [`SpectralRewireScratch`] and are valid until the scratch is next mutated.
pub struct SpectralRewireOutput<'a> {
    /// Purified delta `ΔW* = U_r M V_rᵀ` (on-manifold component). Row-major
    /// `d_out × d_in`.
    pub delta_star: &'a [f32],
    /// Compact rewiring matrix `M = U_rᵀ ΔW V_r` (rank × rank), row-major.
    /// Off-diagonal `M[i][j]` (i≠j) = cross-skill rewiring.
    pub rewiring_matrix: &'a [f32],
    /// Off-manifold residual `ΔW⊥ = ΔW − ΔW*`. Row-major `d_out × d_in`.
    pub residual: &'a [f32],
    /// On-manifold energy fraction `‖ΔW*‖_F / ‖ΔW‖_F` ∈ [0, 1]. High values
    /// indicate spectral concentration (the paper's core assumption).
    pub on_manifold_fraction: f32,
}

// ---------------------------------------------------------------------------
// spectral_rewire_into — zero-alloc hot path
// ---------------------------------------------------------------------------

/// Project a weight delta `ΔW` onto the base weight `W₀`'s SVD spectral
/// subspace, extracting the on-manifold component `ΔW*` and the compact
/// rewiring matrix `M`.
///
/// # Arguments
///
/// - `w0`: base weights, row-major flat (`d_out × d_in`). The SVD is computed
///   over this matrix. For repeated projections against the same base, consider
///   pre-computing the SVD once (future `SpectralRewireIndex` — see Plan 423
///   open question 2).
/// - `delta`: weight delta `ΔW = W_new − W₀`, same layout as `w0`.
/// - `d_out`, `d_in`: matrix dimensions.
/// - `rank`: top-k spectral rank `r` (≤ `min(d_out, d_in)`). Controls the
///   subspace dimension — higher rank captures more of ΔW but M is larger.
/// - `scratch`: caller-owned reusable buffers.
///
/// # Returns
///
/// A [`SpectralRewireOutput`] borrowing `scratch`. The projection is exact up
/// to f32 round-off when ΔW lies entirely in the base SVD subspace.
///
/// # Zero-alloc contract
///
/// No allocation after warmup. Buffers grow only if dimensions exceed capacity
/// (see [`SpectralRewireScratch::with_capacity`]).
///
/// # Panics
///
/// Panics if `w0.len()` or `delta.len()` ≠ `d_out * d_in`, or if `rank` is 0.
pub fn spectral_rewire_into<'a>(
    w0: &[f32],
    delta: &[f32],
    d_out: usize,
    d_in: usize,
    rank: usize,
    scratch: &'a mut SpectralRewireScratch,
) -> SpectralRewireOutput<'a> {
    let total = d_out * d_in;
    assert_eq!(
        w0.len(),
        total,
        "spectral_rewire_into: w0.len() {} != d_out*d_in = {d_out}*{d_in} = {total}",
        w0.len()
    );
    assert_eq!(
        delta.len(),
        total,
        "spectral_rewire_into: delta.len() {} != d_out*d_in = {total}",
        delta.len()
    );
    assert!(rank >= 1, "spectral_rewire_into: rank must be >= 1, got {rank}");
    let r = rank.min(d_out.min(d_in));

    scratch.ensure_capacity(d_out, d_in, r);

    // ── Step 1: SVD of W₀ → U_r, V_r (column-major in svd_result) ──────────
    {
        let svd_result = &mut scratch.svd_result;
        let svd_work = &mut scratch.svd_work;
        thin_svd_into(w0, d_out, d_in, svd_result, svd_work);
    }
    let svd = &scratch.svd_result;

    // ── Step 2: A = U_rᵀ · ΔW  (r × d_in, row-major in a_buf) ──────────────
    //
    // A[i][j] = Σ_k U[k][i] · ΔW[k][j]
    // For each i: accumulate rank-1 updates over ΔW rows weighted by U column i.
    let a_len = r * d_in;
    let a_buf = &mut scratch.a_buf[..a_len];
    for v in a_buf.iter_mut() {
        *v = 0.0;
    }
    for i in 0..r {
        let u_col_i = svd.left_singular_vector(i); // length d_out, contiguous
        let a_row_offset = i * d_in;
        for k in 0..d_out {
            let alpha = u_col_i[k]; // U[k][i]
            let delta_row = &delta[k * d_in..(k + 1) * d_in]; // contiguous
            let a_row = &mut a_buf[a_row_offset..a_row_offset + d_in];
            // a_row[j] += alpha * delta_row[j]  — contiguous axpy
            let mut j = 0;
            while j + 4 <= d_in {
                a_row[j] = alpha.mul_add(delta_row[j], a_row[j]);
                a_row[j + 1] = alpha.mul_add(delta_row[j + 1], a_row[j + 1]);
                a_row[j + 2] = alpha.mul_add(delta_row[j + 2], a_row[j + 2]);
                a_row[j + 3] = alpha.mul_add(delta_row[j + 3], a_row[j + 3]);
                j += 4;
            }
            while j < d_in {
                a_row[j] = alpha.mul_add(delta_row[j], a_row[j]);
                j += 1;
            }
        }
    }

    // ── Step 3: M = A · V_r  (r × r, row-major in m_buf) ───────────────────
    //
    // M[i][j] = Σ_k A[i][k] · V[k][j] = <A_row_i, V_col_j>
    let m_len = r * r;
    let m_buf = &mut scratch.m_buf[..m_len];
    for i in 0..r {
        let a_row_i = &a_buf[i * d_in..(i + 1) * d_in]; // contiguous, length d_in
        for j in 0..r {
            let v_col_j = svd.right_singular_vector(j); // length d_in, contiguous
            m_buf[i * r + j] = simd_dot_f32(a_row_i, v_col_j, d_in);
        }
    }

    // ── Step 4: B = U_r · M  (d_out × r, row-major in b_buf) ───────────────
    //
    // B[i][j] = Σ_k U[i][k] · M[k][j]
    // For each i: accumulate rank-1 updates over M rows weighted by U entries.
    let b_len = d_out * r;
    let b_buf = &mut scratch.b_buf[..b_len];
    for v in b_buf.iter_mut() {
        *v = 0.0;
    }
    for k in 0..r {
        let u_col_k = svd.left_singular_vector(k); // length d_out, contiguous
        let m_row_k = &m_buf[k * r..(k + 1) * r]; // contiguous, length r
        for i in 0..d_out {
            let beta = u_col_k[i]; // U[i][k]
            let b_row = &mut b_buf[i * r..(i + 1) * r]; // contiguous, length r
            let mut j = 0;
            while j + 4 <= r {
                b_row[j] = beta.mul_add(m_row_k[j], b_row[j]);
                b_row[j + 1] = beta.mul_add(m_row_k[j + 1], b_row[j + 1]);
                b_row[j + 2] = beta.mul_add(m_row_k[j + 2], b_row[j + 2]);
                b_row[j + 3] = beta.mul_add(m_row_k[j + 3], b_row[j + 3]);
                j += 4;
            }
            while j < r {
                b_row[j] = beta.mul_add(m_row_k[j], b_row[j]);
                j += 1;
            }
        }
    }

    // ── Step 5: ΔW* = B · V_rᵀ  (d_out × d_in, row-major in delta_star_buf) ─
    //
    // ΔW*[i][j] = Σ_k B[i][k] · V[j][k]
    // Rank-1 sum: for each k, add B[:,k] ⊗ V[:,k].
    let delta_star = &mut scratch.delta_star_buf[..total];
    for v in delta_star.iter_mut() {
        *v = 0.0;
    }
    for k in 0..r {
        let v_col_k = svd.right_singular_vector(k); // length d_in, contiguous
        for i in 0..d_out {
            let gamma = b_buf[i * r + k]; // B[i][k]
            let ds_row = &mut delta_star[i * d_in..(i + 1) * d_in]; // contiguous
            let mut j = 0;
            while j + 4 <= d_in {
                ds_row[j] = gamma.mul_add(v_col_k[j], ds_row[j]);
                ds_row[j + 1] = gamma.mul_add(v_col_k[j + 1], ds_row[j + 1]);
                ds_row[j + 2] = gamma.mul_add(v_col_k[j + 2], ds_row[j + 2]);
                ds_row[j + 3] = gamma.mul_add(v_col_k[j + 3], ds_row[j + 3]);
                j += 4;
            }
            while j < d_in {
                ds_row[j] = gamma.mul_add(v_col_k[j], ds_row[j]);
                j += 1;
            }
        }
    }

    // ── Step 6: residual + on-manifold fraction ───────────────────────────
    let residual = &mut scratch.residual_buf[..total];
    let mut norm_delta_sq = 0.0_f32;
    let mut norm_star_sq = 0.0_f32;
    for idx in 0..total {
        let d = delta[idx];
        let s = delta_star[idx];
        residual[idx] = d - s;
        norm_delta_sq = d.mul_add(d, norm_delta_sq);
        norm_star_sq = s.mul_add(s, norm_star_sq);
    }
    let on_manifold_fraction = if norm_delta_sq > 1e-30 {
        (norm_star_sq / norm_delta_sq).sqrt()
    } else {
        0.0
    };

    SpectralRewireOutput {
        delta_star,
        rewiring_matrix: m_buf,
        residual,
        on_manifold_fraction,
    }
}

// ---------------------------------------------------------------------------
// spectral_rewire — allocating convenience wrapper
// ---------------------------------------------------------------------------

/// Owned result of [`spectral_rewire`] (the allocating convenience wrapper).
///
/// For hot paths, use [`spectral_rewire_into`] with a reused
/// [`SpectralRewireScratch`] to avoid the three `Vec` allocations here.
pub struct SpectralRewireResult {
    /// Purified delta `ΔW* = U_r M V_rᵀ` (on-manifold component).
    pub delta_star: Vec<f32>,
    /// Compact rewiring matrix `M = U_rᵀ ΔW V_r` (rank × rank).
    pub rewiring_matrix: Vec<f32>,
    /// Off-manifold residual `ΔW⊥ = ΔW − ΔW*`.
    pub residual: Vec<f32>,
    /// On-manifold energy fraction `‖ΔW*‖_F / ‖ΔW‖_F` ∈ [0, 1].
    pub on_manifold_fraction: f32,
}

/// Allocating convenience wrapper around [`spectral_rewire_into`]. Allocates a
/// fresh [`SpectralRewireScratch`] internally, so this is for tests, examples,
/// and cold-path callers only.
pub fn spectral_rewire(
    w0: &[f32],
    delta: &[f32],
    d_out: usize,
    d_in: usize,
    rank: usize,
) -> SpectralRewireResult {
    let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, rank);
    let out = spectral_rewire_into(w0, delta, d_out, d_in, rank, &mut scratch);
    SpectralRewireResult {
        delta_star: out.delta_star.to_vec(),
        rewiring_matrix: out.rewiring_matrix.to_vec(),
        residual: out.residual.to_vec(),
        on_manifold_fraction: out.on_manifold_fraction,
    }
}

// ---------------------------------------------------------------------------
// RewiringDiagnostics — structural readout of the compact rewiring matrix M
// ---------------------------------------------------------------------------

/// Default relative threshold for declaring an off-diagonal rewiring entry
/// negligible: 1% of the largest-magnitude entry in M. Tunable via
/// [`rewiring_matrix_diagnostics_with_threshold`].
const DEFAULT_SPARSITY_REL_THRESHOLD: f32 = 0.01;

/// Structural diagnostics of a rewiring matrix `M = U_rᵀ ΔW V_r` (Plan 423
/// Phase 2).
///
/// `M` decomposes a weight delta into in-skill modulation (the diagonal) and
/// cross-skill rewiring (the off-diagonal). These readouts characterize that
/// structure without re-running the projection. All fields are derived purely
/// from the entries of `M` (no second SVD), so the cost is `O(r²)` and the call
/// is allocation-free.
///
/// - [`diagonal_energy`](RewiringDiagnostics::diagonal_energy) — fraction of
///   total `M` energy on the diagonal. Near `1.0` ⇒ the delta is mostly
///   in-skill strength modulation; near `0.0` ⇒ the delta is mostly
///   cross-skill rewiring (the paper's "many-to-one logical synthesis").
/// - [`off_diagonal_energy`](RewiringDiagnostics::off_diagonal_energy) — the
///   complement, `1.0 − diagonal_energy`.
/// - [`spectral_norm_estimate`](RewiringDiagnostics::spectral_norm_estimate) —
///   the matrix ∞-norm `max_i Σ_j |M[i][j]|`, a standard upper bound on the
///   spectral norm `‖M‖₂`. Cheap (`O(r²)`), allocation-free, and tighter than
///   the raw diagonal max. Used as a magnitude scale for the rewiring.
/// - [`rewiring_sparsity`](RewiringDiagnostics::rewiring_sparsity) — fraction
///   of off-diagonal entries whose magnitude is below
///   `rel_threshold · max|M[i][j]|`. Near `1.0` ⇒ only a few cross-skill
///   links carry the rewiring (sparse composition); near `0.0` ⇒ the
///   rewiring is diffuse across all skill pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewiringDiagnostics {
    /// `Σᵢ M[i][i]² / Σᵢⱼ M[i][j]²` — diagonal energy fraction ∈ `[0, 1]`.
    pub diagonal_energy: f32,
    /// `1.0 − diagonal_energy` — off-diagonal (cross-skill) energy fraction.
    pub off_diagonal_energy: f32,
    /// `max_i Σ_j |M[i][j]|` — ∞-norm upper bound on `‖M‖₂`.
    pub spectral_norm_estimate: f32,
    /// Fraction of off-diagonal entries below the relative negligibility
    /// threshold ∈ `[0, 1]`.
    pub rewiring_sparsity: f32,
}

/// Compute [`RewiringDiagnostics`] for a row-major `rank × rank` rewiring
/// matrix `M`, using the default negligibility threshold
/// ([`DEFAULT_SPARSITY_REL_THRESHOLD`], 1% of the max-magnitude entry).
///
/// For a custom threshold, use [`rewiring_matrix_diagnostics_with_threshold`].
///
/// # Panics
///
/// Panics if `m.len() != rank * rank` or `rank == 0`.
///
/// # Example
///
/// A mixed-structure rewiring matrix: strong diagonal (in-skill modulation)
/// plus two cross-skill rewiring links.
///
/// ```
/// use katgpt_spectral::spectral_rewire::rewiring_matrix_diagnostics;
///
/// // Row-major 3×3: diagonal dominates (3, 2, 4), with one strong and one
/// // weak cross-skill link.
/// let m: &[f32] = &[
///     3.0, 0.5, 0.0,
///     0.0, 2.0, 0.0,
///     0.0, 0.0, 4.0,
/// ];
/// let d = rewiring_matrix_diagnostics(m, 3);
///
/// // Most energy is on the diagonal.
/// assert!(d.diagonal_energy > 0.9, "diagonal should dominate: {}", d.diagonal_energy);
/// assert!((d.diagonal_energy + d.off_diagonal_energy - 1.0).abs() < 1e-6);
///
/// // ∞-norm = max row sum: row 0 = |3|+|0.5|+|0| = 3.5; row 2 = 4.0 → max.
/// assert!((d.spectral_norm_estimate - 4.0).abs() < 1e-6);
/// ```
pub fn rewiring_matrix_diagnostics(m: &[f32], rank: usize) -> RewiringDiagnostics {
    rewiring_matrix_diagnostics_with_threshold(m, rank, DEFAULT_SPARSITY_REL_THRESHOLD)
}

/// Compute [`RewiringDiagnostics`] with an explicit relative negligibility
/// threshold for [`RewiringDiagnostics::rewiring_sparsity`].
///
/// `rel_threshold` scales the max-magnitude entry of `M`: an off-diagonal
/// entry counts as negligible when `|M[i][j]| < rel_threshold · max|M|`.
/// A larger threshold declares more entries negligible (higher sparsity);
/// `0.0` means only exact zeros count.
///
/// # Panics
///
/// Panics if `m.len() != rank * rank` or `rank == 0`.
pub fn rewiring_matrix_diagnostics_with_threshold(
    m: &[f32],
    rank: usize,
    rel_threshold: f32,
) -> RewiringDiagnostics {
    assert!(!m.is_empty(), "rewiring_matrix_diagnostics: m is empty");
    assert!(rank > 0, "rewiring_matrix_diagnostics: rank must be > 0");
    assert_eq!(
        m.len(),
        rank * rank,
        "rewiring_matrix_diagnostics: m.len() = {} but rank*rank = {}",
        m.len(),
        rank * rank
    );

    // Single O(r²) pass: accumulate total squared energy, diagonal squared
    // energy, the max-magnitude entry, and the ∞-norm (max absolute row sum).
    let mut total_sq: f32 = 0.0;
    let mut diag_sq: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    let mut max_row_sum: f32 = 0.0;
    for i in 0..rank {
        let row = &m[i * rank..(i + 1) * rank];
        let mut row_sum: f32 = 0.0;
        for (j, &v) in row.iter().enumerate() {
            let av = v.abs();
            total_sq += v * v;
            if i == j {
                diag_sq += v * v;
            }
            if av > max_abs {
                max_abs = av;
            }
            row_sum += av;
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    // All-zero matrix: undefined diagonal fraction and norm → report zeros,
    // vacuously fully "sparse" (no signal to rewire with).
    if total_sq == 0.0 {
        return RewiringDiagnostics {
            diagonal_energy: 0.0,
            off_diagonal_energy: 0.0,
            spectral_norm_estimate: 0.0,
            rewiring_sparsity: 1.0,
        };
    }

    let diagonal_energy = diag_sq / total_sq;
    let off_diagonal_energy = 1.0 - diagonal_energy;
    let spectral_norm_estimate = max_row_sum;

    // Off-diagonal negligibility count. rank == 1 has no off-diagonal entries
    // → sparsity is vacuously 1.0 (nothing to rewire).
    let off_diag_count = rank.saturating_sub(1) * rank;
    let rewiring_sparsity = if off_diag_count == 0 {
        1.0
    } else {
        let threshold = rel_threshold * max_abs;
        let mut negligible = 0usize;
        for i in 0..rank {
            for j in 0..rank {
                if i != j && m[i * rank + j].abs() < threshold {
                    negligible += 1;
                }
            }
        }
        (negligible as f32) / (off_diag_count as f32)
    };

    RewiringDiagnostics {
        diagonal_energy,
        off_diagonal_energy,
        spectral_norm_estimate,
        rewiring_sparsity,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG (xorshift) for reproducible test matrices.
    fn make_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        }
    }

    /// Fill a `rows × cols` row-major matrix with deterministic pseudo-random
    /// values in roughly [-0.5, 0.5).
    fn rand_matrix(rng: &mut impl FnMut() -> f32, rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|_| rng()).collect()
    }

    /// Frobenius norm of a flat row-major matrix.
    fn fro_norm(m: &[f32]) -> f32 {
        let sq: f32 = m.iter().map(|&v| v * v).sum();
        sq.sqrt()
    }

    /// Relative Frobenius error: ‖a − b‖_F / ‖b‖_F.
    fn rel_fro_err(a: &[f32], b: &[f32]) -> f32 {
        let diff: f32 = a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum();
        let base: f32 = b.iter().map(|&v| v * v).sum();
        (diff / base).sqrt()
    }

    // ── T1.7: Synthetic delta round-trip ──────────────────────────────────

    #[test]
    fn synthetic_on_manifold_delta_round_trips() {
        // Construct W₀ = random 8×8, compute its thin SVD, construct ΔW = U_r M_true V_rᵀ
        // for a known M_true. The projection must recover M_true and ΔW* = ΔW exactly.
        let mut rng = make_rng(42);
        let d_out = 8;
        let d_in = 8;
        let r = 4;

        let w0 = rand_matrix(&mut rng, d_out, d_in);

        // SVD of W₀ to get U_r, V_r for constructing the synthetic delta.
        let mut svd_result = SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = SvdScratch::with_capacity(d_in, d_out);
        thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);

        // Random M_true (r × r).
        let m_true = rand_matrix(&mut rng, r, r);

        // ΔW = U_r · M_true · V_rᵀ, i.e. ΔW[i][j] = Σ_{a,b} U[i][a]·M[a][b]·V[j][b].
        let mut delta = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                let mut acc = 0.0_f32;
                for a in 0..r {
                    let u_ia = svd_result.left_singular_vector(a)[i];
                    for b in 0..r {
                        let m_ab = m_true[a * r + b];
                        let v_jb = svd_result.right_singular_vector(b)[j];
                        acc += u_ia * m_ab * v_jb;
                    }
                }
                delta[i * d_in + j] = acc;
            }
        }

        // Run spectral_rewire — should recover M_true and ΔW* ≈ ΔW.
        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        // ΔW* should equal ΔW (the delta is exactly on-manifold by construction).
        let err = rel_fro_err(&result.delta_star, &delta);
        assert!(
            err < 1e-4,
            "on-manifold round-trip: ΔW* should match ΔW within 1e-4 rel, got {err:.2e}"
        );

        // on_manifold_fraction should be ~1.0 (delta is entirely on-manifold).
        assert!(
            result.on_manifold_fraction > 0.999,
            "on_manifold_fraction should be ~1.0 for on-manifold delta, got {}",
            result.on_manifold_fraction
        );

        // Residual should be ~0.
        let res_norm = fro_norm(&result.residual);
        let delta_norm = fro_norm(&delta);
        assert!(
            res_norm / delta_norm < 1e-4,
            "residual should be ~0 for on-manifold delta, got {}/{} = {}",
            res_norm,
            delta_norm,
            res_norm / delta_norm
        );

        // M should be recoverable. Note: sign ambiguity in SVD means M may
        // differ from m_true by sign flips per singular vector. We check the
        // Frobenius norm is preserved (energy conservation).
        let m_norm = fro_norm(&result.rewiring_matrix);
        let m_true_norm = fro_norm(&m_true);
        assert!(
            (m_norm - m_true_norm).abs() / m_true_norm < 1e-3,
            "‖M‖_F should match ‖M_true‖_F, got {} vs {}",
            m_norm,
            m_true_norm
        );
    }

    #[test]
    fn zero_delta_produces_zero_output() {
        let mut rng = make_rng(7);
        let d_out = 6;
        let d_in = 4;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = vec![0.0_f32; d_out * d_in];

        let result = spectral_rewire(&w0, &delta, d_out, d_in, 2);

        assert!(result.delta_star.iter().all(|&v| v.abs() < 1e-10));
        assert!(result.residual.iter().all(|&v| v.abs() < 1e-10));
        assert!(result.on_manifold_fraction.abs() < 1e-10);
    }

    #[test]
    fn on_plus_off_equals_delta() {
        // ΔW_on + ΔW_off must reconstruct ΔW exactly (orthogonal decomposition).
        let mut rng = make_rng(99);
        let d_out = 10;
        let d_in = 8;
        let r = 5;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        // delta_star + residual == delta
        for idx in 0..d_out * d_in {
            let recon = result.delta_star[idx] + result.residual[idx];
            let rel = (recon - delta[idx]).abs() / delta[idx].abs().max(1e-10);
            assert!(
                rel < 1e-5,
                "reconstruction failed at idx {idx}: ΔW_on+ΔW_off={recon:.6e}, ΔW={:.6e}, rel={rel:.2e}",
                delta[idx]
            );
        }
    }

    #[test]
    fn on_manifold_fraction_in_unit_interval() {
        let mut rng = make_rng(123);
        for &(d_out, d_in, r) in &[(8, 8, 4), (16, 12, 6), (6, 4, 2)] {
            let w0 = rand_matrix(&mut rng, d_out, d_in);
            let delta = rand_matrix(&mut rng, d_out, d_in);
            let result = spectral_rewire(&w0, &delta, d_out, d_in, r);
            assert!(
                (0.0..=1.0).contains(&result.on_manifold_fraction),
                "on_manifold_fraction out of [0,1]: {} for ({d_out}×{d_in}, r={r})",
                result.on_manifold_fraction
            );
        }
    }

    #[test]
    fn higher_rank_captures_more_energy() {
        // A random delta has no preferred subspace, but increasing rank should
        // never decrease the on-manifold fraction (monotone non-decreasing).
        let mut rng = make_rng(256);
        let d_out = 12;
        let d_in = 12;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        let mut prev = 0.0_f32;
        for r in [2, 4, 6, 8, 12] {
            let result = spectral_rewire(&w0, &delta, d_out, d_in, r);
            assert!(
                result.on_manifold_fraction >= prev - 1e-5,
                "on_manifold_fraction decreased at rank {r}: {} < {prev}",
                result.on_manifold_fraction
            );
            prev = result.on_manifold_fraction;
        }
        // At full rank, the projection is identity → fraction = 1.0.
        assert!(
            prev > 0.999,
            "at full rank, on_manifold_fraction should be 1.0, got {prev}"
        );
    }

    #[test]
    fn non_square_matrix_works() {
        let mut rng = make_rng(777);
        let d_out = 16;
        let d_in = 4;
        let r = 3;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        assert_eq!(result.delta_star.len(), d_out * d_in);
        assert_eq!(result.rewiring_matrix.len(), r * r);
        assert_eq!(result.residual.len(), d_out * d_in);
        assert!((0.0..=1.0).contains(&result.on_manifold_fraction));
    }

    #[test]
    fn scratch_reuse_is_consistent() {
        let mut rng = make_rng(2024);
        let d_out = 8;
        let d_in = 8;
        let r = 4;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        // First call with fresh scratch.
        let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, r);
        let out1 = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch);
        let ds1 = out1.delta_star.to_vec();
        let m1 = out1.rewiring_matrix.to_vec();
        let frac1 = out1.on_manifold_fraction;

        // Second call reusing the same scratch — must be identical.
        let out2 = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch);
        let ds2 = out2.delta_star.to_vec();
        let m2 = out2.rewiring_matrix.to_vec();
        let frac2 = out2.on_manifold_fraction;

        assert_eq!(ds1, ds2, "delta_star must be identical on scratch reuse");
        assert_eq!(m1, m2, "rewiring_matrix must be identical on scratch reuse");
        assert_eq!(frac1, frac2, "on_manifold_fraction must be identical");
    }

    // ── T2.2: rewiring_matrix_diagnostics on identity M ──────────────────

    #[test]
    fn diagnostics_identity_matrix_is_all_diagonal() {
        // Pure diagonal M (3×3) — all energy on the diagonal, no cross-skill
        // rewiring. Every off-diagonal entry is exactly zero.
        let m: &[f32] = &[
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ];
        let d = rewiring_matrix_diagnostics(m, 3);

        assert!((d.diagonal_energy - 1.0).abs() < 1e-6, "diagonal_energy = {}", d.diagonal_energy);
        assert!(d.off_diagonal_energy.abs() < 1e-6, "off_diagonal_energy = {}", d.off_diagonal_energy);
        // ∞-norm = max row sum = max(1, 2, 3) = 3.0.
        assert!((d.spectral_norm_estimate - 3.0).abs() < 1e-6, "norm = {}", d.spectral_norm_estimate);
        // All 6 off-diagonals are zero → fully sparse.
        assert!((d.rewiring_sparsity - 1.0).abs() < 1e-6, "sparsity = {}", d.rewiring_sparsity);
    }

    // ── T2.2: rewiring_matrix_diagnostics on pure off-diagonal M ─────────

    #[test]
    fn diagnostics_pure_off_diagonal_matrix_is_all_rewiring() {
        // Zero diagonal, all off-diagonal entries equal — pure cross-skill
        // rewiring, fully dense (no negligible off-diagonals).
        let m: &[f32] = &[
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let d = rewiring_matrix_diagnostics(m, 3);

        assert!(d.diagonal_energy.abs() < 1e-6, "diagonal_energy = {}", d.diagonal_energy);
        assert!((d.off_diagonal_energy - 1.0).abs() < 1e-6, "off_diagonal_energy = {}", d.off_diagonal_energy);
        // ∞-norm = max row sum = 2.0 (each row sums |1|+|1| on off-diagonals).
        assert!((d.spectral_norm_estimate - 2.0).abs() < 1e-6, "norm = {}", d.spectral_norm_estimate);
        // No off-diagonal is below 1% of max_abs (1.0) → sparsity = 0.0.
        assert!(d.rewiring_sparsity.abs() < 1e-6, "sparsity = {}", d.rewiring_sparsity);
    }

    // ── T2.2: rewiring_matrix_diagnostics on mixed M + edge cases ────────

    #[test]
    fn diagnostics_mixed_and_edge_cases() {
        // Mixed: dominant diagonal (10²) plus one moderate off-diagonal (3²).
        // diagonal_energy = 100 / (100 + 9 + 9) = 100/118 ≈ 0.847.
        let m: &[f32] = &[
            10.0, 3.0,
            3.0, 0.0,
        ];
        let d = rewiring_matrix_diagnostics(m, 2);
        assert!((d.diagonal_energy - (100.0_f32 / 118.0)).abs() < 1e-5,
            "diagonal_energy = {} expected ~0.847", d.diagonal_energy);
        assert!((d.diagonal_energy + d.off_diagonal_energy - 1.0).abs() < 1e-6,
            "diagonal + off-diagonal must sum to 1.0");
        // ∞-norm = max(|10|+|3|, |3|+|0|) = 13.0.
        assert!((d.spectral_norm_estimate - 13.0).abs() < 1e-6);
        // Off-diagonals are both 3.0; threshold = 0.01*10 = 0.1; neither < 0.1
        // → sparsity = 0/2 = 0.0.
        assert!(d.rewiring_sparsity.abs() < 1e-6, "sparsity = {}", d.rewiring_sparsity);

        // Raising the threshold to 50% makes 3.0 < 0.5*10 = 5.0 → both negligible.
        let d2 = rewiring_matrix_diagnostics_with_threshold(m, 2, 0.5);
        assert!((d2.rewiring_sparsity - 1.0).abs() < 1e-6,
            "sparsity at 50% threshold = {}", d2.rewiring_sparsity);

        // All-zero M: total energy 0 → zeroed diagnostics, vacuously sparse.
        let z: &[f32] = &[0.0, 0.0, 0.0, 0.0];
        let dz = rewiring_matrix_diagnostics(z, 2);
        assert_eq!(dz, RewiringDiagnostics {
            diagonal_energy: 0.0,
            off_diagonal_energy: 0.0,
            spectral_norm_estimate: 0.0,
            rewiring_sparsity: 1.0,
        });

        // rank == 1: no off-diagonal entries → sparsity vacuously 1.0; the
        // single diagonal entry carries all energy.
        let one: &[f32] = &[5.0];
        let d1 = rewiring_matrix_diagnostics(one, 1);
        assert!((d1.diagonal_energy - 1.0).abs() < 1e-6);
        assert!((d1.spectral_norm_estimate - 5.0).abs() < 1e-6);
        assert!((d1.rewiring_sparsity - 1.0).abs() < 1e-6);

        // rank == 0 and wrong-length panic (input validation).
        assert!(std::panic::catch_unwind(|| rewiring_matrix_diagnostics(&[1.0], 0)).is_err());
        assert!(std::panic::catch_unwind(|| rewiring_matrix_diagnostics(&[1.0, 2.0], 2)).is_err());
    }

    // ── T2.3 (integration): diagnostics on a real spectral_rewire output ─

    #[test]
    fn diagnostics_on_synthetic_rewiring_output() {
        // Inject a known on-manifold delta: ΔW = U_r M_true V_rᵀ for a diagonal
        // M_true. The recovered rewiring matrix must be diagonal-dominant →
        // diagnostics report high diagonal_energy and full sparsity.
        let mut rng = make_rng(101);
        let d_out = 8;
        let d_in = 8;
        let r = 4;
        let w0 = rand_matrix(&mut rng, d_out, d_in);

        // Reuse the project's SVD to build a genuinely on-manifold delta.
        let mut svd_result = katgpt_core::SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = katgpt_core::SvdScratch::with_capacity(d_in, d_out);
        katgpt_core::thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);

        // Diagonal M_true: [1, 2, 3, 4]. Build ΔW = U_r · diag · V_rᵀ.
        let m_true = [1.0_f32, 2.0, 3.0, 4.0];
        let mut delta = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                let mut acc = 0.0;
                for k in 0..r {
                    let u_k = svd_result.left_singular_vector(k);
                    let v_k = svd_result.right_singular_vector(k);
                    acc += u_k[i] * m_true[k] * v_k[j];
                }
                delta[i * d_in + j] = acc;
            }
        }

        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        // On-manifold fraction should be ~1.0 (delta lives in the subspace).
        assert!(result.on_manifold_fraction > 0.999,
            "on_manifold_fraction = {}", result.on_manifold_fraction);

        // Diagnostics on the recovered M: diagonal should dominate.
        let d = rewiring_matrix_diagnostics(&result.rewiring_matrix, r);
        assert!(d.diagonal_energy > 0.95,
            "diagonal_energy = {} (expected diagonal-dominant from diag M_true)", d.diagonal_energy);
        assert!((d.diagonal_energy + d.off_diagonal_energy - 1.0).abs() < 1e-5);
        assert!((0.0..=1.0).contains(&d.rewiring_sparsity));
        assert!(d.spectral_norm_estimate > 0.0);
    }
}
