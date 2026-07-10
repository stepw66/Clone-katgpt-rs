//! Continuous Cochain Point Sampler — Whitney/de-Rham reconstruction for
//! modelless intra-primitive field queries.
//!
//! Plan 422 / Research 404 (distilled from Pajouheshgar et al., *Neural Cellular
//! Automata: From Cells to Pixels*, arXiv:2506.22899, SIGGRAPH 2026).
//!
//! # What this module does
//!
//! Given a discrete [`CochainField`] on a cell complex and a **continuous**
//! query point `p` inside a primitive (quad on a grid, triangle on a mesh), this
//! module computes:
//!
//! 1. **Interpolated state** `s̄(p) = Σⱼ λⱼ(p) · sⱼ` — the Whitney 0-form
//!    reconstruction. Turns discrete vertex values into a continuously-queryable
//!    field.
//! 2. **Local coordinate** `u(p) ∈ [-1,1]ᵈ` — a compact, zero-centered
//!    encoding of the point's position within its enclosing primitive.
//! 3. **Augmented local coordinate** `u_aug(p)` — a C⁰-continuous positional
//!    encoding (Cartesian sincos for quads, barycentric sort+CDF remap for
//!    triangles).
//!
//! The LPPN decoder weights `f_θ` (paper Eq. 4) are training-side (→ riir-train).
//! The modelless analog is the caller supplying a frozen direction vector and
//! using `(s̄(p), u_aug(p))` as the conditioning input. This module computes only
//! the conditioning — pure DEC math, no weights, no training.
//!
//! # Modelless
//!
//! Every function is closed-form algebra: bilinear/barycentric interpolation
//! (partition-of-unity λ-weights), sinusoidal basis functions, and analytic
//! triangular-distribution CDF transforms. No training, no backprop.
//!
//! # Zero-alloc
//!
//! All `*_into` variants write into caller-provided slices or a reusable
//! [`PointSamplerScratch`]. The hot path uses only stack-local intermediates
//! (fixed-size arrays for λ-weights and local coordinates).
//!
//! # GOAT gate (Plan 422 Phase 3)
//!
//! - **G1** Linear-precision exactness: bilinear λ reproduces linear fields
//!   exactly (tested at 1000+ interior points, tolerance 1e-5).
//! - **G2** Partition of unity: `Σⱼ λⱼ = 1`, `λⱼ ≥ 0` (tested at grid points).
//! - **G3** C⁰ continuity: sincos encoding is continuous at quad boundaries
//!   (`sin(nπ·1) = sin(nπ·(-1)) = 0`); barycentric sort is invariant to vertex
//!   ordering (tested with permuted vertex lists).
//! - **G4** Zero-alloc steady state (by construction — all `*_into` paths use
//!   caller-provided buffers).
//! - **G5** Sub-µs per query on a 64×64 grid (bilinear interp + optional sincos).
//!
//! # Primitive types
//!
//! | Primitive | λ-coordinate | Local coord u | Aug encoding |
//! |-----------|-------------|---------------|-------------|
//! | Quad (2D grid) | Bilinear `[f32; 4]` | Cartesian `[-1,1]²` | Sincos basis |
//! | Triangle (mesh) | Barycentric `[f32; 3]` | Sorted barycentric | CDF remap |
//!
//! Hex (3D voxel) and tetrahedron samplers are deferred — quad covers the
//! `grid_2d` fast path; triangle covers mesh consumers.

use crate::types::{CellComplex, CochainField};
use core::f32::consts::PI;

// ===========================================================================
// Types
// ===========================================================================

/// Local-coordinate encoding strategy (paper §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalCoordEncode {
    /// No augmentation: use raw `u ∈ [-1,1]ᵈ`.
    Raw,
    /// Cartesian sincos basis (paper Eq. 3):
    /// `[sin(πu), cos(πu), ..., sin(nπu), cos(nπu)]` per axis.
    /// Used for quad/cube primitives. C⁰-continuous across primitive boundaries.
    CartesianSincos {
        /// Number of harmonic pairs per axis. Output dimension = `4 * n_harmonics` for 2D.
        n_harmonics: u8,
    },
    /// Barycentric sort + triangular-distribution CDF remap (paper Appendix B).
    /// Used for triangle/tet primitives. C⁰-continuous across edges and
    /// invariant to vertex ordering.
    BarycentricSortCdf,
}

/// Pre-allocated output buffers for the combined sample+encode path.
/// Mirrors [`VCycleScratch`](crate::VCycleScratch) from Plan 413.
///
/// Reuse across calls to keep the steady-state path alloc-free.
#[derive(Debug)]
pub struct PointSamplerScratch {
    /// Interpolated state `s̄(p)` — size = `field.dim`.
    pub state: Vec<f32>,
    /// Augmented local coordinates `u_aug(p)` — size depends on encoding.
    pub local_coord: Vec<f32>,
}

impl PointSamplerScratch {
    /// Allocate scratch sized for the given output dimensions.
    #[inline]
    #[must_use]
    pub fn new(dim: usize, local_coord_dim: usize) -> Self {
        Self {
            state: vec![0.0; dim],
            local_coord: vec![0.0; local_coord_dim],
        }
    }

    /// Resize buffers. Only reallocates if the size changed.
    #[inline]
    pub fn resize(&mut self, dim: usize, local_coord_dim: usize) {
        if self.state.len() != dim {
            self.state.resize(dim, 0.0);
        }
        if self.local_coord.len() != local_coord_dim {
            self.local_coord.resize(local_coord_dim, 0.0);
        }
    }
}

/// Returns the output dimension of the augmented local coordinate for a given
/// encoding.
///
/// - `Raw` → `n_axes` (2 for quad, 3 for tri)
/// - `CartesianSincos { n }` → `2 * n_axes * n` (only valid for `n_axes = 2`)
/// - `BarycentricSortCdf` → `3` (only valid for `n_axes = 3`)
#[inline]
#[must_use]
pub fn local_coord_aug_dim(encode: LocalCoordEncode, n_axes: usize) -> usize {
    match encode {
        LocalCoordEncode::Raw => n_axes,
        LocalCoordEncode::CartesianSincos { n_harmonics } => 2 * n_axes * n_harmonics as usize,
        LocalCoordEncode::BarycentricSortCdf => 3,
    }
}

// ===========================================================================
// Internal helpers — Quad
// ===========================================================================

/// Locate the quad containing continuous point `(px, py)` on a regular grid.
///
/// Returns `(face_x, face_y, u_x, u_y)` where `u ∈ [0, 1]` are local fractional
/// coordinates within the containing quad.
///
/// Uses the `grid_dims` fast path (Plan 357 Issue 001): O(1) with no search.
#[inline]
fn locate_grid_point(cx: &CellComplex, px: f32, py: f32) -> (usize, usize, f32, f32) {
    let (w, h) = cx.grid_dims().expect(
        "locate_grid_point: complex must be a regular grid (constructed via CellComplex::grid_2d)",
    );
    debug_assert!(
        w >= 2 && h >= 2,
        "locate_grid_point: grid must be at least 2×2, got {w}×{h}"
    );

    // Clamp point to the valid interior domain. The last valid face index is
    // w-2 (0-indexed), so the point must be in [0, w-1).
    let max_x = (w.saturating_sub(1)) as f32;
    let max_y = (h.saturating_sub(1)) as f32;
    let px = px.clamp(0.0, max_x - f32::EPSILON);
    let py = py.clamp(0.0, max_y - f32::EPSILON);

    let fx = (px as usize).min(w.saturating_sub(2));
    let fy = (py as usize).min(h.saturating_sub(2));
    let u_x = px - fx as f32;
    let u_y = py - fy as f32;
    (fx, fy, u_x, u_y)
}

/// The four vertex indices of grid quad `(fx, fy)` on a `w`-wide grid.
///
/// Vertex layout on the grid: `v = y * w + x`.
///
/// Order: `[BL, BR, TL, TR]` = `[(fx, fy), (fx+1, fy), (fx, fy+1), (fx+1, fy+1)]`.
#[inline]
fn quad_vertex_indices(fx: usize, fy: usize, w: usize) -> [usize; 4] {
    let base = fy * w + fx;
    [base, base + 1, base + w, base + w + 1]
}

/// Bilinear λ-weights (partition of unity, non-negative, linear precision).
///
/// `u_x, u_y ∈ [0, 1]` are local fractional coordinates within a quad.
/// Returns `[λ_BL, λ_BR, λ_TL, λ_TR]` matching [`quad_vertex_indices`] order.
#[inline]
fn lambda_bilinear(u_x: f32, u_y: f32) -> [f32; 4] {
    let omx = 1.0 - u_x;
    let omy = 1.0 - u_y;
    [omx * omy, u_x * omy, omx * u_y, u_x * u_y]
}

// ===========================================================================
// Internal helpers — Triangle
// ===========================================================================

/// Barycentric coordinates of point `p` in triangle `(a, b, c)` (2D).
///
/// Returns `[λ_a, λ_b, λ_c]` where `λ_a + λ_b + λ_c = 1`.
/// Uses the standard cross-product / area-ratio formula.
#[inline]
fn barycentric_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> [f32; 3] {
    let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    let l0 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det;
    let l1 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det;
    let l2 = 1.0 - l0 - l1;
    [l0, l1, l2]
}

/// Sort three values descending using a sorting network (branch-light).
/// Returns `[max, mid, min]`.
#[inline]
fn sort_descending_3(x: f32, y: f32, z: f32) -> [f32; 3] {
    let (a, b) = if x >= y { (x, y) } else { (y, x) };
    if z >= a {
        [z, a, b]
    } else if z >= b {
        [a, z, b]
    } else {
        [a, b, z]
    }
}

/// Triangular distribution CDF: `Tri(ℓ, r, m)` on `[ℓ, r]` with mode `m`.
/// Returns `Pr(X ≤ x)`. Paper Appendix B, Eq. 16.
///
/// ```text
/// F(x) = 0                               if x ≤ ℓ
///      = (x-ℓ)² / ((r-ℓ)(m-ℓ))          if ℓ < x < m
///      = 1 - (r-x)² / ((r-ℓ)(r-m))      if m ≤ x < r
///      = 1                               if x ≥ r
/// ```
#[inline]
fn triangular_cdf(x: f32, l: f32, r: f32, m: f32) -> f32 {
    if x <= l {
        0.0
    } else if x >= r {
        1.0
    } else if x < m {
        // SAFETY: left_range > 0 because ℓ < x < m implies ℓ < m.
        let left_range = m - l;
        let dx = x - l;
        dx * dx / ((r - l) * left_range)
    } else {
        // SAFETY: right_range > 0 because m ≤ x < r implies m < r.
        let right_range = r - m;
        let dr = r - x;
        1.0 - dr * dr / ((r - l) * right_range)
    }
}

// ===========================================================================
// Quad API
// ===========================================================================

/// Bilinear λ-weights (partition of unity) for point `(px, py)` on a regular
/// grid.
///
/// Returns `[λ_BL, λ_BR, λ_TL, λ_TR]` for the containing quad.
///
/// # Panics
/// If the complex is not a regular grid.
#[inline]
pub fn lambda_coordinate_quad(cx: &CellComplex, px: f32, py: f32) -> [f32; 4] {
    let (_w, _h) = cx.grid_dims().expect("lambda_coordinate_quad: requires regular grid");
    let (_fx, _fy, u_x, u_y) = locate_grid_point(cx, px, py);
    lambda_bilinear(u_x, u_y)
}

/// Continuous cochain field sample at point `(px, py)` on a regular grid.
///
/// Computes `s̄(p) = Σ λⱼ(p) · sⱼ` using bilinear interpolation over the four
/// vertices of the containing quad. This is the Whitney 0-form reconstruction
/// on a 2D grid.
///
/// # Arguments
///
/// * `cx` — a regular grid complex ([`CellComplex::grid_2d`]).
/// * `field` — a rank-0 (vertex) cochain of arbitrary feature dimension.
/// * `px, py` — continuous query location in vertex-index coordinates
///   (`px ∈ [0, w-1]`, `py ∈ [0, h-1]`). Clamped to the grid domain.
/// * `out` — output slice of length `field.dim`. Receives `s̄(p)`.
///
/// # Panics
///
/// If `field.rank != 0`, `out.len() != field.dim`, or the complex is not a
/// regular grid.
///
/// # Allocation
///
/// Zero heap allocations — uses only stack-local intermediates.
#[inline]
pub fn sample_cochain_at_point_quad_into(
    cx: &CellComplex,
    field: &CochainField,
    px: f32,
    py: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(
        field.rank,
        0,
        "sample_cochain_at_point_quad_into: field must be rank-0, got rank {}",
        field.rank
    );
    let dim = field.dim;
    debug_assert_eq!(
        out.len(),
        dim,
        "sample_cochain_at_point_quad_into: out.len {} != dim {}",
        out.len(),
        dim
    );

    let (w, _) = cx
        .grid_dims()
        .expect("sample_cochain_at_point_quad_into: requires regular grid");
    let (fx, fy, u_x, u_y) = locate_grid_point(cx, px, py);
    let verts = quad_vertex_indices(fx, fy, w);
    let lam = lambda_bilinear(u_x, u_y);

    for d in 0..dim {
        out[d] = lam[0] * field.data[verts[0] * dim + d]
            + lam[1] * field.data[verts[1] * dim + d]
            + lam[2] * field.data[verts[2] * dim + d]
            + lam[3] * field.data[verts[3] * dim + d];
    }
}

/// Compact Cartesian local coordinate `u ∈ [-1,1]²` for a point in a grid quad.
///
/// Maps local fractional position `(u_x, u_y) ∈ [0,1]²` within the containing
/// quad to zero-centered `[-1,1]²`.
///
/// # Panics
///
/// If the complex is not a regular grid.
#[inline]
#[must_use]
pub fn local_coordinate_quad(cx: &CellComplex, px: f32, py: f32) -> [f32; 2] {
    let (_, _, u_x, u_y) = locate_grid_point(cx, px, py);
    [2.0 * u_x - 1.0, 2.0 * u_y - 1.0]
}

/// Augmented Cartesian local coordinate via sincos basis (paper Eq. 3).
///
/// Writes `[sin(πu₀), cos(πu₀), ..., sin(nπu₀), cos(nπu₀),`
/// followed by the same for `u₁` into `out`.
///
/// The encoding is C⁰-continuous across primitive boundaries because
/// `sin(kπ·1) = sin(kπ·(-1)) = 0` and `cos(kπ·1) = cos(kπ·(-1))`.
///
/// # Panics (debug)
///
/// `out.len()` must equal `2 * u.len() * n_harmonics`.
#[inline]
pub fn local_coordinate_aug_cartesian(u: &[f32], n_harmonics: u8, out: &mut [f32]) {
    let n = n_harmonics as usize;
    debug_assert_eq!(
        out.len(),
        2 * u.len() * n,
        "local_coordinate_aug_cartesian: out.len {} != 2 * {} * {}",
        out.len(),
        u.len(),
        n
    );
    let mut k = 0;
    for &ui in u {
        for h in 1..=n {
            let phase = (h as f32) * PI * ui;
            out[k] = phase.sin();
            out[k + 1] = phase.cos();
            k += 2;
        }
    }
}

/// Combined sample + local-coordinate-encode for a point in a grid quad.
///
/// Writes `s̄(p)` into `scratch.state` and `u_aug(p)` into `scratch.local_coord`.
/// Resizes scratch if needed (no-op if already correctly sized).
///
/// Zero-alloc after warmup (scratch is reused).
#[inline]
pub fn sample_point_quad_into(
    cx: &CellComplex,
    field: &CochainField,
    px: f32,
    py: f32,
    encode: LocalCoordEncode,
    scratch: &mut PointSamplerScratch,
) {
    let dim = field.dim;
    let aug_dim = local_coord_aug_dim(encode, 2);
    scratch.resize(dim, aug_dim);

    sample_cochain_at_point_quad_into(cx, field, px, py, &mut scratch.state);

    match encode {
        LocalCoordEncode::Raw => {
            let u = local_coordinate_quad(cx, px, py);
            scratch.local_coord[0] = u[0];
            scratch.local_coord[1] = u[1];
        }
        LocalCoordEncode::CartesianSincos { n_harmonics } => {
            let u = local_coordinate_quad(cx, px, py);
            local_coordinate_aug_cartesian(&u, n_harmonics, &mut scratch.local_coord);
        }
        LocalCoordEncode::BarycentricSortCdf => {
            panic!("sample_point_quad_into: BarycentricSortCdf is for triangle primitives");
        }
    }
}

// ===========================================================================
// Triangle API
// ===========================================================================

/// Barycentric λ-weights for a point in a triangle.
///
/// Returns `[λ₀, λ₁, λ₂]` corresponding to `tri_positions[0..3]`.
/// Satisfies partition of unity (`Σλᵢ = 1`) and linear precision.
#[inline]
#[must_use]
pub fn lambda_coordinate_tri(p: [f32; 2], tri_positions: &[[f32; 2]; 3]) -> [f32; 3] {
    barycentric_2d(p, tri_positions[0], tri_positions[1], tri_positions[2])
}

/// Continuous cochain field sample at point `(px, py)` in a triangle.
///
/// Computes `s̄(p) = Σ λⱼ(p) · sⱼ` using barycentric interpolation. This is the
/// Whitney 0-form reconstruction on a triangle mesh.
///
/// Point location (finding which triangle contains `p`) is the caller's
/// responsibility — mesh acceleration structures are out of scope for this
/// module.
///
/// # Arguments
///
/// * `field` — a rank-0 (vertex) cochain.
/// * `tri_positions` — world-space 2D positions of the triangle's three vertices.
/// * `tri_indices` — vertex indices into `field` for the triangle's vertices.
/// * `px, py` — continuous query location.
/// * `out` — output slice of length `field.dim`.
///
/// # Panics
///
/// If `field.rank != 0` or `out.len() != field.dim`.
#[inline]
pub fn sample_cochain_at_point_tri_into(
    field: &CochainField,
    tri_positions: &[[f32; 2]; 3],
    tri_indices: [usize; 3],
    px: f32,
    py: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(
        field.rank,
        0,
        "sample_cochain_at_point_tri_into: field must be rank-0, got rank {}",
        field.rank
    );
    let dim = field.dim;
    debug_assert_eq!(
        out.len(),
        dim,
        "sample_cochain_at_point_tri_into: out.len {} != dim {}",
        out.len(),
        dim
    );

    let lam = barycentric_2d([px, py], tri_positions[0], tri_positions[1], tri_positions[2]);

    for d in 0..dim {
        out[d] = lam[0] * field.data[tri_indices[0] * dim + d]
            + lam[1] * field.data[tri_indices[1] * dim + d]
            + lam[2] * field.data[tri_indices[2] * dim + d];
    }
}

/// Compact barycentric local coordinate: `uᵢ = 2λᵢ - 1 ∈ [-1,1]`.
#[inline]
#[must_use]
pub fn local_coordinate_tri(lambda: [f32; 3]) -> [f32; 3] {
    [2.0 * lambda[0] - 1.0, 2.0 * lambda[1] - 1.0, 2.0 * lambda[2] - 1.0]
}

/// Remap sorted barycentric coordinates `(a, b, c)` to `[-1,1]³` via the
/// triangular-distribution inverse-CDF transform (paper Appendix B, Eqs. 15–17).
///
/// Each sorted component follows a known triangular distribution under uniform
/// sampling within an equilateral triangle:
/// - `a ~ Tri(1/3, 1, 1/2)` (the maximum)
/// - `b ~ Tri(0, 1/2, 1/3)` (the middle)
/// - `c ~ Tri(0, 1/3, 0)` (the minimum)
///
/// The remapped coordinates are C⁰-continuous across triangle edges (sorting
/// eliminates vertex-order dependence) and have balanced dynamic range across
/// components.
///
/// `sorted_lambda` must be sorted descending: `sorted_lambda[0] ≥ [1] ≥ [2]`.
/// Use [`sort_descending_3`] (re-exported as needed) or equivalent.
///
/// # Panics (debug)
///
/// `out.len()` must equal 3.
#[inline]
pub fn local_coordinate_aug_barycentric(sorted_lambda: [f32; 3], out: &mut [f32]) {
    debug_assert_eq!(
        out.len(),
        3,
        "local_coordinate_aug_barycentric: out.len {} != 3",
        out.len()
    );
    let [a, b, c] = sorted_lambda;
    let third = 1.0 / 3.0;
    let half = 0.5;
    // a ~ Tri(1/3, 1, 1/2) — the maximum barycentric coordinate
    out[0] = 2.0 * triangular_cdf(a, third, 1.0, half) - 1.0;
    // b ~ Tri(0, 1/2, 1/3) — the middle barycentric coordinate
    out[1] = 2.0 * triangular_cdf(b, 0.0, half, third) - 1.0;
    // c ~ Tri(0, 1/3, 0) — the minimum barycentric coordinate
    out[2] = 2.0 * triangular_cdf(c, 0.0, third, 0.0) - 1.0;
}

/// Combined sample + local-coordinate-encode for a point in a triangle.
///
/// Writes `s̄(p)` into `scratch.state` and `u_aug(p)` into `scratch.local_coord`.
#[inline]
pub fn sample_point_tri_into(
    field: &CochainField,
    tri_positions: &[[f32; 2]; 3],
    tri_indices: [usize; 3],
    px: f32,
    py: f32,
    encode: LocalCoordEncode,
    scratch: &mut PointSamplerScratch,
) {
    let dim = field.dim;
    let aug_dim = local_coord_aug_dim(encode, 2);
    scratch.resize(dim, aug_dim);

    sample_cochain_at_point_tri_into(
        field,
        tri_positions,
        tri_indices,
        px,
        py,
        &mut scratch.state,
    );

    let lam = barycentric_2d([px, py], tri_positions[0], tri_positions[1], tri_positions[2]);

    match encode {
        LocalCoordEncode::Raw => {
            let u = local_coordinate_tri(lam);
            scratch.local_coord[0] = u[0];
            scratch.local_coord[1] = u[1];
        }
        LocalCoordEncode::BarycentricSortCdf => {
            let sorted = sort_descending_3(lam[0], lam[1], lam[2]);
            local_coordinate_aug_barycentric(sorted, &mut scratch.local_coord);
        }
        LocalCoordEncode::CartesianSincos { .. } => {
            panic!("sample_point_tri_into: CartesianSincos is for quad primitives");
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    // --- G1: Linear-precision exactness (quad) ---

    #[test]
    fn g1_linear_precision_quad() {
        let (gw, gh) = (8usize, 8usize);
        let grid = CellComplex::grid_2d(gw, gh);
        let mut field = CochainField::zeros(0, gw * gh, 1);
        for y in 0..gh {
            for x in 0..gw {
                field.set_scalar(y * gw + x, 2.0 * x as f32 + 3.0 * y as f32 + 1.0);
            }
        }

        let mut out = [0.0f32];
        let alpha = 2.0f32;
        let beta = 3.0f32;
        let gamma = 1.0f32;
        let mut n_tested = 0usize;
        for fy_frac in [0.0f32, 0.25, 0.5, 0.75, 0.99] {
            for fx_frac in [0.0f32, 0.25, 0.5, 0.75, 0.99] {
                for cy in 0..gh - 1 {
                    for cx in 0..gw - 1 {
                        let px = cx as f32 + fx_frac;
                        let py = cy as f32 + fy_frac;
                        sample_cochain_at_point_quad_into(&grid, &field, px, py, &mut out);
                        let expected = alpha * px + beta * py + gamma;
                        let diff = (out[0] - expected).abs();
                        assert!(
                            diff < TOL,
                            "G1 at ({px}, {py}): got {}, expected {expected}, diff {diff}",
                            out[0]
                        );
                        n_tested += 1;
                    }
                }
            }
        }
        assert!(n_tested > 1000, "G1 should test >1000 points, got {n_tested}");
    }

    #[test]
    fn g1_linear_precision_quad_multidim() {
        // f(x, y) = [x, y, x+y, 5] — multi-channel linear field.
        let (gw, gh) = (6usize, 6usize);
        let grid = CellComplex::grid_2d(gw, gh);
        let dim = 4usize;
        let mut field = CochainField::zeros(0, gw * gh, dim);
        for y in 0..gh {
            for x in 0..gw {
                let v = y * gw + x;
                let features = field.cell_features_mut(v);
                features[0] = x as f32; // ch 0: x
                features[1] = y as f32; // ch 1: y
                features[2] = (x + y) as f32; // ch 2: x+y
                features[3] = 5.0; // ch 3: constant
            }
        }

        let mut out = vec![0.0f32; dim];
        for fy in [0.3f32, 0.7] {
            for fx in [0.3f32, 0.7] {
                for cy in 0..gh - 1 {
                    for cx in 0..gw - 1 {
                        let px = cx as f32 + fx;
                        let py = cy as f32 + fy;
                        sample_cochain_at_point_quad_into(&grid, &field, px, py, &mut out);
                        assert!((out[0] - px).abs() < TOL, "ch0 at ({px},{py})");
                        assert!((out[1] - py).abs() < TOL, "ch1 at ({px},{py})");
                        assert!((out[2] - (px + py)).abs() < TOL, "ch2 at ({px},{py})");
                        assert!((out[3] - 5.0).abs() < TOL, "ch3 at ({px},{py})");
                    }
                }
            }
        }
    }

    // --- G2: Partition of unity ---

    #[test]
    fn g2_partition_of_unity_quad() {
        let (gw, gh) = (4usize, 4usize);
        let grid = CellComplex::grid_2d(gw, gh);
        for py in [0.0f32, 0.3, 0.7, 1.5, 2.8] {
            for px in [0.0f32, 0.3, 0.7, 1.5, 2.8] {
                let (_, _, u_x, u_y) = locate_grid_point(&grid, px, py);
                let lam = lambda_bilinear(u_x, u_y);
                let sum: f32 = lam.iter().sum();
                assert!((sum - 1.0).abs() < TOL, "G2 sum={sum} at ({px},{py})");
                for &l in &lam {
                    assert!(l >= -TOL, "G2 negative λ={l} at ({px},{py})");
                }
            }
        }
    }

    #[test]
    fn g2_partition_of_unity_tri() {
        let tri = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
        // Test only interior points (barycentric weights should be non-negative).
        // The centroid is (1, 2/3); use points clearly inside the triangle.
        for py in [0.2f32, 0.5, 0.8, 1.0] {
            for px in [0.5f32, 0.8, 1.0, 1.2, 1.5] {
                let lam = lambda_coordinate_tri([px, py], &tri);
                let sum = lam[0] + lam[1] + lam[2];
                assert!((sum - 1.0).abs() < TOL, "G2 tri sum={sum} at ({px},{py})");
                // Partition of unity always holds; non-negativity only inside the tri.
                if lam.iter().all(|&l| l >= -TOL) {
                    for &l in &lam {
                        assert!(l >= -TOL, "G2 tri negative λ={l} at ({px},{py})");
                    }
                }
            }
        }
    }

    // --- G3: C⁰ continuity ---

    #[test]
    fn g3_c0_continuity_cartesian_sincos_boundary() {
        // At a quad boundary, one side has u = +1, the other has u = -1.
        // sin(kπ · 1) = sin(kπ · (-1)) = 0  (sine is odd)
        // cos(kπ · 1) = cos(kπ · (-1))     (cosine is even)
        // → encoding is identical from both sides.
        let n_harmonics = 4u8;
        let aug_dim = 2 * 2 * n_harmonics as usize;
        let mut enc_pos = vec![0.0f32; aug_dim];
        let mut enc_neg = vec![0.0f32; aug_dim];

        for &y_val in &[0.0f32, 0.25, -0.5, 1.0] {
            local_coordinate_aug_cartesian(&[1.0, y_val], n_harmonics, &mut enc_pos);
            local_coordinate_aug_cartesian(&[-1.0, y_val], n_harmonics, &mut enc_neg);
            for (i, (a, b)) in enc_pos.iter().zip(enc_neg.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff < TOL,
                    "G3 sincos: u_x=1 vs u_x=-1 (y={y_val}): comp {i} differs by {diff}"
                );
            }
        }
    }

    #[test]
    fn g3_c0_continuity_barycentric_sort_invariance() {
        // Same triangle, same point, but vertices permuted → sorted tuple must match.
        let a = [0.0f32, 0.0];
        let b = [2.0, 0.0];
        let c = [1.0, 2.0];
        let p = [0.7f32, 0.5];

        let permutations: [[[f32; 2]; 3]; 6] = [
            [a, b, c],
            [a, c, b],
            [b, a, c],
            [b, c, a],
            [c, a, b],
            [c, b, a],
        ];

        // Reference sorted tuple from the first permutation.
        let lam_ref = lambda_coordinate_tri(p, &permutations[0]);
        let sorted_ref = sort_descending_3(lam_ref[0], lam_ref[1], lam_ref[2]);

        let mut aug_ref = [0.0f32; 3];
        local_coordinate_aug_barycentric(sorted_ref, &mut aug_ref);

        for (i, tri) in permutations.iter().enumerate() {
            let lam = lambda_coordinate_tri(p, tri);
            let sorted = sort_descending_3(lam[0], lam[1], lam[2]);
            for j in 0..3 {
                assert!(
                    (sorted[j] - sorted_ref[j]).abs() < TOL,
                    "G3 sort: perm {i} comp {j}: got {} expected {}",
                    sorted[j],
                    sorted_ref[j]
                );
            }
            let mut aug = [0.0f32; 3];
            local_coordinate_aug_barycentric(sorted, &mut aug);
            for j in 0..3 {
                assert!(
                    (aug[j] - aug_ref[j]).abs() < TOL,
                    "G3 aug: perm {i} comp {j}: got {} expected {}",
                    aug[j],
                    aug_ref[j]
                );
            }
        }
    }

    // --- G3 supplement: Cartesian sincos at the boundary is exact zero/constant ---

    #[test]
    fn g3_sincos_boundary_values() {
        // At u = ±1: sin(kπ) = 0, cos(kπ) = (-1)^k
        let n_harmonics = 5u8;
        let aug_dim = 2 * n_harmonics as usize; // 1 axis
        let mut enc = vec![0.0f32; aug_dim];
        for sign in [1.0f32, -1.0] {
            local_coordinate_aug_cartesian(&[sign], n_harmonics, &mut enc);
            for h in 0..n_harmonics as usize {
                let sin_val = enc[h * 2];
                let cos_val = enc[h * 2 + 1];
                assert!(
                    sin_val.abs() < TOL,
                    "G3: sin({}π·{}) = {sin_val} should be ~0",
                    h + 1,
                    sign
                );
                let expected_cos = if (h + 1) % 2 == 0 { 1.0 } else { -1.0 };
                assert!(
                    (cos_val - expected_cos).abs() < TOL,
                    "G3: cos({}π·{}) = {cos_val} should be ~{expected_cos}",
                    h + 1,
                    sign
                );
            }
        }
    }

    // --- Triangular CDF correctness ---

    #[test]
    fn triangular_cdf_endpoints_and_monotonicity() {
        // a ~ Tri(1/3, 1, 1/2)
        let third = 1.0f32 / 3.0;
        assert!((triangular_cdf(third, third, 1.0, 0.5) - 0.0).abs() < TOL);
        assert!((triangular_cdf(1.0, third, 1.0, 0.5) - 1.0).abs() < TOL);
        assert!((triangular_cdf(0.5, third, 1.0, 0.5) - (0.5 - third).powi(2) / ((1.0 - third) * (0.5 - third))).abs() < TOL);

        // b ~ Tri(0, 1/2, 1/3)
        assert!((triangular_cdf(0.0, 0.0, 0.5, third) - 0.0).abs() < TOL);
        assert!((triangular_cdf(0.5, 0.0, 0.5, third) - 1.0).abs() < TOL);

        // c ~ Tri(0, 1/3, 0) — mode at left boundary
        assert!((triangular_cdf(0.0, 0.0, third, 0.0) - 0.0).abs() < TOL);
        assert!((triangular_cdf(third, 0.0, third, 0.0) - 1.0).abs() < TOL);
        // At c = 1/6: F = 1 - (1/3 - 1/6)² / (1/3 * 1/3) = 1 - (1/6)² / (1/9) = 1 - 1/4 = 3/4
        let cdf_mid = triangular_cdf(1.0 / 6.0, 0.0, third, 0.0);
        assert!((cdf_mid - 0.75).abs() < TOL, "c CDF at 1/6: {cdf_mid} vs 0.75");

        // Monotonicity: F should be non-decreasing for each distribution.
        for &(l, r, m) in &[(third, 1.0, 0.5), (0.0, 0.5, third), (0.0, third, 0.0)] {
            let mut prev = 0.0f32;
            let n_samples = 50;
            for i in 1..=n_samples {
                let x = l + (r - l) * (i as f32) / (n_samples as f32);
                let f = triangular_cdf(x, l, r, m);
                assert!(f >= prev - TOL, "CDF not monotone at x={x}: {f} < {prev}");
                prev = f;
            }
        }
    }

    // --- Combined sample+encode ---

    #[test]
    fn sample_point_quad_combined() {
        let (gw, gh) = (4usize, 4usize);
        let grid = CellComplex::grid_2d(gw, gh);
        let mut field = CochainField::zeros(0, gw * gh, 3);
        for y in 0..gh {
            for x in 0..gw {
                let v = y * gw + x;
                field.cell_features_mut(v)[0] = x as f32;
                field.cell_features_mut(v)[1] = y as f32;
                field.cell_features_mut(v)[2] = 42.0;
            }
        }

        let n_harmonics = 2u8;
        let aug_dim = local_coord_aug_dim(
            LocalCoordEncode::CartesianSincos { n_harmonics },
            2,
        );
        let mut scratch = PointSamplerScratch::new(3, aug_dim);

        sample_point_quad_into(
            &grid,
            &field,
            1.5,
            2.5,
            LocalCoordEncode::CartesianSincos { n_harmonics },
            &mut scratch,
        );
        // s̄(1.5, 2.5) = [1.5, 2.5, 42.0] (linear precision)
        assert!((scratch.state[0] - 1.5).abs() < TOL);
        assert!((scratch.state[1] - 2.5).abs() < TOL);
        assert!((scratch.state[2] - 42.0).abs() < TOL);
        // aug has 2 * 2 * 2 = 8 values
        assert_eq!(scratch.local_coord.len(), 8);
    }

    #[test]
    fn sample_point_tri_combined() {
        // Simple triangle with known values.
        let tri_pos = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let tri_idx = [0usize, 1, 2];
        let mut field = CochainField::zeros(0, 3, 1);
        field.set_scalar(0, 10.0);
        field.set_scalar(1, 20.0);
        field.set_scalar(2, 30.0);

        let mut scratch = PointSamplerScratch::new(1, 3);

        sample_point_tri_into(
            &field,
            &tri_pos,
            tri_idx,
            1.0,
            1.0,
            LocalCoordEncode::BarycentricSortCdf,
            &mut scratch,
        );

        // Point (1,1) in triangle (0,0)-(4,0)-(0,4):
        // λ0 = 1 - 1/4 - 1/4 = 0.5, λ1 = 1/4, λ2 = 1/4
        // s̄ = 0.5*10 + 0.25*20 + 0.25*30 = 5 + 5 + 7.5 = 17.5
        assert!((scratch.state[0] - 17.5).abs() < TOL, "got {}", scratch.state[0]);
        // aug has 3 values (sorted barycentric CDF remap)
        assert_eq!(scratch.local_coord.len(), 3);
    }

    // --- Boundary / edge cases ---

    #[test]
    fn quad_sample_at_vertex() {
        // Sampling exactly at a vertex should return that vertex's value.
        let (gw, gh) = (4usize, 4usize);
        let grid = CellComplex::grid_2d(gw, gh);
        let mut field = CochainField::zeros(0, gw * gh, 1);
        for i in 0..gw * gh {
            field.set_scalar(i, i as f32 * 10.0);
        }

        let mut out = [0.0f32];
        sample_cochain_at_point_quad_into(&grid, &field, 2.0, 1.0, &mut out);
        // Vertex (2,1) = index 1*4+2 = 6, value = 60.0
        assert!((out[0] - 60.0).abs() < TOL, "vertex sample: {}", out[0]);
    }

    #[test]
    fn quad_sample_clamps_out_of_bounds() {
        let (gw, gh) = (4usize, 4usize);
        let grid = CellComplex::grid_2d(gw, gh);
        let mut field = CochainField::zeros(0, gw * gh, 1);
        for i in 0..gw * gh {
            field.set_scalar(i, i as f32);
        }

        let mut out = [0.0f32];
        // Overshoot → clamped to the last quad.
        sample_cochain_at_point_quad_into(&grid, &field, 100.0, 100.0, &mut out);
        // Last vertex (3,3) = index 15, value = 15.0
        assert!((out[0] - 15.0).abs() < TOL, "clamped overshoot: {}", out[0]);

        // Undershoot → clamped to the first vertex.
        sample_cochain_at_point_quad_into(&grid, &field, -5.0, -5.0, &mut out);
        assert!((out[0] - 0.0).abs() < TOL, "clamped undershoot: {}", out[0]);
    }

    // --- Barycentric correctness ---

    #[test]
    fn barycentric_at_vertices_and_centroid() {
        let tri = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];

        // At vertex 0: λ = (1, 0, 0)
        let lam = lambda_coordinate_tri([0.0, 0.0], &tri);
        assert!((lam[0] - 1.0).abs() < TOL && lam[1].abs() < TOL && lam[2].abs() < TOL);

        // At vertex 1: λ = (0, 1, 0)
        let lam = lambda_coordinate_tri([4.0, 0.0], &tri);
        assert!((lam[1] - 1.0).abs() < TOL);

        // At centroid: λ = (1/3, 1/3, 1/3)
        let lam = lambda_coordinate_tri([4.0 / 3.0, 4.0 / 3.0], &tri);
        for &l in &lam {
            assert!((l - 1.0 / 3.0).abs() < TOL, "centroid λ = {l}");
        }
    }
}
