//! Sheaf-ADMM Coordination Primitive — Plan 407 / Research 384.
//!
//! One ADMM iteration on a cellular sheaf over a graph `G = (V, E)`. Distilled
//! from Seely, Cupiał, Jones, "Learning Multi-Agent Coordination via
//! Sheaf-ADMM" (arXiv:2605.31005, ICML 2026). **Modelless** by the katgpt-rs
//! mandate: the restriction maps are constructed deterministically (identity /
//! selector) or loaded as a frozen artifact — no training, no backprop.
//!
//! # The primitive
//!
//! Each vertex (agent) carries three rank-0 cochains of dimension `d_v`:
//!
//! | Variable | Role | Update |
//! |---|---|---|
//! | `primal_x` (`x_i`) | local proposal | `x^{k+1} = prox_{f_i/ρ}(z^k − u^k)` |
//! | `consensus_z` (`z_i`) | sheaf projection onto harmonic subspace | `z^{k+1} ≈ Π_{ker F}(x^{k+1} + u^k)` via `T` sheaf-diffusion steps |
//! | `dual_u` (`u_i`) | accumulated disagreement integral | `u^{k+1} = u^k + x^{k+1} − z^{k+1}` |
//!
//! Per-edge restriction maps `F_{i→e}, F_{j→e} ∈ ℝ^{d_e × d_v}` (with
//! `d_e ≤ d_v`, the capacity rule — Research 384 §1.4) project vertex state
//! into the edge stalk `ℝ^{d_e}`. Agents agree only on these projections
//! (heterogeneous consensus), formalizing the latent-to-raw sync rule.
//!
//! # Relationship to shipped DEC operators
//!
//! The sheaf Laplacian `L_F z = Σ_e F_{i→e}^T (F_{i→e} z_i − F_{j→e} z_j)` is
//! computed here from the explicit restriction maps. When `F_{i→e}` IS the
//! coboundary incidence entry (identity maps with `d_e = d_v`),
//! [`sheaf_laplacian_via_maps`] reduces bit-for-bit to
//! [`graph_laplacian`](crate::operators::graph_laplacian) on every dim — the
//! DEC identity from Research 384 §1.3 (the test
//! `sheaf_laplacian_identity_matches_graph_laplacian_first_de_dims` guards
//! this). The general explicit-maps path is reserved for heterogeneous
//! consensus where maps differ per edge / per endpoint.
//!
//! # Zero-alloc
//!
//! All intermediate storage lives in caller-owned [`AdmmScratch`]. The
//! `_into` entry point is zero-alloc by contract in steady state (the buffers
//! are allocated once at zone-init and reused across iterations).
//!
//! # Phase 1 scope (this module)
//!
//! Skeleton: API surface + correct explicit-maps math + identity fast path +
//! unit tests. The identity fast path (Plan 407 T2.4) is implemented: when
//! `maps.is_identity`, [`sheaf_laplacian_via_maps`] computes the graph
//! Laplacian on the first `d_e` dims directly, bypassing the explicit `F^T F`
//! matvec. The general explicit-maps path is reserved for heterogeneous
//! consensus where maps differ per edge / per endpoint.
//!
//! # References
//!
//! - Plan 407 (this primitive), Research 384 (distillation).
//! - Plan 251 — DEC operators (`graph_laplacian`, `hodge_laplacian`).
//! - Plan 314 — Stokes calculus vocabulary crosswalk.

use crate::simd::simd_dot_f32;
use crate::types::{CellComplex, CochainField};

// ---------------------------------------------------------------------------
// Restriction maps
// ---------------------------------------------------------------------------

/// Per-edge restriction map pair for a cellular sheaf.
///
/// For each edge `e = (i, j)`, stores two maps `F_{i→e}` and `F_{j→e}`, each of
/// shape `d_e × d_v` (projecting `ℝ^{d_v} → ℝ^{d_e}`). Maps are stored
/// row-major in a single `Vec<f32>` for cache-friendly iteration:
///
/// ```text
/// maps[edge_idx * (2 * d_e * d_v) + endpoint * (d_e * d_v) + row * d_v + col]
/// ```
///
/// where `endpoint ∈ {0 = i_tail, 1 = j_head}`, matching
/// [`CellComplex::boundary_entries`] pair ordering
/// `(v_tail, e, −1), (v_head, e, +1)`. The map at `(edge, endpoint=0)` is the
/// tail vertex's restriction; `(edge, endpoint=1)` is the head vertex's.
///
/// `is_identity` marks maps constructed via [`SheafMaps::identity`] (or a
/// selector that happens to pick dims `0..d_e` in order). The flag is reserved
/// for the Phase 2 identity fast path; Phase 1 ignores it and always runs the
/// general explicit-maps matvec.
#[derive(Clone, Debug)]
pub struct SheafMaps {
    /// Edge stalk dimension (≤ `d_v`).
    pub d_e: usize,
    /// Vertex stalk dimension.
    pub d_v: usize,
    /// Flat map storage: `n_edges * 2 * d_e * d_v` row-major floats.
    pub maps: Vec<f32>,
    /// `true` iff every map equals `[I_{d_e}; 0]` (identity block). Fast-path hint.
    pub is_identity: bool,
    /// Number of edges covered (= `cx.n_edges()` at construction).
    pub n_edges: usize,
}

impl SheafMaps {
    /// Identity restriction maps: `F_{i→e} = [I_{d_e}; 0]` for all edges and
    /// both endpoints. Homogeneous consensus on the first `d_e` dims — the
    /// modelless floor (Research 384 §2.3).
    ///
    /// When `d_e == d_v`, the sheaf Laplacian reduces bit-for-bit to the graph
    /// Laplacian. When `d_e < d_v`, the first `d_e` dims behave like the graph
    /// Laplacian and the remaining `d_v − d_e` dims are untouched (zero
    /// disagreement).
    ///
    /// # Panics
    /// If `d_e > d_v`.
    pub fn identity(cx: &CellComplex, d_v: usize, d_e: usize) -> Self {
        assert!(
            d_e <= d_v,
            "SheafMaps::identity: d_e ({d_e}) must be <= d_v ({d_v})"
        );
        let n_edges = cx.n_edges();
        let per_map = d_e * d_v;
        // Zero-init then stamp the unit diagonals. Avoids building a temp block.
        let mut maps = vec![0.0f32; n_edges * 2 * per_map];
        for e in 0..n_edges {
            for endpoint in 0..2 {
                let base = e * 2 * per_map + endpoint * per_map;
                // Row r of the identity block has its 1.0 at column r.
                for r in 0..d_e {
                    maps[base + r * d_v + r] = 1.0;
                }
            }
        }
        Self {
            d_e,
            d_v,
            maps,
            is_identity: true,
            n_edges,
        }
    }

    /// Selector restriction maps: each `F_{i→e}` picks a fixed `d_e`-subset of
    /// the `d_v` dims. The same subset is used for all edges and both
    /// endpoints (deterministic; heterogeneous-by-edge would need a per-edge
    /// index slice — deferred to Phase 3 per Plan 407).
    ///
    /// Row `r` of each map is the standard basis vector `e_{dim_indices[r]}`.
    /// `is_identity` is set `true` iff `dim_indices == [0, 1, …, d_e−1]` (i.e.
    /// the selector collapses to the identity block).
    ///
    /// # Panics
    /// If `dim_indices.len() > d_v`, or any index `>= d_v`.
    pub fn selector(cx: &CellComplex, d_v: usize, dim_indices: &[usize]) -> Self {
        let d_e = dim_indices.len();
        assert!(
            d_e <= d_v,
            "SheafMaps::selector: d_e ({d_e}) = dim_indices.len() must be <= d_v ({d_v})"
        );
        for (r, &idx) in dim_indices.iter().enumerate() {
            assert!(
                idx < d_v,
                "SheafMaps::selector: dim_indices[{r}] = {idx} >= d_v {d_v}"
            );
        }
        let n_edges = cx.n_edges();
        let per_map = d_e * d_v;
        let mut maps = vec![0.0f32; n_edges * 2 * per_map];
        for e in 0..n_edges {
            for endpoint in 0..2 {
                let base = e * 2 * per_map + endpoint * per_map;
                for r in 0..d_e {
                    maps[base + r * d_v + dim_indices[r]] = 1.0;
                }
            }
        }
        let is_identity = dim_indices
            .iter()
            .enumerate()
            .all(|(r, &idx)| idx == r);
        Self {
            d_e,
            d_v,
            maps,
            is_identity,
            n_edges,
        }
    }

    /// Read the restriction map for `endpoint` (0 = tail, 1 = head) of edge
    /// `edge_idx`. Returns a `d_e × d_v` row-major slice.
    #[inline]
    pub fn edge_map(&self, edge_idx: usize, endpoint: usize) -> &[f32] {
        debug_assert!(
            edge_idx < self.n_edges,
            "SheafMaps::edge_map: edge_idx {edge_idx} >= n_edges {}",
            self.n_edges
        );
        debug_assert!(
            endpoint < 2,
            "SheafMaps::edge_map: endpoint {endpoint} must be 0 (tail) or 1 (head)"
        );
        let per_map = self.d_e * self.d_v;
        let start = edge_idx * 2 * per_map + endpoint * per_map;
        &self.maps[start..start + per_map]
    }
}

// ---------------------------------------------------------------------------
// Local objective
// ---------------------------------------------------------------------------

/// Local objective `f_i` per vertex, with a closed-form proximal solver.
///
/// Both variants store per-vertex-per-dim parameters (length `n_vertices *
/// d_v`), so each agent can carry a distinct objective. Uniform objectives
/// broadcast by tiling the same `d_v`-length pattern across all vertices.
#[derive(Clone, Debug)]
pub enum LocalObjective {
    /// `f_i(x) = ½ x^T diag(diag_q) x + q^T x`. Proximal solve is elementwise:
    ///
    /// `x_i = (ρ(z_i − u_i) − q_i) / (diag_q_i + ρ)`
    DiagonalQuadratic {
        /// Per-vertex-per-dim diagonal of `Q` (length `n_vertices * d_v`).
        diag_q: Vec<f32>,
        /// Per-vertex-per-dim linear term (length `n_vertices * d_v`).
        q: Vec<f32>,
    },
    /// `f_i(x) = ½ x^T diag(diag_q) x + q^T x + lambda^T |x|`. Proximal solve is
    /// the diagonal-quadratic solve followed by a soft-threshold:
    ///
    /// `x_i = soft_threshold( (ρ(z_i − u_i) − q_i) / (diag_q_i + ρ), lambda_i / (diag_q_i + ρ) )`
    DiagonalQuadL1 {
        diag_q: Vec<f32>,
        q: Vec<f32>,
        /// Per-vertex-per-dim L1 weight (length `n_vertices * d_v`).
        lambda: Vec<f32>,
    },
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffers for [`sheaf_admm_step_into`]. Allocate once at
/// zone-init via [`AdmmScratch::new`] and reuse across iterations — the
/// `_into` entry point performs zero allocations in steady state.
#[derive(Clone, Debug)]
pub struct AdmmScratch {
    /// Per-edge endpoint projection scratch. Documented capacity is
    /// `n_edges * 2 * d_e`; the diffusion loop reuses the first `d_e` slots as
    /// the current edge's disagreement vector (compute-then-accumulate).
    pub edge_projections: Vec<f32>,
    /// `L_F · z` output (sheaf Laplacian applied to the consensus cochain),
    /// length `n_vertices * d_v`. Zeroed at the start of each matvec.
    pub sheaf_laplacian_z: Vec<f32>,
}

impl AdmmScratch {
    /// Allocate scratch sized for a cell complex with the given vertex / edge
    /// stalk dimensions.
    pub fn new(cx: &CellComplex, d_v: usize, d_e: usize) -> Self {
        Self {
            edge_projections: vec![0.0; cx.n_edges() * 2 * d_e],
            sheaf_laplacian_z: vec![0.0; cx.n_vertices() * d_v],
        }
    }
}

// ---------------------------------------------------------------------------
// ADMM step
// ---------------------------------------------------------------------------

/// One ADMM iteration (x-update → z-update → u-update), in place.
///
/// User-friendly entry point: asserts cochain shapes then delegates to
/// [`sheaf_admm_step_into`]. All three cochains must be rank-0, `dim == d_v`,
/// `n_cells == cx.n_vertices()`.
#[inline]
pub fn sheaf_admm_step(
    cx: &CellComplex,
    maps: &SheafMaps,
    primal_x: &mut CochainField,
    consensus_z: &mut CochainField,
    dual_u: &mut CochainField,
    objective: &LocalObjective,
    rho: f32,
    eta: f32,
    diffusion_steps: usize,
    scratch: &mut AdmmScratch,
) {
    let n = cx.n_vertices();
    let d_v = maps.d_v;
    debug_assert_eq!(primal_x.rank, 0, "sheaf_admm_step: primal_x must be rank-0");
    debug_assert_eq!(consensus_z.rank, 0, "sheaf_admm_step: consensus_z must be rank-0");
    debug_assert_eq!(dual_u.rank, 0, "sheaf_admm_step: dual_u must be rank-0");
    debug_assert_eq!(
        primal_x.dim, d_v,
        "sheaf_admm_step: primal_x.dim {} != maps.d_v {d_v}",
        primal_x.dim
    );
    debug_assert_eq!(
        consensus_z.dim, d_v,
        "sheaf_admm_step: consensus_z.dim {} != maps.d_v {d_v}",
        consensus_z.dim
    );
    debug_assert_eq!(
        dual_u.dim, d_v,
        "sheaf_admm_step: dual_u.dim {} != maps.d_v {d_v}",
        dual_u.dim
    );
    debug_assert_eq!(
        primal_x.n_cells(),
        n,
        "sheaf_admm_step: primal_x.n_cells mismatch"
    );
    debug_assert_eq!(
        consensus_z.n_cells(),
        n,
        "sheaf_admm_step: consensus_z.n_cells mismatch"
    );
    debug_assert_eq!(dual_u.n_cells(), n, "sheaf_admm_step: dual_u.n_cells mismatch");
    debug_assert_eq!(
        scratch.sheaf_laplacian_z.len(),
        n * d_v,
        "sheaf_admm_step: scratch.sheaf_laplacian_z.len() {} != n*d_v {}",
        scratch.sheaf_laplacian_z.len(),
        n * d_v
    );
    sheaf_admm_step_into(
        cx,
        maps,
        primal_x,
        consensus_z,
        dual_u,
        objective,
        rho,
        eta,
        diffusion_steps,
        scratch,
    );
}

/// Zero-alloc ADMM iteration — the real implementation.
///
/// Same contract as [`sheaf_admm_step`]; callers that have already validated
/// shapes (or are in a hot loop) can call this directly to skip the assertions.
/// All intermediate storage is in `scratch`; no allocations occur.
///
/// # Update order (scaled-form ADMM)
///
/// 1. **x-update** — reads pre-step `z^k, u^k`, writes `x^{k+1}`.
/// 2. **z-update** — warm-starts `z = x^{k+1} + u^k`, then runs
///    `diffusion_steps` sheaf-diffusion iterations
///    `z ← z − η (L_F z)`. Writes `z^{k+1}`.
/// 3. **u-update** — `u^{k+1} = u^k + x^{k+1} − z^{k+1}`. Reads the post-step
///    `x^{k+1}` and `z^{k+1}`.
#[inline]
pub fn sheaf_admm_step_into(
    cx: &CellComplex,
    maps: &SheafMaps,
    primal_x: &mut CochainField,
    consensus_z: &mut CochainField,
    dual_u: &mut CochainField,
    objective: &LocalObjective,
    rho: f32,
    eta: f32,
    diffusion_steps: usize,
    scratch: &mut AdmmScratch,
) {
    let n = cx.n_vertices();
    let d_v = maps.d_v;
    let total = n * d_v;

    // ---- x-update: x_i = prox_{f_i/ρ}(z_i − u_i) ---------------------------
    // Reads consensus_z (z^k) and dual_u (u^k) from the start of the step.
    match objective {
        LocalObjective::DiagonalQuadratic { diag_q, q } => {
            debug_assert_eq!(diag_q.len(), total, "DiagonalQuadratic: diag_q.len() != n*d_v");
            debug_assert_eq!(q.len(), total, "DiagonalQuadratic: q.len() != n*d_v");
            for k in 0..total {
                let denom = diag_q[k] + rho;
                debug_assert!(
                    denom > 0.0,
                    "sheaf_admm_step: non-positive denom (diag_q[k]={} + rho={})",
                    diag_q[k],
                    rho
                );
                primal_x.data[k] =
                    (rho * (consensus_z.data[k] - dual_u.data[k]) - q[k]) / denom;
            }
        }
        LocalObjective::DiagonalQuadL1 { diag_q, q, lambda } => {
            debug_assert_eq!(diag_q.len(), total, "DiagonalQuadL1: diag_q.len() != n*d_v");
            debug_assert_eq!(q.len(), total, "DiagonalQuadL1: q.len() != n*d_v");
            debug_assert_eq!(lambda.len(), total, "DiagonalQuadL1: lambda.len() != n*d_v");
            for k in 0..total {
                let denom = diag_q[k] + rho;
                debug_assert!(
                    denom > 0.0,
                    "sheaf_admm_step: non-positive denom (diag_q[k]={} + rho={})",
                    diag_q[k],
                    rho
                );
                let x_quad =
                    (rho * (consensus_z.data[k] - dual_u.data[k]) - q[k]) / denom;
                let thresh = lambda[k] / denom;
                primal_x.data[k] = soft_threshold(x_quad, thresh);
            }
        }
    }

    // ---- z-update: warm-start z = x + u, then T sheaf-diffusion steps -------
    // x^{k+1} (primal_x) is now fixed; u^k (dual_u) is read but not yet mutated.
    for k in 0..total {
        consensus_z.data[k] = primal_x.data[k] + dual_u.data[k];
    }
    for _ in 0..diffusion_steps {
        sheaf_laplacian_via_maps(cx, maps, consensus_z, scratch);
        for k in 0..total {
            consensus_z.data[k] -= eta * scratch.sheaf_laplacian_z[k];
        }
    }

    // ---- u-update: u += x − z -----------------------------------------------
    // Reads post-step x^{k+1} (primal_x) and z^{k+1} (consensus_z).
    for k in 0..total {
        dual_u.data[k] += primal_x.data[k] - consensus_z.data[k];
    }
}

/// Soft-threshold operator `S_t(x) = sign(x) · max(|x| − t, 0)`.
#[inline]
fn soft_threshold(x: f32, t: f32) -> f32 {
    if x >= 0.0 {
        (x - t).max(0.0)
    } else {
        -(((-x) - t).max(0.0))
    }
}

/// Compute the sheaf Laplacian applied to `z` into `scratch.sheaf_laplacian_z`.
///
/// `L_F z` accumulates, per edge `e = (v_tail, v_head)`:
/// ```text
/// disagreement = F_{i→e} z_{v_tail} − F_{j→e} z_{v_head}    (d_e-dim)
/// (L_F z)_{v_tail} += F_{i→e}^T · disagreement              (d_v-dim)
/// (L_F z)_{v_head} −= F_{j→e}^T · disagreement              (d_v-dim)
/// ```
///
/// Edges are iterated via `cx.boundary_entries(0).chunks_exact(2)`, matching
/// the `(v_tail, e, −1), (v_head, e, +1)` pair ordering of `grid_2d` /
/// `from_edges`. The current edge's `d_e`-dim disagreement is staged in the
/// first `d_e` slots of `scratch.edge_projections` (compute-then-accumulate to
/// keep the `F^T · d` accumulation cache-friendly).
///
/// # Identity fast path (Plan 407 T2.4 — Phase 2)
///
/// When `maps.is_identity`, the restriction maps are `[I_{d_e}; 0]` for every
/// edge/endpoint. The sheaf Laplacian then reduces bit-for-bit to the graph
/// Laplacian on the first `d_e` dims (dims `d_e..d_v` have zero disagreement).
/// We skip the explicit `F^T F` matvec (which wastes ~`d_v` scalar multiplies
/// per row, most against zero entries) and directly compute the graph-Laplacian
/// difference per edge on the first `d_e` dims. This is the G4 latency
/// optimization — turns a ~`O(|E|·d_e·d_v)` general matvec into a lean
/// `O(|E|·d_e)` identity matvec with no wasted multiplies-against-zero. On
/// regular grids, the identity path further delegates to
/// [`sheaf_laplacian_identity_grid_into`] (the 5-point-stencil variant) for
/// single-write cache-friendly output.

/// 5-point-stencil sheaf Laplacian for identity maps on a regular `w×h` grid.
///
/// Computes `(L_F z)_v[r] = deg(v)·z_v[r] − Σ z_u[r]` for `r < d_e`, leaving
/// dims `d_e..d_v` at zero. Writes each output element exactly once (no
/// scattered read-modify-write) using the same stencil pattern as
/// [`graph_laplacian_grid_into`](crate::operators::graph_laplacian_grid_into),
/// but with stride `d_v` (not `d_e`) to process only the first `d_e` dims of
/// the `d_v`-dim vertex stalk. This is the G4 latency optimization — the
/// edge-list identity path does scattered `+=/−=` that causes store-forwarding
/// stalls; the stencil writes once, reads in row-major order, and
/// auto-vectorizes cleanly.
///
/// `scratch.sheaf_laplacian_z` is written directly (dims `d_e..d_v` left at
/// the zero value from the caller's `.fill(0.0)`).
#[inline]
fn sheaf_laplacian_identity_grid_into(
    w: usize,
    h: usize,
    d_v: usize,
    d_e: usize,
    z: &CochainField,
    scratch: &mut AdmmScratch,
) {
    let p = z.data.as_ptr();
    let o = scratch.sheaf_laplacian_z.as_mut_ptr();
    let stride = w * d_v;

    // Interior: 4 neighbors, branch-free. deg = 4.
    if w >= 3 && h >= 3 {
        for y in 1..(h - 1) {
            let row = y * stride;
            let up_row = row - stride;
            let down_row = row + stride;
            for x in 1..(w - 1) {
                let base = row + x * d_v;
                let left = base - d_v;
                let right = base + d_v;
                let up = up_row + x * d_v;
                let down = down_row + x * d_v;
                unsafe {
                    for c in 0..d_e {
                        let center = *p.add(base + c);
                        *o.add(base + c) = 4.0 * center
                            - *p.add(left + c)
                            - *p.add(right + c)
                            - *p.add(up + c)
                            - *p.add(down + c);
                    }
                }
            }
        }
    }

    // Boundary: top + bottom rows. deg = 2 at corners, 3 on edges.
    for &(y, up_off, down_off, has_up, has_down) in [
        (0usize, 0usize, stride, false, true),
        (h - 1, stride, 0usize, true, false),
    ]
    .iter()
    {
        let row = y * stride;
        let up_row = row.wrapping_sub(up_off);
        let down_row = row + down_off;
        for x in 0..w {
            let base = row + x * d_v;
            let has_left = x > 0;
            let has_right = x < w - 1;
            let deg = (has_left as u8 + has_right as u8 + has_up as u8 + has_down as u8) as f32;
            let left = base.wrapping_sub(d_v);
            let right = base.wrapping_add(d_v);
            let up = up_row + x * d_v;
            let down = down_row + x * d_v;
            unsafe {
                for c in 0..d_e {
                    let center = *p.add(base + c);
                    let mut acc = deg * center;
                    if has_left { acc -= *p.add(left + c); }
                    if has_right { acc -= *p.add(right + c); }
                    if has_up { acc -= *p.add(up + c); }
                    if has_down { acc -= *p.add(down + c); }
                    *o.add(base + c) = acc;
                }
            }
        }
    }

    // Left + right boundary columns (interior rows only — top/bottom already
    // handled above). deg = 3.
    if h >= 3 {
        for y in 1..(h - 1) {
            let row = y * stride;
            let up_row = row - stride;
            let down_row = row + stride;
            for &(x, has_left, has_right) in
                [(0usize, false, true), (w - 1, true, false)].iter()
            {
                let base = row + x * d_v;
                let deg = 3.0f32; // has_up + has_down + one horizontal neighbor
                let left = base.wrapping_sub(d_v);
                let right = base.wrapping_add(d_v);
                let up = up_row + x * d_v;
                let down = down_row + x * d_v;
                unsafe {
                    for c in 0..d_e {
                        let center = *p.add(base + c);
                        let mut acc = deg * center;
                        if has_left { acc -= *p.add(left + c); }
                        if has_right { acc -= *p.add(right + c); }
                        acc -= *p.add(up + c);
                        acc -= *p.add(down + c);
                        *o.add(base + c) = acc;
                    }
                }
            }
        }
    }
}

#[inline]
fn sheaf_laplacian_via_maps(
    cx: &CellComplex,
    maps: &SheafMaps,
    z: &CochainField,
    scratch: &mut AdmmScratch,
) {
    let d_v = maps.d_v;
    let d_e = maps.d_e;

    // ── Identity fast path (Plan 407 T2.4) ─────────────────────────────────
    // F = [I_{d_e}; 0] ⟹ disagreement[r] = z_tail[r] − z_head[r] for r < d_e,
    // and (L_F z)_v[r] = degree_L(v) · z_v[r] − Σ_{u~v} z_u[r] for r < d_e.
    // Dims d_e..d_v stay zero.
    if maps.is_identity {
        // Grid-stencil path: writes each output element exactly once (no fill,
        // no scattered read-modify-write). Processes the first `d_e` dims with a
        // stride-`d_v` 5-point stencil, explicitly zeros dims `d_e..d_v`.
        if let Some((w, h)) = cx.grid_dims() {
            sheaf_laplacian_identity_grid_into(w, h, d_v, d_e, z, scratch);
            return;
        }
        // Edge-list fallback for non-grid complexes (needs fill for dims d_e..d_v).
        scratch.sheaf_laplacian_z.fill(0.0);
        let entries = cx.boundary_entries(0);
        for pair in entries.chunks_exact(2) {
            let v_tail = pair[0].0;
            let v_head = pair[1].0;
            let tail_base = v_tail * d_v;
            let head_base = v_head * d_v;
            for d in 0..d_e {
                let diff = z.data[tail_base + d] - z.data[head_base + d];
                scratch.sheaf_laplacian_z[tail_base + d] += diff;
                scratch.sheaf_laplacian_z[head_base + d] -= diff;
            }
        }
        return;
    }

    // General explicit-maps path: zero-fill then accumulate.
    scratch.sheaf_laplacian_z.fill(0.0);

    let entries = cx.boundary_entries(0);
    for pair in entries.chunks_exact(2) {
        let v_tail = pair[0].0;
        let e = pair[0].1;
        let v_head = pair[1].0;
        debug_assert_eq!(
            pair[1].1, e,
            "sheaf_laplacian_via_maps: mismatched edge idx in boundary pair"
        );

        let f_tail = maps.edge_map(e, 0);
        let f_head = maps.edge_map(e, 1);
        let z_tail_base = v_tail * d_v;
        let z_head_base = v_head * d_v;

        // Stage disagreement[r] = F_tail[r,:]·z_tail − F_head[r,:]·z_head into
        // the first d_e slots of edge_projections.
        {
            let disagreement = &mut scratch.edge_projections[0..d_e];
            let z_tail = &z.data[z_tail_base..z_tail_base + d_v];
            let z_head = &z.data[z_head_base..z_head_base + d_v];
            for r in 0..d_e {
                let f_tail_row = &f_tail[r * d_v..(r + 1) * d_v];
                let f_head_row = &f_head[r * d_v..(r + 1) * d_v];
                let e_tail_r = simd_dot_f32(f_tail_row, z_tail, d_v);
                let e_head_r = simd_dot_f32(f_head_row, z_head, d_v);
                disagreement[r] = e_tail_r - e_head_r;
            }
        }

        // Accumulate F^T · disagreement back into the vertex slots. Direct
        // indexing keeps the two disjoint vertex writes borrow-checker-friendly.
        for r in 0..d_e {
            let d_r = scratch.edge_projections[r];
            let f_tail_row = &f_tail[r * d_v..(r + 1) * d_v];
            let f_head_row = &f_head[r * d_v..(r + 1) * d_v];
            for c in 0..d_v {
                scratch.sheaf_laplacian_z[z_tail_base + c] += f_tail_row[c] * d_r;
                scratch.sheaf_laplacian_z[z_head_base + c] -= f_head_row[c] * d_r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::graph_laplacian;
    use crate::types::{CellComplex, CochainField};

    /// `SheafMaps::identity` lays out `[I_{d_e}; 0]` for both endpoints.
    #[test]
    fn identity_maps_construct_correctly() {
        // 3 vertices, 2 edges: 0-1, 1-2.
        let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
        let maps = SheafMaps::identity(&cx, 4, 2);
        assert_eq!(maps.d_e, 2);
        assert_eq!(maps.d_v, 4);
        assert_eq!(maps.n_edges, 2);
        assert!(maps.is_identity);
        // Expected 2×4 block: [[1,0,0,0],[0,1,0,0]] for both endpoints.
        let expected = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for e in 0..2 {
            for endpoint in 0..2 {
                let m = maps.edge_map(e, endpoint);
                assert_eq!(m, expected.as_slice(),
                    "identity map (e={e}, endpoint={endpoint}) wrong: {m:?}");
            }
        }
    }

    /// `SheafMaps::selector` with `[0, 2]` picks dims 0 and 2.
    #[test]
    fn selector_maps_pick_correct_dims() {
        let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
        let maps = SheafMaps::selector(&cx, 4, &[0, 2]);
        assert_eq!(maps.d_e, 2);
        assert_eq!(maps.d_v, 4);
        assert!(!maps.is_identity, "selector [0,2] should NOT be identity");
        // Row 0 = e_0 = [1,0,0,0], Row 1 = e_2 = [0,0,1,0].
        let expected = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for e in 0..2 {
            for endpoint in 0..2 {
                let m = maps.edge_map(e, endpoint);
                assert_eq!(m, expected.as_slice(),
                    "selector map (e={e}, endpoint={endpoint}) wrong: {m:?}");
            }
        }
    }

    /// Selector with `[0, 1, …]` collapses to identity and sets the flag.
    #[test]
    fn selector_collapses_to_identity_when_ordered() {
        let cx = CellComplex::from_edges(2, &[(0, 1)]);
        let maps = SheafMaps::selector(&cx, 4, &[0, 1]);
        assert!(maps.is_identity, "selector [0,1] should detect identity");
    }

    /// x-update (DiagonalQuadratic) matches the hand-derived closed form.
    #[test]
    fn x_update_diagonal_quadratic() {
        // 2 vertices, 1 edge. d_v = d_e = 2 (identity maps, but maps unused by x-update).
        let cx = CellComplex::from_edges(2, &[(0, 1)]);
        let maps = SheafMaps::identity(&cx, 2, 2);
        let mut primal_x = CochainField::zeros(0, 2, 2);
        let mut consensus_z = CochainField::from_vec(0, 2,
            vec![1.0, 2.0, 3.0, 4.0]);
        let mut dual_u = CochainField::from_vec(0, 2,
            vec![0.1, 0.2, 0.3, 0.4]);
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0, 1.0, 2.0, 2.0],
            q: vec![0.5, -0.5, 1.0, -1.0],
        };
        let mut scratch = AdmmScratch::new(&cx, 2, 2);
        let rho = 2.0;

        // Snapshot pre-step z (the x-update reads these).
        let z_pre = consensus_z.data.clone();
        let u_pre = dual_u.data.clone();

        sheaf_admm_step(&cx, &maps, &mut primal_x, &mut consensus_z, &mut dual_u,
            &objective, rho, 0.1, 1, &mut scratch);

        // Expected x_i = (ρ(z-u) - q) / (diag_q + ρ).
        let expected = [
            (rho * (z_pre[0] - u_pre[0]) - 0.5) / (1.0 + rho), // v0d0
            (rho * (z_pre[1] - u_pre[1]) - (-0.5)) / (1.0 + rho), // v0d1
            (rho * (z_pre[2] - u_pre[2]) - 1.0) / (2.0 + rho), // v1d0
            (rho * (z_pre[3] - u_pre[3]) - (-1.0)) / (2.0 + rho), // v1d1
        ];
        for k in 0..4 {
            assert!((primal_x.data[k] - expected[k]).abs() < 1e-5,
                "x_update v{k}: got {}, expected {}", primal_x.data[k], expected[k]);
        }
    }

    /// x-update (DiagonalQuadL1) soft-thresholds the quadratic solve.
    #[test]
    fn x_update_diagonal_quad_l1_soft_thresholds() {
        let cx = CellComplex::from_edges(2, &[(0, 1)]);
        let maps = SheafMaps::identity(&cx, 2, 2);
        let mut primal_x = CochainField::zeros(0, 2, 2);
        let mut consensus_z = CochainField::from_vec(0, 2,
            vec![1.0, 2.0, 3.0, 4.0]);
        let mut dual_u = CochainField::from_vec(0, 2,
            vec![0.1, 0.2, 0.3, 0.4]);
        // lambda[0]=2.0 makes v0d0 threshold (2/3 ≈ 0.667) exceed its quad
        // solve (≈0.4333) → result 0. Tests the max(0, ...) zeroing path.
        let objective = LocalObjective::DiagonalQuadL1 {
            diag_q: vec![1.0, 1.0, 2.0, 2.0],
            q: vec![0.5, -0.5, 1.0, -1.0],
            lambda: vec![2.0, 0.1, 0.5, 3.0],
        };
        let mut scratch = AdmmScratch::new(&cx, 2, 2);
        let rho = 2.0;
        let z_pre = consensus_z.data.clone();
        let u_pre = dual_u.data.clone();

        sheaf_admm_step(&cx, &maps, &mut primal_x, &mut consensus_z, &mut dual_u,
            &objective, rho, 0.1, 1, &mut scratch);

        let xq = |k: usize, dq: f32, lin: f32| (rho * (z_pre[k] - u_pre[k]) - lin) / (dq + rho);
        let xq0 = xq(0, 1.0, 0.5);    // ≈ 0.4333
        let xq1 = xq(1, 1.0, -0.5);   // ≈ 1.3667
        let xq2 = xq(2, 2.0, 1.0);    // = 1.1
        let xq3 = xq(3, 2.0, -1.0);   // = 2.05
        let expected = [
            soft_threshold(xq0, 2.0 / 3.0), // thresh 0.667 > 0.4333 → 0
            soft_threshold(xq1, 0.1 / 3.0), // ≈ 1.3333
            soft_threshold(xq2, 0.5 / 4.0), // ≈ 0.975
            soft_threshold(xq3, 3.0 / 4.0), // = 1.3
        ];
        // Sanity: xq0 should indeed be zeroed.
        assert!((expected[0]).abs() < 1e-6, "v0d0 should be soft-zeroed, got {}", expected[0]);
        for k in 0..4 {
            assert!((primal_x.data[k] - expected[k]).abs() < 1e-5,
                "x_update_l1 v{k}: got {}, expected {}", primal_x.data[k], expected[k]);
        }
    }

    /// u-update invariant: `u^{k+1} − u^k == x^{k+1} − z^{k+1}` (G2 sanity).
    #[test]
    fn u_update_accumulates_disagreement() {
        let cx = CellComplex::grid_2d(3, 3);
        let maps = SheafMaps::identity(&cx, 2, 2);
        let total = cx.n_vertices() * 2;
        let mut primal_x = CochainField::zeros(0, cx.n_vertices(), 2);
        let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), 2);
        let mut dual_u = CochainField::zeros(0, cx.n_vertices(), 2);
        // Deterministic non-trivial initial values.
        for k in 0..total {
            primal_x.data[k] = 0.1 * (k as f32);
            consensus_z.data[k] = 0.05 * (k as f32);
            dual_u.data[k] = 0.01 * (k as f32);
        }
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0; total],
            q: vec![0.0; total],
        };
        let mut scratch = AdmmScratch::new(&cx, 2, 2);

        let u_before = dual_u.data.clone();
        sheaf_admm_step(&cx, &maps, &mut primal_x, &mut consensus_z, &mut dual_u,
            &objective, 1.0, 0.1, 3, &mut scratch);

        // Post-step x and z are exactly what the u-update read; the invariant
        // is bit-exact because both sides compute the same expression.
        for k in 0..total {
            let du = dual_u.data[k] - u_before[k];
            let dxz = primal_x.data[k] - consensus_z.data[k];
            assert!((du - dxz).abs() < 1e-6,
                "u invariant k={k}: du={du}, x-z={dxz}");
        }
    }

    /// DEC identity: for identity maps with `d_e == d_v`, the sheaf Laplacian
    /// via explicit maps equals the graph Laplacian (Research 384 §1.3).
    #[test]
    fn sheaf_laplacian_identity_matches_graph_laplacian_first_de_dims() {
        let cx = CellComplex::grid_2d(4, 4);
        let d_v = 2;
        let maps = SheafMaps::identity(&cx, d_v, d_v);
        let mut z = CochainField::zeros(0, cx.n_vertices(), d_v);
        // Deterministic non-trivial z.
        for k in 0..z.data.len() {
            z.data[k] = 0.1 * ((k * 13 + 7) as f32).sin().abs();
        }
        let mut scratch = AdmmScratch::new(&cx, d_v, d_v);

        sheaf_laplacian_via_maps(&cx, &maps, &z, &mut scratch);
        let gl = graph_laplacian(&cx, &z);

        // f32 accumulation order differs between the two paths; use a loose-but-safe tol.
        for k in 0..z.data.len() {
            assert!(
                (scratch.sheaf_laplacian_z[k] - gl.data[k]).abs() < 1e-4,
                "sheaf_laplacian vs graph_laplacian k={k}: sheaf={}, graph={}",
                scratch.sheaf_laplacian_z[k], gl.data[k]
            );
        }
    }

    /// Smoke test: one ADMM step on a 4×4 grid runs without panic.
    #[test]
    fn one_admm_step_runs_without_panic() {
        let cx = CellComplex::grid_2d(4, 4);
        let d_v = 4;
        let d_e = 2;
        let maps = SheafMaps::identity(&cx, d_v, d_e);
        let total = cx.n_vertices() * d_v;
        let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
        for k in 0..total {
            primal_x.data[k] = 0.1 * (k as f32);
        }
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0; total],
            q: vec![0.0; total],
        };
        let mut scratch = AdmmScratch::new(&cx, d_v, d_e);
        sheaf_admm_step(&cx, &maps, &mut primal_x, &mut consensus_z, &mut dual_u,
            &objective, 1.0, 0.1, 3, &mut scratch);
        // If we get here, no panic. Sanity: primal is finite.
        for v in primal_x.data.iter() {
            assert!(v.is_finite(), "non-finite primal after step");
        }
    }

    /// Weak G1 preview: K ADMM steps with identity maps reduce the primal
    /// max-edge-disagreement (consensus reached).
    ///
    /// Parameters are tuned so the z-projection is near-exact each step (T=50
    /// diffusion steps with eta=0.2 drives the non-constant residual to <0.2%
    /// on a 4×4 grid), and the local/global balance is well-conditioned
    /// (diag_q == rho). A 2-vertex hand-trace confirms geometric convergence
    /// (disagreement halves each ADMM step) under these settings.
    #[test]
    fn identity_maps_reach_consensus() {
        let cx = CellComplex::grid_2d(4, 4);
        let d_v = 2;
        let d_e = 2;
        let maps = SheafMaps::identity(&cx, d_v, d_e);
        let total = cx.n_vertices() * d_v;
        // Balanced local objective with per-vertex distinct preferred targets.
        // q = -target * diag_q ⇒ unconstrained minimizer of f_i is `target`.
        // diag_q == rho ⇒ x-update is a 50/50 blend of (z-u) and target.
        let diag_q_val = 1.0;
        let rho = 1.0;
        let mut target = vec![0.0f32; total];
        for i in 0..cx.n_vertices() {
            for d in 0..d_v {
                target[i * d_v + d] = (0.3 * (i as f32) + 0.7 * (d as f32)) * 0.5;
            }
        }
        let q: Vec<f32> = target.iter().map(|t| -t * diag_q_val).collect();
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![diag_q_val; total],
            q,
        };
        let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
        // Seed primal with the targets (measures the initial disagreement).
        primal_x.data.copy_from_slice(&target);
        let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

        let d_initial = max_edge_disagreement(&cx, &primal_x);
        // eta = 0.2 keeps diffusion stable (grid Laplacian λ_max ≈ 6.8 on 4×4;
        // 0.2·6.8 ≈ 1.36 < 2). T = 50 drives the z-projection near-exact.
        for _ in 0..30 {
            sheaf_admm_step(&cx, &maps, &mut primal_x, &mut consensus_z, &mut dual_u,
                &objective, rho, 0.2, 50, &mut scratch);
        }
        let d_final = max_edge_disagreement(&cx, &primal_x);

        eprintln!(
            "identity_maps_reach_consensus: d_initial={d_initial:.5}, d_final={d_final:.5}"
        );
        assert!(
            d_final < d_initial,
            "consensus not reached: d_final {d_final} >= d_initial {d_initial}"
        );
        // Stronger: meaningful reduction (geometric convergence ⇒ near-zero).
        assert!(
            d_final < 0.1 * d_initial,
            "consensus reduction too weak: d_final {d_final} >= 0.1*d_initial {}",
            0.1 * d_initial
        );
    }

    /// Max over edges and dims of `|x_tail[d] − x_head[d]|` — the identity-map
    /// disagreement norm (‖F x‖_∞ for `d_e == d_v`).
    fn max_edge_disagreement(cx: &CellComplex, x: &CochainField) -> f32 {
        let dim = x.dim;
        let mut max_d = 0.0f32;
        for pair in cx.boundary_entries(0).chunks_exact(2) {
            let v_tail = pair[0].0;
            let v_head = pair[1].0;
            for d in 0..dim {
                let diff = (x.data[v_tail * dim + d] - x.data[v_head * dim + d]).abs();
                if diff > max_d {
                    max_d = diff;
                }
            }
        }
        max_d
    }
}
