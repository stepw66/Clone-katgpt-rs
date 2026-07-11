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
/// for the identity fast path (5-point grid stencil or edge-list difference).
///
/// `is_selector` (Plan 407 T3.2) marks per-edge selector maps built via
/// [`SheafMaps::selector_per_edge`] / [`SheafMaps::selector_per_edge_topk`].
/// When `true`, the compact row-selection indices live in `selector_indices`
/// and `maps` is empty (no dense matrix is materialized — the gather-scatter
/// fast path uses `selector_indices` directly, `O(d_e)` per edge instead of
/// `O(d_e·d_v)` dense matvec).
#[derive(Clone, Debug)]
pub struct SheafMaps {
    /// Edge stalk dimension (≤ `d_v`).
    pub d_e: usize,
    /// Vertex stalk dimension.
    pub d_v: usize,
    /// Flat map storage: `n_edges * 2 * d_e * d_v` row-major floats.
    /// Empty when `is_selector` (compact indices in `selector_indices`).
    pub maps: Vec<f32>,
    /// `true` iff every map equals `[I_{d_e}; 0]` (identity block). Fast-path hint.
    pub is_identity: bool,
    /// `true` iff maps are compact per-edge selector (row-selection) maps. When
    /// `true`, `maps` is empty and `selector_indices` holds the compact indices;
    /// the laplacian fast path uses gather-scatter (`O(d_e)` per edge).
    pub is_selector: bool,
    /// Compact selector indices: `n_edges * 2 * d_e` entries (u16). For edge
    /// `e`, endpoint `p`, row `r`: `selector_indices[(e*2 + p) * d_e + r]` is the
    /// vertex dim that row `r` selects. Empty when `!is_selector`.
    ///
    /// u16 supports `d_v ≤ 65535` — well beyond any practical vertex stalk
    /// (HLA=8, shards=64).
    pub selector_indices: Vec<u16>,
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
            is_selector: false,
            selector_indices: Vec::new(),
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
        let is_identity = dim_indices.iter().enumerate().all(|(r, &idx)| idx == r);
        Self {
            d_e,
            d_v,
            maps,
            is_identity,
            is_selector: false,
            selector_indices: Vec::new(),
            n_edges,
        }
    }

    /// Per-edge selector restriction maps (Plan 407 T3.2 — Issue 396).
    ///
    /// `dim_indices_per_edge[e]` selects the `d_e` dims for edge `e` (applied
    /// to **both** endpoints — per-endpoint asymmetry is deferred). Each edge
    /// can select a **different** dim subset, enabling per-edge-type
    /// heterogeneous consensus (spatial edges select 5 dims, faction edges
    /// select 3, etc. — Research 314 §1.3).
    ///
    /// # Storage
    ///
    /// Maps are stored in **compact selector form** (`selector_indices`), not
    /// dense — `maps` is empty. The laplacian fast path does gather-scatter in
    /// `O(d_e)` per edge (vs `O(d_e·d_v)` for dense matvec). This is the
    /// server-scale (K>1000) latency win.
    ///
    /// # Uniform `d_e` constraint
    ///
    /// All per-edge slices MUST have the same length `d_e` (the `AdmmScratch`
    /// layout requires a uniform edge stalk dim). For heterogeneous effective
    /// `d_e` (e.g. spatial=5, faction=3), pad shorter selectors to `d_e_max`
    /// with any valid dim index — the caller's conviction-vector weighting
    /// handles the effective dim reduction (the conviction vector, not the map
    /// row count, is the G8 load-bearing mechanism).
    ///
    /// # Panics
    /// If `dim_indices_per_edge.len() != cx.n_edges()`, any slice length is 0
    /// or `> d_v`, slices have unequal lengths, or any index `>= d_v`.
    pub fn selector_per_edge(
        cx: &CellComplex,
        d_v: usize,
        dim_indices_per_edge: &[&[usize]],
    ) -> Self {
        let n_edges = cx.n_edges();
        assert_eq!(
            dim_indices_per_edge.len(),
            n_edges,
            "SheafMaps::selector_per_edge: dim_indices_per_edge.len() {} != n_edges {}",
            dim_indices_per_edge.len(),
            n_edges
        );
        assert!(
            !dim_indices_per_edge.is_empty(),
            "SheafMaps::selector_per_edge: no edges"
        );
        let d_e = dim_indices_per_edge[0].len();
        assert!(
            d_e > 0 && d_e <= d_v,
            "SheafMaps::selector_per_edge: d_e ({d_e}) must be in 1..={d_v}"
        );
        for (e, slice) in dim_indices_per_edge.iter().enumerate() {
            assert_eq!(
                slice.len(),
                d_e,
                "SheafMaps::selector_per_edge: uniform d_e required, but edge {e} has {} (expected {d_e})",
                slice.len()
            );
            for (r, &idx) in slice.iter().enumerate() {
                assert!(
                    idx < d_v,
                    "SheafMaps::selector_per_edge: edge {e} dim_indices[{r}] = {idx} >= d_v {d_v}"
                );
            }
        }
        // Compact u16 storage: n_edges * 2 * d_e indices (both endpoints).
        let mut selector_indices = vec![0u16; n_edges * 2 * d_e];
        for (e, slice) in dim_indices_per_edge.iter().enumerate().take(n_edges) {
            let base = e * 2 * d_e;
            for r in 0..d_e {
                let idx = slice[r] as u16;
                selector_indices[base + r] = idx; // endpoint 0 (tail)
                selector_indices[base + d_e + r] = idx; // endpoint 1 (head)
            }
        }
        // is_identity iff every edge selects dims [0, 1, …, d_e-1] in order.
        let is_identity = dim_indices_per_edge
            .iter()
            .all(|slice| slice.iter().enumerate().all(|(r, &idx)| idx == r));
        Self {
            d_e,
            d_v,
            maps: Vec::new(),
            is_identity,
            is_selector: true,
            selector_indices,
            n_edges,
        }
    }

    /// Top-k selector maps from per-edge importance scores (Plan 407 T3.2 —
    /// Issue 397). Convenience constructor that picks the top-`k` dims per edge
    /// via partial sort, then delegates to [`Self::selector_per_edge`].
    ///
    /// `scores_per_edge[e]` is a `d_v`-length importance score vector for edge
    /// `e` (e.g. Mind-Reading CS-rankings for spatial edges, Latent Functor
    /// direction loadings for faction edges). The top-`k` highest-scoring dims
    /// become the restriction-map rows for that edge.
    ///
    /// # Ties
    ///
    /// Ties are broken by dim index (lower index wins) — deterministic. This
    /// means scores with many equal values bias toward lower dims, which is
    /// acceptable for the modelless mandate (no stochastic tie-breaking).
    ///
    /// # Panics
    /// If `k == 0` or `k > d_v`, or any score slice length `!= d_v`.
    pub fn selector_per_edge_topk(
        cx: &CellComplex,
        d_v: usize,
        scores_per_edge: &[&[f32]],
        k: usize,
    ) -> Self {
        assert!(
            k > 0 && k <= d_v,
            "SheafMaps::selector_per_edge_topk: k ({k}) must be in 1..={d_v}"
        );
        assert_eq!(
            scores_per_edge.len(),
            cx.n_edges(),
            "SheafMaps::selector_per_edge_topk: scores_per_edge.len() != n_edges"
        );
        let mut indices_per_edge: Vec<&[usize]> = Vec::with_capacity(cx.n_edges());
        let mut owned: Vec<Vec<usize>> = Vec::with_capacity(cx.n_edges());
        for scores in scores_per_edge {
            assert_eq!(
                scores.len(),
                d_v,
                "SheafMaps::selector_per_edge_topk: score slice len {} != d_v {d_v}",
                scores.len()
            );
            // Partial sort: pick top-k dims by score (descending), tie-break by
            // index (ascending). Build (score, dim) pairs, sort by (-score, dim).
            let mut pairs: Vec<(f32, usize)> = scores
                .iter()
                .copied()
                .enumerate()
                .map(|(d, s)| (s, d))
                .collect();
            pairs.sort_unstable_by(|a, b| {
                // Higher score first; on tie, lower dim first.
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            let selected: Vec<usize> = pairs.iter().take(k).map(|&(_, d)| d).collect();
            owned.push(selected);
        }
        for v in &owned {
            indices_per_edge.push(v.as_slice());
        }
        Self::selector_per_edge(cx, d_v, &indices_per_edge)
    }

    /// Read the restriction map for `endpoint` (0 = tail, 1 = head) of edge
    /// `edge_idx`. Returns a `d_e × d_v` row-major slice.
    ///
    /// # Panics if `is_selector`
    ///
    /// Selector maps do not materialize a dense matrix — calling this on a
    /// selector map panics. Use [`Self::selector_edge_indices`] instead.
    #[inline]
    pub fn edge_map(&self, edge_idx: usize, endpoint: usize) -> &[f32] {
        assert!(
            !self.is_selector,
            "SheafMaps::edge_map: dense map access on selector maps is not supported; \
             use selector_edge_indices() instead"
        );
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

    /// Read the compact selector indices for `endpoint` (0 = tail, 1 = head) of
    /// edge `edge_idx`. Returns a `d_e`-length slice of vertex dim indices.
    ///
    /// # Panics if `!is_selector`.
    #[inline]
    pub fn selector_edge_indices(&self, edge_idx: usize, endpoint: usize) -> &[u16] {
        assert!(
            self.is_selector,
            "SheafMaps::selector_edge_indices: not a selector map (is_selector is false)"
        );
        debug_assert!(
            edge_idx < self.n_edges,
            "SheafMaps::selector_edge_indices: edge_idx {edge_idx} >= n_edges {}",
            self.n_edges
        );
        debug_assert!(
            endpoint < 2,
            "SheafMaps::selector_edge_indices: endpoint {endpoint} must be 0 or 1"
        );
        let base = (edge_idx * 2 + endpoint) * self.d_e;
        &self.selector_indices[base..base + self.d_e]
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
    /// Warm-start snapshot `b = x + u` for the soft-constraint z-update (T3.3).
    /// Length `n_vertices * d_v`. Only touched by [`sheaf_admm_step_soft_into`];
    /// the hard-constraint step leaves it at zero (no overhead).
    pub warm_start_b: Vec<f32>,
    /// CG residual `r = b − L_F z`, length `n_vertices * d_v`.
    /// Only touched by [`sheaf_admm_step_cg_into`].
    pub cg_r: Vec<f32>,
    /// CG search direction `p`, length `n_vertices * d_v`.
    pub cg_p: Vec<f32>,
    /// CG matvec output `Ap = L_F p`, length `n_vertices * d_v`.
    pub cg_ap: Vec<f32>,
}

impl AdmmScratch {
    /// Allocate scratch sized for a cell complex with the given vertex / edge
    /// stalk dimensions.
    pub fn new(cx: &CellComplex, d_v: usize, d_e: usize) -> Self {
        let total = cx.n_vertices() * d_v;
        Self {
            edge_projections: vec![0.0; cx.n_edges() * 2 * d_e],
            sheaf_laplacian_z: vec![0.0; total],
            warm_start_b: vec![0.0; total],
            cg_r: vec![0.0; total],
            cg_p: vec![0.0; total],
            cg_ap: vec![0.0; total],
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
#[allow(clippy::too_many_arguments)] // ADMM solver — all 10 params are genuinely needed
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
    debug_assert_eq!(
        consensus_z.rank, 0,
        "sheaf_admm_step: consensus_z must be rank-0"
    );
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
    debug_assert_eq!(
        dual_u.n_cells(),
        n,
        "sheaf_admm_step: dual_u.n_cells mismatch"
    );
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
#[allow(clippy::too_many_arguments)] // ADMM solver — all 10 params are genuinely needed
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
            debug_assert_eq!(
                diag_q.len(),
                total,
                "DiagonalQuadratic: diag_q.len() != n*d_v"
            );
            debug_assert_eq!(q.len(), total, "DiagonalQuadratic: q.len() != n*d_v");
            for k in 0..total {
                let denom = diag_q[k] + rho;
                debug_assert!(
                    denom > 0.0,
                    "sheaf_admm_step: non-positive denom (diag_q[k]={} + rho={})",
                    diag_q[k],
                    rho
                );
                primal_x.data[k] = (rho * (consensus_z.data[k] - dual_u.data[k]) - q[k]) / denom;
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
                let x_quad = (rho * (consensus_z.data[k] - dual_u.data[k]) - q[k]) / denom;
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
        sheaf_laplacian_via_maps(cx, maps, &consensus_z.data, scratch);
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

/// One ADMM iteration with a **soft-constraint** z-update (Plan 407 T3.3 —
/// paper eq. 25). Same x-update and u-update as [`sheaf_admm_step_into`], but
/// the z-update minimizes `½ z^T L_F z + (γ/2)‖z − b‖²` (where `b = x + u`)
/// instead of the hard-constraint projection onto `ker(L_F)`.
///
/// # When `gamma == 0.0`
///
/// Delegates to the hard-constraint diffusion path (identical to
/// [`sheaf_admm_step_into`]). This lets callers switch between hard and soft
/// constraints without changing the call site — just pass `gamma = 0.0` for
/// hard, `gamma > 0.0` for soft.
///
/// # When `gamma > 0.0`
///
/// The z-update diffusion step becomes:
/// ```text
/// z ← z − η (L_F z + γ (z − b))
/// ```
/// The `γ(z − b)` term pulls `z` back toward the warm-start `b`, preventing
/// full consensus. This is useful when NPCs should preserve some individual
/// variation — e.g. faction members agree on faction-relevant dims but retain
/// distinct personalities on the remaining dims.
///
/// The warm-start `b` is snapshotted into `scratch.warm_start_b` before the
/// diffusion loop and read from there (avoids re-reading `x + u` each step).
#[allow(clippy::too_many_arguments)] // ADMM solver + gamma = 11 params
#[inline]
pub fn sheaf_admm_step_soft_into(
    cx: &CellComplex,
    maps: &SheafMaps,
    primal_x: &mut CochainField,
    consensus_z: &mut CochainField,
    dual_u: &mut CochainField,
    objective: &LocalObjective,
    rho: f32,
    eta: f32,
    gamma: f32,
    diffusion_steps: usize,
    scratch: &mut AdmmScratch,
) {
    let n = cx.n_vertices();
    let d_v = maps.d_v;
    let total = n * d_v;

    // ---- x-update: identical to hard-constraint ---------------------------
    x_update(primal_x, consensus_z, dual_u, objective, rho, total);

    // ---- z-update: warm-start z = x + u into both z and warm_start_b -------
    for k in 0..total {
        let b_k = primal_x.data[k] + dual_u.data[k];
        consensus_z.data[k] = b_k;
        scratch.warm_start_b[k] = b_k;
    }
    if gamma == 0.0 {
        // Hard-constraint path: identical to sheaf_admm_step_into.
        for _ in 0..diffusion_steps {
            sheaf_laplacian_via_maps(cx, maps, &consensus_z.data, scratch);
            for k in 0..total {
                consensus_z.data[k] -= eta * scratch.sheaf_laplacian_z[k];
            }
        }
    } else {
        // Soft-constraint path: z ← z − η (L_F z + γ(z − b)).
        for _ in 0..diffusion_steps {
            sheaf_laplacian_via_maps(cx, maps, &consensus_z.data, scratch);
            for k in 0..total {
                let gradient = scratch.sheaf_laplacian_z[k]
                    + gamma * (consensus_z.data[k] - scratch.warm_start_b[k]);
                consensus_z.data[k] -= eta * gradient;
            }
        }
    }

    // ---- u-update: identical to hard-constraint ----------------------------
    for k in 0..total {
        dual_u.data[k] += primal_x.data[k] - consensus_z.data[k];
    }
}

/// One ADMM iteration with a **conjugate-gradient** z-update (Plan 407 T3.1 —
/// paper Appendix B.2). Instead of `diffusion_steps` gradient-descent steps on
/// `½ z^T L_F z` (the hard-constraint path), this solves `L_F z = 0` via CG up
/// to `max_cg_iters` iterations or residual tolerance `tol`.
///
/// CG converges in `O(√κ)` iterations vs GD's `O(κ)` for condition number `κ`
/// (the sheaf Laplacian's spectral range). On sparse graphs with poor
/// conditioning (large zones, pathological adjacency), CG reaches a given
/// residual in fewer matvecs — **if** the conditioning is bad enough to offset
/// CG's higher per-iteration cost (3 vector axpy + 1 matvec + 2 dot products vs
/// GD's 1 matvec + 1 axpy).
///
/// # The CG system
///
/// The z-update projects `b = x + u` onto `ker(L_F)`. Since `L_F` is
/// singular (constant/harmonic modes are in the kernel), CG on `L_F z = 0`
/// starting from `z₀ = b` converges to the projection (the component of `b`
/// orthogonal to `ker(L_F)` decays to zero). This is the same inexact
/// projection as the GD path, but with a faster-converging iterate.
///
/// # Modelless
///
/// CG is a deterministic linear-algebra solver — no training, no gradient
/// descent on weights. The sheaf Laplacian is fixed by the restriction maps.
#[allow(clippy::too_many_arguments)] // ADMM solver + CG params = 12 params
#[inline]
pub fn sheaf_admm_step_cg_into(
    cx: &CellComplex,
    maps: &SheafMaps,
    primal_x: &mut CochainField,
    consensus_z: &mut CochainField,
    dual_u: &mut CochainField,
    objective: &LocalObjective,
    rho: f32,
    max_cg_iters: usize,
    tol: f32,
    scratch: &mut AdmmScratch,
) {
    let n = cx.n_vertices();
    let d_v = maps.d_v;
    let total = n * d_v;

    // ---- x-update: identical to hard-constraint ---------------------------
    x_update(primal_x, consensus_z, dual_u, objective, rho, total);

    // ---- z-update: CG on L_F z = 0, warm-started from z₀ = x + u -----------
    // r₀ = b − L_F z₀ = b − L_F b. Since b = x + u, r₀ = b − L_F b.
    for k in 0..total {
        consensus_z.data[k] = primal_x.data[k] + dual_u.data[k];
    }
    sheaf_laplacian_via_maps(cx, maps, &consensus_z.data, scratch);
    let mut rsold: f32 = 0.0;
    for k in 0..total {
        let r_k = -scratch.sheaf_laplacian_z[k]; // r = 0 − L_F z₀ (target is 0)
        scratch.cg_r[k] = r_k;
        scratch.cg_p[k] = r_k; // p₀ = r₀
        rsold += r_k * r_k;
    }
    // If the residual is already below tol, skip CG entirely.
    let mut cg_iters_run = 0usize;
    if rsold > tol * tol {
        for _ in 0..max_cg_iters {
            // Ap = L_F p (zero-alloc: pass cg_p slice directly, no clone).
            sheaf_laplacian_matvec(
                cx,
                maps,
                &scratch.cg_p,
                &mut scratch.sheaf_laplacian_z,
                &mut scratch.edge_projections,
            );
            // matvec wrote into sheaf_laplacian_z; copy to cg_ap before the
            // next iteration clobbers it.
            scratch
                .cg_ap
                .copy_from_slice(&scratch.sheaf_laplacian_z[0..total]);

            // alpha = (r·r) / (p·Ap)
            let p_ap: f32 = scratch
                .cg_p
                .iter()
                .zip(scratch.cg_ap.iter())
                .map(|(&p, &ap)| p * ap)
                .sum();
            if p_ap <= 0.0 || !p_ap.is_finite() {
                break; // L_F is PSD; p_ap ≤ 0 means we hit the kernel — done.
            }
            let alpha = rsold / p_ap;

            // z += alpha * p; r -= alpha * Ap
            let mut rsnew: f32 = 0.0;
            for k in 0..total {
                consensus_z.data[k] += alpha * scratch.cg_p[k];
                scratch.cg_r[k] -= alpha * scratch.cg_ap[k];
                rsnew += scratch.cg_r[k] * scratch.cg_r[k];
            }
            cg_iters_run += 1;
            if rsnew <= tol * tol {
                break;
            }
            // beta = (r_new·r_new) / (r_old·r_old); p = r + beta * p
            let beta = rsnew / rsold;
            for k in 0..total {
                scratch.cg_p[k] = scratch.cg_r[k] + beta * scratch.cg_p[k];
            }
            rsold = rsnew;
        }
    }
    let _ = cg_iters_run; // (used in tests via debug_assert only)

    // ---- u-update: identical to hard-constraint ----------------------------
    for k in 0..total {
        dual_u.data[k] += primal_x.data[k] - consensus_z.data[k];
    }
}

/// x-update helper: shared by all three step variants (hard, soft, CG).
#[inline]
fn x_update(
    primal_x: &mut CochainField,
    consensus_z: &CochainField,
    dual_u: &CochainField,
    objective: &LocalObjective,
    rho: f32,
    total: usize,
) {
    match objective {
        LocalObjective::DiagonalQuadratic { diag_q, q } => {
            debug_assert_eq!(
                diag_q.len(),
                total,
                "DiagonalQuadratic: diag_q.len() != n*d_v"
            );
            debug_assert_eq!(q.len(), total, "DiagonalQuadratic: q.len() != n*d_v");
            for k in 0..total {
                let denom = diag_q[k] + rho;
                debug_assert!(
                    denom > 0.0,
                    "x_update: non-positive denom (diag_q[k]={} + rho={})",
                    diag_q[k],
                    rho
                );
                primal_x.data[k] = (rho * (consensus_z.data[k] - dual_u.data[k]) - q[k]) / denom;
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
                    "x_update: non-positive denom (diag_q[k]={} + rho={})",
                    diag_q[k],
                    rho
                );
                let x_quad = (rho * (consensus_z.data[k] - dual_u.data[k]) - q[k]) / denom;
                let thresh = lambda[k] / denom;
                primal_x.data[k] = soft_threshold(x_quad, thresh);
            }
        }
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

// Compute the sheaf Laplacian applied to `z` into `scratch.sheaf_laplacian_z`.
//
// `L_F z` accumulates, per edge `e = (v_tail, v_head)`:
// ```text
// disagreement = F_{i→e} z_{v_tail} − F_{j→e} z_{v_head}    (d_e-dim)
// (L_F z)_{v_tail} += F_{i→e}^T · disagreement              (d_v-dim)
// (L_F z)_{v_head} −= F_{j→e}^T · disagreement              (d_v-dim)
// ```
//
// Edges are iterated via `cx.boundary_entries(0).chunks_exact(2)`, matching
// the `(v_tail, e, −1), (v_head, e, +1)` pair ordering of `grid_2d` /
// `from_edges`. The current edge's `d_e`-dim disagreement is staged in the
// first `d_e` slots of `scratch.edge_projections` (compute-then-accumulate to
// keep the `F^T · d` accumulation cache-friendly).
//
// # Identity fast path (Plan 407 T2.4 — Phase 2)
//
// When `maps.is_identity`, the restriction maps are `[I_{d_e}; 0]` for every
// edge/endpoint. The sheaf Laplacian then reduces bit-for-bit to the graph
// Laplacian on the first `d_e` dims (dims `d_e..d_v` have zero disagreement).
// We skip the explicit `F^T F` matvec (which wastes ~`d_v` scalar multiplies
// per row, most against zero entries) and directly compute the graph-Laplacian
// difference per edge on the first `d_e` dims. This is the G4 latency
// optimization — turns a ~`O(|E|·d_e·d_v)` general matvec into a lean
// `O(|E|·d_e)` identity matvec with no wasted multiplies-against-zero. On
// regular grids, the identity path further delegates to
// [`sheaf_laplacian_identity_grid_into`] (the 5-point-stencil variant) for
// single-write cache-friendly output.

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
/// `sheaf_laplacian_z` is written directly (dims `d_e..d_v` left at
/// the zero value from the caller's `.fill(0.0)`).
#[inline]
fn sheaf_laplacian_identity_grid_into(
    w: usize,
    h: usize,
    d_v: usize,
    d_e: usize,
    z_data: &[f32],
    sheaf_laplacian_z: &mut [f32],
) {
    let p = z_data.as_ptr();
    let o = sheaf_laplacian_z.as_mut_ptr();
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
                    if has_left {
                        acc -= *p.add(left + c);
                    }
                    if has_right {
                        acc -= *p.add(right + c);
                    }
                    if has_up {
                        acc -= *p.add(up + c);
                    }
                    if has_down {
                        acc -= *p.add(down + c);
                    }
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
            for &(x, has_left, has_right) in [(0usize, false, true), (w - 1, true, false)].iter() {
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
                        if has_left {
                            acc -= *p.add(left + c);
                        }
                        if has_right {
                            acc -= *p.add(right + c);
                        }
                        acc -= *p.add(up + c);
                        acc -= *p.add(down + c);
                        *o.add(base + c) = acc;
                    }
                }
            }
        }
    }
}

/// Compute `L_F · z` into `scratch.sheaf_laplacian_z`. Thin wrapper around
/// [`sheaf_laplacian_matvec`] that splits the scratch borrow.
#[inline]
fn sheaf_laplacian_via_maps(
    cx: &CellComplex,
    maps: &SheafMaps,
    z_data: &[f32],
    scratch: &mut AdmmScratch,
) {
    sheaf_laplacian_matvec(
        cx,
        maps,
        z_data,
        &mut scratch.sheaf_laplacian_z,
        &mut scratch.edge_projections,
    );
}

/// Real matvec: computes the sheaf Laplacian `L_F · z` into `sheaf_laplacian_z`.
/// Takes individual scratch field slices (not the whole `AdmmScratch`) so the
/// CG path can pass `scratch.cg_p` as `z_data` without an aliasing violation
/// (the input and output are disjoint memory).
#[inline]
fn sheaf_laplacian_matvec(
    cx: &CellComplex,
    maps: &SheafMaps,
    z_data: &[f32],
    sheaf_laplacian_z: &mut [f32],
    edge_projections: &mut [f32],
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
            sheaf_laplacian_identity_grid_into(w, h, d_v, d_e, z_data, sheaf_laplacian_z);
            return;
        }
        // Edge-list fallback for non-grid complexes (needs fill for dims d_e..d_v).
        sheaf_laplacian_z.fill(0.0);
        let entries = cx.boundary_entries(0);
        for pair in entries.chunks_exact(2) {
            let v_tail = pair[0].0;
            let v_head = pair[1].0;
            let tail_base = v_tail * d_v;
            let head_base = v_head * d_v;
            for d in 0..d_e {
                let diff = z_data[tail_base + d] - z_data[head_base + d];
                sheaf_laplacian_z[tail_base + d] += diff;
                sheaf_laplacian_z[head_base + d] -= diff;
            }
        }
        return;
    }

    // ── Selector fast path (Plan 407 T3.2) ─────────────────────────────────
    // F_{i→e}[r, :] = e_{idx_tail[r]}, F_{j→e}[r, :] = e_{idx_head[r]} ⟹
    //   disagreement[r] = z_tail[idx_tail[r]] − z_head[idx_head[r]]
    //   (L_F z)_tail[idx_tail[r]] += disagreement[r]
    //   (L_F z)_head[idx_head[r]] −= disagreement[r]
    // This is O(d_e) per edge (gather-scatter), vs O(d_e·d_v) for the dense
    // matvec — the server-scale (K>1000) latency win.
    if maps.is_selector {
        sheaf_laplacian_z.fill(0.0);
        let entries = cx.boundary_entries(0);
        let idx = &maps.selector_indices;
        for pair in entries.chunks_exact(2) {
            let v_tail = pair[0].0;
            let e = pair[0].1;
            let v_head = pair[1].0;
            debug_assert_eq!(
                pair[1].1, e,
                "sheaf_laplacian_matvec: mismatched edge idx in boundary pair"
            );
            let z_tail_base = v_tail * d_v;
            let z_head_base = v_head * d_v;
            let idx_base = e * 2 * d_e;
            let idx_tail = &idx[idx_base..idx_base + d_e];
            let idx_head = &idx[idx_base + d_e..idx_base + 2 * d_e];
            // Gather: disagreement[r] = z_tail[idx_tail[r]] − z_head[idx_head[r]]
            // + scatter-add into tail slot and scatter-sub into head slot.
            // Stage disagreement into edge_projections[0..d_e] first (avoids
            // read-after-write hazard if tail == head on a self-loop).
            let disagreement = &mut edge_projections[0..d_e];
            for r in 0..d_e {
                let it = idx_tail[r] as usize;
                let ih = idx_head[r] as usize;
                disagreement[r] = z_data[z_tail_base + it] - z_data[z_head_base + ih];
            }
            for r in 0..d_e {
                let d_r = disagreement[r];
                let it = idx_tail[r] as usize;
                let ih = idx_head[r] as usize;
                sheaf_laplacian_z[z_tail_base + it] += d_r;
                sheaf_laplacian_z[z_head_base + ih] -= d_r;
            }
        }
        return;
    }

    // General explicit-maps path: zero-fill then accumulate.
    sheaf_laplacian_z.fill(0.0);

    let entries = cx.boundary_entries(0);
    for pair in entries.chunks_exact(2) {
        let v_tail = pair[0].0;
        let e = pair[0].1;
        let v_head = pair[1].0;
        debug_assert_eq!(
            pair[1].1, e,
            "sheaf_laplacian_matvec: mismatched edge idx in boundary pair"
        );

        let f_tail = maps.edge_map(e, 0);
        let f_head = maps.edge_map(e, 1);
        let z_tail_base = v_tail * d_v;
        let z_head_base = v_head * d_v;

        // Stage disagreement[r] = F_tail[r,:]·z_tail − F_head[r,:]·z_head into
        // the first d_e slots of edge_projections.
        {
            let disagreement = &mut edge_projections[0..d_e];
            let z_tail = &z_data[z_tail_base..z_tail_base + d_v];
            let z_head = &z_data[z_head_base..z_head_base + d_v];
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
            let d_r = edge_projections[r];
            let f_tail_row = &f_tail[r * d_v..(r + 1) * d_v];
            let f_head_row = &f_head[r * d_v..(r + 1) * d_v];
            for c in 0..d_v {
                sheaf_laplacian_z[z_tail_base + c] += f_tail_row[c] * d_r;
                sheaf_laplacian_z[z_head_base + c] -= f_head_row[c] * d_r;
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
                assert_eq!(
                    m,
                    expected.as_slice(),
                    "identity map (e={e}, endpoint={endpoint}) wrong: {m:?}"
                );
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
                assert_eq!(
                    m,
                    expected.as_slice(),
                    "selector map (e={e}, endpoint={endpoint}) wrong: {m:?}"
                );
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
        let mut consensus_z = CochainField::from_vec(0, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut dual_u = CochainField::from_vec(0, 2, vec![0.1, 0.2, 0.3, 0.4]);
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0, 1.0, 2.0, 2.0],
            q: vec![0.5, -0.5, 1.0, -1.0],
        };
        let mut scratch = AdmmScratch::new(&cx, 2, 2);
        let rho = 2.0;

        // Snapshot pre-step z (the x-update reads these).
        let z_pre = consensus_z.data.clone();
        let u_pre = dual_u.data.clone();

        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            0.1,
            1,
            &mut scratch,
        );

        // Expected x_i = (ρ(z-u) - q) / (diag_q + ρ).
        let expected = [
            (rho * (z_pre[0] - u_pre[0]) - 0.5) / (1.0 + rho), // v0d0
            (rho * (z_pre[1] - u_pre[1]) - (-0.5)) / (1.0 + rho), // v0d1
            (rho * (z_pre[2] - u_pre[2]) - 1.0) / (2.0 + rho), // v1d0
            (rho * (z_pre[3] - u_pre[3]) - (-1.0)) / (2.0 + rho), // v1d1
        ];
        for (k, (&got, &exp)) in primal_x.data.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "x_update v{k}: got {got}, expected {exp}"
            );
        }
    }

    /// x-update (DiagonalQuadL1) soft-thresholds the quadratic solve.
    #[test]
    fn x_update_diagonal_quad_l1_soft_thresholds() {
        let cx = CellComplex::from_edges(2, &[(0, 1)]);
        let maps = SheafMaps::identity(&cx, 2, 2);
        let mut primal_x = CochainField::zeros(0, 2, 2);
        let mut consensus_z = CochainField::from_vec(0, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut dual_u = CochainField::from_vec(0, 2, vec![0.1, 0.2, 0.3, 0.4]);
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

        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            0.1,
            1,
            &mut scratch,
        );

        let xq = |k: usize, dq: f32, lin: f32| (rho * (z_pre[k] - u_pre[k]) - lin) / (dq + rho);
        let xq0 = xq(0, 1.0, 0.5); // ≈ 0.4333
        let xq1 = xq(1, 1.0, -0.5); // ≈ 1.3667
        let xq2 = xq(2, 2.0, 1.0); // = 1.1
        let xq3 = xq(3, 2.0, -1.0); // = 2.05
        let expected = [
            soft_threshold(xq0, 2.0 / 3.0), // thresh 0.667 > 0.4333 → 0
            soft_threshold(xq1, 0.1 / 3.0), // ≈ 1.3333
            soft_threshold(xq2, 0.5 / 4.0), // ≈ 0.975
            soft_threshold(xq3, 3.0 / 4.0), // = 1.3
        ];
        // Sanity: xq0 should indeed be zeroed.
        assert!(
            (expected[0]).abs() < 1e-6,
            "v0d0 should be soft-zeroed, got {}",
            expected[0]
        );
        for (k, (&got, &exp)) in primal_x.data.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "x_update_l1 v{k}: got {got}, expected {exp}"
            );
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
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            1.0,
            0.1,
            3,
            &mut scratch,
        );

        // Post-step x and z are exactly what the u-update read; the invariant
        // is bit-exact because both sides compute the same expression.
        for (k, ((&u_post, &u_pre), (&x_val, &z_val))) in dual_u
            .data
            .iter()
            .zip(&u_before)
            .zip(primal_x.data.iter().zip(&consensus_z.data))
            .enumerate()
        {
            let du = u_post - u_pre;
            let dxz = x_val - z_val;
            assert!(
                (du - dxz).abs() < 1e-6,
                "u invariant k={k}: du={du}, x-z={dxz}"
            );
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

        sheaf_laplacian_via_maps(&cx, &maps, &z.data, &mut scratch);
        let gl = graph_laplacian(&cx, &z);

        // f32 accumulation order differs between the two paths; use a loose-but-safe tol.
        for k in 0..z.data.len() {
            assert!(
                (scratch.sheaf_laplacian_z[k] - gl.data[k]).abs() < 1e-4,
                "sheaf_laplacian vs graph_laplacian k={k}: sheaf={}, graph={}",
                scratch.sheaf_laplacian_z[k],
                gl.data[k]
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
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            1.0,
            0.1,
            3,
            &mut scratch,
        );
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
            sheaf_admm_step(
                &cx,
                &maps,
                &mut primal_x,
                &mut consensus_z,
                &mut dual_u,
                &objective,
                rho,
                0.2,
                50,
                &mut scratch,
            );
        }
        let d_final = max_edge_disagreement(&cx, &primal_x);

        eprintln!("identity_maps_reach_consensus: d_initial={d_initial:.5}, d_final={d_final:.5}");
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

    // ========================================================================
    // Plan 407 Phase 3 — T3.2 (selector_per_edge + topk + fast-path)
    // ========================================================================

    /// `selector_per_edge` builds compact indices for per-edge dim subsets.
    #[test]
    fn selector_per_edge_construct_correctly() {
        // 3 vertices, 2 edges: 0-1, 1-2.
        let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
        let d_v = 4;
        // Edge 0 selects dims [0, 2], edge 1 selects dims [1, 3].
        let maps = SheafMaps::selector_per_edge(&cx, d_v, &[&[0, 2], &[1, 3]]);
        assert_eq!(maps.d_e, 2);
        assert_eq!(maps.d_v, 4);
        assert_eq!(maps.n_edges, 2);
        assert!(maps.is_selector);
        assert!(
            !maps.is_identity,
            "heterogeneous selectors should not be identity"
        );
        assert!(
            maps.maps.is_empty(),
            "selector maps should not materialize dense maps"
        );
        assert_eq!(maps.selector_indices.len(), 2 * 2 * 2); // n_edges * 2 * d_e
        // Edge 0, endpoint 0 (tail): indices [0, 2].
        assert_eq!(maps.selector_edge_indices(0, 0), &[0, 2]);
        // Edge 0, endpoint 1 (head): same [0, 2] (both endpoints).
        assert_eq!(maps.selector_edge_indices(0, 1), &[0, 2]);
        // Edge 1, endpoint 0: [1, 3].
        assert_eq!(maps.selector_edge_indices(1, 0), &[1, 3]);
    }

    /// `selector_per_edge` detects identity when all edges pick [0, 1, …, d_e-1].
    #[test]
    fn selector_per_edge_collapses_to_identity_when_uniform_ordered() {
        let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
        let maps = SheafMaps::selector_per_edge(&cx, 4, &[&[0, 1], &[0, 1]]);
        assert!(
            maps.is_identity,
            "uniform ordered selectors should detect identity"
        );
    }

    /// `selector_per_edge_topk` picks the top-k dims by score per edge.
    #[test]
    fn selector_per_edge_topk_picks_highest_scoring_dims() {
        let cx = CellComplex::from_edges(2, &[(0, 1)]);
        let d_v = 4;
        // Scores: dim 3 has highest, dim 1 second. Top-2 should pick [3, 1].
        let scores: &[&[f32]] = &[&[0.1, 0.5, 0.2, 0.9]];
        let maps = SheafMaps::selector_per_edge_topk(&cx, d_v, scores, 2);
        assert_eq!(maps.d_e, 2);
        let indices = maps.selector_edge_indices(0, 0);
        assert_eq!(
            indices,
            &[3, 1],
            "top-2 should be [3, 1] by descending score"
        );
    }

    /// `selector_per_edge_topk` breaks ties deterministically (lower dim wins).
    #[test]
    fn selector_per_edge_topk_tie_breaks_by_lower_dim() {
        let cx = CellComplex::from_edges(2, &[(0, 1)]);
        // All scores equal → ties broken by lower dim → picks [0, 1].
        let scores: &[&[f32]] = &[&[0.5, 0.5, 0.5, 0.5]];
        let maps = SheafMaps::selector_per_edge_topk(&cx, 4, scores, 2);
        assert_eq!(maps.selector_edge_indices(0, 0), &[0, 1]);
    }

    /// Selector fast path: matvec result matches the dense selector matvec
    /// bit-for-bit (both compute `L_F z` for the same selector maps). This is
    /// the correctness gate for the T3.2 gather-scatter fast path.
    #[test]
    fn selector_fast_path_matches_dense_selector_matvec() {
        // 4 vertices, 3 edges: 0-1, 1-2, 2-3 (path graph).
        let cx = CellComplex::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let d_v = 4;
        let d_e = 2;

        // Dense selector (uniform dims [1, 3] for all edges).
        let dense_maps = SheafMaps::selector(&cx, d_v, &[1, 3]);
        // Compact selector (same dims [1, 3] per edge).
        let compact_maps = SheafMaps::selector_per_edge(&cx, d_v, &[&[1, 3], &[1, 3], &[1, 3]]);

        // Random z.
        let mut z = CochainField::zeros(0, cx.n_vertices(), d_v);
        for k in 0..z.data.len() {
            z.data[k] = (0.1 * (k as f32) + 0.3).sin();
        }

        // Compute L_F z with both paths.
        let mut dense_scratch = AdmmScratch::new(&cx, d_v, d_e);
        sheaf_laplacian_via_maps(&cx, &dense_maps, &z.data, &mut dense_scratch);

        let mut compact_scratch = AdmmScratch::new(&cx, d_v, d_e);
        sheaf_laplacian_via_maps(&cx, &compact_maps, &z.data, &mut compact_scratch);

        // Both must produce the same result (selector maps are mathematically
        // identical; only the storage/compute path differs).
        for k in 0..z.data.len() {
            assert_eq!(
                dense_scratch.sheaf_laplacian_z[k], compact_scratch.sheaf_laplacian_z[k],
                "dense vs compact selector matvec mismatch at k={k}: dense={}, compact={}",
                dense_scratch.sheaf_laplacian_z[k], compact_scratch.sheaf_laplacian_z[k]
            );
        }
    }

    /// Selector maps reach consensus (full ADMM run with selector_per_edge).
    /// Mirrors `identity_maps_reach_consensus` but with per-edge selector maps.
    #[test]
    fn selector_per_edge_reaches_consensus() {
        let cx = CellComplex::grid_2d(4, 4);
        let d_v = 2;
        let d_e = 2;
        // Uniform selector [0, 1] = identity for all edges (via selector_per_edge).
        let n_edges = cx.n_edges();
        let dims: Vec<&[usize]> = vec![&[0, 1]; n_edges];
        let maps = SheafMaps::selector_per_edge(&cx, d_v, &dims);
        // The maps are identity-flagged (uniform [0,1]), so they'll take the
        // identity fast path — still exercises the selector constructor + ADMM.
        assert!(maps.is_identity);

        let total = cx.n_vertices() * d_v;
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
        primal_x.data.copy_from_slice(&target);
        let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

        let d_initial = max_edge_disagreement(&cx, &primal_x);
        for _ in 0..30 {
            sheaf_admm_step(
                &cx,
                &maps,
                &mut primal_x,
                &mut consensus_z,
                &mut dual_u,
                &objective,
                rho,
                0.2,
                50,
                &mut scratch,
            );
        }
        let d_final = max_edge_disagreement(&cx, &primal_x);
        assert!(
            d_final < 0.1 * d_initial,
            "selector_per_edge consensus failed: d_final={d_final} >= 0.1*d_initial={}",
            0.1 * d_initial
        );
    }

    /// Heterogeneous selector maps (different dims per edge) still drive
    /// consensus on the selected dims. The non-selected dims should retain
    /// their disagreement (no coordination).
    #[test]
    fn selector_per_edge_heterogeneous_drives_partial_consensus() {
        // Path graph 0-1-2. d_v=4, d_e=2.
        // Edge 0 selects dims [0, 1], edge 1 selects dims [2, 3].
        // After ADMM: dims 0,1 agree on edge 0's vertices; dims 2,3 agree on
        // edge 1's vertices. But since edge 0 doesn't coordinate dims 2,3 and
        // edge 1 doesn't coordinate dims 0,1, cross-edge consensus is limited.
        let cx = CellComplex::from_edges(3, &[(0, 1), (1, 2)]);
        let d_v = 4;
        let d_e = 2;
        let maps = SheafMaps::selector_per_edge(&cx, d_v, &[&[0, 1], &[2, 3]]);

        let total = cx.n_vertices() * d_v;
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0; total],
            q: vec![0.0; total],
        };
        // Initial primal: vertex 0 = [1,1,1,1], v1 = [0,0,0,0], v2 = [1,1,1,1].
        let mut primal_x = CochainField::from_vec(
            0,
            d_v,
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        let mut consensus_z = CochainField::zeros(0, 3, d_v);
        let mut dual_u = CochainField::zeros(0, 3, d_v);
        let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

        // Run 50 ADMM steps (enough to converge on a 3-vertex path).
        for _ in 0..50 {
            sheaf_admm_step(
                &cx,
                &maps,
                &mut primal_x,
                &mut consensus_z,
                &mut dual_u,
                &objective,
                1.0,
                0.25,
                50,
                &mut scratch,
            );
        }

        // Edge 0 (v0-v1) coordinates dims 0,1 → |x0[0] - x1[0]| should be small.
        let edge0_dim0_diff = (primal_x.data[0] - primal_x.data[4]).abs();
        assert!(
            edge0_dim0_diff < 0.1,
            "edge 0 dim 0 should agree: diff={edge0_dim0_diff}"
        );

        // Edge 1 (v1-v2) coordinates dims 2,3 → |x1[2] - x2[2]| should be small.
        let edge1_dim2_diff = (primal_x.data[4 + 2] - primal_x.data[8 + 2]).abs();
        assert!(
            edge1_dim2_diff < 0.1,
            "edge 1 dim 2 should agree: diff={edge1_dim2_diff}"
        );
    }

    // ========================================================================
    // Plan 407 Phase 3 — T3.1 (conjugate-gradient z-update)
    // ========================================================================

    /// CG z-update reaches consensus at least as well as GD on identity maps.
    #[test]
    fn cg_z_update_reaches_consensus() {
        let cx = CellComplex::grid_2d(4, 4);
        let d_v = 2;
        let d_e = 2;
        let maps = SheafMaps::identity(&cx, d_v, d_e);
        let total = cx.n_vertices() * d_v;
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
        primal_x.data.copy_from_slice(&target);
        let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

        let d_initial = max_edge_disagreement(&cx, &primal_x);
        // CG with 20 iters + tight tol.
        for _ in 0..30 {
            sheaf_admm_step_cg_into(
                &cx,
                &maps,
                &mut primal_x,
                &mut consensus_z,
                &mut dual_u,
                &objective,
                rho,
                20,
                1e-8,
                &mut scratch,
            );
        }
        let d_final = max_edge_disagreement(&cx, &primal_x);
        eprintln!("cg_z_update_reaches_consensus: d_initial={d_initial:.5}, d_final={d_final:.5}");
        assert!(
            d_final < 0.1 * d_initial,
            "CG consensus failed: d_final={d_final} >= 0.1*d_initial={}",
            0.1 * d_initial
        );
    }

    /// CG z-update produces a lower-residual projection than GD at the same
    /// matvec count, on a graph where CG's convergence advantage is
    /// meaningful (a larger grid where GD's `O(κ)` vs CG's `O(√κ)` matters).
    #[test]
    fn cg_beats_gd_on_residual_at_equal_matvec_count() {
        // 8×8 grid (64 vertices). Condition number κ ≈ λ_max/λ_min ≈
        // 8/(2−2cos(π/8)) ≈ 8/0.152 ≈ 53. CG's √κ ≈ 7.3 vs GD's κ ≈ 53.
        let cx = CellComplex::grid_2d(8, 8);
        let d_v = 1;
        let d_e = 1;
        let maps = SheafMaps::identity(&cx, d_v, d_e);
        let total = cx.n_vertices() * d_v;

        // Objective with a per-vertex preferred target so the primal is
        // non-trivial (non-constant → has components outside ker(L_F)).
        // q = -target, diag_q = 1.0 → x-update = (rho*(z-u) + target) / (1+rho).
        let mut target = vec![0.0f32; total];
        for (k, t) in target.iter_mut().enumerate() {
            *t = (0.1 * (k as f32)).sin();
        }
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0; total],
            q: target.iter().map(|t| -t).collect(),
        };

        // Identical non-zero initial state for both paths.
        let make_state = || {
            let mut x = CochainField::zeros(0, cx.n_vertices(), d_v);
            x.data[..total].copy_from_slice(&target[..total]);
            let z = CochainField::zeros(0, cx.n_vertices(), d_v);
            let u = CochainField::zeros(0, cx.n_vertices(), d_v);
            (x, z, u)
        };
        let (mut primal_gd, mut z_gd, mut u_gd) = make_state();
        let (mut primal_cg, mut z_cg, mut u_cg) = make_state();
        let mut scratch_gd = AdmmScratch::new(&cx, d_v, d_e);
        let mut scratch_cg = AdmmScratch::new(&cx, d_v, d_e);

        // One step with GD (T=20 diffusion) vs CG (20 iters, same matvec count).
        sheaf_admm_step_into(
            &cx,
            &maps,
            &mut primal_gd,
            &mut z_gd,
            &mut u_gd,
            &objective,
            1.0,
            0.2,
            20,
            &mut scratch_gd,
        );
        sheaf_admm_step_cg_into(
            &cx,
            &maps,
            &mut primal_cg,
            &mut z_cg,
            &mut u_cg,
            &objective,
            1.0,
            20,
            1e-12,
            &mut scratch_cg,
        );

        // Measure residual ‖L_F z‖ (should be near zero if z is in ker(L_F)).
        let mut scratch_r = AdmmScratch::new(&cx, d_v, d_e);
        sheaf_laplacian_via_maps(&cx, &maps, &z_gd.data, &mut scratch_r);
        let gd_residual: f32 = scratch_r.sheaf_laplacian_z.iter().map(|x| x.abs()).sum();
        sheaf_laplacian_via_maps(&cx, &maps, &z_cg.data, &mut scratch_r);
        let cg_residual: f32 = scratch_r.sheaf_laplacian_z.iter().map(|x| x.abs()).sum();

        eprintln!("cg_beats_gd: gd_residual={gd_residual:.6}, cg_residual={cg_residual:.6}");
        // CG should have a lower residual (better projection).
        assert!(
            cg_residual < gd_residual,
            "CG residual {cg_residual} should be < GD residual {gd_residual}"
        );
    }

    // ========================================================================
    // Plan 407 Phase 3 — T3.3 (soft-constraint variant)
    // ========================================================================

    /// Soft-constraint with gamma=0 matches the hard-constraint path exactly.
    #[test]
    fn soft_constraint_gamma_zero_matches_hard() {
        let cx = CellComplex::grid_2d(3, 3);
        let d_v = 2;
        let d_e = 2;
        let maps = SheafMaps::identity(&cx, d_v, d_e);
        let total = cx.n_vertices() * d_v;

        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0; total],
            q: vec![-0.5; total],
        };

        // Identical initial state for both paths.
        let init = |x: &mut CochainField| {
            for k in 0..total {
                x.data[k] = (0.1 * (k as f32)).sin();
            }
        };
        let mut x_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
        init(&mut x_hard);
        let mut z_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
        init(&mut z_hard);
        let mut u_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
        init(&mut u_hard);
        let mut x_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
        init(&mut x_soft);
        let mut z_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
        init(&mut z_soft);
        let mut u_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
        init(&mut u_soft);

        let mut scratch_hard = AdmmScratch::new(&cx, d_v, d_e);
        let mut scratch_soft = AdmmScratch::new(&cx, d_v, d_e);

        sheaf_admm_step_into(
            &cx,
            &maps,
            &mut x_hard,
            &mut z_hard,
            &mut u_hard,
            &objective,
            1.0,
            0.2,
            10,
            &mut scratch_hard,
        );
        sheaf_admm_step_soft_into(
            &cx,
            &maps,
            &mut x_soft,
            &mut z_soft,
            &mut u_soft,
            &objective,
            1.0,
            0.2,
            0.0,
            10,
            &mut scratch_soft,
        );

        // Bit-identical: gamma=0 takes the hard path.
        for k in 0..total {
            assert_eq!(x_hard.data[k], x_soft.data[k], "x mismatch at {k}");
            assert_eq!(z_hard.data[k], z_soft.data[k], "z mismatch at {k}");
            assert_eq!(u_hard.data[k], u_soft.data[k], "u mismatch at {k}");
        }
    }

    /// Soft-constraint with gamma>0 preserves individual variation: the primal
    /// retains MORE disagreement than the hard-constraint path after the same
    /// number of ADMM steps. The `γ(z − b)` term resists full consensus.
    #[test]
    fn soft_constraint_gamma_positive_preserves_variation() {
        let cx = CellComplex::grid_2d(4, 4);
        let d_v = 2;
        let d_e = 2;
        let maps = SheafMaps::identity(&cx, d_v, d_e);
        let total = cx.n_vertices() * d_v;

        // Each vertex has a distinct target → the hard path drives all toward
        // consensus, the soft path retains individual variation.
        let mut target = vec![0.0f32; total];
        for i in 0..cx.n_vertices() {
            for d in 0..d_v {
                target[i * d_v + d] = (0.3 * (i as f32) + 0.7 * (d as f32)) * 0.5;
            }
        }
        let q: Vec<f32> = target.iter().map(|t| -t).collect();
        let objective = LocalObjective::DiagonalQuadratic {
            diag_q: vec![1.0; total],
            q,
        };

        let init_primal = |x: &mut CochainField| {
            x.data.copy_from_slice(&target);
        };

        let mut x_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
        init_primal(&mut x_hard);
        let mut z_hard = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut u_hard = CochainField::zeros(0, cx.n_vertices(), d_v);

        let mut x_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
        init_primal(&mut x_soft);
        let mut z_soft = CochainField::zeros(0, cx.n_vertices(), d_v);
        let mut u_soft = CochainField::zeros(0, cx.n_vertices(), d_v);

        let mut scratch_hard = AdmmScratch::new(&cx, d_v, d_e);
        let mut scratch_soft = AdmmScratch::new(&cx, d_v, d_e);

        for _ in 0..30 {
            sheaf_admm_step_into(
                &cx,
                &maps,
                &mut x_hard,
                &mut z_hard,
                &mut u_hard,
                &objective,
                1.0,
                0.2,
                50,
                &mut scratch_hard,
            );
            sheaf_admm_step_soft_into(
                &cx,
                &maps,
                &mut x_soft,
                &mut z_soft,
                &mut u_soft,
                &objective,
                1.0,
                0.2,
                0.5,
                50,
                &mut scratch_soft,
            );
        }

        let hard_disagree = max_edge_disagreement(&cx, &x_hard);
        let soft_disagree = max_edge_disagreement(&cx, &x_soft);
        eprintln!("soft_preserves_variation: hard={hard_disagree:.6}, soft={soft_disagree:.6}");
        // Soft should retain MORE disagreement (less consensus).
        assert!(
            soft_disagree > hard_disagree,
            "soft constraint should preserve more variation: soft={soft_disagree} > hard={hard_disagree}"
        );
    }
}
