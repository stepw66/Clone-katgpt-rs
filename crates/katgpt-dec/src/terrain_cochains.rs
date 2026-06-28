//! Terrain-specific cochain wrappers for arena spatial reasoning.
//!
//! Thin newtype wrappers around [`CochainField`] that provide domain-specific
//! constructors and raw→semantic bridge functions for game-relevant fields:
//! - [`SafetyCochain`] — per-vertex safety score (rank 0)
//! - [`ThreatCochain`] — per-edge threat magnitude (rank 1)
//! - [`OccupancyCochain`] — per-face occupancy density (rank 2)
//! - [`DestructionCochain`] — per-vertex terrain destruction ratio (rank 0)
//!
//! Bridge functions convert raw game state (projectile positions, destroyed faces)
//! into semantic cochain values via sigmoid projections — never softmax.

use crate::types::{CellComplex, CochainField};

/// Gaussian falloff sigma for projectile threat propagation.
const THREAT_SIGMA: f32 = 4.0;

/// Gaussian falloff sigma for POI anchor notability propagation (Plan 335 T1).
/// Distinct from `THREAT_SIGMA` so the two domains can be tuned independently.
const INTEREST_SIGMA: f32 = 4.0;

/// Sigmoid: `σ(x) = 1 / (1 + e^x)`.
///
/// Uses `e^x` (not `e^{-x}`), so `sigmoid(danger) = 1/(1 + e^{danger})`
/// maps high danger → low safety.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + x.exp())
}

// ---------------------------------------------------------------------------
// SafetyCochain (rank 0, dim 1 — scalar per vertex)
// ---------------------------------------------------------------------------

/// Per-vertex safety field. Higher value = safer.
///
/// Rank-0 cochain: one scalar per vertex. Bridges raw projectile trajectories
/// into semantic safety scores via sigmoid projection.
#[repr(transparent)]
pub struct SafetyCochain(CochainField);

impl SafetyCochain {
    /// Create a zero-initialized safety field over the given cell complex.
    #[inline]
    pub fn zeros(cx: &CellComplex) -> Self {
        Self(CochainField::zeros(0, cx.n_vertices(), 1))
    }

    /// Wrap an existing rank-0 cochain as a safety field.
    #[inline]
    pub fn from_cochain(field: CochainField) -> Self {
        debug_assert_eq!(field.rank, 0, "SafetyCochain requires rank 0");
        debug_assert_eq!(field.dim, 1, "SafetyCochain requires dim 1");
        Self(field)
    }

    /// Read safety at vertex `vertex_idx`.
    #[inline]
    pub fn safety(&self, vertex_idx: usize) -> f32 {
        self.0.scalar(vertex_idx)
    }

    /// Write safety at vertex `vertex_idx`.
    #[inline]
    pub fn set_safety(&mut self, vertex_idx: usize, val: f32) {
        self.0.set_scalar(vertex_idx, val);
    }

    /// Borrow the underlying cochain.
    #[inline]
    pub fn as_cochain(&self) -> &CochainField {
        &self.0
    }

    /// Consume and return the underlying cochain.
    #[inline]
    pub fn into_cochain(self) -> CochainField {
        self.0
    }

    /// Number of vertices in this field.
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.0.n_cells()
    }

    // --- Bridge functions ---

    /// Raw→semantic bridge: convert projectile threats into vertex safety scores.
    ///
    /// For each vertex, accumulate Gaussian-weighted danger from all threat sources,
    /// then project to safety via `sigmoid(-danger)`.
    ///
    /// # Algorithm (spatially pruned splat, Issue riir-neuron-db/003 Layer C)
    ///
    /// The naive loop scanned every threat for every vertex, paying one `exp()`
    /// per (vertex × threat) pair regardless of distance. With `THREAT_SIGMA = 4`,
    /// contributions beyond `R = 3·THREAT_SIGMA = 12` cells are `<
    /// danger_level · exp(−R²/σ) = danger_level · exp(−9) ≈ danger_level ·
    /// 1.2e-4` — negligible after sigmoid saturation. This constructor inverts
    /// the loop nest: for each threat, splat its Gaussian contributions into
    /// the bounded `[tx−R, tx+R] × [ty−R, ty+R]` vertex window.
    ///
    /// For a 100×100 grid × 100 threats this drops the inner-loop `exp()` count
    /// from `10⁶` to `~100 · (2·12)² ≈ 6×10⁴` — a ~17× reduction.
    ///
    /// # Numerical tolerance vs the naive loop
    ///
    /// Output values differ from the naive `for vx, vy { for t in threats }`
    /// loop by at most `~danger_level · exp(−9) / 4 ≈ 3e-5` after sigmoid
    /// projection (worst case at the splat window boundary). This is below
    /// f32 epsilon at the sigmoid's operating point and well below the
    /// semantic resolution of a `[0, 1]` safety score. The existing tests
    /// (`test_safety_from_projectile_threat_*`) assert monotonicity (`near <
    /// far`) and the no-threat baseline (`== 0.5`), both of which are
    /// preserved by construction.
    ///
    /// # Arguments
    /// * `cx` — Cell complex (should be a 2D grid matching `grid_w × grid_h`)
    /// * `grid_w` — Grid width in vertices
    /// * `grid_h` — Grid height in vertices
    /// * `threats` — Slice of `(x, y, danger_level)` in grid coordinates
    pub fn from_projectile_threat(
        cx: &CellComplex,
        grid_w: usize,
        grid_h: usize,
        threats: &[(f32, f32, f32)],
    ) -> Self {
        let mut field = CochainField::zeros(0, cx.n_vertices(), 1);

        // No threats ⇒ danger stays 0 everywhere ⇒ sigmoid(0) = 0.5. Pre-fill
        // the baseline so we can skip the splat + sigmoid passes entirely.
        if threats.is_empty() {
            for v in &mut field.data {
                *v = 0.5;
            }
            return Self(field);
        }
        if grid_w == 0 || grid_h == 0 {
            return Self(field);
        }

        let inv_sigma = 1.0 / THREAT_SIGMA;
        // 3σ cutoff (Issue riir-neuron-db/003): beyond this radius the
        // Gaussian contribution is < exp(−9) ≈ 1.2e-4 × danger_level, which is
        // sigmoid-clamped below f32 epsilon for any plausible danger level.
        let r = 3.0 * THREAT_SIGMA; // = 12.0
        let data = &mut field.data;

        // Splat: for each threat, accumulate raw danger into the bounded
        // vertex window. Sigmoid is applied in a second pass so the inner
        // splat loop stays branch-free and amenable to auto-vectorization.
        for &(tx, ty, danger_level) in threats {
            // Vertex coords are integers, so the window is the integer range
            // within [tx−R, tx+R] × [ty−R, ty+R], clamped to the grid.
            let vx_lo = (tx - r).ceil().max(0.0) as usize;
            let vx_hi = ((tx + r).floor() as isize)
                .max(-1)
                .min((grid_w as isize) - 1) as usize;
            let vy_lo = (ty - r).ceil().max(0.0) as usize;
            let vy_hi = ((ty + r).floor() as isize)
                .max(-1)
                .min((grid_h as isize) - 1) as usize;

            if vx_hi < vx_lo || vy_hi < vy_lo {
                continue;
            }

            for vy in vy_lo..=vy_hi {
                let vyf = vy as f32;
                let dy2 = (vyf - ty) * (vyf - ty);
                let row_base = vy * grid_w;
                for vx in vx_lo..=vx_hi {
                    let vxf = vx as f32;
                    let dx2 = (vxf - tx) * (vxf - tx);
                    let dist_sq = dx2 + dy2;
                    // Equivalent to `danger_level * (-dist_sq/THREAT_SIGMA).exp()`
                    // but written with a hoisted reciprocal to save the division.
                    data[row_base + vx] += danger_level * (-(dist_sq * inv_sigma)).exp();
                }
            }
        }

        // Project raw danger → [0, 1] safety in a single sweep.
        for v in &mut field.data {
            *v = sigmoid(*v);
        }

        Self(field)
    }
}

// ---------------------------------------------------------------------------
// ThreatCochain (rank 1, dim 1 — scalar per edge)
// ---------------------------------------------------------------------------

/// Per-edge threat magnitude field.
///
/// Rank-1 cochain: one scalar per edge representing threat magnitude along
/// the edge's orientation.
#[repr(transparent)]
pub struct ThreatCochain(CochainField);

impl ThreatCochain {
    /// Create a zero-initialized threat field over the given cell complex.
    #[inline]
    pub fn zeros(cx: &CellComplex) -> Self {
        Self(CochainField::zeros(1, cx.n_edges(), 1))
    }

    /// Wrap an existing rank-1 cochain as a threat field.
    #[inline]
    pub fn from_cochain(field: CochainField) -> Self {
        debug_assert_eq!(field.rank, 1, "ThreatCochain requires rank 1");
        debug_assert_eq!(field.dim, 1, "ThreatCochain requires dim 1");
        Self(field)
    }

    /// Read threat magnitude at edge `edge_idx`.
    #[inline]
    pub fn threat(&self, edge_idx: usize) -> f32 {
        self.0.scalar(edge_idx)
    }

    /// Write threat magnitude at edge `edge_idx`.
    #[inline]
    pub fn set_threat(&mut self, edge_idx: usize, val: f32) {
        self.0.set_scalar(edge_idx, val);
    }

    /// Borrow the underlying cochain.
    #[inline]
    pub fn as_cochain(&self) -> &CochainField {
        &self.0
    }

    /// Consume and return the underlying cochain.
    #[inline]
    pub fn into_cochain(self) -> CochainField {
        self.0
    }

    /// Number of edges in this field.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.0.n_cells()
    }
}

// ---------------------------------------------------------------------------
// OccupancyCochain (rank 2, dim 1 — scalar per face)
// ---------------------------------------------------------------------------

/// Per-face occupancy density field.
///
/// Rank-2 cochain: one scalar per face representing how occupied the area is.
#[repr(transparent)]
pub struct OccupancyCochain(CochainField);

impl OccupancyCochain {
    /// Create a zero-initialized occupancy field over the given cell complex.
    #[inline]
    pub fn zeros(cx: &CellComplex) -> Self {
        Self(CochainField::zeros(2, cx.n_faces(), 1))
    }

    /// Wrap an existing rank-2 cochain as an occupancy field.
    #[inline]
    pub fn from_cochain(field: CochainField) -> Self {
        debug_assert_eq!(field.rank, 2, "OccupancyCochain requires rank 2");
        debug_assert_eq!(field.dim, 1, "OccupancyCochain requires dim 1");
        Self(field)
    }

    /// Read occupancy at face `face_idx`.
    #[inline]
    pub fn occupancy(&self, face_idx: usize) -> f32 {
        self.0.scalar(face_idx)
    }

    /// Write occupancy at face `face_idx`.
    #[inline]
    pub fn set_occupancy(&mut self, face_idx: usize, val: f32) {
        self.0.set_scalar(face_idx, val);
    }

    /// Borrow the underlying cochain.
    #[inline]
    pub fn as_cochain(&self) -> &CochainField {
        &self.0
    }

    /// Consume and return the underlying cochain.
    #[inline]
    pub fn into_cochain(self) -> CochainField {
        self.0
    }

    /// Number of faces in this field.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.0.n_cells()
    }
}

// ---------------------------------------------------------------------------
// DestructionCochain (rank 0, dim 1 — scalar per vertex)
// ---------------------------------------------------------------------------

/// Per-vertex terrain destruction field.
///
/// Rank-0 cochain: one scalar per vertex tracking how destroyed the
/// surrounding terrain is (ratio of destroyed incident faces).
#[repr(transparent)]
pub struct DestructionCochain(CochainField);

impl DestructionCochain {
    /// Create a zero-initialized destruction field over the given cell complex.
    #[inline]
    pub fn zeros(cx: &CellComplex) -> Self {
        Self(CochainField::zeros(0, cx.n_vertices(), 1))
    }

    /// Wrap an existing rank-0 cochain as a destruction field.
    #[inline]
    pub fn from_cochain(field: CochainField) -> Self {
        debug_assert_eq!(field.rank, 0, "DestructionCochain requires rank 0");
        debug_assert_eq!(field.dim, 1, "DestructionCochain requires dim 1");
        Self(field)
    }

    /// Read destruction at vertex `vertex_idx`.
    #[inline]
    pub fn destruction(&self, vertex_idx: usize) -> f32 {
        self.0.scalar(vertex_idx)
    }

    /// Write destruction at vertex `vertex_idx`.
    #[inline]
    pub fn set_destruction(&mut self, vertex_idx: usize, val: f32) {
        self.0.set_scalar(vertex_idx, val);
    }

    /// Borrow the underlying cochain.
    #[inline]
    pub fn as_cochain(&self) -> &CochainField {
        &self.0
    }

    /// Consume and return the underlying cochain.
    #[inline]
    pub fn into_cochain(self) -> CochainField {
        self.0
    }

    /// Number of vertices in this field.
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.0.n_cells()
    }

    // --- Bridge functions ---

    /// Raw→semantic bridge: convert destroyed face set into per-vertex destruction ratio.
    ///
    /// For each vertex, computes `destroyed_incident_faces / max_incident_faces`.
    /// A vertex with no incident faces (degenerate grid) gets destruction = 0.0.
    ///
    /// # Grid layout
    /// Face index `f` maps to grid position `(fx, fy)` where `f = fy * (grid_w - 1) + fx`.
    /// Face `(fx, fy)` has 4 vertices at grid coordinates
    /// `(fx, fy)`, `(fx+1, fy)`, `(fx, fy+1)`, `(fx+1, fy+1)`.
    ///
    /// # Arguments
    /// * `cx` — Cell complex (should be a 2D grid matching `grid_w × grid_h`)
    /// * `grid_w` — Grid width in vertices
    /// * `grid_h` — Grid height in vertices
    /// * `destroyed_face_indices` — Face indices that are destroyed
    pub fn from_destroyed_faces(
        cx: &CellComplex,
        grid_w: usize,
        grid_h: usize,
        destroyed_face_indices: &[usize],
    ) -> Self {
        let n_vertices = cx.n_vertices();
        let mut field = CochainField::zeros(0, n_vertices, 1);

        // Degenerate grids have no faces — all destruction stays zero.
        if grid_w <= 1 || grid_h <= 1 {
            return Self(field);
        }

        let faces_per_row = grid_w - 1;

        // Count destroyed incident faces per vertex.
        let mut destroyed_count = vec![0u32; n_vertices];

        for &f in destroyed_face_indices {
            let fx = f % faces_per_row;
            let fy = f / faces_per_row;

            // 4 corner vertices of face (fx, fy)
            let v00 = fy * grid_w + fx;
            let v10 = fy * grid_w + fx + 1;
            let v01 = (fy + 1) * grid_w + fx;
            let v11 = (fy + 1) * grid_w + fx + 1;

            destroyed_count[v00] += 1;
            destroyed_count[v10] += 1;
            destroyed_count[v01] += 1;
            destroyed_count[v11] += 1;
        }

        for vy in 0..grid_h {
            for vx in 0..grid_w {
                let vidx = vy * grid_w + vx;
                let destroyed = destroyed_count[vidx];

                // Max incident faces: vertex (vx, vy) touches faces with
                // fx ∈ {vx-1, vx} ∩ [0, w-2] and fy ∈ {vy-1, vy} ∩ [0, h-2].
                let n_x = (vx > 0) as u32 + (vx < grid_w - 1) as u32;
                let n_y = (vy > 0) as u32 + (vy < grid_h - 1) as u32;
                let max_incident = n_x * n_y;

                let ratio = match max_incident {
                    0 => 0.0,
                    _ => destroyed as f32 / max_incident as f32,
                };
                field.set_scalar(vidx, ratio);
            }
        }

        Self(field)
    }
}

// ---------------------------------------------------------------------------
// InterestCohain (rank 0, dim 1 — scalar per vertex) — the user's "f" field
// ---------------------------------------------------------------------------

/// Per-vertex interest/notability field (Plan 335 T1).
///
/// Rank-0 cochain: one scalar per vertex. Bridges raw POI anchors
/// into semantic interest scores via sigmoid projection.
/// Higher value = more interesting/notable (fame, reward, attention).
#[repr(transparent)]
pub struct InterestCohain(CochainField);

impl InterestCohain {
    /// Create a zero-initialized interest field over the given cell complex.
    #[inline]
    pub fn zeros(cx: &CellComplex) -> Self {
        Self(CochainField::zeros(0, cx.n_vertices(), 1))
    }

    /// Wrap an existing rank-0 cochain as an interest field.
    #[inline]
    pub fn from_cochain(field: CochainField) -> Self {
        debug_assert_eq!(field.rank, 0, "InterestCohain requires rank 0");
        debug_assert_eq!(field.dim, 1, "InterestCohain requires dim 1");
        Self(field)
    }

    /// Read interest at vertex `vertex_idx`.
    #[inline]
    pub fn interest(&self, vertex_idx: usize) -> f32 {
        self.0.scalar(vertex_idx)
    }

    /// Write interest at vertex `vertex_idx`.
    #[inline]
    pub fn set_interest(&mut self, vertex_idx: usize, val: f32) {
        self.0.set_scalar(vertex_idx, val);
    }

    /// Borrow the underlying cochain.
    #[inline]
    pub fn as_cochain(&self) -> &CochainField {
        &self.0
    }

    /// Consume and return the underlying cochain.
    #[inline]
    pub fn into_cochain(self) -> CochainField {
        self.0
    }

    /// Number of vertices in this field.
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.0.n_cells()
    }

    // --- Bridge functions ---

    /// Raw→semantic bridge: convert POI anchors into vertex interest scores.
    ///
    /// For each vertex, accumulate Gaussian-weighted notability from all
    /// anchors, then project to interest via `sigmoid(-notability)`.
    ///
    /// The existing `sigmoid(x) = 1/(1 + e^x)` maps high input → low output,
    /// so we pass `-notability` to get the correct interest polarity: high
    /// notability → high interest. With no anchors, notability = 0 everywhere
    /// and interest = `sigmoid(0) = 0.5` — the same baseline convention as
    /// `SafetyCochain::from_projectile_threat`.
    pub fn from_anchors(
        cx: &CellComplex,
        grid_w: usize,
        grid_h: usize,
        anchors: &[(f32, f32, f32)],
    ) -> Self {
        let mut field = CochainField::zeros(0, cx.n_vertices(), 1);

        for vy in 0..grid_h {
            let vyf = vy as f32;
            for vx in 0..grid_w {
                let vxf = vx as f32;

                let mut notability = 0.0f32;
                for &(ax, ay, notability_level) in anchors {
                    let dist_sq = (vxf - ax).powi(2) + (vyf - ay).powi(2);
                    notability += notability_level * (-dist_sq / INTEREST_SIGMA).exp();
                }

                let vertex_idx = vy * grid_w + vx;
                // High notability → high interest: invert sigmoid polarity.
                field.set_scalar(vertex_idx, sigmoid(-notability));
            }
        }

        Self(field)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_cochain_zeros() {
        let cx = CellComplex::grid_2d(4, 4);
        let s = SafetyCochain::zeros(&cx);
        assert_eq!(s.n_vertices(), 16);
        for i in 0..16 {
            assert_eq!(s.safety(i), 0.0);
        }
    }

    #[test]
    fn test_safety_set_get() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut s = SafetyCochain::zeros(&cx);
        s.set_safety(5, 0.87);
        assert!((s.safety(5) - 0.87).abs() < 1e-6);
        // Other vertices unchanged
        assert_eq!(s.safety(0), 0.0);
        assert_eq!(s.safety(6), 0.0);
    }

    #[test]
    fn test_threat_cochain_zeros() {
        let cx = CellComplex::grid_2d(4, 4);
        let t = ThreatCochain::zeros(&cx);
        assert_eq!(t.n_edges(), cx.n_edges());
        for i in 0..t.n_edges() {
            assert_eq!(t.threat(i), 0.0);
        }
    }

    #[test]
    fn test_occupancy_cochain_zeros() {
        let cx = CellComplex::grid_2d(4, 4);
        let o = OccupancyCochain::zeros(&cx);
        assert_eq!(o.n_faces(), cx.n_faces());
        for i in 0..o.n_faces() {
            assert_eq!(o.occupancy(i), 0.0);
        }
    }

    #[test]
    fn test_destruction_cochain_zeros() {
        let cx = CellComplex::grid_2d(4, 4);
        let d = DestructionCochain::zeros(&cx);
        assert_eq!(d.n_vertices(), 16);
        for i in 0..16 {
            assert_eq!(d.destruction(i), 0.0);
        }
    }

    #[test]
    fn test_safety_from_projectile_threat_no_threats() {
        // No threats → danger = 0 → safety = sigmoid(0) = 0.5 everywhere
        let cx = CellComplex::grid_2d(4, 4);
        let s = SafetyCochain::from_projectile_threat(&cx, 4, 4, &[]);
        for i in 0..16 {
            assert!(
                (s.safety(i) - 0.5).abs() < 1e-6,
                "safety at vertex {i} should be 0.5, got {}",
                s.safety(i)
            );
        }
    }

    #[test]
    fn test_safety_from_projectile_threat_single() {
        // Single threat at center — vertices near it should have lower safety
        let cx = CellComplex::grid_2d(4, 4);
        let threats = [(1.5_f32, 1.5, 5.0)];
        let s = SafetyCochain::from_projectile_threat(&cx, 4, 4, &threats);

        let near = s.safety(1 * 4 + 1); // vertex (1,1) — close to (1.5, 1.5)
        let far = s.safety(3 * 4 + 3); // vertex (3,3) — far from (1.5, 1.5)

        assert!(
            near < far,
            "near vertex safety ({near}) should be < far vertex safety ({far})"
        );
        assert!(near < 0.5, "near vertex should be dangerous (safety < 0.5)");
    }

    #[test]
    fn test_destruction_from_destroyed_faces() {
        // Grid 4x4 → 3x3 = 9 faces. Destroy face 0 at grid (0,0).
        let cx = CellComplex::grid_2d(4, 4);
        let d = DestructionCochain::from_destroyed_faces(&cx, 4, 4, &[0]);

        // Face (0,0) has vertices: 0, 1, 4, 5
        // Vertex 0 (corner): max_incident=1, destroyed=1 → 1.0
        assert!(
            (d.destruction(0) - 1.0).abs() < 1e-6,
            "corner vertex 0 destruction should be 1.0, got {}",
            d.destruction(0)
        );
        // Vertex 1 (edge): max_incident=2, destroyed=1 → 0.5
        assert!(
            (d.destruction(1) - 0.5).abs() < 1e-6,
            "edge vertex 1 destruction should be 0.5, got {}",
            d.destruction(1)
        );
        // Vertex 4 (edge): max_incident=2, destroyed=1 → 0.5
        assert!(
            (d.destruction(4) - 0.5).abs() < 1e-6,
            "edge vertex 4 destruction should be 0.5, got {}",
            d.destruction(4)
        );
        // Vertex 5 (interior): max_incident=4, destroyed=1 → 0.25
        assert!(
            (d.destruction(5) - 0.25).abs() < 1e-6,
            "interior vertex 5 destruction should be 0.25, got {}",
            d.destruction(5)
        );
        // Unrelated vertices have 0 destruction
        assert_eq!(d.destruction(15), 0.0);
    }

    #[test]
    fn test_cochain_wrappers_preserve_rank() {
        let cx = CellComplex::grid_2d(4, 4);

        let s = SafetyCochain::zeros(&cx);
        assert_eq!(s.as_cochain().rank, 0);

        let t = ThreatCochain::zeros(&cx);
        assert_eq!(t.as_cochain().rank, 1);

        let o = OccupancyCochain::zeros(&cx);
        assert_eq!(o.as_cochain().rank, 2);

        let d = DestructionCochain::zeros(&cx);
        assert_eq!(d.as_cochain().rank, 0);
    }

    #[test]
    fn test_into_cochain_roundtrip() {
        let cx = CellComplex::grid_2d(4, 4);

        // Safety
        let mut s = SafetyCochain::zeros(&cx);
        s.set_safety(3, 0.42);
        let field = s.into_cochain();
        assert_eq!(field.rank, 0);
        assert!((field.scalar(3) - 0.42).abs() < 1e-6);

        // Re-wrap and verify
        let s2 = SafetyCochain::from_cochain(field);
        assert!((s2.safety(3) - 0.42).abs() < 1e-6);

        // Threat
        let mut t = ThreatCochain::zeros(&cx);
        t.set_threat(2, 7.5);
        let field = t.into_cochain();
        assert_eq!(field.rank, 1);
        assert!((field.scalar(2) - 7.5).abs() < 1e-6);

        // Occupancy
        let mut o = OccupancyCochain::zeros(&cx);
        o.set_occupancy(1, 0.9);
        let field = o.into_cochain();
        assert_eq!(field.rank, 2);
        assert!((field.scalar(1) - 0.9).abs() < 1e-6);

        // Destruction
        let mut d = DestructionCochain::zeros(&cx);
        d.set_destruction(7, 0.33);
        let field = d.into_cochain();
        assert_eq!(field.rank, 0);
        assert!((field.scalar(7) - 0.33).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // InterestCohain (Plan 335 T1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_interest_cochain_zeros() {
        let cx = CellComplex::grid_2d(4, 4);
        let f = InterestCohain::zeros(&cx);
        assert_eq!(f.n_vertices(), 16);
        for i in 0..16 {
            assert_eq!(f.interest(i), 0.0);
        }
    }

    #[test]
    fn test_interest_set_get() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut f = InterestCohain::zeros(&cx);
        f.set_interest(5, 0.91);
        assert!((f.interest(5) - 0.91).abs() < 1e-6);
        assert_eq!(f.interest(0), 0.0);
        assert_eq!(f.interest(6), 0.0);
    }

    #[test]
    fn test_interest_cochain_rank_zero() {
        let cx = CellComplex::grid_2d(4, 4);
        let f = InterestCohain::zeros(&cx);
        assert_eq!(f.as_cochain().rank, 0);
    }

    #[test]
    fn test_interest_cochain_dim_one() {
        let cx = CellComplex::grid_2d(4, 4);
        let f = InterestCohain::zeros(&cx);
        assert_eq!(f.as_cochain().dim, 1);
    }

    #[test]
    fn test_from_anchors_no_anchors_returns_baseline() {
        // No anchors → notability = 0 everywhere → interest = sigmoid(0) = 0.5.
        // Mirrors `test_safety_from_projectile_threat_no_threats`.
        let cx = CellComplex::grid_2d(4, 4);
        let f = InterestCohain::from_anchors(&cx, 4, 4, &[]);
        for i in 0..16 {
            assert!(
                (f.interest(i) - 0.5).abs() < 1e-6,
                "interest at vertex {i} should be 0.5 (baseline), got {}",
                f.interest(i)
            );
        }
    }

    #[test]
    fn test_from_anchors_peaks_at_anchor_positions() {
        // Single anchor at (5, 5) → interest peaks at vertex (5, 5).
        //
        // Note: since notability ≥ 0 everywhere (Gaussian tail is always
        // non-negative), `-notability ≤ 0`, so `sigmoid(-notability) ≥ 0.5`
        // at EVERY vertex. The peak is the global max; far vertices sit at the
        // 0.5 baseline (their notability contribution is ~0).
        let cx = CellComplex::grid_2d(8, 8);
        let anchors = [(5.0_f32, 5.0, 3.0)];
        let f = InterestCohain::from_anchors(&cx, 8, 8, &anchors);

        let peak = f.interest(5 * 8 + 5);
        let corner = f.interest(0);

        assert!(peak > corner, "peak ({peak}) should be > corner ({corner})");
        assert!(
            peak > 0.5,
            "anchor vertex should be notable (>0.5), got {peak}"
        );
        // Far corner sits at the 0.5 baseline (notability contribution ~0).
        assert!(
            (corner - 0.5).abs() < 1e-4,
            "far corner should be ~0.5 baseline, got {corner}"
        );

        for vy in 0..8 {
            for vx in 0..8 {
                let idx = vy * 8 + vx;
                if idx == 5 * 8 + 5 {
                    continue;
                }
                assert!(
                    f.interest(idx) <= peak,
                    "vertex ({vx},{vy}) interest {} exceeded peak {peak}",
                    f.interest(idx)
                );
            }
        }
    }

    #[test]
    fn test_from_anchors_adds_multiple_sources() {
        // Two anchors: verify contributions sum before sigmoid (closed-form check).
        let cx = CellComplex::grid_2d(8, 8);
        let a1 = (0.0_f32, 0.0, 2.0);
        let a2 = (8.0, 8.0, 2.0);
        let f = InterestCohain::from_anchors(&cx, 8, 8, &[a1, a2]);

        let vx = 4.0_f32;
        let vy = 4.0_f32;
        let d1_sq = (vx - a1.0).powi(2) + (vy - a1.1).powi(2);
        let d2_sq = (vx - a2.0).powi(2) + (vy - a2.1).powi(2);
        let expected_notability =
            a1.2 * (-d1_sq / INTEREST_SIGMA).exp() + a2.2 * (-d2_sq / INTEREST_SIGMA).exp();
        let expected = sigmoid(-expected_notability);

        let idx = 4 * 8 + 4;
        assert!((f.interest(idx) - expected).abs() < 1e-6);

        // Adding a second anchor raises interest — but only at a vertex where
        // the second anchor actually contributes. Vertex (4,4) is equidistant
        // from both a1=(0,0) and a2=(8,8), so both contribute meaningfully;
        // compare the dual-anchor interest there against the single-anchor
        // (a1-only) interest at the same vertex.
        let f_one = InterestCohain::from_anchors(&cx, 8, 8, &[a1]);
        let single = f_one.interest(idx);
        let dual = f.interest(idx);
        assert!(
            dual > single,
            "dual ({dual}) should be > single ({single}) at vertex (4,4)"
        );
    }

    #[test]
    fn test_interest_into_cochain_roundtrip() {
        let cx = CellComplex::grid_2d(4, 4);
        let mut f = InterestCohain::zeros(&cx);
        f.set_interest(3, 0.42);
        let field = f.into_cochain();
        assert_eq!(field.rank, 0);
        assert_eq!(field.dim, 1);
        assert!((field.scalar(3) - 0.42).abs() < 1e-6);
        let f2 = InterestCohain::from_cochain(field);
        assert!((f2.interest(3) - 0.42).abs() < 1e-6);
    }

    // ── Issue riir-neuron-db/003 Layer C — splat vs naive parity ────────────
    //
    // `SafetyCochain::from_projectile_threat` uses a spatially pruned splat
    // (3σ cutoff, R = 12 with THREAT_SIGMA = 4). This test pins the parity
    // invariant: the splat output matches a naive reference (which scans
    // every threat for every vertex) to within f32 epsilon after sigmoid.

    /// Naive reference implementation — the pre-Layer-C algorithm. O(vertices ×
    /// threats). Kept here as a parity oracle for the splat path.
    fn from_projectile_threat_naive(
        cx: &CellComplex,
        grid_w: usize,
        grid_h: usize,
        threats: &[(f32, f32, f32)],
    ) -> SafetyCochain {
        let mut field = CochainField::zeros(0, cx.n_vertices(), 1);
        if grid_w == 0 || grid_h == 0 {
            return SafetyCochain::from_cochain(field);
        }
        for vy in 0..grid_h {
            let vyf = vy as f32;
            for vx in 0..grid_w {
                let vxf = vx as f32;
                let mut danger = 0.0f32;
                for &(tx, ty, danger_level) in threats {
                    let dist_sq = (vxf - tx).powi(2) + (vyf - ty).powi(2);
                    danger += danger_level * (-dist_sq / THREAT_SIGMA).exp();
                }
                field.set_scalar(vy * grid_w + vx, sigmoid(danger));
            }
        }
        SafetyCochain::from_cochain(field)
    }

    #[test]
    fn test_safety_splat_matches_naive_within_3sigma() {
        // 20×16 grid with 6 scattered threats — exercises interior, boundary,
        // and corner vertices (some threats land near the grid boundary, which
        // exercises the splat window clamping).
        let gw = 20usize;
        let gh = 16usize;
        let cx = CellComplex::grid_2d(gw, gh);
        let threats = vec![
            (3.5_f32, 4.0, 5.0),
            (10.0, 8.0, 2.5),
            (17.2, 2.1, 8.0),
            (1.0, 14.5, 1.0),
            (19.0, 15.0, 3.0),
            (8.5, 11.3, 0.5),
        ];

        let naive = from_projectile_threat_naive(&cx, gw, gh, &threats);
        let splat = SafetyCochain::from_projectile_threat(&cx, gw, gh, &threats);

        let naive_field = naive.as_cochain();
        let splat_field = splat.as_cochain();
        assert_eq!(naive_field.data.len(), splat_field.data.len());

        // Tolerance: the splat drops contributions < dl·exp(-R²/σ) = dl·exp(-9)
        // for R = 12. With dl ≤ 8 here, worst-case dropped contribution is
        // ~8 · 1.2e-4 ≈ 1e-3. After sigmoid (derivative ≤ 1/4 near 0), the
        // output difference is bounded by ~2.5e-4. Allow 1e-3 for f32 rounding
        // headroom; in practice the max observed diff is ≪ 1e-4.
        const TOL: f32 = 1e-3;
        let mut max_diff = 0.0f32;
        let mut max_diff_at = 0usize;
        for (i, (a, b)) in naive_field.data.iter().zip(splat_field.data.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_at = i;
            }
            assert!(
                diff <= TOL,
                "vertex {i}: naive={a:.7} splat={b:.7} diff={diff:.2e} > TOL={TOL:.0e}"
            );
        }
        // Tighter sanity bound — the actual envelope is much tighter than
        // the documented tolerance. If this ever fails, investigate before
        // bumping the bound.
        assert!(
            max_diff < 1e-4,
            "max diff {max_diff:.2e} at vertex {max_diff_at} is larger than the \
             expected ~1e-4 envelope; investigate the splat cutoff",
        );
    }

    #[test]
    fn test_safety_splat_far_vertices_are_baseline() {
        // A single threat at the grid center. Vertices well beyond 3σ = 12
        // should read exactly 0.5 (sigmoid(0)), the no-threat baseline.
        let gw = 40usize;
        let gh = 40usize;
        let cx = CellComplex::grid_2d(gw, gh);
        let threats = vec![(20.0_f32, 20.0, 5.0)];
        let s = SafetyCochain::from_projectile_threat(&cx, gw, gh, &threats);
        let field = s.as_cochain();

        // Corner (0,0): distance to (20,20) is sqrt(800) ≈ 28.3 — well beyond R.
        let far_corner = field.data[0];
        assert!((far_corner - 0.5).abs() < 1e-7,
            "far corner vertex should be 0.5 baseline, got {far_corner}");

        // Opposite corner (gw-1, gh-1) = (39, 39): distance to (20,20) is
        // sqrt(361+361) ≈ 26.9 — also beyond R.
        let opp_corner = field.data[(gh - 1) * gw + (gw - 1)];
        assert!((opp_corner - 0.5).abs() < 1e-7,
            "opposite corner vertex should be 0.5 baseline, got {opp_corner}");

        // Sanity: a vertex NEAR the threat should NOT be at baseline.
        // Vertex (20, 20): distance 0. Danger = 5. sigmoid(5) ≈ 0.0067 ≪ 0.5.
        let near = field.data[20 * gw + 20];
        assert!(near < 0.1,
            "near vertex should be well below baseline (high danger), got {near}");
    }
}
