//! Core types for the Canvas Schema Compiler (Plan 419, Research 398).
//!
//! These types model a declared causal topology over a structured latent
//! space ("canvas"). The compiler lowers them into attention masks + loss
//! weight masks; see [`crate::canvas::compile_schema`].
//!
//! Vocabulary (paper §2):
//! - **Canvas** — a structured latent volume `T × H × W` with `d_model` channels.
//! - **Region** — an axis-aligned box `[t0..t1) × [h0..h1) × [w0..w1)` inside
//!   the canvas, carrying a semantic type + loss role + default attention family.
//! - **Topology** — a directed graph of `Connection`s between regions. A
//!   connection `src → dst` means `dst` may attend to `src`. **Absence of an
//!   edge is a hard prohibition** → exact marginal independence for binary masks.
//! - **Schema** — `CanvasLayout + CanvasTopology`, the input to the compiler.

use core::ops::Range;

/// Fixed dimensionality of a [`SemanticType`]'s frozen embedding.
///
/// Chosen to match the HLA 64-dim affect manifold and the PKM query dim, so a
/// semantic type's embedding lives in the same latent space as the runtime
/// affect/steering vectors. This is a *convention*, not a hard constraint on
/// the math — `transfer_distance` is pure cosine and would work at any dim.
pub const SEMANTIC_EMBED_DIM: usize = 64;

/// Axis-aligned region bounds inside the canvas volume `[0..T) × [0..H) × [0..W)`.
///
/// All bounds are half-open (`start` inclusive, `end` exclusive), matching
/// Rust range semantics. A region with `t0 == t1` is a single-frame slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanvasBounds {
    /// Temporal start (inclusive).
    pub t0: u32,
    /// Temporal end (exclusive).
    pub t1: u32,
    /// Height start (inclusive).
    pub h0: u32,
    /// Height end (exclusive).
    pub h1: u32,
    /// Width start (inclusive).
    pub w0: u32,
    /// Width end (exclusive).
    pub w1: u32,
}

impl CanvasBounds {
    /// Construct bounds. The convention is half-open `[start, end)` on each axis.
    #[inline]
    pub const fn new(t0: u32, t1: u32, h0: u32, h1: u32, w0: u32, w1: u32) -> Self {
        Self { t0, t1, h0, h1, w0, w1 }
    }

    /// Number of temporal frames spanned: `t1 - t0`.
    #[inline]
    pub const fn dt(self) -> u32 {
        self.t1.saturating_sub(self.t0)
    }

    /// Number of height rows spanned: `h1 - h0`.
    #[inline]
    pub const fn dh(self) -> u32 {
        self.h1.saturating_sub(self.h0)
    }

    /// Number of width columns spanned: `w1 - w0`.
    #[inline]
    pub const fn dw(self) -> u32 {
        self.w1.saturating_sub(self.w0)
    }

    /// Volume of the region: `dt · dh · dw`.
    ///
    /// Returns 0 for a degenerate (empty) region. Used by
    /// [`super::region_indices`] to compute the flat position count.
    #[inline]
    pub const fn volume(self) -> usize {
        (self.dt() as usize) * (self.dh() as usize) * (self.dw() as usize)
    }
}

/// Newtype index into a [`CanvasLayout`]'s `regions` vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(pub usize);

impl RegionId {
    /// Construct a region index.
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Underlying index.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl From<usize> for RegionId {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

/// A frozen semantic type carried by a region, used by
/// [`super::transfer_distance`] to compute schema-ABI compatibility.
///
/// The embedding is a fixed-size `[f32; SEMANTIC_EMBED_DIM]` slice so that
/// `transfer_distance` is a zero-allocation cosine. Embeddings are **frozen
/// inputs** — never trained, never mutated at runtime (modelless mandate).
#[derive(Clone, PartialEq)]
pub struct SemanticType {
    /// Human-readable name (e.g. `"camera"`, `"joint_angles"`).
    pub name: &'static str,
    /// Frozen unit-or-arbitrary-norm embedding. `transfer_distance` normalizes
    /// internally, so callers need not pre-normalize.
    pub frozen_embedding: [f32; SEMANTIC_EMBED_DIM],
}

impl SemanticType {
    /// Construct a named semantic type with the given frozen embedding.
    #[inline]
    pub const fn new(name: &'static str, frozen_embedding: [f32; SEMANTIC_EMBED_DIM]) -> Self {
        Self { name, frozen_embedding }
    }

    /// Construct a semantic type whose embedding is a single hot axis (a basis
    /// vector). Two basis-vector types at different axes are maximally distant
    /// (`transfer_distance == 1.0`); at the same axis they are identical (0.0).
    /// Useful for tests and for declaring orthogonal type partitions.
    pub fn basis(name: &'static str, axis: usize) -> Self {
        let mut emb = [0.0_f32; SEMANTIC_EMBED_DIM];
        if axis < SEMANTIC_EMBED_DIM {
            emb[axis] = 1.0;
        }
        Self { name, frozen_embedding: emb }
    }
}

impl core::fmt::Debug for SemanticType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Don't dump the whole 64-f32 array in debug output.
        f.debug_struct("SemanticType")
            .field("name", &self.name)
            .field("dim", &SEMANTIC_EMBED_DIM)
            .finish()
    }
}

/// The family of attention function a region or connection uses. Mirrors
/// paper §2.5's taxonomy. Consumers dispatch on this when materializing the
/// compiled mask into a concrete kernel.
///
/// This enum carries **no behavior** — it is a routing tag. The actual
/// attention computation lives in the consumer's attention path (AC-Prefix,
/// VortexFlow, etc.). The compiler only emits the *mask structure*; the
/// function family tells the consumer *how* to interpret each edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AttentionFnFamily {
    /// Full cross-attention between two regions.
    Cross,
    /// Linear attention (kernelized, no N² materialization).
    Linear,
    /// Sigmoid-gated attention (the codebase default — never softmax, per AGENTS.md).
    Sigmoid,
    /// Multiplicative gate (GLU-style).
    Gated,
    /// Perceiver-style learned-query bottleneck (Plan 245 still-perceiver).
    Perceiver,
    /// Pooling (read-out / global-summary).
    Pooling,
    /// Direct copy (identity passthrough).
    Copy,
    /// Mamba SSM.
    Mamba,
    /// RWKV time-mix.
    Rwkv,
    /// Hyena long-convolution.
    Hyena,
    /// Local windowed attention.
    Local,
    /// Arbitrary sparse pattern (the compiled-mask native form).
    Sparse,
    /// No attention (region is inert / non-participating this step).
    None,
    /// Fixed-random projection (Performer/ESBMB-style).
    RandomFixed,
    /// Mixture over the above (consumer resolves weights).
    Mixture,
}

/// A declared region of the canvas: spatial bounds + temporal period + loss
/// role + optional semantic type + default attention family.
#[derive(Clone)]
pub struct RegionSpec {
    /// Human-readable region name (e.g. `"visual"`, `"action"`).
    pub name: &'static str,
    /// Axis-aligned bounds inside the canvas volume.
    pub bounds: CanvasBounds,
    /// Temporal update period (frames). Period 1 = updates every frame;
    /// period N = persists for N frames before re-evaluating. Documentational
    /// for the compiler (does not affect mask structure); consumed by the
    /// runtime scheduler.
    pub period: u32,
    /// Whether this region contributes to the loss (and is thus an "output"
    /// region for the [`super::build_loss_weight_mask`] computation).
    pub is_output: bool,
    /// Loss weight `ω_r` for this region when it is an output. Non-output
    /// regions get zero loss weight regardless of this field.
    pub loss_weight: f32,
    /// Optional semantic type for [`super::transfer_distance`] compatibility.
    pub semantic_type: Option<SemanticType>,
    /// Default attention family for connections into this region. A
    /// [`Connection`]'s `fn_family` overrides this per-edge.
    pub default_attn: AttentionFnFamily,
}

impl core::fmt::Debug for RegionSpec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RegionSpec")
            .field("name", &self.name)
            .field("bounds", &self.bounds)
            .field("period", &self.period)
            .field("is_output", &self.is_output)
            .field("loss_weight", &self.loss_weight)
            .field("default_attn", &self.default_attn)
            .finish_non_exhaustive()
    }
}

impl RegionSpec {
    /// Construct a region spec with all fields.
    #[inline]
    pub const fn new(
        name: &'static str,
        bounds: CanvasBounds,
        period: u32,
        is_output: bool,
        loss_weight: f32,
        semantic_type: Option<SemanticType>,
        default_attn: AttentionFnFamily,
    ) -> Self {
        Self { name, bounds, period, is_output, loss_weight, semantic_type, default_attn }
    }
}

/// A directed connection `src → dst`: `dst` may attend to `src`.
///
/// Per paper §2.3 convention, content flows `src → dst` because `dst`'s query
/// positions read `src`'s key/value positions. **Absence of a connection is a
/// hard prohibition** — for a binary mask (`weight ∈ {0, 1}`) this gives exact
/// marginal independence (see [`super::can_reach`] / Plan 419 Phase 3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Connection {
    /// Source region (keys/values come from here).
    pub src: RegionId,
    /// Destination region (queries come from here).
    pub dst: RegionId,
    /// Edge weight. `1.0` (or any positive value) = fully present; `0.0` =
    /// absent (equivalent to omitting the edge). For the exact marginal
    /// independence guarantee the mask must be binary (`weight ∈ {0, 1}`).
    pub weight: f32,
    /// Source temporal offset. `None` = unconstrained (any frame of `src` may
    /// align with any frame of `dst`). `Some(k)` = the source frame must be at
    /// reference `ref + k` for some shared `ref` (see [`super::temporal_aligns`]).
    pub t_src: Option<i32>,
    /// Destination temporal offset. See [`Self::t_src`].
    pub t_dst: Option<i32>,
    /// Per-edge attention family override. `None` = use the destination
    /// region's `default_attn`.
    pub fn_family: Option<AttentionFnFamily>,
}

impl Connection {
    /// Construct a binary, temporally-unconstrained connection with no
    /// per-edge family override (the common case).
    #[inline]
    pub const fn new(src: RegionId, dst: RegionId) -> Self {
        Self { src, dst, weight: 1.0, t_src: None, t_dst: None, fn_family: None }
    }

    /// Construct a weighted connection.
    #[inline]
    pub const fn weighted(src: RegionId, dst: RegionId, weight: f32) -> Self {
        Self { src, dst, weight, t_src: None, t_dst: None, fn_family: None }
    }

    /// Whether this connection is "present" (weight > 0). Zero-weight edges are
    /// treated as absent by the mask builder.
    #[inline]
    pub fn is_present(self) -> bool {
        self.weight > 0.0
    }
}

/// The canvas geometry + region declarations.
#[derive(Clone)]
pub struct CanvasLayout {
    /// Temporal extent.
    pub t: u32,
    /// Height extent.
    pub h: u32,
    /// Width extent.
    pub w: u32,
    /// Latent channel count per position.
    pub d_model: u32,
    /// Region declarations. Order defines [`RegionId`] indexing.
    pub regions: Vec<RegionSpec>,
}

impl core::fmt::Debug for CanvasLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CanvasLayout")
            .field("t", &self.t)
            .field("h", &self.h)
            .field("w", &self.w)
            .field("d_model", &self.d_model)
            .field("n_regions", &self.regions.len())
            .finish()
    }
}

impl CanvasLayout {
    /// Total number of latent positions in the canvas: `T · H · W`.
    ///
    /// This is the row/column count of the compiled attention mask. Regions
    /// partition (a subset of) these positions.
    #[inline]
    pub fn n_positions(&self) -> usize {
        (self.t as usize) * (self.h as usize) * (self.w as usize)
    }

    /// Number of declared regions.
    #[inline]
    pub fn n_regions(&self) -> usize {
        self.regions.len()
    }
}

/// The directed connection graph between regions.
#[derive(Debug, Clone, Default)]
pub struct CanvasTopology {
    /// The edges. Order is declaration order; the compiler does not sort.
    pub connections: Vec<Connection>,
}

impl CanvasTopology {
    /// Construct an empty topology (no connections — fully isolated regions).
    #[inline]
    pub fn new() -> Self {
        Self { connections: Vec::new() }
    }

    /// Construct a topology from a vector of connections.
    #[inline]
    pub fn from_connections(connections: Vec<Connection>) -> Self {
        Self { connections }
    }

    /// Number of edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.connections.len()
    }

    /// Add a single connection.
    #[inline]
    pub fn add(&mut self, c: Connection) {
        self.connections.push(c);
    }
}

/// A declared canvas: geometry + regions + the causal topology between them.
#[derive(Clone)]
pub struct CanvasSchema {
    /// Geometry + region declarations.
    pub layout: CanvasLayout,
    /// Directed connection graph.
    pub topology: CanvasTopology,
}

/// The compiled output of [`super::compile_schema`].
///
/// Carries the per-region flat-position index ranges, the sparse attention
/// mask, and the per-position loss weight mask. All allocations happen here,
/// at schema-load time; downstream queries ([`super::can_reach`]) are alloc-free.
#[derive(Debug, Clone)]
pub struct CompiledCanvas {
    /// For each region, the contiguous range of flat positions it occupies.
    /// `region_indices[r]` = `[start, end)` into the flat `[0, N)` position
    /// index where `N = layout.n_positions()`.
    pub region_indices: Vec<Range<usize>>,
    /// The sparse attention mask (paper §2.3 `M ∈ R^{N×N}_{≥0}`, sparse form).
    pub mask: AttentionMaskSpec,
    /// Per-position loss weight `ω_i` (paper §2.3).
    pub loss_mask: LossWeightMask,
}

/// Sparse representation of the compiled attention mask `M ∈ R^{N×N}_{≥0}`.
///
/// This is **not** a dense matrix. Each edge `(i, j, w)` means "query position
/// `i` may attend to key position `j` with weight `w`". Consumers lower this
/// into whatever dense / blocked / bit-packed form their attention kernel
/// needs (e.g. AC-Prefix's bit-packed [`crate::ac_prefix::AcPrefixMask`]).
///
/// For binary masks (`w ∈ {0, 1}`) the reachability guarantee holds: an absent
/// edge ⟹ exact marginal independence.
#[derive(Debug, Clone, Default)]
pub struct AttentionMaskSpec {
    /// Row/column extent `N = layout.n_positions()`.
    pub n_positions: usize,
    /// `(query_i, key_j, weight)` triples, in declaration order.
    pub edges: Vec<(usize, usize, f32)>,
}

impl AttentionMaskSpec {
    /// Number of (present) edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Per-position loss weight vector `ω ∈ R^N` (paper §2.3).
///
/// `ω_i = Σ_r 1[i ∈ I_r] · loss_weight_r · 1[is_output_r]`. Non-output regions
/// contribute nothing; positions outside all regions get zero.
#[derive(Debug, Clone, Default)]
pub struct LossWeightMask {
    /// Length `N = layout.n_positions()`.
    pub weights: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_volume_is_product_of_extents() {
        // 2 frames × 3 rows × 4 cols = 24 positions.
        let b = CanvasBounds::new(0, 2, 0, 3, 0, 4);
        assert_eq!(b.volume(), 24);
        assert_eq!(b.dt(), 2);
        assert_eq!(b.dh(), 3);
        assert_eq!(b.dw(), 4);
    }

    #[test]
    fn degenerate_bounds_have_zero_volume() {
        assert_eq!(CanvasBounds::new(0, 0, 0, 3, 0, 4).volume(), 0);
        assert_eq!(CanvasBounds::new(0, 2, 1, 1, 0, 4).volume(), 0);
        assert_eq!(CanvasBounds::new(0, 2, 0, 3, 5, 5).volume(), 0);
    }

    #[test]
    fn region_id_round_trips() {
        let r = RegionId::new(7);
        assert_eq!(r.get(), 7);
        assert_eq!(r, RegionId::from(7));
    }

    #[test]
    fn basis_vectors_are_orthogonal_or_identical() {
        let a = SemanticType::basis("camera", 0);
        let b = SemanticType::basis("joints", 1);
        let a2 = SemanticType::basis("camera2", 0);
        // Dot products: same-axis → 1, diff-axis → 0.
        let dot_same: f32 =
            a.frozen_embedding.iter().zip(a2.frozen_embedding.iter()).map(|(x, y)| x * y).sum();
        let dot_diff: f32 =
            a.frozen_embedding.iter().zip(b.frozen_embedding.iter()).map(|(x, y)| x * y).sum();
        assert!((dot_same - 1.0).abs() < 1e-6);
        assert!(dot_diff.abs() < 1e-6);
    }

    #[test]
    fn semantic_type_debug_does_not_dump_array() {
        let s = SemanticType::basis("x", 0);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("x"));
        assert!(!dbg.contains("0.0, 0.0"));
    }

    #[test]
    fn connection_is_present_respects_weight() {
        assert!(Connection::new(RegionId(0), RegionId(1)).is_present());
        assert!(Connection::weighted(RegionId(0), RegionId(1), 0.5).is_present());
        assert!(!Connection::weighted(RegionId(0), RegionId(1), 0.0).is_present());
        assert!(!Connection::weighted(RegionId(0), RegionId(1), -1.0).is_present());
    }

    #[test]
    fn layout_n_positions_is_product() {
        let layout = CanvasLayout { t: 2, h: 3, w: 4, d_model: 8, regions: vec![] };
        assert_eq!(layout.n_positions(), 24);
        assert_eq!(layout.n_regions(), 0);
    }

    #[test]
    fn topology_default_is_empty() {
        let t = CanvasTopology::default();
        assert_eq!(t.n_edges(), 0);
    }

    #[test]
    fn repr_u8_attention_family_is_one_byte() {
        assert_eq!(core::mem::size_of::<AttentionFnFamily>(), 1);
    }
}
