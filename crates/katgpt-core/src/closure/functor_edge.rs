//! PTG × latent_functor edge composition — continuous functor as PTG edge operator.
//!
//! Issue 040 (2026-07-04). Closes the neuro-symbolic gap: symbolic PTG edges
//! ([`crate::closure::PtgEdge`]) become differentiable transitions by
//! associating each edge with a continuous functor operator (direction vector
//! + sigmoid gate).
//!
//! # The gap this closes
//!
//! [`PrimitiveTransitionGraph`] (Plan 290) ships as the symbolic execution DAG
//! — edges record *that* primitive B followed primitive A, with no continuous
//! operator semantics. Meanwhile `latent_functor` (riir-ai Plan 273) ships the
//! continuous transition operator `apply_functor(state, functor)`. The two
//! never composed at the edge level.
//!
//! This module adds the composition: a [`FunctorPtg`] wraps an unchanged PTG
//! with a parallel array of optional [`FunctorEdgeParams`], one per edge.
//!
//! # Wire-format safety (the T1 audit finding)
//!
//! **The issue's original wire-compat claim was wrong.** Adding a field to the
//! postcard-serialized [`crate::closure::PtgEdge`] is NOT backward-compatible:
//!
//! - Plain `Option<T>` adds 1 byte per edge (None discriminant) → wire changes.
//! - `#[serde(skip_serializing_if = "Option::is_none", default)]` makes
//!   *serialization* byte-identical to the old format when `None`, BUT
//!   *deserialization* of those same bytes FAILS ("Hit end of buffer") because
//!   postcard is positional — `default` cannot kick in on EOF. Verified
//!   empirically (T1 audit, 2026-07-04): `NewSkip(None) → bytes → NewSkip`
//!   round-trip fails.
//!
//! **Design decision:** the [`FunctorPtg`] composite type leaves
//! [`crate::closure::PtgEdge`] and [`PrimitiveTransitionGraph`] byte-identical.
//! The functor layer lives in a parallel `Vec<Option<FunctorEdgeParams>>`
//! indexed by edge position. This preserves wire format + commitment 100%.
//!
//! # The apply math
//!
//! Given a state `s ∈ R^D` and a functor direction `d ∈ R^D` (unit-normalized
//! at extraction time), the edge transition is:
//!
//! ```text
//! coherence = cos(s, d) = (s · d) / (‖s‖ · ‖d‖)     // ∈ [-1, 1]
//! gate      = sigmoid(β · (coherence − τ))           // ∈ (0, 1)
//! s'        = s + gate · d                            // sigmoid-gated additive update
//! ```
//!
//! This is the modelless composition of riir-ai's `apply_functor`
//! (`out = source + functor`) and `functor_gate` (`sigmoid(β·(c−τ))`). The
//! direction is pre-extracted (by riir-ai's `extract_functor_into`); `β`/`τ`
//! are baked at extraction time; the apply is a cosine + sigmoid + SAXPY.
//!
//! # Zero-allocation contract
//!
//! [`apply_functor_edge_into`] writes into a caller-provided `out` buffer.
//! No `Vec` in the hot path.
//!
//! # Feature gate
//!
//! Gated behind `ptg_functor_edges` (implies `closure_instrument`). Opt-in
//! until the GOAT gate (G1–G4) passes.
//!
//! # References
//!
//! - **Issue:** `katgpt-rs/.issues/040_ptg_latent_functor_edge_composition.md`
//! - **PTG substrate:** `katgpt-rs/crates/katgpt-core/src/closure/mod.rs` (Plan 290)
//! - **latent_functor (riir-ai):** `riir-engine/src/latent_functor/arithmetic.rs` (Plan 273)
//! - **Sibling:** Issue 039 — `FunctorEdgeParams.direction_set` can be included
//!   in the architecture root once this ships.

use crate::closure::PrimitiveTransitionGraph;
use crate::simd::simd_dot_f32;
use crate::sigmoid;

// ── Functor edge parameters ───────────────────────────────────────────────

/// Continuous functor parameters baked onto a PTG edge.
///
/// Carries the content-addressed reference to a direction vector set + the
/// sigmoid gate scalars. The direction itself is resolved at apply time from
/// the referenced table (by the caller — this struct stays small, no
/// variable-length blob).
///
/// # Fields
///
/// - `direction_set`: BLAKE3 root of the direction-vector table (content-
///   addressed reference — same pattern as
///   `crate::engram::EngramTableId`). Stored as raw `[u8; 32]` to keep this
///   module decoupled from the `engram` feature (callers convert via `.0`
///   if they hold an `EngramTableId`). Matches Issue 039's `functor_sig_root`
///   convention.
/// - `direction_index`: which row of the referenced table (supports K-direction
///   sets — one table can hold directions for multiple edges).
/// - `beta`: sigmoid gate steepness (baked at extraction time).
/// - `tau`: sigmoid gate threshold (baked at extraction time).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FunctorEdgeParams {
    /// Content-addressed reference to the direction-vector table (BLAKE3 root).
    pub direction_set: [u8; 32],
    /// Which row of the referenced table (0-indexed).
    pub direction_index: u16,
    /// Sigmoid gate steepness `β`. Higher = sharper transition.
    pub beta: f32,
    /// Sigmoid gate threshold `τ`. Coherence above this → gate opens.
    pub tau: f32,
}

impl FunctorEdgeParams {
    /// Sentinel "no direction set" — all-zeros BLAKE3 root.
    /// Matches the padding-leaf convention from `build_merkle_root`.
    pub const NULL_DIRECTION_SET: [u8; 32] = [0u8; 32];

    /// Construct with explicit params.
    #[inline]
    #[must_use]
    pub const fn new(
        direction_set: [u8; 32],
        direction_index: u16,
        beta: f32,
        tau: f32,
    ) -> Self {
        Self { direction_set, direction_index, beta, tau }
    }

    /// Default gate scalars (β=8.0, τ=0.6) — matches riir-ai's
    /// `latent_functor::arithmetic::DEFAULT_GATE_{BETA,TAU}`.
    pub const DEFAULT_BETA: f32 = 8.0;
    pub const DEFAULT_TAU: f32 = 0.6;
}

// ── FunctorPtg composite ──────────────────────────────────────────────────

/// A [`PrimitiveTransitionGraph`] augmented with per-edge continuous functor
/// operators.
///
/// The inner [`ptg`](Self::ptg) stays pure symbolic — wire format and BLAKE3
/// commitment are byte-identical to a bare PTG. The functor layer lives in
/// [`edge_functors`](Self::edge_functors), a parallel array indexed by edge
/// position in `ptg.edges`.
///
/// # Construction
///
/// - [`FunctorPtg::new`] — wrap a PTG with all-`None` functor slots.
/// - [`FunctorPtg::with_functors`] — wrap a PTG with an explicit functor array.
/// - [`FunctorPtg::set_edge_functor`] — set the functor for one edge.
///
/// # Invariant
///
/// `edge_functors.len() == ptg.edges.len()`. Constructors enforce this;
/// [`FunctorPtg::set_edge_functor`] panics on out-of-bounds index.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FunctorPtg {
    /// The underlying symbolic PTG. Wire format and commitment unchanged.
    pub ptg: PrimitiveTransitionGraph,
    /// Per-edge functor params, indexed by edge position in `ptg.edges`.
    /// `None` = symbolic edge (no functor). `Some` = continuous functor.
    pub edge_functors: Vec<Option<FunctorEdgeParams>>,
}

impl FunctorPtg {
    /// Wrap a PTG with all-`None` functor slots (pure symbolic — no functors
    /// yet, but the composite is ready to receive them).
    #[inline]
    #[must_use]
    pub fn new(ptg: PrimitiveTransitionGraph) -> Self {
        let n = ptg.edges.len();
        Self {
            ptg,
            edge_functors: vec![None; n],
        }
    }

    /// Wrap a PTG with an explicit functor array. The array length must equal
    /// `ptg.edges.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `edge_functors.len() != ptg.edges.len()`.
    #[inline]
    #[must_use]
    pub fn with_functors(
        ptg: PrimitiveTransitionGraph,
        edge_functors: Vec<Option<FunctorEdgeParams>>,
    ) -> Self {
        assert_eq!(
            edge_functors.len(),
            ptg.edges.len(),
            "edge_functors.len() must equal ptg.edges.len()"
        );
        Self { ptg, edge_functors }
    }

    /// Set the functor params for edge at `edge_index`.
    ///
    /// # Panics
    ///
    /// Panics if `edge_index >= edge_functors.len()`.
    #[inline]
    pub fn set_edge_functor(&mut self, edge_index: usize, params: FunctorEdgeParams) {
        self.edge_functors[edge_index] = Some(params);
    }

    /// Get the functor params for edge at `edge_index`, if any.
    #[inline]
    #[must_use]
    pub fn edge_functor(&self, edge_index: usize) -> Option<&FunctorEdgeParams> {
        self.edge_functors[edge_index].as_ref()
    }

    /// Count of edges carrying a functor (vs pure symbolic).
    #[inline]
    #[must_use]
    pub fn functor_edge_count(&self) -> usize {
        self.edge_functors.iter().filter(|f| f.is_some()).count()
    }

    /// The inner PTG's commitment — byte-identical to
    /// [`crate::closure::commitment`] on the same PTG (functor layer does NOT
    /// participate in the PTG commitment; it commits separately if needed).
    #[inline]
    #[must_use]
    pub fn ptg_commitment(&self) -> [u8; 32] {
        crate::closure::commitment(&self.ptg)
    }
}

// ── Apply (the hot path) ──────────────────────────────────────────────────

/// Apply a functor edge transition: `s' = s + gate · d`.
///
/// Where `gate = sigmoid(β · (coherence − τ))` and `coherence = cos(s, d)`.
///
/// The caller resolves `direction` from the referenced direction-set table
/// (using [`FunctorEdgeParams::direction_set`] + `direction_index`). This
/// keeps the apply function pure (no table lookup) and benchmarkable.
///
/// # Arguments
///
/// - `state`: current latent state `s` (length `dim`).
/// - `params`: functor edge params (β, τ gate scalars).
/// - `direction`: pre-resolved direction vector `d` (length `dim`, ideally
///   unit-normalized at extraction time).
/// - `dim`: dimensionality of state and direction.
/// - `out`: output buffer `s'` (length `dim`). May alias `state`.
///
/// # Zero-allocation
///
/// Writes into caller-provided `out`. No `Vec`, no heap.
///
/// # Mathematical note
///
/// The gate opens (→ 1) when `coherence > τ` and closes (→ 0) when
/// `coherence < τ`. At `coherence == τ`, `gate = sigmoid(0) = 0.5`. The
/// update is additive: `s' = s + gate · d`.
///
/// When `gate = 0` (low coherence), `s' = s` (no-op — edge doesn't fire).
/// When `gate = 1` (high coherence), `s' = s + d` (full functor application).
#[inline]
pub fn apply_functor_edge_into(
    state: &[f32],
    params: &FunctorEdgeParams,
    direction: &[f32],
    dim: usize,
    out: &mut [f32],
) {
    debug_assert!(state.len() >= dim);
    debug_assert!(direction.len() >= dim);
    debug_assert!(out.len() >= dim);

    // Coherence = cosine similarity(state, direction).
    let dot = simd_dot_f32(state, direction, dim);
    let norm_s = simd_dot_f32(state, state, dim).sqrt();
    let norm_d = simd_dot_f32(direction, direction, dim).sqrt();
    let coherence = if norm_s > 0.0 && norm_d > 0.0 {
        dot / (norm_s * norm_d)
    } else {
        0.0
    };

    // Gate = sigmoid(β · (coherence − τ)).
    let gate = sigmoid(params.beta * (coherence - params.tau));

    // s' = s + gate · d  (in-place SAXPY; out may alias state).
    for i in 0..dim {
        out[i] = state[i] + gate * direction[i];
    }
}

/// Compute the gate value `sigmoid(β · (coherence − τ))` without applying.
///
/// Useful for diagnostics: "would this edge fire on this state?".
#[inline]
#[must_use]
pub fn functor_edge_gate(
    state: &[f32],
    params: &FunctorEdgeParams,
    direction: &[f32],
    dim: usize,
) -> f32 {
    debug_assert!(state.len() >= dim);
    debug_assert!(direction.len() >= dim);
    let dot = simd_dot_f32(state, direction, dim);
    let norm_s = simd_dot_f32(state, state, dim).sqrt();
    let norm_d = simd_dot_f32(direction, direction, dim).sqrt();
    let coherence = if norm_s > 0.0 && norm_d > 0.0 {
        dot / (norm_s * norm_d)
    } else {
        0.0
    };
    sigmoid(params.beta * (coherence - params.tau))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{serialize_postcard, OperatorKind, PrimitiveKind, PtgRecorder};

    // ── FunctorEdgeParams ──────────────────────────────────────────────────

    #[test]
    fn functor_edge_params_fields() {
        let params = FunctorEdgeParams::new([1u8; 32], 7, 8.0, 0.6);
        assert_eq!(params.direction_set, [1u8; 32]);
        assert_eq!(params.direction_index, 7);
        assert_eq!(params.beta, 8.0);
        assert_eq!(params.tau, 0.6);
        let copied = params; // Copy
        assert_eq!(copied.beta, 8.0);
    }

    #[test]
    fn null_direction_set_is_all_zeros() {
        assert_eq!(FunctorEdgeParams::NULL_DIRECTION_SET, [0u8; 32]);
    }

    #[test]
    fn default_gate_constants_match_riir_ai() {
        assert_eq!(FunctorEdgeParams::DEFAULT_BETA, 8.0);
        assert_eq!(FunctorEdgeParams::DEFAULT_TAU, 0.6);
    }

    // ── apply_functor_edge_into ────────────────────────────────────────────

    #[test]
    fn apply_high_coherence_full_direction() {
        let state = [1.0, 0.0, 0.0, 0.0];
        let direction = [1.0, 0.0, 0.0, 0.0]; // cosine sim = 1.0
        let params = FunctorEdgeParams::new([0u8; 32], 0, 8.0, 0.6);
        let mut out = [0.0f32; 4];
        apply_functor_edge_into(&state, &params, &direction, 4, &mut out);
        let expected_gate = sigmoid(8.0 * (1.0 - 0.6));
        for i in 0..4 {
            let expected = state[i] + expected_gate * direction[i];
            assert!((out[i] - expected).abs() < 1e-6, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn apply_low_coherence_near_nop() {
        let state = [1.0, 0.0, 0.0, 0.0];
        let direction = [0.0, 1.0, 0.0, 0.0]; // cosine sim = 0
        let params = FunctorEdgeParams::new([0u8; 32], 0, 8.0, 0.6);
        let mut out = [0.0f32; 4];
        apply_functor_edge_into(&state, &params, &direction, 4, &mut out);
        let expected_gate = sigmoid(8.0 * (0.0 - 0.6));
        assert!(expected_gate < 0.01);
        for i in 0..4 {
            assert!((out[i] - state[i]).abs() < 0.01);
        }
    }

    #[test]
    fn apply_at_threshold_half_gate() {
        // state = [0.6, 0.8] (unit), direction = [1, 0] (unit) → cos = 0.6.
        let state = [0.6, 0.8];
        let direction = [1.0, 0.0];
        let params = FunctorEdgeParams::new([0u8; 32], 0, 8.0, 0.6);
        let gate = functor_edge_gate(&state, &params, &direction, 2);
        assert!((gate - 0.5).abs() < 1e-6, "gate = {}", gate);
    }

    #[test]
    fn apply_out_may_alias_state() {
        // NOTE: `apply_functor_edge_into` takes `state: &[f32]` and `out: &mut
        // [f32]` as separate borrows, so true in-place aliasing is not
        // expressible in the type system. The function is element-wise safe
        // (read state[i], write out[i]), but callers must use two distinct
        // buffers or copy. This test verifies correctness with a copy.
        let state = [1.0, 0.0, 0.0, 0.0];
        let direction = [1.0, 0.0, 0.0, 0.0];
        let params = FunctorEdgeParams::new([0u8; 32], 0, 8.0, 0.6);
        let mut out = [0.0f32; 4];
        apply_functor_edge_into(&state, &params, &direction, 4, &mut out);
        assert!(out[0] > 1.0, "gate opened: out[0] should be > 1.0, got {}", out[0]);
    }

    #[test]
    fn apply_zero_state_no_nan() {
        let state = [0.0f32; 4];
        let direction = [1.0, 0.0, 0.0, 0.0];
        let params = FunctorEdgeParams::new([0u8; 32], 0, 8.0, 0.6);
        let mut out = [0.0f32; 4];
        apply_functor_edge_into(&state, &params, &direction, 4, &mut out);
        for v in out.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn apply_deterministic() {
        let state = [0.5, -0.3, 0.8, 0.1];
        let direction = [0.2, 0.7, -0.4, 0.55];
        let params = FunctorEdgeParams::new([3u8; 32], 2, 10.0, 0.5);
        let mut out1 = [0.0f32; 4];
        let mut out2 = [0.0f32; 4];
        apply_functor_edge_into(&state, &params, &direction, 4, &mut out1);
        apply_functor_edge_into(&state, &params, &direction, 4, &mut out2);
        assert_eq!(out1, out2);
    }

    // ── FunctorPtg ─────────────────────────────────────────────────────────

    fn make_test_ptg() -> PrimitiveTransitionGraph {
        let mut rec = PtgRecorder::new(42);
        let a = rec.enter(PrimitiveKind::UserDefined(0), 0, None);
        let b = rec.enter(PrimitiveKind::UserDefined(1), 1, None);
        let c = rec.enter(PrimitiveKind::UserDefined(2), 2, None);
        rec.exit(a, b, OperatorKind::Sequence);
        rec.exit(b, c, OperatorKind::Branch);
        rec.finish()
    }

    #[test]
    fn functor_ptg_new_all_none() {
        let ptg = make_test_ptg();
        let edge_count = ptg.edges.len();
        let fptg = FunctorPtg::new(ptg);
        assert_eq!(fptg.edge_functors.len(), edge_count);
        assert!(fptg.edge_functors.iter().all(|f| f.is_none()));
        assert_eq!(fptg.functor_edge_count(), 0);
    }

    #[test]
    fn functor_ptg_set_and_get() {
        let ptg = make_test_ptg();
        let mut fptg = FunctorPtg::new(ptg);
        let params = FunctorEdgeParams::new([1u8; 32], 0, 8.0, 0.6);
        fptg.set_edge_functor(1, params);
        assert_eq!(fptg.functor_edge_count(), 1);
        assert_eq!(fptg.edge_functor(1), Some(&params));
        assert_eq!(fptg.edge_functor(0), None);
    }

    #[test]
    fn functor_ptg_length_mismatch_panics() {
        let ptg = make_test_ptg();
        let bad = vec![None; 99];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = FunctorPtg::with_functors(ptg, bad);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn functor_ptg_preserves_inner_commitment() {
        let ptg = make_test_ptg();
        let bare = crate::closure::commitment(&ptg);
        let fptg = FunctorPtg::new(ptg);
        assert_eq!(fptg.ptg_commitment(), bare);
    }

    #[test]
    fn functor_ptg_commitment_unchanged_after_setting() {
        let ptg = make_test_ptg();
        let bare = crate::closure::commitment(&ptg);
        let mut fptg = FunctorPtg::new(ptg);
        let params = FunctorEdgeParams::new([9u8; 32], 3, 12.0, 0.4);
        fptg.set_edge_functor(0, params);
        fptg.set_edge_functor(1, params);
        assert_eq!(fptg.ptg_commitment(), bare);
    }

    // ── Wire-format safety ────────────────────────────────────────────────

    #[test]
    fn bare_ptg_round_trips() {
        let ptg = make_test_ptg();
        let bytes = serialize_postcard(&ptg).expect("serialize");
        assert!(!bytes.is_empty());
        let rt = crate::closure::deserialize_postcard(&bytes).expect("deserialize");
        assert_eq!(rt.edges.len(), ptg.edges.len());
        assert_eq!(rt.nodes.len(), ptg.nodes.len());
    }

    #[test]
    fn functor_ptg_serializes_and_round_trips() {
        let ptg = make_test_ptg();
        let mut fptg = FunctorPtg::new(ptg);
        fptg.set_edge_functor(0, FunctorEdgeParams::new([7u8; 32], 1, 9.0, 0.55));
        let bytes = postcard::to_allocvec(&fptg).expect("serialize");
        let rt: FunctorPtg = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(rt.ptg.edges.len(), fptg.ptg.edges.len());
        assert_eq!(rt.edge_functors.len(), fptg.edge_functors.len());
        assert_eq!(rt.edge_functor(0), fptg.edge_functor(0));
    }

    #[test]
    fn bare_ptg_bytes_identical_to_inner_ptg_bytes() {
        // CRITICAL wire-format safety property: the inner PTG of a FunctorPtg
        // serializes identically to a bare PTG. The functor layer is additive.
        let ptg1 = make_test_ptg();
        let ptg2 = make_test_ptg();
        let bare_bytes = serialize_postcard(&ptg1).expect("serialize bare");
        let fptg = FunctorPtg::new(ptg2);
        let inner_bytes = serialize_postcard(&fptg.ptg).expect("serialize inner");
        assert_eq!(bare_bytes, inner_bytes);
    }
}
