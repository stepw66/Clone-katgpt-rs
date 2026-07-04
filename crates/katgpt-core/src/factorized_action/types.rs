//! Core data types for the Factorized Transition Action Abstraction primitive.
//!
//! These types are pure data — no behavior beyond lookup — so the
//! modelless factorization/gate/aggregate functions in `kernel.rs` and
//! `codebook.rs` can operate on references without coupling to the
//! concrete codebook contents.
//!
//! # Const generics
//!
//! - `K` — codebook size (number of effect primitives).
//! - `D` — per-primitive latent dimension (each centroid is `D` floats).
//! - `S` — state-vector dimension for the FiLM relevance gate.
//!
//! Paper defaults (verified from `otf_vqvae/default_config.yaml`):
//! `K = 128`, `D = 32`.

use bytemuck::{Pod, Zeroable};

/// Maximum number of patches supported in a single transition
/// factorization. The `assignments` array is sized for the worst case so
/// no allocation is needed on the hot path.
pub const MAX_PATCHES: usize = 64;

/// Maximum codebook size K supported by `TransitionFactors::weights`.
/// Must be ≥ the largest codebook K used by any caller. Paper default
/// is K=128; we round up to 256 for headroom.
pub const MAX_K: usize = 256;

/// Row-major codebook storage of shape `K × D`.
///
/// We store the codebook as a flat `[[f32; D]; K]` (nested arrays) so
/// `bytemuck::Pod` can be derived via the auto-impl for arrays-of-Pod
/// arrays. Const-generic array sizing `[f32; K*D]` would require
/// `feature(generic_const_exprs)` and isn't available on stable Rust.
///
/// Layout guarantee: `size_of::<EffectCodebook<K,D>>() == K * D * 4`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EffectCodebook<const K: usize, const D: usize> {
    /// Nested-array centroid table. `centroids[k]` is the D-dim effect
    /// primitive k. Row-major in memory.
    pub centroids: [[f32; D]; K],
}

// SAFETY: `EffectCodebook` is `#[repr(C)]` and contains only `[[f32; D]; K]`,
// which is itself Pod (f32 is Pod, arrays of Pod are Pod). No padding, no
// uninitialized bytes. This manual impl is required because `derive(Pod)`
// can't handle the nested-array shape with const generics on stable Rust.
unsafe impl<const K: usize, const D: usize> Pod for EffectCodebook<K, D> {}
unsafe impl<const K: usize, const D: usize> Zeroable for EffectCodebook<K, D> {}

impl<const K: usize, const D: usize> EffectCodebook<K, D> {
    /// Read centroid `k` as a `&[f32]` slice (zero-copy view).
    #[inline]
    pub fn centroid(&self, k: usize) -> &[f32] {
        debug_assert!(k < K, "centroid index {k} out of range K={K}");
        &self.centroids[k]
    }

    /// Read centroid `k` as a fixed-size array reference (zero-copy).
    #[inline]
    pub fn centroid_arr(&self, k: usize) -> &[f32; D] {
        debug_assert!(k < K, "centroid index {k} out of range K={K}");
        &self.centroids[k]
    }

    /// Construct an all-zero codebook (used as a placeholder before k-means fit).
    pub const fn zeroed() -> Self {
        Self {
            centroids: [[0.0; D]; K],
        }
    }
}

impl<const K: usize, const D: usize> Default for EffectCodebook<K, D> {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Per-transition factorization output.
///
/// Holds:
/// - `assignments` — Top-1 nearest-neighbor code index for each patch.
/// - `weights` — Per-code activation strength (raw counts in
///   `[0, n_patches]`; finalize divides by `n_patches` to normalize).
/// - `n_active` — Number of distinct codes with non-zero occupancy.
///
/// Designed to be allocated once per transition and reused across calls —
/// every field is a fixed-size array, no allocation in the hot path.
///
/// `assignments` is sized `MAX_PATCHES` (patches per transition).
/// `weights` is sized `MAX_K` (codes per codebook). The two are decoupled
/// because K can be larger than the patch count (paper default K=128,
/// patch count = 16).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TransitionFactors {
    /// `assignments[patch_idx] = k*` (Top-1 code index). `u16` because the
    /// codebook is bounded by `K ≤ MAX_K` (paper default K=128).
    pub assignments: [u16; MAX_PATCHES],
    /// Per-code raw activation count (pre-normalization) or activation
    /// strength (post-normalization). Only the first K entries are used.
    pub weights: [f32; MAX_K],
    /// Number of patches used to fill `assignments` (≤ MAX_PATCHES).
    pub n_patches: usize,
    /// Number of distinct codes with non-zero occupancy.
    pub n_active: usize,
}

impl TransitionFactors {
    /// Construct an all-zero factorization buffer.
    pub fn zeroed() -> Self {
        Self {
            assignments: [0; MAX_PATCHES],
            weights: [0.0; MAX_K],
            n_patches: 0,
            n_active: 0,
        }
    }

    /// Reset the buffer to all-zero state — cheaper than reconstructing
    /// when reusing across calls (zero allocation, just `memset`).
    pub fn reset(&mut self) {
        self.assignments = [0; MAX_PATCHES];
        self.weights = [0.0; MAX_K];
        self.n_patches = 0;
        self.n_active = 0;
    }
}

impl Default for TransitionFactors {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Aggregated factorized action latent.
///
/// The D-dim weighted average over state-conditioned factor tokens.
/// Latent by construction — only its scalar projections cross the sync
/// boundary, same discipline as `latent_functor`.
#[repr(transparent)]
pub struct FactorizedActionLatent<const D: usize>(pub [f32; D]);

// SAFETY: `repr(transparent)` over `[f32; D]`, which is Pod for any D
// (f32 is Pod; arrays-of-Pod are Pod by bytemuck's blanket impl). Manual
// impl needed because derive can't handle the const-generic array shape.
unsafe impl<const D: usize> Pod for FactorizedActionLatent<D> {}
unsafe impl<const D: usize> Zeroable for FactorizedActionLatent<D> {}

impl<const D: usize> Clone for FactorizedActionLatent<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const D: usize> Copy for FactorizedActionLatent<D> {}

impl<const D: usize> std::fmt::Debug for FactorizedActionLatent<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FactorizedActionLatent").field(&&self.0[..]).finish()
    }
}

impl<const D: usize> Default for FactorizedActionLatent<D> {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl<const D: usize> FactorizedActionLatent<D> {
    /// Read the latent as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Read the latent as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }

    /// Construct an all-zero latent.
    pub fn zeroed() -> Self {
        Self([0.0; D])
    }
}

/// Aggregation strategy for combining per-code factor tokens into the
/// action latent.
///
/// Verified against `otf_lam/model.py::aggregator_type`:
/// - `Gate` (default) — sigmoid relevance gate produces `α_k`, then
///   normalized weighted average. The full primitive.
/// - `Mean` — `α_k = 1` for all active codes (uniform weighted average,
///   no learned gate). The G2 ablation baseline proving the sigmoid gate
///   adds value over uniform aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum AggregatorType {
    /// Sigmoid relevance gate `α_k = sigmoid(β·(relevance(r_k) − τ))`.
    #[default]
    Gate = 0,
    /// Uniform `α_k = 1` (ablation).
    Mean = 1,
}

/// Frozen FiLM projection bank for the state-aware factor token.
///
/// For each code `k`, holds two S-length projection vectors `g_proj_k`
/// and `b_proj_k` used to compute state-conditioned scale/shift:
///
/// ```text
/// γ_k = dot(state, g_proj_k)     // scale  (state-conditioned)
/// β_k = dot(state, b_proj_k)     // shift  (state-conditioned)
/// r_k = (1 + γ_k) * c(k) + β_k   // FiLM-modulated codebook vector
/// ```
///
/// The projections are **frozen** — random orthonormal init at construction
/// time, never trained. This is the modelless analog of the paper's
/// learned 4-layer FiLM `GateNetwork`; we use a single deterministic
/// linear projection per code.
///
/// `#[repr(C)]` + Pod so it can be persisted and BLAKE3-committed
/// alongside the codebook (the consumer-side `EffectCodebookShard` in
/// riir-neuron-db will eventually carry both).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FilmProjectionBank<const K: usize, const D: usize, const S: usize> {
    /// Row-major `K × S` scale-projection table. `g_proj[k]` is the
    /// S-length scale-projection for code `k`.
    pub g_proj: [[f32; S]; K],
    /// Row-major `K × S` shift-projection table. `b_proj[k]` is the
    /// S-length shift-projection for code `k`.
    pub b_proj: [[f32; S]; K],
}

unsafe impl<const K: usize, const D: usize, const S: usize> Pod for FilmProjectionBank<K, D, S> {}
unsafe impl<const K: usize, const D: usize, const S: usize> Zeroable
    for FilmProjectionBank<K, D, S>
{}

impl<const K: usize, const D: usize, const S: usize> FilmProjectionBank<K, D, S> {
    /// Construct an identity-style bank: scale proj = 0 (γ=0 → no scaling),
    /// shift proj = 0 (β=0 → no shift). Factor token reduces to `c(k)`.
    pub fn zeroed() -> Self {
        Self {
            g_proj: [[0.0; S]; K],
            b_proj: [[0.0; S]; K],
        }
    }

    /// Scale projection for code `k` — `&[f32; S]` slice.
    #[inline]
    pub fn g_proj_slice(&self, k: usize) -> &[f32] {
        debug_assert!(k < K, "g_proj index {k} out of range K={K}");
        &self.g_proj[k]
    }

    /// Shift projection for code `k` — `&[f32; S]` slice.
    #[inline]
    pub fn b_proj_slice(&self, k: usize) -> &[f32] {
        debug_assert!(k < K, "b_proj index {k} out of range K={K}");
        &self.b_proj[k]
    }
}

impl<const K: usize, const D: usize, const S: usize> Default for FilmProjectionBank<K, D, S> {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_zeroed_reads_zero_centroids() {
        let cb: EffectCodebook<4, 8> = EffectCodebook::zeroed();
        assert_eq!(cb.centroid(0), &[0.0; 8]);
        assert_eq!(cb.centroid(3), &[0.0; 8]);
    }

    #[test]
    fn codebook_centroid_arr_matches_slice() {
        let mut cb: EffectCodebook<4, 8> = EffectCodebook::zeroed();
        cb.centroids[1] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(cb.centroid(1), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cb.centroid_arr(1), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn transition_factors_reset_round_trips() {
        let mut f = TransitionFactors::zeroed();
        f.assignments[0] = 3;
        f.weights[0] = 1.5;
        f.n_patches = 1;
        f.n_active = 1;
        f.reset();
        assert_eq!(f.assignments, [0; MAX_PATCHES]);
        assert_eq!(f.weights, [0.0; MAX_K]);
        assert_eq!(f.n_patches, 0);
        assert_eq!(f.n_active, 0);
    }

    #[test]
    fn aggregator_default_is_gate() {
        assert_eq!(AggregatorType::default(), AggregatorType::Gate);
    }

    #[test]
    fn action_latent_zeroed_reads_zero() {
        let l: FactorizedActionLatent<8> = FactorizedActionLatent::zeroed();
        assert_eq!(l.as_slice(), &[0.0; 8]);
    }

    #[test]
    fn film_bank_zeroed_is_identity_film() {
        // γ=0, β=0 → factor token = (1+0)*c(k) + 0 = c(k).
        let bank: FilmProjectionBank<4, 8, 4> = FilmProjectionBank::zeroed();
        assert_eq!(bank.g_proj_slice(0), &[0.0; 4]);
        assert_eq!(bank.b_proj_slice(0), &[0.0; 4]);
    }

    #[test]
    fn codebook_pod_byte_layout() {
        // EffectCodebook<K=4, D=8> must be 4*8*4 = 128 bytes, no padding.
        assert_eq!(std::mem::size_of::<EffectCodebook<4, 8>>(), 4 * 8 * 4);
        assert_eq!(std::mem::size_of::<EffectCodebook<128, 32>>(), 128 * 32 * 4);
    }

    #[test]
    fn action_latent_pod_byte_layout() {
        assert_eq!(std::mem::size_of::<FactorizedActionLatent<8>>(), 8 * 4);
        assert_eq!(std::mem::size_of::<FactorizedActionLatent<32>>(), 32 * 4);
    }
}
