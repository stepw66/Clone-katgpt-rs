//! Intersection-over-Union (IoU) cheap-proxy for attention head substitution.
//!
//! Implements paper eq. 3 from [arXiv:2606.19317](https://arxiv.org/abs/2606.19317)
//! (Hayes, Li, Andreas — *Explaining Attention with Program Synthesis*, MIT
//! CSAIL / NJIT, 30 Jun 2026):
//!
//! ```text
//! IoU(a, b) = Σ_i min(a_i, b_i) / Σ_i max(a_i, b_i)
//! ```
//!
//! Applied to a pair of attention probability rows (real head vs surrogate),
//! this is the cheap proxy that the paper §3 Fig 5b shows correlates with
//! expensive causal substitution cost at Pearson `r > 0.9`. The
//! [`HeadSubstitutionGate`](super::gate::HeadSubstitutionGate) uses IoU as the
//! first-stage gate before the cached FaithfulnessProbe veto.
//!
//! # Performance contract
//!
//! - Zero heap allocation.
//! - SIMD-friendly chunked 8-wide loop (auto-vectorizes on x86-64 AVX2 and
//!   aarch64 NEON). Tail handled by a scalar remainder loop.
//! - Empty-denominator case (both rows all-zero) returns `0.0` per the
//!   convention "no overlap when nothing attends".
//!
//! Inputs are interpreted as non-negative attention weights; the function does
//! NOT validate non-negativity (caller responsibility — gating the hot path
//! with `max()`-based clamping would defeat the SIMD throughput target).

/// Chunk size for the auto-vectorized accumulator loop. 8 × f32 fits one AVX2
/// / NEON vector lane and lets LLVM emit a single `vminps` / `vmaxps` pair per
/// iteration. Tunable downstream but kept as a `const` so the remainder
/// expression stays compile-time-evaluable.
const CHUNK: usize = 8;

/// Intersection-over-Union of two non-negative attention rows.
///
/// Paper eq. 3: `Σ min(a,b) / Σ max(a,b)`. Returns `0.0` if either slice is
/// empty or the denominator is zero (both rows all-zero — no overlap by
/// convention).
///
/// # Length contract
///
/// `a` and `b` must have equal length. If they don't, the function returns
/// `0.0` (defensive — attention rows in a real head are always the same
/// length, but a noisy caller shouldn't `panic!` on the hot path).
#[inline]
pub fn iou(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut sum_min: f32 = 0.0;
    let mut sum_max: f32 = 0.0;

    // Chunked 8-wide loop for auto-vectorization. LLVM lowers `a[i].min(b[i])`
    // / `a[i].max(b[i])` to `vminps` / `vmaxps` on AVX2 and to `fmin` / `fmax`
    // intrinsics on NEON. The accumulator is a scalar f32 (not an array) so
    // LLVM can keep it in a single xmm register and fold the horizontal add.
    let chunks = n / CHUNK;
    let main_len = chunks * CHUNK;
    let mut i = 0;
    while i < main_len {
        // Unrolled inner body — let the optimizer widen as the target allows.
        // Manual unroll-by-8 keeps the loop branch-free even on targets where
        // auto-vectorization is disabled.
        let mut local_min = 0.0f32;
        let mut local_max = 0.0f32;
        let mut k = 0;
        while k < CHUNK {
            let ai = a[i + k];
            let bi = b[i + k];
            local_min += ai.min(bi);
            local_max += ai.max(bi);
            k += 1;
        }
        sum_min += local_min;
        sum_max += local_max;
        i += CHUNK;
    }

    // Scalar tail.
    while i < n {
        let ai = a[i];
        let bi = b[i];
        sum_min += ai.min(bi);
        sum_max += ai.max(bi);
        i += 1;
    }

    if sum_max <= 0.0 {
        // Both rows all-zero (or pathological negative-input case the caller
        // contract disallows). Treat as no-overlap, not division-by-zero.
        return 0.0;
    }
    sum_min / sum_max
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests — G1 hand-computed cases (Plan 353 T2.1 / T1.3)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed: identity → IoU = 1.0.
    #[test]
    fn iou_identity_is_one() {
        let a = [0.5f32, 0.3, 0.2];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    /// Hand-computed: disjoint supports → IoU = 0.0.
    #[test]
    fn iou_disjoint_is_zero() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 1.0, 0.0];
        assert!(iou(&a, &b).abs() < 1e-6);
    }

    /// Hand-computed half-overlap: `a = [1, 1, 0, 0]`, `b = [1, 0, 1, 0]`.
    /// Σ min = 1, Σ max = 3 → IoU = 1/3.
    #[test]
    fn iou_partial_overlap_known_value() {
        let a = [1.0f32, 1.0, 0.0, 0.0];
        let b = [1.0f32, 0.0, 1.0, 0.0];
        let got = iou(&a, &b);
        assert!((got - 1.0 / 3.0).abs() < 1e-6, "got {got}");
    }

    /// Length-mismatch returns 0.0 (defensive, no panic).
    #[test]
    fn iou_length_mismatch_returns_zero() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0];
        assert_eq!(iou(&a, &b), 0.0);
    }

    /// Both-empty returns 0.0 (empty denominator case).
    #[test]
    fn iou_both_empty_returns_zero() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(iou(&a, &b), 0.0);
    }

    /// Both all-zero returns 0.0 (empty denominator case).
    #[test]
    fn iou_both_all_zero_returns_zero() {
        let a = [0.0f32; 8];
        let b = [0.0f32; 8];
        assert_eq!(iou(&a, &b), 0.0);
    }

    /// Tail handling: length 11 (8 + 3 remainder) must match a reference fold.
    #[test]
    fn iou_chunked_tail_matches_scalar() {
        // Deterministic LCG for reproducibility.
        let mut state = 0xBEEFu64;
        let mut lcg = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f32
        };
        let n = 11;
        let a: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let b: Vec<f32> = (0..n).map(|_| lcg()).collect();

        let chunked = iou(&a, &b);

        // Scalar reference.
        let mut s_min = 0.0f32;
        let mut s_max = 0.0f32;
        for (ai, bi) in a.iter().zip(b.iter()) {
            s_min += ai.min(*bi);
            s_max += ai.max(*bi);
        }
        let reference = if s_max > 0.0 { s_min / s_max } else { 0.0 };

        assert!(
            (chunked - reference).abs() < 1e-6,
            "chunked {chunked} vs ref {reference}"
        );
    }

    /// Symmetry: `iou(a,b) == iou(b,a)`.
    #[test]
    fn iou_is_symmetric() {
        let a = [0.1f32, 0.4, 0.3, 0.2];
        let b = [0.3f32, 0.1, 0.5, 0.1];
        assert!((iou(&a, &b) - iou(&b, &a)).abs() < 1e-6);
    }
}
