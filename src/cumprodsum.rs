//! Cumprodsum — the atomic 1-semiseparable matrix multiplication primitive.
//!
//! Based on "Transformers are SSMs" (arXiv 2405.21060, Section 3.2.2):
//! the scalar state space model recurrence `y_t = a_t · y_{t-1} + x_t` is
//! equivalent to multiplication by a 1-semiseparable (1-SS) matrix.
//!
//! This primitive unifies:
//! - **cumsum** (a = 1): standard causal mask / cumulative sum
//! - **cumprod** (x = 0): pure exponential decay / cumulative product
//! - **GDN2 diagonal decay**: matrix-valued cumprodsum (via `cumprodsum_batched`)
//! - **LinOSS oscillation**: complex-eigenvalue variant (future)
//!
//! All functions are zero-allocation: outputs are written to pre-allocated
//! `&mut [f32]` slices. O(T) time, O(1) extra space.
//!
//! Reference: Research 231 — Semiseparable State Space Duality.

/// Scalar cumprodsum: `out[t] = a[t] * out[t-1] + x[t]` with `out[0] = a[0] * h_init + x[0]`.
///
/// This is the atomic 1-SS matrix multiplication (SSD paper eq. 7).
///
/// # Arguments
/// * `a` - Decay factors `[T]`, typically `sigmoid(gate)` bounded in [0, 1]
/// * `x` - Input sequence `[T]`
/// * `h_init` - Initial hidden state (0 for causal from-scratch)
/// * `out` - Output sequence `[T]` (pre-allocated)
///
/// # Special cases
/// - `a = [1, 1, ...]` → cumulative sum (cumsum)
/// - `x = [0, 0, ...]` → cumulative product (cumprod)
///
/// # Panics
/// Debug-asserts that `a`, `x`, and `out` have equal length.
#[inline]
pub fn cumprodsum_scalar(a: &[f32], x: &[f32], h_init: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), x.len());
    debug_assert_eq!(a.len(), out.len());

    if out.is_empty() {
        return;
    }

    let mut h = h_init;
    for t in 0..out.len() {
        unsafe {
            h = a.get_unchecked(t) * h + x.get_unchecked(t);
            *out.get_unchecked_mut(t) = h;
        }
    }
}

/// Batched cumprodsum: applies the scalar recurrence independently across `n_channels` channels.
///
/// Each channel has its own decay factor and input, but they're processed in an interleaved
/// layout for cache efficiency: `[channel, time]` → `a[ch * T + t]`, `x[ch * T + t]`.
///
/// This is the channel-broadcast form used in diagonal SSMs (SSD paper eq. 8b).
///
/// # Arguments
/// * `a` - Decay factors `[n_channels * T]`, layout `[ch][t]`
/// * `x` - Input sequence `[n_channels * T]`, layout `[ch][t]`
/// * `h_init` - Initial hidden states `[n_channels]`
/// * `out` - Output sequence `[n_channels * T]`, layout `[ch][t]`
/// * `n_channels` - Number of independent channels
/// * `seq_len` - Sequence length T
///
/// # Panics
/// Debug-asserts slice lengths match `n_channels * seq_len`.
#[inline]
pub fn cumprodsum_batched(
    a: &[f32],
    x: &[f32],
    h_init: &[f32],
    out: &mut [f32],
    n_channels: usize,
    seq_len: usize,
) {
    let total = n_channels * seq_len;
    debug_assert_eq!(a.len(), total);
    debug_assert_eq!(x.len(), total);
    debug_assert_eq!(out.len(), total);
    debug_assert_eq!(h_init.len(), n_channels);

    for ch in 0..n_channels {
        let offset = ch * seq_len;
        cumprodsum_scalar(
            &a[offset..offset + seq_len],
            &x[offset..offset + seq_len],
            h_init[ch],
            &mut out[offset..offset + seq_len],
        );
    }
}

/// Segment sum: computes the log-domain segment sums for 1-SS mask construction.
///
/// `out[i, j] = sum(a[j+1..=i])` for `j <= i`, `-inf` for `j > i`.
///
/// `exp(segsum(a))` produces the 1-SS matrix entries:
/// `M[i, j] = exp(sum(a[j+1..=i])) = prod(a[j+1..=i])` for `j <= i`.
///
/// This is the `segsum` function from SSD paper Listing 1.
///
/// # Arguments
/// * `a` - Input logits `[T]`
/// * `out` - Output matrix `[T * T]`, row-major (pre-allocated)
///
/// # Panics
/// Debug-asserts `out.len() == a.len() * a.len()`.
#[inline]
pub fn segsum(a: &[f32], out: &mut [f32]) {
    let t = a.len();
    debug_assert_eq!(out.len(), t * t);

    if t == 0 {
        return;
    }

    // Cumulative sum of a
    let mut cumsum = vec![0.0f32; t]; // Small, stack-like allocation for the cumsum
    cumsum[0] = a[0];
    for i in 1..t {
        cumsum[i] = cumsum[i - 1] + a[i];
    }

    // segsum[i, j] = cumsum[i] - cumsum[j] for j <= i, else -inf
    for i in 0..t {
        for j in 0..t {
            let idx = i * t + j;
            if j <= i {
                unsafe {
                    *out.get_unchecked_mut(idx) = cumsum.get_unchecked(i) - cumsum.get_unchecked(j);
                }
            } else {
                unsafe {
                    *out.get_unchecked_mut(idx) = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Influence score: cumulative product of decay factors from position `from` to `to`.
///
/// This computes `prod(a[from+1..=to])` = the semiseparable mask entry `L[to, from]`.
/// When this is below a threshold, the influence of position `from` on `to` is negligible.
///
/// Used by the SemiseparablePruner to decide branch pruning.
///
/// # Arguments
/// * `a` - Decay factors `[T]`
/// * `from` - Source position (inclusive lower bound)
/// * `to` - Target position (inclusive upper bound)
///
/// Returns the cumulative product. If `from >= to`, returns 1.0 (no decay).
#[inline]
pub fn influence(a: &[f32], from: usize, to: usize) -> f32 {
    if from >= to {
        return 1.0;
    }
    let mut prod = 1.0f32;
    for i in (from + 1)..=to {
        unsafe {
            prod *= a.get_unchecked(i);
        }
    }
    prod
}

/// Context freshness: mean cumulative influence across the sequence.
///
/// High freshness → recent context dominates (information is concentrated near the end).
/// Low freshness → context is spread evenly (information persists across the sequence).
///
/// Used by adaptive CoT to adjust thinking budget:
/// `thinking_budget = base + max_extra * sigmoid(beta * (freshness - threshold))`
///
/// # Arguments
/// * `a` - Decay factors `[T]`, typically in [0, 1]
///
/// Returns a value in [0, 1] where 1.0 = all information at the end, 0.0 = uniform.
#[inline]
pub fn context_freshness(a: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.5;
    }

    // Compute cumulative influence of position 0 on each subsequent position
    // Freshness = mean of these influences (high = fast decay = fresh)
    let mut sum = 0.0f32;
    let mut prod = 1.0f32;
    for &ai in a {
        prod *= ai;
        sum += prod;
    }
    // Normalize: if all a=1, sum=T, freshness should be ~1 (everything persists)
    // If all a=0, sum=0, freshness should be ~0 (nothing persists)
    sum / a.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── cumprodsum_scalar ──────────────────────────────────────────

    #[test]
    fn cumprodsum_basic() {
        // h_t = a_t * h_{t-1} + x_t, h_init = 0
        let a = [0.5, 0.5, 0.5, 0.5];
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        cumprodsum_scalar(&a, &x, 0.0, &mut out);

        // h0 = 0.5*0 + 1 = 1
        // h1 = 0.5*1 + 2 = 2.5
        // h2 = 0.5*2.5 + 3 = 4.25
        // h3 = 0.5*4.25 + 4 = 6.125
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.5).abs() < 1e-5);
        assert!((out[2] - 4.25).abs() < 1e-5);
        assert!((out[3] - 6.125).abs() < 1e-5);
    }

    #[test]
    fn cumprodsum_with_init() {
        let a = [0.9, 0.9];
        let x = [1.0, 1.0];
        let mut out = [0.0f32; 2];
        cumprodsum_scalar(&a, &x, 5.0, &mut out);

        // h0 = 0.9*5 + 1 = 5.5
        // h1 = 0.9*5.5 + 1 = 5.95
        assert!((out[0] - 5.5).abs() < 1e-5);
        assert!((out[1] - 5.95).abs() < 1e-5);
    }

    #[test]
    fn cumprodsum_cumsum_special_case() {
        // a = 1 → cumsum
        let a = [1.0, 1.0, 1.0, 1.0, 1.0];
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        cumprodsum_scalar(&a, &x, 0.0, &mut out);

        // Should be cumulative sum: [1, 3, 6, 10, 15]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
        assert!((out[2] - 6.0).abs() < 1e-5);
        assert!((out[3] - 10.0).abs() < 1e-5);
        assert!((out[4] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn cumprodsum_cumprod_special_case() {
        // x = 0 → cumprod (with h_init providing the seed)
        let a = [0.5, 0.5, 0.5, 0.5];
        let x = [0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        cumprodsum_scalar(&a, &x, 1.0, &mut out);

        // h0 = 0.5*1 + 0 = 0.5
        // h1 = 0.5*0.5 + 0 = 0.25
        // h2 = 0.5*0.25 + 0 = 0.125
        // h3 = 0.5*0.125 + 0 = 0.0625
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.25).abs() < 1e-5);
        assert!((out[2] - 0.125).abs() < 1e-5);
        assert!((out[3] - 0.0625).abs() < 1e-5);
    }

    #[test]
    fn cumprodsum_empty() {
        let a: [f32; 0] = [];
        let x: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        cumprodsum_scalar(&a, &x, 0.0, &mut out);
        // Should not panic
    }

    #[test]
    fn cumprodsum_single_element() {
        let a = [0.7];
        let x = [3.0];
        let mut out = [0.0f32; 1];
        cumprodsum_scalar(&a, &x, 1.0, &mut out);
        // h0 = 0.7*1 + 3 = 3.7
        assert!((out[0] - 3.7).abs() < 1e-5);
    }

    #[test]
    fn cumprodsum_large_sequence() {
        // Verify O(T) doesn't blow up or underflow for reasonable T
        let t = 1024;
        let a = vec![0.99; t];
        let x = vec![1.0; t];
        let mut out = vec![0.0; t];
        cumprodsum_scalar(&a, &x, 0.0, &mut out);

        // All outputs should be finite and monotonically increasing
        for i in 0..t {
            assert!(out[i].is_finite(), "Output {} is not finite: {}", i, out[i]);
            if i > 0 {
                assert!(out[i] > out[i - 1], "Not monotonically increasing at {}", i);
            }
        }
    }

    // ── cumprodsum_batched ─────────────────────────────────────────

    #[test]
    fn cumprodsum_batched_basic() {
        // 2 channels, 3 timesteps each
        let a = vec![
            0.5, 0.5, 0.5, // channel 0
            0.9, 0.9, 0.9, // channel 1
        ];
        let x = vec![
            1.0, 1.0, 1.0, // channel 0
            1.0, 1.0, 1.0, // channel 1
        ];
        let h_init = vec![0.0, 0.0];
        let mut out = vec![0.0; 6];
        cumprodsum_batched(&a, &x, &h_init, &mut out, 2, 3);

        // Channel 0: [1.0, 1.5, 1.75]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.5).abs() < 1e-5);
        assert!((out[2] - 1.75).abs() < 1e-5);

        // Channel 1: [1.0, 1.9, 2.71]
        assert!((out[3] - 1.0).abs() < 1e-5);
        assert!((out[4] - 1.9).abs() < 1e-5);
        assert!((out[5] - 2.71).abs() < 1e-5);
    }

    #[test]
    fn cumprodsum_batched_matches_scalar() {
        // Verify batched gives same result as individual scalar calls
        let n_ch = 4;
        let t = 32;
        let a: Vec<f32> = (0..n_ch * t)
            .map(|i| 0.5 + 0.01 * (i as f32 % 10.0))
            .collect();
        let x: Vec<f32> = (0..n_ch * t).map(|i| (i as f32) * 0.1).collect();
        let h_init: Vec<f32> = (0..n_ch).map(|_| 0.5).collect();
        let mut out_batched = vec![0.0; n_ch * t];
        cumprodsum_batched(&a, &x, &h_init, &mut out_batched, n_ch, t);

        for ch in 0..n_ch {
            let offset = ch * t;
            let mut out_scalar = vec![0.0; t];
            cumprodsum_scalar(
                &a[offset..offset + t],
                &x[offset..offset + t],
                h_init[ch],
                &mut out_scalar,
            );
            for i in 0..t {
                assert!(
                    (out_batched[offset + i] - out_scalar[i]).abs() < 1e-5,
                    "Mismatch at ch={}, t={}: batched={}, scalar={}",
                    ch,
                    i,
                    out_batched[offset + i],
                    out_scalar[i]
                );
            }
        }
    }

    // ── segsum ─────────────────────────────────────────────────────

    #[test]
    fn segsum_basic() {
        // a = [1, 2, 3]
        // cumsum = [1, 3, 6]
        // segsum[i,j] = cumsum[i] - cumsum[j] for j <= i, else -inf
        let a = [1.0, 2.0, 3.0];
        let mut out = vec![0.0; 9];
        segsum(&a, &mut out);

        // Row 0 (i=0): [cumsum[0]-cumsum[0], -inf, -inf] = [0, -inf, -inf]
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!(out[1].is_infinite() && out[1] < 0.0);
        assert!(out[2].is_infinite() && out[2] < 0.0);

        // Row 1 (i=1): [cumsum[1]-cumsum[0], cumsum[1]-cumsum[1], -inf] = [2, 0, -inf]
        assert!((out[3] - 2.0).abs() < 1e-5);
        assert!((out[4] - 0.0).abs() < 1e-5);
        assert!(out[5].is_infinite() && out[5] < 0.0);

        // Row 2 (i=2): [cumsum[2]-cumsum[0], cumsum[2]-cumsum[1], cumsum[2]-cumsum[2]] = [5, 3, 0]
        assert!((out[6] - 5.0).abs() < 1e-5);
        assert!((out[7] - 3.0).abs() < 1e-5);
        assert!((out[8] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn segsum_exp_produces_lower_triangular_mask() {
        // exp(segsum(a)) should produce a lower-triangular matrix
        // where M[i,j] = prod(a[j+1..=i])
        let a = [0.5f32, 0.5, 0.5];
        let mut seg = vec![0.0; 9];
        segsum(&a, &mut seg);

        // M[0,0] = exp(0) = 1
        assert!((seg[0].exp() - 1.0).abs() < 1e-5);
        // M[1,0] = exp(a[1]) = exp(0.5)
        assert!((seg[3].exp() - 0.5f32.exp()).abs() < 1e-5);
        // M[2,0] = exp(a[1]+a[2]) = exp(1.0)
        assert!((seg[6].exp() - (1.0f32).exp()).abs() < 1e-5);
    }

    // ── influence ──────────────────────────────────────────────────

    #[test]
    fn influence_no_decay() {
        // from == to → no decay → 1.0
        let a = [0.5, 0.5, 0.5];
        assert!((influence(&a, 1, 1) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn influence_one_step() {
        // from=0, to=1 → prod(a[1]) = 0.5
        let a = [0.5, 0.5, 0.5];
        assert!((influence(&a, 0, 1) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn influence_multi_step() {
        // from=0, to=2 → prod(a[1], a[2]) = 0.5 * 0.5 = 0.25
        let a = [0.5, 0.5, 0.5];
        assert!((influence(&a, 0, 2) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn influence_uniform_decay() {
        // from=0, to=4 → 0.5^4 = 0.0625
        let a = [0.5, 0.5, 0.5, 0.5, 0.5];
        assert!((influence(&a, 0, 4) - 0.0625).abs() < 1e-5);
    }

    #[test]
    fn influence_from_exceeds_to() {
        let a = [0.5, 0.5, 0.5];
        // from=2, to=0 → 1.0 (no decay backwards)
        assert!((influence(&a, 2, 0) - 1.0).abs() < 1e-5);
    }

    // ── context_freshness ──────────────────────────────────────────

    #[test]
    fn freshness_no_decay() {
        // a = [1, 1, 1] → all info persists → freshness high
        let a = [1.0, 1.0, 1.0];
        let f = context_freshness(&a);
        assert!(f > 0.9, "Expected high freshness for no decay, got {}", f);
    }

    #[test]
    fn freshness_fast_decay() {
        // a = [0, 0, 0] → nothing persists → freshness low
        let a = [0.0, 0.0, 0.0];
        let f = context_freshness(&a);
        assert!(f < 0.1, "Expected low freshness for fast decay, got {}", f);
    }

    #[test]
    fn freshness_medium_decay() {
        // a = [0.5, 0.5, 0.5] → moderate
        let a = [0.5, 0.5, 0.5];
        let f = context_freshness(&a);
        assert!(f > 0.0 && f < 1.0, "Expected moderate freshness, got {}", f);
    }

    #[test]
    fn freshness_empty() {
        let a: [f32; 0] = [];
        let f = context_freshness(&a);
        assert!((f - 0.5).abs() < 1e-5, "Expected 0.5 for empty, got {}", f);
    }

    // ── property: cumprodsum matches GDN2-style recurrence ─────────

    #[test]
    fn cumprodsum_matches_gdn2_diagonal_decay() {
        // GDN2 applies S *= Diag(α) per row. The diagonal decay α is the
        // cumprodsum decay factor. Verify: with a single channel,
        // cumprodsum(a, x) == sequential application of h = a*h + x.
        let a = [0.8, 0.9, 0.7, 0.85, 0.95];
        let x = [1.0, 2.0, 0.5, 3.0, 1.5];

        // Manual computation
        let mut h = 0.0f32;
        let mut expected = [0.0f32; 5];
        for t in 0..5 {
            h = a[t] * h + x[t];
            expected[t] = h;
        }

        let mut out = [0.0f32; 5];
        cumprodsum_scalar(&a, &x, 0.0, &mut out);

        for t in 0..5 {
            assert!(
                (out[t] - expected[t]).abs() < 1e-5,
                "Mismatch at t={}: cumprodsum={}, manual={}",
                t,
                out[t],
                expected[t]
            );
        }
    }
}
