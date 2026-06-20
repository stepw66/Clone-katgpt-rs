//! SIMD argmax — single-pass `(usize, f32)` reducer used by token sampling.

/// Single-pass argmax: returns `(index, value)` of the maximum element.
///
/// Fuses max-finding and index-recovery into one traversal. The naive idiom —
/// `simd_max_f32(x)` followed by `x.iter().position(|&v| v == max)` — scans the
/// buffer twice (one SIMD max pass + one scalar equality scan that can run the
/// full length). On `aarch64` this uses a NEON kernel that tracks per-lane max
/// values *and* indices in one pass, measured at ~5× faster than the two-pass
/// idiom across vocab sizes 27 → 256k.
///
/// Tie-break matches `position(|&v| v == max)`: the **first** index attaining
/// the maximum is returned (strict `>` never replaces an earlier equal value).
///
/// Returns `(0, f32::NEG_INFINITY)` for an empty slice.

#[inline]
pub fn simd_argmax_f32(x: &[f32]) -> (usize, f32) {
    if x.is_empty() {
        return (0, f32::NEG_INFINITY);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_argmax_f32(x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Non-NEON: a scalar index-tracking loop does not auto-vectorize and
        // measured *slower* than the existing SIMD-max + position idiom at large
        // vocab, so reuse that two-pass path here (no regression). An AVX2 kernel
        // mirroring the NEON one could replace this once it can be verified on x86.
        let max_val = simd_max_f32(x);
        let idx = x.iter().position(|&v| v == max_val).unwrap_or(0);
        (idx, max_val)
    }
}

/// NEON single-pass argmax: tracks 4 lanes of (max value, index) simultaneously.
///
/// Each lane `l` accumulates the strided elements `x[l], x[l+4], x[l+8], …`;
/// `vcgtq_f32` (strict greater-than) only updates a lane on a *new* maximum, so
/// within a lane the earliest index wins. The 4 lane candidates are then reduced
/// by (value desc, index asc), and a scalar tail handles the final `len % 4`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_argmax_f32(x: &[f32]) -> (usize, f32) {
    use core::arch::aarch64::{
        vaddq_u32, vbslq_u32, vcgtq_f32, vdupq_n_u32, vld1q_f32, vld1q_u32, vmaxq_f32, vst1q_f32,
        vst1q_u32,
    };
    unsafe {
        let n = x.len();
        // Below one full vector, scalar is cheaper than NEON setup.
        if n < 4 {
            let mut bi = 0usize;
            let mut bv = *x.get_unchecked(0);
            for i in 1..n {
                let v = *x.get_unchecked(i);
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            return (bi, bv);
        }

        let base: [u32; 4] = [0, 1, 2, 3];
        let four = vdupq_n_u32(4);
        let mut vmax = vld1q_f32(x.as_ptr());
        let mut vidx = vld1q_u32(base.as_ptr());
        let mut cur = vaddq_u32(vidx, four); // indices for the next chunk
        let chunks = n / 4;
        let mut i = 4;
        for _ in 1..chunks {
            let v = vld1q_f32(x.as_ptr().add(i));
            let mask = vcgtq_f32(v, vmax); // lanes where v strictly greater
            vmax = vmaxq_f32(vmax, v);
            vidx = vbslq_u32(mask, cur, vidx); // adopt new index only where greater
            cur = vaddq_u32(cur, four);
            i += 4;
        }

        // Reduce the 4 lanes: highest value, smallest index on ties.
        let mut vals = [0f32; 4];
        let mut idxs = [0u32; 4];
        vst1q_f32(vals.as_mut_ptr(), vmax);
        vst1q_u32(idxs.as_mut_ptr(), vidx);
        let mut bv = vals[0];
        let mut bi = idxs[0] as usize;
        for l in 1..4 {
            let vi = idxs[l] as usize;
            if vals[l] > bv || (vals[l] == bv && vi < bi) {
                bv = vals[l];
                bi = vi;
            }
        }

        // Scalar tail (0..3 remaining elements).
        while i < n {
            let v = *x.get_unchecked(i);
            if v > bv {
                bv = v;
                bi = i;
            }
            i += 1;
        }
        (bi, bv)
    }
}
