//! Scale-normalized heterogeneous-branch fusion (paper Eq 13–14, Plan 358).
//!
//! Fuses per-head outputs from two heterogeneous attention branches (e.g. FA
//! with sharp low-entropy distributions + GDN with flat high-entropy outputs)
//! by independently RMSNorming each head, then applying a learnable per-head
//! scalar γ, writing into the index-preserving slot. Ships currently unused
//! (Plan 182 is layer-wise, not head-wise) but ready for any future head-mixing
//! runtime. Static γ beats dynamic softmax gate per paper Table 5.

/// Scale-normalized fusion of heterogeneous attention branch outputs
/// (paper Eq 13–14). Independent RMSNorm per branch + index-preserving
/// concatenation + learnable per-head scalar γ.
///
/// Why: FA softmax produces sharp low-entropy distributions dominated by query
/// norm; GDN normalization cancels query norm → smoother high-entropy outputs.
/// Naive concatenation destabilizes (paper Table 5: -10% RULER Single w/o Norm).
/// Independent RMSNorm per branch unifies feature scales; per-head γ lets the
/// model adaptively recalibrate each head's contribution. Static γ beats
/// dynamic softmax gate (paper Table 5).
///
/// Generic over any two branches identified by a per-head `BranchKind` tag.
#[derive(Clone, Debug)]
pub struct ScaleNormalizedFusion {
    /// Per-head learnable scalar γ. Length = n_heads. Default 1.0 (identity).
    pub gamma: Vec<f32>,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl ScaleNormalizedFusion {
    pub fn new(n_heads: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; n_heads],
            eps,
        }
    }

    /// Fuse per-head outputs in-place into `out` (length `n_heads * head_dim`,
    /// row-major). `per_head_outputs[h]` is the raw output of head h (already
    /// routed from its branch — caller does the FA-vs-GDN dispatch). Each head's
    /// output is independently RMSNormed, then multiplied by `gamma[h]`, then
    /// written into `out` at the head's index-preserving slot.
    ///
    /// Zero-allocation: writes directly into `out`; the RMSNorm + γ-scale are
    /// fused into a single loop (2 loops/head, no scratch buffer needed).
    #[inline]
    pub fn fuse_into(
        &self,
        per_head_outputs: &[&[f32]], // [n_heads][head_dim]
        head_dim: usize,
        out: &mut [f32], // [n_heads * head_dim]
    ) {
        let n_heads = per_head_outputs.len();
        debug_assert_eq!(out.len(), n_heads * head_dim);
        for (h, src) in per_head_outputs.iter().enumerate() {
            debug_assert_eq!(src.len(), head_dim);
            // RMSNorm: compute inv_rms once per head.
            let mut sum_sq = 0.0f32;
            for v in *src {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + self.eps).sqrt();
            // Fused RMSNorm + γ-scale + write into index-preserving slot.
            let g = self.gamma[h];
            let slot = h * head_dim;
            for j in 0..head_dim {
                out[slot + j] = g * src[j] * inv_rms;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference RMSNorm computed straightforwardly for test comparison.
    fn rmsnorm_ref(x: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / n + eps).sqrt();
        x.iter().map(|v| v * inv).collect()
    }

    #[test]
    fn identity_gamma_is_rmsnorm() {
        // γ = 1 → output is exactly the RMSNorm of each head's input.
        let n_heads = 2;
        let head_dim = 4;
        let eps = 1e-5;
        let fusion = ScaleNormalizedFusion::new(n_heads, eps);
        let h0 = [1.0, 2.0, 3.0, 4.0];
        let h1 = [0.5, -0.5, 2.0, -2.0];
        let per_head: Vec<&[f32]> = vec![&h0, &h1];
        let mut out = vec![0.0f32; n_heads * head_dim];
        fusion.fuse_into(&per_head, head_dim, &mut out);

        let ref0 = rmsnorm_ref(&h0, eps);
        let ref1 = rmsnorm_ref(&h1, eps);
        for j in 0..head_dim {
            assert!((out[j] - ref0[j]).abs() < 1e-6, "h0 mismatch at {j}");
            assert!(
                (out[head_dim + j] - ref1[j]).abs() < 1e-6,
                "h1 mismatch at {j}"
            );
        }
    }

    #[test]
    fn gamma_zero_zeros_output() {
        // γ = 0 → all zeros.
        let n_heads = 1;
        let head_dim = 3;
        let mut fusion = ScaleNormalizedFusion::new(n_heads, 1e-5);
        fusion.gamma = vec![0.0];
        let h0 = [3.0, 4.0, 5.0];
        let per_head: Vec<&[f32]> = vec![&h0];
        let mut out = vec![0.0f32; n_heads * head_dim];
        fusion.fuse_into(&per_head, head_dim, &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn gamma_two_doubles_rmsnorm() {
        // γ = 2 → output is 2× the RMSNorm.
        let n_heads = 1;
        let head_dim = 3;
        let eps = 1e-5;
        let mut fusion = ScaleNormalizedFusion::new(n_heads, eps);
        fusion.gamma = vec![2.0];
        let h0 = [1.0, 1.0, 1.0];
        let per_head: Vec<&[f32]> = vec![&h0];
        let mut out = vec![0.0f32; n_heads * head_dim];
        fusion.fuse_into(&per_head, head_dim, &mut out);
        let ref0 = rmsnorm_ref(&h0, eps);
        for j in 0..head_dim {
            assert!((out[j] - 2.0 * ref0[j]).abs() < 1e-6, "mismatch at {j}");
        }
    }

    #[test]
    fn mixed_branches_unified_scale() {
        // Synthetic FA-sharp (large norm) + GDN-flat (small norm) inputs:
        // after RMSNorm both heads land on the same unit-RMS scale (modulo γ).
        let n_heads = 2;
        let head_dim = 4;
        let eps = 1e-5;
        let fusion = ScaleNormalizedFusion::new(n_heads, eps);
        // FA-sharp: large magnitude, peaked.
        let fa = [10.0, 0.0, 0.0, 0.0];
        // GDN-flat: small magnitude, uniform.
        let gdn = [0.1, 0.1, 0.1, 0.1];
        let per_head: Vec<&[f32]> = vec![&fa, &gdn];
        let mut out = vec![0.0f32; n_heads * head_dim];
        fusion.fuse_into(&per_head, head_dim, &mut out);

        // Compute the per-head output RMS (should be ~1 for both since γ=1).
        // Note: RMSNorm with eps pulls RMS slightly below 1.0, more so for
        // smaller-magnitude inputs (eps is relatively larger). The intent of
        // this test is scale UNIFICATION — both heads land on ~the same ~unit
        // scale regardless of input magnitude — not bit-exact unit RMS.
        let rms = |slice: &[f32]| -> f32 {
            let n = slice.len() as f32;
            (slice.iter().map(|v| v * v).sum::<f32>() / n).sqrt()
        };
        let rms0 = rms(&out[0..head_dim]);
        let rms1 = rms(&out[head_dim..2 * head_dim]);
        assert!(
            (rms0 - 1.0).abs() < 1e-3,
            "FA head not ~unit-RMS: {rms0}"
        );
        assert!((rms1 - 1.0).abs() < 1e-3, "GDN head not ~unit-RMS: {rms1}");
        // Scale unification: the ~10× input magnitude difference (FA=10 vs
        // GDN=0.1) collapses to a small relative gap after RMSNorm.
        assert!((rms0 - rms1).abs() < 1e-3, "scales not unified: {rms0} vs {rms1}");
    }
}
