//! RoPE (Rotary Position Embedding) undo/reapply utilities.
//!
//! RoPE applies position-dependent rotation to pairs of dimensions:
//!   For dim pair (2i, 2i+1), rotation angle = pos × inv_freq[i]
//!   where inv_freq[i] = 1.0 / (10000^(2i/d_head))
//!
//! `undo_rope` applies the INVERSE rotation (negated angles).
//! `reapply_rope` applies the FORWARD rotation.
//!
//! These are exact algebraic inverses — roundtrip error is at most float epsilon.

/// Pre-computed RoPE inverse frequencies for a given head_dim.
///
/// Caches `inv_freq[i] = 1.0 / (10000^(2i/d_head))` to avoid recomputing
/// and reallocating on every rotation call.
pub struct RopeFreqs {
    inv_freq: Vec<f32>,
    half: usize,
}

impl RopeFreqs {
    /// Build inverse frequencies for the given head_dim.
    pub fn new(head_dim: usize) -> Self {
        let half = head_dim / 2;
        let base: f32 = 10000.0;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exp = 2.0 * i as f32 / head_dim as f32;
                1.0 / base.powf(exp)
            })
            .collect();
        Self { inv_freq, half }
    }

    /// Apply position-dependent rotation to dim pairs in-place.
    ///
    /// For each pair (2i, 2i+1):
    ///   θ = pos × inv_freq[i]
    ///   [x0', x1'] = [[cos θ, -sin θ], [sin θ, cos θ]] @ [x0, x1]
    ///
    /// When `negate = true`, applies the inverse rotation (negated angle).
    #[inline]
    pub fn apply(&self, x: &mut [f32], pos: usize, negate: bool) {
        let sign: f32 = if negate { -1.0 } else { 1.0 };
        let pos_f = sign * pos as f32;

        for i in 0..self.half {
            let theta = pos_f * self.inv_freq[i];
            let (sin_t, cos_t) = theta.sin_cos();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = cos_t * x0 - sin_t * x1;
            x[2 * i + 1] = sin_t * x0 + cos_t * x1;
        }
    }
}

/// Apply position-dependent rotation to dim pairs in-place.
///
/// For each pair (2i, 2i+1):
///   θ = pos × inv_freq[i]
///   [x0', x1'] = [[cos θ, -sin θ], [sin θ, cos θ]] @ [x0, x1]
///
/// When `negate = true`, applies the inverse rotation (negated angle).
///
/// Prefer [`RopeFreqs::apply`] in hot paths to avoid reallocating the
/// frequency table on every call.
fn apply_rotation(x: &mut [f32], pos: usize, head_dim: usize, negate: bool) {
    let freqs = RopeFreqs::new(head_dim);
    freqs.apply(x, pos, negate);
}

/// Undo RoPE: apply the inverse position-dependent rotation.
///
/// For dim pair (2i, 2i+1), applies rotation by -pos × inv_freq[i].
/// This removes the position-dependent phase structure so that subsequent
/// PCA sees spatially coherent data.
pub fn undo_rope(x: &mut [f32], pos: usize, head_dim: usize) {
    apply_rotation(x, pos, head_dim, true);
}

/// Reapply RoPE: apply the forward position-dependent rotation.
///
/// After inverse PCA rotation, the reconstructed vector needs RoPE
/// reapplied to restore position-dependent structure for attention.
pub fn reapply_rope(x: &mut [f32], pos: usize, head_dim: usize) {
    apply_rotation(x, pos, head_dim, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo_reapply_roundtrip() {
        let head_dim = 128;
        let pos = 42;
        let mut x: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 + 1.0).sin() * 0.5)
            .collect();
        let original = x.clone();

        undo_rope(&mut x, pos, head_dim);
        reapply_rope(&mut x, pos, head_dim);

        for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-5,
                "roundtrip failed at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_undo_changes_vector() {
        let head_dim = 64;
        let pos = 10;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).cos()).collect();
        let original = x.clone();

        undo_rope(&mut x, pos, head_dim);

        // Should change the vector (unless degenerate)
        let diff: f32 = original
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "undo_rope should modify the vector");
    }

    #[test]
    fn test_reapply_changes_vector() {
        let head_dim = 64;
        let pos = 10;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).cos()).collect();
        let original = x.clone();

        reapply_rope(&mut x, pos, head_dim);

        let diff: f32 = original
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "reapply_rope should modify the vector");
    }

    #[test]
    fn test_identity_at_pos_zero() {
        let head_dim = 32;
        let pos = 0;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let original = x.clone();

        // At pos=0, rotation angle = 0 → identity
        reapply_rope(&mut x, pos, head_dim);

        for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "pos=0 should be identity at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_roundtrip_various_positions() {
        let head_dim = 64;
        for pos in [0, 1, 10, 100, 511] {
            let mut x: Vec<f32> = (0..head_dim)
                .map(|i| (i as f32 + 1.0).sin() * 0.5)
                .collect();
            let original = x.clone();

            undo_rope(&mut x, pos, head_dim);
            reapply_rope(&mut x, pos, head_dim);

            for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < 1e-4,
                    "roundtrip failed at pos={pos}, [{i}]: {orig} vs {rec}"
                );
            }
        }
    }
}
