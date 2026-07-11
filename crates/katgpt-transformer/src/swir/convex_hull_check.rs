//! Convex-hull invariant check (Plan 275 GOAT gate G4).
//!
//! The soft embedding `ẽ_t = Σ_v p_t[v] · e(v)` is a convex combination of rows
//! of the embedding matrix, so for every dimension `d`:
//!
//! ```text
//!   min_v e(v)[d]  ≤  ẽ_t[d]  ≤  max_v e(v)[d]
//! ```
//!
//! Any violation indicates a SIMD bug, numerical drift, or a caller passing an
//! unnormalised probability vector. This is a free correctness check — the
//! invariant is mathematical, not heuristic.
//!
//! This kernel is **test/debug only** — O(vocab · embedding_dim) is too
//! expensive for the hot path. It runs in unit tests and in the GOAT-gate
//! benchmark harness, not in `SwiRController::step`.

const HULL_TOL: f32 = 1e-4;

/// Returns `true` iff `soft_embed` lies inside the per-dim
/// `[min_v e(v)[d], max_v e(v)[d]]` box of the vocabulary embeddings, within
/// [`HULL_TOL`] absolute slack (to absorb f32 rounding in the SIMD accumulate).
///
/// `soft_embed`: length `embedding_dim`.
/// `embedding_matrix`: flattened `[vocab, embedding_dim]` row-major.
/// `embedding_dim`: width of each row.
#[inline]
pub fn in_vocab_convex_hull(
    soft_embed: &[f32],
    embedding_matrix: &[f32],
    embedding_dim: usize,
) -> bool {
    let vocab = embedding_matrix.len() / embedding_dim;
    if vocab == 0 || soft_embed.len() != embedding_dim {
        return false;
    }

    // Per-dim min/max scan of the embedding matrix.
    let mut dim_min = vec![f32::INFINITY; embedding_dim];
    let mut dim_max = vec![f32::NEG_INFINITY; embedding_dim];
    for v in 0..vocab {
        let row_off = v * embedding_dim;
        for d in 0..embedding_dim {
            let e = embedding_matrix[row_off + d];
            if e < dim_min[d] {
                dim_min[d] = e;
            }
            if e > dim_max[d] {
                dim_max[d] = e;
            }
        }
    }

    for d in 0..embedding_dim {
        let s = soft_embed[d];
        if s < dim_min[d] - HULL_TOL || s > dim_max[d] + HULL_TOL {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swir::soft_embedding::soft_embedding;

    fn mat(rows: &[Vec<f32>]) -> Vec<f32> {
        let mut out = Vec::new();
        for r in rows {
            out.extend_from_slice(r);
        }
        out
    }

    #[test]
    fn one_hot_lies_in_hull() {
        let emb = mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let mut probs = vec![0.0; 3];
        probs[1] = 1.0;
        let mut out = vec![0.0; 3];
        soft_embedding(&probs, &emb, 3, &mut out);
        assert!(
            in_vocab_convex_hull(&out, &emb, 3),
            "one-hot soft embed must be in hull"
        );
    }

    #[test]
    fn uniform_mixture_lies_in_hull() {
        let emb = mat(&[
            vec![1.0, -2.0, 3.0],
            vec![-4.0, 5.0, -6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, -11.0, 12.0],
        ]);
        let probs = vec![0.25; 4];
        let mut out = vec![0.0; 3];
        soft_embedding(&probs, &emb, 3, &mut out);
        assert!(in_vocab_convex_hull(&out, &emb, 3));
    }

    #[test]
    fn out_of_range_fails_hull_check() {
        let emb = mat(&[vec![1.0, 1.0], vec![2.0, 2.0]]);
        let bad = vec![10.0, 10.0]; // outside the box.
        assert!(!in_vocab_convex_hull(&bad, &emb, 2));
    }

    #[test]
    fn random_soft_embeddings_all_in_hull() {
        // Random probs over a fixed vocab — all mixtures must satisfy G4.
        let emb = mat(&[
            vec![1.0, -2.0, 3.0, 0.5, -1.5],
            vec![-4.0, 5.0, -6.0, 2.5, 3.0],
            vec![7.0, 8.0, 9.0, -1.0, -2.0],
            vec![0.0, -1.0, 2.0, 1.0, 4.0],
        ]);
        let mut rng_state = 0xdead_beef_u32;
        for _ in 0..200 {
            // Cheap xorshift for reproducibility — no external rng dep.
            let mut probs = [0.0f32; 4];
            let mut sum = 0.0;
            for p in probs.iter_mut() {
                rng_state = rng_state.wrapping_mul(2654435761).wrapping_add(12345);
                *p = (rng_state as f32) / (u32::MAX as f32);
                sum += *p;
            }
            for p in probs.iter_mut() {
                *p /= sum;
            }
            let mut out = vec![0.0; 5];
            soft_embedding(&probs, &emb, 5, &mut out);
            assert!(
                in_vocab_convex_hull(&out, &emb, 5),
                "random soft embed violated hull: {out:?}"
            );
        }
    }
}
