//! ProjectionEdge — fixed random projection, modelless fallback.

use super::traits::DenseEdge;
use super::types::{DenseHidden, MeshScratch};

/// An edge that applies a fixed (non-trainable) projection matrix to the
/// hidden state.
///
/// This is the **modelless fallback** when no trained LoRA edge is available.
/// The projection is random but fixed at construction — it provides *some*
/// transformation between nodes without requiring training.
///
/// Not expected to match trained-edge quality (paper's +30.5%), but provides
/// a non-trivial baseline for testing topology mechanics.
pub struct ProjectionEdge {
    /// Row-major projection matrix `[out_dim * in_dim]`.
    matrix: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl ProjectionEdge {
    /// Create a projection edge with a fixed random matrix.
    ///
    /// Uses `fastrand` for deterministic seeding. Scale is `1/sqrt(in_dim)`
    /// to preserve activation magnitude (Xavier-like init).
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let scale = 1.0 / (in_dim as f32).sqrt();
        let matrix = (0..in_dim * out_dim)
            .map(|_| rng.f32() * 2.0 * scale - scale)
            .collect();
        Self {
            matrix,
            in_dim,
            out_dim,
        }
    }

    /// Apply projection to one row of `input` (length `in_dim`).
    #[inline]
    #[allow(clippy::needless_range_loop)] // stride math: o indexes out[o] AND o*self.in_dim offset into matrix
    fn project_row(&self, input_row: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input_row.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        // Zero out.
        for o in out.iter_mut() {
            *o = 0.0;
        }
        // matvec: out[o] = sum_i matrix[o * in_dim + i] * input[i].
        // Chunked inner loop for SIMD.
        for o in 0..self.out_dim {
            let row_offset = o * self.in_dim;
            let mut acc = 0.0f32;
            let mut i = 0;
            while i + 8 <= self.in_dim {
                for j in 0..8 {
                    acc += self.matrix[row_offset + i + j] * input_row[i + j];
                }
                i += 8;
            }
            while i < self.in_dim {
                acc += self.matrix[row_offset + i] * input_row[i];
                i += 1;
            }
            out[o] = acc;
        }
    }
}

impl DenseEdge for ProjectionEdge {
    fn route_into(&self, from: &DenseHidden, scratch: &mut MeshScratch) {
        debug_assert_eq!(from.hidden_dim, self.in_dim);
        debug_assert_eq!(scratch.edge_output.hidden_dim, self.out_dim);
        debug_assert_eq!(scratch.edge_output.seq_len, from.seq_len);
        // Project each row.
        for pos in 0..from.seq_len {
            let in_start = pos * self.in_dim;
            let out_start = pos * self.out_dim;
            self.project_row(
                &from.data[in_start..in_start + self.in_dim],
                &mut scratch.edge_output.data[out_start..out_start + self.out_dim],
            );
        }
    }

    fn cost_hint(&self) -> f32 {
        (self.in_dim * self.out_dim) as f32
    }

    fn name(&self) -> &str {
        "projection"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_edge_shape_preserving() {
        let edge = ProjectionEdge::new(4, 4, 42);
        let mut from = DenseHidden::zeros(2, 4);
        for (i, v) in from.rows_mut().iter_mut().enumerate() {
            *v = i as f32 * 0.1;
        }
        let mut scratch = MeshScratch::new(2, 4);
        edge.route_into(&from, &mut scratch);
        assert_eq!(scratch.edge_output.len(), 8); // seq_len 2 * hidden_dim 4.
    }

    #[test]
    fn test_projection_edge_changes_output() {
        let edge = ProjectionEdge::new(4, 4, 42);
        let mut from = DenseHidden::zeros(1, 4);
        from.data.copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let mut scratch = MeshScratch::new(1, 4);
        edge.route_into(&from, &mut scratch);
        // Output should not equal input (it's a random projection).
        let differs = scratch
            .edge_output
            .rows()
            .iter()
            .zip(from.rows().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "projection must change the hidden state");
    }
}
