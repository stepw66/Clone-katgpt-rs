//! LoraEdge — wraps an existing LoRA adapter as a DenseMesh edge.

use super::traits::DenseEdge;
use super::types::{DenseHidden, MeshScratch};

/// A DenseMesh edge backed by a LoRA adapter.
///
/// Applies a low-rank update `B @ A` to the hidden state, where `A` is
/// `[rank × in_dim]` and `B` is `[out_dim × rank]`. The update is added to
/// the identity pass-through (residual), so with `A = 0` or `B = 0` this
/// degrades to [`super::edge_identity::IdentityEdge`].
///
/// This is the **frozen-vertex setting** from the paper (§4.2.2): the vertex
/// LLM is frozen, only the edge (here, the LoRA `A`/`B`) is trained. The
/// trained LoRA edges live in riir-ai (R122 EdgeLoRA); katgpt-rs loads them
/// as read-only weights and applies them at inference.
pub struct LoraEdge {
    /// Down-projection `[rank * in_dim]`, row-major.
    lora_a: Vec<f32>,
    /// Up-projection `[out_dim * rank]`, row-major.
    lora_b: Vec<f32>,
    in_dim: usize,
    rank: usize,
    out_dim: usize,
    /// Scaling factor applied to the LoRA update (paper uses α/r).
    scale: f32,
    /// Optional sigmoid gate vector for sigmoid-gated multi-edge routing
    /// (riir-ai F3). When `None`, scale is fixed.
    gate: Option<Vec<f32>>,
}

impl LoraEdge {
    /// Build a LoRA edge from flattened weight tensors.
    ///
    /// - `lora_a`: `[rank * in_dim]` row-major (rank rows of in_dim).
    /// - `lora_b`: `[out_dim * rank]` row-major (out_dim rows of rank).
    /// - `scale`: scaling factor (typically `alpha / rank`).
    pub fn new(
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
        in_dim: usize,
        rank: usize,
        out_dim: usize,
        scale: f32,
    ) -> Self {
        debug_assert_eq!(lora_a.len(), rank * in_dim);
        debug_assert_eq!(lora_b.len(), out_dim * rank);
        Self {
            lora_a,
            lora_b,
            in_dim,
            rank,
            out_dim,
            scale,
            gate: None,
        }
    }

    /// Attach a sigmoid gate vector (for sigmoid-gated multi-edge routing).
    ///
    /// When set, the effective scale per position is `sigmoid(gate . hidden)`
    /// instead of the fixed `scale`. This enables multiple edges to be
    /// simultaneously active (per AGENTS.md: sigmoid not softmax).
    pub fn with_gate(mut self, gate: Vec<f32>) -> Self {
        debug_assert_eq!(gate.len(), self.in_dim);
        self.gate = Some(gate);
        self
    }

    /// Apply the LoRA update to one input row, writing the result to `out`.
    ///
    /// out = input + scale * (B @ (A @ input))
    ///
    /// `rank_buf` is a caller-provided scratch of length `rank`.
    #[inline]
    fn lora_row(&self, input_row: &[f32], out: &mut [f32], effective_scale: f32, rank_buf: &mut [f32]) {
        debug_assert_eq!(rank_buf.len(), self.rank);
        // rank_buf = A @ input  (rank-length)
        for r in 0..self.rank {
            let row_offset = r * self.in_dim;
            let mut acc = 0.0f32;
            let mut i = 0;
            while i + 8 <= self.in_dim {
                for j in 0..8 {
                    acc += self.lora_a[row_offset + i + j] * input_row[i + j];
                }
                i += 8;
            }
            while i < self.in_dim {
                acc += self.lora_a[row_offset + i] * input_row[i];
                i += 1;
            }
            rank_buf[r] = acc * effective_scale;
        }
        // out = input + B @ rank_buf
        for (o, out_slot) in out.iter_mut().enumerate() {
            let row_offset = o * self.rank;
            let mut acc = 0.0f32;
            let mut r = 0;
            while r + 8 <= self.rank {
                for j in 0..8 {
                    acc += self.lora_b[row_offset + r + j] * rank_buf[r + j];
                }
                r += 8;
            }
            while r < self.rank {
                acc += self.lora_b[row_offset + r] * rank_buf[r];
                r += 1;
            }
            *out_slot = input_row[o] + acc;
        }
    }

    /// Compute the per-position effective scale (sigmoid gate if present).
    #[inline]
    fn effective_scale(&self, input_row: &[f32]) -> f32 {
        match &self.gate {
            None => self.scale,
            Some(g) => {
                // sigmoid(gate . input)
                let mut dot = 0.0f32;
                let mut i = 0;
                while i + 8 <= self.in_dim {
                    for j in 0..8 {
                        dot += g[i + j] * input_row[i + j];
                    }
                    i += 8;
                }
                while i < self.in_dim {
                    dot += g[i] * input_row[i];
                    i += 1;
                }
                if dot >= 0.0 {
                    self.scale * (1.0 / (1.0 + (-dot).exp()))
                } else {
                    let e = dot.exp();
                    self.scale * (e / (1.0 + e))
                }
            }
        }
    }
}

impl DenseEdge for LoraEdge {
    fn route_into(&self, from: &DenseHidden, scratch: &mut MeshScratch) {
        debug_assert_eq!(from.hidden_dim, self.in_dim);
        debug_assert_eq!(scratch.edge_output.hidden_dim, self.out_dim);
        debug_assert_eq!(scratch.edge_output.seq_len, from.seq_len);
        // Ensure rank_buf is large enough (resize if needed — cold path).
        if scratch.rank_buf.len() < self.rank {
            scratch.rank_buf.resize(self.rank, 0.0);
        }
        let rank_buf_len = self.rank;
        // Split scratch to avoid double-borrow.
        let MeshScratch { edge_output, rank_buf, .. } = scratch;
        for pos in 0..from.seq_len {
            let in_start = pos * self.in_dim;
            let out_start = pos * self.out_dim;
            let effective_scale = self.effective_scale(&from.data[in_start..in_start + self.in_dim]);
            self.lora_row(
                &from.data[in_start..in_start + self.in_dim],
                &mut edge_output.data[out_start..out_start + self.out_dim],
                effective_scale,
                &mut rank_buf[..rank_buf_len],
            );
        }
    }

    fn cost_hint(&self) -> f32 {
        (self.in_dim + self.out_dim) as f32 * self.rank as f32
    }

    fn name(&self) -> &str {
        "lora"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_edge_zero_weights_is_identity() {
        // When LoRA weights are zero, output should equal input (residual).
        let in_dim = 4;
        let rank = 2;
        let out_dim = 4;
        let lora_a = vec![0.0f32; rank * in_dim];
        let lora_b = vec![0.0f32; out_dim * rank];
        let edge = LoraEdge::new(lora_a, lora_b, in_dim, rank, out_dim, 1.0);
        let mut from = DenseHidden::zeros(1, in_dim);
        from.data.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut scratch = MeshScratch::with_rank_capacity(1, out_dim, rank);
        edge.route_into(&from, &mut scratch);
        for (a, b) in scratch.edge_output.rows().iter().zip(from.rows().iter()) {
            assert!((a - b).abs() < 1e-6, "zero LoRA should be identity");
        }
    }

    #[test]
    fn test_lora_edge_nonzero_changes_output() {
        let in_dim = 4;
        let rank = 2;
        let out_dim = 4;
        // Identity-ish A (first rank=2 components) and B picking them out.
        // A = [[1,0,0,0],[0,1,0,0]], B = [[1,0],[0,1],[0,0],[0,0]]
        let lora_a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let lora_b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let edge = LoraEdge::new(lora_a, lora_b, in_dim, rank, out_dim, 1.0);
        let mut from = DenseHidden::zeros(1, in_dim);
        from.data.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut scratch = MeshScratch::with_rank_capacity(1, out_dim, rank);
        edge.route_into(&from, &mut scratch);
        let out = scratch.edge_output.rows();
        // out = input + scale * B @ A @ input = input + 1*[1,2,0,0]
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }
}
