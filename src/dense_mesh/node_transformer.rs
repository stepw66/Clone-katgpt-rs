//! TransformerNode — wraps `crate::transformer::forward` as a [`DenseNode`].
//!
//! This is the "real LLM as vertex" glue missing from the synthetic
//! `IdentityNode` used in `prof_dense_mesh.rs`. With `TransformerNode`, the
//! GOAT gate proofs (G3 easy overhead, G4 hard bound) can measure actual
//! transformer forward cost — not just framework overhead on a no-op node.
//!
//! # What it does
//!
//! Each `forward_dense` call invokes `transformer::forward` once at the
//! configured `(token, pos)`. The returned logits slice (length
//! `config.vocab_size`) is cloned into a fresh [`DenseHidden`] of shape
//! `[1, vocab_size]`.
//!
//! The input [`DenseHidden`] is intentionally ignored — in this proof mode
//! the node is a pure function of `(token, pos)`, not of any latent
//! predecessor state. This matches the "frozen vertex + edges do the work"
//! setting from LMNet §4.2.2: the LLM forward is the same across branches,
//! only the edges differ.
//!
//! # Interior mutability
//!
//! `transformer::forward` requires `&mut ForwardContext` and
//! `&mut MultiLayerKVCache`, but [`DenseNode::forward_dense`] takes `&self`.
//! We bridge with `RefCell`. The runtime borrow check is a single atomic
//! compare-and-swap — negligible vs the transformer forward cost (~ms).
//!
//! # Latent / Raw Compliance
//!
//! The logits vector is **raw** (it's the de-embedding output — token-space).
//! Per AGENTS.md: logits at the output boundary node are raw, and that's
//! exactly what this node produces. Interior nodes in a real stripped-vertex
//! mesh would emit latent residual states; this proof node intentionally
//! uses the full forward (embed + layers + de-embed) so we can compare
//! apples-to-apples against vanilla `transformer::forward` for gate 3/4.
//!
//! Reference: katgpt-rs/.research/234_DenseMesh_Latent_Node_Network.md,
//! katgpt-rs/.plans/266_densemesh_latent_node_network.md Phase 5.

use std::cell::RefCell;

use super::traits::DenseNode;
use super::types::{DenseHidden, MeshScratch};
use crate::transformer::{forward, ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::Config;

/// A [`DenseNode`] backed by a real `transformer::forward` call.
///
/// Holds one `ForwardContext` + `MultiLayerKVCache` (allocated once at
/// construction, reused across `forward_dense` calls — plasma tier).
pub struct TransformerNode {
    config: Config,
    weights: TransformerWeights,
    ctx: RefCell<ForwardContext>,
    cache: RefCell<MultiLayerKVCache>,
    /// Token id fed to `forward` (proof-mode: single-token).
    token: usize,
    /// Position fed to `forward`.
    pos: usize,
}

impl TransformerNode {
    /// Construct a node that will forward `token` at `pos`.
    ///
    /// `ctx` and `cache` are allocated once here (cold tier) and reused
    /// across all `forward_dense` invocations.
    pub fn new(config: Config, weights: TransformerWeights, token: usize, pos: usize) -> Self {
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);
        Self {
            config,
            weights,
            ctx: RefCell::new(ctx),
            cache: RefCell::new(cache),
            token,
            pos,
        }
    }

    /// Reset the KV cache (e.g., between independent queries).
    pub fn reset_cache(&self) {
        self.cache.borrow_mut().reset();
    }

    /// The token id used for forward.
    pub fn token(&self) -> usize {
        self.token
    }

    /// The position used for forward.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Borrow the config (immutable).
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl DenseNode for TransformerNode {
    fn forward_dense(
        &self,
        _input: &DenseHidden,
        _layer_idx: usize,
        _scratch: &mut MeshScratch,
    ) -> DenseHidden {
        // Borrow ctx + cache mutably for the duration of the forward call.
        // RefCell borrow failure would indicate re-entrancy — a bug.
        let mut ctx = self.ctx.borrow_mut();
        let mut cache = self.cache.borrow_mut();
        let out = forward(
            &mut ctx,
            &self.weights,
            &mut cache,
            self.token,
            self.pos,
            &self.config,
        );
        // `out` is `&mut ctx.logits` — clone into an owned `DenseHidden`
        // to release the ctx borrow before returning.
        //
        // hidden_dim = vocab_size (the logits length). This is intentionally
        // raw token-space — see module docs on latent/raw compliance.
        DenseHidden {
            data: out.to_vec().into_boxed_slice(),
            seq_len: 1,
            hidden_dim: self.config.vocab_size,
        }
    }

    fn hidden_dim(&self) -> usize {
        // Note: returns vocab_size (logits dim), not n_embd. Callers building
        // LoRA edges between TransformerNodes must size them to vocab_size,
        // or use IdentityEdge (which is dim-agnostic for correctness but
        // still asserts matching lengths in debug).
        self.config.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rng;

    /// Smoke test: TransformerNode produces non-empty output and reports the
    /// correct hidden_dim (= vocab_size, since `forward` returns logits).
    #[test]
    fn test_transformer_node_basic_forward() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let node = TransformerNode::new(config.clone(), weights, 0, 0);

        let input = DenseHidden::zeros(1, config.vocab_size);
        let mut scratch = MeshScratch::new(1, config.vocab_size);
        let out = node.forward_dense(&input, 0, &mut scratch);

        assert_eq!(out.seq_len, 1, "single-token forward");
        assert_eq!(out.hidden_dim, config.vocab_size, "logits dim");
        assert_eq!(out.len(), config.vocab_size, "buffer length");
        // Logits should not all be zero (random weights produce non-zero output).
        let sum: f32 = out.rows().iter().sum();
        assert!(sum.abs() > 0.0, "logits sum must be non-zero");
    }

    /// Calling forward_dense multiple times on the same node is safe — the
    /// RefCell borrows release cleanly between calls. At `Config::draft()`
    /// scale with n_layer=1 and a single cache slot, repeated calls at the
    /// same `(token, pos)` produce identical output (deterministic write to
    /// the same KV slot). This test confirms reusability without panic.
    #[test]
    fn test_transformer_node_repeated_forward_safe() {
        let config = Config::draft();
        let mut rng = Rng::new(7);
        let weights = TransformerWeights::new(&config, &mut rng);
        let node = TransformerNode::new(config.clone(), weights, 0, 0);

        let input = DenseHidden::zeros(1, config.vocab_size);
        let mut scratch = MeshScratch::new(1, config.vocab_size);

        // Call multiple times — must not panic from RefCell re-borrow.
        let out1 = node.forward_dense(&input, 0, &mut scratch);
        let out2 = node.forward_dense(&input, 0, &mut scratch);
        let out3 = node.forward_dense(&input, 0, &mut scratch);

        // At draft scale + same (token, pos), output is deterministic.
        assert_eq!(out1.len(), out2.len());
        assert_eq!(out2.len(), out3.len());
        for i in 0..out1.len() {
            assert!((out1.rows()[i] - out2.rows()[i]).abs() < 1e-6);
            assert!((out2.rows()[i] - out3.rows()[i]).abs() < 1e-6);
        }
    }

    /// hidden_dim reports vocab_size, not n_embd — documented contract.
    #[test]
    fn test_transformer_node_hidden_dim_is_vocab_size() {
        let config = Config::draft();
        let mut rng = Rng::new(99);
        let weights = TransformerWeights::new(&config, &mut rng);
        let node = TransformerNode::new(config, weights, 0, 0);
        assert_eq!(node.hidden_dim(), 27, "Config::draft() vocab_size is 27");
    }
}
