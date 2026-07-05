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
//! # Interior mutability & thread safety
//!
//! `transformer::forward` requires `&mut ForwardContext` and
//! `&mut MultiLayerKVCache`, but [`DenseNode::forward_dense`] takes `&self`.
//! We bridge with per-slot `Mutex`es. To support rayon vertex parallelism
//! (Issue 020, Path A — the paper's §3.3 vertex-parameter-sharing model where
//! all hidden nodes share one LLM and execute in parallel), the node holds a
//! **pool** of `(ForwardContext, MultiLayerKVCache)` pairs. Each rayon worker
//! picks its own slot via `rayon::current_thread_index()`, so there is no
//! contention — each Mutex is only ever held by the worker that owns that slot.
//! Outside a rayon context, slot 0 is used.
//!
//! `DenseNode: Send + Sync` is satisfied because every field is `Send + Sync`:
//! `Config` / `TransformerWeights` are plain data, and `Vec<Mutex<T>>` is
//! `Sync` when `T: Send`.
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
//! Issue 020 (Path A vertex parallelism).

use std::sync::Mutex;

use katgpt_transformer::dense_mesh::traits::DenseNode;
use katgpt_transformer::dense_mesh::types::{DenseHidden, MeshScratch};
use crate::forward::forward;
use crate::ForwardContext;
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};
use katgpt_core::types::Config;

/// Default pool size used when the caller does not explicitly request a
/// parallelism budget. Sized to the host's logical CPU count so that the
/// default rayon thread pool (which also uses `available_parallelism`) has a
/// unique slot per worker in the common case.
fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// A [`DenseNode`] backed by a real `transformer::forward` call.
///
/// Holds a pool of `ForwardContext` + `MultiLayerKVCache` pairs (allocated
/// once at construction, reused across `forward_dense` calls — plasma tier).
/// Pool size governs the maximum rayon worker parallelism the node can serve
/// without slot contention; see the module docs on thread safety.
pub struct TransformerNode {
    config: Config,
    weights: TransformerWeights,
    /// Per-thread ForwardContext pool. Indexed by `rayon::current_thread_index()`.
    ctx_pool: Vec<Mutex<ForwardContext>>,
    /// Per-thread KV cache pool. Indexed by `rayon::current_thread_index()`.
    /// Each rayon worker has its own cache instance — no data race.
    cache_pool: Vec<Mutex<MultiLayerKVCache>>,
    /// Token id fed to `forward` (proof-mode: single-token).
    token: usize,
    /// Position fed to `forward`.
    pos: usize,
}

impl TransformerNode {
    /// Construct a node that will forward `token` at `pos`.
    ///
    /// The ctx/cache pool is sized to the host's `available_parallelism` so
    /// that rayon vertex parallelism (Issue 020) can give each worker its own
    /// slot. If you need a smaller pool (e.g. memory-constrained), use
    /// [`TransformerNode::new_with_pool_size`].
    pub fn new(config: Config, weights: TransformerWeights, token: usize, pos: usize) -> Self {
        Self::new_with_pool_size(config, weights, token, pos, default_pool_size())
    }

    /// Construct a node with an explicit per-thread pool size.
    ///
    /// `pool_size = 1` disables vertex parallelism (each rayon worker beyond
    /// slot 0 falls back to slot 0, contending on its Mutex — correct but slow).
    /// `pool_size >= rayon worker count` is required for uncontended parallel
    /// execution. Prefer [`TransformerNode::new`] unless you know your rayon
    /// pool is smaller than `available_parallelism`.
    pub fn new_with_pool_size(
        config: Config,
        weights: TransformerWeights,
        token: usize,
        pos: usize,
        pool_size: usize,
    ) -> Self {
        let pool_size = pool_size.max(1);
        let ctx_pool = (0..pool_size)
            .map(|_| Mutex::new(ForwardContext::new(&config)))
            .collect();
        let cache_pool = (0..pool_size)
            .map(|_| Mutex::new(MultiLayerKVCache::new(&config)))
            .collect();
        Self {
            config,
            weights,
            ctx_pool,
            cache_pool,
            token,
            pos,
        }
    }

    /// Pick a pool slot for the current thread. Outside rayon this is slot 0;
    /// inside rayon it is `rayon::current_thread_index()`, clamped to the pool
    /// size. Clamping means an oversized rayon pool will contend on the last
    /// slot — correct, just sub-optimal. Callers wanting no contention should
    /// size the pool via [`TransformerNode::new`] (defaults to
    /// `available_parallelism`).
    #[inline]
    fn slot(&self) -> usize {
        let idx = rayon::current_thread_index().unwrap_or(0);
        idx.min(self.ctx_pool.len().saturating_sub(1))
    }

    /// Reset the KV cache in every pool slot (e.g., between independent queries).
    pub fn reset_cache(&self) {
        for cache in &self.cache_pool {
            if let Ok(mut guard) = cache.lock() {
                guard.reset();
            }
        }
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

    /// Number of per-thread (ctx, cache) slots in the pool.
    pub fn pool_size(&self) -> usize {
        self.ctx_pool.len()
    }
}

impl DenseNode for TransformerNode {
    fn forward_dense(
        &self,
        _input: &DenseHidden,
        _layer_idx: usize,
        _scratch: &mut MeshScratch,
    ) -> DenseHidden {
        // Pick this thread's slot. Each rayon worker gets a unique slot when
        // the pool is sized to the worker count, so the Mutex is uncontended.
        let slot = self.slot();
        let mut ctx = self.ctx_pool[slot]
            .lock()
            .expect("TransformerNode ctx Mutex poisoned");
        let mut cache = self.cache_pool[slot]
            .lock()
            .expect("TransformerNode cache Mutex poisoned");
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
    use katgpt_core::types::Rng;

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
    /// per-slot Mutexes release cleanly between calls. At `Config::draft()`
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

        // Call multiple times — must not panic from Mutex re-lock.
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

    /// Calling `forward_dense` from multiple rayon workers concurrently is
    /// safe — each worker picks its own pool slot via
    /// `rayon::current_thread_index()`, so the per-slot Mutexes are
    /// uncontended and there is no data race on the KV cache.
    ///
    /// This is the core invariant for Issue 020 (vertex parallelism):
    /// the shared `TransformerNode` (vertex parameter sharing, paper §3.3)
    /// must tolerate parallel forward calls from the topology's rayon scope.
    #[test]
    fn test_transformer_node_parallel_forward_is_safe() {
        use rayon::prelude::*;

        let config = Config::draft();
        let mut rng = Rng::new(123);
        let weights = TransformerWeights::new(&config, &mut rng);
        // Pool sized to the host parallelism so each rayon worker gets a slot.
        let node = TransformerNode::new(config.clone(), weights, 0, 0);
        assert!(node.pool_size() >= 1, "pool must have at least one slot");

        let input = DenseHidden::zeros(1, config.vocab_size);
        let n_workers = 8usize.min(node.pool_size().max(1));

        // Run n_workers forward calls in parallel against the SAME node,
        // writing each result into a disjoint slot of a pre-allocated vec via
        // `par_iter_mut` — the same pattern the topology uses. If the Mutex
        // pool failed to isolate per-thread state, this would either deadlock,
        // panic on a poisoned Mutex, or produce torn reads.
        let mut outputs: Vec<DenseHidden> =
            (0..n_workers).map(|_| DenseHidden::zeros(1, config.vocab_size)).collect();
        outputs.par_iter_mut().for_each(|out_slot| {
            let mut scratch = MeshScratch::new(1, config.vocab_size);
            *out_slot = node.forward_dense(&input, 0, &mut scratch);
        });

        // All workers ran the same (token, pos) on the same weights — outputs
        // must be bit-identical regardless of which slot served the call.
        assert_eq!(outputs.len(), n_workers);
        for out in &outputs {
            assert_eq!(out.seq_len, 1);
            assert_eq!(out.hidden_dim, config.vocab_size);
        }
        let first = &outputs[0];
        for (i, out) in outputs.iter().enumerate().skip(1) {
            for (a, b) in first.rows().iter().zip(out.rows().iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "parallel forward outputs diverged at worker {} (a={a}, b={b})",
                    i
                );
            }
        }
    }
}
