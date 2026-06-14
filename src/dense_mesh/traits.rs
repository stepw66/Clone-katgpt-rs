//! Traits for DenseMesh — DenseNode and DenseEdge.

use super::types::{DenseHidden, MeshScratch};

// ---------------------------------------------------------------------------
// DenseNode — a stripped transformer forward pass (vertex)
// ---------------------------------------------------------------------------

/// A node in the DenseMesh — one forward pass through a stripped LLM.
///
/// "Stripped" means: no embedding layer at input, no de-embedding at output
/// (paper §3.1.1). The node consumes a `DenseHidden` (dense vector from the
/// previous node via an edge) and produces a `DenseHidden` for the next layer.
///
/// Only the **input** boundary node accepts tokens (it runs the embedding
/// layer internally); only the **output** boundary node emits tokens (it runs
/// the de-embedding layer). All intermediate nodes are pure `DenseHidden -> DenseHidden`.
///
/// # Vertex Parameter Sharing
///
/// Per paper §3.3, all nodes share the same pre-trained LLM weights `θ_v`.
/// Implementations should reuse a single transformer forward function — only
/// the active LoRA edge differs per branch.
pub trait DenseNode {
    /// Forward pass: consume `input` hidden state, produce output hidden state.
    ///
    /// `scratch` is a pre-allocated reusable buffer (plasma tier) — use it for
    /// intermediate activations, never allocate inside this function.
    ///
    /// `layer_idx` tells the node which topology layer it is in (some nodes
    /// may want to behave differently at input vs hidden vs output layers).
    fn forward_dense(
        &self,
        input: &DenseHidden,
        layer_idx: usize,
        scratch: &mut MeshScratch,
    ) -> DenseHidden;

    /// The hidden dimension this node produces.
    ///
    /// Used by the topology engine to size scratch buffers. Must be constant
    /// across calls (configurable at construction, not at forward time).
    fn hidden_dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// DenseEdge — a communication transformation between nodes
// ---------------------------------------------------------------------------

/// A communication edge between two DenseMesh nodes (paper §3.1.2).
///
/// The edge transforms the output hidden state of a predecessor node into the
/// input hidden state of a successor node. This is where:
/// - **Identity edges** pass through unchanged (baseline, gate 1)
/// - **LoRA edges** apply a small low-rank adaptation (the frozen-vertex
///   setting from the paper — edges are the only trainable part)
/// - **Projection edges** apply a fixed random projection (modelless fallback)
///
/// # Aggregation
///
/// When multiple edges feed into one node, their outputs are summed (paper
/// §3.1.3). The topology engine handles summation; each edge only transforms
/// one predecessor's hidden state.
pub trait DenseEdge: Send + Sync {
    /// Transform `from` (predecessor output) and write the result into
    /// `scratch.edge_output`.
    ///
    /// The topology engine reads `scratch.edge_output` after this returns to
    /// aggregate it into the successor's input. Returning `()` (rather than a
    /// borrow) keeps the borrow of `scratch` confined to this call, so the
    /// topology engine can immediately access `scratch.aggregate` afterwards.
    fn route_into(&self, from: &DenseHidden, scratch: &mut MeshScratch);

    /// Relative cost hint for EdgeBandit ranking.
    ///
    /// Identity = 0.0, LoRA = small constant, projection = higher.
    /// Used to break ties when multiple edges have similar reward.
    fn cost_hint(&self) -> f32 {
        0.0
    }

    /// Human-readable name for debugging / bandit logging.
    fn name(&self) -> &str {
        "edge"
    }
}
