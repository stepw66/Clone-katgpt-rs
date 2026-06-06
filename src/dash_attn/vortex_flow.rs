//! VortexFlow — composable sparse routing trait for KV block selection.
//!
//! Each router implements two phases:
//! 1. `forward_cache` — query-independent cache update when KV blocks are appended
//! 2. `forward_indexer` — query-dependent block selection during decode
//!
//! Feature gate: `vortex_flow` (Plan 196, Phase 1, default-OFF).

use std::fmt::Debug;

// ---------------------------------------------------------------------------
// VortexFlow trait
// ---------------------------------------------------------------------------

/// Composable sparse routing trait for KV block selection.
///
/// Implementors provide:
/// - A cache type for query-independent block summaries
/// - Cache update logic when new KV blocks arrive
/// - Query-dependent top-k block selection
pub trait VortexFlow: Send + Sync {
    /// Router-specific cache type.
    type Cache: Send + Debug;

    /// Update cache when a new KV block is appended (query-independent).
    /// Called during prefill or when KV cache grows.
    ///
    /// # Arguments
    /// * `cache` — mutable router cache to update
    /// * `keys` — flat `[block_size * head_dim]` key vectors for this block
    /// * `values` — flat `[block_size * head_dim]` value vectors for this block
    /// * `block_idx` — index of the block being cached
    /// * `head_dim` — dimension per head
    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    );

    /// Select top-k blocks for the given query (query-dependent).
    /// Called during each decode step.
    ///
    /// # Arguments
    /// * `query` — query vector `[head_dim]`
    /// * `cache` — immutable router cache
    /// * `n_blocks` — total number of blocks currently cached
    /// * `top_k` — maximum number of blocks to select
    /// * `scratch` — reusable scratch buffer for intermediate computations
    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision;

    /// Create a new cache instance pre-allocated for `n_blocks_capacity` blocks.
    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache;
}

// ---------------------------------------------------------------------------
// RoutingDecision
// ---------------------------------------------------------------------------

/// Result of routing: which blocks to attend to and their weights.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected block indices (sorted by relevance, descending).
    pub blocks: Vec<usize>,
    /// Routing weights for selected blocks.
    pub weights: Vec<f32>,
}

impl RoutingDecision {
    /// Create an empty routing decision.
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity for `top_k` entries.
    pub fn with_capacity(top_k: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(top_k),
            weights: Vec::with_capacity(top_k),
        }
    }

    /// Clear for reuse across decode steps without deallocating.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.weights.clear();
    }

    /// Number of selected blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether no blocks were selected.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl Default for RoutingDecision {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VortexScratch
// ---------------------------------------------------------------------------

/// Reusable scratch buffer for routing computations.
#[derive(Debug)]
pub struct VortexScratch {
    /// Block scores buffer `[max_blocks]`.
    pub scores: Vec<f32>,
    /// Top-k index buffer.
    pub indices: Vec<usize>,
}

impl VortexScratch {
    /// Create scratch buffers sized for `max_blocks` blocks.
    pub fn new(max_blocks: usize) -> Self {
        Self {
            scores: vec![0.0; max_blocks],
            indices: Vec::with_capacity(max_blocks),
        }
    }

    /// Ensure buffers can hold at least `n` blocks, growing if needed.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.scores.len() < n {
            self.scores.resize(n, 0.0);
        }
        if self.indices.capacity() < n {
            // Reserve enough for n total elements
            let additional = n.saturating_sub(self.indices.len());
            self.indices.reserve(additional);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_decision_new() {
        let rd = RoutingDecision::new();
        assert!(rd.is_empty());
        assert_eq!(rd.len(), 0);
    }

    #[test]
    fn test_routing_decision_with_capacity() {
        let rd = RoutingDecision::with_capacity(8);
        assert!(rd.is_empty());
        assert!(rd.blocks.capacity() >= 8);
        assert!(rd.weights.capacity() >= 8);
    }

    #[test]
    fn test_routing_decision_clear_reuse() {
        let mut rd = RoutingDecision::with_capacity(4);
        rd.blocks.push(0);
        rd.blocks.push(1);
        rd.weights.push(0.7);
        rd.weights.push(0.3);
        assert_eq!(rd.len(), 2);

        let block_cap = rd.blocks.capacity();
        let weight_cap = rd.weights.capacity();

        rd.clear();
        assert!(rd.is_empty());
        // Capacity preserved after clear
        assert_eq!(rd.blocks.capacity(), block_cap);
        assert_eq!(rd.weights.capacity(), weight_cap);
    }

    #[test]
    fn test_routing_decision_default() {
        let rd = RoutingDecision::default();
        assert!(rd.is_empty());
    }

    #[test]
    fn test_vortex_scratch_new() {
        let scratch = VortexScratch::new(16);
        assert_eq!(scratch.scores.len(), 16);
        assert!(scratch.scores.iter().all(|&s| s == 0.0));
        assert!(scratch.indices.is_empty());
        assert!(scratch.indices.capacity() >= 16);
    }

    #[test]
    fn test_vortex_scratch_ensure_capacity_grow() {
        let mut scratch = VortexScratch::new(4);
        scratch.ensure_capacity(16);
        assert!(scratch.scores.len() >= 16);
        // After ensure_capacity, pushing n elements should not reallocate
        for i in 0..16 {
            scratch.indices.push(i);
        }
        assert_eq!(scratch.indices.len(), 16);
    }

    #[test]
    fn test_vortex_scratch_ensure_capacity_noop_when_sufficient() {
        let mut scratch = VortexScratch::new(32);
        let scores_ptr = scratch.scores.as_ptr();
        scratch.ensure_capacity(16);
        // Should not reallocate
        assert_eq!(scratch.scores.as_ptr(), scores_ptr);
    }
}
