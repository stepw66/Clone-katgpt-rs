//! LayerwiseTopology — the graph orchestration for DenseMesh.
//!
//! Implements paper §3.1.3: a layer-wise fully-connected directed graph where
//! each vertex is a [`DenseNode`] and each edge is a [`DenseEdge`]. Within a
//! layer, vertexes share the same node (vertex parameter sharing, §3.3).

use std::boxed::Box;
use std::sync::Mutex;
use std::vec::Vec;

use super::traits::{DenseEdge, DenseNode};
use super::types::{DenseHidden, MeshConfig, MeshScratch, Topology};

/// Width at which the parallel path switches from per-successor rayon tasks to
/// the batched scratch-pooled dispatch (Issue 020, Path B). Below this, the
/// per-successor rayon path is fine — allocations and spawn overhead are
/// invisible against cheap IdentityNode forwards. Above it, pooling the
/// per-worker scratches and output slots becomes worthwhile (the Path A code
/// comment flagged this as "a future Path B optimisation").
const VERTEX_BATCH_THRESHOLD: usize = 4;

/// Errors from topology construction or forward.
#[derive(Debug)]
pub enum TopologyError {
    /// Topology widths and node list disagree.
    ShapeMismatch { expected: usize, got: usize },
    /// Edge matrix has wrong dimensions for the topology.
    EdgeMatrixMismatch { layer: usize },
    /// Hidden dimensions don't line up across a layer boundary.
    HiddenDimMismatch {
        layer: usize,
        from: usize,
        to: usize,
    },
}

/// A layer-wise fully-connected DenseMesh (paper §3.1.3).
///
/// Holds one shared [`DenseNode`] (vertex parameter sharing) and a matrix of
/// [`DenseEdge`]s. `edges[layer][from_idx * width_l + to_idx]` is the edge
/// from layer `l` node `from_idx` to layer `l+1` node `to_idx`.
pub struct LayerwiseTopology {
    topology: Topology,
    /// Shared vertex node (paper §3.3 — all nodes share the same LLM).
    node: Box<dyn DenseNode>,
    /// Edge matrix, indexed `[layer][from * width_next + to]`.
    /// Length per layer = widths[l] * widths[l+1].
    edges: Vec<Vec<Box<dyn DenseEdge>>>,
    /// Pooled per-worker scratch buffers for the batched parallel path
    /// (Issue 020, Path B). Allocated once on first use, grown as topology
    /// width demands, reused across `forward()` calls — eliminates the
    /// per-call `Vec<MeshScratch>` allocation that Path A's comment flagged as
    /// "out of scope for the ~50 LoC target". Locked once per forward (not
    /// per rayon task) so contention is negligible.
    scratch_pool: Mutex<Vec<MeshScratch>>,
    /// Pooled output slots (`next` buffer) for the batched parallel path.
    /// Same lifetime pattern as `scratch_pool`: allocated once, reused.
    output_pool: Mutex<Vec<DenseHidden>>,
}

impl LayerwiseTopology {
    /// Build a topology with a shared node and a flat edge list.
    ///
    /// `edges_per_layer[l]` must have length `widths[l] * widths[l+1]`.
    pub fn new(
        topology: Topology,
        node: Box<dyn DenseNode>,
        edges_per_layer: Vec<Vec<Box<dyn DenseEdge>>>,
    ) -> Result<Self, TopologyError> {
        if edges_per_layer.len() + 1 != topology.widths.len() {
            return Err(TopologyError::ShapeMismatch {
                expected: topology.widths.len() - 1,
                got: edges_per_layer.len(),
            });
        }
        for (l, layer_edges) in edges_per_layer.iter().enumerate() {
            let expected = topology.widths[l] * topology.widths[l + 1];
            if layer_edges.len() != expected {
                return Err(TopologyError::EdgeMatrixMismatch { layer: l });
            }
        }
        Ok(Self {
            topology,
            node,
            edges: edges_per_layer,
            scratch_pool: Mutex::new(Vec::new()),
            output_pool: Mutex::new(Vec::new()),
        })
    }

    /// Convenience: build a chain `[1, 1, 1]` with a single edge.
    ///
    /// Used for GOAT gate 1 (correctness baseline).
    pub fn chain_with_edge(
        node: Box<dyn DenseNode>,
        edge: Box<dyn DenseEdge>,
    ) -> Result<Self, TopologyError> {
        Self::new(Topology::chain(), node, vec![vec![edge]])
    }

    /// Forward pass through the mesh.
    ///
    /// Paper §3.1.3 eq. (1): for each layer l+1, for each node j in l+1,
    /// aggregate (sum) the outputs of all edges from layer l into j, then
    /// run the node forward on the aggregated input.
    ///
    /// `input` is the first layer's input (e.g., token embeddings).
    /// Returns the last layer's output hidden state.
    ///
    /// `config` controls compute routing. When
    /// [`MeshConfig::enable_vertex_parallelism`] is set and a hidden layer's
    /// width ≥ [`MeshConfig::gpu_width_threshold`], the per-successor-node
    /// work (edge aggregation + node forward) is dispatched to a rayon
    /// parallel iterator so the width-many shared-LLM forwards execute in
    /// parallel — paper §3.3 vertex parameter sharing + Issue 020 Path A.
    /// Below the threshold (or with parallelism disabled) the path stays
    /// sequential to avoid rayon spawn overhead on trivial nodes.
    pub fn forward(
        &self,
        input: &DenseHidden,
        scratch: &mut MeshScratch,
        config: &MeshConfig,
    ) -> DenseHidden {
        // Current layer's hidden states: width-many DenseHidden buffers.
        // Layer 0 has width 1 in all our topologies (input node).
        // For generality we handle width-many at every layer.
        let mut current: Vec<DenseHidden> = {
            let w0 = self.topology.width(0);
            // Replicate input into w0 copies (usually w0 == 1).
            (0..w0).map(|_| input.clone()).collect()
        };

        for l in 0..self.topology.depth() - 1 {
            let width_l = self.topology.width(l);
            let width_next = self.topology.width(l + 1);

            // Parallel vertex path (Issue 020, Path A + Path B). Two modes:
            //
            //   - Path A (`width_next < VERTEX_BATCH_THRESHOLD` or pooling not
            //     yet profitable): per-successor rayon task with fresh
            //     per-call scratch/output allocation. Matches the original
            //     Path A behaviour exactly.
            //   - Path B (`width_next >= VERTEX_BATCH_THRESHOLD`): the same
            //     per-successor rayon parallelism, but using pooled scratch +
            //     output slots from `self.scratch_pool` / `self.output_pool`.
            //     This eliminates the per-call `Vec<MeshScratch>` and
            //     `Vec<DenseHidden>` allocations the Path A code flagged as a
            //     "future Path B optimisation". The transformer forward itself
            //     still runs one `forward_dense` per successor — full
            //     `transformer::forward_batched` fusion requires a DenseNode
            //     trait extension (documented in issue 020 as follow-up).
            //
            // Both paths preserve Path A's per-thread `TransformerNode` (ctx,
            // cache) pool isolation — no data race on the shared vertex.
            if config.enable_vertex_parallelism && width_next >= config.gpu_width_threshold {
                if width_next >= VERTEX_BATCH_THRESHOLD {
                    current =
                        self.forward_layer_parallel_pooled(l, width_l, width_next, &current, input);
                } else {
                    current = self.forward_layer_parallel(l, width_l, width_next, &current, input);
                }
            } else {
                current = self.forward_layer_sequential(l, width_l, width_next, &current, scratch);
            }
        }

        // After the last layer, `current` has width-many buffers. For an output
        // layer of width 1, there's exactly one. Return the first (or sum if >1).
        if current.len() == 1 {
            current.into_iter().next().unwrap()
        } else {
            // Multiple output nodes — sum them (rare; paper uses width-1 output).
            let mut iter = current.into_iter();
            let mut acc = iter.next().unwrap();
            for h in iter {
                acc.add_assign(&h);
            }
            acc
        }
    }

    /// Sequential per-layer forward (paper §3.1.3 eq. (1)).
    ///
    /// Reuses the caller-provided `scratch` for aggregation/edge output —
    /// plasma-tier zero-alloc in the steady state.
    #[allow(clippy::needless_range_loop)] // stride math: i indexes current[i] AND i*width_next+j offset into edges_l
    fn forward_layer_sequential(
        &self,
        l: usize,
        width_l: usize,
        width_next: usize,
        current: &[DenseHidden],
        scratch: &mut MeshScratch,
    ) -> Vec<DenseHidden> {
        let mut next: Vec<DenseHidden> = Vec::with_capacity(width_next);
        for j in 0..width_next {
            scratch.aggregate.clear();
            let mut any = false;
            for i in 0..width_l {
                let edge_idx = i * width_next + j;
                // Route predecessor i's hidden state through the edge.
                // Writes into scratch.edge_output; returns ().
                self.edges[l][edge_idx].route_into(&current[i], scratch);
                let edge_len = scratch.edge_output.len();
                let seq_len = current[i].seq_len;
                // Now safe to read scratch.edge_output (mutable borrow released).
                if !any {
                    // First contributor: copy.
                    scratch.aggregate.data[..edge_len]
                        .copy_from_slice(&scratch.edge_output.data[..edge_len]);
                    scratch.aggregate.seq_len = seq_len;
                    scratch.aggregate.hidden_dim = edge_len / seq_len.max(1);
                    any = true;
                } else {
                    for (dst, src) in scratch.aggregate.data[..edge_len]
                        .iter_mut()
                        .zip(scratch.edge_output.data[..edge_len].iter())
                    {
                        *dst += *src;
                    }
                }
            }
            // Forward the aggregated input through the shared node.
            // Clone aggregate out so node can borrow scratch mutably.
            let agg = scratch.aggregate.clone();
            let out = self.node.forward_dense(&agg, l + 1, scratch);
            next.push(out);
        }
        next
    }

    /// Parallel per-layer forward (Issue 020, Path A).
    ///
    /// Each successor node j's work (edge aggregation + node forward) runs in
    /// its own rayon task. Because the shared `&self.node` is `Send + Sync`
    /// (see `DenseNode` trait bounds), rayon can share the reference across
    /// workers. Each worker gets its own `MeshScratch` for edge aggregation —
    /// obtained as a disjoint `&mut` via `zip` of two `par_iter_mut` slices, so
    /// there is no shared mutable scratch and no `unsafe`. The node itself is
    /// responsible for per-thread isolation of its internal
    /// `(ForwardContext, MultiLayerKVCache)` state (see `TransformerNode`'s
    /// pool keyed by `rayon::current_thread_index()`).
    ///
    /// `_input` is unused; edges route from `current`. Kept in the signature
    /// for symmetry with [`Self::forward_layer_sequential`] and future routing.
    #[allow(clippy::needless_range_loop)] // stride math: i indexes current[i] AND i*width_next+j offset into edges_l
    fn forward_layer_parallel(
        &self,
        l: usize,
        width_l: usize,
        width_next: usize,
        current: &[DenseHidden],
        _input: &DenseHidden,
    ) -> Vec<DenseHidden> {
        use rayon::prelude::*;

        // Size per-worker scratch from the predecessor hidden shape.
        let seq_len = current.first().map(|h| h.seq_len).unwrap_or(1);
        let hidden_dim = current.first().map(|h| h.hidden_dim).unwrap_or(1);

        // Pre-allocate output slots and per-worker scratch. Both are moved into
        // parallel iterators below; rayon splits them into disjoint `&mut`
        // chunks per task, so no two tasks ever alias.
        let mut next: Vec<DenseHidden> = (0..width_next)
            .map(|_| DenseHidden::zeros(seq_len, hidden_dim))
            .collect();
        // One scratch per successor node. Allocated per mesh-forward call; for
        // width-4 that is 4 scratch allocations, dwarfed by the transformer
        // forward cost. (Pooling these in the topology is a future Path B
        // optimisation; out of scope for the ~50 LoC Path A target.)
        let mut local_scratches: Vec<MeshScratch> = (0..width_next)
            .map(|_| MeshScratch::new(seq_len, hidden_dim))
            .collect();

        // Shared (Sync) captures — edges + node + predecessor hidden states.
        let edges_l = &self.edges[l];
        let node = &self.node;

        // Zip the output slots with their per-worker scratch and enumerate so
        // each task knows its successor index `j` (needed for edge lookup).
        // `par_iter_mut().zip(par_iter_mut()).enumerate()` is the idiomatic
        // rayon pattern for fan-out writes into two parallel pre-allocated
        // buffers: the iterator implementation proves disjointness, so no
        // `unsafe` is needed.
        next.par_iter_mut()
            .zip(local_scratches.par_iter_mut())
            .enumerate()
            .for_each(|(j, (out_slot, scratch_j))| {
                // Aggregate edges from all predecessors i into successor j.
                scratch_j.aggregate.clear();
                let mut any = false;
                for i in 0..width_l {
                    let edge_idx = i * width_next + j;
                    edges_l[edge_idx].route_into(&current[i], scratch_j);
                    let edge_len = scratch_j.edge_output.len();
                    let seq_len_i = current[i].seq_len;
                    if !any {
                        scratch_j.aggregate.data[..edge_len]
                            .copy_from_slice(&scratch_j.edge_output.data[..edge_len]);
                        scratch_j.aggregate.seq_len = seq_len_i;
                        scratch_j.aggregate.hidden_dim = edge_len / seq_len_i.max(1);
                        any = true;
                    } else {
                        for (dst, src) in scratch_j.aggregate.data[..edge_len]
                            .iter_mut()
                            .zip(scratch_j.edge_output.data[..edge_len].iter())
                        {
                            *dst += *src;
                        }
                    }
                }

                // Forward the aggregated input through the shared node. Each
                // rayon worker picks up its own (ctx, cache) slot from the
                // node's internal pool — see TransformerNode::forward_dense.
                let agg = scratch_j.aggregate.clone();
                let out = node.forward_dense(&agg, l + 1, scratch_j);
                *out_slot = out;
            });

        next
    }

    /// Parallel per-layer forward with pooled scratch/output (Issue 020, Path B).
    ///
    /// Identical compute shape to [`Self::forward_layer_parallel`] — each
    /// successor node's edge aggregation + node forward runs in its own rayon
    /// task. The difference is allocation: instead of `Vec<MeshScratch>` and
    /// `Vec<DenseHidden>` freshly allocated per call (Path A's "future Path B
    /// optimisation" footnote), these buffers are drawn from
    /// [`Self::scratch_pool`] / [`Self::output_pool`] and returned after the
    /// layer completes. At width 4 with `vocab_size = 4096` this saves ~8
    /// allocations (4 scratch + 4 output DenseHidden) per hidden-layer
    /// transition — the pooled path pays zero allocator cost after the first
    /// call.
    ///
    /// Full `transformer::forward_batched` fusion (one matmul-batched call for
    /// all width-many successors) is NOT done here because the `DenseNode`
    /// trait exposes only `forward_dense` — the shared `TransformerNode`'s
    /// `weights` / `ctx_pool` / `cache_pool` are private. That fusion is
    /// documented in issue 020 as a DenseNode trait extension follow-up.
    #[allow(clippy::needless_range_loop)] // stride math: i indexes current[i] AND i*width_next+j offset into edges_l
    fn forward_layer_parallel_pooled(
        &self,
        l: usize,
        width_l: usize,
        width_next: usize,
        current: &[DenseHidden],
        _input: &DenseHidden,
    ) -> Vec<DenseHidden> {
        use rayon::prelude::*;

        let seq_len = current.first().map(|h| h.seq_len).unwrap_or(1);
        let hidden_dim = current.first().map(|h| h.hidden_dim).unwrap_or(1);

        // Drain pooled scratch + output. Grown on first call (or width bump),
        // then stable — no allocator traffic in steady state.
        let mut scratches: Vec<MeshScratch> = {
            let mut guard = self.scratch_pool.lock().expect("scratch_pool poisoned");
            std::mem::take(&mut *guard)
        };
        let mut next: Vec<DenseHidden> = {
            let mut guard = self.output_pool.lock().expect("output_pool poisoned");
            std::mem::take(&mut *guard)
        };
        while scratches.len() < width_next {
            scratches.push(MeshScratch::new(seq_len, hidden_dim));
        }
        // Truncate extra pooled slots if the topology shrank; keep capacity.
        scratches.truncate(width_next);
        // Ensure `next` has exactly `width_next` slots of the right shape.
        while next.len() < width_next {
            next.push(DenseHidden::zeros(seq_len, hidden_dim));
        }
        next.truncate(width_next);
        // If shapes changed since last call (rare — topology or model resized),
        // re-zero the slots to their new shape.
        for slot in next.iter_mut() {
            if slot.seq_len != seq_len || slot.hidden_dim != hidden_dim {
                *slot = DenseHidden::zeros(seq_len, hidden_dim);
            }
        }

        let edges_l = &self.edges[l];
        let node = &self.node;

        next.par_iter_mut()
            .zip(scratches.par_iter_mut())
            .enumerate()
            .for_each(|(j, (out_slot, scratch_j))| {
                scratch_j.aggregate.clear();
                let mut any = false;
                for i in 0..width_l {
                    let edge_idx = i * width_next + j;
                    edges_l[edge_idx].route_into(&current[i], scratch_j);
                    let edge_len = scratch_j.edge_output.len();
                    let seq_len_i = current[i].seq_len;
                    if !any {
                        scratch_j.aggregate.data[..edge_len]
                            .copy_from_slice(&scratch_j.edge_output.data[..edge_len]);
                        scratch_j.aggregate.seq_len = seq_len_i;
                        scratch_j.aggregate.hidden_dim = edge_len / seq_len_i.max(1);
                        any = true;
                    } else {
                        for (dst, src) in scratch_j.aggregate.data[..edge_len]
                            .iter_mut()
                            .zip(scratch_j.edge_output.data[..edge_len].iter())
                        {
                            *dst += *src;
                        }
                    }
                }

                let agg = scratch_j.aggregate.clone();
                let out = node.forward_dense(&agg, l + 1, scratch_j);
                *out_slot = out;
            });

        // Return buffers to the pool for the next call. Any prior capacity is
        // preserved (pushes don't shrink the underlying allocation).
        {
            let mut guard = self.scratch_pool.lock().expect("scratch_pool poisoned");
            *guard = scratches;
        }

        next
    }

    /// The topology shape.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
}

#[cfg(test)]
mod tests {
    use super::super::edge_identity::IdentityEdge;
    use super::super::types::DenseHidden;
    use super::*;

    /// A trivial node that passes input through unchanged (for gate 1).
    struct IdentityNode {
        hidden_dim: usize,
    }

    impl DenseNode for IdentityNode {
        fn forward_dense(
            &self,
            input: &DenseHidden,
            _layer_idx: usize,
            _scratch: &mut MeshScratch,
        ) -> DenseHidden {
            input.clone()
        }
        #[inline]
        fn hidden_dim(&self) -> usize {
            self.hidden_dim
        }
    }

    #[test]
    fn test_chain_identity_topology_preserves_input() {
        // Gate 1: chain [1,1,1] + IdentityEdge + IdentityNode should preserve input.
        let node = Box::new(IdentityNode { hidden_dim: 4 });
        let edge = Box::new(IdentityEdge::new());
        let topo = LayerwiseTopology::chain_with_edge(node, edge).unwrap();

        let mut input = DenseHidden::zeros(2, 4);
        for (i, v) in input.rows_mut().iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }
        let mut scratch = MeshScratch::new(2, 4);
        let cfg = MeshConfig::default();
        let out = topo.forward(&input, &mut scratch, &cfg);
        // Output should equal input.
        for (a, b) in out.rows().iter().zip(input.rows().iter()) {
            assert!((a - b).abs() < 1e-6, "chain+identity must preserve input");
        }
    }

    #[test]
    fn test_topology_shape_mismatch_errors() {
        let node = Box::new(IdentityNode { hidden_dim: 4 });
        // Topology chain needs 1 layer of edges (2 boundaries), but we give 2.
        let result = LayerwiseTopology::new(
            Topology::chain(),
            node,
            vec![vec![Box::new(IdentityEdge::new())], vec![]],
        );
        assert!(matches!(result, Err(TopologyError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_edge_matrix_mismatch_errors() {
        let node = Box::new(IdentityNode { hidden_dim: 4 });
        // Diamond topology [1,2,1] needs 2 edge layers: layer 0 needs 2 edges
        // (1*2), layer 1 needs 2 edges (2*1). We give correct layer count but
        // wrong edge count in layer 0.
        let result = LayerwiseTopology::new(
            Topology::diamond(),
            node,
            vec![
                vec![Box::new(IdentityEdge::new())], // 1 edge, expected 2
                vec![Box::new(IdentityEdge::new()), Box::new(IdentityEdge::new())],
            ],
        );
        assert!(matches!(
            result,
            Err(TopologyError::EdgeMatrixMismatch { .. })
        ));
    }
}
