//! LayerwiseTopology — the graph orchestration for DenseMesh.
//!
//! Implements paper §3.1.3: a layer-wise fully-connected directed graph where
//! each vertex is a [`DenseNode`] and each edge is a [`DenseEdge`]. Within a
//! layer, vertexes share the same node (vertex parameter sharing, §3.3).

use std::boxed::Box;
use std::vec::Vec;

use super::traits::{DenseEdge, DenseNode};
use super::types::{DenseHidden, MeshConfig, MeshScratch, Topology};

/// Errors from topology construction or forward.
#[derive(Debug)]
pub enum TopologyError {
    /// Topology widths and node list disagree.
    ShapeMismatch { expected: usize, got: usize },
    /// Edge matrix has wrong dimensions for the topology.
    EdgeMatrixMismatch { layer: usize },
    /// Hidden dimensions don't line up across a layer boundary.
    HiddenDimMismatch { layer: usize, from: usize, to: usize },
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
        })
    }

    /// Convenience: build a chain `[1, 1, 1]` with a single edge.
    ///
    /// Used for GOAT gate 1 (correctness baseline).
    pub fn chain_with_edge(
        node: Box<dyn DenseNode>,
        edge: Box<dyn DenseEdge>,
    ) -> Result<Self, TopologyError> {
        Self::new(
            Topology::chain(),
            node,
            vec![vec![edge]],
        )
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
    /// `_config` is reserved for future compute-routing integration; currently
    /// the topology runs all layers on CPU (the actual CPU/GPU/ANE dispatch is
    /// performed by the caller based on [`MeshConfig::pick_compute`]).
    pub fn forward(
        &self,
        input: &DenseHidden,
        scratch: &mut MeshScratch,
        _config: &MeshConfig,
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

            // For each successor node in layer l+1, aggregate incoming edges.
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

            current = next;
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

    /// The topology shape.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::edge_identity::IdentityEdge;
    use super::super::types::DenseHidden;

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
        assert!(matches!(result, Err(TopologyError::EdgeMatrixMismatch { .. })));
    }
}
