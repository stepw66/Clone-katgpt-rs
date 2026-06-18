//! `PtgRecorder` — incremental builder for [`PrimitiveTransitionGraph`]s.
//!
//! Wraps any execution of a [`crate::traits::ConstraintPruner`] (or any other
//! producer) and materializes a PTG node-by-node, edge-by-edge. The recorder
//! keeps `id == index` invariant: [`PtgRecorder::enter`] returns the index of
//! the node it just pushed.
//!
//! ## Hot path
//!
//! Every method is `#[inline]` and the inner [`Vec`]s pre-reserve capacity 16
//! for typical short traces. No allocations occur per-call beyond the
//! occasional `Vec::resize`; in steady state with a warmed-up recorder, calls
//! are push-only.
//!
//! ## Zero cost when disabled
//!
//! The whole module is `#[cfg(feature = "closure_instrument")]`. Callers that
//! want a stable code shape regardless of feature can branch on a `cfg!` at
//! the call site — no code is generated when the feature is off.

use super::{
    OperatorKind, PrimitiveKind, PrimitiveTransitionGraph, PtgEdge, PtgNode,
};

/// Index of a node inside a [`PrimitiveTransitionGraph::nodes`].
///
/// The recorder keeps `NodeId == index` so callers can pass these directly to
/// [`PtgRecorder::exit`] without a lookup.
pub type NodeId = u32;

/// Default pre-reserved capacity for both node and edge [`Vec`]s.
///
/// Most traces are < 16 nodes; spilling past this triggers one reallocation
/// but no per-call allocation thereafter.
pub const DEFAULT_TRACE_CAPACITY: usize = 16;

/// Incremental builder for a [`PrimitiveTransitionGraph`].
///
/// Construct with [`PtgRecorder::new`], drive with
/// [`PtgRecorder::enter`] / [`PtgRecorder::exit`], finalize with
/// [`PtgRecorder::finish`].
///
/// # Example
///
/// ```
/// use katgpt_core::closure::{OperatorKind, PrimitiveKind, PtgRecorder};
///
/// let mut rec = PtgRecorder::new(0);
/// let a = rec.enter(PrimitiveKind::UserDefined(0), 0, [0u8; 32]);
/// let b = rec.enter(PrimitiveKind::UserDefined(1), 1, [1u8; 32]);
/// rec.exit(a, b, OperatorKind::Sequence);
/// let ptg = rec.finish();
/// assert_eq!(ptg.nodes.len(), 2);
/// assert_eq!(ptg.edges.len(), 1);
/// ```
pub struct PtgRecorder {
    nodes: Vec<PtgNode>,
    edges: Vec<PtgEdge>,
    task_family_id: u32,
}

impl PtgRecorder {
    /// New empty recorder for a given task family.
    ///
    /// Pre-reserves [`DEFAULT_TRACE_CAPACITY`] slots for both nodes and edges
    /// (AGENTS.md "Allocation": pre-allocate, reuse across calls).
    #[inline]
    #[must_use]
    pub fn new(task_family_id: u32) -> Self {
        Self {
            nodes: Vec::with_capacity(DEFAULT_TRACE_CAPACITY),
            edges: Vec::with_capacity(DEFAULT_TRACE_CAPACITY),
            task_family_id,
        }
    }

    /// Push a node for `primitive` invoked at `tick` with input commitment
    /// `blake3_in`. Returns its node id (== its index in the final `nodes`).
    ///
    /// The first node entered becomes the root of the resulting PTG.
    #[inline]
    pub fn enter(
        &mut self,
        primitive: PrimitiveKind,
        tick: u32,
        blake3_in: [u8; 32],
    ) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(PtgNode {
            primitive,
            tick,
            blake3_in,
        });
        id
    }

    /// Push an edge `parent_id --op--> child_id`.
    ///
    /// The caller is responsible for ordering — `parent_id` is whatever makes
    /// semantic sense for the operator (e.g. the parent in a `Sequence`).
    #[inline]
    pub fn exit(&mut self, parent_id: NodeId, child_id: NodeId, op: OperatorKind) {
        self.edges.push(PtgEdge {
            op,
            from: parent_id,
            to: child_id,
        });
    }

    /// Finalize into an immutable [`PrimitiveTransitionGraph`].
    ///
    /// `root` is set to `0` if any nodes were entered, else `0` (the empty
    /// PTG's default root — there is no entry node).
    #[inline]
    #[must_use]
    pub fn finish(self) -> PrimitiveTransitionGraph {
        PrimitiveTransitionGraph {
            nodes: self.nodes,
            edges: self.edges,
            // Entry node is the first one entered (id == 0) when non-empty.
            root: 0,
            task_family_id: self.task_family_id,
        }
    }
}

// ── Tests (T1.4 + T1.5) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{commitment, serialize_postcard};

    /// Same call sequence ⇒ byte-identical PTG (determinism gate).
    #[test]
    fn recorder_is_deterministic() {
        fn build() -> PrimitiveTransitionGraph {
            let mut rec = PtgRecorder::new(7);
            let a = rec.enter(PrimitiveKind::UserDefined(0), 10, [1u8; 32]);
            let b = rec.enter(PrimitiveKind::UserDefined(1), 11, [2u8; 32]);
            rec.exit(a, b, OperatorKind::Sequence);
            rec.finish()
        }
        let bytes1 = serialize_postcard(&build()).expect("s1");
        let bytes2 = serialize_postcard(&build()).expect("s2");
        assert_eq!(bytes1, bytes2, "byte-identical PTG required");
        // And the commitment too.
        let h1 = commitment(&build());
        let h2 = commitment(&build());
        assert_eq!(h1, h2);
    }

    #[test]
    fn empty_recorder_produces_empty_ptg() {
        let rec = PtgRecorder::new(3);
        let ptg = rec.finish();
        assert_eq!(ptg.task_family_id, 3);
        assert!(ptg.nodes.is_empty());
        assert!(ptg.edges.is_empty());
        assert_eq!(ptg.root, 0);
    }

    #[test]
    fn single_node_ptg() {
        let mut rec = PtgRecorder::new(1);
        let id = rec.enter(PrimitiveKind::UserDefined(42), 100, [9u8; 32]);
        let ptg = rec.finish();
        assert_eq!(ptg.nodes.len(), 1);
        assert_eq!(ptg.edges.len(), 0);
        assert_eq!(ptg.root, 0);
        assert_eq!(id, 0);
        assert_eq!(ptg.nodes[0].primitive, PrimitiveKind::UserDefined(42));
        assert_eq!(ptg.nodes[0].tick, 100);
    }

    #[test]
    fn id_equals_index_invariant() {
        let mut rec = PtgRecorder::new(0);
        for i in 0..10u32 {
            let id = rec.enter(PrimitiveKind::UserDefined(i), i, [i as u8; 32]);
            assert_eq!(id, i, "id must equal index");
        }
        let ptg = rec.finish();
        assert_eq!(ptg.nodes.len(), 10);
        for (i, n) in ptg.nodes.iter().enumerate() {
            assert_eq!(n.primitive, PrimitiveKind::UserDefined(i as u32));
        }
    }

    /// T1.5 — compose 4 primitive nodes with various operators; verify final
    /// graph has the expected shape.
    #[test]
    fn four_node_composition_materializes() {
        let mut rec = PtgRecorder::new(5);
        // Build: A --Seq--> B --Branch--> C --ParJoin--> D
        // plus  D --Recurse--> A  (back-edge to root).
        let a = rec.enter(PrimitiveKind::UserDefined(0), 0, [0u8; 32]);
        let b = rec.enter(PrimitiveKind::UserDefined(1), 1, [1u8; 32]);
        let c = rec.enter(PrimitiveKind::UserDefined(2), 2, [2u8; 32]);
        let d = rec.enter(PrimitiveKind::UserDefined(3), 3, [3u8; 32]);
        rec.exit(a, b, OperatorKind::Sequence);
        rec.exit(b, c, OperatorKind::Branch);
        rec.exit(c, d, OperatorKind::ParallelJoin);
        rec.exit(d, a, OperatorKind::Recurse);

        let ptg = rec.finish();
        assert_eq!(ptg.nodes.len(), 4, "T1.5 nodes");
        assert_eq!(ptg.edges.len(), 4, "T1.5 edges");
        assert_eq!(ptg.root, a, "T1.5 root");
        assert_eq!(ptg.task_family_id, 5);
        // Operator sanity check.
        assert_eq!(ptg.edges[0].op, OperatorKind::Sequence);
        assert_eq!(ptg.edges[1].op, OperatorKind::Branch);
        assert_eq!(ptg.edges[2].op, OperatorKind::ParallelJoin);
        assert_eq!(ptg.edges[3].op, OperatorKind::Recurse);
    }

    /// Capacity hint avoids per-call allocation in the common case.
    #[test]
    fn capacity_hint_holds_for_short_traces() {
        let rec = PtgRecorder::new(0);
        // Both Vecs should have been pre-reserved at DEFAULT_TRACE_CAPACITY.
        assert!(rec.nodes.capacity() >= DEFAULT_TRACE_CAPACITY);
        assert!(rec.edges.capacity() >= DEFAULT_TRACE_CAPACITY);
    }
}
