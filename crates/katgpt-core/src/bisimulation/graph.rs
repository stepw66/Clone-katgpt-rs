//! `TransitionGraph` — observed transition set + sorted adjacency index
//! (Plan 324 T1.5, T1.9).
//!
//! This is the **input** to the bisimulation algorithm. It is built from a
//! stream of `(from, to, op)` observations and serves:
//!
//! 1. **Adjacency queries** — `for_each_adjacent(state, op, f)` is O(log) to
//!    locate the (state, op) window in `edges`, then a contiguous scan.
//! 2. **Canonicalization** — the builder sorts and dedups edges so two graphs
//!    constructed from the same multiset of transitions are byte-identical.
//! 3. **Bound checks** — `n_states()` / `n_edges()` cap partition-refinement
//!    memory at construction time.
//!
//! # Layout invariants
//!
//! After `build()`:
//! - `states` is strictly increasing and dense `0..n`.
//! - `edges` is sorted by `(from, op.discriminant(), to)` and deduped.
//! - `edge_index` has one entry per state, holding `(state, offset)` where
//!   `offset` is the index into `edges` of the *first* edge with that `from`.
//!   The last entry's `offset` equals `edges.len()` (sentinel).
//!
//! # Latent vs raw boundary
//!
//! The graph itself is raw — the consumer builds it from observed (state, op,
//! state') triples. The state ids are dense `u32` indices; the consumer is
//! responsible for any abstraction `φ` that maps raw domain observations to
//! these dense ids *before* construction (cf. R3 in Plan 324: bisimulation
//! acts on whatever graph it's given; the right state abstraction is the
//! consumer's job).

use super::types::{OperatorLabel, StateId, Transition};

/// Sorted, deduped, indexed transition graph.
///
/// Constructed via [`TransitionGraphBuilder`]; after `build()` the struct is
/// immutable and safe to share by `&` reference across threads (all fields
/// are owned and `Vec` is `Sync` when `T: Sync`).
#[derive(Clone, Debug, Default)]
pub struct TransitionGraph {
    /// Dense `0..n` state ids, strictly increasing. The `Vec<StateId>` is
    /// redundant with the index `0..n` but is kept so the graph is
    /// self-describing (a future `from_raw_edges` constructor can verify
    /// that every `from`/`to` in `edges` appears in `states`).
    pub(crate) states: Vec<StateId>,
    /// Edges sorted by `(from, op.discriminant(), to)`, deduped. See
    /// [`Transition`] for layout.
    pub(crate) edges: Vec<Transition>,
    /// One entry per state. Entry `i` is `(StateId(i), offset)` where
    /// `offset` is the index into `edges` of the first edge whose `from ==
    /// StateId(i)`. The final entry's offset is `edges.len()` (acts as the
    /// right boundary for the last state's adjacency window).
    pub(crate) edge_index: Vec<(StateId, usize)>,
}

impl TransitionGraph {
    /// Number of states in the graph (dense `0..n`).
    #[inline]
    pub fn n_states(&self) -> u32 {
        // `states.len()` is always `u32`-safe because `StateId` itself is a
        // `u32` — if the graph had `>= 2^32` states, construction would have
        // panicked on `StateId` assignment.
        self.states.len() as u32
    }

    /// Number of edges (transitions) in the graph.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Borrow the underlying sorted edge slice. Useful for tests + the
    /// canonical-byte serialization in `refine.rs`.
    #[inline]
    pub fn edges(&self) -> &[Transition] {
        &self.edges
    }

    /// Borrow the underlying state slice.
    #[inline]
    pub fn states(&self) -> &[StateId] {
        &self.states
    }

    /// Lookup the offset window in `edges` for `state`'s outgoing edges.
    ///
    /// Returns `(start, end)` such that `self.edges[start..end]` are exactly
    /// the edges with `from == state`, sorted by `(op, to)`. If `state` has
    /// no outgoing edges, returns an empty window `(k, k)`.
    ///
    /// O(log |states|) via binary search on `edge_index`. No allocation.
    #[inline]
    pub fn adjacency_window(&self, state: StateId) -> (usize, usize) {
        // Binary search `edge_index` for `state`. The vec is sorted by
        // `StateId` ascending (it's `0..n` in order), so `partition_point`
        // gives us the index in `O(log n)`.
        let idx = self.edge_index.partition_point(|(s, _)| s.0 < state.0);
        if idx >= self.edge_index.len() || self.edge_index[idx].0 != state {
            // State not in graph — empty window.
            return (0, 0);
        }
        let start = self.edge_index[idx].1;
        // The end is the next state's start, or `edges.len()` for the last
        // state.
        let end = if idx + 1 < self.edge_index.len() {
            self.edge_index[idx + 1].1
        } else {
            self.edges.len()
        };
        (start, end)
    }

    /// Call `f` for every `(op, to)` pair outgoing from `state`, in sorted
    /// `(op, to)` order. O(degree); no allocation.
    ///
    /// This is the **hot-path** adjacency accessor used by the partition
    /// refinement loop (Phase 2) — it must remain allocation-free.
    #[inline]
    pub fn for_each_adjacent<F: FnMut(OperatorLabel, StateId)>(&self, state: StateId, mut f: F) {
        let (start, end) = self.adjacency_window(state);
        for edge in &self.edges[start..end] {
            f(edge.op, edge.to);
        }
    }

    /// True iff the graph contains no states (and therefore no edges).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

// ─── Builder ───────────────────────────────────────────────────────────────

/// Streaming builder for [`TransitionGraph`].
///
/// Push transitions in any order; `build()` sorts, dedups, builds the
/// `edge_index`, and verifies that every `from`/`to` is a known state.
pub struct TransitionGraphBuilder {
    states: Vec<StateId>,
    edges: Vec<Transition>,
    seen_states: Vec<bool>,
    max_state_plus_one: u32,
}

impl Default for TransitionGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionGraphBuilder {
    /// Construct an empty builder.
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            edges: Vec::new(),
            seen_states: Vec::new(),
            max_state_plus_one: 0,
        }
    }

    /// Construct a builder pre-sized for `n_states` states and `n_edges`
    /// edges. Avoids reallocation during `push_transition` for the common
    /// case where the caller knows the bounds up-front.
    pub fn with_capacity(n_states: usize, n_edges: usize) -> Self {
        Self {
            states: Vec::with_capacity(n_states),
            edges: Vec::with_capacity(n_edges),
            seen_states: Vec::with_capacity(n_states),
            max_state_plus_one: 0,
        }
    }

    /// Record a transition observation. States are auto-registered: the
    /// first time a `StateId` appears (as `from` or `to`) it is added to
    /// the state set.
    ///
    /// Duplicate transitions (same `(from, to, op)`) are kept as separate
    /// observations here — `build()` dedups them. This lets the caller
    /// observe a stream of (possibly duplicated) trajectories without a
    /// pre-dedup pass.
    pub fn push_transition(&mut self, from: StateId, to: StateId, op: OperatorLabel) {
        self.register_state(from);
        self.register_state(to);
        self.edges.push(Transition::new(from, to, op));
    }

    /// Register a state id, growing the `seen_states` bitset if necessary.
    fn register_state(&mut self, s: StateId) {
        let idx = s.0 as usize;
        if idx >= self.max_state_plus_one as usize {
            // Grow seen_states to idx+1.
            let new_len = (idx + 1).next_power_of_two().max(16);
            self.seen_states.resize(new_len, false);
            self.max_state_plus_one = new_len as u32;
        }
        if !self.seen_states[idx] {
            self.seen_states[idx] = true;
            self.states.push(s);
        }
    }

    /// Finalize the graph. Sorts states (ascending), sorts + dedups edges
    /// (by `(from, op, to)`), and builds the `edge_index`.
    ///
    /// After `build()` the builder is consumed and cannot be reused. If
    /// you need a fresh builder, call [`Self::new`] again.
    pub fn build(mut self) -> TransitionGraph {
        // 1. Sort + dedup states.
        self.states.sort_unstable();
        self.states.dedup();

        // 2. Sort + dedup edges.
        self.edges.sort_unstable();
        self.edges.dedup();

        // 3. Build the edge_index: one entry per state.
        //
        // `edge_index[k] = (StateId(k), offset)` where `offset` is the index
        // into `edges` of the FIRST edge with `from == StateId(k)`. For
        // zero-out-degree states, `offset` points to where the next state's
        // edges begin, so the adjacency window `(offset, next_offset)` is
        // empty.
        let mut edge_index: Vec<(StateId, usize)> = Vec::with_capacity(self.states.len());
        let mut cur_state_idx: usize = 0;
        for (i, edge) in self.edges.iter().enumerate() {
            // Flush zero-out-degree states that sort before `edge.from`.
            // Their offset is `i` — the current edge position, which is
            // also where the NEXT state with outgoing edges will start.
            // This makes their adjacency window empty: (i, i).
            while cur_state_idx < self.states.len() && self.states[cur_state_idx] < edge.from {
                edge_index.push((self.states[cur_state_idx], i));
                cur_state_idx += 1;
            }
            // First edge of a new state — record its starting offset.
            if cur_state_idx < self.states.len()
                && self.states[cur_state_idx] == edge.from
                && edge_index.len() == cur_state_idx
            {
                edge_index.push((self.states[cur_state_idx], i));
                cur_state_idx += 1;
            }
        }
        // Flush remaining zero-out-degree states at the tail.
        while cur_state_idx < self.states.len() {
            edge_index.push((self.states[cur_state_idx], self.edges.len()));
            cur_state_idx += 1;
        }

        TransitionGraph {
            states: self.states,
            edges: self.edges,
            edge_index,
        }
    }
}

// Convenience comparison for tests + canonical-byte construction.
impl PartialEq for TransitionGraph {
    fn eq(&self, other: &Self) -> bool {
        // Field-wise equality — `Vec` derives `PartialEq` so this Just Works.
        // We hand-write it only because we don't want to derive `Eq` on the
        // struct (the builder is mutable and the invariant "edge_index is
        // consistent with edges" isn't statically enforced).
        self.states == other.states
            && self.edges == other.edges
            && self.edge_index == other.edge_index
    }
}

impl Eq for TransitionGraph {}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: u32) -> StateId {
        StateId(v)
    }

    #[test]
    fn empty_builder_yields_empty_graph() {
        let g = TransitionGraphBuilder::new().build();
        assert_eq!(g.n_states(), 0);
        assert_eq!(g.n_edges(), 0);
        assert!(g.is_empty());
        assert_eq!(g.adjacency_window(s(0)), (0, 0));
    }

    #[test]
    fn builder_sorts_and_dedups_edges() {
        // Push in arbitrary order, including a duplicate.
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(2), s(0), OperatorLabel::PickTop);
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(0), s(1), OperatorLabel::PickTop); // dup
        b.push_transition(s(1), s(2), OperatorLabel::PlaceOn);
        b.push_transition(s(0), s(2), OperatorLabel::PickTop);
        let g = b.build();

        // 3 states (0, 1, 2), 4 unique edges after dedup.
        assert_eq!(g.n_states(), 3);
        assert_eq!(g.n_edges(), 4);

        // Edges must be sorted by (from, op, to):
        // (0, PickTop, 1), (0, PickTop, 2), (1, PlaceOn, 2), (2, PickTop, 0)
        let e = g.edges();
        assert_eq!(e[0], Transition::new(s(0), s(1), OperatorLabel::PickTop));
        assert_eq!(e[1], Transition::new(s(0), s(2), OperatorLabel::PickTop));
        assert_eq!(e[2], Transition::new(s(1), s(2), OperatorLabel::PlaceOn));
        assert_eq!(e[3], Transition::new(s(2), s(0), OperatorLabel::PickTop));
    }

    #[test]
    fn adjacency_window_is_correct() {
        let mut b = TransitionGraphBuilder::new();
        // State 0 has 2 outgoing edges, state 1 has 1, state 2 has 0.
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(0), s(2), OperatorLabel::PlaceOn);
        b.push_transition(s(1), s(2), OperatorLabel::PickTop);
        let g = b.build();

        // State 0: edges[0..2]
        assert_eq!(g.adjacency_window(s(0)), (0, 2));
        // State 1: edges[2..3]
        assert_eq!(g.adjacency_window(s(1)), (2, 3));
        // State 2: zero-out-degree — empty window.
        assert_eq!(g.adjacency_window(s(2)), (3, 3));

        // State 99 doesn't exist.
        assert_eq!(g.adjacency_window(s(99)), (0, 0));
    }

    #[test]
    fn for_each_adjacent_visits_in_sorted_order() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(2), OperatorLabel::PickTop);
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(0), s(3), OperatorLabel::PlaceOn);
        let g = b.build();

        let mut visited: Vec<(OperatorLabel, u32)> = Vec::new();
        g.for_each_adjacent(s(0), |op, to| visited.push((op, to.0)));

        // Sorted by (op.discriminant, to): PickTop before PlaceOn.
        // PickTop edges: (1), (2) — sorted by to.
        // PlaceOn edges: (3).
        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], (OperatorLabel::PickTop, 1));
        assert_eq!(visited[1], (OperatorLabel::PickTop, 2));
        assert_eq!(visited[2], (OperatorLabel::PlaceOn, 3));
    }

    #[test]
    fn isolated_state_is_in_state_set() {
        let mut b = TransitionGraphBuilder::new();
        // State 5 appears only as `to`, never as `from`.
        b.push_transition(s(0), s(5), OperatorLabel::NoOp);
        let g = b.build();
        assert_eq!(g.n_states(), 2);
        assert!(g.states().contains(&s(5)));
        // State 5 has zero out-degree.
        assert_eq!(g.adjacency_window(s(5)), (1, 1));
    }

    #[test]
    fn self_loop_is_supported() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(0), OperatorLabel::NoOp);
        let g = b.build();
        assert_eq!(g.n_states(), 1);
        assert_eq!(g.n_edges(), 1);
        assert_eq!(g.adjacency_window(s(0)), (0, 1));
    }
}
