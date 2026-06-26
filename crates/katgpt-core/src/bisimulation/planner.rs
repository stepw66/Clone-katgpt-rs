//! `planner` — BFS planner over a bisimulation quotient's operator-labeled
//! edges (Plan 324 T4.1–T4.3).
//!
//! Given an [`OperatorSchema`] inferred from a [`BisimulationQuotient`], find
//! the shortest operator-label sequence that takes `start_class` to
//! `goal_class`. Breadth-first search over the quotient graph is sufficient
//! for G3 (plan validity): BFS finds the minimum-length plan, and replaying
//! it against the original [`TransitionGraph`] verifies no precondition
//! violations.
//!
//! **Out of scope:** MetricFF-grade classical planning, heuristic search,
//! cost-optimal planning with arbitrary cost models. The quotient graph is
//! small (post-bisimulation), so BFS is adequate. Downstream consumers that
//! need richer planning can wire up [`crate::induced_cwm::ismcts`] or an
//! external PDDL solver.

use super::operator::OperatorSchema;
use super::refine::BisimulationQuotient;
use super::types::{OperatorLabel, StateClassId};
use std::collections::VecDeque;

/// Result of a planning query: either a sequence of operator labels or
/// `None` if no path exists in the quotient graph.
pub type Plan = Vec<OperatorLabel>;

/// Find the shortest operator-label sequence from `start` to `goal` in the
/// quotient graph (Plan 324 T4.1).
///
/// BFS over quotient edges. Returns `Some(sequence)` if a path exists,
/// `None` otherwise. The returned sequence is minimum-length (BFS
/// guarantee) but not necessarily unique.
///
/// # Behavior
///
/// - If `start == goal`, returns `Some(vec![])` (empty plan — already at
///   goal).
/// - If no path exists, returns `None`.
///
/// # Complexity
///
/// `O(n_classes + n_quotient_edges)` — standard BFS over the quotient graph.
pub fn plan(
    _schema: &OperatorSchema,
    quotient: &BisimulationQuotient,
    start: StateClassId,
    goal: StateClassId,
) -> Option<Plan> {
    // Trivial case: already at goal.
    if start == goal {
        return Some(Vec::new());
    }

    let n = quotient.n_classes as usize;
    if start.0 as usize >= n || goal.0 as usize >= n {
        // Out-of-range class ids → no plan (shouldn't happen for well-formed
        // inputs; defensive guard).
        return None;
    }

    // BFS state.
    // `visited[c]` = predecessor class + operator label that reached `c`.
    // `None` = unvisited; `Some((pred, op))` = visited via `(pred --op--> c)`.
    // The start class is marked visited with a sentinel predecessor.
    let mut visited: Vec<Option<(StateClassId, OperatorLabel)>> = vec![None; n];
    // Mark start as visited with a self-loop sentinel so the reconstruction
    // loop knows where to stop.
    visited[start.0 as usize] = Some((start, OperatorLabel::NoOp));

    let mut queue: VecDeque<StateClassId> = VecDeque::with_capacity(n);
    queue.push_back(start);

    while let Some(cur) = queue.pop_front() {
        // Iterate outgoing edges of `cur` in the quotient. The quotient's
        // edges are already sorted by (from, op, to), so for a given `cur`
        // we can binary-search — but `n_quotient_edges` is small enough
        // (post-bisimulation) that a linear scan is fine.
        for edge in &quotient.quotient_edges {
            if edge.from != cur {
                continue;
            }
            let next = edge.to;
            if visited[next.0 as usize].is_some() {
                continue; // already visited
            }
            visited[next.0 as usize] = Some((cur, edge.op));
            if next == goal {
                // Reconstruct the plan by walking predecessors backward.
                return Some(reconstruct_plan(&visited, start, goal));
            }
            queue.push_back(next);
        }
    }

    // Goal not reachable from start.
    None
}

/// Walk `visited` backward from `goal` to `start`, collecting operator
/// labels in reverse, then reverse the result.
fn reconstruct_plan(
    visited: &[Option<(StateClassId, OperatorLabel)>],
    start: StateClassId,
    goal: StateClassId,
) -> Plan {
    let mut plan_rev: Plan = Vec::new();
    let mut cur = goal;
    while cur != start {
        let (pred, op) = visited[cur.0 as usize].expect("visited chain must be intact");
        plan_rev.push(op);
        cur = pred;
    }
    plan_rev.reverse();
    plan_rev
}

// ─── Replay validation (G3) ────────────────────────────────────────────────

impl OperatorSchema {
    /// Replay a plan against the quotient graph, verifying that every step
    /// respects operator preconditions and that the final class is `goal`.
    ///
    /// Returns `Ok(final_class)` if the replay succeeds and lands on `goal`,
    /// or `Err(step_index)` with the index of the first failing step.
    ///
    /// Used by the G3 plan-validity tests (Plan 324 T4.2).
    pub fn replay_plan(
        &self,
        quotient: &BisimulationQuotient,
        start: StateClassId,
        plan: &[OperatorLabel],
        goal: StateClassId,
    ) -> Result<StateClassId, usize> {
        let mut cur = start;
        for (i, &op) in plan.iter().enumerate() {
            // Check precondition: `cur` must admit `op`.
            if !self.admits(cur, op) {
                return Err(i);
            }
            // Find the quotient edge `(cur, op, next)` — there may be
            // multiple; pick the first (BFS found one valid path, any
            // valid edge works for replay).
            let next = quotient
                .quotient_edges
                .iter()
                .find(|e| e.from == cur && e.op == op)
                .map(|e| e.to);
            match next {
                Some(n) => cur = n,
                None => return Err(i),
            }
        }
        if cur == goal {
            Ok(cur)
        } else {
            // Reached end of plan but not at goal — return plan.len() as
            // the "failure step" (one past the last step).
            Err(plan.len())
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bisimulation::graph::TransitionGraphBuilder;
    use crate::bisimulation::operator::infer_operators;
    use crate::bisimulation::refine::partition_refine;
    use crate::bisimulation::types::{OperatorLabel, StateId};

    fn s(v: u32) -> StateId {
        StateId(v)
    }

    fn c(v: u32) -> StateClassId {
        StateClassId(v)
    }

    #[test]
    fn empty_plan_when_start_equals_goal() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        let g = b.build();
        let q = partition_refine(&g);
        let schema = infer_operators(&q);

        let p = plan(&schema, &q, c(0), c(0)).expect("start==goal → empty plan");
        assert!(p.is_empty());
    }

    #[test]
    fn single_step_plan() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        let g = b.build();
        let q = partition_refine(&g);
        let schema = infer_operators(&q);

        // Class 0 → class 1 via PickTop.
        let p = plan(&schema, &q, c(0), c(1)).expect("path must exist");
        assert_eq!(p, vec![OperatorLabel::PickTop]);

        // Replay must succeed.
        let final_class = schema
            .replay_plan(&q, c(0), &p, c(1))
            .expect("replay must succeed");
        assert_eq!(final_class, c(1));
    }

    #[test]
    fn multi_step_plan() {
        // 0 --A--> 1 --B--> 2 --C--> 3 : a chain.
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(1), s(2), OperatorLabel::PlaceOn);
        b.push_transition(s(2), s(3), OperatorLabel::PlaceOnEmpty);
        let g = b.build();
        let q = partition_refine(&g);
        let schema = infer_operators(&q);

        let p = plan(&schema, &q, c(0), c(3)).expect("path must exist");
        assert_eq!(p.len(), 3, "shortest plan from 0 to 3 has 3 steps");
        assert_eq!(p[0], OperatorLabel::PickTop);
        assert_eq!(p[1], OperatorLabel::PlaceOn);
        assert_eq!(p[2], OperatorLabel::PlaceOnEmpty);

        // Replay succeeds and lands on goal.
        schema
            .replay_plan(&q, c(0), &p, c(3))
            .expect("replay must succeed");
    }

    #[test]
    fn unreachable_goal_returns_none() {
        // Two disconnected components.
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(2), s(3), OperatorLabel::PlaceOn);
        let g = b.build();
        let q = partition_refine(&g);
        let schema = infer_operators(&q);

        // After bisimulation: {0,2} collapse (both have one edge to a
        // sink), {1,3} collapse (both sinks). So classes are {0,2}→0,
        // {1,3}→1. The quotient has one edge: 0 --(both ops?)--> 1.
        //
        // Actually: state 0 has edge PickTop→1 (sink), state 2 has edge
        // PlaceOn→3 (sink). Sinks {1,3} are bisim-equiv. Sources {0,2}
        // are bisim-equiv iff their signatures match. sig(0) = [(PickTop,
        // class(1))], sig(2) = [(PlaceOn, class(3))]. Since PickTop≠PlaceOn,
        // the signatures differ → 0 and 2 are in DIFFERENT classes.
        //
        // So: 3 classes — {1,3}→0 (canonical: smallest member is 1),
        // {0}→1, {2}→2. Wait, canonicalization: walk states 0,1,2,3.
        // State 0 → old class A; state 1 → old class B; state 2 → old
        // class C; state 3 → old class B (sink). So {1,3} share class B.
        // Canonical: state 0's class → new 0; state 1's class → new 1;
        // state 2's class → new 2.
        //
        // Quotient edges:
        //   class(0)=0 --PickTop--> class(1)=1
        //   class(2)=2 --PlaceOn--> class(3)=1
        //
        // So from class 0, we can reach class 1 (via PickTop). From class
        // 2, we can reach class 1 (via PlaceOn). Class 1 has no outgoing
        // edges (sink). So class 0 can NOT reach class 2.
        let result = plan(&schema, &q, c(0), c(2));
        assert!(result.is_none(), "class 0 cannot reach class 2");
    }

    #[test]
    fn replay_detects_precondition_violation() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        let g = b.build();
        let q = partition_refine(&g);
        let schema = infer_operators(&q);

        // Try to replay a plan that starts with PlaceOn from class 0 —
        // class 0 doesn't admit PlaceOn.
        let bad_plan = vec![OperatorLabel::PlaceOn];
        let result = schema.replay_plan(&q, c(0), &bad_plan, c(1));
        assert!(result.is_err(), "replay must detect precondition violation");
        assert_eq!(result.unwrap_err(), 0, "failure at step 0");
    }
}
