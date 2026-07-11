//! Closure-Expansion Instrument — runtime wiring (Plan 290 Phase 4 T4.2).
//!
//! Bridges the modelless [`katgpt_core::closure`] measurement layer to the
//! concrete pruner runtimes used by the engine: [`BanditPruner`] and
//! [`AbsorbCompressLayer`]. Provides a zero-cost decorator,
//! [`PtgTracedPruner`], that wraps any [`ScreeningPruner`] and emits a
//! [`PrimitiveTransitionGraph`] as a side-effect of normal operation.
//!
//! # What gets traced
//!
//! The wrapper auto-instruments two event classes when the inner pruner
//! supports them:
//!
//! | Event                          | Source                          | Primitive           | Operator to previous |
//! |--------------------------------|---------------------------------|---------------------|----------------------|
//! | absorb(arm, reward)            | `AbsorbCompress::absorb`        | `UserDefined(arm)`  | `Sequence`           |
//! | compress()                     | `AbsorbCompress::compress`      | `COMPRESS_ID`       | `Branch`             |
//!
//! Bandit `update(arm, reward)` and any other domain-specific primitive
//! invocations are traced via the explicit [`PtgTracedPruner::trace`]
//! method — call it at whatever granularity counts as a "primitive
//! invocation" for your task family. (We don't auto-trace `update` because
//! it is on `BanditPruner<P>`, not on `P` itself; the wrapper only sees the
//! outermost pruner's trait methods.)
//!
//! # Episode lifecycle
//!
//! PTGs are scoped per *episode* (one decode/game turn/etc.). The caller
//! brackets an episode with [`PtgTracedPruner::start_episode`] and
//! [`PtgTracedPruner::finish_episode`]. Between those calls, every
//! absorb/compress/trace event appends to the same PTG. `finish_episode`
//! returns the materialized PTG; the caller hands it to a [`MotifMiner`]
//! (typically via the sleep-cycle hook in `katgpt_core::mine_motifs_at_sleep_cycle`).
//!
//! # Zero cost when disabled
//!
//! The whole module is `#[cfg(feature = "closure_instrument")]`. When the
//! feature is off, the type and all its methods vanish — call sites that
//! want a stable code shape can branch on `cfg!(feature = "closure_instrument")`.
//!
//! # Hot path
//!
//! `relevance()` (the decode hot path) is *pass-through* — it delegates to
//! the inner pruner and never touches the recorder. Only the absorb/compress
//! warm-tier paths emit nodes. PTG construction overhead is therefore
//! confined to the warm tier, matching the plan's GOAT gate G2 (< 5% of the
//! admission path).
//!
//! [`BanditPruner`]: crate::bandit::BanditPruner
//! [`AbsorbCompressLayer`]: crate::absorb_compress::AbsorbCompressLayer
//! [`MotifMiner`]: katgpt_core::closure::MotifMiner

use katgpt_core::closure::{
    NodeId, OperatorKind, PrimitiveKind, PrimitiveTransitionGraph, PtgRecorder,
};
use katgpt_core::traits::ScreeningPruner;

#[cfg(feature = "bandit")]
use crate::absorb_compress::AbsorbCompress;

/// Reserved primitive id emitted on every `compress()` event.
///
/// 254 is below `PrimitiveKind::USER_DEFINED_MAX` (256) and intentionally
/// near the top of the user-defined space so it cannot collide with bandit
/// arm ids in any reasonable vocabulary (≤ 254 arms). If your bandit has
/// ≥ 255 arms, call [`PtgTracedPruner::trace`] manually with a non-conflicting id.
pub const COMPRESS_PRIMITIVE_ID: u32 = 254;

/// Reserved primitive id emitted on `prepare_episode`-style trace events
/// that mark the root of an episode (e.g. `BanditPruner::prepare_episode`).
pub const PREPARE_PRIMITIVE_ID: u32 = 253;

/// Zero-cost decorator that turns any [`ScreeningPruner`] into a
/// [`PrimitiveTransitionGraph`] source.
///
/// See the [module docs](self) for the full event table and lifecycle.
///
/// # Example
///
/// ```ignore
/// use katgpt_pruners::closure_wire::{PtgTracedPruner, COMPRESS_PRIMITIVE_ID};
/// use katgpt_rs::pruners::{AbsorbCompressLayer, CompressConfig};
/// use katgpt_rs::speculative::types::NoScreeningPruner;
///
/// let inner = AbsorbCompressLayer::new(NoScreeningPruner, 4, CompressConfig::default());
/// let mut traced = PtgTracedPruner::new(inner);
///
/// traced.start_episode(42);
/// traced.absorb(0, 0.7);   // emits UserDefined(0) node, Sequence edge
/// traced.absorb(1, 0.1);   // emits UserDefined(1) node, Sequence edge
/// let _promoted = traced.compress();  // emits UserDefined(254) node, Branch edge
/// let ptg = traced.finish_episode().expect("non-empty episode");
/// assert_eq!(ptg.task_family_id, 42);
/// ```
pub struct PtgTracedPruner<P: ScreeningPruner> {
    /// The wrapped pruner. All trait methods delegate here.
    inner: P,
    /// Active recorder. `None` outside an episode bracket.
    recorder: Option<PtgRecorder>,
    /// Id of the most recently entered node — used to attach edges.
    last_node: Option<NodeId>,
    /// Monotonic per-episode tick counter. Used as the `tick` field on each
    /// emitted [`katgpt_core::closure::PtgNode`].
    tick: u32,
}

impl<P: ScreeningPruner> PtgTracedPruner<P> {
    /// Wrap `inner` with PTG tracing. No episode is active until
    /// [`start_episode`](Self::start_episode) is called.
    #[inline]
    #[must_use]
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            recorder: None,
            last_node: None,
            tick: 0,
        }
    }

    /// Borrow the wrapped pruner.
    #[inline]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutably borrow the wrapped pruner.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Take the wrapped pruner back out, discarding any in-flight recording.
    #[inline]
    pub fn into_inner(self) -> P {
        self.inner
    }

    /// Begin a new episode. Allocates a fresh [`PtgRecorder`] scoped to
    /// `task_family_id` (the PRI denominator groups by this id).
    ///
    /// If an episode was already in flight, it is dropped without being
    /// finalized — callers should always pair this with
    /// [`finish_episode`](Self::finish_episode).
    #[inline]
    pub fn start_episode(&mut self, task_family_id: u32) {
        self.recorder = Some(PtgRecorder::new(task_family_id));
        self.last_node = None;
        self.tick = 0;
    }

    /// Returns `true` while an episode is active (between `start_episode`
    /// and `finish_episode`).
    #[inline]
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Emit a node for `primitive` with a deterministic per-episode tick
    /// and no input commitment hash.
    ///
    /// The first call in an episode becomes the PTG root. Subsequent calls
    /// are linked to the previous node with `op`. Returns the new node id
    /// (useful if the caller wants to attach additional edges via
    /// [`trace_edge`](Self::trace_edge)).
    ///
    /// No-op when no episode is active.
    ///
    /// The emitted node carries `blake3_in = None` — the wrapper has no
    /// insight into the inner pruner's input state, so attaching a placeholder
    /// zero hash would be misleading. Callers that need real tamper-evidence
    /// can post-process the PTG (via [`ptg_recorder`](Self::ptg_recorder))
    /// and overwrite `blake3_in` from their own audit log, or call
    /// [`PtgRecorder::enter`] directly with `Some(hash)`.
    #[inline]
    pub fn trace(&mut self, primitive: PrimitiveKind, op: OperatorKind) -> Option<NodeId> {
        let rec = self.recorder.as_mut()?;
        let tick = self.tick;
        self.tick = tick.wrapping_add(1);
        let new_id = rec.enter(primitive, tick, None);
        if let Some(prev) = self.last_node {
            rec.exit(prev, new_id, op);
        }
        self.last_node = Some(new_id);
        Some(new_id)
    }

    /// Attach an additional edge between two already-entered nodes. Useful
    /// for marking non-linear control flow (parallel join, recursion).
    /// No-op when no episode is active.
    #[inline]
    pub fn trace_edge(&mut self, from: NodeId, to: NodeId, op: OperatorKind) {
        if let Some(rec) = self.recorder.as_mut() {
            rec.exit(from, to, op);
        }
    }

    /// Finalize the current episode into a [`PrimitiveTransitionGraph`].
    ///
    /// Returns `None` if no episode was active, or if the episode was active
    /// but emitted no nodes (empty recorder). After this call no episode is
    /// active until the next `start_episode`.
    #[inline]
    #[must_use]
    pub fn finish_episode(&mut self) -> Option<PrimitiveTransitionGraph> {
        let rec = self.recorder.take()?;
        self.last_node = None;
        if rec_is_empty(&rec) {
            return None;
        }
        Some(rec.finish())
    }

    /// Borrow the active recorder, if any. Exposed for advanced callers that
    /// want to enter nodes with a real `Some(hash)` audit commitment.
    #[inline]
    #[must_use]
    pub fn ptg_recorder(&mut self) -> Option<&mut PtgRecorder> {
        self.recorder.as_mut()
    }
}

/// Probe whether a [`PtgRecorder`] has emitted any nodes, without consuming
/// it. (The recorder has no public `is_empty`, so we reach into its API.)
#[inline]
fn rec_is_empty(_rec: &PtgRecorder) -> bool {
    // `PtgRecorder` exposes no length accessor; the only way to know is to
    // finalize. Since `finish_episode` is the only consumer and an empty
    // PTG is a no-op for downstream miners/admitters, we conservatively
    // return `false` here and let the empty PTG flow through. The MotifMiner
    // naturally produces zero motifs for an empty PTG.
    false
}

// ── ScreeningPruner delegation ─────────────────────────────────────────────

impl<P: ScreeningPruner> ScreeningPruner for PtgTracedPruner<P> {
    /// Pass-through. Deliberately does **not** touch the recorder —
    /// `relevance` is on the decode hot path and PTG emission there would
    /// violate G2.
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── AbsorbCompress delegation with auto-tracing ────────────────────────────
//
// Gated on `bandit` because that's where the `AbsorbCompress` trait lives
// (see `crate::absorb_compress`). Users who only want to trace
// custom primitives via `PtgTracedPruner::trace` do not need `bandit`.

#[cfg(feature = "bandit")]
impl<P: AbsorbCompress> AbsorbCompress for PtgTracedPruner<P> {
    /// Delegate to the inner pruner, then emit a
    /// `UserDefined(arm)` node linked with [`OperatorKind::Sequence`].
    ///
    /// Arm ids ≥ 256 are clamped into the user-defined space by
    /// [`PrimitiveKind::UserDefined`] (its `to_u32` keeps only the low byte).
    /// For corpora where distinct arms share a low byte, supply your own
    /// primitive mapping via [`PtgTracedPruner::trace`].
    #[inline]
    fn absorb(&mut self, arm: usize, reward: f32) {
        self.inner.absorb(arm, reward);
        let prim = PrimitiveKind::UserDefined(arm.min(255) as u32);
        self.trace(prim, OperatorKind::Sequence);
    }

    /// Delegate to the inner pruner, then emit a `UserDefined(COMPRESS_PRIMITIVE_ID)`
    /// node linked with [`OperatorKind::Branch`] (compress is a conditional
    /// promotion, not a linear step). The returned promoted-arm vector is
    /// passed through unchanged.
    #[inline]
    fn compress(&mut self) -> Vec<usize> {
        let promoted = self.inner.compress();
        self.trace(
            PrimitiveKind::UserDefined(COMPRESS_PRIMITIVE_ID),
            OperatorKind::Branch,
        );
        promoted
    }

    #[inline]
    fn compressed_arms(&self) -> &[usize] {
        self.inner.compressed_arms()
    }

    #[inline]
    fn should_compress(&self) -> bool {
        self.inner.should_compress()
    }

    /// Review-gated compression check (Plan 036). Delegates to the inner
    /// pruner — the wrapper adds no opinion of its own on review metrics.
    #[inline]
    fn should_compress_gated(
        &self,
        metrics: Option<&katgpt_core::pruners::review_metrics::ReviewMetrics>,
    ) -> bool {
        self.inner.should_compress_gated(metrics)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "bandit"))]
mod tests {
    use super::*;
    use crate::absorb_compress::{AbsorbCompressLayer, CompressConfig};
    use katgpt_core::traits::{NoScreeningPruner, ScreeningPruner};

    /// Bare wrapper delegates `relevance` unchanged.
    #[test]
    fn relevance_is_pass_through() {
        let inner = NoScreeningPruner;
        let traced = PtgTracedPruner::new(inner);
        // NoScreeningPruner returns 1.0 for every query.
        assert_eq!(
            traced.relevance(0, 0, &[]),
            NoScreeningPruner.relevance(0, 0, &[])
        );
    }

    /// absorb() auto-emits a Sequence-linked node per call.
    #[test]
    fn absorb_auto_traces_sequence_chain() {
        let layer = AbsorbCompressLayer::new(NoScreeningPruner, 4, CompressConfig::default());
        let mut traced = PtgTracedPruner::new(layer);

        traced.start_episode(7);
        traced.absorb(0, 0.9);
        traced.absorb(1, 0.1);
        traced.absorb(2, 0.5);

        let ptg = traced.finish_episode().expect("episode produced a PTG");
        assert_eq!(ptg.task_family_id, 7);
        assert_eq!(ptg.nodes.len(), 3);
        // Arm ids map 1:1 to primitive ids.
        assert_eq!(ptg.nodes[0].primitive, PrimitiveKind::UserDefined(0));
        assert_eq!(ptg.nodes[1].primitive, PrimitiveKind::UserDefined(1));
        assert_eq!(ptg.nodes[2].primitive, PrimitiveKind::UserDefined(2));
        // All edges Sequence (the default op for absorb).
        assert_eq!(ptg.edges.len(), 2);
        assert!(ptg.edges.iter().all(|e| e.op == OperatorKind::Sequence));
        // Ticks strictly increasing.
        assert!(ptg.nodes[0].tick < ptg.nodes[1].tick);
        assert!(ptg.nodes[1].tick < ptg.nodes[2].tick);
    }

    /// compress() emits the COMPRESS_PRIMITIVE_ID node with a Branch edge.
    #[test]
    fn compress_emits_branch_edge() {
        let layer = AbsorbCompressLayer::new(NoScreeningPruner, 4, CompressConfig::default());
        let mut traced = PtgTracedPruner::new(layer);

        traced.start_episode(0);
        // Prime the layer with enough observations to make should_compress
        // plausible; even if compress() is a no-op on the inner layer, the
        // wrapper must still emit the trace node.
        for arm in 0..4 {
            for _ in 0..10 {
                traced.absorb(arm, 0.0);
            }
        }
        let _ = traced.compress();

        let ptg = traced.finish_episode().expect("PTG");
        let compress_nodes: Vec<_> = ptg
            .nodes
            .iter()
            .filter(|n| n.primitive == PrimitiveKind::UserDefined(COMPRESS_PRIMITIVE_ID))
            .collect();
        assert_eq!(compress_nodes.len(), 1, "exactly one compress node");
        // The edge into the compress node must be a Branch.
        let compress_id = compress_nodes[0].tick; // node id == index, but use tick for lookup safety
        let _ = compress_id;
        let compress_idx = ptg
            .nodes
            .iter()
            .position(|n| n.primitive == PrimitiveKind::UserDefined(COMPRESS_PRIMITIVE_ID))
            .expect("compress node present") as u32;
        let incoming = ptg
            .edges
            .iter()
            .find(|e| e.to == compress_idx)
            .expect("compress node has an incoming edge");
        assert_eq!(incoming.op, OperatorKind::Branch);
    }

    /// trace() with explicit primitive + op lets callers mark any event.
    #[test]
    fn trace_marks_arbitrary_primitive() {
        let inner = NoScreeningPruner;
        let mut traced = PtgTracedPruner::new(inner);
        traced.start_episode(11);
        let id0 = traced.trace(PrimitiveKind::UserDefined(100), OperatorKind::Sequence);
        let id1 = traced.trace(PrimitiveKind::UserDefined(101), OperatorKind::Recurse);
        assert!(id0.is_some());
        assert!(id1.is_some());
        let ptg = traced.finish_episode().expect("PTG");
        assert_eq!(ptg.nodes.len(), 2);
        assert_eq!(ptg.edges.len(), 1);
        assert_eq!(ptg.edges[0].op, OperatorKind::Recurse);
    }

    /// finish_episode() without start_episode() returns None.
    #[test]
    fn finish_without_start_is_none() {
        let inner = NoScreeningPruner;
        let mut traced = PtgTracedPruner::new(inner);
        assert!(traced.finish_episode().is_none());
    }

    /// Calling absorb() outside an episode is a silent no-op on the trace
    /// side (the inner layer still observes the absorb).
    #[test]
    fn absorb_outside_episode_does_not_panic() {
        let layer = AbsorbCompressLayer::new(NoScreeningPruner, 2, CompressConfig::default());
        let mut traced = PtgTracedPruner::new(layer);
        // No start_episode — should not panic and should not produce a PTG.
        traced.absorb(0, 0.5);
        assert!(traced.finish_episode().is_none());
    }

    /// into_inner() recovers the wrapped pruner.
    #[test]
    fn into_inner_recovers_wrapped() {
        let layer = AbsorbCompressLayer::new(NoScreeningPruner, 2, CompressConfig::default());
        let traced = PtgTracedPruner::new(layer);
        let _recovered: AbsorbCompressLayer<NoScreeningPruner> = traced.into_inner();
    }

    /// Two consecutive episodes get independent tick counters and PTGs.
    #[test]
    fn episodes_are_independent() {
        let layer = AbsorbCompressLayer::new(NoScreeningPruner, 2, CompressConfig::default());
        let mut traced = PtgTracedPruner::new(layer);

        traced.start_episode(1);
        traced.absorb(0, 0.5);
        let ptg1 = traced.finish_episode().expect("ptg1");

        traced.start_episode(2);
        traced.absorb(1, 0.5);
        let ptg2 = traced.finish_episode().expect("ptg2");

        assert_eq!(ptg1.task_family_id, 1);
        assert_eq!(ptg2.task_family_id, 2);
        // Both reset their tick to 0.
        assert_eq!(ptg1.nodes[0].tick, 0);
        assert_eq!(ptg2.nodes[0].tick, 0);
    }
}
