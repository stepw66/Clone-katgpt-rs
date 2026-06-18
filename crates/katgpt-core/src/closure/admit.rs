//! `MotifAdmitter` — promotion gate from recurring [`Motif`] → composite primitive.
//!
//! Implements the paper's §5.2 "wrapped motifs become higher-order primitives"
//! admission test, fusing with Plan 215's RegimeTransitionGate. The gate is
//! intentionally self-contained: it consumes a [`Motif`] + a few scalars and
//! emits a [`GateResult`] that Phase 4 wiring can route into the actual
//! `RegimeTransitionGate` (avoids a circular dependency on Plan 215 types).
//!
//! ## Admission rule
//!
//! A motif is admitted iff **all three** hold:
//! 1. `PRI ≥ 0.1` — appears across at least 10% of task families.
//! 2. `occurrence_count ≥ 3` — seen at least 3 times.
//! 3. The corpus description length `dl_old_bits` exceeds the admission cost
//!    (`admission_cost = admission_cost_per_node_bits * node_count`) — i.e.
//!    the corpus is large enough to amortize the cost of registering a new
//!    vocabulary symbol.
//!
//! `admission_cost = admission_cost_per_node_bits * node_count` (MDL-style
//! overhead — bigger motifs need to pay off more to be worth wrapping).
//!
//! ## MDL approximation
//!
//! Plan 290's literal MDL rule (`DL_new < DL_old − admission_cost`) is
//! algebraically circular since `DL_new := DL_old − admission_cost` is the
//! only model we have without Plan 215 wiring. We interpret the intent as
//! "the corpus is large enough to amortize the admission cost":
//!
//! ```text
//! admit iff dl_old_bits > admission_cost
//! ```
//!
//! The real `DL_new` from Plan 215's MDL formula replaces this comparison in
//! Phase 4 — the [`GateResult`] shape is stable across the swap.

use super::{Motif, PrimitiveKind};

/// Outcome of [`MotifAdmitter::evaluate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateResult {
    /// The motif is admitted as a new composite primitive.
    Admitted {
        /// New composite primitive to register.
        new_primitive: PrimitiveKind,
        /// Bits saved by wrapping (`admission_cost`).
        dl_gain_bits: f32,
    },
    /// The motif was rejected.
    Rejected {
        /// Why.
        reason: RejectionReason,
    },
}

/// Reason a motif was not admitted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RejectionReason {
    /// PRI below the `0.1` threshold. Carries the offending value.
    LowPri(f32),
    /// `occurrence_count` below the `3` minimum. Carries the offending value.
    InsufficientOccurrences(u32),
    /// MDL cost — wrapping would not pay off.
    MdlCost,
}

/// Promotion gate. Stateless — holds only configuration.
#[derive(Clone, Copy, Debug)]
pub struct MotifAdmitter {
    /// MDL-style admission cost per node, in bits. Default `8.0` — i.e. a
    /// 4-node motif costs 32 bits to admit and must save more than that.
    pub admission_cost_per_node_bits: f32,
}

/// Default per-node admission cost (bits). Mirrors MDL conventions: the cost
/// of "introducing a new symbol in the vocabulary" ≈ 1 byte.
pub const DEFAULT_ADMISSION_COST_PER_NODE_BITS: f32 = 8.0;

/// Minimum PRI for admission.
pub const PRI_ADMISSION_THRESHOLD: f32 = 0.1;

/// Minimum occurrence count for admission.
pub const OCCURRENCE_ADMISSION_THRESHOLD: u32 = 3;

impl Default for MotifAdmitter {
    #[inline]
    fn default() -> Self {
        Self {
            admission_cost_per_node_bits: DEFAULT_ADMISSION_COST_PER_NODE_BITS,
        }
    }
}

impl MotifAdmitter {
    /// Construct with default config (8 bits/node).
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a custom per-node admission cost.
    #[inline]
    #[must_use]
    pub fn with_cost_per_node(admission_cost_per_node_bits: f32) -> Self {
        Self {
            admission_cost_per_node_bits,
        }
    }

    /// Evaluate the admission gate.
    ///
    /// `total_task_families` is the denominator for PRI (e.g. corpus-wide
    /// count of distinct task families observed). `dl_old_bits` is the
    /// pre-wrap description length of the corpus; we admit only when it
    /// exceeds the per-node admission cost (see module doc for the MDL
    /// approximation).
    #[inline]
    #[must_use]
    pub fn evaluate(
        &self,
        motif: &Motif,
        total_task_families: u32,
        dl_old_bits: f32,
    ) -> GateResult {
        // 1. PRI gate.
        let pri = motif.primitive_reuse_index(total_task_families);
        if pri < PRI_ADMISSION_THRESHOLD {
            return GateResult::Rejected {
                reason: RejectionReason::LowPri(pri),
            };
        }

        // 2. Occurrence-count gate.
        if motif.occurrence_count < OCCURRENCE_ADMISSION_THRESHOLD {
            return GateResult::Rejected {
                reason: RejectionReason::InsufficientOccurrences(motif.occurrence_count),
            };
        }

        // 3. MDL gate — corpus must be large enough to amortize the cost.
        let admission_cost = self.admission_cost_per_node_bits * motif.node_count as f32;
        if dl_old_bits <= admission_cost {
            return GateResult::Rejected {
                reason: RejectionReason::MdlCost,
            };
        }

        // Admitted: derive the composite primitive id from the BE prefix of
        // the motif's BLAKE3 hash (4 bytes → u32 → offset into Composite
        // space).
        let mut prefix_bytes = [0u8; 4];
        let n = motif.subgraph_hash.len().min(4);
        prefix_bytes[..n].copy_from_slice(&motif.subgraph_hash[..n]);
        let prefix = u32::from_be_bytes(prefix_bytes);
        GateResult::Admitted {
            new_primitive: PrimitiveKind::Composite(prefix),
            dl_gain_bits: admission_cost,
        }
    }
}

// ── Tests (T2.6 + T2.7) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{
        FixedU32Set, MotifMiner, OperatorKind, PrimitiveKind, PtgRecorder,
    };

    fn make_search_verify_branch_ptg(task_family_id: u32) -> crate::closure::PrimitiveTransitionGraph {
        let mut rec = PtgRecorder::new(task_family_id);
        let a = rec.enter(PrimitiveKind::UserDefined(0), 0, [0u8; 32]);
        let b = rec.enter(PrimitiveKind::UserDefined(1), 1, [1u8; 32]);
        let c = rec.enter(PrimitiveKind::UserDefined(2), 2, [2u8; 32]);
        rec.exit(a, b, OperatorKind::Sequence);
        rec.exit(b, c, OperatorKind::Branch);
        rec.finish()
    }

    /// T2.6 — a motif across 3 task families × 100 traces is admitted.
    #[test]
    fn motif_is_admitted_when_pri_and_count_meet_thresholds() {
        let mut miner = MotifMiner::new();
        for i in 0..100u32 {
            miner.observe(make_search_verify_branch_ptg(i % 3));
        }
        let motifs = miner.mine_batch();
        let found = motifs
            .iter()
            .find(|m| m.node_count == 3)
            .expect("3-node motif present");

        let gate = MotifAdmitter::new();
        let result = gate.evaluate(found, 3, 10_000.0);
        match result {
            GateResult::Admitted {
                new_primitive,
                dl_gain_bits,
            } => {
                assert!(new_primitive.is_composite());
                assert!(dl_gain_bits > 0.0, "should report a gain");
            }
            other => panic!("expected Admitted, got {other:?}"),
        }
    }

    /// T2.7 — high occurrence but only 1 task family ⇒ low PRI ⇒ rejected.
    #[test]
    fn motif_in_one_family_is_rejected_for_low_pri() {
        let mut miner = MotifMiner::new();
        for _ in 0..100u32 {
            miner.observe(make_search_verify_branch_ptg(0));
        }
        let motifs = miner.mine_batch();
        let found = motifs
            .iter()
            .find(|m| m.node_count == 3)
            .expect("3-node motif present");

        // 20 task families in the corpus, but this motif only in 1 ⇒ PRI = 0.05.
        let gate = MotifAdmitter::new();
        let result = gate.evaluate(found, 20, 10_000.0);
        match result {
            GateResult::Rejected {
                reason: RejectionReason::LowPri(pri),
            } => {
                assert!(pri < PRI_ADMISSION_THRESHOLD, "pri={pri}");
            }
            other => panic!("expected Rejected::LowPri, got {other:?}"),
        }
    }

    /// MDL cost branch is exercised when wrapping saves nothing.
    #[test]
    fn motif_rejected_on_mdl_when_no_gain() {
        // Construct a motif with PRI and occ above thresholds.
        let mut families = FixedU32Set::<16>::new();
        for f in 0..5 {
            families.insert(f);
        }
        let motif = Motif {
            subgraph_hash: [1u8; 32],
            node_count: 4,
            edge_count: 3,
            occurrence_count: 10,
            task_family_ids: families,
        };
        // PRI = 5/10 = 0.5 ≥ 0.1, occ = 10 ≥ 3.
        // admission_cost = 8 * 4 = 32 bits; dl_old = 10 ≤ 32 ⇒ MdlCost.
        let gate = MotifAdmitter::new();
        let result = gate.evaluate(&motif, 10, 10.0);
        assert!(
            matches!(
                result,
                GateResult::Rejected {
                    reason: RejectionReason::MdlCost
                }
            ),
            "expected MdlCost, got {result:?}"
        );
    }

    /// Occurrence-count gate.
    #[test]
    fn motif_rejected_on_insufficient_occurrences() {
        let mut families = FixedU32Set::<16>::new();
        for f in 0..5 {
            families.insert(f);
        }
        let motif = Motif {
            subgraph_hash: [1u8; 32],
            node_count: 2,
            edge_count: 1,
            occurrence_count: 1, // below threshold of 3
            task_family_ids: families,
        };
        let gate = MotifAdmitter::new();
        let result = gate.evaluate(&motif, 5, 1_000.0);
        match result {
            GateResult::Rejected {
                reason: RejectionReason::InsufficientOccurrences(n),
            } => {
                assert_eq!(n, 1);
            }
            other => panic!("expected InsufficientOccurrences, got {other:?}"),
        }
    }

    /// Admitted motif yields a composite primitive whose prefix is taken from
    /// the motif's BLAKE3 hash (deterministic).
    #[test]
    fn admitted_primitive_id_is_deterministic() {
        let mut families = FixedU32Set::<16>::new();
        for f in 0..5 {
            families.insert(f);
        }
        let motif = Motif {
            subgraph_hash: [
                0xAB, 0xCD, 0xEF, 0x12, 0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            node_count: 2,
            edge_count: 1,
            occurrence_count: 10,
            task_family_ids: families,
        };
        let gate = MotifAdmitter::new();
        // PRI = 5/10 = 0.5 ≥ 0.1, occ = 10, dl_old = 10000, cost = 16 ⇒ admit.
        let result = gate.evaluate(&motif, 10, 10_000.0);
        match result {
            GateResult::Admitted { new_primitive, .. } => {
                let expected = u32::from_be_bytes([0xAB, 0xCD, 0xEF, 0x12]);
                assert_eq!(new_primitive, PrimitiveKind::Composite(expected));
            }
            other => panic!("expected Admitted, got {other:?}"),
        }
    }
}
