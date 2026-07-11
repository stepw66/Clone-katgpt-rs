//! Staging table for surgical per-slot CREATE / UPDATE / DELETE.
//!
//! Plan 360 / Proposal 003 P1. A COW (copy-on-write) mutation buffer over an
//! existing [`InMemoryEngramTable`]. Buffers per-slot UPDATE and DELETE
//! operations, then produces a new immutable table on [`commit`](StagingEngramTable::commit)
//! with the source untouched.
//!
//! # Why COW
//!
//! The source table may be **live-published** under an [`EngramHotSwap`](super::EngramHotSwap)
//! Arc — readers may be mid-[`lookup_into`](super::EngramTable::lookup_into) at any
//! moment. In-place mutation would violate the G5 reader-atomicity guarantee
//! (no torn reads during freeze/thaw). COW builds a new table off to the side;
//! the caller then publishes it atomically via `swap`.
//!
//! # When to use this vs [`EngramTableBuilder`]
//!
//! - **Builder**: bulk-populate a new table from scratch (CREATE-only).
//! - **Staging**: surgical edits to an already-built table (per-slot UPDATE /
//!   DELETE) without re-deriving every other pattern. The win is **CPU** (no
//!   per-slot re-population) and **API** (caller doesn't need to re-derive).
//!
//! # Allocation budget
//!
//! - `update_slot` / `delete_slot`: amortized O(1) via `Vec` capacity growth
//!   on the pending list, plus one `Vec<f32>` copy per UPDATE (caller may
//!   mutate the slice after queueing). No per-call `HashMap`, no format strings.
//! - `commit`: one `Box<[f32]>` allocation for the COW slot array copy. Same
//!   memory cost as a full table rebuild — the win is CPU + API, not memory.

use super::EngramTable;
use super::table::InMemoryEngramTable;
use std::fmt;

/// Error returned by [`StagingEngramTable`] mutation methods.
///
/// Deliberately a small manual enum (no `thiserror` dep — matches the
/// `SurjectiveMapLoadError` pattern in `tokenizer.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingError {
    /// `slot_idx >= source.num_slots()`. Carries the offending index + the
    /// table's slot count for diagnostics.
    SlotOutOfBounds { slot_idx: usize, n_slots: usize },
    /// `pattern.len() != source.dim()`. Carries the actual + expected lengths.
    WrongPatternLen { actual: usize, expected: usize },
    /// `commit()` was called with no pending mutations. Committing an empty
    /// staging table is a wasteful full copy — caller should keep the source.
    NoPendingMutations,
}

impl fmt::Display for StagingError {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlotOutOfBounds { slot_idx, n_slots } => write!(
                f,
                "StagingEngramTable: slot_idx {slot_idx} out of bounds (n_slots={n_slots})"
            ),
            Self::WrongPatternLen { actual, expected } => write!(
                f,
                "StagingEngramTable: pattern.len()={actual} must equal dim()={expected}"
            ),
            Self::NoPendingMutations => {
                write!(f, "StagingEngramTable::commit: no pending mutations")
            }
        }
    }
}

impl std::error::Error for StagingError {}

/// A single buffered mutation. `None` = DELETE (zero out); `Some` = UPDATE
/// (replace with this pattern). Stored in insertion order; later same-slot
/// wins at commit time.
#[derive(Debug, Clone)]
struct PendingMutation {
    slot_idx: usize,
    new_pattern: Option<Vec<f32>>,
}

/// COW mutation buffer over an [`InMemoryEngramTable`].
///
/// Borrows the source immutably, buffers per-slot UPDATE / DELETE mutations,
/// and produces a new immutable table on `commit()`. The source is untouched
/// (copy-on-write semantics).
///
/// # Example
///
/// ```ignore
/// use katgpt_core::engram::{EngramTableBuilder, EngramHash, StagingEngramTable};
///
/// let mut b = EngramTableBuilder::new(1024, 32);
/// // ... populate ...
/// let table = b.build();
///
/// // Surgical edit: replace slot 47's pattern, zero out slot 92, leave the
/// // other 1022 slots untouched. Source table is NOT mutated.
/// let new_table = StagingEngramTable::from_table(&table)
///     .update_slot(47, &[1.0f32; 32])
///     .expect("47 in bounds, pattern len 32 == dim 32")
///     .delete_slot(92)
///     .expect("92 in bounds")
///     .commit()
///     .expect("at least one pending mutation");
/// ```
#[derive(Debug)]
pub struct StagingEngramTable<'a> {
    source: &'a InMemoryEngramTable,
    /// Insertion-ordered mutations. Later same-slot wins at commit (last-write
    /// semantics, mirroring `EngramTableBuilder::add_pattern`).
    pending: Vec<PendingMutation>,
}

impl<'a> StagingEngramTable<'a> {
    /// Borrow `source` immutably with an empty pending list.
    #[inline]
    pub fn from_table(source: &'a InMemoryEngramTable) -> Self {
        Self {
            source,
            pending: Vec::new(),
        }
    }

    /// Borrow `source`, pre-allocating `capacity` slots in the pending list.
    /// Use when the caller knows the approximate edit count to avoid Vec
    /// reallocation as mutations are queued.
    #[inline]
    pub fn with_capacity(source: &'a InMemoryEngramTable, capacity: usize) -> Self {
        Self {
            source,
            pending: Vec::with_capacity(capacity),
        }
    }

    /// UPDATE-slot: queue a pattern replacement at `slot_idx`.
    ///
    /// `pattern.len()` MUST equal `source.dim()` — checked at queue time
    /// (returns `Err` early, not at commit time). The pattern is **copied**
    /// into the pending list so the caller may mutate or drop the slice after
    /// this call returns.
    ///
    /// Queueing two UPDATEs on the same slot is allowed — the later one wins
    /// at commit time (last-write semantics, same as `add_pattern`).
    #[inline]
    pub fn update_slot(
        &mut self,
        slot_idx: usize,
        pattern: &[f32],
    ) -> Result<&mut Self, StagingError> {
        let n_slots = self.source.num_slots();
        if slot_idx >= n_slots {
            return Err(StagingError::SlotOutOfBounds { slot_idx, n_slots });
        }
        let expected = self.source.dim();
        if pattern.len() != expected {
            return Err(StagingError::WrongPatternLen {
                actual: pattern.len(),
                expected,
            });
        }
        self.pending.push(PendingMutation {
            slot_idx,
            new_pattern: Some(pattern.to_vec()),
        });
        Ok(self)
    }

    /// DELETE-slot: queue a zero-out at `slot_idx`.
    ///
    /// Idempotent — deleting an already-deleted slot is a no-op at commit
    /// (the second DELETE just re-zeroes an already-zero slot). Queueing an
    /// UPDATE after a DELETE on the same slot makes the UPDATE win (last-write).
    #[inline]
    pub fn delete_slot(&mut self, slot_idx: usize) -> Result<&mut Self, StagingError> {
        let n_slots = self.source.num_slots();
        if slot_idx >= n_slots {
            return Err(StagingError::SlotOutOfBounds { slot_idx, n_slots });
        }
        self.pending.push(PendingMutation {
            slot_idx,
            new_pattern: None,
        });
        Ok(self)
    }

    /// Apply all pending mutations, producing a new immutable table.
    ///
    /// The source table is **untouched** (COW). The returned table has a
    /// fresh lazy BLAKE3 commitment (computed on first `commitment()` call).
    ///
    /// After `commit`, the staging table's pending list is drained — a second
    /// `commit` call returns [`StagingError::NoPendingMutations`]. The source
    /// borrow remains valid.
    ///
    /// Returns [`StagingError::NoPendingMutations`] if no mutations are
    /// queued — committing an empty staging table is a wasteful full copy;
    /// the caller should keep using the source as-is in that case.
    ///
    /// Takes `&mut self` (not `self`) so it composes with the `&mut Self`
    /// return of [`update_slot`](Self::update_slot) / [`delete_slot`](Self::delete_slot)
    /// for fluent chaining: `from_table(t).update_slot(i, p)?.commit()?`.
    pub fn commit(&mut self) -> Result<InMemoryEngramTable, StagingError> {
        if self.pending.is_empty() {
            return Err(StagingError::NoPendingMutations);
        }
        // Drain pending into a local — we're borrowing `&mut self`, so we
        // can't move out of `self.pending` directly. `take` swaps in an empty
        // Vec and hands us ownership of the original. A second commit call
        // would see `self.pending.is_empty()` and return Err.
        let pending = std::mem::take(&mut self.pending);

        let d = self.source.dim();
        let n_slots = self.source.num_slots();

        // COW step 1: clone the source slot array into a fresh allocation.
        // `to_vec` → `into_boxed_slice` mirrors what the builder does. For a
        // 1M×64 table this is ~256MB — same memory cost as a rebuild, the win
        // is CPU (no per-slot re-derivation) and API (caller doesn't need to
        // re-derive patterns).
        let mut new_slots = self.source.slots().to_vec().into_boxed_slice();

        // COW step 2: apply pending mutations in insertion order; later same-slot
        // wins. Linear scan O(pending²) worst case — fine for the typical GM-edit
        // workload (O(10s) of mutations). For pathological workloads (>1000
        // mutations) the caller should use `EngramTableBuilder` instead.
        for m in &pending {
            let start = m.slot_idx * d;
            let end = start + d;
            match &m.new_pattern {
                Some(pattern) => {
                    new_slots[start..end].copy_from_slice(pattern);
                }
                None => {
                    new_slots[start..end].fill(0.0);
                }
            }
        }

        // COW step 3: clone the heads (HashHead is Copy — cheap clone of the
        // boxed array) and construct the new table via the crate-visible
        // `from_parts` constructor.
        let new_heads = Box::new(*self.source.heads());
        Ok(InMemoryEngramTable::from_parts(
            new_slots, new_heads, n_slots, d,
        ))
    }

    /// Number of pending mutations queued (not yet committed).
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Discard all pending mutations, keeping the source borrow. Useful for
    /// "cancel edit" UI flows.
    #[inline]
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

// The `EngramTable` trait import above brings `num_slots` / `dim` into scope.
// The source table's `heads()` accessor is called via `Box::new(*source.heads())`
// without naming `HashHead` — the type flows through inference.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{EngramHash, EngramTableBuilder};

    /// Helper: build a table with all slots populated with distinct non-zero patterns.
    /// Slot `i` gets pattern `[(i+1) as f32; d]` for all `i in 0..n` — the `+1`
    /// ensures slot 0 is NOT all-zeros (which would be indistinguishable from
    /// an empty/unpopulated slot, breaking commitment-difference checks).
    fn make_distinct_table(n_slots: usize, d: usize) -> InMemoryEngramTable {
        let mut b = EngramTableBuilder::new(n_slots, d);
        for i in 0..n_slots as u64 {
            let pat: Vec<f32> = vec![(i + 1) as f32; d];
            b.add_pattern(EngramHash(i % n_slots.max(1) as u64), &pat);
        }
        b.build()
    }

    /// Helper: read slot `i` from a table into a fresh Vec.
    fn read_slot(table: &InMemoryEngramTable, slot_idx: usize) -> Vec<f32> {
        let d = table.dim();
        let src = table.slots();
        src[slot_idx * d..(slot_idx + 1) * d].to_vec()
    }

    #[test]
    fn update_slot_writes_to_new_table_only() {
        // T1.7: stage an update, commit, verify the new table has the new
        // pattern at the slot AND the source table is unchanged (via slots()).
        let d = 4;
        let source = make_distinct_table(16, d);
        let src_slot_5_before = read_slot(&source, 5);
        let src_commitment_before = source.commitment();

        let new_pattern = vec![42.0f32; d];
        let new_table = StagingEngramTable::from_table(&source)
            .update_slot(5, &new_pattern)
            .unwrap()
            .commit()
            .unwrap();

        // (a) new table has the new pattern at slot 5
        assert_eq!(read_slot(&new_table, 5), new_pattern);
        // (b) source table's slot 5 is unchanged
        assert_eq!(read_slot(&source, 5), src_slot_5_before);
        // (c) source table's commitment is unchanged
        assert_eq!(source.commitment(), src_commitment_before);
    }

    #[test]
    fn delete_slot_zeros_in_new_table() {
        // T1.7: stage a delete, commit, verify the slot is zero in the new
        // table AND non-zero in the source.
        let d = 4;
        let source = make_distinct_table(16, d);

        // Sanity: slot 7 should be non-zero in the source.
        let src_slot_7 = read_slot(&source, 7);
        assert!(
            src_slot_7.iter().any(|&v| v != 0.0),
            "slot 7 non-zero in source"
        );

        let new_table = StagingEngramTable::from_table(&source)
            .delete_slot(7)
            .unwrap()
            .commit()
            .unwrap();

        // New table: slot 7 is all zeros.
        let new_slot_7 = read_slot(&new_table, 7);
        assert!(
            new_slot_7.iter().all(|&v| v == 0.0),
            "slot 7 zero in new table"
        );
        // Source still non-zero at slot 7.
        assert_eq!(read_slot(&source, 7), src_slot_7);
    }

    #[test]
    fn commit_empty_returns_err() {
        // T1.7: committing with no pending mutations returns NoPendingMutations.
        let source = make_distinct_table(8, 4);
        let mut staging = StagingEngramTable::from_table(&source);
        let result = staging.commit();
        assert_eq!(result.unwrap_err(), StagingError::NoPendingMutations);
    }

    #[test]
    fn out_of_bounds_slot_returns_err() {
        // T1.7: update_slot(n_slots, ...) returns SlotOutOfBounds.
        let n_slots = 16;
        let source = make_distinct_table(n_slots, 4);
        let mut staging = StagingEngramTable::from_table(&source);
        let err = staging.update_slot(n_slots, &[0.0f32; 4]).unwrap_err();
        assert_eq!(
            err,
            StagingError::SlotOutOfBounds {
                slot_idx: n_slots,
                n_slots
            }
        );
        // delete_slot too
        let err = staging.delete_slot(n_slots).unwrap_err();
        assert_eq!(
            err,
            StagingError::SlotOutOfBounds {
                slot_idx: n_slots,
                n_slots
            }
        );
    }

    #[test]
    fn wrong_pattern_len_returns_err() {
        // T1.7: update_slot(i, &vec_of_d+1) returns WrongPatternLen.
        let d = 4;
        let source = make_distinct_table(16, d);
        let mut staging = StagingEngramTable::from_table(&source);
        let too_long = vec![0.0f32; d + 1];
        let err = staging.update_slot(0, &too_long).unwrap_err();
        assert_eq!(
            err,
            StagingError::WrongPatternLen {
                actual: d + 1,
                expected: d
            }
        );
    }

    #[test]
    fn later_same_slot_wins() {
        // T1.7: stage update_slot(5, A) then update_slot(5, B), commit, verify
        // slot 5 == B (last-write semantics).
        let d = 4;
        let source = make_distinct_table(16, d);
        let pattern_a = vec![1.0f32; d];
        let pattern_b = vec![2.0f32; d];

        let new_table = StagingEngramTable::from_table(&source)
            .update_slot(5, &pattern_a)
            .unwrap()
            .update_slot(5, &pattern_b)
            .unwrap()
            .commit()
            .unwrap();

        assert_eq!(read_slot(&new_table, 5), pattern_b);
    }

    #[test]
    fn mixed_update_delete_update() {
        // T1.7: stage U/D/U on different slots, commit, verify each.
        let d = 2;
        let source = make_distinct_table(32, d);
        let p3 = vec![3.0f32; d];
        let p11 = vec![11.0f32; d];

        let new_table = StagingEngramTable::from_table(&source)
            .update_slot(3, &p3)
            .unwrap()
            .delete_slot(7)
            .unwrap()
            .update_slot(11, &p11)
            .unwrap()
            .commit()
            .unwrap();

        assert_eq!(read_slot(&new_table, 3), p3);
        assert!(read_slot(&new_table, 7).iter().all(|&v| v == 0.0));
        assert_eq!(read_slot(&new_table, 11), p11);
    }

    #[test]
    fn clear_discards_pending() {
        // T1.7: stage mutations, clear(), pending_count() == 0.
        let source = make_distinct_table(16, 4);
        let mut staging = StagingEngramTable::from_table(&source);
        staging.update_slot(0, &[1.0f32; 4]).unwrap();
        staging.update_slot(1, &[2.0f32; 4]).unwrap();
        assert_eq!(staging.pending_count(), 2);

        staging.clear();
        assert_eq!(staging.pending_count(), 0);

        // After clear, commit returns NoPendingMutations.
        let result = staging.commit();
        assert_eq!(result.unwrap_err(), StagingError::NoPendingMutations);
    }

    #[test]
    fn commitment_recomputed() {
        // T1.7: source table has cached commitment C1; after staging + commit,
        // new table has commitment C2 != C1 (BLAKE3 differs because contents differ).
        let source = make_distinct_table(16, 4);
        let c1 = source.commitment();

        let new_table = StagingEngramTable::from_table(&source)
            .delete_slot(0)
            .unwrap()
            .commit()
            .unwrap();

        let c2 = new_table.commitment();
        assert_ne!(c1, c2, "commitment must change after a mutation");
    }

    #[test]
    fn idempotent_delete() {
        // T1.7: stage delete_slot(7) twice, commit succeeds, slot 7 is zero.
        let source = make_distinct_table(16, 4);
        let new_table = StagingEngramTable::from_table(&source)
            .delete_slot(7)
            .unwrap()
            .delete_slot(7)
            .unwrap()
            .commit()
            .unwrap();

        let slot_7 = read_slot(&new_table, 7);
        assert!(
            slot_7.iter().all(|&v| v == 0.0),
            "double-delete zeros the slot"
        );
    }

    #[test]
    fn with_capacity_pre_allocates() {
        // Sanity: with_capacity doesn't allocate on first push (capacity hint).
        let source = make_distinct_table(16, 4);
        let mut staging = StagingEngramTable::with_capacity(&source, 3);
        let cap_before = staging.pending.capacity();
        assert!(cap_before >= 3, "with_capacity reserves at least 3");

        staging.update_slot(0, &[1.0f32; 4]).unwrap();
        staging.update_slot(1, &[2.0f32; 4]).unwrap();
        staging.update_slot(2, &[3.0f32; 4]).unwrap();
        assert_eq!(
            staging.pending.capacity(),
            cap_before,
            "no realloc within hint"
        );
    }

    #[test]
    fn unaffected_slots_match_source_bit_for_bit() {
        // T1.7 sub-check: slots NOT in the pending list must match the source
        // bit-for-bit (no mutation leakage). This is the G1 correctness core.
        let d = 4;
        let n_slots = 32;
        let source = make_distinct_table(n_slots, d);

        // Mutate slots 3, 7, 11; leave the other 29 untouched.
        let p3 = vec![99.0f32; d];
        let p11 = vec![42.0f32; d];
        let new_table = StagingEngramTable::from_table(&source)
            .update_slot(3, &p3)
            .unwrap()
            .delete_slot(7)
            .unwrap()
            .update_slot(11, &p11)
            .unwrap()
            .commit()
            .unwrap();

        // Check every non-mutated slot matches the source.
        for i in 0..n_slots {
            if i == 3 || i == 7 || i == 11 {
                continue;
            }
            assert_eq!(
                read_slot(&new_table, i),
                read_slot(&source, i),
                "untouched slot {i} must match source bit-for-bit"
            );
        }
    }

    #[test]
    fn builder_chaining_idiom() {
        // The `-> Result<&mut Self, _>` return enables fluent chaining.
        // Verify the chained form compiles + runs end-to-end.
        let source = make_distinct_table(8, 2);
        let new_table = StagingEngramTable::from_table(&source)
            .update_slot(0, &[1.0, 2.0])
            .unwrap()
            .delete_slot(1)
            .unwrap()
            .commit()
            .unwrap();

        assert_eq!(read_slot(&new_table, 0), vec![1.0, 2.0]);
        assert!(read_slot(&new_table, 1).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn pending_count_tracks_queue() {
        let source = make_distinct_table(16, 4);
        let mut staging = StagingEngramTable::from_table(&source);
        assert_eq!(staging.pending_count(), 0);
        staging.update_slot(0, &[1.0f32; 4]).unwrap();
        assert_eq!(staging.pending_count(), 1);
        staging.delete_slot(1).unwrap();
        assert_eq!(staging.pending_count(), 2);
        staging.update_slot(0, &[2.0f32; 4]).unwrap(); // same slot, new entry
        assert_eq!(
            staging.pending_count(),
            3,
            "duplicate slot adds a new entry"
        );
    }

    #[test]
    fn delete_then_update_same_slot_update_wins() {
        // Edge case: delete slot then update it. Last-write means UPDATE wins.
        let source = make_distinct_table(16, 4);
        let new_pattern = vec![7.0f32; 4];
        let new_table = StagingEngramTable::from_table(&source)
            .delete_slot(5)
            .unwrap()
            .update_slot(5, &new_pattern)
            .unwrap()
            .commit()
            .unwrap();
        assert_eq!(read_slot(&new_table, 5), new_pattern);
    }

    #[test]
    fn zero_dim_table_hands_back_unchanged_committed_arithmetic() {
        // Edge case: n_slots=0. update_slot(0,...) returns SlotOutOfBounds
        // (0 >= 0). The staging table should never reach commit with pending
        // mutations on a 0-slot table — but if it somehow does, commit must
        // not panic. Verify the bounds check fires first.
        let source = EngramTableBuilder::new(0, 4).build();
        let mut staging = StagingEngramTable::from_table(&source);
        let err = staging.update_slot(0, &[0.0f32; 4]).unwrap_err();
        assert_eq!(
            err,
            StagingError::SlotOutOfBounds {
                slot_idx: 0,
                n_slots: 0
            }
        );
    }

    #[test]
    fn staging_error_display_is_human_readable() {
        // Sanity: Display impl produces a useful message (not just `Debug`).
        let e1 = StagingError::SlotOutOfBounds {
            slot_idx: 99,
            n_slots: 16,
        };
        assert!(format!("{e1}").contains("99"));
        assert!(format!("{e1}").contains("16"));

        let e2 = StagingError::WrongPatternLen {
            actual: 5,
            expected: 4,
        };
        assert!(format!("{e2}").contains("5"));
        assert!(format!("{e2}").contains("4"));

        let e3 = StagingError::NoPendingMutations;
        assert!(format!("{e3}").contains("no pending"));
    }
}
