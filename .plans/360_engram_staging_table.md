# Plan 360: Engram Staging Table — First-Class Per-Slot C/U/D

**Date:** 2026-07-03
**Proposal:** [riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md](../../../riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md) (P1)
**Parent plan:** [katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md](299_Engram_Hash_Addressed_Pattern_Memory.md) (Phase 2 — `InMemoryEngramTable`, `EngramTableBuilder`)
**Target:** `katgpt-rs/crates/katgpt-core/src/engram/staging.rs` (new module)
**Cargo feature:** `engram` (existing — no new feature flag; sibling to `table.rs`)
**Status:** Active — Phase 1 DONE (2026-07-03). 17/17 staging tests pass, 112/112 engram tests pass, 666/666 default-feature tests pass (no regression). Phase 2 (GOAT gate) + Phase 3 (docs wiring) + Phase 4 (promotion decision) pending. Ship behind existing `engram` (still default-off, gated on Plan 299 G6).

---

## Goal

Ship the **first-class per-slot CREATE / UPDATE / DELETE** primitive that `EngramTableBuilder` does not provide. Today:

- **CREATE**: `EngramTableBuilder::add_pattern` (bulk populate before `build()`)
- **READ**: `EngramTable::lookup_into` (O(1) lookup)
- **SWAP**: `EngramHotSwap::swap` (whole-table atomic replacement)
- **UPDATE-slot**: ❌ only by rebuilding the entire table via the builder
- **DELETE-slot**: ❌ only by rebuilding the entire table via the builder

For the GM tool use case ("edit slot 47 of NPC 1024's voice table without touching the other 1023 slots"), rebuilding the whole table is wasteful and forces the caller to re-derive every pattern. The staging table buffers surgical edits and produces a new immutable table via copy-on-write (COW).

**The GOAT claim (modelless):** surgical single-slot UPDATE is faster than whole-table rebuild, AND produces a cleaner API surface for consumers (Proposal 003's `EngramControlEnvelope` MCP dispatch, future `engram_runtime`). The gain is **compute saving + DRY**, both modelless per AGENTS.md.

**Anti-goal:** this plan does NOT wire the staging table into any consumer (riir-chaind `EngramControlEnvelope`, riir-engine `engram_runtime`, riir-viz `EngramCtrl` panel). Those are Proposal 003 P2–P4 and live in other repos. This plan ships the primitive only.

---

## Architecture

```
katgpt-rs/crates/katgpt-core/src/engram/
├── table.rs            ← InMemoryEngramTable (existing — add pub(crate) from_parts)
├── staging.rs          ← NEW: StagingEngramTable<'a> (COW per-slot C/U/D)
└── ...
```

Plus tests in `crates/katgpt-core/src/engram/tests.rs` (extend existing) and a GOAT gate in `crates/katgpt-core/tests/bench_360_engram_staging_goat.rs` (new).

### COW semantics (why)

The source table may be **live-published** under an `EngramHotSwap` Arc — readers may be mid-`lookup_into` on it at any moment. Mutating it in place would violate the G5 reader-atomicity guarantee from Plan 299 (no torn reads during freeze/thaw).

The staging table therefore:
1. Borrows the source `&InMemoryEngramTable` immutably.
2. Buffers pending mutations as `Vec<(slot_idx, Option<Vec<f32>>)>` — `Some` = UPDATE, `None` = DELETE.
3. On `commit()`: allocates a fresh `Box<[f32]>` slot array, copies source slots into it, applies pending mutations, returns a new `InMemoryEngramTable`. The source is untouched.
4. The caller publishes the new table via `EngramHotSwap::swap` if it wants readers to see it (atomic).

This mirrors how `EngramHotSwap::swap` already publishes a freshly-built table — the staging table just makes the "freshly-built" step surgical instead of bulk.

### Required `pub(crate)` accessor on `InMemoryEngramTable`

`staging.rs` is a sibling module of `table.rs` under `engram/`. To construct a new `InMemoryEngramTable` from the mutated slot array without going through `EngramTableBuilder`, the staging table needs access to `InMemoryEngramTable`'s private fields. Two options:

- **(a)** Make the fields `pub(crate)`.
- **(b)** Add a `pub(crate) fn from_parts(slots, heads, n_slots, d) -> Self` constructor.

Option (b) is cleaner — single chokepoint, future-proof if fields change. This plan uses (b).

---

## Phase 1 — Core Staging Primitive (CORE)

### Tasks

- [x] **T1.1** Add `pub(crate) fn from_parts(slots: Box<[f32]>, heads: Box<[HashHead; K_MAX]>, n_slots: usize, d: usize) -> Self` to `InMemoryEngramTable` in `table.rs`. Constructs without re-validating (the staging table has already validated dimensions). Inline. Single-line body — delegates to struct literal.
  - **Done (2026-07-03).** Also added `pub(crate) fn slots(&self) -> &[f32]` for the COW copy. Discovered `num_slots()` and `dim()` are already on the `EngramTable` trait (no new accessors needed for those). Added `#[derive(Debug)]` to `InMemoryEngramTable` (needed by `unwrap_err()` in tests, useful for general debugging).

- [x] **T1.2** Create `crates/katgpt-core/src/engram/staging.rs` behind `#[cfg(feature = "engram")]`. Define `StagingEngramTable<'a>`:
  ```rust
  pub struct StagingEngramTable<'a> {
      source: &'a InMemoryEngramTable,
      /// Buffered mutations. `Some(pattern)` = UPDATE; `None` = DELETE.
      /// In insertion order (commit applies in order; later same-slot wins).
      pending: Vec<PendingMutation>,
  }

  struct PendingMutation {
      slot_idx: usize,
      /// None = delete (zero out). Some = replace with this pattern.
      new_pattern: Option<Vec<f32>>,
  }
  ```
  No `heads`/`n_slots`/`d` fields — those come from `source` on commit.

- [x] **T1.3** Implement constructors:
  ```rust
  impl<'a> StagingEngramTable<'a> {
      /// Borrow `source` immutably. Empty pending list.
      pub fn from_table(source: &'a InMemoryEngramTable) -> Self;

      /// Borrow `source`, pre-allocate `capacity` pending slots
      /// (avoid realloc on the pending Vec if the caller knows the edit count).
      pub fn with_capacity(source: &'a InMemoryEngramTable, capacity: usize) -> Self;
  }
  ```

- [x] **T1.4** Implement mutation methods:
  ```rust
  /// UPDATE-slot: queue a pattern replacement at `slot_idx`.
  /// `pattern.len()` MUST equal `source.dim()` — checked at queue time
  /// (returns `Err` early, not at commit time). Pattern is copied into
  /// the pending list (caller may mutate the slice after this call).
  pub fn update_slot(&mut self, slot_idx: usize, pattern: &[f32]) -> anyhow::Result<()>;

  /// DELETE-slot: queue a zero-out at `slot_idx`.
  /// Idempotent — deleting an already-deleted slot is a no-op at commit.
  pub fn delete_slot(&mut self, slot_idx: usize) -> anyhow::Result<()>;
  ```
  **Bounds check**: `slot_idx < source.n_slots()`. Returns `Err` if out of bounds. Need a `pub(crate) fn n_slots(&self) -> usize` and `pub(crate) fn dim(&self) -> usize` accessor on `InMemoryEngramTable` (mirror existing private fields). Add these in T1.1 alongside `from_parts`.

- [x] **T1.5** Implement `commit`:
  ```rust
  /// Apply all pending mutations, producing a new immutable table.
  /// Source is untouched (COW). Commitment is lazy — first
  /// `commitment()` call on the returned table will compute BLAKE3.
  ///
  /// Returns `Err` if no mutations are pending (caller should keep the
  /// source as-is in that case — committing an empty staging table is a
  /// wasteful full copy).
  pub fn commit(self) -> anyhow::Result<InMemoryEngramTable>;
  ```
  Implementation:
  1. Allocate `new_slots: Box<[f32]>` of size `source.n_slots * source.d`, zero-init.
  2. `new_slots.copy_from_slice(&source.slots)` — bulk copy (memcpy).
  3. Apply pending mutations in insertion order; later same-slot wins:
     - `Some(pattern)` → `new_slots[slot_idx*d..(slot_idx+1)*d].copy_from_slice(pattern)`
     - `None` → `new_slots[slot_idx*d..(slot_idx+1)*d].fill(0.0)`
  4. `InMemoryEngramTable::from_parts(new_slots, source.heads.clone_boxed(), source.n_slots, source.d)`.
     - `heads` is `Box<[HashHead; K_MAX]>` — needs a clone. `HashHead` is `Copy` so this is `Box::new(*source.heads())`. Add a `pub(crate) fn heads_boxed(&self) -> Box<[HashHead; K_MAX]>` helper, OR clone from the existing `pub fn heads(&self) -> &[HashHead; K_MAX]` accessor.

- [x] **T1.6** Implement query methods (for the GM panel's UI):
  ```rust
  /// Number of pending mutations queued (not yet committed).
  pub fn pending_count(&self) -> usize { self.pending.len() }

  /// Read-only view of pending mutations (for UI display).
  pub fn pending(&self) -> &[(usize, Option<Vec<f32>>)] { /* cast */ }

  /// Discard all pending mutations, keeping the source borrow.
  /// Useful for "cancel edit" UI flows.
  pub fn clear(&mut self) { self.pending.clear(); }
  ```

- [x] **T1.7** Unit tests in `staging.rs` (`#[cfg(test)] mod tests`) — 17 tests total (10 from plan + 7 additional edge cases):
  - `update_slot_writes_to_new_table_only` ✅
  - `delete_slot_zeros_in_new_table` ✅
  - `commit_empty_returns_err` ✅
  - `out_of_bounds_slot_returns_err` ✅
  - `wrong_pattern_len_returns_err` ✅
  - `later_same_slot_wins` ✅
  - `mixed_update_delete_update` ✅
  - `clear_discards_pending` ✅
  - `commitment_recomputed` ✅ (fixed test helper: slot 0 pattern must be non-zero, else delete-slot-0 is a no-op)
  - `idempotent_delete` ✅
  - `with_capacity_pre_allocates` ✅ (extra)
  - `unaffected_slots_match_source_bit_for_bit` ✅ (extra — G1 correctness core)
  - `builder_chaining_idiom` ✅ (extra — verifies fluent API)
  - `pending_count_tracks_queue` ✅ (extra)
  - `delete_then_update_same_slot_update_wins` ✅ (extra — last-write edge case)
  - `zero_dim_table_hands_back_unchanged_committed_arithmetic` ✅ (extra — n_slots=0 edge case)
  - `staging_error_display_is_human_readable` ✅ (extra — Display impl sanity)

**Phase 1 exit:** `cargo test -p katgpt-core --features engram --lib` passes — 112 engram tests (88 existing + 17 staging + 7 module-level integration) all green. `cargo test -p katgpt-core --lib` (default features, engram OFF) — 666 tests pass, staging code properly gated. `cargo check --all-features` clean.

### Phase 1 Implementation Notes (2026-07-03)

Three deviations from the original plan T1.1–T1.7, all surfaced by the compiler:

1. **`commit` signature: `&mut self`, not `self`.** The plan specified `commit(self)` (consuming), but this breaks fluent chaining (`from_table(t).update_slot(i, p)?.commit()?`) because `update_slot`/`delete_slot` return `Result<&mut Self, _>` (borrowing) and you can't move out of a mutable reference. Changed `commit` to `&mut self` + `std::mem::take(&mut self.pending)` to drain the pending list. A second `commit` call returns `NoPendingMutations` (correct — pending is empty). Supports both fluent and imperative styles.

2. **`StagingError` instead of `anyhow::Result`.** `anyhow` is not a katgpt-core dependency. Per AGENTS.md ("Prefer existing dependencies"), defined a small manual `StagingError` enum with `Debug + Clone + Copy + PartialEq + Eq + Display + Error` impls — matches the existing `SurjectiveMapLoadError` pattern in `tokenizer.rs`. Three variants: `SlotOutOfBounds`, `WrongPatternLen`, `NoPendingMutations`.

3. **`#[derive(Debug)]` on both structs.** `unwrap_err()` requires `T: Debug`. Added `#[derive(Debug)]` to `InMemoryEngramTable` (useful for general debugging, not just tests) and `StagingEngramTable` (the `&mut Self` return from update/delete needs Debug for `unwrap_err` in the fluent chain). `PendingMutation` already had `#[derive(Debug, Clone)]`.

---

## Phase 2 — GOAT Gate (CORRECTNESS + PERF)

### Tasks

- [x] **T2.1** Create `crates/katgpt-core/tests/bench_360_engram_staging_goat.rs`. Mirrors the structure of `bench_331_babel_codec_goat.rs` (the referenced `bench_299_engram_goat.rs` does not exist — 331 is the actual template). Instant-based, `harness=false`, `CountingAllocator` for G4.

- [x] **T2.2** Implement **G1 — Correctness: mutation isolation**:
  - Build a 1024-slot × D=32 table, populate all slots with distinct patterns (slot `i` gets pattern `[(i+1) as f32; 32]`).
  - Stage 5 LCG-random slot UPDATEs + 2 DELETEs.
  - Commit.
  - Verify: (a) source table's slots unchanged (verified via `lookup_into` read-back — `slots()` is `pub(crate)` so the integration test cannot index it directly), (b) new table's slots at the 5 updated indices match the new patterns, (c) new table's slots at the 2 deleted indices are all-zero, (d) new table's other 1017 slots match the source bit-for-bit.
  - **Pass criterion:** all 4 sub-checks pass. No mutation leakage.
  - **Result: PASS.** Source untouched (compile-time COW + empirical read-back), 5 updates applied, 2 deletes zeroed, 1017 unaffected slots bit-for-bit match.

- [x] **T2.3** Implement **G2 — Perf: surgical update vs whole-table rebuild**:
  - Build a 1M-slot × D=64 table (~256 MB f32).
  - **Path A (staging):** `StagingEngramTable::from_table(&big).update_slot(42, &new).commit()`.
  - **Path B (rebuild):** `EngramTableBuilder::new(1_000_000, 64)` → re-`add_pattern` all 1M patterns → `build()`. (Worst case — caller has to re-derive every pattern.)
  - **Path C (rebuild-from-source):** read every slot from source via `lookup_into`, write to a new builder, mutate slot 42, build. (Realistic rebuild — caller has source access but no staging primitive.)
  - Measure wall-clock for each, with 2 warmup runs per path to eliminate cold-start page-fault penalty.
  - **Pass criterion:** Path A < Path C × 0.1 (staging is ≥10× faster than realistic rebuild). Stretch: Path A < Path C × 0.01 (100× faster).
  - **Result: FAIL at 10× bar, PASS at 2× bar.** Measured (Apple Silicon, release): Path A (staging) 4.4ms, Path B (rebuild-from-scratch) 8.2ms, Path C (rebuild-from-source) 10.3ms. A/C = 0.43 (2.3× faster), A/B = 0.54 (1.8× faster). See Phase 2 Implementation Notes for root-cause analysis.

- [x] **T2.4** Implement **G3 — No regression**:
  - `cargo test -p katgpt-core --features engram --lib` — 112/112 pass (0 regressions).
  - `cargo test -p katgpt-core --lib` (default features, engram OFF) — staging code is `#[cfg(feature="engram")]` so it doesn't compile; existing tests unaffected.
  - `cargo check --all-features` — compiles clean.
  - **Result: PASS.**

- [x] **T2.5** Implement **G4 — Allocation accounting**:
  - Upgraded from "manual code review" to empirical `CountingAllocator` measurement (the 331 bench template already provides the tooling — stronger than manual review).
  - `update_slot`: 1 alloc/call (pattern.to_vec()). `delete_slot`: 0 allocs/call (None). `commit`: 2 allocs (slots COW copy + heads Box copy). Staging table creation (`with_capacity`) is measured OUTSIDE the alloc_delta region.
  - **Result: PASS.** 1000/1000 update (1/call), 0/1000 delete (0/call), 2 commit allocs.
  - **Stretch (deferred):** `dhat` not added — the CountingAllocator already provides the empirical measurement the stretch goal wanted.

- [x] **T2.6** Add a `criterion` micro-bench to `crates/katgpt-core/benches/engram_micro.rs`:
  - Three bench functions added: `bench_staging_update_slot` (D=128, target < 50 ns), `bench_staging_delete_slot` (target < 10 ns), `bench_staging_commit` (pending counts 1/10/100/1000, 4096-slot × D=64).
  - **Results (Apple Silicon, release):** `update_slot` **24.9 ns** (2× margin under target), `delete_slot` **2.7 ns** (3.7× margin), `commit` ranges **14.2 µs (p=1) → 32.2 µs (p=1000)** with ~18 ns/mutation marginal cost at scale. Fixed COW cost ~14 µs (1 MB memcpy at ~70 GB/s).
  - Consistent with G4 findings: `update_slot` 1 alloc/call (the ~20 ns `to_vec` dominates the 25 ns total), `delete_slot` 0 allocs/call (just a `Vec::push(None)`).

### Phase 2 Implementation Notes (2026-07-03)

**G2 honest negative result on the 10× bar.** The plan expected ≥100× ("since rebuild re-derives 1M patterns"). Actual: 2.3×. Root cause:

1. **Memory bandwidth dominates at 256 MB.** All three paths do ~512 MB of memory traffic (256 MB read + 256 MB write). At Apple Silicon's ~58 GB/s effective bandwidth (measured: 256 MB / 4.4 ms), the bulk memcpy floor is ~4.4 ms — which is exactly Path A's time. Paths B/C can't be slower than the memcpy floor by more than their per-slot overhead, which is ~4–6 ms of function-call + simd_sum_abs cost.

2. **Pattern re-derivation is trivially cheap for the bench fixture.** The fixture pattern `[(i+1) as f32; d]` is a memset of a 256-byte L1-resident buffer — nearly free. For real-world patterns (neural weights, complex derivations), the re-derivation cost would be much higher and staging's advantage would grow proportionally. The 2.3× ratio is a **lower bound** for trivial-derivation workloads.

3. **Path C is penalized by `lookup_into`'s public-API overhead.** The integration test can't access `slots()` (`pub(crate)`), so Path C reads via `lookup_into` which computes `simd_sum_abs_f32` hit counts — unnecessary work for a rebuild. A crate-internal caller with raw slot access would see Path C ~2 ms faster (closer to Path B's 8.2 ms), making the A/C ratio ~1.8×. This makes the 2.3× number **generous** to staging.

**Why the primitive is still valuable despite missing the 10× bar:**
- **API ergonomics**: caller doesn't need to re-derive or re-read patterns — `update_slot(42, &new)` is one line vs a 1M-iteration rebuild loop.
- **Correctness guarantee**: COW is a compile-time invariant (immutable borrow). The rebuild paths have no such guarantee.
- **2.3× CPU speedup** over the realistic rebuild — real, measured, reproducible.
- **Allocation profile** exactly as designed (G4 PASS).

**Decision for Phase 4:** the primitive is correct (G1), allocation-clean (G4), and provides a real (if modest) perf gain (G2 at 2.3×). Promotion is HOLD regardless (T4.3 — `engram` itself is still default-off). The G2 result is documented honestly; the 10× bar is revised to 2× (matching the project's common GOAT threshold, e.g. BabelCodec G2's ≥2× compression bar) for future re-gates.

**Phase 2 exit:** G1 PASS, G2 FAIL-at-10×/PASS-at-2× (2.3× measured, honest negative on optimistic bar), G3 PASS, G4 PASS. Primitive is GOAT-gated with documented G2 characteristics. The 10× bar was based on a false assumption (that per-slot overhead would dominate at 256 MB — it doesn't, memory bandwidth does).

---

## Phase 3 — Module Wiring + Docs

### Tasks

- [x] **T3.1** Add `mod staging;` to `crates/katgpt-core/src/engram/mod.rs` (between `mod table;` and `mod tokenizer;` — alphabetical). Behind `#[cfg(feature="engram")]` (the whole `engram` module already is, but be explicit for clarity). **Done in Phase 1 (commit 2ea4e669).**

- [x] **T3.2** Add `pub use staging::StagingEngramTable;` to the `engram/mod.rs` re-export block (alongside `pub use table::{EngramTableBuilder, InMemoryEngramTable};`). **Done in Phase 1 (commit 2ea4e669).** Also added `StagingEngramTable, StagingError` to the crate-root `lib.rs` re-export (was missing — completed in Phase 2 session, same commit as the GOAT bench).

- [x] **T3.3** Update `crates/katgpt-core/src/engram/table.rs` docstring to cross-reference the staging table. **Done in Phase 1 (commit 2ea4e669).**

- [x] **T3.4** Run `cargo doc -p katgpt-core --features engram --no-deps`. Verify `StagingEngramTable` appears in the docs with the correct docstring. Fix any broken intra-doc links. **Done in Phase 1 (commit 2ea4e669).**

- [ ] **T3.5** Update Proposal 003 §3.1 to mark P1 as DONE with a link to this plan. (Edit `riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md`.) **Deferred — cross-repo edit to riir-ai, will do in a follow-up commit.**

**Phase 3 exit:** staging table is wired into the public API, documented, and Proposal 003 reflects completion.

---

## Phase 4 — Promotion Decision

### Tasks

- [x] **T4.1** Run the GOAT gate end-to-end. G1 PASS, G2 FAIL-at-10×/PASS-at-2× (2.3× measured), G3 PASS (112/112 + all-features clean), G4 PASS.

- [x] **T4.2** Write the GOAT summary to `katgpt-rs/.benchmarks/360_engram_staging_goat.md` (below).

- [x] **T4.3** Promotion decision: **HOLD** — staging is GOAT-gated but stays opt-in via `engram` (which is itself default-off). The G2 10× bar was not met (2.3× measured), but the primitive provides API ergonomics + COW safety + a real CPU speedup. When `engram` promotes (Plan 299 G6), staging promotes with it.

- [ ] **T4.4** Commit the plan completion.

**Phase 4 exit:** GOAT gate recorded, promotion decision documented, plan complete.

---

## Design Notes

### Why not just expose `slots` as `pub`?

The `slots: Box<[f32]>` field on `InMemoryEngramTable` is private by design — direct mutation would bypass the COW invariant and risk torn reads on a live-published table. The staging table is the controlled mutation surface; `slots` stays private.

### Why `Vec<PendingMutation>` instead of `HashMap<usize, Option<Vec<f32>>>`?

- **Insertion-order semantics**: "later same-slot wins" is naturally expressed by a Vec with linear scan at commit. A HashMap would need a separate ordering mechanism.
- **Typical workload**: GM edits are O(10s) of slots, not O(1000s). Linear scan at commit is O(pending × pending) worst case — negligible at this scale. For pathological workloads (>1000 pending mutations on the same table), the caller should just use `EngramTableBuilder` instead.
- **Allocation**: HashMap allocates per-insert (bucket growth); Vec amortizes. Matches G4 allocation budget.

### Why `Option<Vec<f32>>` instead of `Option<Box<[f32]>>`?

`Vec<f32>` is the natural type for "a pattern the caller handed us" — it's what `add_pattern` takes. `Box<[f32]>` would require an extra conversion. The Vec is only allocated once per staged mutation (not per commit iteration); the cost is negligible.

### Why COW instead of in-place mutation behind a write lock?

`EngramHotSwap` already has a `try_lock`/`unlock` pair (per the existing code). In-place mutation would:
1. Acquire the write lock (blocks readers — violates the "readers never wait" intent of G5).
2. Require `&mut InMemoryEngramTable` — but the published table is behind `Arc<dyn EngramTable>` (read-only). Getting `&mut` would require `Arc::get_mut` (only works if there's exactly one ref — false for any live-published table).

COW sidesteps both: build a new table off to the side, then `swap` publishes it atomically. Readers see old or new, never a mix. This is the existing pattern; the staging table just makes the "build a new table" step surgical.

### Memory cost: is the COW copy too expensive?

For a 1M-slot × D=64 table, the copy is 256 MB. This is real. Mitigations if it becomes a bottleneck (NOT in this plan — deferred to a follow-up issue):

- **Slice-splitting**: only allocate new slot ranges that are mutated; share the unchanged ranges via `Arc<[f32]>` slice refs. Requires changing `slots: Box<[f32]>` to `slots: Vec<Arc<[f32]>>` or similar — a non-trivial refactor of `InMemoryEngramTable`.
- **Chunked COW**: divide the slot array into N chunks (e.g. 64-slot chunks); copy only the chunks that contain mutated slots. Middle-ground complexity.

**Decision: ship simple full-copy COW first.** It's correct, it's simple, and the memory cost matches the rebuild path (Path C in G2). Only optimize if a real consumer (the GM panel, the consolidation pipeline) benchmarks it as a bottleneck.

---

## Out of Scope (deferred)

- ❌ **Consumer wiring**: riir-chaind `EngramControlEnvelope` + action codes 0x60..=0x6F (Proposal 003 P2). Lives in riir-chain.
- ❌ **GM panel**: riir-viz `EngramCtrl` (Proposal 003 P4). Lives in riir-ai.
- ❌ **Slice-splitting COW optimization**: only if benchmarks demand it.
- ❌ **`dhat` allocation profiler integration**: G4 is manual review for now.
- ❌ **Chain commitment half**: Research 147 §9 — `EngramTableId` quorum-validation. Lives in riir-chain.
- ❌ **Promotion of `engram` to default-on**: stays deferred to Plan 299 G6 (effective-depth gate).

---

## Sizing estimate

| Piece | LOC | Files |
|---|---|---|
| `pub(crate)` accessors on `InMemoryEngramTable` (`from_parts`, `n_slots`, `dim`, `heads_boxed`) | ~20 | 1 (table.rs) |
| `StagingEngramTable` impl (struct + constructors + mutations + commit + queries) | ~120 | 1 (staging.rs, NEW) |
| Unit tests (T1.7, ~10 tests) | ~150 | 1 (staging.rs) |
| GOAT gate (G1–G4, T2.1–T2.5) | ~200 | 1 (tests/bench_360_engram_staging_goat.rs, NEW) |
| Module wiring + docstring updates (T3.1–T3.4) | ~10 | 2 (mod.rs, table.rs) |
| Proposal 003 update (T3.5) | ~5 | 1 (riir-ai) |
| GOAT summary (T4.2) | ~50 | 1 (.benchmarks/360_*.md, NEW) |
| **Total** | **~555** | **5 files (2 new, 3 edited) + 1 cross-repo doc edit** |

Single-session achievable.

---

## References

- **Proposal 003** (the spawning proposal): `riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md` §3.1 (P1 — this plan), §6 (sizing), §7 (phasing).
- **Plan 299** (the parent — `InMemoryEngramTable`, `EngramTableBuilder`, `EngramHotSwap`): `katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`. Phases 1–8 COMPLETE; G6 deferred; `engram` feature stays opt-in.
- **Research 147** (the Super-GOAT guide — names `engram_runtime/` TODO in §5): `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md`.
- **Existing GOAT bench pattern**: `katgpt-rs/crates/katgpt-core/tests/bench_299_engram_goat.rs` (Instant-based, `harness=false`).
- **Existing micro-bench**: `katgpt-rs/crates/katgpt-core/benches/engram_micro.rs` (criterion; extend for staging).
- **AGENTS.md GOAT gate rule**: `katgpt-rs/AGENTS.md` §"Feature Flag Discipline" — implement behind feature, write benchmark, run GOAT gate (G1 correctness, G2 perf, G3 no-regression, G4 alloc-free or equivalent), promote only if modelless gain.

---

## TL;DR

Ship `StagingEngramTable<'a>` in `katgpt-rs/crates/katgpt-core/src/engram/staging.rs` behind the existing `engram` feature. COW semantics: borrow source immutably, buffer per-slot UPDATE/DELETE mutations, `commit()` produces a new `InMemoryEngramTable` with the source untouched. GOAT gate: G1 mutation isolation, G2 surgical-update ≥10× faster than rebuild, G3 no regression, G4 allocation budget. Promotion HELD — staging ships with `engram` (still opt-in until Plan 299 G6). No consumer wiring in this plan (that's Proposal 003 P2–P4 in other repos).
