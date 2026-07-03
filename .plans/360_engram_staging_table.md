# Plan 360: Engram Staging Table — First-Class Per-Slot C/U/D

**Date:** 2026-07-03
**Proposal:** [riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md](../../../riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md) (P1)
**Parent plan:** [katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md](299_Engram_Hash_Addressed_Pattern_Memory.md) (Phase 2 — `InMemoryEngramTable`, `EngramTableBuilder`)
**Target:** `katgpt-rs/crates/katgpt-core/src/engram/staging.rs` (new module)
**Cargo feature:** `engram` (existing — no new feature flag; sibling to `table.rs`)
**Status:** Active — proposed. Ship behind existing `engram` (still default-off, gated on Plan 299 G6).

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

- [ ] **T1.1** Add `pub(crate) fn from_parts(slots: Box<[f32]>, heads: Box<[HashHead; K_MAX]>, n_slots: usize, d: usize) -> Self` to `InMemoryEngramTable` in `table.rs`. Constructs without re-validating (the staging table has already validated dimensions). Inline. Single-line body — delegates to struct literal.

- [ ] **T1.2** Create `crates/katgpt-core/src/engram/staging.rs` behind `#[cfg(feature = "engram")]`. Define `StagingEngramTable<'a>`:
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

- [ ] **T1.3** Implement constructors:
  ```rust
  impl<'a> StagingEngramTable<'a> {
      /// Borrow `source` immutably. Empty pending list.
      pub fn from_table(source: &'a InMemoryEngramTable) -> Self;

      /// Borrow `source`, pre-allocate `capacity` pending slots
      /// (avoid realloc on the pending Vec if the caller knows the edit count).
      pub fn with_capacity(source: &'a InMemoryEngramTable, capacity: usize) -> Self;
  }
  ```

- [ ] **T1.4** Implement mutation methods:
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

- [ ] **T1.5** Implement `commit`:
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

- [ ] **T1.6** Implement query methods (for the GM panel's UI):
  ```rust
  /// Number of pending mutations queued (not yet committed).
  pub fn pending_count(&self) -> usize { self.pending.len() }

  /// Read-only view of pending mutations (for UI display).
  pub fn pending(&self) -> &[(usize, Option<Vec<f32>>)] { /* cast */ }

  /// Discard all pending mutations, keeping the source borrow.
  /// Useful for "cancel edit" UI flows.
  pub fn clear(&mut self) { self.pending.clear(); }
  ```

- [ ] **T1.7** Unit tests in `staging.rs` (`#[cfg(test)] mod tests`):
  - `update_slot_writes_to_new_table_only` — stage an update, commit, verify the new table has the new pattern at the slot AND the source table is unchanged (compare via `commitment()`).
  - `delete_slot_zeros_in_new_table` — stage a delete, commit, verify the slot is zero in the new table AND non-zero in the source.
  - `commit_empty_returns_err` — committing with no pending mutations returns `Err`.
  - `out_of_bounds_slot_returns_err` — `update_slot(n_slots, ...)` returns `Err`.
  - `wrong_pattern_len_returns_err` — `update_slot(i, &[0.0; d+1])` returns `Err`.
  - `later_same_slot_wins` — stage `update_slot(5, A)` then `update_slot(5, B)`, commit, verify slot 5 == B.
  - `mixed_update_delete_update` — stage U/D/U on different slots, commit, verify each.
  - `clear_discards_pending` — stage mutations, `clear()`, `pending_count() == 0`.
  - `commitment_recomputed` — source table has cached commitment C1; after staging + commit, new table has commitment C2 ≠ C1 (BLAKE3 differs because contents differ).
  - `idempotent_delete` — stage `delete_slot(7)` twice, commit succeeds, slot 7 is zero.

**Phase 1 exit:** `cargo test -p katgpt-core --features engram --lib` passes (existing 88 engram tests + ~10 new staging tests). The staging table is correct COW.

---

## Phase 2 — GOAT Gate (CORRECTNESS + PERF)

### Tasks

- [ ] **T2.1** Create `crates/katgpt-core/tests/bench_360_engram_staging_goat.rs`. Mirrors the structure of `tests/bench_299_engram_goat.rs` (Instant-based, `harness=false`).

- [ ] **T2.2** Implement **G1 — Correctness: mutation isolation**:
  - Build a 1024-slot × D=32 table, populate all slots with distinct patterns (slot `i` gets pattern `[i as f32; 32]`).
  - Stage 5 random slot UPDATEs + 2 DELETEs.
  - Commit.
  - Verify: (a) source table's `commitment()` unchanged, (b) new table's slots at the 5 updated indices match the new patterns, (c) new table's slots at the 2 deleted indices are all-zero, (d) new table's other 1017 slots match the source bit-for-bit.
  - **Pass criterion:** all 4 sub-checks pass. No mutation leakage.

- [ ] **T2.3** Implement **G2 — Perf: surgical update vs whole-table rebuild**:
  - Build a 1M-slot × D=64 table (~256 MB f32).
  - **Path A (staging):** `StagingEngramTable::from_table(&big).update_slot(42, &new).commit()`.
  - **Path B (rebuild):** `EngramTableBuilder::new(1_000_000, 64)` → re-`add_pattern` all 1M patterns → `build()`. (Worst case — caller has to re-derive every pattern.)
  - **Path C (rebuild-from-source):** read every slot from source, write to a new builder, mutate slot 42, build. (Realistic rebuild — caller has source access.)
  - Measure wall-clock for each.
  - **Pass criterion:** Path A < Path C × 0.1 (staging is ≥10× faster than realistic rebuild). Stretch: Path A < Path C × 0.01 (100× faster).
  - **Note on the COW memory cost:** Path A allocates a fresh 256 MB slot array (the COW copy). This is the same cost as Path C. The win is **CPU** (no per-slot `add_pattern` overhead, no hash-mod-N re-derivation) and **API** (caller doesn't need to re-derive patterns). The benchmark measures CPU time; memory cost is identical. The risk noted in Proposal 003 §8 (slice-splitting optimization) is deferred — only optimize if a real consumer benchmarks it as a bottleneck.

- [ ] **T2.4** Implement **G3 — No regression**:
  - `cargo test -p katgpt-core --features engram --lib` — all 88 + ~10 new tests pass.
  - `cargo test -p katgpt-core --lib` (default features, engram OFF) — staging code is `#[cfg(feature="engram")]` so it doesn't compile; existing 7400+ tests unaffected.
  - `cargo check --all-features` — compiles clean (catches feature-combo regressions per the `merkle_root` lesson).

- [ ] **T2.5** Implement **G4 — Allocation accounting**:
  - The staging table allocates: (1) the `pending: Vec<PendingMutation>` (one entry per staged mutation — unavoidable, caller-driven), (2) each `Some(Vec<f32>)` pattern copy (unavoidable — caller may mutate the slice after queueing), (3) the new `Box<[f32]>` slot array on commit (unavoidable — COW).
  - **Pass criterion:** the staging table does NOT allocate inside `update_slot`/`delete_slot` beyond the `pending.push` (amortized O(1) via Vec capacity) and the pattern copy. Specifically: no per-call `HashMap`, no per-call `Box`, no format strings. Verify via manual code review (the `dhat` crate is not currently a katgpt-core dep; adding it for one bench is overkill).
  - **Stretch (deferred):** add `dhat` as an optional dev-dep gated behind a `bench_alloc` feature, measure heap allocations during commit. Defer unless G2 reveals a surprise.

- [ ] **T2.6** Add a `criterion` micro-bench to `crates/katgpt-core/benches/engram_micro.rs` (existing — extend, don't create new):
  - Bench group `staging`: `update_slot` latency (target < 50 ns, just a Vec push + memcpy of D f32s), `commit` latency at varying pending counts (1, 10, 100, 1000 mutations on a 1024-slot table).
  - Mirrors the existing `lookup_into` / `multi_head_hash` / `sigmoid_fuse_into` bench entries.

**Phase 2 exit:** G1 PASS (mutation isolation proven), G2 PASS (staging ≥10× faster than rebuild — expected to be ≥100× since rebuild re-derives 1M patterns), G3 PASS (no regressions), G4 PASS (allocation budget honored). The primitive is GOAT-gated.

---

## Phase 3 — Module Wiring + Docs

### Tasks

- [ ] **T3.1** Add `mod staging;` to `crates/katgpt-core/src/engram/mod.rs` (between `mod table;` and `mod tokenizer;` — alphabetical). Behind `#[cfg(feature="engram")]` (the whole `engram` module already is, but be explicit for clarity).

- [ ] **T3.2** Add `pub use staging::StagingEngramTable;` to the `engram/mod.rs` re-export block (alongside `pub use table::{EngramTableBuilder, InMemoryEngramTable};`).

- [ ] **T3.3** Update `crates/katgpt-core/src/engram/table.rs` docstring to cross-reference the staging table:
  ```rust
  //! After `build()`, the slots and heads are immutable; the only lazy state
  //! is the cached BLAKE3 commitment. For surgical per-slot edits without
  //! rebuilding the whole table, see [`StagingEngramTable`].
  ```

- [ ] **T3.4** Run `cargo doc -p katgpt-core --features engram --no-deps`. Verify `StagingEngramTable` appears in the docs with the correct docstring. Fix any broken intra-doc links.

- [ ] **T3.5** Update Proposal 003 §3.1 to mark P1 as DONE with a link to this plan. (Edit `riir-ai/.proposals/003_engram_crud_table_tier_access_matrix.md`.)

**Phase 3 exit:** staging table is wired into the public API, documented, and Proposal 003 reflects completion.

---

## Phase 4 — Promotion Decision

### Tasks

- [ ] **T4.1** Run the GOAT gate end-to-end: `cargo test -p katgpt-core --features engram --lib && cargo test -p katgpt-core --features engram --test bench_360_engram_staging_goat`. All gates must pass.

- [ ] **T4.2** Write the GOAT summary to `katgpt-rs/.benchmarks/360_engram_staging_goat.md`:
  - G1 result (pass/fail + evidence).
  - G2 result (latency numbers: staging vs rebuild, ratio).
  - G3 result (test counts before/after, no regressions).
  - G4 result (allocation review).
  - Decision: PROMOTE / DEMOTE / HOLD.

- [ ] **T4.3** Promotion decision:
  - **PROMOTE the staging table to default-on?** — NO. The `engram` feature itself is still default-off (deferred to Plan 299 G6 effective-depth gate). Promoting staging alone would be inconsistent. The staging table is GOAT-gated and ready, but it ships **with** `engram` — when `engram` promotes, staging promotes with it.
  - **Decision: HOLD** — staging is GOAT-PASS but stays opt-in via `engram`. Update Proposal 003 §3.1 and §7 P5 to reflect this.

- [ ] **T4.4** Commit the plan completion: update task checkboxes, commit with `feat:` prefix (this is a new primitive, not just docs).

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
