# Plan 336: Controlled-Utility Primitives — Criterion-Versioned Records + Best-Belief Selector

**Date:** 2026-06-28
**Research:** [katgpt-rs/.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md](../.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md)
**Source paper:** [arXiv:2606.26294](https://arxiv.org/pdf/2606.26294) — Iacob et al., Red Queen Gödel Machine, §3.4–3.5 + App. F (Prop. 2, 4, 6).
**Target:** `katgpt-rs/crates/katgpt-core/src/criterion_store.rs` + `katgpt-rs/crates/katgpt-core/src/best_belief.rs`
**Status:** Active — Phase 1 (skeleton) pending

---

## Goal

Ship two generic modelless inference primitives distilled from RQGM's freeze/thaw consistency contract:

1. **`CriterionVersionedRecords<D>`** — a record store where each record carries a `dep_set` of evaluator/criterion slots and per-slot BLAKE3-tagged criterion versions. `erase_slot(m)` removes only records depending on slot `m` whose criterion tag mismatches the current one. Modelless analog of RQGM Prop. 2 (selective erasure preserves criterion consistency).
2. **`best_belief_score()` + `select_best_belief()`** — ε-quantile Beta lower bound `BB_ε = I⁻¹_ε(1 + S, 1 + F)` for conservative *selection* of frozen snapshots / adapters / direction vectors. Complements `BanditStrategy::ThompsonSampling` (exploration) with a conservative-exploitation counterpart. Modelless analog of RQGM Prop. 4.

Both ship behind a single `controlled_utility` feature flag (opt-in). GOAT gate must pass before promotion to default. This plan does **not** implement the Super-GOAT fusion (per-NPC selective forgetting) — that is tracked in [Issue 004](../.issues/004_per_npc_selective_forgetting_super_goat_fusion.md).

## Non-Goals

- The co-evolution training loop (LLM evaluator improvement) → riir-train.
- Wiring into Plan 279 / Plan 315 / Issue 001 as a refactor — that is a Phase 2 follow-up after the GOAT gate passes on the standalone primitives. Phase 1 ships the primitives + their own tests + a benchmark.
- Multi-slot commutative `erase_slots(&[m1, m2])` (RQGM Rem. 1) — Phase 2 refinement.

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/criterion_store.rs`. Define:
  ```rust
  pub struct CriterionVersionedRecord<D> {
      pub data: D,
      pub dep_set: SmallBitSet,         // slots whose criterion affected this record
      pub criterion_tags: SmallVec<[blake3::Hash; 4]>, // per-slot tag, indexed by slot
      pub epoch_vector: SmallVec<[u32; 4]>,
  }

  pub struct CriterionVersionedRecords<D> {
      records: Vec<CriterionVersionedRecord<D>>,
      current_criteria: Vec<blake3::Hash>, // per-slot current criterion tag
      current_epochs: Vec<u32>,
  }

  impl<D> CriterionVersionedRecords<D> {
      pub fn insert(&mut self, record: CriterionVersionedRecord<D>);
      pub fn erase_slot(&mut self, slot: usize); // retain where slot ∉ dep_set OR tag matches
      pub fn set_criterion(&mut self, slot: usize, tag: blake3::Hash, epoch: u32);
      pub fn iter_valid(&self) -> impl Iterator<Item = &D>;
      pub fn len(&self) -> usize;
      pub fn is_empty(&self) -> bool { self.len() == 0 }
  }
  ```
  Use `smallvec` + `smallbitset` (or hand-rolled `u64` bitset for ≤64 slots — typical case is ≤4 slots). Zero-allocation on the no-op erasure path (slot unchanged → early return).

- [ ] **T1.2** Implement `erase_slot` semantics precisely per RQGM Def. F.2:
  ```
  retain z where (slot ∉ z.dep_set) OR (z.criterion_tags[slot] == current_criteria[slot])
  ```
  Use `Vec::retain` for in-place filtering. Document the consistency invariant in a doc-test.

- [ ] **T1.3** Create `katgpt-rs/crates/katgpt-core/src/best_belief.rs`. Implement:
  ```rust
  /// ε-quantile of Beta(1 + successes, 1 + failures) — the conservative lower bound
  /// the candidate's true utility exceeds with probability 1 − ε (RQGM Prop. 4).
  /// epsilon ∈ (0, 1), typically 0.05.
  pub fn best_belief_score(successes: u32, failures: u32, epsilon: f32) -> f32;

  /// Select the candidate with the highest best-belief score. Ties favor incumbent_idx
  /// (if provided) to avoid unnecessary slot transitions and the erasure they trigger.
  pub fn select_best_belief(
      candidates: &[(u32, u32)],
      epsilon: f32,
      incumbent_idx: Option<usize>,
  ) -> usize;
  ```

- [ ] **T1.4** Implement the inverse regularized incomplete Beta function `I⁻¹_ε(a, b)`. Two options — pick the one with lower latency on the benchmark:
  - Newton iteration on `I_x(a, b)` (continued-fraction forward eval from Numerical Recipes §6.4).
  - Bisection on the CDF (slower but simpler; acceptable for ≤100 ns target if vectorized).
  Do NOT pull a new dependency. Reuse `libm` / existing math.

- [ ] **T1.5** Add `controlled_utility` feature to `katgpt-core/Cargo.toml`. Wire both modules behind it. Add to the parent `katgpt-rs/Cargo.toml` as an opt-in feature (NOT default).

- [ ] **T1.6** Unit tests:
  - `erase_slot_preserves_criterion_consistency` — random transition sequence, assert invariant holds after each transition (modelless analog of RQGM Prop. 2).
  - `erase_slot_noop_when_criterion_unchanged` — `set_criterion` with same tag → `erase_slot` is a no-op, zero allocations.
  - `erase_slot_only_removes_dependent_records` — records with `dep_set` not containing the slot survive; records containing it but with matching tag survive; records containing it with mismatched tag are erased.
  - `best_belief_score_monotone_in_successes` — holding failures fixed, more successes → higher score.
  - `best_belief_score_epsilon_zero_is_mean` — at ε → 0.5 the score → posterior mean (sanity check).
  - `select_best_belief_favors_incumbent_on_tie` — exact tie → incumbent_idx returned.
  - `select_best_belief_picks_higher_lower_bound` — candidate with (S=10, F=1) beats (S=100, F=90) at ε=0.05 (conservative).
  - Fuzz: random `(records, transitions)` sequences, assert consistency invariant never violated.

## Phase 2 — GOAT Gate (BLOCKS PROMOTION)

### Tasks

- [ ] **T2.1 (G1 correctness)** Property test: for any sequence of `set_criterion` + `erase_slot` operations, the store's retained records all satisfy the criterion-consistency invariant (every retained record's per-slot tags match the current criteria on every slot in its dep_set). 10k random sequences, seeds fixed.

- [ ] **T2.2 (G2 perf)** Benchmark on `criterion_store_bench`:
  - `erase_slot` on N=1000 records, k=4 slots, 25% dependency rate: ≤ 5 µs.
  - `erase_slot` no-op (slot unchanged): ≤ 5 ns (early return).
  - `best_belief_score`: ≤ 100 ns (closed-form Beta quantile).
  - `select_best_belief` on 8 candidates: ≤ 500 ns.

- [ ] **T2.3 (G3 no regression)** — DEFERRED to a separate plan. Wiring `CriterionVersionedRecords` into Plan 279 (Gram cache), Plan 315 (cascade invalidation), Issue 001 (HLA eigenbasis) is a refactor that produces bit-identical behavior. This is a follow-up plan, NOT a blocker for promoting the standalone primitives to default. The standalone primitives can promote once G1 + G2 + G4 pass.

- [ ] **T2.4 (G4 alloc-free)** Hot path (no-op erasure, `best_belief_score` lookup) is allocation-free, verified with `#[track_caller]` + `Alloc`-hooked test or `dhat` heap profile. `erase_slot` reuses `Vec::retain` (in-place, no alloc).

- [ ] **T2.5** If G1 + G2 + G4 pass → promote `controlled_utility` to the `default` feature list in `katgpt-rs/Cargo.toml`. Run `cargo check --all-features` + `cargo test -p katgpt-core --lib` to confirm.

## Phase 3 — Wiring (FOLLOW-UP, SEPARATE PLAN)

Deferred. After Phase 2 promotes the primitives, a follow-up plan (TBD number) will:
- Refactor Plan 279 Gram cache to use `CriterionVersionedRecords`.
- Refactor Plan 315 cascade invalidation to use `erase_slot(ZONE_SLOT)`.
- Refactor Issue 001 HLA eigenbasis check to use criterion-tag verification.
- Each refactor must produce bit-identical behavior to the existing hand-rolled invalidation (G3 of the follow-up plan).

## Open Questions

- **OQ1:** Should `CriterionVersionedRecords` implement a `recompute_derived_stats` hook (for CMP, Thompson accumulators), or should that stay in the consumer? Lean: consumer-side — the primitive is a pure record store, derived stats are domain-specific. Document this in the doc-comment.
- **OQ2:** For >64 slots, fall back to `bitvec` or hand-roll a `SmallVec<[u64; 1]>`? Lean: `SmallVec<[u64; 1]>` — covers up to 64 slots inline, heaps beyond. Typical case is ≤4 slots.
- **OQ3:** Should `best_belief_score` expose a `Beta` struct (reusable across calls) or stay as a free function? Lean: free function for now; if Phase 2 perf misses, consider a precomputed table for common (S, F) pairs.
- **OQ4 (crate-promotion coordination — riir-ai Issue 007 Phase E):** Issue 007 (riir-ai/.issues/007) Phase E defers breaking `katgpt-core` into smaller publishable crates (`katgpt-dec`, `katgpt-simd`, `katgpt-micro-belief`, `katgpt-hla`, `katgpt-personality`, `katgpt-sleep`, `katgpt-sense`, ...). When that lands, the two files added by this plan should be re-homed:
  - `best_belief.rs` → future `katgpt-bandits` (alongside `BanditPruner` / `BanditStrategy`). It's a Beta-quantile selection function, same family.
  - `criterion_store.rs` → future `katgpt-consistency` or `katgpt-freeze` (generic consistency primitive, no bandit semantics).
  Adding them to `katgpt-core/src/` now is fine — Phase E is explicitly deferred ("separate plans"), and re-homing 2 files later is trivial. This OQ exists so the Phase E implementer knows to claim both files. **No hold on this plan** — Issue 007's refactor is subtractive (removes `sense/` IP) + splits; this plan is additive (new generic substrate); zero file overlap.

## References

- Research 320 (originating distillation).
- RQGM paper §3.4 (Selective Erasure, Def. F.2), §3.5 (Controlled Utility Evolution), App. F Prop. 2 (consistency), Prop. 4 (best-belief lower bound), Prop. 6 (linear cost under exponential checkpoints).
- Plan 279 (Gram cache invalidation on snapshot bump — scattered instance to unify).
- Plan 315-riir-ai (`invalidate_zone_on_collapse` — scattered instance to unify).
- Issue 001 (HLA eigenbasis BLAKE3-check on reload — scattered instance to unify).
- Issue 004 (Super-GOAT fusion follow-up — NOT in scope for this plan).
