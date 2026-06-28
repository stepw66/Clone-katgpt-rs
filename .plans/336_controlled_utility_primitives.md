# Plan 336: Best-Belief ε-Quantile Beta Selector + Criterion-Versioned Cache Trait

**Date:** 2026-06-28 (revised 2026-06-28 after corpus read — see "Revision history" below)
**Research:** [katgpt-rs/.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md](../.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md)
**Source paper:** [arXiv:2606.26294](https://arxiv.org/pdf/2606.26294) — Iacob et al., Red Queen Gödel Machine, §3.5 + App. F Prop. 4.
**Target:** `katgpt-rs/crates/katgpt-core/src/best_belief.rs` (new) + trait extraction over `dec/cache.rs` + `dec/zone_cache.rs`
**Status:** Active — Phase 1 (`best_belief`) pending; Phase 2 (DRY trait) deferred

---

## Revision history

**v1 (2026-06-28 initial):** Proposed two new primitives: `CriterionVersionedRecords<D>` (selective erasure) + `best_belief_score()` (ε-quantile Beta). Verdict: GOAT for both. Super-GOAT fusion tracked as Issue 004.

**v2 (2026-06-28 correction, this version):** After reading the riir-ai Super-GOAT corpus (R158 Committed Personality Blend, R161 Cognitive Branch, R155 Sub-Goal Compaction) and grepping the shipped code (`dec/cache.rs` `DecCache`, Plan 335 `ZoneGeometryCache`):

- **`CriterionVersionedRecords<D>` is NOT a new primitive.** `DecCache` (katgpt-core `dec/cache.rs`) and `ZoneGeometryCache` (Plan 335 Phase 2, katgpt-core `dec/zone_cache.rs`) already ship the criterion-versioned erasure pattern. `DecCache` even includes derived-stat recomputation (`store_hodge`, `store_betti`). The value is a **DRY trait extraction**, not a new capability → **Gain**, not GOAT.
- **`best_belief_score()` IS genuinely new.** Grep confirms `sample_beta` exists (Jöhnk's algorithm, Thompson sampling) but no inverse-CDF / Beta quantile function for conservative *selection*. → **GOAT** (standalone).
- **Super-GOAT fusion (Issue 004) is dead.** The candidate selling point ("per-NPC selective forgetting on personality swap") is a paraphrase of Research 158 §1.3 property #3 (sampling invariance) + §2.4 (sync boundary). Issue 004 closed.
- **LoRA vocabulary was stale.** The actual modelless erasure substrate is freeze/thaw (`MerkleFrozenEnvelope`) + geometry bins (`ZoneGeometryPod` / `ZoneGeometryCache` with `topology_version` + `SourceShardHashMismatch`). LoRA hot-swap is pre-spinoff framing. Corrected throughout.

## Goal

Ship one genuinely-new GOAT primitive + one Gain-tier DRY extraction:

1. **`best_belief_score()` / `select_best_belief()`** (GOAT) — ε-quantile Beta lower bound `BB_ε = I⁻¹_ε(1 + S, 1 + F)` for conservative *selection* of frozen snapshots / archetype blend shards / zone geometry pods. Complements the existing `sample_beta` Thompson sampling (exploration) with a conservative-exploitation counterpart. Modelless analog of RQGM Prop. 4.

2. **`CriterionVersionedCache<V>` trait** (Gain, deferred) — extract the common interface already implemented independently by `DecCache` and `ZoneGeometryCache`. Both have: a version tag (`topology_version`), an `is_valid(version)` check, an `invalidate()` / `invalidate(key, new_version)` method, and a BLAKE3 / hash integrity check. The trait unifies them; existing impls get `impl CriterionVersionedCache for DecCache` / `ZoneGeometryCache`. Pure DRY, no new behavior.

**Not in scope:** Super-GOAT fusion, per-NPC selective forgetting (Issue 004 — closed, covered by R158/R161/R155), controlled-utility-evolution reframe of MAPE-K (architectural observation only, lives in Research 320 §2.2.3).

## Phase 1 — `best_belief` GOAT primitive (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/best_belief.rs`. Define:
  ```rust
  /// ε-quantile of Beta(1 + successes, 1 + failures) — the conservative lower bound
  /// the candidate's true utility exceeds with probability 1 − ε (RQGM Prop. 4).
  /// Used for SELECTION (which frozen snapshot / archetype shard / zone pod to promote),
  /// complementing `sample_beta` (Thompson sampling for EXPLORATION).
  ///
  /// epsilon ∈ (0, 1), typically 0.05. Lower ε = more conservative.
  pub fn best_belief_score(successes: u32, failures: u32, epsilon: f32) -> f32;

  /// Select the candidate with the highest best-belief score. Ties favor `incumbent_idx`
  /// (if provided) to avoid unnecessary snapshot swaps and the cache invalidation they trigger.
  pub fn select_best_belief(
      candidates: &[(u32 /* S */, u32 /* F */)],
      epsilon: f32,
      incumbent_idx: Option<usize>,
  ) -> usize;
  ```

- [ ] **T1.2** Implement the inverse regularized incomplete Beta function `I⁻¹_ε(a, b)`. Options (pick lowest-latency on the benchmark):
  - Newton iteration on the continued-fraction forward eval of `I_x(a, b)` (Numerical Recipes §6.4).
  - Bisection on the CDF (simpler, acceptable if ≤100 ns target is met).
  Do NOT pull a new dependency. Reuse `libm` / existing math. Note: existing `sample_beta` uses Jöhnk's algorithm (rejection sampling) — that's for *drawing* from Beta, not for the *quantile*. Different algorithm needed.

- [ ] **T1.3** Add `best_belief` feature to `katgpt-core/Cargo.toml` (leaf feature, no deps). Wire to parent `katgpt-rs/Cargo.toml` as opt-in (NOT default).

- [ ] **T1.4** Unit tests:
  - `best_belief_score_monotone_in_successes` — holding failures fixed, more successes → higher score.
  - `best_belief_score_at_epsilon_half_is_mean` — at ε = 0.5 the score → posterior mean `α/(α+β)` (sanity check, within 1e-3).
  - `best_belief_score_lower_epsilon_is_more_conservive` — ε=0.05 < ε=0.5 → lower score for same (S, F).
  - `best_belief_score_bounds` — always in `(0, 1)` for finite S, F.
  - `select_best_belief_favors_incumbent_on_tie` — exact tie → incumbent_idx returned.
  - `select_best_belief_picks_higher_lower_bound` — candidate (S=10, F=1) beats (S=100, F=90) at ε=0.05 (the conservative property: small-but-confident beats large-but-noisy).
  - `select_best_belief_with_none_incumbent_picks_argmax` — no incumbent → pure argmax.
  - Numerical accuracy vs a reference (e.g. `statrs` behind a `dev-dependency` only, gated to `#[cfg(test)]`): max abs error < 1e-4 across a grid of (S, F, ε).

## Phase 2 — GOAT Gate (BLOCKS PROMOTION of `best_belief`)

### Tasks

- [ ] **T2.1 (G1 correctness)** Property test against a reference Beta quantile (e.g. `statrs` test-only dep): for a grid of `(S ∈ {0..100}, F ∈ {0..100}, ε ∈ {0.01, 0.05, 0.1, 0.25, 0.5})`, max abs error < 1e-4.

- [ ] **T2.2 (G2 perf)** Benchmark `best_belief_bench`:
  - `best_belief_score`: ≤ 100 ns (closed-form Beta quantile, no alloc).
  - `select_best_belief` on 8 candidates: ≤ 500 ns.

- [ ] **T2.3 (G3 no regression)** N/A — new module, nothing to regress. Documented for completeness.

- [ ] **T2.4 (G4 alloc-free)** `best_belief_score` hot path is allocation-free, verified via `dhat` or `TrackingAllocator`. `select_best_belief` iterates `&[(u32, u32)]` slice — no alloc.

- [ ] **T2.5** If G1 + G2 + G4 pass → promote `best_belief` to the `default` feature list in `katgpt-rs/Cargo.toml`. Run `cargo check --all-features` + `cargo test -p katgpt-core --lib`.

## Phase 3 — `CriterionVersionedCache<V>` DRY trait (GAIN, DEFERRED)

Deferred until Phase 2 ships. The trait extracts the common interface already implemented by `DecCache` and `ZoneGeometryCache`:

```rust
/// Common interface for caches keyed by a criterion version (topology_version,
/// snapshot BLAKE3, archetype library hash, etc.). Already implemented independently
/// by `DecCache` (katgpt-core::dec::cache) and `ZoneGeometryCache` (katgpt-core::dec::zone_cache).
pub trait CriterionVersionedCache {
    type Version;
    type Key;
    fn is_valid(&self, key: &Self::Key, version: &Self::Version) -> bool;
    fn invalidate(&mut self, key: &Self::Key, new_version: Self::Version);
    fn current_version(&self, key: &Self::Key) -> Option<&Self::Version>;
}
```

This is pure DRY — no new behavior, no new capability. Existing impls get blanket or explicit trait impls. The value is: (a) future caches (e.g. HLA eigenbasis cache from Issue 001, Gram cache from Plan 279) implement one trait instead of inventing their own `is_valid` / `invalidate` vocabulary; (b) generic ` invalidated_keys(cache)` / `bulk_invalidate` helpers.

### Tasks (Phase 3, deferred)

- [ ] **T3.1** Define `CriterionVersionedCache` trait in `katgpt-core/src/cache_version.rs`.
- [ ] **T3.2** `impl CriterionVersionedCache for DecCache` (single-slot — Key = ()).
- [ ] **T3.3** `impl CriterionVersionedCache for ZoneGeometryCache` (multi-entry — Key = ZoneHash).
- [ ] **T3.4** Document the pattern in `katgpt-core/src/dec/cache.rs` doc-comment, pointing to the trait.
- [ ] **T3.5** No GOAT gate (Gain-tier DRY refactor). Just `cargo check --all-features` + existing tests pass bit-identically.

## Open Questions

- **OQ1:** `best_belief_score` — free function vs `Beta` struct? Lean: free function. If Phase 2 perf misses, consider a precomputed lookup table for common `(S, F)` pairs (e.g. S, F ∈ {0..32}).
- **OQ2:** `select_best_belief` — should it expose per-candidate scores (for diagnostics) or just return the winner? Lean: return winner; add `best_belief_scores(candidates, epsilon) -> Vec<f32>` as a separate function for diagnostics.
- **OQ3 (crate-promotion coordination — riir-ai Issue 007 Phase E):** When Issue 007 Phase E promotes `katgpt-core` sub-crates, `best_belief.rs` should land in a future `katgpt-bandits` crate (alongside `BanditPruner` / `BanditStrategy` / `sample_beta`). Adding it to `katgpt-core/src/` now is fine — Phase E is deferred, re-homing 1 file is trivial. **No hold on this plan** — Issue 007's refactor is subtractive (removes `sense/` IP); this plan is additive (new generic math); zero file overlap.
- **OQ4 (use-case framing):** The intended consumers are freeze/thaw selection (which `ArchetypeBlendShard` to promote for an NPC), zone cache promotion (which `ZoneGeometryPod` to keep hot), and anchor-gated snapshot swaps (RQGM §3.5 pattern: promote challenger only if it strictly raises ε-best-belief on anchor). NOT LoRA hot-swap — LoRA is pre-spinoff vocabulary. The runtime substrate is `MerkleFrozenEnvelope` + `ZoneGeometryCache` + `ArchetypeBlendShard`.

## Non-Goals

- Super-GOAT fusion / per-NPC selective forgetting (Issue 004 — CLOSED, covered by R158/R161/R155).
- Co-evolution training loop (LLM evaluator improvement) → riir-train.
- Controlled-utility-evolution runtime (epoch freeze + boundary replacement + selective erasure as a unified runtime abstraction) — architectural observation only, no plan. Research 320 §2.2.3 documents the mapping to existing modules (`MerkleFrozenEnvelope`, `mape_k.rs`, `consolidation.rs`, `latent_functor/reestimation.rs`, `committed_blend/`).

## References

- [Research 320](../.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md) (corrected).
- [Issue 004](../.issues/004_per_npc_selective_forgetting_super_goat_fusion.md) (CLOSED — not novel).
- RQGM paper §3.5 (Controlled Utility Evolution), App. F Prop. 4 (best-belief lower bound).
- `katgpt-core/src/pruners/bandit.rs` `sample_beta` (Jöhnk's) — the existing Beta *sampler* (exploration) that `best_belief_score` complements as the Beta *quantile* (conservative selection).
- `katgpt-core/src/dec/cache.rs` `DecCache` — existing criterion-versioned cache (single-slot, with derived stats). Phase 3 trait extraction target.
- `katgpt-core/src/dec/zone_cache.rs` `ZoneGeometryCache` (Plan 335) — existing criterion-versioned cache (multi-entry, papaya lock-free, BLAKE3-tagged). Phase 3 trait extraction target.
- riir-ai Research 158 (Committed Personality Blend) — the already-committed Super-GOAT that ships the per-NPC committed-personality-with-survives-swap capability.
