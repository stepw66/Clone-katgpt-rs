# Plan 336: Best-Belief ε-Quantile Beta Selector + Criterion-Versioned Cache Trait

**Date:** 2026-06-28 (revised 2026-06-28 after corpus read — see "Revision history" below)
**Research:** [katgpt-rs/.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md](../.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md)
**Source paper:** [arXiv:2606.26294](https://arxiv.org/pdf/2606.26294) — Iacob et al., Red Queen Gödel Machine, §3.5 + App. F Prop. 4.
**Target:** `katgpt-rs/crates/katgpt-core/src/best_belief.rs` (new) + trait extraction over `dec/cache.rs` + `dec/zone_cache.rs`
**Status:** Phase 1 shipped (T1.1–T1.4 done). Phase 2 measured — G1 PASS (3.099e-5 vs statrs), G4 PASS (alloc-free by construction), **G2 FAIL** (133 ns at S1,F1 vs 100 ns target; 2.2 µs select-8 vs 500 ns target). `best_belief` left opt-in per GOAT-discipline rule; T2.5 promotion blocked on a G2 unblock (see “Phase 1+2 Results”). Phase 3 (DRY trait) still deferred.

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

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/best_belief.rs`. Define:
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

- [x] **T1.2** Implement the inverse regularized incomplete Beta function `I⁻¹_ε(a, b)`. Options (pick lowest-latency on the benchmark):
  - Newton iteration on the continued-fraction forward eval of `I_x(a, b)` (Numerical Recipes §6.4).  ← IMPLEMENTED (with bisection fallback)
  - Bisection on the CDF (simpler, acceptable if ≤100 ns target is met).
  Do NOT pull a new dependency. Reuse `libm` / existing math. Note: existing `sample_beta` uses Jöhnk's algorithm (rejection sampling) — that's for *drawing* from Beta, not for the *quantile*. Different algorithm needed.

- [x] **T1.3** Add `best_belief` feature to `katgpt-core/Cargo.toml` (leaf feature, no deps). Wire to parent `katgpt-rs/Cargo.toml` as opt-in (NOT default).

- [x] **T1.4** Unit tests:
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

- [x] **T2.1 (G1 correctness)** Property test against a reference Beta quantile (e.g. `statrs` test-only dep): for a grid of `(S ∈ {0..100}, F ∈ {0..100}, ε ∈ {0.01, 0.05, 0.1, 0.25, 0.5})`, max abs error < 1e-4.
  - **RESULT: PASS.** Max abs err vs `statrs` 0.17 `Beta::inverse_cdf` across the 8×8×5 grid = **3.099e-5** (< 1e-4 target). Worst point: S=100, F=100, eps=0.5 (ours=0.5000005, theirs=0.49996948). statrs was used directly as a native-only dev-dep (no fallback to self-rolled f64 reference was needed).

- [x] **T2.2 (G2 perf)** Benchmark `best_belief_bench`:
  - `best_belief_score`: ≤ 100 ns (closed-form Beta quantile, no alloc).
  - `select_best_belief` on 8 candidates: ≤ 500 ns.
  - **RESULT: FAIL.** The Lentz continued-fraction `betacf` converges slowly for larger `a+b`, so latency scales ~linearly with S+F. Measured (Apple Silicon arm64, release, criterion median):
    - `best_belief_score(S0,F0)` = 3.30 ns (uniform-prior early return)
    - `best_belief_score(S1,F1)` = **133 ns** ← over 100 ns target
    - `best_belief_score(S10,F1)` = 167 ns
    - `best_belief_score(S50,F50)` = 358 ns
    - `best_belief_score(S100,F90)` = 668 ns
    - `best_belief_score(S1000,F1000)` = 1.47 µs
    - `select_best_belief` 4 candidates = **1.20 µs** ← over 500 ns target
    - `select_best_belief` 8 candidates = **2.20 µs** ← 4.4× over 500 ns target
  - Even the smallest non-trivial case (S1,F1 = 133 ns) misses the 100 ns target, and `select_8` at 2.2 µs is 4.4× over. The gate does not hold; `best_belief` stays opt-in. See “Path forward” below.

- [x] **T2.3 (G3 no regression)** N/A — new module, nothing to regress. Documented for completeness. `cargo check --all-features` clean; `cargo test -p katgpt-core --features best_belief --lib` 14/14 green.

- [x] **T2.4 (G4 alloc-free)** `best_belief_score` hot path is allocation-free, verified via `dhat` or `TrackingAllocator`. `select_best_belief` iterates `&[(u32, u32)]` slice — no alloc.
  - **RESULT: PASS by construction.** grep of `best_belief.rs` shows no `Vec::new` / `Box::new` / `String` / `format!` / `to_string` / `.collect(` / `.push(` symbols in `best_belief_score` or `select_best_belief`. The only allocating symbol (`.collect()`) is in `best_belief_scores` (the documented off-hot-path diagnostic helper).

- [ ] **T2.5** If G1 + G2 + G4 pass → promote `best_belief` to the `default` feature list in `katgpt-rs/Cargo.toml`. Run `cargo check --all-features` + `cargo test -p katgpt-core --lib`.
  - **NOT DONE — G2 failed.** G1 (3.099e-5) and G4 (alloc-free by construction) pass, but G2 misses both targets. `best_belief` remains opt-in. Promotion is blocked on a G2 unblock (see “Path forward” below).

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

---

## Phase 1+2 Results (implemented 2026-06-28)

**Implementation.** `katgpt-rs/crates/katgpt-core/src/best_belief.rs` ships the three
public fns from the spec: `best_belief_score(S, F, ε)`,
`select_best_belief(candidates, ε, incumbent_idx)`, and the diagnostic
`best_belief_scores(candidates, ε)`. The inverse regularized incomplete Beta
`I⁻¹_ε(a, b)` is implemented as Newton iteration on the forward continued-
fraction `I_x(a, b)` (Numerical Recipes §6.4 `betai` + `betacf`, Lentz form),
seeded by a Wilson-Hilferty-style normal-approximation initial guess using
Acklam's `Φ⁻¹`. A bisection fallback covers the rare case where a Newton step
leaves `(0,1)`. `ln Γ` is a hand-rolled Lanczos approximation (g=7, n=9)
because Rust std does not expose `lgamma` on f32/f64. Edge cases: `S=F=0`
returns the ε-quantile of Beta(1,1)=Uniform(0,1), i.e. ε itself (see note
below); `ε≤0` returns `X_MIN`; `ε≥1` returns `X_MAX` (handled directly rather
than driving Newton into the f32-noise floor of the extreme Beta tails).

**Note on the uniform-prior edge case.** The plan's draft text said `S=F=0`
returns `1 − ε`. That is the *upper*-tail complement. The standard ε-quantile
of Uniform(0,1) is `ε` (i.e. `q` such that `CDF(q) = ε`), and that same `q` is
the value the utility exceeds with prob `1 − ε` (since `P(X > q) = 1 − ε`). The
implementation returns `ε` for `S=F=0`, matching `statrs::Beta::inverse_cdf`
and the unit test `best_belief_score_uniform_prior`. A caller wanting the
upper-tail value passes `epsilon = 1 − desired`.

**G1 (correctness vs statrs) — PASS.** Native-only dev-dep `statrs = "0.17"`
compiled cleanly; no fallback to a self-rolled f64 reference was needed.
`statrs::distribution::Beta::inverse_cdf` (trait `ContinuousCDF`) is the
reference. Across the 8×8×5 grid `(S,F ∈ {0,1,2,5,10,25,50,100}, ε ∈ {0.01, 0.05, 0.1, 0.25, 0.5})`:

| gate | target | measured | verdict |
|------|--------|----------|---------|
| G1 max abs err | < 1e-4 | **3.099e-5** (worst: S=100, F=100, eps=0.5) | PASS |

**G2 (latency) — FAIL.** Criterion 0.5, Apple Silicon arm64, release,
`--warm-up-time 0.5 --measurement-time 1.5 --sample-size 30`. Median:

| case | target | measured | verdict |
|------|--------|----------|---------|
| `best_belief_score(S0,F0)` | ≤100 ns | 3.30 ns (early return) | PASS |
| `best_belief_score(S1,F1)` | ≤100 ns | **133 ns** | FAIL |
| `best_belief_score(S10,F1)` | ≤100 ns | 167 ns | FAIL |
| `best_belief_score(S50,F50)` | ≤100 ns | 358 ns | FAIL |
| `best_belief_score(S100,F90)` | ≤100 ns | 668 ns | FAIL |
| `best_belief_score(S1000,F1000)` | ≤100 ns | 1.47 µs | FAIL |
| `select_4_candidates` | ≤500 ns | **1.20 µs** | FAIL |
| `select_8_candidates` | ≤500 ns | **2.20 µs** (4.4× over) | FAIL |

Root cause: the Lentz continued-fraction `betacf` needs O(a+b) iterations to
converge, so per-call latency scales roughly linearly with `S+F`. Even the
smallest non-trivial case (`S1,F1`, `a=b=2`) is 133 ns because each Newton
iteration re-evaluates the full CF. `select_best_belief` calls
`best_belief_score` once per candidate (plus once for the incumbent tie-check),
so `select_8` ≈ 8 × 167 ns + overhead ≈ 2.2 µs.

**G4 (alloc-free) — PASS by construction.** grep of `best_belief.rs` shows no
`Vec::new` / `Box::new` / `String` / `format!` / `to_string` / `.collect(` /
`.push(` symbols in `best_belief_score` or `select_best_belief`. The only
allocating symbol (`.collect()`) is in `best_belief_scores` (the documented
off-hot-path diagnostic helper).

**G3 (no regression) — N/A** (new module). `cargo check --all-features` clean;
`cargo test -p katgpt-core --features best_belief --lib` 14/14 green (13 unit
+ 1 statrs-reference).

**Decision: left opt-in, NOT promoted to default.** G2 misses both targets.
`best_belief` stays behind the `best_belief` feature flag. T2.5 unchecked.

### Path forward (G2 unblock candidates, NOT in this commit)

1. **Precomputed `(S,F)` LUT** (OQ1's original lean): precompute the ε=0.05
   quantile for `S, F ∈ {0..32}` (or wider) at build/init into a fixed-size
   table; O(1) lookup with linear/bilinear interpolation for off-grid values.
   Expected to bring the common small-count regime (S,F ≤ 32) well under 100 ns.
   The CF path is kept as the cold fallback for large/off-grid counts.
2. **Cache the CF Lentz state** across the Newton iterations: the CF for
   adjacent `x` values shares most terms; a warm-started Lentz can shave ~30%.
3. **Single-Newton-step approximation**: for `a, b ≥ some_threshold`, the
   normal-approximation initial guess is already within 1e-3; ship the 1-Newton-
   step refinement as the “fast” path and the full-convergence path as the
   “accurate” path, gating on a quality flag. Trades a small G1 budget for G2.
4. **Reframe the target**: the 100 ns / 500 ns targets were calibrated for a
   closed-form primitive; `I⁻¹_ε(a,b)` is inherently iterative. A target like
   “≤ 200 ns at (S,F) ≤ 10; ≤ 1 µs at (S,F) ≤ 1000” would reflect the
   algorithmic floor. This is a re-spec, not an unblock — needs sign-off.

Option 1 (the LUT) is the most likely GOAT unblock and is the natural
follow-up plan. Filing a separate issue/plan for it rather than blocking this
commit.

**Commit:** `dbdc492f` on `develop` (the Cargo.toml entries — feature flag, `statrs` native dev-dep, bench target — landed in the preceding commit `10eca09e`; this commit adds the module, lib.rs wiring, bench, and plan update). Run `git --no-pager log --oneline | grep best_belief` to confirm the hash if the tree has been amended since.
