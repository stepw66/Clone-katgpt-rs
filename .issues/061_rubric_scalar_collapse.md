# Issue 061: RubricBanditPruner Scalar Collapse — Rubric ≡ GZero

**Severity:** Bug (design flaw)
**Plan:** 071 (ROPD Rubric Modelless Distillation)
**Affected:** `bomber_09_rubric_tournament`, `fft_02_rubric_tournament`
**Test:** `tests/test_rubric_scalar_collapse.rs`

## Problem

`RubricPlayer` produces identical results to `GZeroPlayer` in arena tournaments:

- **Bomber:** Rubric 8.0% = GZero 8.0% — tied
- **FFT:** Rubric 60.0% = GZero 60.0% — tied
- **Head-to-head:** 40 games, 100% draws

The multi-criterion rubric vector (`RubricVector` with N criteria) is constructed and stored but **never used for differentiated decision-making**. It collapses to a single scalar before feeding to the bandit, making `RubricBanditPruner` mathematically equivalent to `DeltaBanditPruner`.

## Root Cause

`RubricBanditPruner::compute_reward()` collapses N criteria to 1 scalar:

```rust
// src/pruners/ropd_rubric/rubric_bandit.rs:261-272
fn compute_reward(&self, student: &RubricVector, reference: &RubricVector) -> f32 {
    let gap = reference.weighted_score() - student.weighted_score();
    //       ^^^^^^^^^^^^^^^^^^^^^^^^ COLLAPSE: N criteria → 1 scalar
    ...
    self.inner.update(arm, reward); // feeds scalar to bandit
}
```

Two `RubricVector`s with identical `weighted_score()` but different per-criterion profiles get the same reward:

| Profile | Scores | weighted_score | reward |
|---------|--------|---------------|--------|
| A | survival=1.0, safety=0.0, efficiency=0.0 | 0.571 | 0.429 |
| C | survival=0.5, safety=0.5, efficiency=1.0 | 0.571 | 0.429 |

These are strategically different game states but the bandit cannot distinguish them.

Additionally, `reference_rubric` is always `RubricVector::perfect(weights, 0)` (all 1.0), so `reward = 1.0 - student.weighted_score()` — a monotonic function of the same inputs GZero uses.

## Proof (3 tests)

Run: `cargo test --features "ropd_rubric,g_zero" --test test_rubric_scalar_collapse -- --nocapture`

1. **`test_rubric_scalar_collapse_bandit_reward`** — proves same `ws` → same reward, per-criterion gaps discarded
2. **`test_rubric_vs_scalar_delta_equivalence`** — proves `RubricBanditPruner ≡ DeltaBanditPruner` (diff=0.000000 for all arms)
3. **`test_rubric_absorb_uses_per_criterion_but_final_sum_collapses`** — absorb filtering works per-criterion ✅ but reward still scalar ❌

## Partial Win

`RubricGatedAbsorbCompress::observe_rubric()` DOES use per-criterion gaps correctly for filtering:

- High-weight gap (weight=4.0) → `above_threshold=true` ✅
- Low-weight gap (weight=1.0) → `above_threshold=false` ✅

But `compute_absorb_reward()` still collapses via `Σ(weight × gap)` — a weighted sum → single scalar.

## Impact

- Plan 071 hypothesis **rejected**: rubric adds ZERO improvement over scalar δ in both domains
- `bomber_09_rubric_tournament` and `fft_02_rubric_tournament` results are misleading — Rubric = GZero by construction
- The rubric infrastructure (types, scorer, templates) is valuable but the reward pathway wastes the multi-criterion signal

## Possible Fixes

1. **Per-criterion bandits** — one `BanditPruner` per criterion, arm selection aggregates across criteria
2. **Gap vector as multi-dimensional reward** — don't collapse to scalar, use vector reward
3. **Criterion-weighted arm mapping** — map criterion index to arm selection signal
4. **Dynamic reference rubrics** — don't use perfect (1.0, 1.0, 1.0), use outcome-based references that vary per round

## Files

| File | Role |
|------|------|
| `src/pruners/ropd_rubric/rubric_bandit.rs` | Bug: `compute_reward()` collapses to scalar |
| `src/pruners/ropd_rubric/rubric_absorb.rs` | Partial: filtering works, reward still scalar |
| `src/pruners/ropd_rubric/types.rs` | `weighted_score()`, `gap_criteria()` (exists but unused) |
| `src/pruners/bomber/rubric_player.rs` | Always uses `RubricVector::perfect()` as reference |
| `src/pruners/fft/rubric_player.rs` | Same issue |
| `tests/test_rubric_scalar_collapse.rs` | 3 tests proving the bug |
| `.benchmarks/009_arena_integration.md` | Arena results showing Rubric = GZero |