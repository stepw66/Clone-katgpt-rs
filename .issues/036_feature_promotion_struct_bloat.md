# Issue 036 — Feature-Promotion Struct Bloat in BanditPruner/AbsorbCompressLayer

**Filed:** 2026-07-03
**Priority:** P2 (perf — not blocking, but accounts for remaining ~17% gap)
**Related:** `.benchmarks/372_crate_refactor_lto_regression_fix.md`

## Problem

Between May 29 and June 12, 2026, several features were promoted to default-on
that add fields to `BanditPruner` and `AbsorbCompressLayer`:

| Feature | Fields added to `BanditPruner` |
|---------|-------------------------------|
| `partial_scoring` | `partial_scorer: Option<Box<dyn PartialScorer>>` |
| `idea_divergence` | `idea_divergence: Option<IdeaDivergence>` + `arm_score_vectors: Vec<Vec<f32>>` |
| `skill_lifecycle` | `memory: PrunerMemory` |
| `bandit` | `shared_stats: Option<Arc<SharedBanditStats>>` |

Each field is individually justified (passed GOAT G1–G7 quality gates). But
collectively, they push `BanditPruner` from ~3 fields (May 29) to 13+ fields,
spreading the struct across multiple cache lines. The `Bandit update()`
benchmark (tight inner loop touching only `self.stats`) regressed ~30% from
the struct layout change alone (separate from the LTO fix in Bench 372).

## Root Cause

The GOAT gate checks the feature being promoted, but does NOT check whether
adding fields to a shared struct degrades OTHER benchmarks that don't use the
feature. This is a systemic blind spot.

## Proposed Fix (deferred — not blocking)

Group all optional/feature-gated extension fields behind a single `Box<Extensions>`:

```rust
pub struct BanditPruner<P: ScreeningPruner> {
    inner: P,
    strategy: BanditStrategy,
    stats: BanditStats,           // ← hot path: update(), relevance()
    thompson_cache: Vec<f32>,     // ← hot path: prepare_episode()
    // ── cold path: grouped behind one indirection ──
    extensions: Option<Box<BanditExtensions>>,
}

struct BanditExtensions {
    shared_stats: Option<Arc<SharedBanditStats>>,
    dual_cutoff: f32,
    soft_route: bool,
    soft_route_tau: f32,
    partial_scorer: Option<Box<dyn PartialScorer>>,
    idea_divergence: Option<IdeaDivergence>,
    arm_score_vectors: Vec<Vec<f32>>,
    memory: PrunerMemory,
    soft_route_scores: Option<Mutex<Vec<f32>>>,
    soft_route_weights: Option<Mutex<Vec<f32>>>,
}
```

This keeps the hot-path fields (`inner`, `strategy`, `stats`, `thompson_cache`)
in one cache line, and pushes all rarely-used extension fields behind a single
pointer indirection. The `Option<Box<>>` is 8 bytes when `None` (the fast path
for benchmarks that don't use extensions).

### Expected Gain

Bandit update() should recover the remaining ~17% gap (415M → ~500M) by
keeping `BanditStats` in a hotter cache line.

## Tasks

- [ ] Benchmark `BanditPruner` struct size before/after the `Box<Extensions>` refactor
- [ ] Implement `BanditExtensions` grouping
- [ ] Verify all 130 katgpt-pruners tests pass
- [ ] Run full bench suite, confirm Bandit update() ≥ 480M (within 5% of peak)
- [ ] Apply same pattern to `AbsorbCompressLayer` if bench shows gain

## Deferral Rationale

This is P2 because:
1. The LTO + lazy-Mutex + Vec-compress fixes (Bench 372) already recovered the
   biggest regressions (Bandit +69%, AbsorbCompress +190%).
2. The remaining 17% gap is within run-to-run thermal variance (~25%).
3. The `Box<Extensions>` refactor touches every constructor and every field
   access — higher risk than the Bench 372 fixes.

## TL;DR

Feature promotions (May 29 → June 12) bloated `BanditPruner` from 3 → 13
fields, causing cache-line sprawl. Fix: group cold fields behind
`Box<Extensions>`. Deferred as P2 — the acute regressions are fixed in Bench 372.
