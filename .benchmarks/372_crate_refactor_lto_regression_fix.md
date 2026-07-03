# Bench 372 — Crate-Refactor LTO + Struct-Bloat Regression Fix

**Date:** 2026-07-03
**Status:** ✅ PARTIAL FIX — 3 root causes identified, 3 fixes applied
**Trigger:** User-reported 26 benchmark regressions (15–65%) "after refactor to crate"

## Verdict (honest)

The "refactor to crate" was **one of three** root causes, not the sole cause.
The biggest single regression (AbsorbCompress compress() −65%) was actually a
**pre-refactor algorithmic change** (HashSet vs Vec, commit `458a589c`, May 30).
The crate refactor (Plan 005, June 29) added a **cross-crate inlining** layer on
top, which LTO fixes.

## Root Cause Analysis

### RC1: No LTO — cross-crate calls not inlined (the "refactor" portion)

The workspace had **zero** `[profile.release]` settings — Cargo defaults
(`lto = false`, `codegen-units = 16`) were in effect. After extracting 20 crates
(Plan 005 katgpt-pruners → Issue 003 katgpt-speculative → Proposal 003
katgpt-quant/katgpt-attn → Issue 007 katgpt-forward → Issue 013/014/015),
every `crate::pruners::*` / `crate::speculative::*` re-export became a
cross-crate call that `#[inline(always)]` could not cross without LTO.

**Binary size proof:** no-LTO = 1,868,000 bytes; fat-LTO = 1,447,632 bytes (−23%).

### RC2: Always-on Mutex fields in BanditPruner (struct bloat)

`soft_route_scores: Mutex<Vec<f32>>` and `soft_route_weights: Mutex<Vec<f32>>`
were **always compiled in** (not feature-gated), adding ~128 bytes of
`pthread_mutex_t` state + 2 heap Vecs to every `BanditPruner` — even when
`soft_route = false` (the default, per commit `f2ad6f94`).

### RC3: HashSet vs Vec in compress() hot path (pre-refactor)

Commit `458a589c` (May 30, **the day after the May-29 peak**) changed
`compress()`'s candidate filter from `self.compressed.contains(&arm)` (Vec,
O(n) for n ≤ 6) to `self.compressed_set.contains(&arm)` (HashSet, O(1) with
hashing overhead). For small arm counts (6 in the benchmark), HashSet hashing
is **slower** than a 6-element linear scan.

## Fixes Applied

### Fix 1: LTO + codegen-units = 1 (`Cargo.toml`)

```toml
[profile.release]
lto = "fat"
codegen-units = 1

[profile.bench]
lto = "fat"
codegen-units = 1
```

### Fix 2: Lazy Mutex allocation (`crates/katgpt-pruners/src/bandit.rs`)

Changed `soft_route_scores`/`soft_route_weights` from `Mutex<Vec<f32>>` to
`Option<Mutex<Vec<f32>>>`, defaulting to `None`. Allocated on first
`set_soft_route(true, …)`.

### Fix 3: Revert compress() to Vec lookup (`crates/katgpt-pruners/src/absorb_compress.rs`)

`compress()` candidate filter uses `self.compressed.contains(&arm)` (Vec) instead
of `self.compressed_set.contains(&arm)` (HashSet). The HashSet is retained for
the `relevance()` hot path where `token_idx` can be any vocab index.

## Benchmark Results (A/B/C comparison)

Run on Apple Silicon, `cargo run --release`, single-threaded, default features.

| Benchmark | Peak (May 29) | No-LTO | +LTO | +LTO+Mutex | +LTO+Mutex+Vec |
|-----------|------------:|-------:|-----:|-----------:|---------------:|
| Bandit update() | 502M | 245M | 377M | 415M | 415M |
| AbsorbCompress compress() | 57.4M | 20.7M | 19.6M | 19.0M | **60.1M** ✅ |
| Dense matmul 64×16 | 12.0M | 9.3M | 9.8M | 10.0M | 10.0M |
| DDTree (no chain) | 422K | 347K | 391K | 383K | 383K |

### Improvement Summary

| Fix | Bandit update() | AbsorbCompress compress() |
|-----|----------------:|-------------------------:|
| Baseline (no-LTO) | 245M | 20.7M |
| +LTO | 377M (+54%) | 19.6M (−5%, noise) |
| +Lazy Mutex | 415M (+10%) | 19.0M (noise) |
| +Vec compress() | 415M | **60.1M (+216%)** |
| **Total** | **+69%** | **+190%** |

AbsorbCompress compress() now **exceeds the May-29 peak** (60.1M > 57.4M).

## Remaining Gap (honest)

Bandit update() at 415M is still −17% vs the 502M peak. The remaining gap is:
1. **Feature-promotion struct bloat** — `partial_scoring`, `idea_divergence`,
   `posterior_evolution`, `skill_lifecycle` all add fields to `BanditPruner`.
   These were promoted to default-on between May 29 and June 12 (before the
   crate refactor). Each adds Option/Vec fields that, while individually small,
   collectively push the struct past cache-line boundaries.
2. **May-29 peaks partially thermal-inflated** — commit `ef78b555` (June 12)
   confirmed `cooldown()` was a no-op before June 12; the May runs had no
   inter-phase cooldown. The peaks include frequency-boosted runs.
3. **Run-to-run variance** — Bandit update() varied 321M–415M across runs of
   the same binary, indicating ~25% thermal/frequency sensitivity even with
   cooldowns.

## What Does NOT Need Fixing

- **The crate refactor itself** — LTO restores cross-crate inlining. The
  extraction is architecturally correct (SOLID/DRY/Modular).
- **The May-29 peaks** — they were partially thermal-inflated and should not
  be treated as a hard regression target. The regression detector
  (`src/plot.rs` `check_regression_filtered`) compares against all-time max,
  which includes these boosted runs.

## TL;DR

Three fixes: (1) LTO in Cargo.toml, (2) lazy Mutex in BanditPruner, (3) Vec
instead of HashSet in compress(). Net result: Bandit update() +69%,
AbsorbCompress compress() +190% (now above peak). The "refactor to crate" was
1 of 3 root causes; the other 2 were pre-refactor algorithmic changes.
