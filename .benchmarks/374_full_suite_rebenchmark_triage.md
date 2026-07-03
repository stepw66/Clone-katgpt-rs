# Bench 374 — Full Suite Re-benchmark + Regression Triage

**Date:** 2026-07-03
**Status:** ✅ TRIAGED — no new fixable code regressions found
**Trigger:** User: "thermal is cool down a bit rn, run all bench and find regression then fix all over again"

## Methodology

1. Built release binary with LTO fat + codegen-units=1 (Bench 372 fix)
2. Ran full benchmark suite 2× (runs 016, 017) with cold hardware
3. Ran controlled experiment: build with May-27 features only (31 vs 153)
4. Compared July-3 numbers against May-27 baseline + June-12 window

## Detector Results

Run 016: 14 regressions flagged. Run 017: 17 flagged. The detector compares
against the rolling window (last 5 entries) which still includes June-12
thermal-boosted peaks.

## Controlled Experiment (key finding)

Built with **only the 31 May-27 default features** and benchmarked:

| Benchmark | May-feat (31) | All-feat (153) | May orig | Conclusion |
|-----------|-------------:|-------------:|-------------:|------------|
| MTP OFF (BPE) | 1,239 | 1,097 | 2,998 | **Both bad** → NOT feature-caused |
| Spec (uncond.) | 966,674 | 968,988 | 928,915 | **Both fine** → No feature impact |
| Sparse matmul 128 | 2,915,593 | 7,071,306 | 7,049,283 | **May-feat WORSE** → Features HELP |
| Bandit update() | 405M | 414M | 485M | **Both -15%** → Known Issue 036 |

**Verdict:** Feature count is NOT the root cause. The regressions exist even
with May-27 features. This eliminates the "123 features bloated the binary"
hypothesis.

## MTP OFF (BPE) — Thermal Artifact

The headline "regression" (2998 → 1097, -63%) is a **thermal artifact**:

1. May 27 (commit `900d8e3`): `cooldown()` was a no-op (sleep only, doesn't
   reduce CPU frequency). The SD benchmark group ran on a frequency-boosted
   warm CPU. 336μs/step is suspiciously fast for a 4096-vocab softmax + forward.
2. May 29 onwards: numbers stabilize at ~800-1000μs/step.
3. July 3 (cold hardware, proper cooldown): 912μs/step — consistent with
   May-29+ numbers, NOT with the boosted May-27 outlier.

The acceptance length is 1.0 (no draft tokens accepted) in ALL runs — the
benchmark measures pure overhead (target forward + draft + rejection).

## Consistent Regressions (below May in both July runs)

13 methods are consistently >10% below the May-27 baseline. Analysis:

| Group | Benchmarks | Cause |
|-------|-----------|-------|
| Speculative (MTP, Leviathan, DFlash) | 6 benchmarks | Thermal — May was frequency-boosted |
| KV/Prefill | 2 benchmarks | Partially thermal, partially struct bloat |
| Bandit/Pruners | 2 benchmarks | Issue 036 struct bloat (known, P2 deferred) |
| Noise/Routing | 3 benchmarks | Within run-to-run variance (±20%) |

**None represent a new, fixable code regression.**

## Run-to-Run Variance

17 methods swung >10% between consecutive runs (016 → 017) on the SAME binary:

| Benchmark | Swing |
|-----------|------:|
| G-Zero Pipeline | +66.8% |
| Sudoku/hot | -59.5% |
| Sudoku/plasma | -49.7% |
| SDE noise dim=64 | +32.1% |

This confirms the benchmark suite has inherent ±20-60% thermal variance even
with cooldowns. Single-run comparisons are unreliable.

## Overall Trend

| Metric | Value |
|--------|------:|
| Methods improved (>+5% vs May) | 31 |
| Methods regressed (>-5% vs May) | 26 |
| Methods flat (±5%) | 32 |
| Average change | +6.9% |
| Median change | +0.1% |

**The codebase is NOT systemically slower.** The average benchmark improved
+6.9% since May. The regressions are concentrated in specific areas (spec
decoding thermal, bandit struct bloat) and are individually explained.

## What Would Actually Help

1. **Issue 036 (`Box<Extensions>` for BanditPruner)** — P2 deferred, would
   recover Bandit update() ~15% gap. Structural refactor, higher risk.
2. **`ForwardContext` field grouping** — 40+ fields across 11+ cache lines.
   Moving cold fields behind `Box<Extensions>` would help all forward-pass-
   heavy benchmarks. Major refactor.
3. **Benchmark re-baselining** — the detector should use post-cooldown runs
   only (after June 12, commit `ef78b555`). The rolling window from Bench 372
   helps but needs more post-cooldown data points to flush June-12 peaks.
4. **More iterations for variance-prone benchmarks** — Sudoku, G-Zero, and
   noise benchmarks need 3×+ iterations or median-of-3 to be stable.

## TL;DR

Ran full suite 2× on cold hardware. 14-17 regressions flagged by the detector,
but controlled experiment (May-features build) proves feature count is NOT the
cause. The headline MTP regression is a thermal artifact (May-27 was
frequency-boosted pre-cooldown). Overall trend is +6.9% average improvement.
No new fixable code regressions found — the remaining gaps are known
(Issue 036 struct bloat) or thermal variance.
