# Issue 012: RCD GOAT Gate Benchmark Infrastructure

**Date:** 2026-06-13
**Plan:** 258 (Task 5.5)
**Priority:** Medium
**Type:** Benchmark Infrastructure

## Problem

Plan 258 Task 5.5 requires a GOAT gate comparison: RCD (Residual Context Diffusion) vs DMax baseline at equivalent TPS (tokens per second). The RCD implementation is complete (Phases 1-4), but the benchmark infrastructure to measure "accuracy at throughput-matched decode speed" does not exist yet.

## What's Needed

1. **TPS-matched benchmark harness** — run DMax and RCD at the same tokens/sec, measure output quality
2. **Accuracy metric** — compare generated tokens against a reference (e.g., GSM8K-equivalent)
3. **Step-count comparison** — measure denoise steps to convergence with vs without RCD

## Current State

- RCD core implementation: ✅ Done (Phases 1-4)
- Unit tests: ✅ 6 tests passing
- MUX-RCD fusion: ✅ Done (Task 4.1)
- MUX-RCD DDTree wiring: ⏸ Deferred (Task 4.2 — needs DDTree score access)
- GOAT gate benchmark: ❌ This issue

## Acceptance Criteria

- [-] Benchmark can run DMax vs RCD with identical input/seeds — **scope ambiguity**: DMax (Plan 109, `dmax_spd`) is a decode strategy, RCD (Plan 258, `rcd_residual`) is residual injection — they compose, not compete. Need user decision: compare RCD-only vs DMax-only, or RCD+DMax vs DMax-only?
- [-] Reports: accuracy diff, step reduction ratio, per-step overhead — feasible at synthetic scale (~400 LoC, follows `riir-ai/crates/riir-gpu/tests/goat_289_rcd_ab_tests.rs::goat_t64` template). Real-model TPS comparison blocked on trained dLLM (none exists).
- [-] GOAT verdict: promote RCD to default if ≥2pp accuracy OR ≥1.5× step reduction — blocked on above scoping decision + benchmark implementation

## Findings from investigation

- **Training-side RCD GOAT is already done**: `riir-ai/crates/riir-gpu/tests/goat_289_rcd_ab_tests.rs::goat_t64_dflare_rcd_vs_dflare` reports accuracy diff, step reduction, and overhead with thresholds (3pp accuracy, 1.3× step reduction, 10% overhead).
- **Inference-side RCD exists in katgpt-rs**: `src/dllm.rs::denoise_loop_rcd` + `RcdConfig`. Existing tests verify correctness (`test_rcd_disabled_matches_baseline`, `test_rcd_enabled_converges_and_injects`, `test_rcd_vs_baseline_no_regression`) but no GOAT benchmark.
- **DMax exists in katgpt-rs**: `src/speculative/d2f.rs`, feature `dmax_spd`, Plan 109.
- **`dmax_spd` and `rcd_residual` have never been co-enabled in a test** — unknown if they compose cleanly. This is the #1 thing to verify before writing any benchmark.

## Path forward (when unblocked)

1. **Verify `dmax_spd` + `rcd_residual` compose**: small test that enables both and confirms no panic / no NaN.
2. **Decide scope**: RCD-only vs DMax-only, or RCD+DMax vs DMax-only.
3. **Build synthetic-scale GOAT benchmark** in katgpt-rs following the `goat_289` template. ~400 LoC, single test file. Reuses `micro_dllm()` config + pattern-data trainer.
4. **Defer real-model TPS comparison** until trained dLLM exists.

## Blocks

Plan 258 promotion of `rcd_residual` from opt-in to default feature.
