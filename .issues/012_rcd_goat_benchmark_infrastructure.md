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

- [ ] Benchmark can run DMax vs RCD with identical input/seeds
- [ ] Reports: accuracy diff, step reduction ratio, per-step overhead
- [ ] GOAT verdict: promote RCD to default if ≥2pp accuracy OR ≥1.5× step reduction

## Blocks

Plan 258 promotion of `rcd_residual` from opt-in to default feature.
