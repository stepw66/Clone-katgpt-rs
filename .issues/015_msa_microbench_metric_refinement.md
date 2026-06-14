# MSA Microbench Metric Refinement

**Source**: Issue 014 (MSA Arena RULER Benchmark Infrastructure) — Optimization candidates
**Priority**: Low
**Blocked**: No — purely diagnostic metric improvements on existing micro-benchmarks
**Depends**: Nothing (uses existing `tests/bench_256_*.goat.rs` infrastructure)

## Summary

Issue 014's full RULER arena is blocked on trained model weights + RULER dataset +
attention inference harness — none of which exist in katgpt-rs (modelless inference).
However, three metric redesigns on the **existing** synthetic micro-benchmarks are
fully tractable today and would sharpen Plan 256's GOAT verdict rationale.

These do **not** flip the GOAT verdict (per-group / KV-outer / adaptive-k all failed
their original micro-benchmark gates). They reframe *why* and *where* each strategy
wins or loses, which informs future promotion decisions.

## Acceptance Criteria

- [ ] **O1 — Per-group coverage metric redesign**: measure per-call partition spread
      (Jaccard distance between groups within a single call) instead of cross-query
      union. The current metric saturates at ~1.0× because 128 queries × 32 top-k
      touch all reachable blocks regardless of per-call diversity.
      - File: `tests/bench_256_per_group.goat.rs`
      - Predicted outcome: shows per-group DOES diversify per-call (design goal met)
        even though cross-query union saturates. Does NOT flip GOAT — per-group's
        real value is high-top_k latency (already measured, already a pass at 0.98×).

- [ ] **O2 — KV-outer query batching sweep**: sweep `N_QUERIES ∈ {256, 512, 1024, 2048}`
      instead of hardcoded 256. At 512K context with top_k=32, avg queries/block ≈ 1
      so reverse-index amortization gives nothing. Plan 256 line 120 already names
      this root cause.
      - File: `tests/bench_256_kv_outer.goat.rs`
      - Predicted outcome: at N_QUERIES=2048, KV-outer beats Q-outer at 128K because
        avg queries/block rises to ~8. Confirms existing analysis, sharpens regime
        boundary in the recommendation.

- [ ] **O3 — Adaptive-k precision@k**: add two alternative metrics using existing data:
      1. `precision@adaptive_k` = `|adapt ∩ dense_top{adapt_k}| / adapt_k`
      2. `weighted recall` = `Σ scores(adapt ∩ dense) / Σ scores(dense_top32)`
      Current `bench_256_adaptive_k.goat.rs:166` recall is mathematically capped at
      20/32 = 0.625 because adaptive k ≈ 20 < 32.
      - File: `tests/bench_256_adaptive_k.goat.rs`
      - Predicted outcome: precision@k likely shows adaptive-k picks well (just fewer).
        Weighted recall likely > 0.90 because high-score blocks dominate. Reframes
        recommendation: "compute saver at near-equivalent precision" instead of
        "fails recall."

## Why these are tracked separately from Issue 014

Issue 014's acceptance criteria are all transitively blocked on the arena
prerequisites (trained model, RULER dataset, harness). The optimization candidates
are the only items that don't need that infrastructure — they're micro-bench
refinements using synthetic data that already exists. Splitting them out lets the
micro-bench work proceed without waiting on the (possibly never-arriving) arena
prerequisites.

## Notes

- All three benchmarks already collect the data needed for the new metrics —
  no new measurement code is required, just new analysis passes over existing
  arrays.
- Each item is ~1 day of work: read existing bench, add metric calc, re-run,
  update Plan 256 verdict text.
- None of these flip the GOAT verdict — they add nuance to *why* each strategy
  wins or loses in its specific regime.
