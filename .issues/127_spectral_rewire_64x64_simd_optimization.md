# Issue 127 — Spectral Rewiring 64×64 Cached-Index SIMD Optimization

> **Spawned from:** Plan 423 (Spectral Rewiring), Benchmark 423 §G5 "Open Follow-ups"
> **Date:** 2026-07-10
> **Type:** optimization (SIMD / auto-vectorization)
> **Severity:** LOW — the 64×64 case is NOT a real per-NPC workload (see below)
> **Status:** OPEN (future, not blocking)

## Context

The cached-index hot path (`spectral_rewire_with_index_into`) in
`crates/katgpt-spectral/src/spectral_rewire.rs` does four rank-r matmuls per
delta after a one-time SVD index build. Benchmark 423 §G5 measured:

| Scale | rank | mean | target | result |
|---|---|---|---|---|
| 8×8   | 4  | 0.41µs | ≤1µs  | PASS (true per-NPC hot-loop) |
| 512×64 | 32 | 947µs | ≤1ms  | PASS (LoRA-scale rows) |
| 64×64 | 8  | 29µs | ≤50µs | PASS (recalibrated) |

The 64×64 r=8 case lands at ~29µs (~75K flops of memory-bound rank-1 axpy at
~2.5 GFLOP/s effective). The original Plan 423 target was 10µs; it was
recalibrated to 50µs because the flop count doesn't support 10µs at this
memory-bound effective throughput.

## Why this is LOW priority

**The 64×64 case is not a real per-NPC workload.** Benchmark 423 §G5 documents
that Plan 423's "64×64 (reshaped style_weights)" was a misread:
`NeuronShard::style_weights[64]` has 64 *elements*, which reshape to **8×8**,
not 64×64. The 8×8 case (0.41µs) is the true per-NPC hot-loop size, and it
already passes its 1µs target by 2.4×.

The 64×64 case only matters if a future consumer reshapes a 64×64 weight block
through `spectral_rewire`. No such consumer exists today.

## Optimization opportunity

If the 64×64 path ever becomes a real workload, the four matmuls in the
cached-index hot loop could likely hit the original 10µs target with:

1. **Chunked loops (4 or 8 elements)** to help LLVM auto-vectorize the rank-r
   matmuls (per global optimization guidelines).
2. **Explicit SIMD** (`std::simd` or wide f32x8) for the inner axpy.
3. **Keeping the inner loop branch-free** (per hot-loop rules).

The current implementation is scalar; the ~2.5 GFLOP/s effective throughput
suggests it is memory-bound, so SIMD gains may be modest unless the access
pattern is also improved (e.g., contiguous rank-stride reads).

## Tasks

- [ ] **T1** Only if a real 64×64 (or larger) `spectral_rewire` consumer
  emerges: profile the four cached-index matmuls, confirm the memory-bound
  diagnosis, and decide whether SIMD or access-pattern reform is the right
  lever.
- [ ] **T2** If T1 justifies the work: implement chunked/SIMD inner loop, gate
  behind the existing `spectral_rewire` feature, re-run G5, confirm the 10µs
  target is met without regressing G1a/G2/G3/G4.

## Don't do this yet

Do not implement until a real >8×8 `spectral_rewire` consumer appears. The
8×8 hot-loop (the only confirmed per-NPC workload) is already 2.4× under
target. Filing this as a captured-for-the-record issue; no plan to be created
unless T1's precondition is met.

## Cross-references

- `.plans/423_spectral_rewire_primitive.md` — COMPLETE, opt-in
- `.benchmarks/423_spectral_rewire_goat.md` §G5, §"Open Follow-ups"
- `.issues/123_fusion_b_two_component_delta_decomposition.md` — CLOSED
- `.issues/124_spectral_rewire_svd_col_cap.md` — RESOLVED
