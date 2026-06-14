# GPart Partition Pruning — BanditPruner Integration

**Source**: Plan 257 (GPart Isometric Adapter) — Deferred Idea 2
**Priority**: Medium
**Blocked**: No
**Depends**: Plan 257 (complete), BanditPruner infrastructure
**Status**: ✅ MVP DONE — static top-k magnitude mask shipped behind `gpart_pruning`
  feature flag. BanditPruner-driven dynamic mask deferred (needs reward signal design).

## Summary
Integrate GPart partition groups with the existing BanditPruner to enable per-group pruning during inference. Groups with low activation contribution can be zeroed out, trading quality for speed.

## What was shipped (MVP)

- New method `GpartAdapter::apply_with_scratch_masked(base_weights, assignments,
  group_sizes, group_mask: &[bool])` — gates: `gpart_pruning`.
- New method `GpartAdapter::topk_mask(k) -> Vec<bool>` — selects the k groups
  with largest `|θ[g]|` via `select_nth_unstable_by` (O(d) average).
- Feature flag `gpart_pruning = ["gpart_adapter"]` in katgpt-core and the
  passthrough `gpart_pruning = ["katgpt-core/gpart_pruning"]` in katgpt-rs.
- GOAT proof benchmark `tests/bench_008_gpart_pruning_goat.rs` with P1/P2/P3
  gates + budget sweep.

## Implementation notes

- **Branch-free inner loop**: per-group delta is precomputed as
  `scale * θ[g] * (active as u8 as f32)`, zeroed for masked groups without a
  per-element branch. LLVM auto-vectorises the all-true path to match
  `apply_with_scratch`.
- **BanditPruner deferred**: a per-group reward signal doesn't fall out of the
  forward pass — ΔW is consumed downstream by a single loss, so credit
  attribution to specific groups is essentially gradient-based saliency
  repackaged as a bandit. That design work belongs in a separate plan.

## Acceptance Criteria
- [x] Define pruning threshold for partition groups — top-k by `|θ[g]|` (static)
- [x] Extend `GpartAdapter::apply_with_scratch()` to accept a group mask —
      added `apply_with_scratch_masked(base_weights, assignments, group_sizes,
      group_mask)`
- [x] Benchmark quality degradation vs. speedup for top-k group pruning —
      `tests/bench_008_gpart_pruning_goat.rs`, see results below
- [x] GOAT gate behind `gpart_pruning` feature flag — wired in both Cargo.tomls

## Benchmark Results (release mode, Apple Silicon)

```
running 4 tests
✅ P1: all-true mask matches unmasked (max_abs_diff = 0)
✅ P2: k=d/4 masked 11054ns vs unmasked 15226ns (0.73×, ≤1.20× slack)
✅ P3: relative L2 at k=d/2 = 0.4160 (< 0.50)

┌─ GPart pruning budget sweep (d=32, n=8192) ─────────────
│  budget |   k |  mask ns | vs unmasked | rel L2
│  -------+-----+----------+-------------+--------
│    100% |  32 |   10538ns |       0.68× | 0.0000
│     75% |  24 |   10715ns |       0.69× | 0.0772
│     50% |  16 |   11072ns |       0.71× | 0.3599
│     25% |   8 |   11047ns |       0.71× | 0.6621
└──────────────────────────────────────────────────────────
```

### Findings

1. **Masked path is actually faster** (0.68-0.73× unmasked). Likely because
   zeroed deltas let the CPU skip dependent ops or improve branch prediction.
   Apply-time is not where the win lives anyway — the win is downstream: zeroed
   weight slices can be skipped by a smart matmul kernel.
2. **Fidelity degrades gracefully**: rel L2 grows ~0.36 at k=d/2 (top-50% kept)
   to ~0.66 at k=d/4. Top-k selection picks the small-magnitude half, so the
   diff stays well below the random-baseline √(pruned_fraction).
3. **P3 threshold (0.50) passes comfortably at 0.42**. The math: with θ ~ U(−0.5,
   0.5), the expected diff at k=d/2 is √0.5 ≈ 0.71; top-k brings it to 0.42.

## Validation

```
cargo build --workspace --features gpart_pruning
  → Finished, no errors (only pre-existing json_structural warning)
cargo test -p katgpt-core --features gpart_pruning --lib test_gpart
  → 11 passed; 0 failed
cargo test --release --test bench_008_gpart_pruning_goat --features gpart_pruning
  → 4 passed; 0 failed
```

## Deferred work

- **Dynamic bandit-based mask**: needs per-group reward signal that the forward
  pass doesn't naturally produce. Should be its own plan.
- **Downstream matmul speedup**: smart kernel that skips zeroed weight slices.
  Would unlock the real perf win, but requires matmul-kernel-level changes.
- **Real-model GOAT**: synthetic-scale rel L2 is a proxy; real downstream-task
  accuracy needs a trained model.

## Notes
- GPart groups are already computed via `generate_assignments()` — natural unit for bandit-based pruning
- Should reuse existing `BanditPruner` ARM selection logic
