# GPart Partition Pruning — BanditPruner Integration

**Source**: Plan 257 (GPart Isometric Adapter) — Deferred Idea 2
**Priority**: Medium
**Blocked**: No
**Depends**: Plan 257 (complete), BanditPruner infrastructure

## Summary
Integrate GPart partition groups with the existing BanditPruner to enable per-group pruning during inference. Groups with low activation contribution can be zeroed out, trading quality for speed.

## Acceptance Criteria
- [ ] Define pruning threshold for partition groups
- [ ] Extend `GpartAdapter::apply_with_scratch()` to accept a group mask
- [ ] Benchmark quality degradation vs. speedup for top-k group pruning
- [ ] GOAT gate behind `gpart_pruning` feature flag

## Notes
- GPart groups are already computed via `generate_assignments()` — natural unit for bandit-based pruning
- Should reuse existing `BanditPruner` ARM selection logic
