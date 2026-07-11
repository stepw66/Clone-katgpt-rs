# Plan 003: StillKV Heuristic β Optimization

**Date:** 2026-06-11
**Status:** Done
**Issue:** 003 (still-KV heuristic β optimization) — issue closed + removed (done).

## Tasks
- [x] T1: Add `with_beta_strategy()` builder to `IterativeChunkCompactor`
- [x] T2: Add β benchmark test (T25): compare β-A vs β-D on synthetic KV
- [x] T3: Add attention distribution verification: no single latent >50%, entropy < max × 0.8
- [x] T4: Update `run_compaction` helper to accept beta strategy
- [x] T5: Update issue acceptance criteria checkboxes
- [x] T6: Verify all tests pass (80 passed, 1 pre-existing GOAT failure)

## Changes
- `src/still_kv/beta_bias.rs`: Added `AttentionDistribution` struct with `from_cross_attn`, `is_non_degenerate`, `is_not_collapsed`
- `src/still_kv/iterative.rs`: Added `with_beta_strategy()` builder
- `src/still_kv/mod.rs`: Added `run_compaction_with_beta`, T25 benchmark, T26/T27 verification tests
- `src/speculative/dd_tree.rs`: Fixed `u8` vs `usize` comparison
- `crates/katgpt-core/src/slod.rs`: Fixed extra `}` and `usize` vs `f32` type mismatch
