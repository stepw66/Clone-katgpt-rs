# Issue 096: PagedKVCache Hot-Path Allocations in ensure_pages() and rollback()

## Severity: Medium
## Files: `katgpt-rs/src/transformer.rs`

## Description
Two methods on `PagedKVCache` allocate on the heap on every call, which happens per `forward_paged()` invocation and per speculative decoding rejection.

### `ensure_pages()` (L3437)
Allocates a `Vec<usize>` for `all_new` on every call. Should be a pre-allocated scratch buffer on `PagedKVCache`.

### `rollback()` (L3521, L3538)
Allocates a `HashSet<usize>` and per-layer `Vec<usize>` on every rollback. For frequent speculative decoding rollbacks, this is avoidable allocation.

Per optimization.md: "Cache allocations: Vec::with_capacity() once, clear() + reuse across calls"

## Fix
1. Add `all_new_buf: Vec<usize>` field to `PagedKVCache`, pre-allocate in `new()`, use `clear()` + reuse in `ensure_pages()`.
2. Add `rollback_set: HashSet<usize>` and `rollback_removed: Vec<usize>` fields, pre-allocate and reuse in `rollback()`.

## Impact
Medium — eliminates 2-3 heap allocations per forward pass in paged attention and speculative decoding paths. For batched inference with many sequences, this compounds.
