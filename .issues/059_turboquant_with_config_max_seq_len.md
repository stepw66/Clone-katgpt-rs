# Issue 059: TurboQuant `with_config` Uses `kv_dim` as `max_seq_len`

> **Status**: Open
> **Severity**: Bug (data corruption / silent failure)
> **Affected**: `src/turboquant/kv_cache.rs`, `src/turboquant/types.rs`

## Summary

`TurboQuantKVCache::with_config` incorrectly uses `tq_config.kv_dim` everywhere `max_seq_len` should be used. The `TurboQuantKVCacheConfig` struct is also missing a `max_seq_len` field entirely.

## Root Cause

1. `TurboQuantKVCacheConfig` (in `types.rs`) has no `max_seq_len` field.
2. `with_config` (in `kv_cache.rs`) uses `tq_config.kv_dim` for:
   - `key_indices` inner dimension (positions) — should be `max_seq_len`
   - `key_norms` inner dimension (positions) — should be `max_seq_len`
   - `val_indices` inner dimension (positions) — should be `max_seq_len`
   - `val_norms` inner dimension (positions) — should be `max_seq_len`
   - `self.max_seq_len` field — should be actual max sequence length

Compare with `new()` which correctly uses `config.block_size` (the max sequence length) for all position-dimension allocations.

## Impact

When `kv_dim ≠ max_seq_len` (e.g., `kv_dim=64`, `max_seq_len=256`):
- If `kv_dim < max_seq_len`: cache is too small → out-of-bounds access / panic on positions beyond `kv_dim`
- If `kv_dim > max_seq_len`: cache is oversized → wasted memory, but functionally correct
- `reset()` iterates `0..self.max_seq_len` → iterates wrong range after `with_config`
- `compression_ratio()` and `bytes_per_token()` use `self.max_seq_len` indirectly — unaffected but semantically wrong

## Evidence

### `new()` (correct):
```rust
// kv_cache.rs L77-79
key_indices: vec![vec![vec![0u8; packed_key_len]; max_seq_len]; n_layers],
key_norms:   vec![vec![0.0f32; max_seq_len]; n_layers],
// ...
max_seq_len,
```

### `with_config()` (broken):
```rust
// kv_cache.rs L112-119
key_indices: vec![vec![vec![0u8; packed_key_len]; tq_config.kv_dim]; tq_config.n_layers],
key_norms:   vec![vec![0.0f32; tq_config.kv_dim]; tq_config.n_layers],
// ...
max_seq_len: tq_config.kv_dim,  // ← WRONG
```

## Fix

1. Add `max_seq_len: usize` to `TurboQuantKVCacheConfig` in `types.rs`
2. Replace all `tq_config.kv_dim` (used as position dimension) with `tq_config.max_seq_len` in `with_config()`
3. Add a test that creates cache via `with_config` with `max_seq_len != kv_dim` and verifies dimensions

## Tasks

- [x] Add `max_seq_len` field to `TurboQuantKVCacheConfig`
- [x] Fix `with_config` to use `tq_config.max_seq_len` for position-dimension allocations
- [x] Add regression test for `with_config` with `max_seq_len != kv_dim`
