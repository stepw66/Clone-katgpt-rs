# Issue 042 — KVarN fork-drift: riir-gpu/src/kvarn/ duplicates katgpt-kv/src/kvarn/ (slower)

## Status: RESOLVED (2026-07-10)
## Priority: P2 (DRY violation + perf regression in fork)
## Discovered: 2026-07-10 (quarterly GOAT cherry-pick audit follow-up)

## Problem

`riir-ai/crates/riir-gpu/src/kvarn/` is a fork of `katgpt-rs/crates/katgpt-kv/src/kvarn/`
that has **drifted behind** the canonical version. The fork duplicates three substrate
files and the RTN helper functions, all of which are slower than the canonical copies:

### Duplicated + drifted-behind files

| File | katgpt-kv (canonical) | riir-gpu (fork) | Drift |
|---|---|---|---|
| `hadamard.rs` | Unsafe ptr arithmetic for LLVM auto-vec | Safe indexing (older) | **Missing auto-vec optimization** |
| `var_norm.rs` | Pre-allocates scratch ONCE, delegates to `variance_normalize_into` (zero-alloc) | Allocates 6 Vecs per Sinkhorn iteration (48 allocs/call for 8 iters) | **48→6 allocations per call** |
| RTN helpers (`packed_bytes_per_row`, `rtn_quantize_rows`, `pack_value`, `unpack_value`) | `mul_add(inv_scale, bias)` hot loop | `(v - lo) / scale` (division per element) | **Missing mul_add optimization** |

### Dead code in riir-gpu (riir-engine has its own copies)

| File | riir-gpu | riir-engine | Status |
|---|---|---|---|
| `kv_quality.rs` | `KvCacheQualityReport`, `estimate_kv_quality` | `kvarn_quality.rs` — same structs + more | **Dead code** — mod.rs line 37 says "no external consumer" |
| `kvarn_tier.rs` | `MemoryTier`, `KvarnTierManager` | `kvarn_tier.rs` — same structs + more | **Dead code** — riir-games has its own `MemoryTier` |

### NOT duplicated (legitimate riir-gpu/game-specific code)

| File | Status |
|---|---|
| `kvarn_zone_weights.rs` | Game-specific zone hot-swap scale registry. Only in riir-gpu. Only consumed by `kvarn-npc` example. |
| `kvarn_quant.rs` (tile-level API) | Different abstraction than katgpt-kv's `KVarNKVCache`. riir-gpu exposes per-tile `kvarn_quantize_key_tile` / `kvarn_quantize_value_tile`. |
| `kvarn_dequant.rs` | Tile-level dequant API. riir-gpu-specific. |
| `kvarn_dispatch.rs` | CPU/GPU auto-route. riir-gpu-specific. |

### Root cause

The RTN helpers (`rtn_quantize_rows`, `pack_value`, `unpack_value`, `packed_bytes_per_row`)
are **private** (`fn`, not `pub fn`) in `katgpt-kv/src/kvarn/kv_cache.rs`. So riir-gpu
cannot `use katgpt_kv::kvarn::...` to access them — the fork happened because the helpers
were not exported.

### Consumer analysis

- `riir_gpu::kvarn::*` is consumed only by `riir-examples/examples/kvarn_npc_thinking.rs`
- `riir-engine` already has `kvarn = ["dep:katgpt-kv", "katgpt-kv/kvarn"]` feature
- `riir-gpu` does NOT depend on `katgpt-kv` (no cycle: riir-gpu → riir-engine, not reverse)

## Fix

### Phase 1: Export RTN helpers from katgpt-kv (katgpt-rs side)

Make the following `fn` → `pub fn` in `katgpt-kv/src/kvarn/kv_cache.rs`:
- `packed_bytes_per_row` [L972]
- `rtn_quantize_rows` [L980]
- `rtn_quantize_rows_grouped` [L1037]
- `pack_value` [L1095]
- `unpack_value` [L1116]
- `unpack_row` [L1141]

Re-export them from `katgpt-kv/src/kvarn/mod.rs`:
```rust
pub use kv_cache::{
    packed_bytes_per_row, rtn_quantize_rows, rtn_quantize_rows_grouped,
    pack_value, unpack_value, unpack_row,
};
```

Also export `VarNormConfig` from mod.rs (currently only `VarianceNormScales` and
`variance_normalize` are re-exported; `VarNormConfig` is pub in the module but not
re-exported at the mod level).

### Phase 2: Consume katgpt-kv from riir-gpu (riir-ai side)

1. Add `katgpt-kv` as optional dep in `riir-gpu/Cargo.toml`:
   ```toml
   katgpt-kv = { path = "../../../katgpt-rs/crates/katgpt-kv", default-features = false, optional = true }
   ```

2. Add `kvarn` feature to `riir-gpu/Cargo.toml` that pulls `katgpt-kv/kvarn`:
   ```toml
   kvarn = ["dep:katgpt-kv", "katgpt-kv/kvarn"]
   ```

3. Replace `riir-gpu/src/kvarn/hadamard.rs` with:
   ```rust
   pub use katgpt_kv::kvarn::hadamard::{hadamard_transform_inplace, hadamard_rows, hadamard_cols, hadamard_cols_into};
   ```

4. Replace `riir-gpu/src/kvarn/var_norm.rs` with:
   ```rust
   pub use katgpt_kv::kvarn::var_norm::{VarianceNormScales, VarNormConfig, variance_normalize, variance_normalize_into};
   ```

5. In `riir-gpu/src/kvarn/kvarn_quant.rs`, replace local RTN helpers with:
   ```rust
   use katgpt_kv::kvarn::{packed_bytes_per_row, rtn_quantize_rows, pack_value, unpack_value};
   ```
   Remove the local copies of `packed_bytes_per_row`, `rtn_quantize_rows`, `pack_value`,
   `unpack_value`. Keep `KvarnQuantParams`, `KvarnTileResult`, and the tile-level API
   functions (`kvarn_quantize_key_tile`, `kvarn_quantize_value_tile`, `kvarn_quantize_cpu`).

6. In `riir-gpu/src/kvarn/kvarn_dequant.rs`, replace `use super::kvarn_quant::unpack_value`
   with `use katgpt_kv::kvarn::unpack_value`. Keep the tile-level dequant API.

### Phase 3: Delete dead code

- Delete `riir-gpu/src/kvarn/kv_quality.rs` (riir-engine has `kvarn_quality.rs`)
- Delete `riir-gpu/src/kvarn/kvarn_tier.rs` (riir-engine has `kvarn_tier.rs`)
- Remove their `mod` declarations from `riir-gpu/src/kvarn/mod.rs`

### Phase 4: Verify

- `cargo check -p katgpt-kv --features kvarn` (katgpt-rs)
- `cargo check -p riir-gpu --features kvarn` (riir-ai)
- `cargo check -p riir-gpu --features kvarn_game` (riir-ai)
- `cargo test -p riir-gpu --features kvarn` (riir-ai)
- `cargo test -p katgpt-kv --features kvarn` (katgpt-rs)
- `cargo run --example kvarn_npc_thinking` (riir-ai, if feasible)

## Perf gain (expected)

- Hadamard: unsafe ptr → LLVM auto-vec (measurable on large tiles)
- VarN: 48→6 allocations per `variance_normalize` call (significant for 128×128 tiles)
- RTN: division → mul_add per element in hot loop

## Risk

- **Low.** The tile-level API signatures don't change. Only the internal helper
  implementations swap from local-copy to katgpt-kv-canonical.
- The `kvarn_game` feature gates `kvarn_zone_weights.rs` which calls
  `kvarn_quantize_cpu` — that function's signature is unchanged.
- `VarNormConfig` fields are identical between the two copies (verified).
- `VarianceNormScales` fields are identical (verified).
