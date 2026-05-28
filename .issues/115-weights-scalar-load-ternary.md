# Issue 116: `load_ternary_bits` Uses Per-Element Scalar Read Loop

## Severity: Medium
## Files: `katgpt-rs/src/weights.rs` (L269-286)

## Problem

`load_ternary_bits` reads `row_scale`, `pos_bits`, and `neg_bits` element-by-element:

```rust
for val in row_scale.iter_mut() {
    *val = f32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]);
    off += 4;
}
```

For large files with millions of elements, this scalar loop is significantly slower than a bulk copy or SIMD-friendly parse.

## Fix

Use bulk `copy_from_slice` after validating buffer size:

```rust
// For f32 row_scale:
let byte_len = rows * 4;
row_scale.copy_from_slice(bytemuck::cast_slice(&buf[off..off + byte_len]));
off += byte_len;

// For u64 pos_bits/neg_bits:
let byte_len = pos_count * 8;
pos_bits.copy_from_slice(bytemuck::cast_slice(&buf[off..off + byte_len]));
off += byte_len;
```

Or without bytemuck, use `chunks_exact(4)` with `f32::from_le_bytes` via `collect()` which is still faster than individual reads due to iterator optimizations.
