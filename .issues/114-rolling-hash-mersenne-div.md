# Issue 115: `mersenne_mul` Uses u128 Division Instead of Fast Mersenne Reduction

## Severity: Medium
## Files: `katgpt-rs/src/cache_prune/rolling_hash.rs` (L123)

## Problem

`mersenne_mul` uses `((a as u128) * (b as u128) % (m as u128)) as u64`. The comment acknowledges this. For the Mersenne prime 2^61−1, the canonical fast reduction avoids the expensive 128-bit modulo and compiles to 2-3 instructions instead of a division:

```rust
fn mersenne_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128) * (b as u128) % (m as u128)) as u64  // slow: 128-bit modulo
}
```

## Fix

Use fast Mersenne reduction for 2^61−1:

```rust
fn mersenne_mul(a: u64, b: u64, m: u64) -> u64 {
    let prod = (a as u128) * (b as u128);
    let lo = (prod & ((1u128 << 61) - 1)) as u64;
    let hi = (prod >> 61) as u64;
    let sum = lo + hi;
    if sum >= m { sum - m } else { sum }
}
```

This replaces a 128-bit modulo (which may compile to a software division) with 2 shifts + 1 add + 1 conditional subtract.
