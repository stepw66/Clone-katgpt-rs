# Issue 100: Newton-Schulz Allocates Temp Buffers Per Call + Scalar Transpose

## Severity: Medium-High
## Files: `katgpt-rs/src/newton_schulz.rs`

## Description

### Per-call allocations (L113-125)
`newton_schulz5_square()` allocates 6 `Vec`s (`x`, `a_mat`, `b_mat`, `bx`, `at`, `xt_buf`) on every call. For training workloads where this is called per gradient update, these allocations add up significantly.

For a 512×512 matrix: `2*m*n + 4*m*m` floats ≈ 5MB per call. At 1000 calls/step, that's 5GB of allocation traffic.

### Scalar transpose with poor cache locality (L23-29)
The `transpose()` function copies one element at a time with strided writes. For matrices larger than L1 cache, this causes cache thrashing. Called ~15 times per `newton_schulz5` call (5 iterations × 3 transposes).

## Fix
1. Create a `NewtonSchulzScratch` struct owning the 6 buffers, or accept pre-allocated scratch buffers as parameters.
2. Implement tiled transpose (8×8 or 16×16 tiles) that fits in L1 cache, or use in-place transpose for square matrices.

## Impact
Medium-High — eliminates ~5MB allocation per call and improves transpose performance 2-4× for typical matrix sizes.
