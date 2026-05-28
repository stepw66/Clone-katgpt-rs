# Issue 118: `peira_aux_loss` Uses O(k³) Triple-Nested Loop Instead of `matmul_into`

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/peira.rs` (L874-886)

## Problem

The triple loop computes `P* @ M` where `M = N_sample + λI`. The inner loop has a branch `if l == j` for adding λ to the diagonal, which prevents auto-vectorization:

```rust
for i in 0..k {
    for j in 0..k {
        for l in 0..k {
            let m_val = n_sample[l * k + j] + if l == j { lambda } else { 0.0 };
            result[i * k + j] += p_star[i * k + l] * m_val;
        }
    }
}
```

## Fix

Build M = N_sample + λI into the scratch buffer first (branch-free), then call the existing `matmul_into`:

```rust
// Build M = N_sample + λI (branch-free diagonal add)
pm[..k*k].copy_from_slice(n_sample);
for i in 0..k {
    pm[i * k + i] += lambda;
}
// P*M = matmul(p_star, pm) — uses SIMD-optimized matmul
matmul_into(&mut result, bt_scratch, p_star, &pm[..k*k], k);
```
