# attention_head Pass 3: cache-hostile loop order

## Status
✅ **DONE** — loop order swapped to `t` outer, `d` inner for contiguous value_cache access

## Severity
🔴 HIGH — called once per head per layer per token

## Location
`src/transformer.rs:542-556` (`attention_head` Pass 3)

## Problem
The value accumulation loop iterates `d` (head_dim) in the outer loop and `t` (sequence positions) in the inner loop. Each `value_cache` access strides by `kv_dim` elements — a **scatter access pattern** that defeats hardware prefetching and causes L1 cache thrashing.

### Current
```rust
// Pass 3: normalize + weighted value accumulation
let inv_sum = 1.0 / sum;
for d in 0..hd {                    // outer: head_dim (64-128)
    let mut val = 0.0f32;
    for t in 0..t_n {               // inner: sequence length
        val += scores[t] * inv_sum
            * value_cache[t * kv_dim + kv_group_offset + d];  // stride = kv_dim ❌
    }
    attn_out[q_head_offset + d] = val;
}
```

**Cache behavior**: For `kv_dim=128` and 128 cache lines, each `t` iteration touches a different cache line. With `t_n > 16`, this spills L1 and causes repeated loads from L2.

## Proposed Fix
Swap loop order: `t` outer, `d` inner. This gives **contiguous row access** on `value_cache`. Pre-scale `scores[t] * inv_sum` once.

```rust
// Pre-scale scores once
for t in 0..t_n {
    unsafe { *scores_buf.get_unchecked_mut(t) *= inv_sum; }
}

// Accumulate: t outer → contiguous value_cache row access
for t in 0..t_n {
    let s = unsafe { *scores_buf.get_unchecked(t) };
    let v_row = unsafe {
        std::slice::from_raw_parts(
            value_cache.as_ptr().add(t * kv_dim + kv_group_offset),
            hd,
        )
    };
    for d in 0..hd {
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) += s * v_row[d];
        }
    }
}
```

Better yet, use `simd_fma_row` or a small SIMD kernel for the inner `d` loop.

## Estimated Impact
- **1.3-2× faster** attention for sequences ≥ 32 tokens
- Dramatically fewer L1 misses on `value_cache`
- Benefit scales with sequence length (worse cache behavior → bigger win)

## Acceptance Criteria
- [x] Loop order swapped to `t` outer, `d` inner
- [x] `attn_out` zero-initialized before accumulation (already done by caller)
- [x] All attention tests pass unchanged
- [x] Benchmark: measure attention time at seq_len=64, 128, 256, 512
