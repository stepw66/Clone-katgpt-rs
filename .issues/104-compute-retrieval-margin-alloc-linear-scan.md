# Issue 104: `compute_retrieval_margin` Allocates Per Query + O(k) Linear Scan Per Doc

## Severity: High
## Files: `katgpt-rs/crates/katgpt-core/src/simd.rs` (L2384-2414)

## Problem

Two compounding issues:

1. **Per-query heap allocation**: `Vec<usize>` built on every outer loop iteration:
   ```rust
   let pos_set: Vec<usize> = (pos_start..pos_start + k)
       .map(|p| neighborhoods[p])
       .collect();  // ← allocates per query
   ```

2. **O(k) linear scan per document**: `pos_set.contains(&j)` is O(k) inside the O(n_docs) inner loop:
   ```rust
   for j in 0..n_docs {
       if pos_set.contains(&j) {  // ← O(k) linear scan per doc
           continue;
       }
   ```
   Total: O(n_queries × n_docs × k) membership tests.

## Fix

Pre-allocate a bitmap outside the loop, reuse with `fill(false)` per query:

```rust
let mut pos_bitmap = vec![false; n_docs];  // allocate once
for i in 0..n_queries {
    pos_bitmap.fill(false);  // reset: ~2ns for typical n_docs
    let pos_start = i * k;
    for p in pos_start..pos_start + k {
        let idx = neighborhoods[p];
        if idx < n_docs { pos_bitmap[idx] = true; }
    }
    // pos_bitmap[j] is now O(1) lookup instead of O(k) linear scan
}
```

For very large n_docs, consider a `HashSet` or sorted vec + binary search instead of a dense bitmap.
