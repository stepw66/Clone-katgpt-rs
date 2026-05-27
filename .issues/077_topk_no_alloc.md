# select_topk_indices: eliminate per-call Vec allocation

## Status
✅ **DONE** — `select_topk_indices_into_buf` added; `clustered_lm_head` uses it

## Severity
🟡 MEDIUM — called every token when clustered LM head is active

## Location
`src/transformer.rs:1240-1257` (`select_topk_indices`)

## Problem
`select_topk_indices` allocates a `Vec<usize>` return value every call:
```rust
indexed[..k].iter().map(|(i, _)| *i).collect() // heap allocation
```

When the clustered LM head is active, this is called per forward pass (per token during generation). The allocation + deallocation adds GC pressure and cache pollution.

There's already `select_topk_indices_into` that reuses the indexed buffer, but it still returns a new `Vec<usize>`.

## Proposed Fix

### Option A: write into pre-allocated output buffer
Add an `into` variant that writes indices into a caller-provided buffer:
```rust
pub fn select_topk_indices_into_buf(
    scores: &[f32],
    k: usize,
    indexed_buf: &mut Vec<(usize, f32)>,
    output_buf: &mut Vec<usize>,
) {
    // ... partial sort into indexed_buf ...
    output_buf.clear();
    output_buf.extend(indexed_buf[..k].iter().map(|(i, _)| *i));
}
```

### Option B: iterator-based (zero allocation)
Return an iterator over the top-K indices without collecting:
```rust
pub fn select_topk_iter<'a>(
    indexed_buf: &'a [(usize, f32)],
    k: usize,
) -> impl Iterator<Item = usize> + 'a {
    indexed_buf[..k].iter().map(|(i, _)| *i)
}
```

Then the clustered_lm_head loop would consume the iterator directly.

## Estimated Impact
- Eliminates **1 allocation per token** during generation with clustered LM head
- Minor but consistent latency improvement

## Acceptance Criteria
- [x] `clustered_lm_head` doesn't allocate per call for index selection
- [x] Top-K selection reuses pre-allocated buffers
- [x] All clustered LM head tests pass
