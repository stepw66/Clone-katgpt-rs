# Issue 102: tiled_attention_batched Uses Rayon Unconditionally for Tiny Workloads

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/attention.rs` (L335-358)

## Description
`tiled_attention_batched` uses `par_chunks_mut` for all batched attention, even when `total = batch × heads` is small (e.g., 1 batch × 1 head). Rayon's task scheduling overhead (~1-10μs per task) can exceed computation time for small seq_len.

Per optimization.md: "Threshold: rayon wins only at m ≥ 64 with μs/row work, or m ≥ 1000 with ns/row work." and "Benchmark serial vs parallel at actual workload size."

## Fix
Add a threshold check before parallelizing:
```rust
if total <= 2 || seq_len * head_dim < 1024 {
    // Sequential fallback for tiny workloads
    for idx in 0..total { ... }
} else {
    output.par_chunks_mut(head_size).enumerate().for_each(...);
}
```

## Impact
Medium — for single-token decode (common in autoregressive generation), this avoids thread pool overhead per attention computation.
