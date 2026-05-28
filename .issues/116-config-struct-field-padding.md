# Issue 117: `Config` Struct Field Ordering Causes Suboptimal Padding

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/types.rs` (L395-480)

## Problem

The `Config` struct has fields interspersed between `usize`, `f32`, `bool`, enums, and `Vec<usize>`. While a comment claims "ordered by descending alignment to minimize padding", several `f32` fields are placed between `usize` fields and `Vec` fields, causing padding gaps. On 64-bit platforms, each `bool` or `u8` followed by a `usize` or `f64` wastes 7 bytes of padding.

## Fix

Reorder fields to group by alignment: `usize`/`u64` first, then `f32`, then `bool`/`u8`, then `Vec`/pointer fields:

```rust
pub struct Config {
    // 8-byte aligned fields
    pub vocab_size: usize,
    pub block_size: usize,
    // ... all other usize fields ...

    // 4-byte aligned fields
    pub temperature: f32,
    pub lora_alpha: f32,
    // ... all other f32 fields ...

    // 1-byte aligned fields
    pub tied_embeddings: bool,
    pub use_rope: bool,
    // ... all other bool/u8 fields ...

    // Pointer fields (8-byte aligned)
    pub lora_targets: Vec<usize>,
    pub mls_layers: Vec<usize>,
}
```
