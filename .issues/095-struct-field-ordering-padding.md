# Issue 095: Struct Field Ordering Causes Padding Waste

## Severity: Low
## Files: `katgpt-rs/crates/katgpt-core/src/types.rs`

## Description
`RtTurboConfig` fields alternate between `f32` and `usize`, causing 8 bytes of padding per instance (48 bytes vs 40 bytes optimal). Per optimization.md, fields should be grouped by alignment (u64/usize → u32/f32 → u8/bool).

## Current Layout (48 bytes)
```
retrieval_head_ratio: f32   // 4 bytes + 4 padding
low_dim: usize              // 8 bytes
top_p: f32                  // 4 bytes + 4 padding
sliding_window: usize       // 8 bytes
sink_tokens: usize          // 8 bytes
block_size: usize           // 8 bytes
```

## Optimal Layout (40 bytes)
```
low_dim: usize              // 8 bytes
sliding_window: usize       // 8 bytes
sink_tokens: usize          // 8 bytes
block_size: usize           // 8 bytes
retrieval_head_ratio: f32   // 4 bytes
top_p: f32                  // 4 bytes
```

## Fix
Reorder `RtTurboConfig` fields: group `usize` first, then `f32`.

## Impact
Low — single instance per model, but struct is `Copy` so it moves frequently. Saves 8 bytes per copy.
