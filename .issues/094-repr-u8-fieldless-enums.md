# Issue 094: Missing `#[repr(u8)]` on Field-less Enums

## Severity: Low
## Files: `katgpt-rs/crates/katgpt-core/src/simd.rs`, `katgpt-rs/crates/katgpt-core/src/types.rs`, `katgpt-rs/crates/katgpt-core/src/traits.rs`, `katgpt-rs/src/transformer.rs`

## Description
Multiple field-less enums across the codebase lack `#[repr(u8)]`, defaulting to `usize`-sized discriminants (8 bytes on 64-bit). Per optimization.md guideline, all field-less enums should use `#[repr(u8)]` for guaranteed 1-byte size.

## Affected Enums
- `SimdLevel` (simd.rs L19-26) — 3 variants, used as runtime dispatch key
- `HlaMode` (types.rs L15-22) — 3 variants, stored in Config
- `ActingMode` (traits.rs L509-523) — 5 variants
- `IterationMode` (types.rs L2288-2294) — 2 variants
- `CacheStrategy` (types.rs L2300-2306) — 2 variants
- `DecodeStage` (transformer.rs L12-21) — 4 variants, used in hot decode path

## Fix
Add `#[repr(u8)]` to each enum definition.

## Impact
Low — improves data density and eliminates zero-extension in match dispatch. Marginal but follows established patterns.
