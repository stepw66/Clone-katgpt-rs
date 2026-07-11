# Plan 393 — speculative Phase 8: SpeculativeContext + forward_decode_stage substrate extraction

Status: **DONE** (2026-07-05). All builds clean, 859/859 root lib tests pass
(exact parity with Plan 392 baseline), 12/12 katgpt-forward tests pass,
10/10 bench_102 GOAT tests pass.
Branch: `develop`.
Predecessor: Plan 392 (Phase 7 dd_tree wrappers extraction — CLOSED `f7d1c853`).

## Problem

Plan 392's summary identified the **forward-cycle cluster** (`dflash`,
`verifier`, `drafter_lora`, `step`, `prefill` — ~6K LOC across 5 files) as the
highest-value remaining speculative consolidation target, but noted it was
blocked on "root-only types" (`SpeculativeContext`) and `forward_decode_stage`.

**Audit reveals both blockers are now dissolvable:**

1. **`SpeculativeContext`** (root `src/speculative/types.rs:100-236`, ~137 LOC)
   is **fully leaf-dependent**. Its fields are `ForwardContext`
   (katgpt-forward), `MultiLayerKVCache` (katgpt-transformer), `Config`
   (katgpt-types/core), `SdeConfig` (katgpt-core). The katgpt-core doc comment
   that said it "needs katgpt-transformer" is **stale** — katgpt-transformer is
   a leaf now. The struct + impl can move to the composition layer
   (katgpt-forward), which already hosts `ForwardContext`.

2. **`forward_decode_stage`** + `forward_draft` + `forward_verify`
   (root `src/transformer.rs:222-308`, ~87 LOC) are **fully leaf-dependent**.
   They dispatch to `forward_base` (already in katgpt-forward since Plan 385)
   and use `DecodeStage` (katgpt-transformer). They can move to katgpt-forward's
   `forward.rs`.

**Dependency-layering constraint (the cycle guard):**

`katgpt-forward` already depends on `katgpt-speculative` (for the `DflashCtx`
impl that travels with `ForwardContext`). Therefore `SpeculativeContext` CANNOT
live in `katgpt-speculative` — it would create
`katgpt-forward → katgpt-speculative → katgpt-forward`. The natural home is
**katgpt-forward** (the composition layer above katgpt-speculative).

## Why this is the right increment

This is the **linchpin unblock**. Once `SpeculativeContext` and
`forward_decode_stage` move to katgpt-forward, the 5 forward-cycle files
(`dflash`, `verifier`, `drafter_lora`, `step`, `prefill`) become movable to
katgpt-forward in Plan 394 — every `crate::speculative::types::SpeculativeContext`
and `crate::transformer::forward_decode_stage` reference will resolve through the
composition layer.

Same cadence as Plans 391→392: substrate first (Plan 393), wrappers next
(Plan 394).

## Strategy: two-phase increment

### Phase 1 — Move `SpeculativeContext` to katgpt-forward

- [x] T1.1 Audit `SpeculativeContext` struct + impl for any non-leaf deps
      (expected: none — ForwardContext, MultiLayerKVCache, Config, SdeConfig
      all leaf-resident)
- [x] T1.2 Create `crates/katgpt-forward/src/speculative_context.rs` with the
      moved struct + impl.
- [x] T1.3 Declare `pub mod speculative_context;` +
      `pub use speculative_context::SpeculativeContext;` in katgpt-forward lib.rs
- [x] T1.4 Root `src/speculative/types.rs`: deleted struct + impl (lines
      88-236), replaced with `pub use katgpt_forward::SpeculativeContext;`
- [x] T1.5 All `crate::speculative::types::SpeculativeContext` +
      `crate::speculative::SpeculativeContext` paths resolve via re-export chain
- [x] T1.6 `cargo check --workspace` clean (removed now-unused
      `MultiLayerKVCache` import from root types.rs — only `SpeculativeContext`
      used it; DDTreeBranchCache uses PagedKVCache)

### Phase 2 — Move `forward_decode_stage` to katgpt-forward

- [x] T2.1 Audit confirmed all 3 functions dispatch to `forward_base` (already
      in katgpt-forward::forward since Plan 385)
- [x] T2.2 Moved 3 functions to `crates/katgpt-forward/src/forward.rs` (appended
      at end, gated `decode_specialize`). Added `DecodeStage` import
      (gated to avoid unused warning under default features).
- [x] T2.3 Added `decode_specialize` feature to katgpt-forward Cargo.toml
      (forwards to katgpt-transformer + katgpt-pruners, matching root)
- [x] T2.4 Root `src/transformer.rs`: deleted 3 functions (~94 LOC), replaced
      with `#[cfg(feature = "decode_specialize")] pub use
      katgpt_forward::forward::forward_decode_stage;`
- [x] T2.5 Root Cargo.toml: added `"katgpt-forward/decode_specialize"` to root's
      `decode_specialize` feature forwarding
- [x] T2.6 `cargo check --workspace` clean

### Phase 3 — Verification

- [x] T3.1 `cargo check --workspace` (default) clean — NO warnings
- [x] T3.2 `cargo check --workspace --all-features` clean — NO warnings
- [x] T3.3 `cargo check --workspace --no-default-features` clean — NO warnings
- [x] T3.4 `cargo test -p katgpt-forward --lib` — 8/8 PASS (default), 12/12
      PASS (--all-features)
- [x] T3.5 `cargo test -p katgpt-rs --lib speculative::types` — 9/9 PASS
      (DDTreeBranchCache tests stay root — blocked on forward_paged)
- [x] T3.6 `cargo test -p katgpt-rs --lib --all-features` — 859/859 PASS
      (exact parity with Plan 392 baseline)
- [x] T3.7 GOAT test: `cargo test -p katgpt-rs --test bench_102_tilert_pipeline_goat`
      — 10/10 PASS (exercises forward_decode_stage via re-export, including
      the decode_specialize_proofs mod that tests all 5 DecodeStage variants)
- [x] T3.8 Plan file updated with final LOC numbers; all tasks marked `[x]`

## Items that stay root (documented blockers)

| Item | LOC | Blocker | Unblock condition |
|---|---:|---|---|
| `DDTreeBranchCache` | ~70 | Uses `forward_paged` (heavy forward variant, root `transformer.rs:2462`) | `forward_paged` move to katgpt-forward (separate plan) |
| `SelfSpecConfig` (gated `tri_mode`) | ~25 | Uses `crate::speculative::d2f::D2fDecodeConfig` + `diffusion_sampler::DiffusionSampler` (forward-cycle files) | Plan 394 (5-file forward-cycle move) |

## Final LOC

| File | Plan 392 end | Plan 393 end | Delta |
|---|---:|---:|---:|
| `src/speculative/types.rs` | 609 | 467 | -142 (-23%)
| `src/transformer.rs` | 5697 | 5615 | -82 (-1.4%)
| `crates/katgpt-forward/src/speculative_context.rs` | 0 | 165 | +165 (new)
| `crates/katgpt-forward/src/forward.rs` | 1097 | 1206 | +109
| `crates/katgpt-forward/src/lib.rs` | 454 | 464 | +10 |

Net root reduction: **-224 LOC** (matches ~226 estimate). The win is
**architectural**: this dissolves the two blockers that kept the 5-file
forward-cycle cluster (~6K LOC) root-bound.

## Estimated impact

| File | Before | After (actual) | Delta |
|---|---:|---:|---:|
| `src/speculative/types.rs` | 609 | 467 | -142 LOC (-23%). SpeculativeContext struct+impl removed; re-export shim added. |
| `src/transformer.rs` | 5697 | 5615 | -82 LOC (-1.4%). forward_decode_stage + forward_draft + forward_verify removed; re-export shim added. |
| `crates/katgpt-forward/src/speculative_context.rs` | 0 | 165 | +165 LOC (new file: struct + impl). |
| `crates/katgpt-forward/src/forward.rs` | 1097 | 1206 | +109 LOC (3 functions + DecodeStage import). |

## Import rewrite table

| Root pattern | katgpt-forward rewrite |
|---|---|
| `crate::transformer::{ForwardContext, ...}` | `crate::{ForwardContext, ...}` (ForwardContext already in katgpt-forward) |
| `crate::transformer::{MultiLayerKVCache, TransformerWeights}` | `katgpt_transformer::{MultiLayerKVCache, TransformerWeights}` |
| `crate::types::Config` | `katgpt_types::Config` |
| `super::types::SdeConfig` (from speculative/types.rs) | `katgpt_core::speculative::types::SdeConfig` |
| `crate::transformer::forward_base` (in forward_decode_stage) | `crate::forward::forward_base` (same crate, forward submodule) |

## GOAT Gate

- G1 correctness: `cargo test` parity (no test count change for moved substrate;
  the DDTreeBranchCache tests stay root, the forward_decode_stage test in
  bench_102 runs via re-export)
- G2 perf: identical code paths, just relocated
- G3 no-regression: workspace builds clean at default / all-features / no-default
- G4 alloc-free: not applicable (substrate relocation only)

## Risks

- `SpeculativeContext` is widely used (benchmarks, examples, tests, leaf).
  The root re-export (`pub use katgpt_forward::SpeculativeContext`) preserves
  all `crate::speculative::types::SpeculativeContext` +
  `crate::speculative::SpeculativeContext` paths, so call sites need no changes.
- `forward_decode_stage` has a GOAT test (`bench_102_tilert_pipeline_goat.rs`)
  that exercises all 5 `DecodeStage` variants. Must pass unchanged via re-export.
- Feature gate `decode_specialize` must be added to katgpt-forward Cargo.toml
  and forwarded from root, otherwise `forward_draft`/`forward_verify` won't
  compile under the gate.

## Next session (Plan 394 candidate)

With Plan 393 done, the 5 forward-cycle files (`dflash`, `verifier`,
`drafter_lora`, `step`, `prefill`) become movable to katgpt-forward. They form
an internal cycle that dissolves within katgpt-forward (referencing each other
as `crate::dflash::*`, `crate::verifier::*`, etc.). Estimated ~6K LOC move.
