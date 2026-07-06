# Issue 042: Extract Inline Tests from `dd_tree.rs` (4790) + `bandit.rs` (3279) — File-Size Guideline Compliance

**Status:** ✅ RESOLVED
**Date:** 2026-07-07
**Priority:** P3 — Tech debt (file-size guideline violation; no correctness impact)
**Blocked:** No — pure move/refactor
**Depends:** Nothing
**Related:** riir-train Issues 370/371/372 (same `#[path]` extraction pattern),
katgpt-rs AGENTS.md ("Keep files less than 2048 lines for .rs file as possible",
"Keep files smaller than 3200 lines as possible")

## Summary

Two `.rs` files in katgpt-rs exceeded the 3200-line hard cap from AGENTS.md,
with the bloat coming entirely from inline `#[cfg(test)] mod tests { ... }`
blocks at the end of each file. Same class of mechanical refactor as the
resolved riir-train Issues 370/371/372: extract the test module to a sibling
file via `#[cfg(test)] #[path = "<name>_tests.rs"] mod tests;`.

| File | Before | Production | Tests (extracted) | After |
|------|--------|------------|-------------------|-------|
| `crates/katgpt-speculative/src/dd_tree.rs` | 4790 | 4136 | 653 | 4140 |
| `crates/katgpt-pruners/src/bandit.rs` | 3279 | 2104 | 1174 | 2107 |

After extraction:
- `dd_tree.rs` → 4140 lines (production is genuinely large — 14 `build_dd_tree_*`
  variants + tree-walk primitives; further split is a design decision, not
  mechanical). Tests → `dd_tree_tests.rs` (651 lines).
- `bandit.rs` → 2107 lines (just over the 2048 soft guideline; production code
  is cohesive — `BanditPruner` + env types + stats). Tests → `bandit_tests.rs`
  (1172 lines).

## Pattern Applied

Identical to riir-train Issue 371 (`gzero_loop.rs`) and Issue 372
(`loss_nextlat` / `cpu_lora_train` / `optimizer_lora_muon`):

```rust
// At the end of the source file (replacing the inline `mod tests { ... }`):
#[cfg(test)]
#[path = "<name>_tests.rs"]
mod tests;
```

The sibling `<name>_tests.rs` file retains the original test content verbatim
(including the leading `use super::*;`), so private-item access and the module
hierarchy are preserved bit-for-bit. No `#![cfg(test)]` inner attribute is
needed in the sibling — the gate lives on the `mod tests;` declaration.

## Why Now

1. The prior session (2026-07-06) did a full-workspace clippy sweep but did
   NOT audit file sizes. These two files were the only 3200+ line violations
   in katgpt-rs where the bloat was mechanical test code (vs. production
   design bloat like GPU kernel dispatch tables).
2. `dd_tree.rs` at 4790 lines was the **largest `.rs` file in katgpt-rs** and
   the 3rd-largest across the 5-repo quintet (after riir-ai GPU kernels
   `forward.rs` 7894, `gemma2_cubecl.rs` 5354).
3. No active feature work on either file's test module — pure move, zero
   behavior change.

## Verification (G3 no-regression)

- `cargo check -p katgpt-pruners --lib --tests` — 0 errors, 0 warnings.
- `cargo check -p katgpt-speculative --lib --tests` — 0 errors (1 pre-existing
  `unused_imports` warning for `HashMap` under `cfg(any(feature = "elf_sde",
  test))` vs usage under `cfg(feature = "elf_sde")` — NOT caused by this
  extraction; the import's `test` gate predates this refactor).
- `cargo test -p katgpt-pruners --lib --features bandit` — **197 passed**
  (was 197 before; includes all 49 `bandit::tests::*`).
- `cargo test -p katgpt-pruners --lib` (default features) — 126 passed
  (bandit module is `#[cfg(feature = "bandit")]`-gated; unaffected).
- `cargo test -p katgpt-speculative --lib` (default features) — **300 passed**
  (unchanged; includes all 24 `dd_tree::tests::*`).
- `cargo clippy -p katgpt-pruners --features bandit --lib --tests` — 0 warnings.
- `cargo check --workspace --all-features` — clean (no combo regression; the
  `merkle_root` / `can_freeze` lesson class).

## Out of Scope

- **`dd_tree.rs` remaining size (4140 lines)**: the production code itself is
  large (14 `build_dd_tree_*` variants). Further reduction would require
  grouping variants into sub-modules (e.g. `dd_tree/sde.rs`, `dd_tree/lodestar.rs`)
  — a design refactor, not mechanical. Defer until a concrete edit-pain
  trigger (next `build_dd_tree_*` variant addition).
- **Pre-existing `HashMap` cfg mismatch in `dd_tree.rs`**: the import gate
  `#[cfg(any(feature = "elf_sde", test))]` includes `test` but the sole usage
  (`best_of_k_rollouts`) is `#[cfg(feature = "elf_sde")]`-only. This produces
  an `unused_imports` warning in default-feature `--lib --tests` builds.
  Pre-existing (not caused by this extraction); left untouched to keep this
  commit purely mechanical.
- **riir-ai GPU kernel files** (`forward.rs` 7894, `gemma2_cubecl.rs` 5354,
  `gemma2_forward.rs` 4904): over the cap but are monolithic GPU kernel
  dispatch tables — splitting risks breaking fused-op dispatch. Not the same
  class as test-block extraction.
- **riir-train `training_loop.rs` (4047 lines)**: Issue 302, explicitly blocked
  on Plan 296 T7.1-T7.3.

## TL;DR

`dd_tree.rs` (4790→4140) and `bandit.rs` (3279→2107) had inline test modules
extracted to sibling `*_tests.rs` files via the established `#[path]` pattern.
`dd_tree.rs` remains over the 3200 hard cap (production code is genuinely
large — 14 `build_dd_tree_*` variants); the remaining size is a design-split
question, not a mechanical one. Both files' test counts unchanged:
katgpt-speculative 300 passed, katgpt-pruners+bandit 197 passed.
