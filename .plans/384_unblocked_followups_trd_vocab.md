# Plan 384 — Unblocked Follow-ups: trd.rs + vocab_channel_pruner.rs

## TL;DR

Two Phase 12 DEFER items whose blockers dissolved during Phase 12 itself:

1. **`src/distill/trd.rs`** (1107 LOC) — blocker was `crate::fold` dep.
   Phase 12 T4.5 moved `fold/` to `katgpt-speculative`. **NOW MOVABLE**
   to `katgpt-speculative/src/distill/trd.rs`.

2. **`src/speculative/vocab_channel_pruner.rs`** (2048 LOC) — blocker was
   `crate::lattice_operad` dep.
   Phase 12 T4.4 moved `lattice_operad/` to `katgpt-pruners`. **NOW MOVABLE**
   to `katgpt-pruners/src/vocab_channel_pruner.rs`.

**Bonus discovery:** vocab_channel_pruner's other "blocker"
(`crate::transformer::TransformerWeights`, `crate::types::Config`) is also
already leaf-resident:
- `TransformerWeights` is in `katgpt-transformer/src/weights.rs`
- `Config` is in `katgpt-types/src/config.rs`

So vocab_channel_pruner is FULLY decoupled, not partially. The original
"transformer-bound" classification was stale.

**Latent bug uncovered:** `katgpt-speculative/src/lib.rs` had
`pub mod distill` gated on `ilc_distill` ALONE. Enabling `trd_refined_draft`
without `ilc_distill` silently cfg'd out the entire distill umbrella — trd
compiled in `cargo check` but produced zero test symbols. Fixed to
`#[cfg(any(...))]`. Root's own `pub mod distill` was already correctly gated.

## Verification (pre-move)

- [x] `crate::fold::{ChainFolder, FoldContext, FoldDecision, StepBoundary}`
      resolves via re-export → real home: `katgpt_speculative::fold::*`
- [x] `crate::lattice_operad::{PrunerExpr, ComposedPruner}` resolves via
      re-export → real home: `katgpt_pruners::lattice_operad::*`
- [x] `crate::transformer::TransformerWeights` is in katgpt-transformer
- [x] `crate::types::Config` is `katgpt_types::Config`
- [x] `katgpt-pruners` Cargo.toml already deps `katgpt-transformer` +
      `katgpt-types` + `bytemuck` + `blake3` + `serde`/`serde_json`
- [x] `katgpt-speculative` Cargo.toml has `chain_fold` feature already
      (gates the `fold` module)
- [x] Root feature gates `trd_refined_draft` + `vocab_channel_pruner` exist
- [x] Root lib.rs re-export paths preserve `katgpt_rs::distill::trd::*` and
      `katgpt_rs::speculative::vocab_channel_pruner::*`

## Tasks

- [x] **T1.** Move `src/distill/trd.rs` → `katgpt-speculative/src/distill/trd.rs`
      - T1.1 `git mv` file ✅
      - T1.2 Imports unchanged — `crate::fold` is now native to katgpt-speculative ✅
      - T1.3 Added `pub mod trd;` to `katgpt-speculative/src/distill/mod.rs`
            under `#[cfg(feature = "trd_refined_draft")]` ✅
      - T1.4 Added `trd_refined_draft = []` + `plasma_path = []` + `gpu = []`
            tracking flags to katgpt-speculative Cargo.toml (the latter two
            silence `unexpected_cfgs` lints on dead-code stubs inside trd.rs) ✅
      - T1.5 Root `src/distill/mod.rs`: replaced `pub mod trd;` with
            `pub use katgpt_speculative::distill::trd;` re-export ✅
      - T1.6 Root Cargo.toml: extended `trd_refined_draft` with
            `katgpt-speculative/trd_refined_draft` ✅

- [x] **T2.** Move `src/speculative/vocab_channel_pruner.rs` →
            `katgpt-pruners/src/vocab_channel_pruner.rs`
      - T2.1 `git mv` file ✅
      - T2.2 Rewrote imports:
            - `crate::lattice_operad::*` stays (native to katgpt-pruners)
            - `crate::transformer::TransformerWeights` → `katgpt_transformer::TransformerWeights`
            - `crate::types::Config` → `katgpt_types::Config` ✅
      - T2.3 Added `pub mod vocab_channel_pruner;` to `katgpt-pruners/src/lib.rs`
            under `#[cfg(feature = "vocab_channel_pruner")]` ✅
      - T2.4 Added `vocab_channel_pruner = []` feature to katgpt-pruners Cargo.toml ✅
      - T2.5 Root `src/speculative/mod.rs`: replaced `pub mod vocab_channel_pruner;`
            with `pub use katgpt_pruners::vocab_channel_pruner;` re-export ✅
      - T2.6 Root Cargo.toml: extended `vocab_channel_pruner` with
            `katgpt-pruners/vocab_channel_pruner` ✅

- [x] **T2.5b (bonus).** Fixed latent `pub mod distill` umbrella gate in
      `katgpt-speculative/src/lib.rs`:
      `#[cfg(feature = "ilc_distill")]` →
      `#[cfg(any(feature = "ilc_distill", feature = "trd_refined_draft"))]`.
      Without this, trd compiled but produced zero test symbols.

- [x] **T3.** GOAT gate G3 — workspace check
      - T3.1 `cargo check --workspace` (default features) clean ✅
      - T3.2 `cargo check --workspace --all-features` clean ✅
      - T3.3 `cargo check --workspace --no-default-features` clean ✅
      - T3.4 `cargo test -p katgpt-speculative --features trd_refined_draft,chain_fold --lib`:
            282/282 PASS (incl. all 12 trd tests) ✅
      - T3.5 `cargo test -p katgpt-pruners --features vocab_channel_pruner --lib`:
            180/180 PASS (incl. all 30 vocab tests) ✅
      - T3.6 Root lib tests: 769/769 PASS (12 fewer than pre-Plan-384 781 —
            exactly the moved trd tests, now resident in katgpt-speculative) ✅
      - T3.7 `cargo run --example vocab_channel_pruner_demo`: runs end-to-end
            through the re-export path ✅

- [x] **T4.** Update Proposal 003 — added Phase 14 entry (DONE 2026-07-05) +
      updated Phase 12 "Final src/ state" line + updated TL;DR to reflect
      14 phases.

- [x] **T5.** Commit on `develop`.

## Notes

- Used `CARGO_TARGET_DIR=/tmp/plan384` per AGENTS.md to avoid lock contention
  with sibling agents.
- Both moves were done sequentially by the main agent (small enough scope
  that subagent overhead would dominate).
