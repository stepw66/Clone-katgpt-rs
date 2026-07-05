# Plan 396 — speculative Phase 10: dd_tree full relocation to katgpt-forward

Status: **DONE** (2026-07-05). All builds clean (default / all-features /
no-default — zero warnings). 709 root lib tests pass + 162 katgpt-forward lib
tests pass = 871 total (Plan 394 baseline was 787 + 84 = 871 — test count
parity preserved, the 78 dd_tree tests moved cleanly from root to
katgpt-forward). 10/10 GOAT tests pass on bench_102_tilert_pipeline_goat
(default). 1/1 bench_165_hydra_budget_goat with hydra_budget + decode_specialize.
Branch: `develop`.
Predecessor: Plan 394 (Phase 9 forward-cycle 5-file move — CLOSED `98b24c69`).

NOTE: Plan 395 was taken by a parallel agent session (HOLA hippocampal exact KV
cache, commit `4a7ec2a0`). This plan renumbered to 396 to avoid collision.
Commit: `c88d1b0a`.

## Problem

`src/speculative/dd_tree.rs` is 2556 LOC. The audit revealed:

- **Production** (~60 LOC): 2 root-only functions
  - `build_dd_tree_screened_with_schedule` (gated `thinking_prune`)
  - `build_dd_tree_gdsd` (gated `gdsd_distill`)
- **Tests** (~2380 LOC, lines 177-2556): a massive `mod tests` block.

The leaf `crates/katgpt-speculative/src/dd_tree.rs` (4789 LOC) already hosts the
bulk production code + its own 652-LOC test module. Root's tests are
**additional** coverage that exercise root-resident helpers + the dd_tree+dflash
combination.

### Dependency audit (the key finding)

EVERY `crate::*` reference in root `dd_tree.rs` resolves to a leaf that
katgpt-forward already depends on:

| Root path | Resolves to |
|---|---|
| `crate::speculative::dflash::dflash_predict` | `katgpt_forward::dflash::dflash_predict` (moved Plan 394) |
| `crate::speculative::belief_drafter::*` | `katgpt_speculative::belief_drafter` |
| `crate::speculative::best_buddies::*` | `katgpt_speculative::best_buddies` |
| `crate::speculative::domino::*` | `katgpt_speculative::domino` |
| `crate::speculative::spec_generator::*` | `katgpt_speculative::spec_generator` |
| `crate::speculative::types::SdeConfig` | `katgpt_core::speculative::types::SdeConfig` |
| `crate::pruners::*` | `katgpt_pruners::*` (re-export shim) |
| `crate::transformer::TransformerWeights` | `katgpt_transformer::TransformerWeights` |
| `crate::types::{Config, Rng}` | `katgpt_types::{Config, Rng}` |

→ The entire file can move to katgpt-forward. No split-moves required.

## Strategy

Single-phase move: copy file → rewrite imports → register module → slim root
to re-export shim → verify build + tests.

### Feature gates to add to katgpt-forward

The test module + 2 production functions use these feature gates. katgpt-forward
already has `domino_correction`. Missing 7:

- `speculative_generator` (forward to `katgpt-speculative/speculative_generator`)
- `best_buddies` (forward to `katgpt-speculative/best_buddies`)
- `belief_drafter` (forward to `katgpt-speculative/belief_drafter`)
- `dflare_progressive_budget` (forward to `katgpt-speculative/dflare_progressive_budget`)
- `elf_sde` (forward to `katgpt-speculative/elf_sde`)
- `thinking_prune` (forward to `katgpt-pruners/thinking_prune` + `katgpt-speculative/thinking_prune`)
- `gdsd_distill` (forward to `katgpt-pruners/gdsd_distill`)

## Phase 1 — Move dd_tree.rs to katgpt-forward

- [x] T1.1 Copy `src/speculative/dd_tree.rs` → `crates/katgpt-forward/src/dd_tree.rs`.
- [x] T1.2 Rewrite imports in moved file:
  - `crate::speculative::dflash::dflash_predict` → `crate::dflash::dflash_predict`
  - `crate::speculative::types::SdeConfig` → `katgpt_core::speculative::types::SdeConfig`
  - `crate::speculative::belief_drafter::*` → `katgpt_speculative::belief_drafter::*`
  - `crate::speculative::best_buddies::*` → `katgpt_speculative::best_buddies::*`
  - `crate::speculative::domino::*` → `katgpt_speculative::domino::*`
  - `crate::speculative::spec_generator::*` → `katgpt_speculative::spec_generator::*`
  - `crate::pruners::*` → `katgpt_pruners::*`
  - `crate::transformer::TransformerWeights` → `katgpt_transformer::TransformerWeights`
  - `crate::types::*` → `katgpt_types::*`
  - `crate::Config` → `katgpt_types::Config`
  - `super::types::{ScreeningPruner,TreeNode,...}` → split:
    `katgpt_core::traits::*` (traits) + `katgpt_core::speculative::types::TreeNode`
- [x] T1.3 Add 7 tracking features to katgpt-forward/Cargo.toml + `fastrand` dev-dep.
- [x] T1.4 Forward the 7 new features from root Cargo.toml.
- [x] T1.5 Register module in katgpt-forward/src/lib.rs: `pub mod dd_tree;` +
      feature-gated re-exports of the 2 production fns.
- [x] T1.6 Slim root `src/speculative/dd_tree.rs` to a 23-LOC re-export shim:
      `pub use katgpt_forward::dd_tree::*;` + feature-gated re-exports of
      `build_dd_tree_screened_with_schedule` + `build_dd_tree_gdsd`.
- [x] T1.7 `cargo check --workspace` clean (default).

### Corrections during implementation

- `gdsd_distill` is NOT an empty tracking flag — `GdsdPruner`/`GdsdConfig`/
  `identity_advantage` are gated `gdsd_distill` in katgpt-pruners. Forwarded to
  `katgpt-pruners/gdsd_distill`.
- `dflare_progressive_budget` needs to forward to BOTH `katgpt-core` (for
  `PositionWeightedBudget`) AND `katgpt-speculative` (for
  `build_dd_tree_screened_progressive` + `TreeBuilder::build_screened_progressive`).
- `fastrand` is used by the spec_generator tests → added as dev-dep.

## Phase 2 — Verification

- [x] T2.1 `cargo check --workspace` (default) — zero warnings.
- [x] T2.2 `cargo check --workspace --all-features` — zero warnings.
- [x] T2.3 `cargo check --workspace --no-default-features` — zero warnings.
- [x] T2.4 `cargo test -p katgpt-forward --lib` (default) — **108/108** PASS.
- [x] T2.5 `cargo test -p katgpt-forward --lib --all-features` — **162/162** PASS.
- [x] T2.6 `cargo test -p katgpt-rs --lib --all-features` — **709/709** PASS
      (test count dropped from Plan 394's 787 because the 78 dd_tree tests
      moved to katgpt-forward; total across both crates = 871, identical to
      Plan 394 — perfect test parity).
- [x] T2.7 GOAT `bench_102_tilert_pipeline_goat` (default) — **10/10** PASS.
- [x] T2.8 GOAT `bench_165_hydra_budget_goat` (hydra_budget + decode_specialize) — **1/1** PASS.
- [x] T2.9 Commit on develop.

## LOC impact

| File | Before | After | Δ |
|---|---:|---:|---:|
| `src/speculative/dd_tree.rs` (root) | 2556 | 23 | **-2533** |
| `crates/katgpt-forward/src/dd_tree.rs` | 0 | 2562 | +2562 |
| **Net root reduction** | | | **-2533 LOC** |

## GOAT Gate

- G1 correctness: moved tests stay green (test parity).
- G2 perf: identical code paths, just relocated.
- G3 no-regression: workspace builds clean at default / all-features / no-default.
- G4 alloc-free: N/A (substrate relocation only).

## Risks

- Feature flag plumbing: 7 new tracking features. `cargo check --all-features` catches misses.
- Test count parity: root tests drop, katgpt-forward tests grow by same amount.
- The 2 root-only production functions reference `crate::pruners::{PrunerSchedule, GdsdPruner, ...}` — these resolve to `katgpt_pruners::*` after rewrite.

## Next session (Plan 397 candidate)

With Plan 396 done, the speculative cluster's largest remaining root-resident
targets are:

1. **`dllm-cycle` cluster** (~5.7K LOC, 4 files): `d2f.rs` (2301), `d2f_verifier.rs`
   (311), `diffusion_sampler.rs` (1463), `set_diffusion.rs` (1631). Root-only,
   gated `dllm` / `tri_mode` / `set_diffusion`. Heavily coupled to
   `TransformerWeights` and the root `dllm` module. Apply the Plan 393→394
   cadence: substrate extraction first, wrapper move next. The `d2f_verifier.rs`
   (311 LOC) depends on `crate::speculative::verifier::SpeculativeVerifier`
   which now lives in katgpt-forward — likely movable directly.

2. **`flashar_anchor` + `flashar_consensus`** (~1.6K LOC): `flashar_anchor.rs`
   (728), `flashar_consensus.rs` (853). Gated `flashar_anchor` / `flashar_consensus`.
   May be movable to katgpt-forward if their deps dissolve — they consume the
   d2f/diffusion_sampler cluster, so they're blocked on (1).

3. **`proof_6_decode_stages_match_forward` floating-point fix** — pre-existing
   precision issue (~1e-6 diff between forward and forward_draft). Independent
   of consolidation but worth fixing.
