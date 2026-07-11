# Plan 391 — speculative Phase 6: dd_tree substrate extraction

Status: **CLOSED — SUPERSEDED by Plan 396** (2026-07-05). Phase 1+2 landed
directly; Phase 3 was rendered obsolete by Plan 396 (`c88d1b0a`), which moved
the entire remaining `dd_tree.rs` (root-only production fns + 2380-LOC test
module) to `katgpt-forward` in a single-phase relocation. The variant
wrappers that Phase 3 planned to move to `katgpt-speculative` are already
there (confirmed by grep, 2026-07-06).
Branch: `develop`
Predecessor: Plan 390 (CLOSED `97205828`) — Phase 5 prefill substrate extraction
Superseded by: Plan 396 (CLOSED `c88d1b0a`) — Phase 10 dd_tree full relocation

## Problem

Root `src/speculative/dd_tree.rs` is **5990 LOC** but most of it is duplicated
or moveable substrate:

| Section | LOC | Status |
|---|---:|---|
| Module doc + imports + `pub use katgpt_speculative::dd_tree::*` | 1-72 | Shim |
| `ManifoldValidWrapper` + `build_dd_tree_manifold` | 73-114 | Pure substrate (movable) |
| `build_dd_tree_lodestar` + `a_star_score` + `find_forced_token` | 116-324 | **DUPLICATE of leaf** (dead code) |
| `build_dd_tree_domino` | 326-413 | Movable (uses leaf DominoPruner + domino.rs) |
| `build_dd_tree_speculative` | 415-470 | Movable (uses leaf spec_generator) |
| `build_dd_tree_belief(_collapse_aware)` | 472-556 | Movable (uses leaf belief_drafter) |
| `build_dd_tree_speculative_kurtosis/_best_buddies` | 558-695 | Movable (uses leaf siblings) |
| `build_dd_tree_screened_progressive/_corr/_flow_budget` | 697-810 | Movable (uses leaf siblings) |
| `build_dd_tree_screened_with_schedule` | 812-835 | Root-bound (uses `crate::pruners::PrunerSchedule`) |
| `build_dd_tree_screened_recfm` | 837-861 | Movable (uses leaf CrossScaleConfig after move) |
| `build_dd_tree_gdsd` | 863-913 | Root-bound (uses `crate::pruners::{GdsdConfig, GdsdPruner}`) |
| `build_dd_tree_and_or` + helpers | 914-1050 | Movable (uses leaf and_or_builder) |
| `build_dd_tree_sde`, `_balanced_sde` | 1052-1096 | Pure substrate (movable) |
| `WidthSelectionMode`, `WidthScaleConfig`, `ResidualTracker` | 1107-1230 | Pure substrate (movable) |
| `CrossScaleConfig`, `branch_velocity_at`, `cross_scale_consistent` | 1241-1300 | Pure substrate (movable) |
| `best_of_k_rollouts`, `cumulative_relevance` | 1302-1443 | Pure substrate (movable) |
| `TreeBuilder` (duplicated+extended from leaf) | 1450-3412 | DEDUP base + move 3 extended methods |
| `entropy_truncate_horizon` | 3431-3439 | Pure substrate (movable) |
| `build_dd_tree_dendritic`, `top_k_indices` | 3465-3610 | Movable (uses leaf dendritic_gate) |
| `mod tests` | 3613-5990 | Root-bound (need TransformerWeights for dflash_predict) |

### Discovery: leaf already hosts LodestarConfig

`crate::pruners::LodestarConfig` is actually a re-export of
`katgpt_speculative::dd_tree::LodestarConfig` (via
`katgpt_pruners::lodestar::LodestarConfig` re-export at line 570). So the
root's `build_dd_tree_lodestar` is *literal duplicate code* that uses an
indirect path through pruners.

## Strategy: three-phase increment

### Phase 1 — Kill dead duplicates (zero-risk)

- [x] T1.1 Remove root's `build_dd_tree_lodestar`, `a_star_score`, `find_forced_token`
      (lines 116-324). They're already re-exported via
      `pub use katgpt_speculative::dd_tree::*`.
- [x] T1.2 `cargo check --workspace` clean
- [x] T1.3 `cargo test -p katgpt-rs --lib speculative::dd_tree` passes (71/71)

### Phase 2 — Move pure substrate (no root deps)

- [x] T2.1 Move `WidthSelectionMode`, `WidthScaleConfig`,
      `From<ConvergenceSelector>` (elf_sde/eqr_convergence gated)
- [x] T2.2 Move `ResidualTracker` (eqr_convergence gated)
- [x] T2.3 Move `CrossScaleConfig`, `branch_velocity_at`, `cross_scale_consistent`
      (recfm gated)
- [x] T2.4 Move `entropy_truncate_horizon` (sr2am_configurator gated)
- [x] T2.5 Move `build_dd_tree_sde`, `build_dd_tree_balanced_sde` (always-on)
- [x] T2.6 Move `best_of_k_rollouts`, `cumulative_relevance` (elf_sde gated)
- [x] T2.7 Move `build_dd_tree_dendritic`, `top_k_indices` (dendritic_gate gated)
- [x] T2.8 Move `ManifoldValidWrapper`, `build_dd_tree_manifold` (manifold_pruner gated)
- [x] T2.9 Root: re-export via `pub use katgpt_speculative::dd_tree::*`
      (already happens via glob; local definitions removed)
- [x] T2.10 Cargo.toml: add forward features for elf_sde/eqr_convergence/recfm/
      sr2am_configurator/dendritic_gate/manifold_pruner to katgpt-speculative
      (added: elf_sde, eqr_convergence, manifold_pruner, bt_rank to leaf;
       recfm/sr2am_configurator/dendritic_gate already existed)
- [x] T2.11 `cargo check --workspace` clean (default, all-features, no-default)
- [x] T2.12 Tests pass: 431/431 default, 859/859 all-features, 1054/1054 leaf
      all-features, 71/71 dd_tree root tests, 34/34 lodestar pruners tests,
      1/1 parallel_probe GOAT, 3/3 speculative_generator GOAT.

**Bonus**: ported root's `span_parents_buf` allocation-reuse optimization
for `build_dd_tree_lodestar` jump-ahead into the leaf (was a microscopic
perf regression in the leaf's `to_vec()` version). Now leaf is the
perf-canonical implementation.

### Phase 3 — Move feature-gated variant wrappers (uses leaf siblings)

**SUPERSEDED by Plan 396 (`c88d1b0a`, 2026-07-05).** Plan 396 chose a cleaner
strategy: instead of moving these variant wrappers to `katgpt-speculative`
piecemeal (Phase 3's plan), it moved the *entire* remaining root
`dd_tree.rs` — the two root-only production fns AND the 2380-LOC test
module — to `katgpt-forward` in a single-phase relocation. The variant
wrappers themselves (T3.1-T3.10) already live in `katgpt-speculative`
(verified by grep on 2026-07-06: all 11 `build_dd_tree_*` fns present at
lines 3607-4027). The root is now a 23--LOC re-export shim.

The tasks below are marked complete to reflect the supersession; no new
work was done under Plan 391 for these items.

- [x] T3.1 Move `build_dd_tree_domino` — DONE (in `katgpt-speculative/src/dd_tree.rs:3607`)
- [x] T3.2 Move `build_dd_tree_speculative` — DONE (`katgpt-speculative/src/dd_tree.rs:3645`)
- [x] T3.3 Move `build_dd_tree_speculative_kurtosis` — DONE (`katgpt-speculative/src/dd_tree.rs:3798`)
- [x] T3.4 Move `build_dd_tree_speculative_best_buddies` — DONE (`katgpt-speculative/src/dd_tree.rs:3876`)
- [x] T3.5 Move `build_dd_tree_belief`, `_belief_collapse_aware` — DONE (`katgpt-speculative/src/dd_tree.rs:3720`, `:3762`)
- [x] T3.6 Move `build_dd_tree_screened_progressive` — DONE (`katgpt-speculative/src/dd_tree.rs:3903`)
- [x] T3.7 Move `build_dd_tree_screened_corr` — DONE (`katgpt-speculative/src/dd_tree.rs:3920`)
- [x] T3.8 Move `build_dd_tree_screened_flow_budget` — DONE (`katgpt-speculative/src/dd_tree.rs:3945`)
- [x] T3.9 Move `build_dd_tree_screened_recfm` — DONE (`katgpt-speculative/src/dd_tree.rs:3989`)
- [x] T3.10 Move `build_dd_tree_and_or` + helpers — DONE (`katgpt-speculative/src/dd_tree.rs:4017`)
- [x] T3.11 Move `TreeBuilder::build_screened_progressive/_with_depth_budgets/_recfm`
      — DONE (`katgpt-speculative/src/dd_tree.rs:1970`, `:2311`, `:2584`)
- [x] T3.12 Root keeps `build_dd_tree_screened_with_schedule`, `build_dd_tree_gdsd`,
      and the test module — **OVERTAKEN by Plan 396**: all three moved to
      `katgpt-forward/src/dd_tree.rs`. Root is now a 23-LOC re-export shim
      (`pub use katgpt_forward::dd_tree::*` + feature-gated re-exports).
      This is *more* than Phase 3 planned — Plan 396's strategy was cleaner
      because katgpt-forward already depends on every sibling leaf the
      tests/production fns need.
- [x] T3.13 `cargo check --workspace` clean — VERIFIED by Plan 396 T2.1-T2.3
      (default / all-features / no-default all zero-warning).
- [x] T3.14 Tests pass — VERIFIED by Plan 396 T2.4-T2.8 (709 root + 162
      katgpt-forward = 871 total; test parity preserved from Plan 394
      baseline).

## Root-bound items remaining after Plan 391

**All resolved by Plan 396.** The items below were the expected post-Plan-391
residue; Plan 396 moved every one of them to `katgpt-forward`, leaving the
root file as a 23-LOC pure re-export shim.

- ~~`build_dd_tree_screened_with_schedule`~~ — moved to `katgpt-forward/src/dd_tree.rs` (Plan 396)
- ~~`build_dd_tree_gdsd`~~ — moved to `katgpt-forward/src/dd_tree.rs` (Plan 396)
- ~~`mod tests`~~ — moved to `katgpt-forward/src/dd_tree.rs` (Plan 396)
- Module shim + re-export — root is now `pub use katgpt_forward::dd_tree::*`
  (changed from `katgpt_speculative` because Plan 396 hosts the test module
  + root-only production fns in katgpt-forward, which already depends on
  katgpt-speculative for the variant wrappers)

Actual root file size after Plans 391 + 396: **23 LOC** (down from 5990 —
far beyond the original ~1500 LOC target).

## GOAT Gate

- G1 correctness: `cargo test` parity (no test count change for moved substrate)
- G2 perf: identical code paths, just relocated
- G3 no-regression: workspace builds clean at default / all-features / no-default
- G4 alloc-free: not applicable (substrate relocation only)

## Risks

- Feature-gate bookkeeping: the leaf has features for many but not all of the
  gates used here. May need to add `elf_sde`, `eqr_convergence`, `recfm`,
  `sr2am_configurator`, `manifold_pruner` tracking flags to leaf Cargo.toml.
- TreeBuilder has 3 root-only methods that need to merge with the leaf's impl.
  The leaf's TreeBuilder is in the same crate, so adding methods is straightforward
  but requires careful handling of `super::types::PositionWeightedBudget` →
  `crate::speculative::types::PositionWeightedBudget` (which the leaf re-exports
  from katgpt-core).
