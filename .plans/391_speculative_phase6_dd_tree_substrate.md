# Plan 391 — speculative Phase 6: dd_tree substrate extraction

Status: **Phase 1+2 DONE** (2026-07-05). Phase 3 deferred to next session.
Branch: `develop`
Predecessor: Plan 390 (CLOSED `97205828`) — Phase 5 prefill substrate extraction

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

These wrappers compose leaf-resident siblings (already in katgpt-speculative).
They can move to the leaf with import rewrites only — no architectural change.

- [ ] T3.1 Move `build_dd_tree_domino` (uses leaf DominoPruner + domino.rs)
- [ ] T3.2 Move `build_dd_tree_speculative` (uses leaf spec_generator)
- [ ] T3.3 Move `build_dd_tree_speculative_kurtosis` (uses leaf kurtosis_gate)
- [ ] T3.4 Move `build_dd_tree_speculative_best_buddies` (uses leaf best_buddies)
- [ ] T3.5 Move `build_dd_tree_belief`, `_belief_collapse_aware` (uses leaf belief_drafter)
- [ ] T3.6 Move `build_dd_tree_screened_progressive` (uses leaf PositionWeightedBudget)
- [ ] T3.7 Move `build_dd_tree_screened_corr` (uses leaf correlation_budget)
- [ ] T3.8 Move `build_dd_tree_screened_flow_budget` (uses leaf nf_flow_budget)
- [ ] T3.9 Move `build_dd_tree_screened_recfm` (uses moved CrossScaleConfig)
- [ ] T3.10 Move `build_dd_tree_and_or` + helpers (uses leaf and_or_builder + katgpt_core::AndOrNode)
- [ ] T3.11 Move `TreeBuilder::build_screened_progressive/_with_depth_budgets/_recfm` into leaf's TreeBuilder
- [ ] T3.12 Root: keep `build_dd_tree_screened_with_schedule` (crate::pruners::PrunerSchedule dep),
       `build_dd_tree_gdsd` (crate::pruners::GdsdPruner dep), and the test module.
- [ ] T3.13 `cargo check --workspace` clean
- [ ] T3.14 Tests pass

## Root-bound items remaining after Plan 391

- `build_dd_tree_screened_with_schedule` — uses `crate::pruners::PrunerSchedule`
- `build_dd_tree_gdsd` — uses `crate::pruners::{GdsdConfig, GdsdPruner}`
- `mod tests` — needs `TransformerWeights` + `dflash_predict`
- Module shim + `pub use katgpt_speculative::dd_tree::*`

Estimated root file size after Plan 391: ~1500 LOC (down from 5990).

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
