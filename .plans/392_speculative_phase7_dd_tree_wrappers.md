# Plan 392 — speculative Phase 7: dd_tree feature-gated wrappers extraction

Status: **Phase 1+2 DONE** (2026-07-05). All builds clean, 71/71 root
 dd_tree tests pass, 1054 leaf tests pass. Predecessor (Plan 391) landed
 Status: **Phase 1+2+3 DONE** (2026-07-05). All tasks complete.
 Branch: `develop`
 Predecessor: Plan 391 (Phase 1+2 DONE) — pure substrate + dead-duplicate
 removal (~841 LOC moved/killed).

## Problem

Root `src/speculative/dd_tree.rs` is **5149 LOC** after Plan 391. The
"feature-gated wrappers that depend on root-only sibling modules" framing
(Plan 391 Phase 3) is now **stale** — every referenced sibling module
(`domino`, `spec_generator`, `belief_drafter`, `kurtosis_gate`,
`best_buddies`, `correlation_budget`, `nf_flow_budget`, `and_or_builder`,
`blueprint`, `decomp_reviewer`) was absorbed into `katgpt-speculative`
between Plans 386–390. Root's `super::X` paths resolve through
`pub use katgpt_speculative::X` shims in `src/speculative/mod.rs`.

So all Phase 3 wrappers (Plan 391) can move with **import rewrites only** —
no architectural change. Only the items with genuine `crate::pruners::*`
dependencies stay root-bound.

## Strategy: three-phase increment

### Phase 1 — Port TreeBuilder extended methods into leaf

The root TreeBuilder has 3 methods the leaf lacks:
- `build_screened_progressive` (gated `dflare_progressive_budget`)
- `build_screened_with_depth_budgets` (gated `corr_budget` or `nf_flow_budget`)
- `build_screened_recfm` (gated `recfm`)

These must move first because the wrappers in Phase 2 call them. Port
verbatim with import rewrites.

- [x] T1.1 Audit root's 3 methods for any non-leaf deps (expected: none
      beyond `crate::types::Config`, `super::types::PositionWeightedBudget`,
      `extract_parent_tokens_into` — all leaf-accessible)
- [x] T1.2 Port `build_screened_progressive` to leaf TreeBuilder
      (gate `dflare_progressive_budget`; needs leaf feature flag)
- [x] T1.3 Port `build_screened_with_depth_budgets` to leaf TreeBuilder
      (gate `corr_budget` OR `nf_flow_budget`; both already in leaf)
- [x] T1.4 Port `build_screened_recfm` to leaf TreeBuilder
      (gate `recfm`; needs leaf feature flag — `CrossScaleConfig` is in leaf)
- [x] T1.5 Add forward features in root Cargo.toml if leaf lacks them
      (`dflare_progressive_budget`, `corr_budget`, `nf_flow_budget`, `recfm`)
- [x] T1.6 Remove the 3 methods from root TreeBuilder (PLUS removed entire
      root TreeBuilder struct + impl — verified dead duplicate after Phase 2)
- [x] T1.7 `cargo check --workspace` clean

### Phase 2 — Move feature-gated wrappers to leaf

All wrappers below compose leaf-resident siblings (now in `katgpt-speculative`)
or `katgpt_core::speculative::types`. Move with import rewrites only.

- [x] T2.1 Move `build_dd_tree_domino` (gate `domino_correction`)
- [x] T2.2 Move `build_dd_tree_speculative` (gate `speculative_generator`)
- [x] T2.3 Move `belief_sigmoid`, `build_dd_tree_belief`,
      `build_dd_tree_belief_collapse_aware` (gate `belief_drafter`)
- [x] T2.4 Move `build_dd_tree_speculative_kurtosis`
      (gates `speculative_generator` + `kurtosis_gate`)
- [x] T2.5 Move `build_dd_tree_speculative_best_buddies`
      (gates `speculative_generator` + `best_buddies`)
- [x] T2.6 Move `build_dd_tree_screened_progressive` (gate `dflare_progressive_budget`)
- [x] T2.7 Move `build_dd_tree_screened_corr` (gate `corr_budget`)
- [x] T2.8 Move `build_dd_tree_screened_flow_budget` (gate `nf_flow_budget`)
- [x] T2.9 Move `build_dd_tree_screened_recfm` (gate `recfm`)
- [x] T2.10 Move `build_dd_tree_and_or`, `collect_solved_path`, `path_to_tree_nodes`
       (gate `and_or_dtree`; uses `katgpt_core::{AndOrNode, proof_cache::ProofGoalCache}`)
- [x] T2.11 Root: re-export via `pub use katgpt_speculative::dd_tree::*` (already happens)

### Phase 3 — Root-bound items + verification

Root retains:
- `build_dd_tree_screened_with_schedule` (uses `crate::pruners::PrunerSchedule`)
- `build_dd_tree_gdsd` (uses `crate::pruners::{GdsdConfig, GdsdPruner, identity_advantage}`)
- `mod tests`
- Module shim + `pub use katgpt_speculative::dd_tree::*`

- [x] T3.1 Root Cargo.toml: forward any new leaf features (audit each)
- [x] T3.2 `cargo check --workspace --all-features` clean
- [x] T3.3 `cargo check --workspace --no-default-features` clean
- [x] T3.4 `cargo test -p katgpt-rs --lib speculative::dd_tree` 71/71
- [x] T3.5 `cargo test -p katgpt-speculative --lib --all-features` 1054/1054 passes
- [x] T3.6 GOAT tests pass:
      - `test_133_parallel_probe_ablation` 1/1 PASS
      - `speculative_generator_goat` 3/3 PASS
      - `examples/lodestar_01_bench.rs` builds clean
      - `cargo test -p katgpt-rs --lib --all-features` 859/859 PASS
- [x] T3.7 Plan file updated with final LOC numbers; all tasks marked `[x]`

## Final LOC

| File | Plan 391 end | Plan 392 end | Delta |
|---|---:|---:|---:|
| `src/speculative/dd_tree.rs` | 5149 | 2556 | -2593 (-50%)
| `crates/katgpt-speculative/src/dd_tree.rs` | 3254 | 4789 | +1535 (+47%)

Root file is larger than the original ~1500 estimate because the test module
(~2380 LOC) was untouched per plan rules — it remains root-bound due to
`TransformerWeights` + `dflash_predict` deps. Excluding tests, root's actual
production code is now ~150 LOC (2 root-bound wrappers + module shim).

## Bonus deletion: root TreeBuilder duplicate

After Phase 2 moved all wrappers, the root's `TreeBuilder` struct + impl
(~998 LOC) became dead duplicate code. Verified zero `TreeBuilder::new` call
sites in non-test root code (only `build_dd_tree_screened_with_schedule` and
`build_dd_tree_gdsd` remained, and they only call the leaf's
`build_dd_tree_screened`). Deleted the entire struct + impl; the leaf's
TreeBuilder surfaces via the existing `pub use katgpt_speculative::dd_tree::*`
glob.

## Pre-existing flaky test (NOT related to Plan 392)

`workflow_lattice::tests::test_bench_lattice_vs_noop` in katgpt-pruners
failed once under full `--all-features --lib` parallel load, then passed when
run in isolation. The test asserts overhead < 500ns per call — inherently
perf-flaky under parallel test execution load. Unrelated to Plan 392 changes
(no code in `workflow_lattice.rs` was touched).

## Estimated impact

| File | Before | After Phase 3 (actual) | Notes |
|---|---:|---:|---|
| `src/speculative/dd_tree.rs` | 5149 | 2556 | -2593 LOC (-50%). Root retains 2 root-bound wrappers + 2380-line test module (untouched per plan rules). Bonus: deleted dead-duplicate TreeBuilder struct+impl (~998 LOC) after verifying zero non-test `TreeBuilder::new` call sites. |
| `crates/katgpt-speculative/src/dd_tree.rs` | 3254 | 4789 | +1535 LOC: 3 TreeBuilder methods (~980 LOC) + 13 wrapper functions (~555 LOC). |

## Import rewrite table (extends Plan 391)

| Root pattern | Leaf rewrite |
|---|---|
| `crate::types::Config` | `katgpt_types::Config` |
| `super::types::PositionWeightedBudget` | `katgpt_core::speculative::types::PositionWeightedBudget` |
| `super::types::RejectionReason` | `katgpt_core::speculative::types::RejectionReason` |
| `super::domino::{compute_prefix_strength, domino_score}` | `crate::domino::{compute_prefix_strength, domino_score}` |
| `super::spec_generator::*` | `crate::spec_generator::*` |
| `super::belief_drafter::BeliefDrafter` | `crate::belief_drafter::BeliefDrafter` |
| `super::kurtosis_gate::excess_kurtosis` | `crate::kurtosis_gate::excess_kurtosis` |
| `super::best_buddies::MarginalBestBuddyAligner` | `crate::best_buddies::MarginalBestBuddyAligner` |
| `super::correlation_budget::CorrelationBudgetAllocator` | `crate::correlation_budget::CorrelationBudgetAllocator` |
| `super::nf_flow_budget::FlowBudgetAllocator` | `crate::nf_flow_budget::FlowBudgetAllocator` |
| `super::and_or_builder::AndOrBuilder` | `crate::and_or_builder::AndOrBuilder` |
| `super::blueprint::BlueprintPass` | `crate::blueprint::BlueprintPass` |
| `super::decomp_reviewer::DecompositionReviewer` | `crate::decomp_reviewer::DecompositionReviewer` |
| `crate::pruners::proof::ProofGoalCache` | `katgpt_core::proof_cache::ProofGoalCache` |
| `crate::types::Rng` / `crate::types::Config::...` | `katgpt_types::Rng` / `katgpt_types::Config::...` |

## GOAT Gate

- G1 correctness: `cargo test` parity (no test count change for moved substrate)
- G2 perf: identical code paths, just relocated
- G3 no-regression: workspace builds clean at default / all-features / no-default
- G4 alloc-free: not applicable (substrate relocation only)

## Risks

- TreeBuilder method port: the 3 methods are ~300 LOC each. Must port
  verbatim — no behavior change. The leaf TreeBuilder's `build_screened`
  may have minor differences from root's; verify before/after equivalence
  by test parity.
- Feature-gate bookkeeping: the leaf Cargo.toml may need new tracking
  flags (`dflare_progressive_budget`, `recfm`, `and_or_dtree` if missing).
- The `build_dd_tree_and_or` move pulls in `katgpt_core::proof_cache` —
  already a leaf dep (via the existing `katgpt-core` dep), but verify the
  feature flag routing.
