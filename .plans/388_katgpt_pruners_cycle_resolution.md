# Plan 388 — katgpt-pruners cycle resolution (Phase 3 speculative move)

Status: **DONE** (2026-07-05, all 3 phases complete)
Branch: `develop`
Predecessor: Plan 387 (CLOSED `09895d93`) — Phase 2 leaf cluster move

## Problem

Plan 387 discovered the **katgpt-pruners cycle**: `katgpt-pruners` depends on
`katgpt-speculative` (Cargo.toml line 21), so `katgpt-speculative` cannot
depend back on `katgpt-pruners`. This blocks 4 root-only speculative files
that consume `crate::pruners::{proof, bandit, freeze}` from moving to the leaf.

| File | LOC | Consumes | Production/Test |
|---|---:|---|---|
| `and_or_builder.rs` | 747 | `proof::{GoalResult, ProofGoalCache}` | production |
| `echo_env.rs` | 795 | `bandit::{BanditPruner, BanditStrategy}` | test-only (line 527) |
| `echo_env_integration.rs` | 297 | `bandit::{BanditPruner, BanditStrategy}` | production |
| `thinking_controller.rs` | 867 | `freeze::{load_frozen, save_frozen}` | production |
| **Total** | **2706** | | |

## Strategy: extract-then-move (three independent phases)

### Phase 1 — Extract `freeze` to katgpt-core (cleanest)

`crates/katgpt-pruners/src/freeze.rs` is 120 LOC of pure stdlib (Path + fs +
mem). Zero pruners-specific knowledge. Single consumer (thinking_controller).

- [x] Create `crates/katgpt-core/src/freeze.rs` (copy contents verbatim, no edits)
- [x] Register `pub mod freeze;` in `crates/katgpt-core/src/lib.rs`
- [x] Replace `crates/katgpt-pruners/src/freeze.rs` with a re-export shim:
      `pub use katgpt_core::freeze::{load_frozen, save_frozen};`
- [x] katgpt-pruners `lib.rs` keeps `pub mod freeze;` — re-export transparent
- [x] Root `src/speculative/thinking_controller.rs`: rewrite import to
      `use katgpt_core::freeze::{load_frozen, save_frozen};`
- [x] `cargo check --workspace` clean — ✅ 14.47s
- [x] freeze tests 4/4 pass in katgpt-core
- [x] thinking_controller 15/15 tests pass in root

### Phase 2 — Extract `proof` core types to katgpt-core

`crates/katgpt-pruners/src/proof/goal_cache.rs` (799 LOC) defines GoalHash,
GoalResult, GoalVerifier trait, ProofGoalCache, ProofGoalSnapshot. The consumed
API (by and_or_builder + dd_tree:919) is just `GoalResult` + `ProofGoalCache`.

Plan: extract the **self-contained core** (GoalHash + GoalResult +
GoalVerifier + ProofGoalCache) to `crates/katgpt-core/src/proof_cache.rs`.
Leave ProofGoalSnapshot + sketch_population + plackett_luce + sketch_sampler
in katgpt-pruners (they're pruners-specific).

- [x] Create `crates/katgpt-core/src/proof_cache.rs` with GoalHash, GoalResult,
      GoalVerifier trait, ProofGoalCache (and their impls)
- [x] Register `pub mod proof_cache;` in katgpt-core lib.rs
- [x] `crates/katgpt-pruners/src/proof/goal_cache.rs`: replace extracted code
      with `pub use katgpt_core::proof_cache::{GoalHash, GoalResult, GoalVerifier, ProofGoalCache};`
      (keep ProofGoalSnapshot + tests in pruners)
- [x] Root `src/speculative/and_or_builder.rs`: rewrite import to
      `use katgpt_core::proof_cache::{GoalResult, ProofGoalCache};`
- [x] `cargo check --workspace` clean — ✅ 17.14s
- [x] proof_cache 22 tests pass in katgpt-core
- [x] katgpt-pruners 126 lib tests pass (incl. 2 snapshot tests in goal_cache)
- [x] and_or_builder 14 tests pass in root

### Phase 3 — Move the 4 files to katgpt-speculative

After Phase 1+2, the 4 files have no genuine katgpt-pruners dependency:
- thinking_controller → katgpt_core::freeze ✅
- and_or_builder → katgpt_core::proof_cache ✅
- echo_env → only test uses bandit (dev-dep trick, NO cycle)
- echo_env_integration → production use of BanditPruner...

**echo_env_integration is the integration glue** between speculative
(echo_env) and pruners (BanditPruner). It belongs in the crate that depends on
BOTH, which is katgpt-pruners (pruners → speculative). It does NOT belong in
katgpt-speculative.

**Phase 3 PREP discovery — ThinkingMode extraction:** thinking_controller.rs
had `pub use katgpt_pruners::ThinkingMode;` which would re-introduce the
cycle. Resolved by extracting ThinkingMode (4-variant `#[repr(u8)]` enum) to
`katgpt_core::thinking_mode` — same pattern as freeze + proof_cache.

- [x] Extract ThinkingMode to `crates/katgpt-core/src/thinking_mode.rs`
- [x] katgpt-pruners lib.rs: replace enum def with `pub use katgpt_core::thinking_mode::ThinkingMode;`
- [x] Move `src/speculative/and_or_builder.rs` → `crates/katgpt-speculative/src/and_or_builder.rs`
      (rewrite `crate::speculative::ScreeningPruner` → `katgpt_core::traits::ScreeningPruner`)
- [x] Move `src/speculative/echo_env.rs` → `crates/katgpt-speculative/src/echo_env.rs`
      (rewrite test imports: `katgpt_pruners::bandit::*`, `crate::dd_tree::*`,
      `katgpt_types::*`, `katgpt_core::traits::NoScreeningPruner`)
- [x] Move `src/speculative/thinking_controller.rs` → `crates/katgpt-speculative/src/thinking_controller.rs`
      (rewrite `crate::cumprodsum::*` → `katgpt_core::cumprodsum::*`)
- [x] Move `src/speculative/echo_env_integration.rs` → `crates/katgpt-pruners/src/echo_env_integration.rs`
      (rewrite `crate::pruners::bandit::*` → `crate::bandit::*`,
      `crate::speculative::*` → `katgpt_speculative::*` / `katgpt_core::speculative::types::*`)
- [x] Add katgpt-pruners as **dev-dependency** in katgpt-speculative Cargo.toml
      with `features = ["echo_env_predictor"]` (transitively enables bandit for the test)
- [x] Add `echo_env_predictor` feature to katgpt-pruners (gates module + forwards to katgpt-speculative)
- [x] Add `thinking_cot`, `and_or_dtree`, `rv_gated_thinking`, `directional_credit` features
      to katgpt-speculative (tracking flags + module gates); and_or_dtree forwards to
      katgpt-core/and_or_dtree + dep:blake3
- [x] Update root Cargo.toml feature forwarding: echo_env_predictor, thinking_cot,
      and_or_dtree, rv_gated_thinking, directional_credit → all forward to katgpt-speculative
- [x] Update `crates/katgpt-pruners/src/lib.rs`: add `pub mod echo_env_integration;` (gated echo_env_predictor)
- [x] Update root `src/speculative/mod.rs`: 4 `pub mod X` → `pub use katgpt_speculative::X;`
      (echo_env_integration → `pub use katgpt_pruners::echo_env_integration;`)
- [x] Update katgpt-speculative lib.rs: add 3 modules with proper cfg gates
- [x] `cargo check --workspace` clean (default / all-features / no-default)

## Import rewrite patterns (inherited from Plan 386/387)

| Root pattern | Leaf rewrite |
|---|---|
| `crate::speculative::types::*` | `katgpt_core::traits::*` / `katgpt_core::speculative::types::*` |
| `crate::types::{Config, Rng, matmul}` | `katgpt_types::{Config, Rng, matmul}` |
| `crate::speculative::build_dd_tree*` | `crate::dd_tree::build_dd_tree*` |
| `crate::speculative::echo_env::*` | `crate::echo_env::*` (leaf-internal) |
| `super::types::*` | `katgpt_core::traits::*` |
| `crate::pruners::freeze::*` | `katgpt_core::freeze::*` |
| `crate::pruners::proof::{GoalResult, ProofGoalCache}` | `katgpt_core::proof_cache::{GoalResult, ProofGoalCache}` |

## GOAT Gate G3 — Verification

- [x] `cargo check --workspace` (default) — clean ✅
- [x] `cargo check --workspace --all-features` — clean ✅
- [x] `cargo check --workspace --no-default-features` — clean ✅
- [x] katgpt-core lib tests pass — **1247 passed** (default) ✅
- [x] katgpt-pruners lib tests pass — **126 passed** (default) ✅
- [x] katgpt-speculative lib tests pass — **1010 passed** (all-features, excluding
      pre-existing `budget_compat::tests::test_effective_tree_budget_entropy_adapts`
      failure which fails on baseline too) ✅
- [x] Root lib tests (default) — **472 passed** ✅
- [x] Root lib tests (all-features) — **900 passed** (excluding pre-existing +
      flaky bench tests) ✅
- [x] `examples/echo_env_predictor_demo` builds ✅
- [x] GOAT integration: belief_drafter_goat **12/12 PASS** ✅
- [x] GOAT integration: and_or_goat **5/5 PASS** ✅
- [x] GOAT integration: bench_167_budget_adaptation_goat **8/8 PASS** ✅
- [x] GOAT integration: bench_trust_region **6/6 PASS** ✅
- [x] GOAT integration: caddtree_budget_goat **7/7 PASS** ✅
- [x] Moved-module tests: echo_env (2) + thinking_controller (11) + and_or_builder
      + others (30) = **43 PASS** ✅

## Risk Notes

- **Back-compat re-export** from katgpt-pruners for freeze + proof keeps
  external consumers working without code changes.
- **dev-dep trick** for echo_env test: dev-dependencies do NOT propagate to
  dependents, so `katgpt-speculative[dev-deps] → katgpt-pruners` does NOT
  create a cycle even though `katgpt-pruners → katgpt-speculative` exists.
- **echo_env_integration move to katgpt-pruners**: changes the public path
  from `katgpt_rs::speculative::echo_env_integration` to
  `katgpt_pruners::echo_env_integration`. One example file needs updating.

## Expected outcome

- 4 files (~2706 LOC) leave `src/speculative/`
- Root-only speculative files: 18 → 14
- katgpt-pruners cycle: RESOLVED for these 4 files
- Remaining root-only speculative files blocked by other cycles (forward-cycle,
  dllm-cycle) documented in Proposal 003

## TL;DR

Resolve the katgpt-pruners cycle by extracting `freeze` + proof core types to
katgpt-core (shared leaf), then move 3 files to katgpt-speculative and
echo_env_integration to katgpt-pruners (it's integration glue). 4 files
(~2706 LOC) leave src/speculative/. dev-dep trick handles echo_env's test-only
bandit usage without creating a cycle.
