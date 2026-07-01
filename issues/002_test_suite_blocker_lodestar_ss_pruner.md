# Issue 002 — Test-suite blocker: refactor lost lodestar glue + ss_pruner cross-crate path

Status: **RESOLVED** (2026-07-01) — T1–T4 done, Phase 0.5 unblocked
Created: 2026-07-01
Related: `proposals/003_src_consolidation_master.md` (Phase 0.5 is gated on this)

## TL;DR

`cargo check --workspace` is clean, but `cargo test --workspace` FAILS to
compile in `katgpt-pruners` due to two pre-existing refactor bugs. This
**blocks** the loser-sweep re-bench (Proposal 003 Phase 0.5): the GOAT
gates cannot be re-run until tests compile.

One of the two bugs is exactly the "shame" scenario the user flagged:
**lodestar is a GOAT-passed DEFAULT-ON winner whose integration glue was
dropped during the speculative-crate extraction**, making it *look* broken
when it is actually a winner. Re-benching in this state would have
misclassified a winner as a loser.

## Bug 1 — `build_dd_tree_lodestar` lost in speculative-crate extraction

**Symptom:** `crates/katgpt-pruners/src/lodestar.rs` tests reference
`katgpt_speculative::dd_tree::{build_dd_tree_lodestar, extract_parent_tokens,
build_dd_tree_pruned}`. `extract_parent_tokens` and `build_dd_tree_pruned`
exist in `katgpt-speculative/src/dd_tree.rs`; **`build_dd_tree_lodestar` does
not.** 5 compile errors (E0432).

**Provenance (this is NOT a loser):**
- Plan 207 T6–T8 explicitly implemented `build_dd_tree_lodestar` with the
  (A) budget mask, (B) jump-ahead, (C) A* heap ordering.
- `.benchmarks/055_lodestar_overhead_goat.md` records **GOAT 5/5 PASS** —
  per-call ~4–8ns, default-0 path +4.3% (within noise), budget-mask path
  −86.7% (faster than baseline).
- Plan 207 status: "Promoted to DEFAULT-ON. ALL 15/15 TASKS COMPLETE."

**What happened:** when `dd_tree.rs` moved to the `katgpt-speculative` crate,
`build_dd_tree`, `build_dd_tree_pruned`, `build_dd_tree_screened`, and
`build_dd_tree_balanced` were ported. `build_dd_tree_lodestar` + its
`CompletionHorizon` trait integration were **dropped**. The `LodestarPruner`
stayed in `katgpt-pruners`, but its consumer (the tree builder) is gone.

**Fix options:**
1. **Restore** `build_dd_tree_lodestar` + `CompletionHorizon` trait into
   `katgpt-speculative/src/dd_tree.rs` (preferred — re-establishes the
   GOAT-passed integration). Source: Plan 207 + the deleted history.
2. **Exile** lodestar to `katgpt-deprecated` (WRONG — it's a verified winner;
   exile would be the "shame" misclassification).

→ Option 1. This is a restore, not a demote.

## Bug 2 — `ss_pruner.rs` references root `cumprodsum`

**Symptom:** `crates/katgpt-pruners/src/ss_pruner.rs:209`:
`crate::cumprodsum::influence(&decay_factors, 0, depth)` → E0433 "could not
find `cumprodsum` in the crate root". `cumprodsum` lives in ROOT
(`src/cumprodsum.rs`), not in `katgpt-pruners`.

**Root cause:** `ss_pruner` was moved to the pruners crate but its test still
references the root-crate path. Classic move-and-lose-the-dep bug — the exact
class Proposal 003 is designed to prevent by moving `cumprodsum` → `katgpt-core`.

**Fix options:**
1. **Short-term (unblock tests now):** make the test compute the expected
   value inline or add a `katgpt-core` re-export of `influence` and use
   `katgpt_core::cumprodsum::influence`.
2. **Long-term (Proposal 003 Phase 10):** move `cumprodsum` → `katgpt-core`
   so it's accessible to all crates. Then `ss_pruner` references
   `katgpt_core::cumprodsum::influence`.

→ Short-term fix to unblock the bench; long-term handled by Phase 10.

## Impact on Proposal 003

- **Phase 0.5 (loser-sweep) is BLOCKED** — cannot re-bench until tests
  compile. The whole point of the sweep is to avoid false-loser
  classification; running it against a broken test suite would defeat that.
- **lodestar is a confirmed winner** — GOAT 5/5, default-ON. Do NOT exile.
  Its `katgpt-pruners` home is correct; only the speculative-side glue is
  missing. After Bug 1 fix, lodestar tests pass and it stays default-ON.
- **This validates the user's concern** — "ensure loser is lose not bc of
  bug." Lodestar would have been a shame-misclassification without the
  re-bench discipline.

## Tasks

- [x] **T1 (Bug 1):** restored `build_dd_tree_lodestar` + `LodestarConfig` into
      `katgpt-speculative/src/dd_tree.rs` (recovered verbatim from commit `6684b5d5`,
      paths adapted: `crate::types::Config` → `katgpt_types::Config`,
      `super::types::CompletionHorizon` → `katgpt_core::traits::CompletionHorizon`).
      `LodestarConfig` moved out of `katgpt-pruners` (avoids a speculative→pruners
      cycle) and re-exported from pruners for back-compat. Added `lodestar`
      feature to `katgpt-speculative`, forwarded by `katgpt-pruners/lodestar`.
      Also fixed a latent feature-combo bug: `lodestar` now implies `bandit`
      (lodestar_cot needs BanditStats — the `merkle_root` class).
- [x] **T2 (Bug 2):** inlined the `influence` oracle in `ss_pruner.rs` test
      (one-line `product()` helper). Long-term fix is Proposal 003 Phase 10
      (cumprodsum → katgpt-core).
- [x] **T2b (bonus bug found during T3):** `tes_loop.rs:465` imported
      `TesConfig` from `katgpt_speculative`, but `TesConfig` moved to the
      pruners crate (Plan 005). Fixed to use the local `super::*` scope.
      Third move-and-lose-the-path bug from the speculative extraction.
- [x] **T3:** `cargo test --workspace --lib` fully green — **5,266 tests,
      0 failures** across all 16 crates.
- [x] **T4:** lodestar test subset green — **34/34 PASS**, including
      `test_dd_tree_lodestar_budget_guarantee`,
      `test_dd_tree_lodestar_nopruner_matches_pruned`, jump-ahead, A*.
- [x] **T5:** Proposal 003 Phase 0.5 (loser-sweep) is unblocked.

## Resolution note

The re-bench discipline the user demanded paid off immediately: the test
suite was broken by THREE move-and-lose-the-path bugs from the speculative
+ pruners crate extractions. Without fixing these first, the loser-sweep
would have run against a non-compiling suite, misclassifying every
feature-gated primitive as a "loser" (false failure). Lodestar specifically
— a GOAT-5/5 default-ON winner — would have been the headline shame.

## References

- Plan 207: `.plans/207_lodestar_completion_distance_pruning.md`
- Bench 055: `.benchmarks/055_lodestar_overhead_goat.md` (GOAT 5/5)
- Proposal 003: `proposals/003_src_consolidation_master.md` (Phase 0.5)

## TL;DR

Test suite won't compile (3 refactor bugs — two originally identified +
one more found during the fix). All three fixed: lodestar glue restored
(GOAT-5/5 winner, NOT a loser), ss_pruner path inlined, tes_loop path
corrected. `cargo test --workspace --lib` now green: 5,266 tests, 0
failures. Phase 0.5 loser-sweep is unblocked.
