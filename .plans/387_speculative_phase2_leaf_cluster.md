# Plan 387 — Speculative Phase 2 Leaf Cluster Move

**Status:** CLOSED
**Branch:** develop
**Predecessor:** Plan 386 (Phase 1 speculative cluster move, CLOSED `c97170c0`)
**Date:** 2026-07-05

## Goal

Move the second wave of root-only `src/speculative/*.rs` files to
`crates/katgpt-speculative/src/`, applying Plan 386's R296-class lesson
(line-range grep the body, not the signature). Plan 386 moved 12 files
(~3855 LOC); this plan moves 10 more (~4545 LOC).

## Re-audit methodology

Line-range grep of all 28 remaining root-only speculative files:

```bash
grep -oE "crate::[a-zA-Z_:]+(::[a-zA-Z_:]+)*" src/speculative/*.rs | sort -u
```

### Genuine root-only blockers (unchanged from Plan 386)

| Blocker | Files | Resolution |
|---|---|---|
| `crate::dllm` | d2f, d2f_verifier, diffusion_sampler, flashar_anchor, flashar_consensus, set_diffusion | DEFER (Plan 383) |
| `crate::dash_attn` | prefill | DEFER |
| `crate::transformer::forward` (cycle) | dflash, verifier, drafter_lora, step | DEFER — needs architectural decision |
| `crate::transformer::forward_paged/forward_decode_stage` | step, types (partial) | DEFER |

### NEW blocker discovered in Phase 2 audit — katgpt-pruners cycle

**`katgpt-pruners` depends on `katgpt-speculative`** (Cargo.toml line 21).
Therefore `katgpt-speculative` **cannot** depend on `katgpt-pruners` — it
would create a cycle.

This blocks 4 files that use `crate::pruners::*`:
- `and_or_builder.rs` — uses `crate::pruners::proof::*`
- `echo_env.rs` — uses `crate::pruners::bandit::*`
- `echo_env_integration.rs` — uses `crate::pruners::bandit::*`
- `thinking_controller.rs` — uses `crate::pruners::freeze::*`

These join the DEFER list. Resolution options for Phase 3:
(a) extract the consumed pruners items to katgpt-core, (b) accept root
residency, (c) invert the katgpt-pruners → katgpt-speculative dep.

### parallel_probe.rs — DEFER

Uses `super::verifier::SpeculativeVerifier` (trait in root `verifier.rs`).
`verifier.rs` is blocked by the `forward`-cycle. Deferring parallel_probe
keeps Phase 2 safe; it moves when the cycle resolves.

## Move set (10 files, ~4545 LOC)

| File | LOC | Feature gate | Leaf deps only |
|---|---:|---|---|
| `alpha.rs` | 636 | `lattice_deduction` | `speculative::types::ScreeningPruner` |
| `budget.rs` | 321 | `budget_adaptation` | `speculative::types::BudgetAdaptation` |
| `budget_compat.rs` | 137 | always-on | `speculative::budget::*` (moves with budget) |
| `caddtree_budget.rs` | 1056 | `caddtree_budget` | `mux_demux`, `build_dd_tree` (both LEAF) |
| `flow_pruner.rs` | 328 | `bandit` | `speculative::types::NoScreeningPruner` |
| `peira_pruner.rs` | 336 | `peira_distill` | `speculative::types::NoScreeningPruner` |
| `precision_aware_generator.rs` | 114 | `precision_aware_draft + speculative_generator` | `precision_aware_draft`, `speculative::spec_generator::*` |
| `residency_audit.rs` | 322 | always-on | `types::Config` |
| `trust_region.rs` | 734 | always-on | `types::Rng` |
| `domino_lora.rs` | 561 | `domino_lora` | `types::matmul` |

## Tasks

- [x] T1. Create plan file
- [x] T2. Move 10 files via `git mv`
- [x] T3. Add features + mod declarations to `katgpt-speculative`
- [x] T4. Update `src/speculative/mod.rs` re-exports
- [x] T5. Forward features in root `Cargo.toml`
- [x] T6. Build verify (default / all-features / no-default)
- [x] T7. Test verify (katgpt-speculative + root lib + integration goats)
- [x] T8. Update Proposal 003
- [x] T9. Commit on develop

## Verification matrix (G3 — no regression)

| Check | Command | Expected | Result |
|---|---|---|---|
| Default check | `cargo check --workspace` | Clean | ✅ PASS |
| All-features check | `cargo check --workspace --all-features` | Clean | ✅ PASS |
| No-default check | `cargo check --workspace --no-default-features` | Clean | ✅ PASS |
| Leaf lib tests | `cargo test -p katgpt-speculative --lib --all-features` | All pass | ✅ **970 passed** (was 848, +122 moved) |
| Root lib tests (default) | `cargo test --lib` | All pass | ✅ **501 passed** |
| Root lib tests (all-feat) | `cargo test --lib --all-features` | All pass | ✅ **949 passed** (was 1071, −122 moved = 949 exact) |
| belief_drafter_goat | integration | All pass | ✅ 12/12 PASS |
| prof_dense_mesh | integration | All pass | ✅ 5/5 PASS |
| and_or_goat | integration | All pass | ✅ 5/5 PASS |
| caddtree_budget_goat | integration | All pass | ✅ 7/7 PASS |
| bench_167_budget_adaptation_goat | integration | All pass | ✅ 8/8 PASS |
| bench_trust_region | integration | All pass | ✅ 6/6 PASS |
| bench_ldt_lattice_deduction | integration | All pass | ✅ 2/2 PASS |
| precision_aware_draft_goat | integration | All pass | ✅ 6/6 PASS |
| bench_023_adaptive_gamma smoke | integration | PASS | ✅ PASS |

## TL;DR

Plan 387 moves 10 more speculative files (~4545 LOC) from root to
katgpt-speculative leaf. Discovered a NEW blocker (katgpt-pruners cycle)
that defers 4 pruners-using files. parallel_probe deferred (depends on
root verifier trait). After this plan, root-only speculative drops from
28 → 18 files.

**GOAT gate G3 PASS**: all build configs clean, 970 leaf tests + 949
root all-features tests (exact arithmetic match: 1071 − 122 moved = 949),
8 integration goat suites all green.
