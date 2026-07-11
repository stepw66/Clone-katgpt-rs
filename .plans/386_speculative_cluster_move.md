# Plan 386 — Speculative Cluster Move to katgpt-speculative (Phase 1)

## TL;DR

Apply the Plan 385 lesson (line-range grep the body, not the signature) to
`src/speculative/`. Re-audit reveals **28 of 40** root-only speculative files
have ZERO genuine root dependencies — their `crate::` references are all leaf
re-exports (`katgpt-core`, `katgpt-types`, `katgpt-pruners`, `katgpt-transformer`).
The "transformer-bound" / "root-only sibling" classifications are stale, exactly
as Plan 385 found for `forward`.

Phase 1 moves the **12 zero-crate-ref / leaf-only-dep files** (the safest
cluster). Phase 2+ (follow-up plans) will move the remaining 16 leaf-only-dep
files + tackle the `forward`-blocked trio (`dflash`, `verifier`, `drafter_lora`).

## Background — the Plan 385 lesson applied

Plan 385 (2026-07-05) discovered that `forward`'s "linchpin" classification
was a signature-only misread: line-range grep of the body showed all `crate::*`
refs were leaf re-exports. The same audit applied to `src/speculative/`:

```
grep -oE "crate::[a-z_:]+(::[a-zA-Z_:]+)*" src/speculative/*.rs | sort -u
```

Result: the only genuine root-only blockers across all 40 files are:
1. `crate::dllm` (root-only, DEFER) — blocks 6 files (d2f, d2f_verifier, diffusion_sampler, flashar_anchor, flashar_consensus, set_diffusion)
2. `crate::dash_attn` (root-only) — blocks prefill.rs
3. `crate::transformer::forward_paged` (root-only) — blocks types.rs (partially)
4. `crate::transformer::forward_decode_stage` (root-only) — blocks step.rs
5. `crate::transformer::forward` — **CYCLE CONSTRAINT**: katgpt-forward depends
   on katgpt-speculative, so files using `forward` cannot move to katgpt-speculative
   (would create katgpt-speculative → katgpt-forward → katgpt-speculative cycle).
   Blocks: dflash, verifier, drafter_lora (+ dllm-blocked: d2f_verifier, flashar_*).

Everything else (`crate::types::*`, `crate::pruners::*`, `crate::speculative::types::*`,
`crate::transformer::TransformerWeights`, `crate::cumprodsum::*`,
`crate::precision_aware_draft::*`, `crate::spechop::*`, etc.) resolves through
leaf re-exports.

## Phase 1 — Move set (12 files)

All files have zero `crate::` refs OR only leaf-reexport refs. Grouped by gate:

| File | Gate | LOC | Deps |
|---|---|---:|---|
| acceptance_forecast.rs | (ungated) | ~200 | 0 |
| belief_cache.rs | belief_drafter | ~150 | 0 |
| belief_drafter.rs | belief_drafter | ~700 | katgpt_core::{Config,SpeculativeGenerator}, simd |
| best_buddies.rs | best_buddies | ~130 | katgpt_core::traits |
| domino.rs | domino_correction | ~360 | std only |
| spec_generator.rs | speculative_generator | ~215 | katgpt_core |
| answer_extract.rs | parallel_probe | ~115 | 0 |
| dendritic_gate.rs | dendritic_gate | ~90 | 0 |
| kurtosis_gate.rs | kurtosis_gate | ~400 | 0 |
| nf_flow_generator.rs | nf_flow_score + speculative_generator | ~280 | super::nf_flow (leaf), super::spec_generator (moves with) |
| nf_flow_qgf.rs | nf_flow_score + qgf_drafter | ~625 | super::nf_flow, super::nf_flow_generator, super::spec_generator, katgpt_core::qgf |
| selectivity_router.rs | selectivity_router | ~590 | 0 |

**Total: ~3855 LOC moved.**

## Tasks

### T1: katgpt-speculative/Cargo.toml — add features
- [x] Add tracking features: `belief_drafter`, `best_buddies`, `domino_correction`, `speculative_generator`, `dendritic_gate`, `kurtosis_gate`, `selectivity_router`, `qgf_drafter`, `nf_flow_score`, `parallel_probe`, `acceptance_forecast`
- [x] Add `papaya`, `bytemuck` optional deps
- [x] Add `depth_invariance`, `self_cond_draft` tracking flags (discovered during compile)

### T2: Move 12 files to crates/katgpt-speculative/src/
- [x] Physical `git mv` for each file
- [x] Fix import paths: `super::nf_flow` → `crate::nf_flow`, `super::spec_generator` → `crate::spec_generator`, `super::nf_flow_generator` → `crate::nf_flow_generator` (in nf_flow_qgf)

### T3: katgpt-speculative/src/lib.rs — declare modules
- [x] Add `pub mod` declarations with matching feature gates

### T4: Root src/speculative/mod.rs — convert to re-exports
- [x] Replace `pub mod X;` with `pub use katgpt_speculative::X;` for each moved file
- [x] Preserve all existing `pub use X::*` re-exports (resolve via the leaf)

### T5: Root Cargo.toml — forward features to katgpt-speculative
- [x] Update each feature to also enable `katgpt-speculative/<feature>`
- [x] Removed `papaya` from root `belief_drafter` (now lives in leaf)
- [x] Forward `depth_invariance` + `self_cond_draft` to katgpt-speculative

### T6: GOAT gate G3 — compile + test
- [x] `cargo check --workspace` (default features) — clean
- [x] `cargo check --workspace --all-features` — clean
- [x] `cargo check --workspace --no-default-features` — clean
- [x] `cargo test -p katgpt-speculative --all-features --lib` — **848 passed** (178 from moved modules)
- [x] `cargo test --lib` (root, default) — **613 passed**
- [x] `cargo test --lib --all-features` (root) — **1071 passed** (1249 pre-Plan-385 − 178 moved = 1071 ✅)
- [x] `belief_drafter_goat` integration — **12/12 PASS**
- [x] `prof_dense_mesh` integration — **5/5 PASS**
- [x] `and_or_goat` integration — **5/5 PASS**
- [x] `bench_023_adaptive_gamma` smoke — PASS

### T7: Docs + commit
- [x] Update .proposals/003_src_consolidation_master.md (Phase 16 entry)
- [x] Commit on develop

## Defers (Phase 2+ candidates)

| Item | LOC | Blocker |
|---|---:|---|
| dd_tree.rs | 5990 | Test dep on `crate::speculative::dflash::dflash_predict` (dflash blocked by forward cycle) |
| dflash.rs | 1731 | `crate::transformer::forward` (CYCLE: katgpt-forward → katgpt-speculative) |
| verifier.rs | 930 | `crate::transformer::forward` (CYCLE) |
| drafter_lora.rs | 1107 | `crate::transformer::forward` (CYCLE) |
| alpha, budget, flow_pruner, peira_pruner, trust_region, and_or_builder, precision_aware_generator, residency_audit, thinking_controller, echo_env*, caddtree_budget, parallel_probe, budget_compat | ~8K | Leaf-only deps — moveable but deferred to keep Phase 1 safe |

The `forward`-cycle block (dflash/verifier/drafter_lora) needs architectural
decision: either (a) move `forward` down from katgpt-forward to a lower crate,
or (b) accept these stay in root, or (c) create a new crate above katgpt-forward.
Tracked as a Phase 2 follow-up.
