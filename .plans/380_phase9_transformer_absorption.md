# Plan 380 — Proposal 003 Phase 9: katgpt-transformer absorption

## Goal

Move four `src/` items into the `katgpt-transformer` workspace crate:

| Item | LOC | Files |
|---|---|---|
| `mbu.rs` | 223 | 1 |
| `tf_loop.rs` | 639 | 1 |
| `dense_mesh/` | ~104KB | 12 (mod + 11 children) |
| `swir/` | ~124KB | 11 (mod + 9 children + 2 docs) |

**Total**: ~25 files. Destination: `crates/katgpt-transformer/src/`.

## Why this unblocks Phase 7 deferred work

Phase 7 deferred `sleep/` because `forward_looped` in `src/transformer.rs`
directly consumes `crate::sleep::SleepConfig`. Once `mbu`/`dense_mesh`/`swir`/
`tf_loop` are out of root, the residual `crate::transformer` surface shrinks,
which is the prerequisite for either (a) Phase 12 final sweep or (b) a future
re-attempt at moving `sleep/` once `ForwardContext` is the only transformer-bound
glue left in root.

## Pre-move audit checklist (per module)

For each module, identify:
1. External deps (other crates — `katgpt_core`, `katgpt_transformer`, etc.)
2. Root deps via shim (`crate::xxx` that resolves through `pub use katgpt_yyy::*`)
3. True root deps (symbols that genuinely live in `src/` and have no crate home)
4. Feature gates (root `Cargo.toml` + `src/lib.rs` `#[cfg(feature=...)]`)
5. External consumers (`riir-ai`, `tests/`, `examples/`, `benches/`)

## Tasks

- [x] **T1.** Audit `mbu.rs` deps — DONE: clean move, 1 import rewrite needed.
- [x] **T2.** Audit `tf_loop.rs` deps — DONE: clean move for bulk; `should_apply_pruner_at_iteration` retained at root (katgpt-pruners cycle).
- [x] **T3.** Audit `dense_mesh/` deps — DONE: 11 of 12 files clean; `node_transformer.rs` deferred (`crate::transformer::forward` dep).
- [x] **T4.** Audit `swir/` deps — DONE: 8 of 9 .rs files clean; `strategy_adapter.rs` deferred (`crate::thinking_cot::*` dep). `ControlToken::resolve_via` stripped from moved types.rs (also depended on thinking_cot).
- [x] **T5.** Decide per-module: clean move vs. partial move vs. defer — DONE: 3 partial moves (dense_mesh, swir, tf_loop) + 1 clean (mbu).
- [x] **T6.** Add features to `katgpt-transformer/Cargo.toml` for any feature-gated modules — DONE: 6 new features (`tf_loop`, `recfm`, `thinking_prune`, `dense_mesh`, `swir_switch_thinking`, `collapse_aware_thinking`, `breakeven_routing`) + `fastrand` dep.
- [x] **T7.** Copy modules to `crates/katgpt-transformer/src/` — DONE.
- [x] **T8.** Rewrite imports inside moved files — DONE: `mbu.rs` (1), `tf_loop.rs` (1 + strip pruner fn), `swir/types.rs` (strip resolve_via), `swir/mod.rs` (strip strategy_adapter), `dense_mesh/mod.rs` (strip node_transformer).
- [x] **T9.** Wire re-exports in root `src/lib.rs` + `src/dense_mesh/mod.rs` + `src/swir/mod.rs` + `src/tf_loop.rs` shims — DONE.
- [x] **T10.** Delete original files from `src/` — DONE (kept only shims + deferred files).
- [x] **T11.** Update root `Cargo.toml` features to forward to crate — DONE.
- [x] **T12.** GOAT gate G3 — DONE: workspace check (3 configs) + katgpt-transformer lib tests (122 PASS) + root lib tests (1397 PASS) + external consumer tests (mbu/tf_loop/dense_mesh/swir) + clippy.
- [x] **T13.** Update `proposals/003_src_consolidation_master.md` Phase 9 → DONE — DONE.
- [x] **T14.** Commit on `develop` with `refactor(transformer):` prefix — pending.

## Deferral protocol

If a module has a true root dep that can't be resolved via shim or import rewrite,
defer that module to Phase 12 (final sweep). Do NOT move half a module — either
the whole thing moves cleanly or it stays. Document the blocker.

If a module is feature-gated and the feature forwards work cleanly, move it. If
the feature has cross-crate coupling that doesn't forward, defer.

## Reference

- Proposal: `proposals/003_src_consolidation_master.md` Phase 9 (L636-637)
- Phase 8 prior art: `.plans/378_phase8_pruners_attn_match_absorption.md`
- Crate current state: `crates/katgpt-transformer/` (6 src files, 5 features)
