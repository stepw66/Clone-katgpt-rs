# Plan 381 — Proposal 003 Phase 10: katgpt-core absorption

## Goal

Move 11 `src/` items into the `katgpt-core` workspace crate.

| Item | LOC/Size | Files | Root feature gate |
|---|---|---|---|
| `alloc.rs` | 7.3KB | 1 | `#[cfg(debug_assertions)]` (special: `#[global_allocator]` static stays root) |
| `cce/` | ~130KB | 8 (mod + 7 children) | `cce_moderator` (default-ON) |
| `cumprodsum.rs` | 30.6KB | 1 | always-on (`cumprodsum = []` reserved) |
| `llmexec_guard.rs` | 3.9KB | 1 | `llmexec_guard` (default-ON) |
| `memory_soup_lora.rs` | 15.7KB | 1 | `memory_soup_lora` (opt-in) |
| `mux_demux.rs` | 10.9KB | 1 | `mux_demux` (chains to `katgpt-pruners/mux_demux` — tracking alias only, no module in pruners) |
| `salience/` | ~58KB | 4 (mod + gate + pending + types) | `salience_tri_gate` (default-ON) |
| `trigger_gate.rs` | 29.4KB | 1 | always-on (no feature gate) |
| `skill_opt/` | 7 files | 7 | `skill_opt` (opt-in) |
| `ssd_block.rs` | 32.3KB | 1 | `ssd_block` (default-ON) |
| `channel_simd.rs` | 9.5KB | 1 | `channel_simd_align` (opt-in) |

**Total**: ~30 files. Destination: `crates/katgpt-core/src/`.

## Out of scope (already done or exiled)

- `alien_sampler/` — already exiled to `katgpt-deprecated` (Phase 3a). The Phase 10 line in the proposal still lists it but that's a stale reference.
- `sigmoid` hoist — Phase 0 DONE.
- `closure_mining.rs` — Phase 7 DONE (now in `katgpt-core/src/closure/mining.rs`).

## Pre-move audit (DONE via grep)

### Dep landscape (clean)

All module-internal `crate::` imports are **intra-group** (both endpoints move together):

- `cce/{bregman,external_regret,heterogeneous,lp,primal_dual}.rs` → `crate::cce::types::*` and `crate::cce::*` (intra-module)
- `salience/pending.rs` → `crate::salience::types::DelegateToken` (intra-module)
- `ssd_block.rs` → `crate::cumprodsum::{cumprodsum_scalar, segsum}` (both move together)

**No root-glue consumers** (no `crate::transformer`, `crate::sleep`, `crate::thinking_cot`, `crate::speculative`, `crate::pruners`, `crate::gdn2`, `crate::hla`, `crate::forward`, etc.) inside any of the 11 modules.

### External consumers (all via `katgpt_rs::*` re-export — preserved by shims)

- `cce` — 5+ files: `tests/cce_convergence.rs`, `tests/cce_vs_nash.rs`, `benches/heterogeneous_cce.rs`, `examples/cce_demo.rs`
- `salience` — 3 files: `examples/salience_tri_gate_basic.rs`, `examples/salience_tri_gate_batch.rs`, `benches/salience_tri_gate_bench.rs`
- `trigger_gate` — `tests/bench_176_trigger_gate.rs`, `tests/bench_250_breakeven_goat.rs`, `src/breakeven/mod.rs` (intra-root, resolves via shim)
- `alloc` — 4 test files: `tests/bench_271_attn_match_goat.rs`, `tests/bench_280_cs_kv_probe_goat.rs`, `tests/bench_284_clr_goat_g4.rs`, `tests/bench_294_ict_g5.rs`
- `llmexec_guard` — `examples/llmexec_guard_demo.rs`, `src/benchmark/llmexec_guard.rs` (intra-root)
- `cumprodsum`, `ssd_block` — `examples/ssd_demo.rs`
- `memory_soup_lora` — has doc-link `katgpt_rs::memory_soup_lora::import_memory_soup_artifact` (self-ref, may need rewrite)

### `mux_demux` feature chain discovery

Root `Cargo.toml` line 395: `mux_demux = ["katgpt-pruners/mux_demux"]`. Verified: `crates/katgpt-pruners/src/` has **NO** `mux_demux.rs` file — the forward is a tracking alias only (katgpt-pruners has the feature declared but no module behind it). Moving `src/mux_demux.rs` to katgpt-core is correct; root feature becomes `mux_demux = ["katgpt-core/mux_demux", "katgpt-pruners/mux_demux"]` (preserves tracking chain, adds the actual module gate).

### `alloc.rs` special case

`#[global_allocator]` is process-global and MUST be declared in the final binary/library crate (root). The pattern:

- Move `TrackingAllocator` struct + `reset_alloc_stats`/`get_alloc_stats` functions + tests to `katgpt_core::alloc`
- Root `lib.rs` keeps: `#[cfg(debug_assertions)] #[global_allocator] static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;`
- Root `lib.rs` re-export: `#[cfg(debug_assertions)] pub use katgpt_core::alloc;`

katgpt-core gains an always-compiled `alloc` module (no feature gate; the `debug_assertions` gate is at the consumer side in root).

## Tasks

- [x] **T0.** Audit all 11 modules — DONE (see above).
- [x] **T1.** Add features to `crates/katgpt-core/Cargo.toml` (`cce_moderator`, `llmexec_guard`, `memory_soup_lora`, `mux_demux` already exists, `salience_tri_gate`, `skill_opt`, `ssd_block`, `channel_simd_align`). `cumprodsum`, `trigger_gate`, `alloc` are always-on (no feature gate needed). Update katgpt-core `default` to include the default-ON ones (`cce_moderator`, `llmexec_guard`, `ssd_block`, `salience_tri_gate`).
- [x] **T2.** Copy 11 modules to `crates/katgpt-core/src/`. Verified byte-identical (8/11) or intentionally rewritten (memory_soup_lora.rs + salience/pending.rs: `katgpt_core::simd::*` → `crate::simd::*`, doc-link `katgpt_rs::` → `katgpt_core::`).
- [x] **T3.** Imports inside moved files verified clean (intra-group `crate::` refs all move together; no cross-crate leakage).
- [x] **T4.** Wire module declarations in `crates/katgpt-core/src/lib.rs` (L1297-1322: always-on `alloc`/`cumprodsum`/`trigger_gate`; 7 feature-gated modules mirror root feature names; `salience` re-export preserved).
- [x] **T4b.** Self-reference fix: 3 files had `katgpt_core::simd::*` calls (root-crate vocabulary) — rewritten to `crate::simd::*` post-move. Files: `memory_soup_lora.rs` (3 sites), `ssd_block.rs` (3 sites), `channel_simd.rs` (1 site). No remaining `katgpt_core::` code refs (only doc-link + comments, which are correct as-is).
- [x] **T4c.** `trigger_gate.rs` toml dep: `from_toml`/`to_toml` were test-only API (no external caller — verified via grep across `src/`). Gated behind `#[cfg(test)]`; added `toml = "0.8"` to katgpt-core `[dev-dependencies]` so katgpt-core stays leaf-clean for downstream consumers (no `toml` in the non-test dep set).
- [x] **T4d.** `alloc::tests::*` was failing because the test binary had no `#[global_allocator]` installed (counters stayed 0). Added a test-only `#[cfg(all(test, debug_assertions))] #[global_allocator] static TEST_GLOBAL_ALLOC: alloc::TrackingAllocator` at the end of katgpt-core/src/lib.rs. `cfg(test)` means it does not exist when katgpt-core is consumed as a library dep — no double-declare conflict with the root crate's own `#[global_allocator]`. All 1221 katgpt-core lib tests now pass (was 1216 pass + 5 fail).
- [x] **T5.** Rewrite root shims: `src/lib.rs` `pub mod X` → `pub use katgpt_core::X` for the 11 modules. Handle `alloc` special case (keep `#[global_allocator]` static). — DONE (previous agent)
- [x] **T6.** Update root `Cargo.toml` features to forward to katgpt-core (8 features; `cumprodsum`/`trigger_gate`/`alloc` need no forward — always-on). — DONE (previous agent). Verified: `cce_moderator`, `llmexec_guard`, `memory_soup_lora`, `salience_tri_gate`, `skill_opt`, `ssd_block`, `channel_simd_align`, `mux_demux` all forward to `katgpt-core/<feature>`.
- [x] **T7.** Delete original files from `src/` (the 8 single files + 3 directory trees). — DONE: all 11 items removed from `src/` (8 single `.rs` files + `src/skill_opt/` (7 files) + `src/cce/` (8 files) + `src/salience/` (4 files)). Verified via `find_path` glob — zero matches for any of the 11 item names.
- [x] **T8.** GOAT gate G3: `cargo check --workspace --all-features` + default + `--no-default-features`; katgpt-core lib tests; root lib tests; consumer tests (cce/salience/trigger_gate/alloc); clippy. — PASS:
  - **cargo check**: workspace default ✅, all-features ✅, no-default-features ✅ (only 2 pre-existing unused-var warnings in `src/main.rs`).
  - **katgpt-core lib tests**: 1221/1221 pass (includes moved `trigger_gate::tests::*`).
  - **root lib tests**: 1270/1270 pass.
  - **consumer tests**: `bench_176_trigger_gate` 5/5 ✅, `cce_convergence` 4/4 ✅, `cce_vs_nash` 3/3 ✅, `bench_284_clr_goat_g4` (alloc-zero-alloc gate) 1/1 ✅. Pre-existing failures (NOT from Plan 381): `bench_250_breakeven_goat::t1_overhead_per_forward` (debug-build perf gate, passes in release), `bench_271_attn_match_goat::g5_reconstruction_quality` (pre-existing quality gate, fails identically before our changes — pure file move cannot regress reconstruction quality).
  - **examples**: `salience_tri_gate_basic`, `salience_tri_gate_batch`, `cce_demo`, `llmexec_guard_demo`, `ssd_demo` all compile clean.
  - **clippy**: workspace zero warnings, zero errors.
  - **Mid-fix discovery**: `trigger_gate.rs`'s `#[cfg(feature = "rv_gated_routing")]` gate (RvThresholds + rv_tier_boost) was dead post-move because `rv_gated_routing` was root-only. Added `rv_gated_routing = []` feature to katgpt-core + updated root's `rv_gated_routing` forward to include `katgpt-core/rv_gated_routing`. Workspace check then passed.
  - **Mid-fix discovery**: `mux_demux.rs`'s `#[cfg(all(feature = "mux_demux", feature = "rcd_residual"))]` gate was dead post-move (rcd_residual is root-only). Dropped the dead half of the AND-gate → `#[cfg(feature = "mux_demux")]`. Strictly more permissive (fn now available whenever mux_demux is on); no regression risk.
- [x] **T9.** Update `.proposals/003_src_consolidation_master.md` Phase 10 → DONE. — DONE: comprehensive Phase 10 entry added with all 4 mid-fixes documented + GOAT gate G3 results.
- [x] **T10.** Commit on `develop` with `refactor(core):` prefix. — DONE: commit `61f18225` on `develop` (33 files changed: 22 renames + 8 modifies + 1 add + Cargo.lock + plan). Explicit pathspec used to avoid sweeping sibling WIP.

## Deferral protocol

If a module has a true root dep that can't be resolved via shim or import rewrite, defer to Phase 12. Do NOT move half a module — either the whole thing moves cleanly or it stays. Document the blocker.

## Reference

- Proposal: `.proposals/003_src_consolidation_master.md` Phase 10 (L684-686) + destination map (L143-167)
- Phase 9 prior art: `.plans/380_phase9_transformer_absorption.md`
- Crate current state: `crates/katgpt-core/` (77 modules, 60+ features, default list of ~40 features)
