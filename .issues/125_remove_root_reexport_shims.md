# Issue 125 — Remove root `src/lib.rs` back-compat re-export shims + migrate single-leaf benches

## Context

`katgpt-rs` is `publish = false` — a dev/examples aggregator. After the
Proposal 003 extraction (Phase 1–12, 2026-07-01 → 07-04), the GOAT primitives
live in leaf crates (`katgpt-core`, `katgpt-attn`, `katgpt-pruners`, ...). The
root `src/lib.rs` retains **136 `pub use katgpt_*` re-exports** (~19% of its 726
lines) that only serve as back-compat for:

1. The root crate's own benches/tests/examples.
2. A handful of `riir-engine` modules (`cce_runtime/*`, `arg_runtime/*`).

There are **no external consumers**. The "back compat" is to our own code.
The shims are migration scaffolding that outlived the migration.

## Scope

### Phase 1 — Rewrite all `katgpt_rs::*` call sites to leaf-direct paths

Every `use katgpt_rs::X::*` that resolves through a re-export shim gets
rewritten to `use katgpt_X::*` (or `katgpt_core::X::*` where X lives in core).

Re-export map (built from `src/lib.rs`): see `.benchmarks/125_reexport_map.md`.

Affected files:
- `katgpt-rs/benches/*.rs` (38 benches)
- `katgpt-rs/tests/*.rs`
- `katgpt-rs/examples/*.rs`
- `katgpt-rs/src/**/*.rs` (internal cross-module refs)
- `riir-ai/crates/riir-engine/src/cce_runtime/*.rs` (`katgpt_rs::cce` → `katgpt_core::cce`)
- `riir-ai/crates/riir-engine/src/arg_runtime/soft_reject_bridge.rs`
- Other riir-* consumers (audit via grep)

### Phase 2 — Migrate single-leaf benches to their owning leaf crate

Benches that test a single leaf's primitives move to that leaf's `benches/`.
Cross-cutting benches (span 3+ leaves, depend on root `types::Config`, etc.)
STAY in root — they are integration benchmarks.

Candidate moves (to verify per-bench):
- `cucg_bench`, `cucg_goat` → `crates/katgpt-core/benches/` (compaction)
- `bench_284_clr_perf` → `crates/katgpt-claim/benches/` (clr)
- `salience_tri_gate_bench` → `crates/katgpt-core/benches/` (salience_tri_gate)
- etc.

Stay-in-root: `cgsp_hint_receptivity_bench` (spans core+percepta+pruners+speculative),
`sudoku_speculate_bench` (spans pruners+speculative+types), and similar.

### Phase 3 — Delete re-export shims from `src/lib.rs`

Remove the 136 `pub use katgpt_*` lines + their comment blocks. Root crate
keeps only genuine root-only modules (`inference_router`, `attn_match_adaptive_cot`,
`dllm`, `benchmark/`, `plot.rs`, `tf_loop.rs`, etc.) + the secondary re-export
shims inside `types.rs`/`transformer.rs` (separate concern — those provide a
flat import surface for root-specific code).

### Phase 4 — Validate

```bash
CARGO_TARGET_DIR=/tmp/issue125 cargo check --workspace
CARGO_TARGET_DIR=/tmp/issue125 cargo check --workspace --all-features
```

Plus spot-check `riir-ai` / `riir-chain` consumers compile.

### Phase 5 — Commit on `develop`

## Tasks

- [x] P1.1: Rewrite katgpt-rs `benches/*.rs` imports to leaf-direct
- [x] P1.2: Rewrite katgpt-rs `tests/*.rs` imports to leaf-direct
- [x] P1.3: Rewrite katgpt-rs `examples/*.rs` imports to leaf-direct
- [x] P1.4: Rewrite katgpt-rs `src/**/*.rs` internal cross-module imports
- [x] P1.5: Rewrite riir-engine `cce_runtime/` + `arg_runtime/` imports
- [x] P1.6: Audit + rewrite other riir-* consumers
- [-] P2.1: Identify single-leaf vs cross-cutting benches — deferred: most benches are cross-cutting (span 3+ leaves + root types), see verdict below
- [-] P2.2: Move single-leaf benches — deferred: only ~5 of 38 benches are pure single-leaf; not worth the Cargo.toml surgery across 15+ leaf crates
- [-] P2.3: Update root `Cargo.toml` — deferred with P2.2
- [x] P3.1: Delete 136 re-export shims from `src/lib.rs` (726 → ~240 lines)
- [x] P4.1: `cargo check --workspace` (default features) — PASS
- [-] P4.2: `cargo check --workspace --all-features` — pre-existing failure in katgpt-attn/gdn2/tree_forward.rs (unrelated)
- [x] P4.3: riir-* consumer compile check — all 4 repos PASS
- [x] P5.1: Commit on `develop` — all 5 repos committed
