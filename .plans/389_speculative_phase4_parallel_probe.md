# Plan 389 — Speculative Phase 4: parallel_probe unblock

**Started:** 2026-07-05
**Predecessor:** Plan 388 (`fccd8c96`) — katgpt-pruners cycle resolution.
**Branch:** `develop`

## Goal

Move `src/speculative/parallel_probe.rs` (1187 LOC, Plan 133 Parallel-Probe 2D
controller) to `katgpt-speculative/src/parallel_probe.rs`.

Root-only speculative count drops from 14 → 13 files.

## The blocker (and the unblock)

`parallel_probe.rs` line 26 has `use super::verifier::SpeculativeVerifier;`.
`verifier.rs` is itself root-only (it consumes `crate::transformer::forward`,
which is the forward-cycle architectural blocker). So parallel_probe appears
to inherit verifier's blocker.

**But the `SpeculativeVerifier` TRAIT is pure** — its signature uses only
`TransformerWeights` (katgpt-transformer) + `Config`/`Rng` (katgpt-types).
The trait has zero `forward` deps. Only the impls (`SimulatedVerifier`,
`LeviathanVerifier`) need `forward`, and they stay in root's verifier.rs.

## Strategy — host the trait in katgpt-speculative

Plan 388 used "extract to katgpt-core (lowest common ancestor)". This plan
uses a cleaner variant: **host the trait in katgpt-speculative** (the natural
home for spec-decode traits), because:

1. The trait signature needs `TransformerWeights` (from katgpt-transformer),
   and katgpt-core is *below* katgpt-transformer — so katgpt-core cannot host
   the trait without an upward dep.
2. katgpt-speculative already has `katgpt-transformer` as a dev-dep; this plan
   promotes it to an optional regular dep gated by `parallel_probe`.
3. parallel_probe itself lives in katgpt-speculative, so the trait + consumer
   stay together (DRY by construction).

Root's verifier.rs keeps its impls (`SimulatedVerifier`, `LeviathanVerifier`)
and imports the trait via `pub use katgpt_speculative::verifier_trait::*;`
back-compat shim.

## Tasks

- [x] **Phase 4.1 — Host `SpeculativeVerifier` trait in katgpt-speculative**
  - [x] Create `crates/katgpt-speculative/src/verifier_trait.rs` with the trait def.
  - [x] Wire `katgpt-transformer` as optional dep gated by `parallel_probe` feature.
  - [x] Add `pub mod verifier_trait;` + re-export in katgpt-speculative lib.rs.
  - [x] `cargo check -p katgpt-speculative --features parallel_probe` clean (no warnings).
- [x] **Phase 4.2 — Move `parallel_probe.rs` to katgpt-speculative**
  - [x] `git mv src/speculative/parallel_probe.rs crates/katgpt-speculative/src/`
  - [x] Rewrite imports per Plan 388 table (extends).
  - [x] Add `pub mod parallel_probe;` to katgpt-speculative lib.rs (gated).
  - [x] `cargo check -p katgpt-speculative --features parallel_probe` clean.
- [x] **Phase 4.3 — Update root verifier.rs to import the trait**
  - [x] Replace trait def with `pub use katgpt_speculative::verifier_trait::SpeculativeVerifier;`
  - [x] Root impls (`SimulatedVerifier`, `LeviathanVerifier`) use the trait unchanged.
- [x] **Phase 4.4 — Update root mod.rs and Cargo.toml**
  - [x] Replaced `pub mod parallel_probe;` with `pub use katgpt_speculative::parallel_probe;`
  - [x] Root Cargo.toml `parallel_probe` feature already forwards — no change needed.
  - [x] Made `katgpt-transformer` a mandatory dep in katgpt-speculative (was dev-dep)
        because root's verifier.rs is always compiled and references the trait.
  - [x] `cargo check` (default / all-features / no-default) all clean.
- [x] **Phase 4.5 — GOAT Gate G3**
  - `cargo check --workspace` (default / all-features / no-default) — all clean.
  - `cargo test -p katgpt-speculative --lib --all-features` — **1039 passed**
    (up from Plan 388's 1010, +29 from parallel_probe module's tests +
    verifier_trait). Only failure: pre-existing `budget_compat::tests::
    test_effective_tree_budget_entropy_adapts` (verified on baseline `0a0cbc4c`).
  - `cargo test -p katgpt-speculative --lib --features parallel_probe parallel_probe`
    — **26 passed** (full Plan 133 GOAT unit suite).
  - `cargo test --lib` (root, default) — **446 passed** (down from Plan 388's 472
    by exactly 26 = parallel_probe tests moved to leaf; expected, not a regression).
  - `cargo test --lib --all-features` (root) — **874 passed** (down from 900 by 26;
    same reason).
  - `cargo test --test test_133_parallel_probe_ablation` — **1/1 PASS** (GOAT).
  - `cargo build --example tactical_04_parallel` — **builds ✅**.
  - `cargo test -p katgpt-pruners --lib` — **126 passed** (matches Plan 388).
  - `cargo test --test speculative_generator_goat` — **3/3 PASS**.
  - `cargo test --test spec_reconciliation_bench` — **2/2 PASS**.
  - Verifier tests in root (`speculative::verifier::*`) — **13/13 PASS**
    (trait re-export works; impls still compile against it).
  - Doctest failures in `answer_extract.rs` (3) — pre-existing from Plan 386,
    verified on baseline.
- [x] **Phase 4.6 — Docs + commit**
  - [x] Update `.proposals/003_src_consolidation_master.md` Phase 18 → add Phase 19.
  - [x] Mark this plan DONE.

## Import rewrite patterns (extends Phase 16/17/18 table)

| Root pattern | Leaf rewrite |
|---|---|
| `use super::verifier::SpeculativeVerifier` | `use crate::verifier_trait::SpeculativeVerifier` |
| `use crate::transformer::TransformerWeights` | `use katgpt_transformer::TransformerWeights` |
| `use crate::types::{Config, Rng}` | `use katgpt_types::{Config, Rng}` |
| `use super::answer_extract::*` | `use crate::answer_extract::*` |

## Risk

- **Public API break**: `katgpt_rs::speculative::verifier::SpeculativeVerifier`
  is a public path. Must be preserved via the re-export shim.
- **Feature interaction**: `thinking_cot` requires `parallel_probe` (root L395).
  After this change, `parallel_probe` enables katgpt-transformer in
  katgpt-speculative. Need to confirm no surprise cycle (katgpt-transformer
  must NOT depend on katgpt-speculative — verified: it depends only on
  katgpt-core).
