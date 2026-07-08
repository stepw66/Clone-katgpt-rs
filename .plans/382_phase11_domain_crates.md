# Plan 382 — Proposal 003 Phase 11: new domain crates

## Goal

Create 5 new domain crates absorbing the remaining scattered `src/` modules.
A 6th crate (`katgpt-bench`) is **deferred** — its `src/benchmark/` content has
heavy root-glue coupling (calls `crate::transformer::forward`, constructs
`ForwardContext`, etc., self-documented as "Root-resident by design").
Per Phase 12, `main.rs` is deleted and benchmark stays root-side as forward glue.

| Crate | Absorbs | Files | LOC/Size | Cross-domain deps | Status |
|---|---|---|---|---|---|
| `katgpt-band` | `band_conditioner.rs`, `bckvss.rs`, `collider_pruner.rs`, `adaptive_cot_stopper.rs` | 4 | ~2800 LOC | `katgpt-core::{sigmoid, ConstraintPruner, PreservationScorer}` + `fastrand` | doable |
| `katgpt-validator` | `validator/` | 4 | 28KB | `katgpt-core::traits::ConstraintPruner` (already leaf); test-only `katgpt-tokenizer` | doable |
| `katgpt-sparse` | `sparse_task_vector.rs`, `specialist_projection.rs` | 2 | ~1640 LOC | `katgpt-band::ComputeTarget` (cross-crate edge) | doable (after band) |
| `katgpt-claim` | `claim_rubric/`, `clr/` | 15 | ~148KB | `katgpt-core::{traits, simd}` + `blake3` (test) | doable |
| `katgpt-ruliology` | `ruliology/` | 13 | 176KB | `katgpt-pruners::g_zero::delta_absorb::DeltaGatedConfig` (single field read) | doable |
| `katgpt-bench` | `benchmark/`, `plot.rs` | 17 | 220KB+665 LOC | **BLOCKED** — transformer-bound (`forward`, `ForwardContext`) | **DEFER** to Phase 12 |

**Pattern (per Phase 1–10 precedent):**
1. Copy modules byte-identical to `crates/<new>/src/`.
2. Rewrite `crate::` paths that no longer resolve (cross-domain edges →
   `katgpt_X::` paths; intra-group stay as `crate::`).
3. Add `Cargo.toml` with features mirroring root feature names.
4. Add new crate `src/lib.rs` with module declarations.
5. Update root `Cargo.toml`: workspace members + path dep + feature forwards.
6. Update root `src/lib.rs`: `pub mod X;` → `pub use katgpt_X::X;` shim
   (preserves every historical `katgpt_rs::X::*` path).
7. Delete originals from `src/`.
8. GOAT gate G3: workspace cargo check on default / all-features /
   no-default-features; consumer tests pass; clippy clean.

## Pre-extraction audit (DONE via 4 parallel subagents)

Key findings per crate:

### `katgpt-band`
- **Internal coupling graph**: `band_conditioner` is the sink; `bckvss` and
  `collider_pruner` import from it; `adaptive_cot_stopper` is **standalone**
  (lib.rs comment "depends on band_conditioner" is inaccurate — verified zero
  `crate::` imports in the file).
- **Feature name mismatches**: `collider_pruner` is gated by feature
  `collider_consistency`; `adaptive_cot_stopper` by `adaptive_cot_identifiability`.
  Preserve these names in the new crate.
- **External inbound edge**: `src/specialist_projection.rs:44` imports
  `crate::band_conditioner::ComputeTarget` — will become
  `crate::katgpt_band::ComputeTarget` via root shim (back-compat preserved).
- **External crate deps**: `katgpt_core::sigmoid` (all 4);
  `katgpt_core::{ConstraintPruner, PreservationScorer}` traits implemented by
  `collider_pruner`; `fastrand` (bckvss prod, others test).
- **Zero root-glue** leakage.

### `katgpt-validator`
- **External crate deps**: `syn`, `proc-macro2` (feature-gated);
  `katgpt_core::traits::ConstraintPruner` (via root
  `crate::speculative::types::ConstraintPruner` re-export — resolves to leaf).
- **Test-only dep**: `crate::tokenizer::BpeTrainer` — needs
  `katgpt-tokenizer` as `[dev-dependencies]`.
- **Zero root-glue** leakage.

### `katgpt-sparse`
- **Cross-domain edge**: `specialist_projection.rs:43`
  `use crate::band_conditioner::ComputeTarget` → becomes
  `use katgpt_band::ComputeTarget`.
- **Intra-group edge**: `specialist_projection` → `sparse_task_vector`
  (both move together; stays `crate::sparse_task_vector`).
- **Test-only cross-crate**: `sparse_task_vector` `mod gauge_tests` uses
  `crate::gauge_invariant::*` for parity testing — gate behind
  `#[cfg(feature = "gauge_invariant")]` (already gated) and forward to
  `katgpt-spectral` in the new crate's Cargo.toml.
- **Cargo feature dep**: `specialist_projection = ["sparse_task_vector",
  "band_conditioner"]` — becomes
  `specialist_projection = ["sparse_task_vector", "katgpt-band/band_conditioner"]`.

### `katgpt-claim`
- **Two siblings, no internal coupling** (claim_rubric ↔ clr do NOT reference
  each other). The bridge lives in `riir-ai/npc_clr/claim_rubric_bridge.rs`
  (consumer-side).
- **External crate deps**: `katgpt_core::traits::FeatureClass` (claim_rubric);
  `katgpt_core::simd` + `blake3` test-only (clr).
- **Zero root-glue** leakage.

### `katgpt-ruliology`
- **Cross-couple**: single struct field read at `mutation.rs:233`:
  `crate::pruners::g_zero::delta_absorb::DeltaGatedConfig` (the
  `delta_threshold: f32` field). Gated behind `#[cfg(feature = "ruliology")]`.
  → becomes `katgpt_pruners::g_zero::delta_absorb::DeltaGatedConfig`.
- **External crate deps**: `fastrand::Rng` (mutation.rs only).
- **Zero root-glue** leakage.

### `katgpt-bench` (DEFERRED)
- **Root-glue blocker**: 6+ files in `src/benchmark/` construct
  `crate::transformer::ForwardContext` and call `crate::transformer::forward*`.
  Self-documented as "Root-resident by design (Issue 033 §C, Option C)".
- **Coupling to `main.rs`**: 6 call sites in `src/main.rs` — but Phase 12
  deletes `main.rs`, so this coupling dissolves when that lands.
- **Decision**: defer to Phase 12. When `main.rs` is deleted, the benchmark
  surface can either (a) stay in root as forward glue, or (b) move to a
  new crate if `katgpt-transformer` grows a forward module. Per Phase 12
  proposal text, `src/` is allowed to retain forward-glue — benchmark
  qualifies. **Defer, do not attempt in Phase 11.**

## Tasks

- [x] **T0.** Audit (DONE — see above).
- [x] **T1.** `katgpt-band` crate. ✅ DONE 2026-07-04.
  - T1.1 Copy 4 files to `crates/katgpt-band/src/`. ✅
  - T1.2 Imports: zero rewrites needed — `crate::band_conditioner::*` refs
    are intra-group (all 4 modules moved together); `katgpt_core::sigmoid` /
    `katgpt_core::{ConstraintPruner, PreservationScorer}` external refs
    resolve through the new crate's `katgpt-core` dep. ✅
  - T1.3 Add `Cargo.toml` with 4 features mirroring root names. ✅
  - T1.4 Add `src/lib.rs` with module declarations. ✅
  - T1.5 Root `Cargo.toml`: workspace member + path dep + 4 feature forwards
    (incl. preserving `katgpt-core/local_branch_routing` inside
    `collider_consistency`). Updated `specialist_projection` feature's
    `band_conditioner` dep to forward to `katgpt-band/band_conditioner`. ✅
  - T1.6 Root `src/lib.rs`: 4 `pub mod X;` → `pub use katgpt_band::X;` shims. ✅
  - T1.7 Delete 4 originals from `src/`. ✅
  - T1.8 GOAT gate G3 — PASS:
    - workspace `cargo check`: default ✅, all-features ✅, no-default ✅
      (only pre-existing `src/main.rs` warnings).
    - katgpt-band lib tests: 24/24 pass.
    - consumer test `bench_176_trigger_gate`: 5/5 pass.
    - 3 examples (`bckvss_vs_dense`, `cccp_vs_nopruner`, `adaptive_cot_stopping`)
      compile clean via root shims.
    - clippy `katgpt-band --all-features`: zero warnings.
- [x] **T2.** `katgpt-validator` crate. ✅ DONE 2026-07-04.
  - T2.1 Copy `validator/` (4 files) to `crates/katgpt-validator/src/`. ✅
  - T2.2 Import rewrites in `syn_pruner.rs`: `crate::speculative::types::ConstraintPruner` → `katgpt_core::ConstraintPruner`; `crate::tokenizer::{BpeTokenizer, BpeTokenizerImpl, BpeTrainer}` → `katgpt_tokenizer::*` (3 test sites via sed). ✅
  - T2.3 `Cargo.toml`: `validator` feature (back-compat gate, empty), `hoare_pruner` feature (gates `SynPruner::propagate` impl). Deps: `katgpt-core`, `katgpt-tokenizer`, `syn`, `proc-macro2`. ✅
  - T2.4 `src/lib.rs` — dropped inner `#[cfg(feature = "validator")]` gates (crate IS the validator; the feature exists for back-compat). ✅
  - T2.5 Root `Cargo.toml`: workspace member + path dep. `validator` feature forwards to `katgpt-validator/validator`. **Dropped root `syn`/`proc-macro2` optional deps** (now in leaf). Updated `hoare_pruner` feature to forward to `katgpt-validator/hoare_pruner` so the propagate impl compiles when both gates on. ✅
  - T2.6 Root `src/lib.rs`: `pub mod validator` → `pub use katgpt_validator as validator;` shim. ✅
  - T2.7 Delete `src/validator/`. ✅
  - T2.8 GOAT gate G3 — PASS:
    - workspace `cargo check`: default ✅, all-features ✅, no-default ✅
      (only pre-existing `src/main.rs` warnings).
    - katgpt-validator lib tests: 7/7 pass (4 partial_parser + 3 syn_pruner).
    - consumer example `core_01_validator` compiles via root shim.
    - clippy `katgpt-validator --all-features`: zero warnings.
- [x] **T3.** `katgpt-sparse` crate (depends on T1). ✅ DONE 2026-07-04.
  - T3.1 Copy 2 files to `crates/katgpt-sparse/src/`. ✅
  - T3.2 Import rewrites:
    - `specialist_projection.rs:43`: `crate::band_conditioner::ComputeTarget` → `katgpt_band::band_conditioner::ComputeTarget`.
    - `sparse_task_vector.rs:794` (test only): `crate::gauge_invariant::*` → `katgpt_spectral::gauge_invariant::*`.
    - Intra-group `crate::sparse_task_vector::*` (in specialist_projection) stays unchanged.
  - T3.3 `Cargo.toml`: features `sparse_task_vector`, `specialist_projection` (implies sparse_task_vector + `dep:katgpt-band` + `katgpt-band/band_conditioner`), `gauge_invariant`. Deps: `katgpt-core` (always), `katgpt-band` (opt). dev-deps: `katgpt-spectral` (with `gauge_invariant` feature), `fastrand`. ✅
  - T3.4 `src/lib.rs` with module declarations. ✅
  - T3.5 Root `Cargo.toml`: workspace member + path dep + 3 feature forwards (`sparse_task_vector`, `specialist_projection`, `gauge_invariant`). ✅
  - T3.6 Root `src/lib.rs`: 2 shims (`pub use katgpt_sparse::{sparse_task_vector, specialist_projection}`). ✅
  - T3.7 Delete 2 originals from `src/`. ✅
  - T3.8 GOAT gate G3 — PASS:
    - workspace `cargo check`: default ✅, all-features ✅, no-default ✅.
    - katgpt-sparse lib tests: 22/22 default, 29/29 with `--features gauge_invariant`
      (7 additional gauge_invariant parity tests).
    - example `splat_vs_dense_attention` compiles.
    - consumer test `bench_300_tjs_msa_rescue_goat`: 1/1 pass.
    - consumer test `bench_270_gauge_invariant_goat`: 16/17 pass; the 1 failure
      (`t08_throughput_rebalance_256x16`) is pre-existing perf-gate flakiness
      (32.5 μs > 5 μs target). Verified by running on prior commit (HEAD before
      Plan 382) — fails identically (31.7 μs > 5 μs target). NOT caused by
      Plan 382 — pure file move cannot regress perf.
    - clippy `katgpt-sparse --all-features`: zero warnings.
- [x] **T4.** `katgpt-claim` crate. ✅ DONE 2026-07-04.
  - T4.1 Copy `claim_rubric/` (5 files) + `clr/` (10 files) to
    `crates/katgpt-claim/src/`. Byte-identical — zero import rewrites needed
    (all `crate::` refs are intra-group; external `katgpt_core::*` already
    resolves correctly). ✅
  - T4.2 Imports: zero rewrites. ✅
  - T4.3 `Cargo.toml`: features `claim_rubric`, `clr` (both default-ON).
    Deps: `katgpt-core`. dev-deps: `blake3`, `bytemuck` (both test-only,
    used by clr test fixtures for hashing direction-vector tamper-evidence). ✅
  - T4.4 `src/lib.rs` with module declarations. ✅
  - T4.5 Root `Cargo.toml`: workspace member + path dep + 2 feature forwards. ✅
  - T4.6 Root `src/lib.rs`: shims preserving the flat re-export shape
    (module + 20-symbol flat surface for clr; module + 6-symbol flat surface
    for claim_rubric). ✅
  - T4.7 Delete `src/claim_rubric/` + `src/clr/`. ✅
  - T4.8 GOAT gate G3 — PASS:
    - workspace `cargo check`: default ✅, all-features ✅, no-default ✅.
    - katgpt-claim lib tests: 54/54 pass.
    - 4 examples (`clr_minimal`, `clr_brevity_tiebreak`, `clr_learning_potential`,
      `claim_rubric_minimal`) compile via root shims.
    - consumer tests: `bench_284_clr_goat_g4` 1/1, `bench_284_clr_goat` 3/3,
      `claim_rubric_test` 17/17, `bench_307_claim_rubric_goat` 1/1.
    - clippy `katgpt-claim --all-features`: zero warnings.
- [x] **T5.** `katgpt-ruliology` crate. ✅ DONE 2026-07-04.
  - T5.1 Copy `ruliology/` (13 files) to `crates/katgpt-ruliology/src/`. ✅
  - T5.2 Import rewrites:
    - `mutation.rs:12`: `crate::pruners::g_zero::delta_absorb::DeltaGatedConfig` → `katgpt_pruners::g_zero::delta_absorb::DeltaGatedConfig`. Gated behind `#[cfg(feature = "ruliology")]` so the import resolves only when needed (avoids unused-import warning when ruliology is off).
    - All intra-module `crate::ruliology::*` paths rewritten to `crate::*` via sed (the crate root IS ruliology now). 19 sites across 9 files.
    - Doc-link `crate::pruners::*` → `katgpt_pruners::*` (mutation.rs:198).
  - T5.3 `Cargo.toml`: feature `ruliology`. Deps: `katgpt-core`, `katgpt-pruners` (with `g_zero` feature, non-optional — mutation.rs unconditionally imports DeltaGatedConfig, but only the `delta_gated_co_evolve` fn uses it), `fastrand`, `blake3` (prod — behavioral fingerprint hashing in fsm/ca/tm). ✅
  - T5.4 `src/lib.rs` — mod.rs renamed to lib.rs; added `#![allow(unexpected_cfgs)]`. ✅
  - T5.5 Root `Cargo.toml`: workspace member + path dep + feature forward (preserves bandit + katgpt-pruners/ruliology chain). ✅
  - T5.6 Root `src/lib.rs`: `pub mod ruliology` → `pub use katgpt_ruliology as ruliology;` shim. ✅
  - T5.7 Delete `src/ruliology/`. ✅
  - T5.8 GOAT gate G3 — PASS:
    - workspace `cargo check`: default ✅, all-features ✅, no-default ✅.
    - katgpt-ruliology lib tests: 94/94 pass (1 ignored).
    - example `ruliology_demo` compiles via root shim.
    - clippy `katgpt-ruliology --all-features`: zero warnings.
- [x] **T6.** Defer `katgpt-bench`. Documented in proposal Phase 11 entry
  with the blocker (transformer-bound glue: 6+ files in `src/benchmark/`
  construct `crate::transformer::ForwardContext` and call `crate::transformer::forward*`;
  self-documented as "Root-resident by design, Issue 033 §C Option C").
  Also coupled to `main.rs` (6 call sites) — Phase 12 deletes `main.rs`,
  dissolving that coupling. Re-evaluate after Phase 12. ✅
- [x] **T7.** Update `.proposals/003_src_consolidation_master.md` Phase 11 →
  DONE (5 of 6 crates; bench deferred). ✅ (see T7 edit below)
- [x] **T8.** Commit on `develop` with `refactor:` prefix. ✅
  - Commit `f18b128b` on `develop`. 52 files changed: 5 new crate Cargo.toml +
    5 new crate lib.rs + 38 renames (src/* → crates/*/src/*) + root Cargo.toml
    feature forwards + root src/lib.rs shims + Cargo.lock + Plan 382 +
    Proposal 003 Phase 11 entry + 1 delete (validator/mod.rs superseded by
    lib.rs).

## GOAT gate G3 (per crate)

For each crate after wiring:
1. `cargo check --workspace` (default features).
2. `cargo check --workspace --all-features`.
3. `cargo check --workspace --no-default-features`.
4. New crate lib tests pass.
5. Root lib tests pass (shim re-exports resolve).
6. At least one consumer test per moved module passes (from `tests/`,
   `benches/`, or `examples/`).
7. `cargo clippy --workspace` zero warnings.

Use `CARGO_TARGET_DIR=/tmp/plan382` if cargo lock contention with sibling
agents. Remove the dir when done.

## Deferral protocol

If a module has a true root dep that can't be resolved via shim or import
rewrite, defer to Phase 12. Document the blocker. Do NOT move half a module.

## Reference

- Proposal: `.proposals/003_src_consolidation_master.md` Phase 11 (L744-745).
- Phase 10 prior art: `.plans/381_phase10_core_absorption.md`.
- Audit reports: this file's "Pre-extraction audit" section above.
