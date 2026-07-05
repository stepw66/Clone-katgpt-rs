# Plan 404 — Data Probe Substrate Extraction to katgpt-core

**Status:** DONE (committed locally, not pushed)
**Branch:** `develop`
**Parent:** Proposal 003 (Master `src/` Consolidation)
**Predecessor:** Plan 403 (`d8c34206` — denoise-loop cluster extraction, CLOSED)

## (1) Task

Continue the master `src/` consolidation. Plan 403's handoff named the
**Proposal 003 endgame audit** as the natural next step:

> The denoise cluster was the last large pure-inference cluster in
> `src/dllm.rs`. What remains (~3008 LOC) is predominantly training code …
> not extractable to katgpt-forward without crossing the train/infer boundary.
> A Proposal 003 endgame audit would be the natural next step — quantify how
> much root remains, classify what's extractable vs permanently root-resident,
> and decide whether the consolidation is essentially complete or whether
> another plan-sized cluster exists.

### Endgame audit (the R296-class "verify the blocker" lesson, applied repo-wide)

Surveying all remaining `src/` files classified each into:

| Class | Files | Verdict |
|---|---|---|
| **Pure re-export shim** (DONE) | `distill/mod.rs`, `dense_mesh/mod.rs`, `speculative/{d2f,d2f_verifier,diffusion_sampler,flashar_anchor,flashar_consensus,set_diffusion}.rs` | Already extracted by Plans 398–403. Nothing to do. |
| **Forward-coupled glue** (BLOCKED) | `transformer.rs`, `speculative/{types,step_paged,prefill,dd_tree}.rs`, `speculative/ppot/`, `pruners/bomber/`, `sleep/`, `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs`, backend files | Need `crate::transformer::{ForwardContext, forward, forward_paged}`. Root-resident until katgpt-transformer grows a forward module. |
| **Training infrastructure** (ROOT-RESIDENT) | `dllm.rs` (3008 LOC) | Crosses train/infer boundary. Per modelless mandate, stays root. |
| **Tooling** (ROOT-RESIDENT) | `benchmark/`, `plot.rs` | Depends on everything; tooling, not library. |
| **Pure substrate, extractable** | **`data_probe/`** (6 modules, ~1644 LOC) | Only deps: katgpt-core + intra-module. **Plan 404 target.** |

#### The "dllm-cycle defer until dllm moves" blocker DISSOLVED

Phase 18 documented:

> **dllm-cycle** (~6K LOC): d2f, d2f_verifier, diffusion_sampler,
> flashar_anchor, flashar_consensus, set_diffusion — defer until dllm moves.

Plans 399–403 moved all of those to katgpt-forward. The root files with those
names are now thin re-export shims + training-bridge code. **The blocker is
gone; the deferred work is complete.** This is the same R296-class lesson as
Plan 403's `dllm_solver` investigation: inherited "BLOCKED" claims go stale.

### The data_probe candidate

`src/data_probe/` (7 modules + mod.rs, ~2384 LOC total) is the only remaining
plan-sized pure-substrate cluster in `src/`. Its dependency profile:

| Module | LOC | Deps | Extractable? |
|---|---:|---|---|
| `markov.rs` | 279 | none (pure stdlib) | ✅ always-on |
| `nll.rs` | 146 | `super::markov` (intra) | ✅ always-on |
| `typical_set.rs` | 267 | `super::{markov,nll}` (intra) | ✅ always-on |
| `claim.rs` | 167 | none (pure stdlib) | ✅ always-on |
| `dirichlet_energy.rs` | 214 | `katgpt_core::{dirichlet,spectral_hierarchy}` | ✅ always-on |
| `geometry.rs` | 565 | `super::sink_classify` (intra) | ✅ gated `sink_aware_attn` |
| `sink_classify.rs` | 657 | re-export shim from `katgpt_core::data_probe` | already extracted |

**Natural home: katgpt-core.** katgpt-core already hosts the sink-aware
classifier half (`data_probe.rs`, 1337 LOC, gated `sink_aware_attn`). The root
modules are pure information-theoretic substrate that join naturally.

### Gating design (the one real decision)

katgpt-core's existing `data_probe.rs` is gated `#[cfg(feature =
"sink_aware_attn")]`. The root modules split into:

- **Always-on group:** markov, nll, typical_set, claim, dirichlet_energy —
  pure information theory, no sink dependency.
- **`sink_aware_attn`-gated group:** geometry (depends on sink_classify),
  sink_classify (the existing content).

**Decision: un-gate the katgpt-core `data_probe` parent module** (make it
always-on), gate only `sink_classify` + `geometry` as `sink_aware_attn`
submodules. The parent `mod.rs` re-exports the always-on items unconditionally
and the sink items under `#[cfg(feature = "sink_aware_attn")]`.

This preserves all existing paths:
- `crate::data_probe::SinkKind` (used by `parallax_attn.rs` in katgpt-core) →
  resolves via mod.rs `#[cfg(sink_aware_attn)] pub use sink_classify::*;`
- `crate::data_probe::markov::*` → resolves via mod.rs (always-on)
- Root's `crate::data_probe::*` → resolves via root's re-export shims

The change is strictly more permissive: the module always exists, but sink
items still require the feature. No existing consumer can break.

## (2) Plan

### T1 — Convert katgpt-core `data_probe.rs` → `data_probe/` directory

- `git mv crates/katgpt-core/src/data_probe.rs crates/katgpt-core/src/data_probe/sink_classify.rs`
- Create `crates/katgpt-core/src/data_probe/mod.rs` with:
  - Always-on: `pub mod {markov,nll,typical_set,claim,dirichlet_energy};`
  - Gated: `#[cfg(feature = "sink_aware_attn")] pub mod {sink_classify,geometry};`
  - Re-export blocks mirroring the existing `pub use data_probe::{...}` in lib.rs
    (gated `sink_aware_attn`) so `crate::data_probe::SinkKind` etc. still resolve
    for `parallax_attn.rs`.

### T2 — Move root modules to katgpt-core

- `git mv src/data_probe/{markov,nll,typical_set,claim,dirichlet_energy,geometry}.rs`
  → `crates/katgpt-core/src/data_probe/`
- Import rewrites:
  - `super::markov` → `crate::data_probe::markov` (or keep `super::` since
    they're now siblings in the same katgpt-core module — actually `super::`
    still works since they're all under `data_probe/`)
  - `super::sink_classify` (in geometry.rs) → unchanged (intra-module)
  - `katgpt_core::dirichlet` / `katgpt_core::spectral_hierarchy`
    (dirichlet_energy.rs) → `crate::dirichlet` / `crate::spectral_hierarchy`
    (now intra-katgpt-core)

### T3 — katgpt-core `lib.rs`

- Change `#[cfg(feature = "sink_aware_attn")] pub mod data_probe;` → `pub mod data_probe;`
  (un-gate the parent).
- The existing `#[cfg(feature = "sink_aware_attn")] pub use data_probe::{...}`
  block stays unchanged (the sink items are still feature-gated inside the
  module).

### T4 — Root `src/data_probe/mod.rs` → re-export shims

Replace `pub mod X;` declarations with `pub use katgpt_core::data_probe::X;`
for all 7 submodules. Preserve the existing re-export block at the bottom.

### T5 — Root `src/data_probe/sink_classify.rs`

Update the re-export path from `katgpt_core::data_probe::{...}` to
`katgpt_core::data_probe::sink_classify::{...}` (the items moved one level
deeper). OR keep it working via the mod.rs re-export in katgpt-core (which
re-exports sink_classify items at the `data_probe::` level). **Prefer the
latter** — no change needed if katgpt-core's mod.rs re-exports at the parent
level.

### T6 — Validate (GOAT gate G3)

- `cargo check --workspace` (default / all-features / no-default) clean
- `cargo test -p katgpt-core --lib` — data_probe tests pass
- `cargo test -p katgpt-rs --lib --features data_probe` — root data_probe tests pass via re-export
- 0 warnings, 0 errors

## (3) Validation (GOAT Gate G3)

- [x] `cargo check --workspace` (default) — clean, 0 warnings (from Plan 404 changes)
- [x] `cargo check --workspace --all-features` — clean
- [x] `cargo check --workspace --no-default-features` — clean
- [x] `cargo test -p katgpt-core --lib` — **1272/1272** PASS (default), **2658/2658** PASS (all-features)
- [x] `cargo test -p katgpt-rs --lib` — **252/252** PASS (default), **571/571** PASS (all-features)
- [x] `cargo test -p katgpt-rs --lib --features "data_probe,sink_aware_attn" data_probe` — **18/18** PASS (sink_classify tests via re-export)
- [x] `cargo test -p katgpt-core --lib --features "sink_aware_attn,dirichlet_energy,spectral_hierarchy" data_probe` — **44/44** PASS (moved tests)
- [x] 3 pre-existing warnings in `src/speculative/prefill.rs` (unused imports, unrelated to Plan 404)

**Test parity:** all moved tests pass in katgpt-core; all root tests pass via re-export.

**Gating fix during execution:**
- `dirichlet_energy.rs` re-exports gated `dirichlet_energy` (upstream `crate::dirichlet` is feature-gated)
- `dirichlet_energy.rs` test module gated `#[cfg(all(test, feature = "dirichlet_energy"))]` (tests exercise the re-exported functions)
- `sink_classify.rs` + `geometry.rs` submodules gated `sink_aware_attn` (unchanged from katgpt-core's original gating)
- Root's flat re-export of `dirichlet_energy` (the function) removed due to name collision with `pub mod dirichlet_energy`; function accessible via `katgpt_rs::data_probe::dirichlet_energy::dirichlet_energy` (mirrors historical structure)

## (4) Tasks

- [x] T1 Convert katgpt-core `data_probe.rs` → `data_probe/` directory
- [x] T2 Move 6 root modules to katgpt-core `data_probe/`
- [x] T3 Un-gate katgpt-core `data_probe` parent module in lib.rs
- [x] T4 Root `src/data_probe/mod.rs` → re-export shims
- [x] T5 Verify sink_classify re-export path (works via katgpt-core mod.rs parent-level re-export)
- [x] T6 Validate (G3 GOAT gate) — all green
- [x] T7 Commit on `develop`

## (5) LOC Impact

| File | Before | After | Delta |
|---|---:|---:|---:|
| `src/data_probe/{markov,nll,typical_set,claim,dirichlet_energy,geometry}.rs` | 1638 | 0 | **-1638** |
| `src/data_probe/mod.rs` | 69 | 72 | +3 (re-export shim) |
| `crates/katgpt-core/src/data_probe.rs` → `data_probe/{mod,sink_classify}.rs` | 1337 | 1440 | +103 (mod.rs + edits) |
| `crates/katgpt-core/src/data_probe/{markov,nll,...}.rs` | 0 | 1644 | +1644 (moved) |

**Net root reduction: -1635 LOC.**

Cumulative root reduction (Plans 399–404): **-4322 LOC**.
