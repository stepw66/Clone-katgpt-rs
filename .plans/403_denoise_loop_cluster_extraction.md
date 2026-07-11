# Plan 403 — Denoise-Loop Cluster Extraction to katgpt-forward

**Status:** DONE (committed locally, not pushed)
**Branch:** `develop`
**Parent:** Proposal 003 (Master `src/` Consolidation)
**Predecessor:** Plan 402 (`695917c8` — forward-positions cluster extraction, CLOSED)

## (1) Task

Continue the master `src/` consolidation. Plan 402's handoff named this as the
next candidate:

> **Plan 403 — Denoise-loop cluster extraction** (~700 LOC). Pure inference but
> coupled to root's `crate::dllm_solver::*` helpers (`compute_residual`,
> `interpolate_residual`, `normalized_entropy`). Items to move:
> - `denoise_loop`, `denoise_loop_scheduled`, `denoise_loop_rcd`, `denoise_loop_rcd_3sr`
> - `DenoiseConstraint` trait + `NoConstraint` / `NoRepeatConstraint` impls
>
> They now access `BidirectionalContext` via pub fields through the re-export —
> moving them requires also extracting or re-exporting the `dllm_solver` helpers.

### Investigation that refined the scope (the "verify the blocker" lesson)

The inherited "BLOCKED on `crate::dllm_solver::*`" claim was **stale**. The
helpers are NOT in root `src/dllm_solver.rs` (no such file). They live in
**katgpt-core** at `crates/katgpt-core/src/dllm_solver.rs`, gated
`#[cfg(feature = "critical_interval_gate")]` (see katgpt-core's
`src/lib.rs:1320-1321`). Root re-exports them:

```rust
// katgpt-rs/src/lib.rs:133-138
#[cfg(feature = "critical_interval_gate")]
pub use katgpt_core::dllm_solver;
```

So `crate::dllm_solver::*` in root resolves through to `katgpt_core::dllm_solver`.
**katgpt-forward can use the same path directly** — `katgpt_core::dllm_solver::*`
— provided the feature plumbing is in place.

### Feature plumbing requirement

- `katgpt_core::dllm_solver` module is gated `critical_interval_gate`
- Root's `rcd_residual` transitively enables `critical_interval_gate`
  (`rcd_residual = ["critical_interval_gate", ..., "katgpt-forward/rcd_residual"]`)
- katgpt-forward's current `rcd_residual = ["katgpt-core/rcd_residual"]` does
  NOT enable `katgpt-core/critical_interval_gate`
- **Fix:** extend katgpt-forward's `rcd_residual` to also forward
  `katgpt-core/critical_interval_gate`. Then `denoise_loop_rcd` (gated
  `rcd_residual`) and `denoise_loop_rcd_3sr` (gated `d2f_3sr_warm_start`,
  which depends on `rcd_residual`) can both reach
  `katgpt_core::dllm_solver::*`.

### Test scope discipline

The 9 denoise tests (test_denoise_loop_converges, test_constraint_improves_denoising,
test_no_repeat_constraint, test_rcd_*, test_denoise_loop_rcd_3sr_*, test_3sr_*)
all depend on **root-only training helpers** (`train_mini_dllm`,
`generate_pattern_dataset`). They CANNOT move to katgpt-forward without either
moving training infrastructure (out of scope — training stays in root) or
rewriting the tests with synthetic weights (risky — changes test surface).

**Decision: tests stay in root.** They exercise the public API via the
re-exported `denoise_loop` etc., so they remain valid end-to-end tests of the
extracted code. This matches the Plan 399-402 pattern (move production code,
leave consumers in root).

## (2) Plan

### T1 — New file `crates/katgpt-forward/src/denoise_loops.rs`

Module gated `dllm` (mirrors `forward_positions` from Plan 402 — the loops
depend on `BidirectionalContext` + `forward_bidirectional_positions_into`
from that module, which is `dllm`-gated).

Moved production items:
- `DenoiseConstraint` trait (always-on within module)
- `NoConstraint` struct + `impl DenoiseConstraint`
- `NoRepeatConstraint` struct + `impl Default` + `impl NoRepeatConstraint` +
  `impl DenoiseConstraint`
- `denoise_loop` (always-on)
- `denoise_loop_scheduled` (always-on)
- `denoise_loop_rcd` (gated `rcd_residual`)
- `denoise_loop_rcd_3sr` (gated `d2f_3sr_warm_start`)

### T2 — katgpt-forward `Cargo.toml`

Extend `rcd_residual` feature to forward `katgpt-core/critical_interval_gate`:

```toml
rcd_residual = ["katgpt-core/rcd_residual", "katgpt-core/critical_interval_gate"]
```

This is the **only** feature-def change required. `d2f_3sr_warm_start` already
exists as an empty tracking flag (Plan 402) and transitively depends on
`rcd_residual` at the root level.

### T3 — katgpt-forward `lib.rs`

Register module + 7-item re-export:

```rust
#[cfg(feature = "dllm")]
pub mod denoise_loops;

#[cfg(feature = "dllm")]
pub use denoise_loops::{
    denoise_loop, denoise_loop_scheduled, DenoiseConstraint, NoConstraint,
    NoRepeatConstraint,
};
#[cfg(all(feature = "dllm", feature = "rcd_residual"))]
pub use denoise_loops::denoise_loop_rcd;
#[cfg(all(feature = "dllm", feature = "d2f_3sr_warm_start"))]
pub use denoise_loops::denoise_loop_rcd_3sr;
```

### T4 — Root `src/dllm.rs`

Delete moved code (L1801-2486, ~686 LOC). Replace with re-export shims at the
original locations.

### T5 — Import rewrites

| Old (root `crate::*`) | New (katgpt-forward) |
|---|---|
| `crate::dllm_solver::{compute_residual, interpolate_residual, normalized_entropy}` | `katgpt_core::dllm_solver::{compute_residual, interpolate_residual, normalized_entropy}` |
| `crate::dllm_solver::{ThreeStateReuseConfig, classify_transitions, compute_gammas, warm_start_lerp}` | `katgpt_core::dllm_solver::{ThreeStateReuseConfig, classify_transitions, compute_gammas, warm_start_lerp}` |
| `crate::dllm_solver::TransitionType::UnchangedVisible` | `katgpt_core::dllm_solver::TransitionType::UnchangedVisible` |
| `BidirectionalContext::new` | `crate::forward_positions::BidirectionalContext::new` (intra-crate) |
| `forward_bidirectional_positions_into` | `crate::forward_positions::forward_bidirectional_positions_into` (intra-crate) |
| `PositionOffsetSchedule` | `katgpt_core::PositionOffsetSchedule` |
| `TransformerWeights`, `Config`, `Rng` | `katgpt_transformer::TransformerWeights`, `katgpt_types::*`, `katgpt_types::Rng` |
| `katgpt_core::simd::*` | unchanged |

## (3) Validation (GOAT Gate G3)

- [x] `cargo check --workspace` clean (default / `--all-features` / `--no-default-features`) — zero warnings
- [x] `cargo test -p katgpt-forward --lib --all-features` — **256/256** PASS (no new tests moved)
- [x] `cargo test -p katgpt-rs --lib --all-features` — **615/615** PASS (9 denoise tests exercise via re-export)
- [x] Total test parity: **871 = 871**
- [x] `bench_102_tilert_pipeline_goat`: **13/13** (first run had flaky `bench_e_stability_profile` CV=1.17, re-run 13/13 — same flaky timing test noted in Plan 402)
- [x] `bench_166_flashar_consensus_goat`: **9/9**
- [x] `test_diffusion_sampler_goat`: **5/5**
- [x] Diagnostics: **0 errors, 0 warnings**

## (4) Tasks

- [x] T1 Create `crates/katgpt-forward/src/denoise_loops.rs` (722 LOC)
- [x] T2 Extend katgpt-forward `Cargo.toml` `rcd_residual` feature (forward `katgpt-core/critical_interval_gate`)
- [x] T2b Extend katgpt-forward `Cargo.toml` `d2f_3sr_warm_start` feature (forward `katgpt-core/d2f_3sr_warm_start` + imply local `rcd_residual`)
- [x] T3 Register module + re-export in katgpt-forward `lib.rs`
- [x] T4 Delete moved code from root `src/dllm.rs`, add re-export shims (-651 LOC)
- [x] T5 Apply import rewrites in the new file
- [x] T6 Validate (G3 GOAT gate) — all green
- [x] T7 Commit on `develop`

## (5) LOC Impact

| File | Before | After | Delta |
|---|---:|---:|---:|
| `src/dllm.rs` | 3659 | 3008 | **-651** |
| `crates/katgpt-forward/src/denoise_loops.rs` | — | 722 | +722 |

Cumulative root reduction (Plans 399–403): **-2687 LOC**.

## (6) What stays in root

- **All 9 denoise tests** — they use root-only training helpers (`train_mini_dllm`, `generate_pattern_dataset`) and exercise the public API via the re-export.
- **Training infrastructure** — `evaluate_accuracy`, `train_mini_dllm`, `backward`, `sgd_update`, `forward_save*`, gradient contexts, etc.
- **Noise schedules** — `NoiseSchedule`, `AdaptiveNoiseSchedule`, `LossAveraging` (training-coupled, small).
- **Corruption helpers** — `corrupt_block`, `corrupt_block_into` (used by both training and inference).
- **All `replaid_tests`** — adaptive schedule tests (training-dependent).

## (7) Next-session candidate

The denoise cluster was the last large pure-inference cluster in `src/dllm.rs`.
What remains is predominantly training code (not extractable to katgpt-forward
without crossing the train/infer boundary) plus the noise/corruption helpers.

A Proposal 003 endgame audit would be the natural next step — quantify how
much root remains, classify what's extractable vs permanently root-resident,
and decide whether the consolidation is essentially complete or whether
another plan-sized cluster exists.
