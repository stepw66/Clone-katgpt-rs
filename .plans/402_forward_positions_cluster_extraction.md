# Plan 402 — Forward-Positions Cluster Extraction to katgpt-forward

## (0) Origin

Handoff from Plan 401 (CLOSED `ed52538a`). Plan 401's stated Plan 402
candidate was:

> Full `src/dllm.rs` inference extraction — move
> `forward_bidirectional_positions`, `forward_block_causal_positions`,
> and their shared `BidirectionalContext` type into a cohesive inference
> module. This would unblock the 2 comparison tests to also move, and
> significantly shrink root further.

This plan executes that candidate.

## (1) Investigation findings (pre-scope)

### The `BidirectionalContext` coupling is real but tractable

Unlike `forward_set_causal_positions` (Plan 401 — fully self-contained),
`BidirectionalContext` is **written to** by root's inference code
`denoise_loop_rcd` (fields `rcd_residual_embeddings`, `rcd_active`) and
`denoise_loop_rcd_3sr` (fields `tsr_warm_start_embeddings`, `tsr_active`).
It is also **read** by training code `evaluate_accuracy` (field
`all_logits`) and by every `denoise_loop*` (fields `all_logits`,
`all_attn_weights`).

**Resolution**: the struct moves to katgpt-forward with all fields made
`pub`. Root's `denoise_loop*` / `evaluate_accuracy` continue to construct
and mutate it via the re-exported type — same pattern as
`forward_set_causal_positions`, just with a struct instead of a fn. Rust
`pub` fields are accessible across crates, so root's writes
(`bctx.rcd_residual_embeddings[..] = ...`) compile unchanged.

### `d2f_3sr_warm_start` is NOT yet forwarded to katgpt-forward

The `tsr_*` fields are gated `#[cfg(feature = "d2f_3sr_warm_start")]`. Root
forwards `d2f_3sr_warm_start` to `katgpt-core/d2f_3sr_warm_start` but NOT
to katgpt-forward. Moving the struct requires adding a tracking feature to
katgpt-forward + extending root's feature definition.

### `attention_forward_safe` (allocating wrapper) is dead code

It's `#[allow(dead_code)]`; only the `_into` variant is actually called.
Moving it is purely for cohesion (keeps the attention family together).
It will become `pub` so root can re-export it for API compatibility.

### What STAYS in root (scope discipline)

- **Training code**: `evaluate_accuracy`, `train_mini_dllm`,
  `train_mini_set_causal`, `evaluate_set_causal_nelbo`, `backward`,
  `sgd_update`, `masked_loss*`, `forward_save`, `forward_save_set_causal`,
  `ForwardActivations`, `ForwardSaveContext`, `TrainingGradients`,
  `BackwardContext`, all `*_backward*` helpers.
- **Denoise loops**: `denoise_loop`, `denoise_loop_scheduled`,
  `denoise_loop_rcd`, `denoise_loop_rcd_3sr`, `DenoiseConstraint` trait +
  `NoConstraint` / `NoRepeatConstraint` impls. These are pure inference
  but tightly coupled to root's `crate::dllm_solver::*` helpers
  (`compute_residual`, `interpolate_residual`, `normalized_entropy`).
  Moving them is a separate, larger extraction (Plan 403 candidate) —
  they access the re-exported `BidirectionalContext` via pub fields.
- **Noise schedules**: `NoiseSchedule`, `AdaptiveNoiseSchedule`,
  `LossAveraging` — used by training; small; stay.
- **Corruption helpers**: `corrupt_block`, `corrupt_block_into` — used by
  both training (dataset gen) and inference (denoise callers pre-corrupt);
  small; stay.

## (2) Tasks

### T1 — Create `crates/katgpt-forward/src/forward_positions.rs` ✅

- [x] Move `BidirectionalContext` struct + impl (fields → `pub`,
      `new` → `pub fn`). Final: 567 LOC.
- [x] Move `forward_bidirectional_positions` (pub).
- [x] Move `forward_bidirectional_positions_into` (fn → `pub fn`).
- [x] Move `attention_forward_safe` allocating wrapper (fn → `pub fn`,
      dropped `#[allow(dead_code)]`).
- [x] Move `forward_block_causal_positions` (pub).
- [x] Move the 2 Research 376 T0.2 comparison tests
      (`test_set_causal_matches_block_causal_when_block_ordered`,
      `test_set_causal_mdlm_all_one_set_is_bidirectional`).
- [x] Imports rewritten per the map in §3.

### T2 — katgpt-forward `Cargo.toml`: add `d2f_3sr_warm_start` tracking feature ✅

- [x] Add `d2f_3sr_warm_start = []` empty tracking flag (gates the
      `tsr_*` struct fields + cfg branches).

### T3 — katgpt-forward `lib.rs`: register module + re-exports ✅

- [x] `#[cfg(feature = "dllm")] pub mod forward_positions;`
- [x] `pub use forward_positions::{BidirectionalContext,
      forward_bidirectional_positions, forward_bidirectional_positions_into,
      forward_block_causal_positions, attention_forward_safe};`
      (gated `dllm` to match `d2f_context` gating — the module depends on
      `attention_forward_safe_into` from `d2f_context`).

### T4 — Root `src/dllm.rs`: replace moved code with re-export shims ✅

- [x] Delete `BidirectionalContext` struct + impl.
- [x] Delete `forward_bidirectional_positions`.
- [x] Delete `forward_bidirectional_positions_into`.
- [x] Delete `attention_forward_safe` allocating wrapper.
- [x] Delete `forward_block_causal_positions`.
- [x] Delete the 2 comparison tests from `mod tests`.
- [x] Insert re-export blocks (gated `dllm`):
      `pub use katgpt_forward::forward_positions::{...};` (4 items at the
      original struct location) + `pub use katgpt_forward::forward_positions::
      forward_block_causal_positions;` (at the original fn location).
- [x] Keep `pub(crate) use katgpt_forward::attention_forward_safe_into;`
      (the `_into` variant is still used by `forward_save` which stays in root).

### T5 — Root `Cargo.toml`: forward `d2f_3sr_warm_start` to katgpt-forward ✅

- [x] Extend `d2f_3sr_warm_start` feature with `"katgpt-forward/d2f_3sr_warm_start"`.

### T6 — GOAT gate G3 validation ✅

- [x] `cargo check --workspace` clean at default / `--all-features` /
      `--no-default-features` — **zero warnings**.
- [x] `cargo test -p katgpt-forward --lib --all-features`: **256/256** PASS
      (Plan 401: 254 → +2 comparison tests).
- [x] `cargo test -p katgpt-rs --lib --all-features`: **615/615** PASS
      (Plan 401: 617 → -2 comparison tests).
- [x] **Total test parity: 871 = 871** (perfect — tests moved, none lost).
- [x] `bench_102_tilert_pipeline_goat`: **13/13** (no regression; proof_6
      still green after Plan 401's tolerance fix).
- [x] `bench_166_flashar_consensus_goat`: **9/9** (no regression).
- [x] `test_diffusion_sampler_goat`: **5/5** (no regression).
- [x] Project-wide diagnostics: **0 errors, 0 warnings**.

### T7 — Commit

- [x] Single commit on `develop`:
      `refactor: Plan 402 — forward-positions cluster extraction to katgpt-forward`.

## (3) Import Rewrite Map

| Old (root `crate::*`) | New (katgpt-forward `forward_positions.rs`) |
|---|---|
| `crate::transformer::TransformerWeights` | `katgpt_transformer::TransformerWeights` |
| `crate::types::{Config, kv_dim, matmul, matmul_relu, rmsnorm}` | `katgpt_types::{Config, kv_dim, matmul, matmul_relu, rmsnorm}` |
| `katgpt_core::simd::*` | unchanged |
| `pub(crate) use katgpt_forward::attention_forward_safe_into;` (root L520) | `use crate::d2f_context::attention_forward_safe_into;` (intra-crate) |
| `forward_set_causal_positions` (root shim, used by comparison tests) | `use crate::forward_set_causal::forward_set_causal_positions;` (intra-crate) |

## (4) Root re-export shim (T4)

Inserted near the original `BidirectionalContext` location (replacing
~280 deleted lines), gated `dllm` to match the moved code's feature
dependency:

```rust
// ═══════════════════════════════════════════════════════════════
// Forward-Positions Cluster — re-export from katgpt-forward
// ═══════════════════════════════════════════════════════════════
//
// Plan 402 (2026-07-06): BidirectionalContext, forward_bidirectional_positions,
// forward_bidirectional_positions_into, attention_forward_safe (allocating
// wrapper), and forward_block_causal_positions moved to
// `katgpt_forward::forward_positions`. This module re-exports them so every
// historical `crate::dllm::*` import path (notably the denoise_loop* family,
// evaluate_accuracy training code, and the 2 comparison tests' former home)
// continues to resolve.
//
// The struct's fields are `pub` in katgpt-forward because root's
// `denoise_loop_rcd` / `denoise_loop_rcd_3sr` write directly to the
// cfg-gated `rcd_residual_embeddings` / `tsr_warm_start_embeddings` buffers
// (and the `rcd_active` / `tsr_active` flags) after each commitment phase.
// This mirrors the standard "move type, re-export, leave consumers in root"
// pattern (same as forward_set_causal_positions in Plan 401).
#[cfg(feature = "dllm")]
pub use katgpt_forward::forward_positions::{
    attention_forward_safe, forward_bidirectional_positions, forward_bidirectional_positions_into,
    forward_block_causal_positions, BidirectionalContext,
};
// `BidirectionalContext::new` is the constructor — re-exported via the type's
// inherent impl, accessible directly through the re-exported type.
```

## (5) Scope Discipline — what is NOT in this plan

- **`denoise_loop*` family + `DenoiseConstraint`** — pure inference but
  coupled to `crate::dllm_solver::*`. Plan 403 candidate.
- **Training code** (`backward`, `train_mini_*`, `evaluate_*`,
  `forward_save*`, gradient contexts) — must stay in root per modelless
  mandate (training is a research concern).
- **`NoiseSchedule` / `AdaptiveNoiseSchedule` / `LossAveraging`** — small,
  training-coupled; stay.

## (6) LOC Impact (actual)

| File | Before | After | Delta |
|---|---:|---:|---:|
| `src/dllm.rs` | 4137 | 3659 | **-478** |
| `crates/katgpt-forward/src/forward_positions.rs` | — | 567 | +567 |

Root net reduction: **-478 LOC** (Plan 401: -1558 → cumulative -2036 LOC from
the Plan 399-402 extraction series). katgpt-forward continues to grow as the
inference consolidation target.
