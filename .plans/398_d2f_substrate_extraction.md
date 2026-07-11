# Plan 398 — D2F Inference Substrate Extraction to katgpt-forward

**Status:** DONE — committed `8d688bde` (on `develop`)
**Branch:** `develop`
**Predecessor:** Plan 396 (CLOSED `4f59e97b`) — speculative Phase 10 dd_tree full relocation
**Successor (planned):** Plan 399 — d2f wrapper files (`d2f.rs`, `d2f_verifier.rs`, `diffusion_sampler.rs`) move to katgpt-forward

## (1) Problem

The dllm-cycle cluster (~5.7K LOC across `src/speculative/{d2f,d2f_verifier,diffusion_sampler,set_diffusion}.rs`) cannot move to `katgpt-forward` because every file depends on **`crate::dllm::D2fContext`** + **`crate::dllm::forward_block_causal_with`** + **`crate::dllm::denoising_accuracy`**, all of which live in the root-only 4782-LOC `src/dllm.rs` training-and-inference module.

Plan 393→394 established the cadence: **substrate extraction first, wrapper move next**. This plan executes Phase A only — extract the inference substrate; leave the wrapper moves to Plan 399.

## (2) Substrate Inventory (the moveable inference-only subset of `src/dllm.rs`)

| Symbol | Type | Lines | External deps |
|---|---|---:|---|
| `D2fContext` | `pub struct` + `impl { new, reset, commit }` | ~125 | `Config`, `kv_dim` (both katgpt-types) |
| `attention_forward_safe_into` | private `fn` | ~62 | `katgpt_core::simd` only |
| `forward_block_causal_with` | `pub fn` | ~110 | `D2fContext`, `TransformerWeights`, `Config`, `kv_dim`, `rmsnorm`, `matmul`, `matmul_relu`, `attention_forward_safe_into`, `katgpt_core::simd` |
| `denoising_accuracy` | `pub fn` | ~8 | pure (no deps) |

**Total: ~305 LOC of substrate.** All deps already resolve in katgpt-forward:
- `Config`, `kv_dim` → `katgpt_types` (already a dep)
- `TransformerWeights` → `katgpt_transformer` (already a dep)
- `rmsnorm`, `matmul`, `matmul_relu` → `katgpt_types` (already transitively via katgpt-core)
- `katgpt_core::simd` → `katgpt_core` (already a dep)

## (3) What stays in root `src/dllm.rs`

- **All training code**: `train_mini_dllm`, `train_mini_set_causal`, `train_mini_dllm_adaptive`, `evaluate_set_causal_nelbo`, `evaluate_accuracy`, `generate_pattern_dataset`
- **Noise schedules**: `NoiseSchedule`, `AdaptiveNoiseSchedule`, `LossAveraging`
- **Corruption helpers**: `corrupt_block`, `corrupt_block_into`
- **Bidirectional forward**: `forward_bidirectional_positions`, `BidirectionalContext`
- **Block-causal positions (allocating)**: `forward_block_causal_positions` — used by training; not a hot-path concern
- **Set-causal forward**: `forward_set_causal_positions`, `PositionOffsetSchedule` (the latter is already a re-export from katgpt-core)
- **Constraint denoise loop**: `denoise_loop`, `DenoiseConstraint`, `NoConstraint`, `NoRepeatConstraint`
- **All tests**

These are training-research concerns that belong in root (or eventually riir-train). They are out of scope for this plan.

## (4) Tasks

- [x] **T1**: Create `crates/katgpt-forward/src/d2f_context.rs` with the 4 substrate items ported from root `src/dllm.rs`. Imports rewritten to absolute leaf paths (`katgpt_types::Config`, `katgpt_transformer::TransformerWeights`, `katgpt_core::simd`, `katgpt_types::{kv_dim, matmul, matmul_relu, rmsnorm}`). `attention_forward_safe_into` ships as `pub fn` (workspace-internal; katgpt-forward is `publish = false`) — DRY preserved by single source-of-truth in katgpt-forward.
- [x] **T2**: Add `pub mod d2f_context;` + re-exports (`D2fContext`, `forward_block_causal_with`, `denoising_accuracy`, `attention_forward_safe_into`) to `crates/katgpt-forward/src/lib.rs`. Feature-gate the module behind `dllm` (mirrors root's gate on the same name).
- [x] **T3**: Add `dllm = []` tracking feature to `crates/katgpt-forward/Cargo.toml` `[features]` (empty flag — substrate code itself has no `cfg(feature)` branches beyond the optional `rcd_residual` field).
- [x] **T4**: Add `rcd_residual = ["katgpt-core/rcd_residual"]` tracking feature (gates the 3 RCD fields in `D2fContext`). Mirror root's existing `rcd_residual` forwarding.
- [x] **T5**: Slim root `src/dllm.rs`: replace the 4 substrate items with `pub use katgpt_forward::d2f_context::{D2fContext, forward_block_causal_with, denoising_accuracy};` and delete the bodies. The 4 stay-in-root callers of `attention_forward_safe_into` continue to resolve via the `pub(crate) use katgpt_forward::attention_forward_safe_into;` re-export — no other edits needed.
- [x] **T6**: Update root `Cargo.toml`: forward `dllm` and `rcd_residual` features to `katgpt-forward/<feature>` (mirror the existing katgpt-core / katgpt-speculative forwarding lines).
- [x] **T7**: GOAT gate validation:
  - `cargo check --workspace` clean (default) ✅
  - `cargo check --workspace --all-features` clean (combo check — the `merkle_root` lesson class) ✅
  - `cargo check --workspace --no-default-features` clean ✅
  - `cargo test -p katgpt-forward --lib --all-features`: 162/162 PASS ✅
  - `cargo test -p katgpt-rs --lib --all-features`: 709/709 PASS (identical to Plan 396 — perfect parity) ✅
  - `cargo test -p katgpt-rs --lib` (default): 301/301 PASS ✅
  - `cargo test -p katgpt-rs --lib --all-features -- speculative::d2f speculative::d2f_verifier speculative::diffusion_sampler dllm::`: 81/81 PASS (includes rcd_residual feature exercising D2fContext's gated fields) ✅
  - GOAT `bench_102_tilert_pipeline_goat`: 10/10 PASS ✅ (one flake on first run — `bench_e_stability_profile` CV=2.32 noise; re-run passed cleanly; my changes don't touch any stability-metric code path)
  - GOAT `bench_165_hydra_budget_goat` (hydra_budget + decode_specialize): 1/1 PASS ✅
  - Project-wide diagnostics: 0 errors, 0 warnings ✅
- [x] **T8**: Commit on `develop` per global rules. Use `refactor:` prefix. ✅ (commit `8d688bde`)

## (5) Validation Strategy

The substrate is **byte-for-byte structural move** with `crate::*` → `katgpt_*::*` import rewrites only. No semantic change. The root re-export shim preserves every historical `katgpt_rs::dllm::D2fContext` / `katgpt_rs::dllm::forward_block_causal_with` / `katgpt_rs::dllm::denoising_accuracy` import path.

**Public API preservation** is the gate: if any test or external call site breaks, the move is wrong.

## (6) Risks

1. **`attention_forward_safe_into` is shared by 5 root callers** (revised during execution, originally reported as 1):
   - `forward_bidirectional_positions_into` (stays in root)
   - `forward_save` (stays in root)
   - `forward_block_causal_positions` (stays in root — training variant)
   - `forward_block_causal_with` (MOVES to katgpt-forward)
   - `attention_forward_safe` allocating wrapper (stays in root)
   
   **Resolution**: Move the function to `katgpt-forward::d2f_context` as `pub` (workspace-internal — katgpt-forward is `publish = false`). Root imports it via `use katgpt_forward::attention_forward_safe_into;`. DRY preserved. Originally planned as private; revised to `pub` after grep found 5 callers.
2. **`rcd_residual` feature gate on `D2fContext` fields**: The 3 fields (`residual_embeddings`, `entropy_weights`, `rcd_softmax_scratch`) are conditionally compiled. Must preserve this in the moved struct.
3. **`D2fContext::reset` references the same `#[cfg(feature = "rcd_residual")]`**: must keep the cfg block intact.

## (7) Handoff to Plan 399

Once this plan lands, the wrapper files can move:
- `src/speculative/d2f.rs` → `crates/katgpt-forward/src/d2f.rs` — depends on `crate::dllm::{D2fContext, denoising_accuracy, forward_block_causal_with}` which will become `crate::d2f_context::*`. The training-dependent tests (`train_mini_dllm`, `generate_pattern_dataset`) either stay in root or move to riir-train.
- `src/speculative/d2f_verifier.rs` → `crates/katgpt-forward/src/d2f_verifier.rs` — straightforward after `d2f.rs` moves.
- `src/speculative/diffusion_sampler.rs` → `crates/katgpt-forward/src/diffusion_sampler.rs` — same.

`set_diffusion.rs` is more entangled (consumes `forward_set_causal_positions` which stays in root training code). Plan 399 will evaluate whether it moves or stays.
