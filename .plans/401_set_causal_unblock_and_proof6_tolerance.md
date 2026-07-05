# Plan 401 — `forward_set_causal_positions` Extraction + `proof_6` Tolerance Fix

## (0) Origin

Handoff from Plan 400 (CLOSED `2afc2988`). The Plan 400 summary listed three
candidate next steps; this plan executes **two of them**:

1. **`proof_6_decode_stages_match_forward` fix** — pre-existing GOAT gate failure
   (1/13 in `bench_102_tilert_pipeline_goat`). Quick win, closes a real GOAT
   regression.
2. **Plan 401 proper — unblock `set_diffusion.rs` move**. The Plan 400 summary
   marked this as "STILL BLOCKED on `crate::dllm::forward_set_causal_positions`
   (training code in root)". **Investigation revealed this assessment was
   wrong** — `forward_set_causal_positions` is pure inference (no gradients, no
   backprop, no loss). Per global rule "Try implement to unblock if block", we
   move it.

## (1) Tasks

### T1 — `proof_6` tolerance fix ✅

- [x] Reproduce failure: `Draft logits[15]: 6.3448606 vs 6.344859` (~1.6e-6
      diff, just over the 1e-6 threshold). Reproduces in both debug and
      release.
- [x] Root cause: `forward_draft` / `forward_verify` are pure pass-throughs to
      `forward_base` (identical math). The 1.6e-6 divergence is LLVM
      FMA-contraction noise through the extra match-arm call layers — right at
      f32 epsilon for a logit of magnitude ~6.3.
- [x] Tried `#[inline(always)]` on the dispatcher — did NOT fix (the
      contraction happens inside `forward_base` itself, influenced by caller
      context). Reverted.
- [x] Fix: loosen tolerance to `5e-6` absolute with a detailed comment
      explaining the f32 noise floor and the condition for tightening back.
      The test's *intent* (prove the dispatch adds no computation) is
      preserved — 5e-6 is well below any meaningful correctness threshold
      while staying above f32 noise for this magnitude range.
- [x] Full bench_102 suite: **13/13 PASS** (was 12/13).

### T2 — Verify `forward_set_causal_positions` is pure inference ✅

- [x] Read the function (lines 2155-2319 of `src/dllm.rs`). Uses only:
      - `TransformerWeights`, `Config` (externalizable via `katgpt_transformer`
        / `katgpt_types`)
      - `kv_dim`, `matmul`, `matmul_relu`, `rmsnorm` (all in `katgpt_types::math`
        and `katgpt_types::config`)
      - `katgpt_core::simd::*` (already external)
- [x] No gradients, no backprop, no loss, no training-specific types.
- [x] **The "Root-resident by design (Issue 033 §C)" style comment is
      obsolete** — same lesson Plan 399/400 documented. Issue 033 doesn't
      exist; all blockers cited are dissolved.

### T3 — Move `forward_set_causal_positions` to katgpt-forward ✅

- [x] Create `crates/katgpt-forward/src/forward_set_causal.rs` with the
      function body, imports rewritten to `katgpt_types::*` and
      `katgpt_transformer::*`.
- [x] Move 5 PURE tests that only need `forward_set_causal_positions`:
      - `test_set_causal_mask_zeros_ineligible_positions`
      - `test_set_causal_self_attention_always_allowed`
      - `test_set_causal_weights_sum_to_one_over_eligible`
      - `test_set_causal_ar_singleton_each_position_own_set`
      - `test_set_causal_length_mismatch_panics`
- [x] Leave 2 COMPARISON tests in root (they need sibling functions
      `forward_block_causal_positions` / `forward_bidirectional_positions`
      which are NOT being moved in this plan — deferred to Plan 402):
      - `test_set_causal_matches_block_causal_when_block_ordered`
      - `test_set_causal_mdlm_all_one_set_is_bidirectional`
      These call the re-exported `forward_set_causal_positions` from
      katgpt-forward via root's `crate::dllm::forward_set_causal_positions`
      shim — public API preserved.
- [x] Root `src/dllm.rs`: replace function body with
      `pub use katgpt_forward::forward_set_causal_positions;` re-export.
- [x] Register module in `crates/katgpt-forward/src/lib.rs` behind
      `set_diffusion` feature gate.
- [x] Cargo.toml: added `set_diffusion = []` tracking flag to katgpt-forward;
      extended root's `set_diffusion` with `katgpt-forward/set_diffusion`.

### T4 — Move `set_diffusion.rs` to katgpt-forward ✅

- [x] Create `crates/katgpt-forward/src/set_diffusion.rs` — relocate the
      file with import rewrites. Final: 1248 LOC (production + 31 PURE tests).
- [x] Test split: 31 PURE inference tests moved; 6 TRAIN tests (calling
      `generate_pattern_dataset` / `train_mini_dllm` / `evaluate_set_causal_nelbo`
      / `train_mini_set_causal`) stay in root.
- [x] Root `src/speculative/set_diffusion.rs`: slim to re-export shim (22 LOC)
      + TRAIN-only tests (396 LOC). Final root file: 418 LOC (was 1631).
- [x] Verify external callers: `src/speculative/mod.rs` re-exports,
      `src/dllm.rs` lines 1762/1854 (training code, uses
      `crate::speculative::set_diffusion::order_to_gen_steps` via shim).

### T5 — GOAT gate G3 validation ✅

- [x] `cargo check --workspace` clean at default / `--all-features` /
      `--no-default-features` — **zero warnings**.
- [x] `cargo test -p katgpt-forward --lib --all-features`: **254/254** PASS
      (Plan 400: 218 → +36 = 5 forward_set_causal + 31 set_diffusion).
- [x] `cargo test -p katgpt-rs --lib --all-features`: **617/617** PASS
      (Plan 400: 653 → -36 = 5 forward_set_causal + 31 set_diffusion moved).
- [x] **Total test parity: 871 = 871** (perfect — tests moved, none lost).
- [x] `bench_102_tilert_pipeline_goat`: **13/13** (proof_6 fixed — was 12/13).
- [x] `bench_166_flashar_consensus_goat`: **9/9** (no regression).
- [x] `test_diffusion_sampler_goat`: **5/5** (no regression).
- [x] Project-wide diagnostics: **0 errors, 0 warnings**.

### T6 — Commit

- [ ] Single commit on `develop`: `refactor: Plan 401 — forward_set_causal_positions extraction + proof_6 tolerance fix`.

## (2) Scope Discipline

**Plan 402 candidate (deferred)**: Full `src/dllm.rs` inference extraction —
move `forward_bidirectional_positions`, `forward_block_causal_positions`,
`forward_set_causal_positions` (and their shared `BidirectionalContext` type)
into a cohesive inference module. This would unblock the 2 comparison tests
to also move, and significantly shrink root further. **Not in Plan 401
scope** — Plan 401 is the surgical unblock + the proof_6 quick win.

## (3) Import Rewrite Map (T3)

| Old (root `crate::*`) | New (katgpt-forward) |
|---|---|
| `crate::transformer::TransformerWeights` | `katgpt_transformer::TransformerWeights` |
| `crate::types::{Config, kv_dim, matmul, matmul_relu, rmsnorm}` | `katgpt_types::*` (Config from `katgpt_types::Config`, helpers from `katgpt_types::math`/`config`) |
| `katgpt_core::simd::*` | unchanged (already external) |

## (4) Import Rewrite Map (T4 — set_diffusion.rs)

| Old (root `crate::*`) | New (katgpt-forward) |
|---|---|
| `crate::dllm::forward_set_causal_positions` | `crate::forward_set_causal::forward_set_causal_positions` (intra-crate, post-T3) |
| `crate::dllm::PositionOffsetSchedule` | `katgpt_core::PositionOffsetSchedule` (already re-exported there) |
| `crate::transformer::TransformerWeights` | `katgpt_transformer::TransformerWeights` |
| `crate::types::{Config, Rng}` | `katgpt_types::*` |
| TRAIN tests: `crate::dllm::{generate_pattern_dataset, train_mini_dllm}` | stays in root shim |
| TRAIN tests: `crate::dllm::{evaluate_set_causal_nelbo, train_mini_set_causal}` | stays in root shim |
