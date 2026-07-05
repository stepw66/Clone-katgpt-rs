# Plan 394 — speculative Phase 9: forward-cycle 5-file move to katgpt-forward

Status: **DONE** (2026-07-05). All builds clean (default / all-features /
no-default — zero warnings). 787 root lib tests pass + 84 katgpt-forward lib
tests pass = 871 total (Plan 393 baseline was 859 root-only). 10/10 GOAT
tests pass on bench_102_tilert_pipeline_goat (default), 12/13 with
--features decode_specialize (the 1 failure `proof_6_decode_stages_match_forward`
is a pre-existing floating-point precision issue, present on the Plan 393
baseline unchanged). 1/1 bench_165_hydra_budget_goat with decode_specialize.
Branch: `develop`.
Predecessor: Plan 393 (Phase 8 SpeculativeContext + forward_decode_stage — CLOSED `ba2b366a`).

## Problem

Plan 393 unblocked the **forward-cycle cluster** — 5 root-resident files in
`src/speculative/` that form an internal call cycle rooted on `forward()`:

| File | LOC | Sibling deps (within the 5) |
|---|---:|---|
| `drafter_lora.rs` | 1107 | (none) |
| `dflash.rs` | 1731 | (none) |
| `verifier.rs` | 920 | calls into `dflash`, `drafter_lora` |
| `step.rs` | 1852 | calls into `verifier`, `dflash` |
| `prefill.rs` | 497 | test-only call into `verifier`, `step` |
| **Total** | **6107** | |

After Plan 393, every external dependency of these files resolves to a leaf
(katgpt-core, katgpt-types, katgpt-transformer, katgpt-pruners, katgpt-hla,
katgpt-speculative) OR to katgpt-forward itself (SpeculativeContext,
forward_decode_stage, forward, ForwardContext). The internal cycle dissolves
once all five land in katgpt-forward.

## Strategy: sequential move with build+test after each

The 5 files form a strict DAG inside the cluster. Move them in topological
order so each file's `crate::*` paths resolve to either katgpt-forward-self
or already-moved siblings:

```
drafter_lora  ──┐
                ├──► verifier ──► step
dflash ─────────┘                ──► (prefill test-only)
```

Order: `drafter_lora` + `dflash` first (no sibling deps), then `verifier`
(depends on first two), then `step` (depends on `verifier` + `dflash`),
then `prefill` scorers (only test dep on `verifier`/`step`).

## Items that stay root (split-moves)

| Item | Why it stays root |
|---|---|
| `block_select_entmax` (~73 LOC + 3 tests in `prefill.rs`) | Uses `crate::dash_attn::{entmax_1p5, entmax_support}` which is a re-export of `katgpt-attn`. `katgpt-attn` depends on katgpt-forward — adding the reverse dep creates a cycle. The preflight file's own doc comment anticipated this split (Plan 390). |
| `speculative_step_rollback_paged` + `DDTreeBranchCache` (~70 LOC in step.rs + types.rs) | Uses `PagedKVCache` + `forward_paged` (heavy forward variant, root `transformer.rs:2462`). `forward_paged` has genuine root deps (`crate::sleep::*`, `crate::gdn2::*`, `crate::tf_loop`). `speculative_step_rollback_paged` is `#[deprecated]` and not re-exported by mod.rs. |
| `test_extract_ddtree_paths` (1 test in `dflash.rs`) | Calls `crate::speculative::dd_tree::{build_dd_tree, extract_best_path}` — `dd_tree` is root-only (2556 LOC). Defer this single test (stay in root as a thin smoke-test file). |

## Phase 1 — Move `drafter_lora.rs` (1107 LOC, no sibling deps)

- [x] T1.1 Create `crates/katgpt-forward/src/drafter_lora.rs` with content
      from root.
- [x] T1.2 Rewrite imports:
  - `crate::transformer::{ForwardContext, forward}` → `crate::{ForwardContext, forward}`
  - `crate::transformer::{MultiLayerKVCache, TransformerWeights}` → `katgpt_transformer::{...}`
  - `crate::types::*` → `katgpt_types::*`
- [x] T1.3 Update katgpt-forward `lib.rs`: add `pub mod drafter_lora;` + re-exports.
- [x] T1.4 Root `src/speculative/mod.rs`: replace `pub mod drafter_lora;` with
      `pub use katgpt_forward::drafter_lora;`. Update `pub use drafter_lora::{...}`
      (lines 176-179) to keep working via the re-export.
- [x] T1.5 Delete root `src/speculative/drafter_lora.rs`.
- [x] T1.6 `cargo check --workspace` clean.

## Phase 2 — Move `dflash.rs` (1731 LOC, no sibling deps)

- [x] T2.1 Create `crates/katgpt-forward/src/dflash.rs` with content from root.
      (No test deferred — `dd_tree` reference resolved via `katgpt_speculative::dd_tree`.)
- [x] T2.2 Rewrite imports:
  - `crate::speculative::types::{DraftResult, SpeculativeContext}` → `crate::SpeculativeContext` + `katgpt_core::speculative::types::DraftResult`
  - `crate::transformer::*` → mix of `crate::*` and `katgpt_transformer::*`
  - `crate::types::*` → `katgpt_types::*` (kv_dim gated to dflare_kv_routing)
  - `super::domino_lora::*` → `katgpt_speculative::domino_lora::*`
  - `super::domino::*` → `katgpt_speculative::domino::*`
  - `crate::speculative::types::*` (in-body refs) → `katgpt_core::speculative::types::*`
  - `crate::speculative::dd_tree::*` (in tests) → `katgpt_speculative::dd_tree::*`
- [x] T2.3 Update katgpt-forward `lib.rs`: add `pub mod dflash;` + re-exports
      (including feature-gated `_with_fusion`, `_with_domino`, `_with_routing`).
- [x] T2.4 Root `src/speculative/mod.rs`: replace `pub mod dflash;` with
      `pub use katgpt_forward::dflash;`.
- [x] T2.5 Delete root `src/speculative/dflash.rs`.
- [x] T2.6 `cargo check --workspace` clean (default / all-features).

## Phase 3 — Move `verifier.rs` (920 LOC, depends on dflash + drafter_lora)

- [x] T3.1 Create `crates/katgpt-forward/src/verifier.rs` with content from root.
- [x] T3.2 Rewrite imports:
  - `crate::speculative::dflash::*` → `crate::dflash::*`
  - `crate::speculative::drafter_lora::*` → `crate::drafter_lora::*`
  - `crate::speculative::dd_tree::*` → `katgpt_speculative::dd_tree::*`
  - `crate::speculative::types::*` → mix of `crate::*` and `katgpt_core::*`
  - `crate::speculative::acceptance_forecast` → `katgpt_speculative::acceptance_forecast`
  - `crate::transformer::*` → mix of `crate::*` and `katgpt_transformer::*`
  - `crate::types::kv_dim` → moved to test-only import (production unused)
- [x] T3.3 Update katgpt-forward `lib.rs`: add `pub mod verifier;` + re-exports.
- [x] T3.4 Root `src/speculative/mod.rs`: replace `pub mod verifier;` with
      `pub use katgpt_forward::verifier;`.
- [x] T3.5 `d2f_verifier.rs` + `flashar_consensus.rs` continue to resolve
      `crate::speculative::verifier::SpeculativeVerifier` via the re-export.
- [x] T3.6 Delete root `src/speculative/verifier.rs`.
- [x] T3.7 `cargo check --workspace` clean (default / all-features / no-default).

## Phase 4 — Move `step.rs` (1852 LOC, depends on verifier + dflash)

- [x] T4.1 Create `crates/katgpt-forward/src/step.rs` with content from root
      EXCEPT `speculative_step_rollback_paged` + its 3 tests (deferred to root).
      Extracted via `sed` to remove lines 577-757 (the deprecated paged fn) and
      1346-1481 (its 3 tests).
- [x] T4.2 Rewrite imports (same pattern as Phase 3).
- [x] T4.3 Update katgpt-forward `lib.rs`: add `pub mod step;` + re-exports
      (including feature-gated `_with_router` and `_with_configurator`).
- [x] T4.4 Root `src/speculative/mod.rs`: replace `pub mod step;` with
      `pub use katgpt_forward::step;`. Add `pub mod step_paged;` for the
      deferred deprecated paged fn + tests.
- [x] T4.5 Create `src/speculative/step_paged.rs` with `speculative_step_rollback_paged`
      + its 3 tests. Calls `extract_ddtree_paths` via `crate::speculative::step::extract_ddtree_paths`
      (re-export from katgpt-forward).
- [x] T4.6 Make `extract_ddtree_paths` `pub` in katgpt-forward step.rs so the
      root-side `step_paged.rs` can call it.
- [x] T4.7 Add `log` dep to katgpt-forward (step.rs uses `log::debug!` under
      `stability_metrics`).
- [x] T4.8 `cargo check --workspace` clean (default / all-features / no-default).

## Phase 5 — Move `prefill.rs` scorers (split: scorers move, entmax stays)

- [x] T5.1 Create `crates/katgpt-forward/src/prefill.rs` with:
  - Substrate re-export shim (`pub use katgpt_speculative::prefill::*`)
  - `AttentionScorer` + `PrefillScorer` impl
  - `BlockAttentionScorer` + `PrefillScorer` impl
  - `test_attention_scorer_produces_scores`
- [x] T5.2 Keep `block_select_entmax` + its 3 dash_attn-gated tests + the
      rest-gated bridge test in root `src/speculative/prefill.rs` (slim file).
      Root slim file re-exports the moved symbols:
      `pub use katgpt_forward::prefill::{AttentionScorer, BlockAttentionScorer, ...};`
- [x] T5.3 Rewrite imports in moved file (same pattern as Phase 3/4).
- [x] T5.4 Update katgpt-forward `lib.rs`: add `pub mod prefill;` + re-exports.
- [x] T5.5 Root `src/speculative/mod.rs` keeps `pub mod prefill;` — the slim
      file handles the re-export chain to katgpt-forward internally.
- [x] T5.6 Gate `FlashPrefillConfig` import in root slim prefill.rs behind
      `dash_attn` (only block_select_entmax uses it). Gate `make_draft` test
      helper behind `rest` (only the bridge test uses it).
- [x] T5.7 `cargo check --workspace` clean (default / all-features / no-default).

## Phase 6 — Verification

- [x] T6.1 `cargo check --workspace` (default) clean — NO warnings
- [x] T6.2 `cargo check --workspace --all-features` clean — NO warnings
- [x] T6.3 `cargo check --workspace --no-default-features` clean — NO warnings
- [x] T6.4 `cargo test -p katgpt-forward --lib` (default) — 62/62 PASS
- [x] T6.5 `cargo test -p katgpt-forward --lib --all-features` — 84/84 PASS
- [x] T6.6 `cargo test -p katgpt-rs --lib --all-features` — 787/787 PASS
      (test count dropped from Plan 393's 859 because the moved tests now live
      in katgpt-forward; total across both crates = 871, +12 vs Plan 393)
- [x] T6.7 `cargo test -p katgpt-rs --lib speculative::step_paged` — 3/3 PASS
      (paged-KV tests stay root)
- [x] T6.8 GOAT test: `cargo test -p katgpt-rs --test bench_102_tilert_pipeline_goat`
      — 10/10 PASS (default features)
- [x] T6.9 GOAT test with decode_specialize: 12/13 PASS — the 1 failure
      (`proof_6_decode_stages_match_forward`) is a pre-existing floating-point
      precision issue (`6.3448606 vs 6.344859` — diff ~1e-6) present on the
      Plan 393 baseline unchanged. Not caused by Plan 394.
- [x] T6.10 GOAT test: `cargo test -p katgpt-rs --test bench_165_hydra_budget_goat --features "hydra_budget decode_specialize"` — 1/1 PASS
- [x] T6.11 Plan file updated with final LOC; all tasks marked `[x]`; commit on develop

## GOAT Gate

- G1 correctness: `cargo test` parity — moved tests stay green
- G2 perf: identical code paths, just relocated
- G3 no-regression: workspace builds clean at default / all-features / no-default
- G4 alloc-free: not applicable (substrate relocation only)

## Risks

- **Feature flag plumbing**: 11 new tracking features. Easy to forget one —
  `cargo check --workspace --all-features` will catch missing flags.
- **`DDTreeBranchCache` / `forward_paged` cycle**: stays root, do not move.
- **`block_select_entmax` cycle via katgpt-attn**: stays root.
- **`test_extract_ddtree_paths`**: needs `dd_tree` which is root-only. Defer.
- **Sibling path rewrites**: must be consistent — verifier.rs uses
  `crate::dflash::*` AFTER dflash moves into katgpt-forward. If verifier moves
  BEFORE dflash, the path won't resolve. Strict topological order required.

## Next session (Plan 395 candidate)

With Plan 394 done, the speculative cluster is essentially dissolved. The
remaining root-resident speculative files are:

- `dd_tree.rs` (2556 LOC) — root-only. Has a partial substrate in
  `katgpt_speculative::dd_tree` (re-exported via `pub use`). Could be a future
  extraction target once the remaining `ManifoldValidWrapper` wiring is
  examined (Plan 391).
- `d2f.rs` + `d2f_verifier.rs` + `diffusion_sampler.rs` + `set_diffusion.rs`
  (~5.7K LOC total) — root-only, gated `dllm` / `tri_mode` / `set_diffusion`.
  These are the "dllm-cycle" cluster (Plan 391 stretch goal). Heavily coupled
  to `TransformerWeights` and the root dllm module.
- `flashar_anchor.rs` + `flashar_consensus.rs` (~1.6K LOC) — gated
  `flashar_anchor` / `flashar_consensus`. May be movable to katgpt-forward if
  their deps dissolve.

The highest-value remaining target is likely the **test module relocation**
for `dd_tree.rs` — it's currently 2556 LOC and contains a massive `mod tests`
(~2380 LOC). Extracting the test fixture behind a trait would let the tests
move to katgpt-speculative, dramatically shrinking the root.

Alternative: continue with the `dllm-cycle` cluster. Plan 394 demonstrated the
exact pattern: substrate extraction (Plan 393), wrapper move (Plan 394). The
same cadence would apply to dllm-cycle: identify root-only blockers (likely
`d2f::*` types), extract them to katgpt-forward or a new substrate crate,
then move the wrappers.
