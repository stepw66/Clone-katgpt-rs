# Plan 385 — Unblock `dense_mesh/node_transformer.rs` by extracting `forward` to `katgpt-forward`

**Started:** 2026-07-05
**Status:** CLOSED
**Branch:** `develop`
**Commit:** `2fefd0aa`

## Motivation

Plan 384 (Phase 14) closed with `dense_mesh/node_transformer.rs` (334 LOC) flagged as the
next-session candidate for re-audit. The documented blocker was
`crate::transformer::forward` (the cognitive-primitive composer). Re-audit confirms:

- **All leaf-resident EXCEPT `forward` itself.** `ForwardContext` (katgpt-forward),
  `MultiLayerKVCache` / `TransformerWeights` (katgpt-transformer), `Config` / `Rng`
  (katgpt-core), `DenseNode` / `DenseHidden` / `MeshScratch` (katgpt-transformer) —
  all leaves. The only blocker is the `forward` function (and its private dispatchers
  `forward_base` / `forward_coda`), which live in root.

- **`forward` + `forward_base` + `forward_coda` are themselves fully leaf-dependent.**
  Verified by line-range grep: their only `crate::` references are to `crate::types::*`
  (= `pub use katgpt_core::types::*`) and `crate::pruners::*`
  (= `pub use katgpt_pruners::*`). They use `katgpt_core::simd::*` for kernels, plus
  root-local helpers `attention_head` / `standard_lm_head` / `clustered_lm_head`
  (which themselves only call `katgpt_core::simd::*` + `matmul_parallel` from katgpt-core).

- **Cycle claim was real but trivially breakable.** The cycle was:
  `katgpt-transformer → root (for forward) → katgpt-transformer`. User hint
  ("feel free to create new primitive or shared crate if it fix cyclic redundant")
  applies: extract `forward` + its dispatch tree into `katgpt-forward` (which already
  depends on `katgpt-transformer` + `katgpt-pruners` + `katgpt-core`), dissolving the
  cycle. `node_transformer.rs` then moves to `katgpt-forward` (the only viable home —
  it can't live in katgpt-transformer because that would re-create the cycle via the
  `katgpt-forward → katgpt-pruners → katgpt-transformer` chain).

## Tasks

- [x] **T1** — Create `crates/katgpt-forward/src/forward.rs` with extracted functions:
  `forward`, `forward_base`, `forward_coda` (the trio), plus the helpers they depend on
  that are also used by other root forward variants:
  `attention_head`, `standard_lm_head`, `clustered_lm_head`, `select_topk_indices`,
  `select_topk_indices_into_buf`, `cluster_map_round_robin`,
  `cluster_map_from_embeddings`. All become `pub` in katgpt-forward.
- [x] **T2** — Update `katgpt-forward/Cargo.toml`: added `domain_latent`, `kog_cpu_fusion`,
  `kog_cpu_fusion = ["katgpt-transformer/kog_cpu_fusion"]` features (forward trio reads
  these via `#[cfg(feature = ...)]`).
- [x] **T3** — Update `katgpt-forward/src/lib.rs`: declared `pub mod forward;` +
  `pub use forward::*;` so root can `use katgpt_forward::forward_base;` etc.
- [x] **T4** — In `src/transformer.rs`: deleted moved code (1057 LOC), replaced with
  `pub use katgpt_forward::forward;` (for callers) and
  `use katgpt_forward::{attention_head, forward_base, forward_coda, standard_lm_head,
  clustered_lm_head, select_topk_indices, select_topk_indices_into_buf,
  cluster_map_round_robin, cluster_map_from_embeddings};` (for root-local forward
  variants that still call them). Test fns inside `transformer.rs` that called the
  helpers directly continue to resolve via the `use` import.
- [x] **T5** — Update root `Cargo.toml` features: `domain_latent` and `kog_cpu_fusion`
  forward to `katgpt-forward` (so the cfg-gated code paths in katgpt-forward compile
  when root enables these features).
- [x] **T6** — Moved `src/dense_mesh/node_transformer.rs` →
  `crates/katgpt-forward/src/dense_mesh_node_transformer.rs`. Update imports:
  `crate::transformer::{forward, ForwardContext, MultiLayerKVCache, TransformerWeights}`
  → `crate::{forward::forward, ForwardContext}` (ForwardContext is already in katgpt-forward)
  + `katgpt_transformer::{MultiLayerKVCache, TransformerWeights}`. Same for `Config`/`Rng`
  via `katgpt_core::types::`. The `super::traits::DenseNode` / `super::types::*`
  references become `katgpt_transformer::dense_mesh::{traits::DenseNode, types::*}`.
- [x] **T7** — Updated `src/dense_mesh/mod.rs` shim: re-export from katgpt-forward.
- [x] **T8** — GOAT gate G3 PASS. `cargo check --workspace` clean on default /
  all-features / no-default-features. katgpt-forward lib tests: 12/12 (8 hla_forward
  + 4 dense_mesh_node_transformer) with --all-features. Root lib tests: 769/769
  default, 1249/1249 --all-features (4 fewer than pre-Plan-385 — the dense_mesh
  tests moved with the file). dense_mesh_goat_gates 5/5 PASS via re-export.
  prof_dense_mesh 5/5 PASS via re-export. transformer:: 80/80 PASS.
- [x] **T9** — Updated Proposal 003 with Phase 15 entry + TL;DR + Final src/ state.
- [x] **T10** — Committed on `develop` with `refactor:` prefix (commit `2fefd0aa`).

## Acceptance criteria

- `cargo check --workspace` clean on default / all-features / no-default-features.
- `katgpt-forward` lib tests pass; count reflects the moved tests (forward trio has
  none in transformer.rs that move with it — they call into root testing harness —
  so test count delta is mostly from node_transformer's 4 tests).
- `katgpt-transformer` lib tests pass (unchanged count).
- Root lib tests pass (slight decrease as node_transformer tests move to katgpt-forward).
- `cargo run --example dense_mesh_goat_gates` (if exists) runs end-to-end via re-export.
- `katgpt_rs::dense_mesh::TransformerNode` still resolves.

## Out of scope

- Moving `forward_looped`, `forward_prefill`, `forward_paged`, `forward_quantized`,
  `forward_raven`, `forward_training_free_loop`, `generate*`. They have genuine root
  deps (`crate::hla::ahla_step` is leaf, but `crate::sleep::*`, `crate::gdn2::*`,
  `crate::tf_loop::{anchor_blend, sub_step_damped_euler}` are root shim + bulk leaf).
  Those can move later if/when sleep/gdn2 move.
- The full transformer.rs dissolution (~5K LOC remaining after T1-T4).
