# Issue 126: GDN Tree Verify Speculative Step Integration (T4.3 Architectural Blocker)

**Date:** 2026-07-10
**Plan:** [424 GDN Tree Verification](../.plans/424_gdn_tree_verification_primitive.md) T4.3
**Status:** RESOLVED for pure-GDN2 models (T4.3 shipped). Hybrid QwenDeltaNet (mixed Attention+DeltaNet layers) remains a riir-ai follow-up.

## Problem

Plan 424 T4.3 requires routing GDN layers through `verify_gdn_tree` when
`Config::architecture == QwenDeltaNet` in the speculative decode step. This
is blocked by two architectural gaps:

### Gap 1: Speculative step operates on KV cache, not GDN2 state

The production speculative decode function
(`katgpt-forward/src/step.rs::speculative_step_rollback_with`) takes a
`MultiLayerKVCache` (attention KV cache) and performs snapshot/restore for
rollback. It never touches `MultiLayerGdn2Cache` (the GDN2 recurrent state).

GDN2 models (`forward_gdn2` in `katgpt-attn/src/gdn2/forward.rs`) maintain
their own `MultiLayerGdn2Cache` and run a separate forward path. There is no
speculative decode pipeline that operates on GDN2 recurrent state.

**To unblock:** A GDN2-aware speculative decode path needs to be built. This
is a separate plan — the current speculative infrastructure is
attention-KV-centric.

### Gap 2: QwenDeltaNet forward lives in riir-ai, not katgpt-rs

The `Config::qwen_deltanet()` constructor is in katgpt-types, but the actual
`forward_qwen_deltanet` forward pass lives in
`riir-ai/crates/riir-engine/src/deltanet/forward.rs`. The katgpt-rs tree
verify primitive cannot directly integrate into the riir-ai forward path
without a cross-repo dependency (which violates the 5-repo quintet
architecture — katgpt-rs is the public engine, riir-ai is the private
runtime).

**To unblock:** riir-ai needs to consume `katgpt_core::gdn_tree_verify` +
`katgpt_attn::gdn2::tree_verify_bridge` directly (both are available as
public katgpt-rs APIs). The speculative step integration for QwenDeltaNet
models would happen in riir-ai's deltanet forward module, not in katgpt-rs.

## What IS shipped

### T4.2 ✅ — Bridge adapter

The bridge adapter (`katgpt-attn/src/gdn2/tree_verify_bridge.rs`) is complete:

- `verify_gdn2_tree_layer` — reads S₀ from `MultiLayerGdn2Cache`, verifies a
  draft tree across all heads, returns per-node outputs without rollback
- `commit_gdn2_tree_layer` — commits the accepted path back to the cache
- `gdn2_scalar_alpha` — extracts scalar α from per-channel decay (exact when
  uniform, geometric-mean approximation otherwise)
- `gdn2_layer_is_paper_compatible` — checks exact-verification conditions
- 6 tests passing: scalar extraction, paper-compat detection, chain/branching
  verify matches sequential, commit matches sequential

### T4.3 ✅ — Pure-GDN2 speculative step (commit `e7cceb53`)

`speculative_step_gdn_tree` in `src/speculative/step_gdn_tree.rs` (root crate,
gated `gdn_tree_verify`) ships a GDN2-aware speculative decode path:

- Uses `forward_tree_gdn2` (`katgpt-attn/src/gdn2/tree_forward.rs`) to process
  all tree nodes in one pass via tree verify
- p/q rejection sampling along the best path
- Commits via sequential `forward_gdn2` replay (GDN2 kernel convention)
- Also ships `build_topology_from_tree_nodes` in `katgpt-core/src/gdn_tree_verify/mod.rs`
  (DDTree path-encoded nodes → parent-index topology)
- **Convention note:** tree verify uses paper's decay→read→update with 1/√dₖ
  scaling; GDN2 kernel uses update-then-read without scaling. `forward_tree_gdn2`
  applies √dₖ scale correction. The commit path uses the GDN2 kernel convention
  (sequential replay), keeping subsequent decode steps consistent.

## What remains (riir-ai follow-up)

**T4.3b — Hybrid QwenDeltaNet** (mixed Attention + DeltaNet layers). The
shipped `speculative_step_gdn_tree` handles pure-GDN2 models. Real
QwenDeltaNet configs mix attention and delta-rule layers; routing each layer
type to its respective verifier (KV-rollback for attention, tree-verify for
GDN) in a single speculative step is a riir-ai concern (QwenDeltaNet forward
lives in `riir-ai/crates/riir-engine/src/deltanet/forward.rs`).

riir-ai should consume `katgpt_core::gdn_tree_verify` +
`katgpt_attn::gdn2::tree_verify_bridge` (both public APIs) when it needs
hybrid speculative decode.
