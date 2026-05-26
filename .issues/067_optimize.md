# Performance Issues — katgpt-rs

Auto-generated from optimization audit against `.agent/optimization.md` guidelines.
Items are ordered by priority. Checked items have been fixed.

## ✅ Fixed

- [x] **P0** `katgpt-core/src/simd.rs` L55-75: `is_avx2_fma_available()` cached via `Once` + `AtomicBool`
- [x] **P0** `katgpt-core/src/simd.rs` L162: AVX2 dot uses `_mm256_fmadd_ps` (FMA fused)
- [x] **P0** `octopus/codebook.rs` L312, `turboquant/codebook.rs` L201: `ScalarCodebook::quantize` uses `partition_point` (binary search O(log n))
- [x] **P0** `transformer.rs` L334: `attn_scale` pre-computed in `ForwardContext::new()`
- [x] **P0** `gdn2/types.rs` L89-101: GDN2 scratch buffers pre-allocated in `Gdn2LayerState`
- [x] **P1** `katgpt-core/src/attention.rs` L103: `o_tile` hoisted outside query tile loop, `.fill(0.0)` per iteration
- [x] **P1** `katgpt-core/src/attention.rs` L130: Redundant max loop uses `actual_br` instead of `BR`
- [x] **P1** `katgpt-core/src/attention.rs` L220-230: Fallback matmul loop reordered to `(i, j, d)` for contiguous V access
- [x] **P1** `katgpt-core/src/coda.rs` L63: `sqrt_2_over_pi` replaced with `const` value
- [x] **P2** `katgpt-core/src/coda.rs` L446: RoPE bounds check replaced with `debug_assert` + `unsafe get_unchecked`
- [x] **P2** `katgpt-core/src/traits.rs` L314-326: `avg_action_space_for` rewritten as single-pass (zero alloc)
- [x] **P2** `katgpt-core/src/types.rs` L81: `DeltaRoutingMode` annotated `#[repr(u8)]`
- [x] **P2** `katgpt-core/src/types.rs` L2060: `TaskType` annotated `#[repr(u8)]`
- [x] **P1** `katgpt-core/src/attention.rs` L200: Fallback `scores = vec![0.0; seq_len²]` per call — added `scores_buf` param + `tiled_attention_forward_with_scores`
- [x] **P1** `katgpt-core/src/simd.rs` L1833,1925: NEON/AVX2 ternary remainder — scalar accumulation + single SIMD add
- [x] **P1** `katgpt-core/src/types.rs` L1816-1823: `lora_apply` already uses `simd_dot_f32` per row (was done previously)
- [x] **P1** `transformer.rs` (6 sites): `Vec::new()` in delta routing `source_refs` — `delta_source_indices` + `depth_route_with_indices` in `ForwardContext`
- [x] **P1** `transformer.rs` L1211,1173: `clustered_lm_head` + `select_topk_indices` — `cluster_scores_buf` + `topk_indexed_buf` in `ForwardContext`
- [x] **P1** `transformer.rs` L1108: `kv_group` per head per layer — `kv_group_lut` lookup table in `ForwardContext` (11 sites updated)
- [x] **P1** `dllm.rs` L449-461: 13 `vec!` in `forward_save` — `ForwardSaveContext` with pre-allocated buffers
- [x] **P1** `dllm.rs` L468,500,523: Per-position `vec!` — moved to pre-allocated `x_buf`, `x_proj_buf`, `x_mlp_buf`
- [x] **P1** `dllm.rs` L620-917: Per-position `vec!` in `backward` — `BackwardContext` with pre-allocated scratch buffers
- [x] **P1** `dllm.rs` L826-877: Redundant `d_logits`/`d_hf` recomputation — saved in `d_after_attn_res_saved`, eliminated ~60 lines
- [x] **P1** `ega_attn.rs` L63: `compute_energy_gate` — added `compute_energy_gate_into` variant
- [x] **P1** `ega_attn.rs` L121: `energy_scores` — added `energy_scores_into` variant
- [x] **P1** `dash_attn/routing.rs` L36-82: 3+ Vec per routing call — `RoutingScratch` + `score_blocks_entmax_into`
- [x] **P1** `dash_attn/entmax.rs` L28-29: Alloc + sort per call — `entmax_1p5_into` with reusable buffers
- [x] **P1** `dash_attn/forward.rs` L73: `.to_vec()` in prefill — pass slice directly
- [x] **P1** `dash_attn/forward.rs` L158,163: `.to_vec()` + `.clone()` in decode — use references
- [x] **P1** `dash_attn/chunk_summary.rs` L139-178: 2 alloc per `summarize_chunk` — `summarize_chunk_into` + `mean_pool_keys_into`
- [x] **P1** `dash_attn/chunk_summary.rs` L58: `is_zero_init()` scans all elements — cached `zero_initialized` bool field
- [x] **P1** `octopus/encode.rs` L107-123: `encode_vector` — added `encode_vector_into` variant
- [x] **P1** `octopus/encode.rs` L230-250: `unpack_triplet_indices` — added `unpack_triplet_indices_into` variant
- [x] **P1** `octopus/forward.rs` L219: `dequantize_key()` in maxsim — uses `dequantize_key_into` with pre-allocated key_buf
- [x] **P1** `turboquant/kv_cache.rs` L447-459: Scalar `mat_vec_t_into` — documented SIMD limitation, kept optimized scalar with unsafe
- [x] **P1** `octopus/kv_cache.rs` L449-459: Scalar matmul in dequantize — SIMD dot product for column-wise access + scratch buffer for unpack
- [x] **P2** `katgpt-core/src/types.rs` L390-466: `Config` struct fields reordered by descending alignment
- [x] **P2** `katgpt-core/src/questbench.rs` L413-414: `find_sufficient_set` — pre-allocated scratch buffers
- [x] **P2** `katgpt-core/src/questbench.rs` L447: `count_valid_extensions` — `count_valid_extensions_with` avoids `to_vec()` per sort
- [x] **P2** `katgpt-core/src/questbench.rs` L578: `NarrowingPruner::is_valid` — `Vec<Vec<usize>>` indexed by token (O(1) lookup)
- [x] **P2** 4 cache types `reset()`: KVCache → no-op; TurboQuant/Octopus → `max_used_pos` tracker; MultiLayer → `fill_pos` metadata
- [x] **P2** `tf_loop.rs` L136-160: O(N×D) scan — O(1) per-layer via `cache.fill_pos()`
- [x] **P2** `rerank.rs` L179,192: 2 Vec per doc — `cosine_rerank_score_into` with scratch buffers
- [x] **P2** `feedback.rs` L47: `thread::spawn` per call — `OnceLock<Sender>` + worker thread

## Remaining Issues

### 🟢 P2 — Lower Impact / Infrastructure

- [ ] `dash_attn/chunk_summary.rs` L73: `Vec<Vec<Vec<f32>>>` triple indirection — flatten to `Vec<f32>` with stride arithmetic
- [ ] `katgpt-core/src/simd.rs` L1956: `simd_ternary_matmul_batch` lacks parallelism — consider rayon for large batches
