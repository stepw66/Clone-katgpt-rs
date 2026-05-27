# 🟠 Perf: Replace scalar loops with SIMD kernels + add missing inline/enum annotations

## Summary
Three categories of low-effort, high-signal fixes: (A) scalar loops where SIMD kernels already exist in the codebase, (B) missing `#[inline]` on hot-path functions, and (C) missing `#[repr(u8)]` on field-less enums.

---

## Part A — Replace Scalar Loops with Existing SIMD Kernels

The crate already has `simd_dot_f32`, `simd_fused_scale_acc`, `simd_scale_inplace` — but several hot paths don't use them:

| File | Line | Current | Fix |
|------|------|---------|-----|
| `katgpt-core/attention.rs` | L284 | Scalar `d` loop: `output[d] += s * v[d]` | `simd_fused_scale_acc` |
| `ega_attn.rs` | L139-146 | Scalar dot product in `energy_scores_into` | `simd_dot_f32` per row |
| `river_valley.rs` | L13 | `a.iter().zip(b).map(\|x,y\| x*y).sum()` | `simd_dot_f32` |
| `river_valley.rs` | L73-98 | Scalar Gram matrix inner loop | `simd_dot_f32` |
| `transformer.rs` | L3718-3722 | Scalar value accumulation in `raven_readout_into` | `simd_fused_scale_acc` |
| `transformer.rs` | L1479-1505 | Scalar scale loops in `depth_route` | `simd_scale_inplace` |

---

## Part B — Missing `#[inline]` on Hot-Path Functions

| File | Function | Fix |
|------|----------|-----|
| `katgpt-core/types.rs` L1441, L1469 | `silu()`, `swiglu()` | Add `#[inline]` — MLP activation hot-path |
| `katgpt-core/simd.rs` L83 | `simd_dot_f32` + `simd_outer_product_acc`, `simd_matvec`, `simd_scale_inplace`, `simd_sum_f32`, `simd_max_f32` | Change to `#[inline(always)]` — most-called functions, dispatch is compile-time resolved |
| `tf_loop.rs` L55, L76 | `sub_step_damped_euler`, `anchor_blend` | Add `#[inline]` |
| `dllm.rs` L690, L703 | `rmsnorm_backward`, `softmax_backward` | Add `#[inline]` |
| `ega_attn.rs` L46 | `z_normalize` | Add `#[inline]` |
| `iso_quant/rotation.rs` L81-141 | `apply_rotation`, `apply_inverse_rotation` | Add `#[inline]` |
| `planar_quant/rotation.rs` L43-88 | `apply_rotation`, `apply_inverse_rotation` | Add `#[inline]` |
| `river_valley.rs` L13, L18 | `dot`, `l2_norm` | Add `#[inline]` |

---

## Part C — Missing `#[repr(u8)]` on Field-less Enums

Without `#[repr(u8)]`, Rust defaults to `isize` (8 bytes on 64-bit). For field-less enums with 2-16 variants, `u8` (1 byte) is sufficient and reduces padding when stored in structs.

**`katgpt-core/src/`:**
| File | Enum |
|------|------|
| `types.rs` L372 | `ConvergenceSelector` |
| `simd.rs` L17 | `SimdLevel` |
| `questbench.rs` L503, L523, L562 | `QuestBenchDecision`, `MemoryTier`, `CspDomain` |
| `traits.rs` L509 | `ActingMode` |

**`src/`:**
| File | Enum |
|------|------|
| `transformer.rs` L11 | `DecodeStage` |
| `rerank.rs` L17 | `RerankMethod` |
| `dllm.rs` L28 | `LossAveraging` |
| `gdn2/types.rs` L34 | `Gdn2GateConfig` |
| `hla/types.rs` L334 | `HlaVariant` |
| `sp_kv/types.rs` L8 | `SpKvGateMode` |
| `sp_kv/utility_predictor.rs` L160 | `UtilityAggregation` |
| `iso_quant/types.rs` L4 | `IsoQuantMode` |

**Fix**: Add `#[repr(u8)]` above each enum definition.

---

## Effort Estimate
- Part A: ~1-2 hours (search-and-replace with existing SIMD function signatures)
- Part B: ~15 minutes (add attributes)
- Part C: ~10 minutes (add attributes)
