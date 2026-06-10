# GOAT Proof 049: MoA — Mixture of Activations, Token-Adaptive FFN (Plan 158)

> **Date:** 2026-05-29
> **Feature Gate:** `moa_inference` (root, pulls `coda_fusion`) → `katgpt-core/moa_inference`
> **Depends on:** Plan 103 (CODA fused SIMD kernels), Plan 158 (`MoaActivation`, `MoaConfig`, `compute_moa_gates`, `moa_swiglu`, `simd_matmul_rmsnorm_moa_swiglu`)
> **Research:** 126 — "More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations" (arXiv:2605.26647, ByteDance Seed + PKU)

## Summary

GOAT proof for MoA inference — a token-adaptive replacement for the single fixed activation in
FFN layers. A lightweight sigmoid gate `π_k(x) = σ(u_kᵀx)` mixes a fixed dictionary of 7
activations `{Id, ReLU, ReLU², LeakyReLU, GELU, SiLU, Tanh}`; the same W₁/W₂/W₃ projections are
shared, so only **O(d·|K|) ≈ 28d** extra gating parameters are added (negligible vs the O(d²)
matmul). Research 126 proves a strict expressivity hierarchy **fixed ⊊ LA ⊊ MoA** and reports
consistent loss reduction across 0.12B–2B dense/MoE models at 1.03–1.13× wall-clock, unchanged
memory.

Core result: **10/10 MoA unit tests passing**, including the bi-MoA SwiGLU elementwise reference
and the fused SIMD kernel. During this proof a **sign error in the test reference**
(`test_moa_swiglu_correctness_elementwise`) was found and fixed — the implementation was already
correct; the hand-computed expected value for a negative gate under the identity activation had an
extra negation.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Dictionary | `{Id, ReLU, ReLU², LeakyReLU, GELU, SiLU, Tanh}` (7 entries) |
| Gate | `π_k(x) = sigmoid(u_kᵀx)`, bi-MoA on both SwiGLU branches |
| Forward | `moa_swiglu` + fused `simd_matmul_rmsnorm_moa_swiglu` |
| Build | Release (`--release`, core crate) |
| Platform | macOS (aarch64) |

## GOAT Proof Results

### G1: Activation Dictionary

**Claim:** All 7 dictionary activations evaluate correctly; dictionary size is fixed at 7.

| Test | Result |
|------|--------|
| `test_moa_activation_values` | ✅ Id/ReLU/ReLU²/LeakyReLU/GELU/SiLU/Tanh exact |
| `test_moa_activation_dict_size` | ✅ `MOA_DICT_SIZE == 7` |

### G2: Gating

**Claim:** `compute_moa_gates` produces `σ(u_kᵀx)` per entry — uniform under zero weights, biased under skewed weights.

| Test | Result |
|------|--------|
| `test_moa_config_new` | ✅ |
| `test_compute_moa_gates_uniform` | ✅ (all gates ≈ 0.5) |
| `test_compute_moa_gates_biased` | ✅ |

### G3: bi-MoA SwiGLU Forward

**Claim:** `moa_swiglu` computes `[Σ_k ρ_k σ_k(W₁x)] ⊙ [Σ_ℓ π_ℓ σ_ℓ(W₂x)]` correctly and degrades to plain SiLU-SwiGLU when only the SiLU entry is gated on.

| Test | Result |
|------|--------|
| `test_moa_swiglu_uniform_gates` | ✅ |
| `test_moa_swiglu_fallback_to_silu` | ✅ |
| `test_moa_vs_standard_silu_equivalence` | ✅ |
| `test_moa_swiglu_correctness_elementwise` | ✅ **(fixed test sign error 2026-05-29)** |

### G4: Fused SIMD Kernel

**Claim:** `simd_matmul_rmsnorm_moa_swiglu` (matmul + delayed RMS scale + MoA mixing in one pass) matches the unfused reference.

| Test | Result |
|------|--------|
| `test_simd_matmul_rmsnorm_moa_swiglu_correctness` | ✅ |

## GOAT Gate Summary

| # | Proof | Gate | Result |
|---|-------|------|--------|
| G1 | Activation dictionary | 7 activations exact, size fixed | ✅ PASS |
| G2 | Gating | `σ(u_kᵀx)` uniform/biased | ✅ PASS |
| G3 | bi-MoA SwiGLU forward | elementwise + SiLU fallback | ✅ PASS |
| G4 | Fused SIMD kernel | matches unfused reference | ✅ PASS |

**Overall: 10/10 unit tests PASS.**

## Commands to Reproduce

```bash
# Run all MoA proof tests (coda_fusion is pulled transitively by moa_inference at root)
cargo test --release -p katgpt-core --features "coda_fusion,moa_inference" moa -- --nocapture

# Verify builds without feature
cargo check -p katgpt-core
```

## Key Findings

1. **Expressivity hierarchy holds in code** — bi-MoA represents input-dependent activation mixes a
   fixed-activation FFN cannot (Research 126 Thm 4.1/4.2); the elementwise test pins the exact
   `Σρσ(y) ⊙ Σπσ(z)` form.
2. **Overhead is negligible by construction** — O(28d) gating vs O(d²) matmul; the fused kernel
   folds mixing into the existing matmul + RMSNorm pass (no extra memory).
3. **Sigmoid gate, not softmax** — matches the paper's finding that `σ(0)=0.5` gives better
   optimization dynamics than softmax/tanh gating.
4. **Test bug, not impl bug** — the elementwise reference had a sign error for a negative gate under
   identity; fixing the test (not the kernel) turned MoA green.

## Feature Gate

```toml
# katgpt-core/Cargo.toml
moa_inference = []  # MoA Mixture of Activations — token-adaptive activation mixing (Research 126, Plan 158)

# katgpt-rs/Cargo.toml
moa_inference = ["coda_fusion", "katgpt-core/moa_inference"]
```

**Status:** 10/10 unit tests pass — **default-on**.

## Files

| File | Role |
|------|------|
| `crates/katgpt-core/src/coda.rs` | `MoaActivation`, `MoaConfig`, `compute_moa_gates`, `moa_swiglu`, `simd_matmul_rmsnorm_moa_swiglu` + tests (incl. elementwise sign-error fix) |
| `crates/katgpt-core/src/lib.rs` | Re-exports under the `moa_inference` gate |
| `.benchmarks/049_moa_inference_goat.md` | NEW: this file |

## Related

- `.research/126_MoA_Mixture_of_Activations.md`
- `.benchmarks/030_coda_fusion_simd.md` (fused-kernel baseline)
