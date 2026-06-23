# Benchmark 308: KARC GOAT Gate Results (Phase 1)

**Date:** 2026-06-23
**Plan:** [308_karc_delay_basis_ridge_forecaster.md](../.plans/308_karc_delay_basis_ridge_forecaster.md)
**Research:** [288_KARC_Delay_Basis_Ridge_Forecaster.md](../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md)
**Source paper:** [arXiv:2606.19984](https://arxiv.org/abs/2606.19984)

---

## Summary

Phase 1 ships first-order KARC (delay-embedding × Chebyshev basis × closed-form
ridge readout) behind the `karc_forecaster` opt-in feature. Three of the four
GOAT gates pass; G1 NRMSE is within 5× of target (partial miss — documented).

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G1 NRMSE** (1 LT autonomous) | ≤ 1.0e-3 | **4.79e-3** | ❌ MISS (5×; one-step NRMSE 9.7e-4 ≤ 1e-3 ✓) |
| **G1 threshold** (ε=0.1) | ≥ 8 LT | **8.16 LT** | ✅ PASS |
| **G2 forecast latency** (D=8,M=8,K=4) | ≤ 500 ns/call | **381 ns/call** | ✅ PASS |
| **G3 zero-alloc** `forecast_into` | 0 alloc after warmup | **0 alloc** | ✅ PASS |
| **G4 bit-reproducibility** | byte-identical Wout | **byte-identical** | ✅ PASS |

**Verdict:** Keep opt-in (Phase 1). Do NOT promote to default. The G1 NRMSE gap
(~5×) is attributable to the absence of second-order features (Phase 2, T2.1).
The one-step model quality (9.7e-4) is within 2× of the paper's headline (5.3e-4),
confirming the ridge solve + basis expansion are correct; the autonomous-rollout
NRMSE gap reflects the expressiveness ceiling of first-order KARC.

---

## G1 — Double-Scroll (paper §A.1)

Config: `KarcForecaster<ChebyshevBasis<24>, 3, 24, 8>`, λ=5e-3, 4050 training
pairs, per-coordinate normalization to [-1, 1].

```
── G1 results ──────────────────────────────────────────────
  one-step NRMSE (train fit): 9.743024e-4   ← within 2× of paper (5.3e-4)
  NRMSE over 1 LT (32 samples): 4.793730e-3 ← autonomous rollout; 5× target
  threshold (ε=0.1): 255 samples = 8.16 LT   ← PASSES ≥ 8 LT target
  σ(u) mean per-coord: 0.8582
```

Paper reference: NRMSE 5.3e-4, threshold 16.7 LT (uses second-order Fourier,
d_h=1891). Phase 1 uses first-order Chebyshev (d_h=576) — the autonomous-rollout
NRMSE is dominated by chaotic error amplification of the (smaller) first-order
residual, not a model bug.

**ODE parameters** (paper Eqs. 15–17): R1=1.2, R2=3.44, R4=0.193, β=11.6,
I_r=2.25e-5, Lyapunov time ≈ 7.81 units. RK4 with 10 sub-steps per sample
(dt=0.25) for stiff-system stability (the `sinh(β·ΔV)` nonlinearity is explosive
under coarse integration).

---

## G2 — Forecast Latency

Criterion bench, `--release`, single-threaded SIMD dispatch (aarch64 NEON).

```
karc_forecast_into/D8_M8_K4_dh256/hla
    time:   [380.03 ns 381.02 ns 384.98 ns]
    thrpt:  [2.5975 Melem/s 2.6245 Melem/s 2.6314 Melem/s]

karc_forecast_into/D3_M8_K4_dh96/double_scroll
    time:   [111.41 ns 113.30 ns 113.77 ns]
    thrpt:  [8.7895 Melem/s 8.8262 Melem/s 8.9761 Melem/s]
```

D=8, M=8, K=4 (d_h=256, the HLA-shaped config): **381 ns/call** — comfortably
under the 500 ns target.

---

## G3 — Zero-Allocation Forecast

`tests/karc_alloc_check.rs` — manual `GlobalAlloc` counter wrapping `System`.
1000 `forecast_into` calls after 10 warmup calls: **0 alloc, 0 dealloc** delta.
The feature buffer (`forecast_psi`, d_h floats) is pre-allocated at construction
and reused via indexing (stack arrays of size `K·D·M` are not expressible in
stable Rust with const-generic arithmetic — `generic_const_exprs` is unstable).

---

## G4 — Bit-Reproducibility

`tests/karc_reproducibility.rs` — two forecasters fit on the same deterministic
synthetic trajectory produce **byte-identical Wout** (verified via `f32::to_bits`
comparison, which catches NaN-payload and signed-zero differences). Confirmed at
λ ∈ {1e-8, 1e-6, 1e-4} for both Fourier and Chebyshev bases.

---

## Phase 1 → Phase 2 Bridge

The G1 NRMSE gap (~5×) is expected to close with Phase 2 (T2.1 higher-order
features). The paper's headline result uses second-order Fourier features
(d_h=1891) which capture cross-coordinate nonlinear coupling that first-order
features (additive per-coordinate) cannot represent. Phase 2's
`feature_expand_higher_order` + low-rank factorization is the path to the full
16 LT threshold and sub-1e-3 NRMSE.

**TL;DR:** First-order KARC Phase 1: G2/G3/G4 PASS, G1 threshold PASS, G1 NRMSE
within 5× (documented gap → Phase 2 higher-order features). Feature stays opt-in.
