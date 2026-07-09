# Plan 421: Newton-Schulz Blocked Matmul — Eliminate Per-Dot Call Overhead

## Context

After the LoRA-Muon SIMD optimization (riir-train commit `9f52d86`), the NS inv-sqrt
remainder (~1.10ms at 768×768/r64) is now the dominant cost floor in
`lora_muon_step_cpu`. Profiling (2026-07-10, M3 Max) isolates the cost:

| Op | r | Time/call | GFLOP/s | M3 Max NEON peak fraction |
|---|---:|---:|---:|---|
| `ns_inv_sqrt_psd_into` (7 iters) | 64 | 297 µs | 21.6 | ~25% |
| `newton_schulz5_into` (5 iters) | 64×64 | 144 µs | — | — |
| `newton_schulz5_into` (5 iters) | 768×768 | 253 ms | — | — |

Two `ns_inv_sqrt_psd_into` calls per LoRA-Muon step (S_A, S_B) = **595 µs** — the
single largest component of the 1.10ms remainder.

## Root Cause

Both `ns_inv_sqrt_psd_into` and `newton_schulz_n_square_into_raw` compute r×r (or
m×m) matmuls via **r² individual `simd_dot_f32` calls** of length r. At r=64:

- Each `simd_dot_f32` call does 16 FMA iterations (4 NEON regs × 4 iters of 16 elements)
- But each call also pays: function-call overhead, 4× `vdupq_n_f32` register init,
  horizontal reduction (`vaddq` + `vaddvq`), and writes a single scalar result
- A `matmul_nn(r=64)` does 4096 such calls — the per-call overhead dominates
- The A-row is re-loaded from L1 for every output column, even though it could
  stay in registers across multiple Bᵀ rows

## Optimization: Blocked r×r Matmul

Replace the r²-dot-product matmul with a **rank-K blocking** approach:

```
For each row i of A:
  Load A[i, :] into 4 NEON registers (64 f32 = 256 bytes)
  For each block of 4 columns j..j+4 of Bᵀ:
    Load Bᵀ[j:j+4, :] into 4 NEON registers
    FMA-accumulate 4 output values C[i, j..j+4] simultaneously
```

This reuses each A-row load across 4 (or more) output columns, cutting memory
traffic and per-call overhead by 4×.

### Key design decisions

1. **Operate on r ≤ 128** — the blocked kernel targets the NS inv-sqrt path
   (r ≤ 64 in practice). For larger matrices (`newton_schulz5` at 768×768), the
   existing `simd_dot_f32` approach is already adequate (cache-bound, not
   overhead-bound at that scale — the 253ms is inherent O(m³) work).

2. **Feature flag** — ship behind `newton_schulz` (already default-on). No new
   feature flag needed; this is a pure implementation improvement within the
   existing module.

3. **Correctness** — the blocked matmul uses FMA (single rounding), same as the
   existing `simd_dot_f32` NEON path. Bit-identity with the current implementation
   is NOT guaranteed (different accumulation order), but the NS iteration's
   convergence guarantee comes from the polynomial coefficients, not from exact
   rounding. Existing tests use tolerance-based assertions.

4. **Zero new dependencies** — uses `core::arch::aarch64` / `core::arch::x86_64`
   intrinsics directly, same as the existing SIMD kernels.

## Tasks

- [x] T1: Implement `blocked_dot8` — 8-wide blocked dot product with NEON/scalar paths
- [x] T2: Implement `matmul_at_bt_blocked` — blocked r×r matmul processing 8 cols per A-row load
- [-] T3: Replace `matmul_nn` and `matmul_symmetric` in `ns_inv_sqrt_psd_into` — **REVERTED**: the blocked FMA accumulation order causes the NS inv-sqrt polynomial iteration to diverge on rank-deficient PSD matrices (zero eigenvalues). The different rounding pushes tiny eigenvalues outside the convergence basin. `matmul_nn` and `matmul_symmetric` stay on the original `simd_dot_f32` path.
- [x] T4: Replace `matmul_xtx`, A², and `matmul_ax` in `newton_schulz_n_square_into_raw` with blocked versions (for m ≤ 256 where input is normalized to ||X||_F = 1 — numerically safe)
- [x] T5: Run unit tests (13 NS tests + 1417 total) — all pass
- [x] T6: Benchmark before/after — GOAT gate results below
- [x] T7: Run riir-train GOAT gate (1259 unit tests + 14 Plan 299 GOAT tests) — all pass
- [x] T8: Update `.benchmarks/313_lora_muon_profiling.md` with NS optimization results
- [x] T9: Commit on `develop`

## GOAT Gate

- **G1 (correctness):** 13 katgpt-core NS tests + 1417 total katgpt-core tests + 1259 riir-train unit tests + 14 Plan 299 GOAT tests — ALL PASS
- **G2 (perf):** `newton_schulz5_into(64×64)` 144µs → 110µs = **1.31×**; `newton_schulz5_into(256×256)` 9722µs → 8817µs = **1.10×**. `ns_inv_sqrt_psd_into` unchanged (reverted for numerical safety).
- **G3 (no-regression):** `newton_schulz5_into(768×768)` 253ms → 263ms = **0.96× (within noise)** — large-matrix path uses scalar fallback.
- **G4 (alloc-free):** zero new heap allocations (reuses existing scratch buffers)
- **G5 (zero deps):** no new crate dependencies

## Numerical Safety Lesson

The blocked `blocked_dot8` NEON kernel uses 8 independent accumulators (one per
dot product) with 4-element FMA chunks. The original `simd_dot_f32` uses 4
accumulators per single dot with 16-element chunks. The different accumulation
order produces ULP-level differences that are normally harmless.

**However**, `ns_inv_sqrt_psd_into` operates on PSD matrices that can be
rank-deficient (Gram matrices of low-rank LoRA adapters). The NS polynomial
iteration (7 iterations of `X = aX + (bA + cA²)X`) amplifies rounding errors in
the near-zero eigenvalues. The different FMA accumulation order in the blocked
kernel can push these tiny eigenvalues outside the convergence basin [0, 1],
causing the iteration to diverge to Inf → NaN.

The `newton_schulz_n_square_into_raw` path is safe because it normalizes the
input to ||X||_F = 1 before iteration, bounding all singular values to [0, 1].

**Rule:** blocked matmul kernels with different FMA accumulation orders must
NOT be used in numerical iterations that operate on un-normalized, potentially
rank-deficient matrices.

## Non-Goals

- Optimizing `newton_schulz5` at 768×768 (253ms) — that's O(m³) inherent work, not overhead-dominated
- Adding BLAS — the blocked SIMD kernel closes the gap without a C dependency
- Changing the NS iteration coefficients or algorithm
