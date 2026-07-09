# Issue 043: `ns_inv_sqrt_psd_into` Blocked Matmul — Numerical Robustness Blocker

> **Spawned from:** Plan 421 (Newton-Schulz blocked dot8 matmul — commit `4116ad37`)
> **Confidence:** HIGH — the divergence is reproduced and root-caused; the blocker is algorithmic (FMA accumulation order), not a bug.
> **Date:** 2026-07-10
> **Status:** OPEN

---

## TL;DR

Plan 421 shipped a blocked 8-wide NEON matmul kernel (`blocked_dot8`) that gives **1.31×** on `newton_schulz5(64×64)`. The same kernel was applied to `ns_inv_sqrt_psd_into` (the LoRA-Muon inv-sqrt bottleneck, ~595µs/pair for 2× r=64 calls) and **reverted** because it causes NaN divergence on rank-deficient PSD matrices. The `ns_inv_sqrt_psd_into` path is **unchanged** — it still uses the per-dot `simd_dot_f32` at ~79% of NEON peak. This issue tracks the investigation into a numerically-safe blocked kernel for that path.

**The blocker is the FMA accumulation order.** `simd_dot_f32` splits one dot product into 4 interleaved partial sums ([0,4,8,12],[1,5,9,13],…); `blocked_dot8` processes 8 separate dots, each with consecutive 4-element chunks ([0,1,2,3],[4,5,6,7],…). The ULP-level difference pushes near-zero eigenvalues of rank-deficient Gram matrices outside the NS polynomial convergence basin [0,1], and 7 iterations of `X = aX + (bA + cA²)X` amplifies the divergence to Inf→NaN by iteration 4.

---

## The Bottleneck

`ns_inv_sqrt_psd_into` computes P⁻¹ᵐ² for symmetric PSD r×r matrices via 7 Newton-Schulz polynomial iterations. It is called **twice per `lora_muon_step_cpu`** step (once for S_A, once for S_B Gram matrices of the LoRA adapter pair).

| Operation | Current cost | % of NEON peak | Calls/step |
|---|---:|---:|---:|
| `ns_inv_sqrt_psd_into(r=64, 7 iters)` | 297 µs | ~79% (~22 GFLOP/s of 28 peak) | 2 |
| **Total per LoRA-Muon step** | **595 µs** | — | — |

Each iteration calls `matmul_symmetric` (P²) and `matmul_nn` (X·W, P·W²), each doing r² individual `simd_dot_f32` calls of length r. At r=64, that's 3×64² = 12,288 dot-product calls per iteration × 7 iterations ≈ 86K dot calls — the per-dot overhead (function call, `vdupq_n_f32` init, horizontal `vaddvq_f32` reduction) is the ~21% gap to peak.

---

## Root Cause: Accumulation Order Divergence

### `simd_dot_f32` (NEON `neon_dot_f32`, the current path)

`katgpt-rs/crates/katgpt-types/src/simd/dot.rs:97`:

```
4 accumulators × 16-element outer loop
acc0 ← FMA over elements [0, 4, 8, 12, …]   (stride-4 interleaved)
acc1 ← FMA over elements [1, 5, 9, 13, …]
acc2 ← FMA over elements [2, 6, 10, 14, …]
acc3 ← FMA over elements [3, 7, 11, 15, …]
reduce: ((acc0 + acc1) + (acc2 + acc3))
```

### `blocked_dot8_neon` (the reverted kernel)

`katgpt-rs/crates/katgpt-core/src/newton_schulz.rs:663`:

```
8 accumulators (one per dot product, sharing one `a` operand)
each acc ← FMA over consecutive 4-element chunks [0,1,2,3], [4,5,6,7], …
reduce: each acc independently via vaddvq_f32
```

### Why it diverges

- LoRA adapter Gram matrices (S = GᵀG for rank-r adapter G) are **rank-deficient** when the adapter rank < matrix dimension — they have zero eigenvalues.
- The NS polynomial iteration `X_{k+1} = a_k·X_k + (b_k·P + c_k·P²)·X_k` converges only when all eigenvalues of P stay in [0, 1] throughout the iteration.
- The Frobenius normalization (`P / ||P||_F`) bounds the *largest* singular value to ≤1, but near-zero eigenvalues sit at the edge of the convergence basin.
- The different FMA accumulation order produces ULP-level differences in the squared matrix P². For eigenvalues near zero, a 1-ULP difference can flip the sign of a tiny eigenvalue or push it slightly outside [0,1].
- 7 iterations of the polynomial amplify the error exponentially — by iteration 4, the near-zero eigenvalues have diverged to Inf, contaminating the whole matrix as NaN.

**Reproduced:** rank-2 PSD matrix at r=32. Original code converges (max_abs=19.69). Blocked code diverges by iteration 4 (max_abs=Inf → NaN).

### Why `newton_schulz5` is safe but `ns_inv_sqrt_psd_into` is not

`newton_schulz_n_square_into_raw` normalizes its input to ||X||_F = 1 **before** iterating, which bounds **all** singular values to [0, 1]. The NS iteration operates entirely within the safe basin, so ULP-level accumulation differences don't cause divergence. `ns_inv_sqrt_psd_into` normalizes by the Frobenius norm too, but PSD matrices with zero eigenvalues still have eigenvalues at the *edge* of the basin (near 0), where the polynomial is maximally sensitive to rounding.

---

## Candidate Approaches

### Approach A — Match the exact `simd_dot_f32` accumulation order (HARD)

Use 4 interleaved accumulators per dot product (matching `neon_dot_f32`'s [0,4,8,12]/[1,5,9,13]/… pattern), blocked across 8 dots. That's 8 × 4 = 32 NEON accumulators — **exactly the full NEON register file** (Q0–Q31), leaving zero registers for `a`/`b` loads or address computation. The compiler would spill to stack, killing the perf gain.

A 4-dot blocked variant (4 × 4 = 16 accumulators, 16 registers for loads/addresses) is feasible but halves the amortization benefit — the speedup would be much smaller than the 1.31× seen on `newton_schulz5`.

**Verdict:** technically possible but the register pressure likely eats the gain. Would need a PoC to confirm.

### Approach B — Eigendecomposition-based inv-sqrt (DIFFERENT ALGORITHM)

For symmetric PSD matrices, P⁻¹ᵐ² = V·diag(λᵢ⁻¹ᵐ²)·Vᵀ where P = V·diag(λᵢ)·Vᵀ. Compute the eigendecomposition once, apply the scalar inv-sqrt to eigenvalues, reconstruct.

- **Pros:** no iterative amplification of rounding errors; numerically robust; the eigendecomposition itself can use LAPACK-style blocked routines.
- **Cons:** eigendecomposition is O(r³) with a larger constant than 7 NS iterations; for r=64 the crossover may not be favorable. Requires a symmetric eigensolver (Jacobi rotation is the simplest modelless option — no external dep). Also needs a tolerance threshold for zero eigenvalues (λ < ε → λ⁻¹ᵐ² = 0, which is the correct PSD pseudo-inverse behavior).

**Verdict:** promising for numerical robustness, but needs a PoC to check if the constant factor is competitive at r=64. Jacobi eigenvalue algorithm is ~10× the FLOPs of one NS iteration, so 7 NS iters vs 1 Jacobi pass might be a wash.

### Approach C — Different polynomial basis (RESEARCH)

The current NS polynomial (coefficients in `INV_SQRT_COEFFS`) is a degree-5 minimax approximation to x⁻¹ᵐ² on [0, 1]. A basis with better numerical conditioning near zero (e.g., Chebyshev, or a rational approximation) might be more robust to accumulation-order differences.

**Verdict:** research-heavy, uncertain payoff. The convergence basin issue is fundamental to polynomial iteration on near-zero eigenvalues — a different basis might widen the basin but won't eliminate the sensitivity.

### Approach D — Regularize the PSD matrix (CHEAP, PARTIAL)

Add a larger ε·I regularization to P before iterating (currently `INV_SQRT_EPS`). This pushes zero eigenvalues away from the convergence basin edge, making the iteration more robust to accumulation-order differences. The cost is a small bias in the inv-sqrt result.

**Verdict:** cheapest to try. The question is whether the bias is acceptable for LoRA-Muon convergence (the optimizer should tolerate small inv-sqrt errors — it's an orthogonalization preconditioner, not a precision-critical computation). Would need to re-run the Plan 299 GOAT tests with a larger ε to check.

---

## PoC Gate (mandatory before any impl)

Before implementing any approach, a PoC must show:

1. **Numerical safety:** the blocked/alternative kernel converges on rank-deficient PSD matrices (rank-2 at r=32, rank-8 at r=64, rank-16 at r=64) — no NaN/Inf within 7 iterations.
2. **Perf gain:** `ns_inv_sqrt_psd_into(r=64, 7 iters)` < 250 µs/call (current: 297 µs). The 2× per-step cost would drop from 595 µs to < 500 µs.
3. **G1 correctness:** all 14 Plan 299 GOAT tests pass, especially `g1_cross_rank_lr_transfer_predicts_r32` (the test that caught the original divergence).
4. **Bit-identical or provably-bounded:** either the new kernel produces bit-identical output to `simd_dot_f32` on full-rank PSD matrices, or the output difference is bounded and documented as acceptable for the LoRA-Muon use case.

**Where the PoC would live:** `katgpt-rs/crates/katgpt-core/benches/bench_ns_inv_sqrt_blocked.goat.rs` — compare current `simd_dot_f32` path vs the candidate blocked/alternative kernel on rank-deficient PSD matrices.

---

## Decision Matrix

| Approach | Numerically safe? | Perf gain likely? | Effort | Impl now? |
|---|---|---|---|---|
| A — Match `simd_dot_f32` order (8-dot) | Yes (bit-identical) | Unlikely (register spill) | Medium | **No** — PoC first |
| A' — 4-dot blocked (16 regs) | Yes (bit-identical) | Small (~1.1×?) | Medium | **No** — PoC first |
| B — Eigendecomposition (Jacobi) | Yes (robust) | Uncertain at r=64 | Large | **No** — PoC first |
| C — Different polynomial basis | Maybe | Uncertain | Large (research) | **No** |
| D — Larger ε regularization | Maybe | N/A (same kernel) | Tiny | **Maybe** — cheapest test |

**Recommendation:** start with **Approach D** (5-minute test: bump `INV_SQRT_EPS`, re-run GOAT tests with the blocked kernel). If D works, the blocked kernel can be re-enabled with a larger ε. If D doesn't work (bias too large or still diverges), fall back to **Approach A'** (4-dot blocked, bit-identical) and accept the smaller speedup.

---

## Tasks

- [ ] **T1** Try Approach D: bump `INV_SQRT_EPS` by 10×/100×, re-enable `blocked_dot8` in `matmul_symmetric`/`matmul_nn`, run Plan 299 GOAT tests. If pass → measure perf gain. If fail → document the ε threshold at which divergence stops.
- [ ] **T2** If T1 fails, try Approach A': implement a 4-dot blocked variant (`blocked_dot4_interleaved`) that matches `simd_dot_f32`'s 4-accumulator interleaved pattern across 4 dots (16 NEON registers). PoC the perf gain at r=64.
- [ ] **T3** If T1+T2 fail, evaluate Approach B: implement a Jacobi eigensolver for symmetric PSD matrices, compute inv-sqrt via eigenvalue scaling. PoC the perf at r=64 and r=32.
- [-] **T4** (deferred) Approach C (polynomial basis research) — only if A/B/D all fail and the perf gain is worth the research investment.

---

## Cross-references

- **Plan 421** (`katgpt-rs/.plans/421_newton_schulz_blocked_matmul.md`) — the blocked dot8 implementation, GOAT gate results, and the numerical safety lesson.
- **Plan 299** (`katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`) — the GOAT test suite that caught the divergence (`g1_cross_rank_lr_transfer_predicts_r32`).
- **`.benchmarks/313_lora_muon_profiling.md`** (`riir-train`) — LoRA-Muon profiling data showing `ns_inv_sqrt_psd_into` as the ~595µs/pair bottleneck.
- **`katgpt-rs/crates/katgpt-core/src/newton_schulz.rs:436`** — `ns_inv_sqrt_psd_into` (the bottleneck function).
- **`katgpt-rs/crates/katgpt-core/src/newton_schulz.rs:555`** — `matmul_symmetric` (reverted to `simd_dot_f32`, with comment documenting the safety reason).
- **`katgpt-rs/crates/katgpt-core/src/newton_schulz.rs:577`** — `matmul_nn` (same).
- **`katgpt-rs/crates/katgpt-types/src/simd/dot.rs:97`** — `neon_dot_f32` (the 4-accumulator interleaved reference pattern).

---

## TL;DR

The `ns_inv_sqrt_psd_into` LoRA-Muon bottleneck (~595µs/pair) remains unoptimized. The blocked `blocked_dot8` kernel was reverted because its different FMA accumulation order causes NaN divergence on rank-deficient PSD matrices. Four candidate approaches are documented (A: match accumulation order, B: eigendecomposition, C: different polynomial basis, D: larger ε regularization). **Start with Approach D** (cheapest — bump ε, re-test); fall back to **A'** (4-dot bit-identical blocked) if D fails. The PoC gate requires convergence on rank-deficient matrices + < 250µs/call + all Plan 299 GOAT tests passing.
