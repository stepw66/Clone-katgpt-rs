/-
! Spec for the SSMax (log-N Attention Temperature) dilution bound (Plan 411 S3).

This Lean 4 model is the mathematical specification of the property that
`katgpt-rs/crates/katgpt-core/src/ssmax.rs` relies on: the **attention-dilution
bound** from Gollapudi et al., *Can Language Models Actually Retrieve In-Context?
Drowning in Documents at Million Token Scale* (arXiv:2607.01538, 2026).

## The bound

As corpus size `N` grows, the post-normalization attention mass on the gold
document collapses:

    α_gold ≈ 1 / (1 + (N − 1) · N^(−c))

where `c` is the *effective sharpening exponent* in the pre-softmax gold-vs-
distractor logit gap. For the base attention, `c = s · Δ` (the score scale times
the gold–distractor gap). For SSMax, each logit is rescaled by `s_L · log(N)`,
so the effective exponent becomes

    c_SSMax = s_L · log(N) · s · Δ = (s_L · log N) · c_base.

The paper's claim — and the whole reason SSMax ships — is that this extra
`s_L · log(N)` factor is enough to cancel the `(N − 1)` denominator growth.

## What this file proves

This file is the **spec**: it defines `alphaGold N c = 1 / (1 + (N−1) · N^(−c))`
and the elementary building blocks (positivity, boundedness, monotonicity of
`N^c` in `c` for `N > 1`). The headline theorems live in `DilutionBound.lean`.

The companion Rust spec-match test is
`crates/katgpt-core/tests/ssmax_spec_match.rs`; it asserts the Rust
`apply_ssmax_inplace` semantics and the dilution-bound formula match this spec
to f32 precision at sampled `(N, c)` points.

## Why Mathlib

Like `KatgptProof.Bridge`, this proof cannot avoid Mathlib: `N^(−c)` requires
`Real.rpow`, whose strict monotonicity depends on the transcendental analysis
of `exp` / `log` (`Real.log_pos`, `Real.strictMono_exp`, …). Same toolchain:
`leanprover/lean4:v4.32.0-rc1`.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace KatgptProof.Ssmax

open Real

/-! ## The dilution bound

`alphaGold N c` is the post-normalization attention mass on the gold document
when the corpus has `N` keys and the effective sharpening exponent (the
pre-softmax gold–distractor logit gap, scaled by the score scale) is `c`.

For `N > 1` this is well-defined, positive, and strictly less than `1`.
-/

/-- The post-normalization gold attention weight under the paper's dilution
    bound (arXiv:2607.01538 §2). `alphaGold N c = 1 / (1 + (N−1) · N^(−c))`.

    - `N` — number of attended keys (corpus size). Must satisfy `1 < N`.
    - `c` — effective sharpening exponent. Base attention: `c = s · Δ`.
      SSMax: `c = s_L · log(N) · s · Δ`.

    Rust reference: `crates/katgpt-core/src/ssmax.rs` doc comment (the bound is
    stated in the module-level `//!` header, lines 11–17). -/
noncomputable def alphaGold (N c : ℝ) : ℝ := 1 / (1 + (N - 1) * N^(-c))

/-! ## Elementary facts

The denominator `1 + (N−1) · N^(−c)` is strictly greater than `1` for `N > 1`
(since `(N−1) > 0` and `N^(−c) > 0`), so `alphaGold` is well-defined and lies in
the open interval `(0, 1)`. These are the preconditions every theorem below
relies on.
-/

/-- For `N > 1`, the denominator `1 + (N−1) · N^(−c)` is strictly positive
    (in fact `> 1`), so `alphaGold` is well-defined and positive. -/
lemma alphaGold_denom_pos {N : ℝ} (hN : 1 < N) (c : ℝ) :
    0 < 1 + (N - 1) * N^(-c) := by
  have hNpow : 0 < N^(-c) := Real.rpow_pos_of_pos (by linarith) _
  have hNm1 : 0 < N - 1 := by linarith
  have hprod : 0 < (N - 1) * N^(-c) := mul_pos hNm1 hNpow
  linarith

/-- `alphaGold` lies strictly in `(0, 1)` for `N > 1`. The mass on the gold
    document is always a well-defined probability strictly between zero and one,
    regardless of how large `N` gets — the question SSMax answers is *how close
    to one* it can be pushed. -/
lemma alphaGold_bounded {N : ℝ} (hN : 1 < N) (c : ℝ) :
    0 < alphaGold N c ∧ alphaGold N c < 1 := by
  have hNpow : 0 < N^(-c) := Real.rpow_pos_of_pos (by linarith) _
  have hNm1 : 0 < N - 1 := by linarith
  have hprod : 0 < (N - 1) * N^(-c) := mul_pos hNm1 hNpow
  -- Denominator `1 + (N-1)·N^(-c)` is `> 1` (since `(N-1)·N^(-c) > 0`).
  have hdenom_gt1 : 1 < 1 + (N - 1) * N^(-c) := by linarith
  constructor
  · -- 0 < 1/(1 + positive): trivially, `1/(>1) > 0`.
    exact one_div_pos.mpr (by linarith)
  · -- 1/(1 + positive) < 1: `1/(b) < 1 ⟺ 1 < b` for `b > 0` (`div_lt_one`).
    rw [alphaGold]
    exact (div_lt_one (by linarith)).mpr (by linarith)

/-! ## Monotonicity building block: `N^c` is strictly increasing in `c`

For `N > 1`, `N^c = exp(c · log N)` is strictly increasing in `c` (since
`log N > 0` and `exp` is strictly increasing). This is the engine that makes
SSMax work: increasing the effective exponent `c` (by rescaling logits) pushes
more mass onto the gold document.
-/

/-- For `N > 1`, the function `c ↦ N^c` is strictly monotonically increasing.
    Proof: `N^c = exp(log N · c)` (Mathlib `Real.rpow_def_of_pos`), the map
    `c ↦ log N · c` is strictly monotone (multiplication by `log N > 0`), and
    `exp` is strictly monotone — the composition is strictly monotone. -/
lemma strictMono_rpow_of_gt_one {N : ℝ} (hN : 1 < N) :
    StrictMono (fun c : ℝ ↦ N^c) := by
  intro x y hxy
  -- N^x = exp(log N · x) and N^y = exp(log N · y); log N > 0, x < y ⇒
  -- log N · x < log N · y ⇒ exp(log N · x) < exp(log N · y).
  show N^x < N^y
  have hlog : 0 < Real.log N := Real.log_pos hN
  have hmul : Real.log N * x < Real.log N * y :=
    mul_lt_mul_of_pos_left hxy hlog
  -- Unfold `N^c` to `exp(log N · c)` for `N > 0` (Mathlib `rpow_def_of_pos`).
  have hNpos : 0 < N := by linarith
  rw [Real.rpow_def_of_pos hNpos x, Real.rpow_def_of_pos hNpos y]
  exact Real.exp_strictMono hmul

end KatgptProof.Ssmax
