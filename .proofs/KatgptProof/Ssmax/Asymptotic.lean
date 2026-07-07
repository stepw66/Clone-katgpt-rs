/-
! Theorem: SSMax asymptotically defeats attention dilution (Plan 411 S3 follow-up).

This file proves the asymptotic complement to `DilutionBound.lean`. The finite-N
theorems there show that for any fixed `N > 1`, SSMax (with `s_L · log N ≥ 1`)
pushes *more* mass onto the gold document than base attention. This file proves
the limit statement: **as the corpus size `N` grows without bound, SSMax drives
the gold attention mass all the way to `1`** — i.e., SSMax doesn't just slow
the dilution, it defeats it entirely in the large-`N` limit.

## The statement

For fixed `s_L > 0` and `Δ > 0`:

    lim_{N → ∞} α_gold(N, s_L · log(N) · Δ) = 1

where `α_gold(N, c) = 1 / (1 + (N − 1) · N^(−c))` is the dilution bound
(`Basic.lean`). Equivalently, the "leakage" term `(N − 1) · N^(−s_L · log N · Δ)`
vanishes as `N → ∞`.

## Proof strategy (squeeze)

Let `f(N) = (N − 1) · N^(−s_L · log(N) · Δ)` (the leakage).

1. **Lower bound**: `f(N) ≥ 0` for `N > 1`.
2. **Upper bound**: For `N` large enough that `s_L · Δ · log N ≥ 2`,
   `f(N) ≤ 1/N`.
   - `f(N) ≤ N · N^(−s_L · log N · Δ) = N^(1 − s_L · Δ · log N)` (algebra)
   - `1 − s_L · Δ · log N ≤ −1` iff `s_L · Δ · log N ≥ 2`
   - `N^(1 − s_L · Δ · log N) ≤ N^(−1) = 1/N` (monotonicity of `N^c`, `Basic.lean`)
3. **Squeeze**: `1/N → 0` (`tendsto_inv_atTop_zero`), `0 → 0`, so `f(N) → 0`.
4. **Continuity**: `α_gold = 1/(1 + f)`, denominator `1 + f → 1 ≠ 0`, so `→ 1`.

The key rate comparison: `(log N)²` grows faster than `log N`, so
`N^(−s_L · log N · Δ) = exp(−s_L · Δ · (log N)²)` collapses faster than any
polynomial in `N` grows. SSMax's `log(N)` multiplier produces *super-polynomial*
decay of the leakage term.

## Why this was deferred from S3

The finite-N theorems are the practically important results — they tell you
SSMax helps at every fixed `N ≥ 3` (for `s_L = 1`). The asymptotic theorem is a
"completeness" statement: SSMax doesn't just slow the bleeding, it stops it.
Useful for arguing about scale behavior; not load-bearing for any runtime
decision.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Algebra.Order.Field
import KatgptProof.Ssmax.Basic
import KatgptProof.Ssmax.DilutionBound

namespace KatgptProof.Ssmax

open Real Topology Filter

/-! ## Step 1: The leakage term

`leakage N s_L Δ = (N − 1) · N^(−s_L · log N · Δ)` is the "extra denominator"
beyond `1` in `alphaGold`. We want to show it vanishes.
-/

/-- The leakage term `(N − 1) · N^(−s_L · log N · Δ)`. -/
noncomputable def leakage (N s_L Δ : ℝ) : ℝ :=
  (N - 1) * N^(-(s_L * Real.log N * Δ))

/-- `alphaGold` in terms of `leakage`: `1 / (1 + leakage)`. -/
lemma alphaGold_eq {N s_L Δ : ℝ} :
    alphaGold N (s_L * Real.log N * Δ) = 1 / (1 + leakage N s_L Δ) := by
  rfl

/-- The leakage is non-negative for `N > 1` (both factors positive). -/
lemma leakage_nonneg {N s_L Δ : ℝ} (hN : 1 < N) :
    0 ≤ leakage N s_L Δ := by
  have hNpos : 0 < N := by linarith
  have hNm1 : 0 ≤ N - 1 := by linarith
  have hNpow : 0 < N^(-(s_L * Real.log N * Δ)) :=
    Real.rpow_pos_of_pos hNpos _
  exact mul_nonneg hNm1 (le_of_lt hNpow)

/-! ## Step 2: For large `N`, `leakage ≤ 1/N`

Algebra:

    leakage ≤ N · N^(−c)           [(N−1) ≤ N, factor ≥ 0]
            = N^(1 − c)            [rpow_add]
            ≤ N^(−1)                [if c ≥ 2, exponent ≤ −1, N^c monotone]
            = 1 / N
-/

/-- For `1 < N` and `s_L · Δ · log N ≥ 2`: `leakage N s_L Δ ≤ 1/N`. -/
lemma leakage_le_inv {N s_L Δ : ℝ} (hN : 1 < N)
    (hlog_bound : 2 ≤ s_L * Δ * Real.log N) :
    leakage N s_L Δ ≤ 1 / N := by
  have hNpos : 0 < N := by linarith
  have hNpow_nn : 0 ≤ N^(-(s_L * Real.log N * Δ)) :=
    le_of_lt (Real.rpow_pos_of_pos hNpos _)
  -- Step A: (N − 1) ≤ N, multiply by the (nonneg) power.
  have hA : (N - 1) * N^(-(s_L * Real.log N * Δ)) ≤
            N * N^(-(s_L * Real.log N * Δ)) :=
    mul_le_mul_of_nonneg_right (by linarith) hNpow_nn
  -- Step B: N · N^(-c) = N^(1 - c).
  have hrw : N * N^(-(s_L * Real.log N * Δ)) =
             N^(1 - s_L * Real.log N * Δ) := by
    have h1 : 1 - s_L * Real.log N * Δ = 1 + (-(s_L * Real.log N * Δ)) := by ring
    rw [h1, Real.rpow_add hNpos, Real.rpow_one]
  rw [hrw] at hA
  -- Step C: 1 - s_L · log N · Δ ≤ -1, i.e. s_L · log N · Δ ≥ 2.
  -- Note s_L · log N · Δ = s_L · Δ · log N by ring.
  have h_exponent_le : 1 - s_L * Real.log N * Δ ≤ -1 := by
    have h_eq : s_L * Real.log N * Δ = s_L * Δ * Real.log N := by ring
    rw [h_eq]; linarith
  -- strictMono_rpow_of_gt_one (Basic.lean): N^c is strictly increasing in c.
  have h_mono_le : N^(1 - s_L * Real.log N * Δ) ≤ N^(-1 : ℝ) := by
    rw [show N^(1 - s_L * Real.log N * Δ) ≤ N^(-1 : ℝ) ↔
        1 - s_L * Real.log N * Δ ≤ (-1 : ℝ) from
      (strictMono_rpow_of_gt_one hN).le_iff_le]
    exact h_exponent_le
  -- N^(-1) = N⁻¹ = 1/N.
  have h_neg_one : N^(-1 : ℝ) = N⁻¹ := Real.rpow_neg_one N
  have h_inv_div : N⁻¹ = 1 / N := by field_simp
  -- Chain via transitivity.
  calc leakage N s_L Δ
      = (N - 1) * N^(-(s_L * Real.log N * Δ)) := rfl
    _ ≤ N^(1 - s_L * Real.log N * Δ) := hA
    _ ≤ N^(-1 : ℝ) := h_mono_le
    _ = N⁻¹ := h_neg_one
    _ = 1 / N := h_inv_div

/-! ## Step 3: Eventually `s_L · Δ · log N ≥ 2`

Since `log N → ∞` (`Real.tendsto_log_atTop`), for `s_L · Δ > 0` the product
`s_L · Δ · log N → ∞`, so it is eventually `≥ 2`.
-/

/-- For `0 < s_L · Δ`, we have `s_L · Δ · log N ≥ 2` for all large enough `N`. -/
lemma eventually_c_ge_two {s_L Δ : ℝ} (hprod : 0 < s_L * Δ) :
    ∀ᶠ N in atTop, 2 ≤ s_L * Δ * Real.log N := by
  have h_log : Tendsto Real.log atTop atTop := Real.tendsto_log_atTop
  have h_log_ge : ∀ᶠ N in atTop, 2 / (s_L * Δ) ≤ Real.log N :=
    h_log.eventually (eventually_ge_atTop (2 / (s_L * Δ)))
  exact h_log_ge.mono (fun N hN ↦ by
    have hprod_nn : 0 ≤ s_L * Δ := le_of_lt hprod
    have hprod_ne : s_L * Δ ≠ 0 := ne_of_gt hprod
    have h_scaled : s_L * Δ * Real.log N ≥ s_L * Δ * (2 / (s_L * Δ)) :=
      mul_le_mul_of_nonneg_left hN hprod_nn
    have h_simpl : s_L * Δ * (2 / (s_L * Δ)) = 2 :=
      mul_div_cancel₀ 2 hprod_ne
    linarith [h_scaled, h_simpl])

/-! ## Step 4: The squeeze — `leakage → 0`

For all large `N`: `0 ≤ leakage N ≤ 1/N`. Since `0 → 0` and `1/N → 0`,
`squeeze` gives `leakage → 0`.
-/

/-- **The leakage term vanishes.** For `0 < s_L · Δ`:

        lim_{N → ∞} (N − 1) · N^(−s_L · log N · Δ) = 0
-/
theorem tendsto_leakage_zero {s_L Δ : ℝ} (hprod : 0 < s_L * Δ) :
    Tendsto (fun N : ℝ ↦ leakage N s_L Δ) atTop (𝓝 0) := by
  -- Upper bound function: 1/N → 0 (we rewrite N⁻¹ as 1/N via field_simp).
  have h_inv_tendsto : Tendsto (fun N : ℝ ↦ 1 / N) atTop (𝓝 0) := by
    have h := tendsto_inv_atTop_zero (𝕜 := ℝ)
    convert h using 1
    ext N
    field_simp
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le'
  · exact tendsto_const_nhds
  · exact h_inv_tendsto
  · -- Eventually: 0 ≤ leakage (for N > 1).
    exact (eventually_gt_atTop 1).mono (fun N hN ↦ leakage_nonneg hN)
  · -- Eventually: leakage ≤ 1/N.
    exact ((eventually_gt_atTop 1).and (eventually_c_ge_two hprod)).mono
      (fun N ⟨hN1, hN2⟩ ↦ leakage_le_inv hN1 hN2)

/-! ## Step 5: `alphaGold → 1`

`alphaGold N (s_L · log N · Δ) = 1 / (1 + leakage N s_L Δ)`. Since
`leakage → 0` and the denominator `1 + leakage → 1 ≠ 0`, the quotient rule
for limits gives `alphaGold → 1/1 = 1`.
-/

/-- **SSMax asymptotically defeats dilution.** For `0 < s_L · Δ`:

        lim_{N → ∞} α_gold(N, s_L · log N · Δ) = 1

The gold attention mass converges to `1` — SSMax completely undoes the
attention dilution in the large-corpus limit. -/
theorem tendsto_alphaGold_one {s_L Δ : ℝ} (hprod : 0 < s_L * Δ) :
    Tendsto (fun N : ℝ ↦ alphaGold N (s_L * Real.log N * Δ)) atTop (𝓝 1) := by
  -- alphaGold = 1 / (1 + leakage) = (1 + leakage)⁻¹
  have h_denom : Tendsto (fun N : ℝ ↦ 1 + leakage N s_L Δ) atTop (𝓝 (1 : ℝ)) := by
    have h_base : Tendsto (fun N : ℝ ↦ (1 : ℝ)) atTop (𝓝 1) := tendsto_const_nhds
    have h_leak : Tendsto (fun N : ℝ ↦ leakage N s_L Δ) atTop (𝓝 0) :=
      tendsto_leakage_zero hprod
    convert h_base.add h_leak using 1
    simp
  have h_denom_ne : (1 : ℝ) ≠ 0 := one_ne_zero
  -- Inverse tendsto: (1 + leakage)⁻¹ → 1⁻¹ = 1.
  have h_inv_tendsto : Tendsto (fun N : ℝ ↦ (1 + leakage N s_L Δ)⁻¹) atTop (𝓝 (1:ℝ)) := by
    have h := h_denom.inv₀ h_denom_ne
    simpa [inv_one] using h
  -- Rewrite the goal function as (1 + leakage)⁻¹.
  have h_eq : ∀ N : ℝ, alphaGold N (s_L * Real.log N * Δ) = (1 + leakage N s_L Δ)⁻¹ := by
    intro N
    rw [alphaGold_eq, one_div]
  -- Conclude via filter ext.
  rw [show (fun N : ℝ ↦ alphaGold N (s_L * Real.log N * Δ)) =
        (fun N : ℝ ↦ (1 + leakage N s_L Δ)⁻¹) from funext h_eq]
  exact h_inv_tendsto

end KatgptProof.Ssmax
