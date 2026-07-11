/-
! Theorem: SSMax's log-N rescaling increases gold attention mass.

This is the Plan 411 S3 formal-verification result. It proves the two properties
that justify SSMax (log-N attention temperature, arXiv:2607.01538) at the
mathematical level:

1. **`alphaGold` is strictly increasing in the sharpening exponent `c`**
   (`alphaGold_strictMono_in_c`). This is the monotonicity that makes SSMax work:
   rescaling logits to increase the effective exponent `c` provably pushes more
   mass onto the gold document. It is the dilution-bound analog of
   `KatgptProof.Bridge.action_bridge_ranking_preserved` (sigmoid strict
   monotonicity).

2. **SSMax dominates base when `s_L · log(N) ≥ 1`** (`ssmax_dominates_base`).
   SSMax replaces `c_base = s · Δ` with `c_SSMax = s_L · log(N) · s · Δ =
   (s_L · log N) · c_base`. By monotonicity, `alphaGold` increases iff `c`
   increases, iff `s_L · log(N) · s · Δ ≥ s · Δ`, iff `s_L · log(N) ≥ 1`
   (assuming `s · Δ > 0`).

   **This sharpens the plan's informal statement.** Plan 411 S3 sketches
   "`s_L = 1.0, N ≥ 2 ⇒ α_gold(SSMax) ≥ α_gold(base)`", but `log(2) ≈ 0.693 < 1`,
   so at `s_L = 1, N = 2` SSMax is in fact *milder* than base (the bound says the
   gold mass is slightly *lower* with SSMax than without). The correct threshold
   is `s_L · log(N) ≥ 1`, i.e. `N ≥ 3` when `s_L = 1` (since `log(3) ≈ 1.099`).
   This is the kind of off-by-one-in-the-threshold that formal verification
   catches and the empirical G1/G2 tests (which sample N ∈ {64, 1k, 10k, 100k},
   all comfortably above the threshold) miss.

The proofs are two-liners once the monotonicity of `N^c` in `c` is established
(`strictMono_rpow_of_gt_one` in `Basic.lean`). No `sorry`. No admitted axioms
beyond Mathlib's standard foundations (`propext`, `Classical.choice`,
`Quot.sound`).

## Rust spec-match

The paired Rust test `crates/katgpt-core/tests/ssmax_spec_match.rs` asserts that
the Rust dilution-bound formula (as documented in `ssmax.rs` lines 11–17) and
`SsmaxMode::multiplier` agree with this Lean spec to f32 precision. If the Rust
drifts (e.g. someone changes the bound formula or the multiplier semantics), the
test fails and the proof must be re-validated.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import KatgptProof.Ssmax.Basic

namespace KatgptProof.Ssmax

open Real

/-! ## Theorem 1: `alphaGold` is strictly increasing in `c`

This is the headline monotonicity. Unwinding the definition, the chain of
strict-monotonicity / strict-antimonotonicity arguments is:

    c ↦ N^(−c)         strictly decreasing  (N^c increasing, negation reverses)
    c ↦ (N−1) · N^(−c) strictly decreasing  (positive scalar multiple)
    c ↦ 1 + (N−1)·N^(−c) strictly decreasing (shift)
    c ↦ 1 / (1 + (N−1)·N^(−c)) strictly increasing (reciprocal on positives)

The direct algebraic form is easier in Lean: `alphaGold N c₁ < alphaGold N c₂`
(for `c₁ < c₂`) reduces — via positivity of all terms — to `N^(c₁) < N^(c₂)`,
which is exactly `strictMono_rpow_of_gt_one`.
-/

/-- **SSMax monotonicity (the headline theorem).** For `N > 1`, the gold
    attention mass `alphaGold N c` is strictly increasing in the effective
    sharpening exponent `c`.

    This is the property that justifies SSMax: rescaling logits to increase `c`
    (via the `s_L · log(N)` multiplier) provably pushes more mass onto the gold
    document. It is the dilution-bound counterpart of
    `KatgptProof.Bridge.action_bridge_ranking_preserved`. -/
theorem alphaGold_strictMono_in_c {N : ℝ} (hN : 1 < N) :
    StrictMono (alphaGold N) := by
  intro c₁ c₂ hc
  -- Reduce alphaGold N c₁ < alphaGold N c₂ to N^c₁ < N^c₂.
  --
  -- Chain of equivalences (all terms positive for N > 1):
  --   alphaGold N c₁ < alphaGold N c₂
  -- =  1 / (1 + (N-1)·N^(-c₁)) < 1 / (1 + (N-1)·N^(-c₂))
  -- ⟺ 1 + (N-1)·N^(-c₂) < 1 + (N-1)·N^(-c₁)   [reciprocal reverses order]
  -- ⟺ (N-1)·N^(-c₂) < (N-1)·N^(-c₁)            [cancel the +1]
  -- ⟺ N^(-c₂) < N^(-c₁)                        [divide by N-1 > 0]
  -- ⟺ N^c₁ < N^c₂                              [reciprocal reverses order]
  have hNpos : 0 < N := by linarith
  have hNnle : 0 ≤ N := le_of_lt hNpos
  have hNm1_pos : 0 < N - 1 := by linarith
  have hNpow_c1_pos : 0 < N^(-c₁) := Real.rpow_pos_of_pos hNpos _
  have hNpow_c2_pos : 0 < N^(-c₂) := Real.rpow_pos_of_pos hNpos _
  have hNpow_c1_pos' : 0 < N^(c₁) := Real.rpow_pos_of_pos hNpos _
  have hNpow_c2_pos' : 0 < N^(c₂) := Real.rpow_pos_of_pos hNpos _
  -- `N^(-c) = (N^c)⁻¹` for `N ≥ 0` (Mathlib `Real.rpow_neg`).
  have hneg1 : N^(-c₁) = (N^(c₁))⁻¹ := Real.rpow_neg hNnle c₁
  have hneg2 : N^(-c₂) = (N^(c₂))⁻¹ := Real.rpow_neg hNnle c₂
  -- `N^c₁ < N^c₂` from the strict monotonicity of `N^c`.
  have hkey : N^(c₁) < N^(c₂) := strictMono_rpow_of_gt_one hN hc
  -- Chain: N^c₁ < N^c₂  ⟹ (N^c₂)⁻¹ < (N^c₁)⁻¹  ⟺ N^(-c₂) < N^(-c₁).
  -- `inv_lt_inv₀` (field version of `inv_lt_inv_iff` with positivity hypotheses).
  have hrecip : (N^(c₂))⁻¹ < (N^(c₁))⁻¹ :=
    (inv_lt_inv₀ hNpow_c2_pos' hNpow_c1_pos').mpr hkey
  rw [← hneg1, ← hneg2] at hrecip
  -- hrecip : N^(-c₂) < N^(-c₁). Scale by (N-1) > 0, shift by 1.
  have hscaled : (N - 1) * N^(-c₂) < (N - 1) * N^(-c₁) :=
    mul_lt_mul_of_pos_left hrecip hNm1_pos
  have hshifted : 1 + (N - 1) * N^(-c₂) < 1 + (N - 1) * N^(-c₁) := by linarith
  -- Denominators are positive.
  have hdenom_c1_pos : 0 < 1 + (N - 1) * N^(-c₁) := by
    have : 0 < (N - 1) * N^(-c₁) := mul_pos hNm1_pos hNpow_c1_pos
    linarith
  have hdenom_c2_pos : 0 < 1 + (N - 1) * N^(-c₂) := by
    have : 0 < (N - 1) * N^(-c₂) := mul_pos hNm1_pos hNpow_c2_pos
    linarith
  -- Final step: denom₂ < denom₁ ⟹ 1/denom₁ < 1/denom₂ (`one_div_lt_one_div`).
  show alphaGold N c₁ < alphaGold N c₂
  rw [alphaGold, alphaGold]
  exact (one_div_lt_one_div hdenom_c1_pos hdenom_c2_pos).mpr hshifted

/-- Direct (non-`StrictMono`) form of the monotonicity: for `N > 1` and
    `c₁ < c₂`, the gold mass at `c₁` is strictly less than at `c₂`. -/
theorem alphaGold_lt_of_c_lt {N c₁ c₂ : ℝ} (hN : 1 < N) (hc : c₁ < c₂) :
    alphaGold N c₁ < alphaGold N c₂ := by
  exact alphaGold_strictMono_in_c hN hc

/-! ## Theorem 2: SSMax dominates base when `s_L · log(N) ≥ 1`

SSMax replaces the base exponent `c_base = s · Δ` with `c_SSMax = s_L · log(N) ·
s · Δ`. By Theorem 1, `alphaGold` increases iff `c` increases, i.e. iff
`s_L · log(N) · s · Δ ≥ s · Δ`. For `s · Δ > 0` this reduces to
`s_L · log(N) ≥ 1`.

The plan's informal threshold was `s_L = 1, N ≥ 2`, but `log(2) ≈ 0.693 < 1`,
so SSMax at `s_L = 1, N = 2` is in fact *milder* than base. The correct threshold
is `s_L · log(N) ≥ 1` (i.e. `N ≥ 3` for `s_L = 1`).
-/

/-- **SSMax dominates base.** For `N > 1`, `s_L · log(N) ≥ 1`, and a positive
    base exponent `c_base > 0`: applying SSMax (replacing `c_base` with
    `s_L · log(N) · c_base`) does not decrease the gold attention mass, and
    strictly increases it when `s_L · log(N) > 1`.

    The hypothesis `s_L · log(N) ≥ 1` is tight: at `s_L = 1, N = 2` we have
    `log(2) < 1`, so SSMax is milder than base and the inequality reverses. -/
theorem ssmax_dominates_base
    {N s_L c_base : ℝ} (hN : 1 < N) (h_log : s_L * Real.log N ≥ 1)
    (h_cbase : 0 < c_base) :
    alphaGold N (s_L * Real.log N * c_base) ≥ alphaGold N c_base := by
  -- By Theorem 1 (StrictMono), alphaGold is strictly increasing in c.
  -- So it suffices to show s_L · log N · c_base ≥ c_base, i.e. s_L · log N ≥ 1
  -- (dividing by c_base > 0).
  by_cases h : s_L * Real.log N = 1
  · -- Equality: c_SSMax = 1 · c_base = c_base, so alphaGold is equal.
    rw [h, one_mul]
  · -- Strict: s_L · log N > 1 (given ≥ 1 and ≠ 1).
    have h_strict : 1 < s_L * Real.log N := by
      rcases lt_or_gt_of_ne h with hlt | hgt
      · linarith
      · exact hgt
    -- c_SSMax = (s_L · log N) · c_base > 1 · c_base = c_base.
    have h_cSSMax : c_base < s_L * Real.log N * c_base :=
      lt_mul_of_one_lt_left h_cbase h_strict
    -- StrictMono gives strict <, which implies ≥.
    exact le_of_lt (alphaGold_lt_of_c_lt hN h_cSSMax)

end KatgptProof.Ssmax
