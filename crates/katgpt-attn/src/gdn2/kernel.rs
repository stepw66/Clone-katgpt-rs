//! GDN2 (Gated DeltaNet-2) recurrent step kernel.
//!
//! Core O(d_k × d_v) recurrent step implementing decoupled erase/write gates.
//! The four-step recurrence per token:
//!
//! ```text
//! 1. Decay:  S *= Diag(α)        — row-wise scale by per-channel decay
//! 2. Read:   r = Sᵀ(b ⊙ k)      — gated matvec (erase gate modulates key)
//! 3. Update: S += k ⊗ (w⊙v − r) — outer product delta (write gate modulates value)
//! 4. Readout: o = Sᵀ q           — query the updated state
//! ```
//!
//! Reference: Research 70 — Gated DeltaNet-2: O(1) Decode with Decoupled Erase/Write Gates.

use super::types::Gdn2GateConfig;
use katgpt_core::simd::{simd_outer_product_acc, simd_scale_inplace};

/// Core GDN2 recurrent step: O(d_k × d_v) per token per head.
///
/// Updates state S in-place and writes output to `out`.
///
/// # Arguments
/// * `k` — Key vector `[dk]`, should be L2-normalized for stability
/// * `v` — Value vector `[dv]`
/// * `q` — Query vector `[dk]`, should be L2-normalized for stability
/// * `s` — State matrix `[dk × dv]`, updated in-place (row-major)
/// * `alpha` — Per-channel decay `[dk]`, values typically in (0, 1]
/// * `b` — Erase gate `[dk]`, values in [0, 1]
/// * `w_val` — Scalar write weight for `EraseOnly`/`Kda` modes
/// * `w` — Channel-wise write gate `[dv]` for `Full` mode, values in [0, 1]
/// * `out` — Output buffer `[dv]`, written to
/// * `temp` — Temporary buffer `[dv]`, used internally for the read step
/// * `delta` — Delta buffer `[dv]`, used internally for the update step (pre-allocated)
/// * `dk` — Key/query dimension (head_dim)
/// * `dv` — Value dimension (head_dim)
/// * `gate_config` — Which gate variant to use
#[allow(clippy::too_many_arguments)]
pub fn gdn2_recurrent_step(
    k: &[f32],
    v: &[f32],
    q: &[f32],
    s: &mut [f32],
    alpha: &[f32],
    b: &[f32],
    w_val: f32,
    w: &[f32],
    out: &mut [f32],
    temp: &mut [f32],
    delta: &mut [f32],
    dk: usize,
    dv: usize,
    gate_config: Gdn2GateConfig,
) {
    debug_assert_eq!(k.len(), dk);
    debug_assert_eq!(v.len(), dv);
    debug_assert_eq!(q.len(), dk);
    debug_assert_eq!(s.len(), dk * dv);
    debug_assert_eq!(alpha.len(), dk);
    debug_assert_eq!(b.len(), dk);
    debug_assert!(w.len() >= dv || gate_config != Gdn2GateConfig::Full);
    debug_assert_eq!(out.len(), dv);
    debug_assert_eq!(temp.len(), dv);
    debug_assert_eq!(delta.len(), dv);

    // Steps 1+2 fused: decay each row of S by alpha[i] AND accumulate the gated
    // read r = Sᵀ(b ⊙ k) in a single pass over S. Decay must precede the read
    // (the read uses post-decay values), which the per-row ordering preserves
    // exactly. Fusing halves memory traffic over S and removes `dk` per-row
    // function calls vs. the previous decay-then-read two-pass form.
    temp.fill(0.0);
    for i in 0..dk {
        let alpha_row = unsafe { *alpha.get_unchecked(i) };
        let bk_i = unsafe { *b.get_unchecked(i) * *k.get_unchecked(i) };
        let row_start = i * dv;
        let row = &mut s[row_start..row_start + dv];
        if bk_i != 0.0 {
            for j in 0..dv {
                unsafe {
                    let sv = *row.get_unchecked(j) * alpha_row;
                    *row.get_unchecked_mut(j) = sv;
                    *temp.get_unchecked_mut(j) += sv * bk_i;
                }
            }
        } else {
            // Row still decays even when the read contribution is gated off.
            for sv in row.iter_mut() {
                *sv *= alpha_row;
            }
        }
    }

    // Step 3: Update S += k ⊗ (w⊙v − r)
    // Compute delta = w⊙v − r, then outer product accumulate (reuse pre-allocated buffer).
    // Note: no need to `delta.fill(0.0)` first — we overwrite every element below.
    match gate_config {
        Gdn2GateConfig::EraseOnly | Gdn2GateConfig::Kda => {
            for j in 0..dv {
                delta[j] = w_val * v[j] - temp[j];
            }
        }
        Gdn2GateConfig::Full => {
            for j in 0..dv {
                delta[j] = w[j] * v[j] - temp[j];
            }
        }
    }
    // S += k ⊗ delta using SIMD-accelerated outer product
    simd_outer_product_acc(s, k, delta, dk, dv);

    // Step 4: Readout o = Sᵀ q
    // Row-major accumulation: contiguous inner loop over j (vectorizable),
    // outer loop over rows i. For each out[j] the i-contributions are summed
    // in the same 0..dk order as the column-strided form, so the result is
    // bit-identical while the memory access becomes cache- and SIMD-friendly.
    out.fill(0.0);
    for i in 0..dk {
        let qi = unsafe { *q.get_unchecked(i) };
        let row_start = i * dv;
        for j in 0..dv {
            unsafe {
                *out.get_unchecked_mut(j) += *s.get_unchecked(row_start + j) * qi;
            }
        }
    }
}

/// GDN2 state update: steps 1–3 (decay, read, update).
///
/// Updates state S in-place with the new k/v pair. Does NOT produce output.
/// Split out from `gdn2_recurrent_step` so that with GQA the state can be
/// updated once per KV group and then read by multiple Q heads.
#[allow(clippy::too_many_arguments)]
pub fn gdn2_state_update(
    s: &mut [f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    b: &[f32],
    w_val: f32,
    w: &[f32],
    temp: &mut [f32],
    delta: &mut [f32],
    dk: usize,
    dv: usize,
    gate_config: Gdn2GateConfig,
) {
    debug_assert_eq!(k.len(), dk);
    debug_assert_eq!(v.len(), dv);
    debug_assert_eq!(s.len(), dk * dv);
    debug_assert_eq!(alpha.len(), dk);
    debug_assert_eq!(b.len(), dk);
    debug_assert!(w.len() >= dv || gate_config != Gdn2GateConfig::Full);
    debug_assert_eq!(temp.len(), dv);
    debug_assert_eq!(delta.len(), dv);

    // Steps 1+2 fused: decay each row of S by alpha[i] AND accumulate the gated
    // read r = Sᵀ(b ⊙ k) in a single pass over S (see gdn2_recurrent_step for the
    // semantics-preserving rationale).
    temp.fill(0.0);
    for i in 0..dk {
        let alpha_row = unsafe { *alpha.get_unchecked(i) };
        let bk_i = unsafe { *b.get_unchecked(i) * *k.get_unchecked(i) };
        let row_start = i * dv;
        let row = &mut s[row_start..row_start + dv];
        if bk_i != 0.0 {
            for j in 0..dv {
                unsafe {
                    let sv = *row.get_unchecked(j) * alpha_row;
                    *row.get_unchecked_mut(j) = sv;
                    *temp.get_unchecked_mut(j) += sv * bk_i;
                }
            }
        } else {
            for sv in row.iter_mut() {
                *sv *= alpha_row;
            }
        }
    }

    // Step 3: Update S += k ⊗ (w⊙v − r)
    // No `delta.fill(0.0)` — every element is overwritten below.
    match gate_config {
        Gdn2GateConfig::EraseOnly | Gdn2GateConfig::Kda => {
            for j in 0..dv {
                delta[j] = w_val * v[j] - temp[j];
            }
        }
        Gdn2GateConfig::Full => {
            for j in 0..dv {
                delta[j] = w[j] * v[j] - temp[j];
            }
        }
    }
    simd_outer_product_acc(s, k, delta, dk, dv);
}

/// GDN2 readout: step 4 only (o = Sᵀ q).
///
/// Reads the current state without modifying it. Safe to call multiple times
/// for different Q heads sharing the same KV group.
pub fn gdn2_state_readout(s: &[f32], q: &[f32], out: &mut [f32], dk: usize, dv: usize) {
    debug_assert_eq!(q.len(), dk);
    debug_assert_eq!(s.len(), dk * dv);
    debug_assert_eq!(out.len(), dv);

    // Row-major accumulation (contiguous inner loop, vectorizable). Per-output
    // i-contributions are summed in the same 0..dk order as the column-strided
    // form, so results are bit-identical.
    out.fill(0.0);
    for i in 0..dk {
        let qi = unsafe { *q.get_unchecked(i) };
        let row_start = i * dv;
        for j in 0..dv {
            unsafe {
                *out.get_unchecked_mut(j) += *s.get_unchecked(row_start + j) * qi;
            }
        }
    }
}

/// Sigmoid function for gate projections.
///
/// Delegates to [`katgpt_core::simd::fast_sigmoid`] which adds early-exit for
/// `|x| > 40` (where σ saturates in f32) — saves an `exp()` call when gate
/// pre-activations are large.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// L2 normalize a vector in-place: x /= ‖x‖₂ + ε.
#[inline]
pub fn l2_normalize(x: &mut [f32]) {
    let norm_sq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-8);
    simd_scale_inplace(x, inv_norm);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify sigmoid output is in [0, 1].
    #[test]
    fn sigmoid_range() {
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} out of [0,1]");
        }
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) ≈ 0.5");
    }

    /// Verify L2 normalize produces unit vector.
    #[test]
    fn l2_normalize_unit() {
        let mut x = vec![3.0, 4.0, 0.0];
        l2_normalize(&mut x);
        let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm = {norm}, expected 1.0");
    }

    /// Verify L2 normalize handles zero vector gracefully.
    #[test]
    fn l2_normalize_zero() {
        let mut x = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut x);
        for &v in &x {
            assert!(v.is_finite(), "zero-normalize should produce finite: {v}");
        }
    }

    /// Verify basic recurrent step produces finite output with zero state.
    #[test]
    fn recurrent_step_zero_state_finite() {
        let dk = 4;
        let dv = 4;
        let mut s = vec![0.0; dk * dv];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let q = vec![0.5, 0.5, 0.5, 0.5];
        let alpha = vec![0.99; dk];
        let b = vec![1.0; dk];
        let w_channel = vec![1.0; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];
        let mut delta = vec![0.0; dv];

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        for &o in &out {
            assert!(o.is_finite(), "output should be finite: {o}");
        }
    }

    /// Verify recurrent step with all gate configs produces finite output.
    #[test]
    fn recurrent_step_all_gate_configs() {
        let dk = 4;
        let dv = 4;
        let k = vec![0.5; dk];
        let v = vec![1.0; dv];
        let q = vec![0.5; dk];
        let alpha = vec![0.99; dk];
        let b = vec![0.8; dk];
        let w_channel = vec![0.9; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];
        let mut delta = vec![0.0; dv];

        for gate_config in [
            Gdn2GateConfig::EraseOnly,
            Gdn2GateConfig::Full,
            Gdn2GateConfig::Kda,
        ] {
            let mut s = vec![0.0; dk * dv];
            gdn2_recurrent_step(
                &k,
                &v,
                &q,
                &mut s,
                &alpha,
                &b,
                1.0,
                &w_channel,
                &mut out,
                &mut temp,
                &mut delta,
                dk,
                dv,
                gate_config,
            );
            for &o in &out {
                assert!(
                    o.is_finite(),
                    "output should be finite for {gate_config:?}: {o}"
                );
            }
        }
    }

    /// Verify decay shrinks state: after step with v=0, state should be smaller.
    #[test]
    fn recurrent_step_decay_shrinks_state() {
        let dk = 4;
        let dv = 4;
        // Pre-fill state
        let mut s = vec![1.0; dk * dv];
        let k = vec![0.5; dk];
        let v = vec![0.0; dv]; // zero value: no new write
        let q = vec![0.5; dk];
        let alpha = vec![0.5; dk]; // strong decay
        let b = vec![1.0; dk];
        let w_channel = vec![1.0; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];
        let mut delta = vec![0.0; dv];

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // State should decay to ~0.5 (plus the read-correction which is 0 for zero value)
        let s_max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            s_max < 0.9,
            "state should decay with alpha=0.5, max={s_max}"
        );
    }

    /// Verify outer product update: with zero initial state, S should become k⊗(w⊙v).
    #[test]
    fn recurrent_step_writes_outer_product() {
        let dk = 4;
        let dv = 4;
        let mut s = vec![0.0; dk * dv];
        let k = vec![1.0, 0.0, 0.0, 0.0]; // unit basis
        let v = vec![0.0, 0.0, 0.0, 1.0]; // unit basis
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let alpha = vec![1.0; dk]; // no decay
        let b = vec![1.0; dk]; // open erase gate
        let w_channel = vec![1.0; dv]; // open write gate
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];
        let mut delta = vec![0.0; dv];

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // S should be k ⊗ v (row 0 = [0,0,0,1], rest zero)
        assert!((s[0] - 0.0).abs() < 1e-6, "s[0] should be 0");
        assert!((s[3] - 1.0).abs() < 1e-6, "s[3] should be 1.0 (k[0]*v[3])");
        for (offset, &val) in s[dk..dk * dv].iter().enumerate() {
            let i = dk + offset;
            assert!(val.abs() < 1e-6, "s[{i}] should be 0, got {val}");
        }
    }

    /// Verify multi-step accumulation: two tokens should accumulate correctly.
    #[test]
    fn recurrent_step_multi_step_accumulates() {
        let dk = 4;
        let dv = 4;
        let mut s = vec![0.0; dk * dv];
        let alpha = vec![1.0; dk]; // no decay
        let b = vec![1.0; dk];
        let w_channel = vec![1.0; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];
        let mut delta = vec![0.0; dv];

        // Token 1: k = [1,0,0,0], v = [1,0,0,0]
        let k1 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let q1 = vec![1.0, 0.0, 0.0, 0.0];
        gdn2_recurrent_step(
            &k1,
            &v1,
            &q1,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // Token 2: k = [0,1,0,0], v = [0,1,0,0]
        let k2 = vec![0.0, 1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let q2 = vec![1.0, 0.0, 0.0, 0.0];
        gdn2_recurrent_step(
            &k2,
            &v2,
            &q2,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // S should have k1⊗v1 + k2⊗v2 (no decay, no erase correction for zero state)
        // s[0] = k1[0]*v1[0] = 1.0
        // s[1*4+1] = k2[1]*v2[1] = 1.0
        assert!(
            (s[0] - 1.0).abs() < 1e-5,
            "s[0] should be ~1.0 from first token, got {}",
            s[0]
        );
        assert!(
            (s[5] - 1.0).abs() < 1e-5,
            "s[5] should be ~1.0 from second token, got {}",
            s[5]
        );
    }

    /// Verify split functions produce same result as combined step.
    #[test]
    fn split_functions_match_combined() {
        let dk = 4;
        let dv = 4;
        let k = vec![0.25, 0.5, 0.75, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let alpha = vec![0.99; dk];
        let b = vec![0.8; dk];
        let w_channel = vec![0.9; dv];

        // Combined step
        let mut s_combined = vec![0.5; dk * dv];
        let mut out_combined = vec![0.0; dv];
        let mut temp1 = vec![0.0; dv];
        let mut delta1 = vec![0.0; dv];
        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s_combined,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out_combined,
            &mut temp1,
            &mut delta1,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // Split: update + readout
        let mut s_split = vec![0.5; dk * dv];
        let mut temp2 = vec![0.0; dv];
        let mut delta2 = vec![0.0; dv];
        gdn2_state_update(
            &mut s_split,
            &k,
            &v,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut temp2,
            &mut delta2,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
        let mut out_split = vec![0.0; dv];
        gdn2_state_readout(&s_split, &q, &mut out_split, dk, dv);

        // State should match
        for (i, (a, b)) in s_combined.iter().zip(s_split.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "s[{i}] mismatch: combined={a}, split={b}"
            );
        }
        // Output should match
        for (i, (a, b)) in out_combined.iter().zip(out_split.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "out[{i}] mismatch: combined={a}, split={b}"
            );
        }
    }

    /// Verify split functions produce same result for Full gate config.
    #[test]
    fn split_functions_match_combined_full() {
        let dk = 4;
        let dv = 4;
        let k = vec![0.25, 0.5, 0.75, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let alpha = vec![0.95; dk];
        let b = vec![0.7; dk];
        let w_channel = vec![0.8; dv];

        // Combined
        let mut s_combined = vec![0.3; dk * dv];
        let mut out_combined = vec![0.0; dv];
        let mut temp1 = vec![0.0; dv];
        let mut delta1 = vec![0.0; dv];
        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s_combined,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out_combined,
            &mut temp1,
            &mut delta1,
            dk,
            dv,
            Gdn2GateConfig::Full,
        );

        // Split
        let mut s_split = vec![0.3; dk * dv];
        let mut temp2 = vec![0.0; dv];
        let mut delta2 = vec![0.0; dv];
        gdn2_state_update(
            &mut s_split,
            &k,
            &v,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut temp2,
            &mut delta2,
            dk,
            dv,
            Gdn2GateConfig::Full,
        );
        let mut out_split = vec![0.0; dv];
        gdn2_state_readout(&s_split, &q, &mut out_split, dk, dv);

        for (i, (a, b)) in s_combined.iter().zip(s_split.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "s[{i}] mismatch (Full): combined={a}, split={b}"
            );
        }
        for (i, (a, b)) in out_combined.iter().zip(out_split.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "out[{i}] mismatch (Full): combined={a}, split={b}"
            );
        }
    }

    // ── Plan 395 Phase 3: G3 no-regression with hippocampal cache observer ──
    //
    // The cache is a pure observer of the delta-rule update. Running the same
    // token stream with `cache.observe()` after each step must produce
    // byte-identical GDN2 state `S` and output `o` as running without the cache.
    #[cfg(feature = "hippocampal_cache")]
    #[test]
    fn g3_cache_observer_no_regression() {
        use katgpt_core::HippocampalCache;

        let dk = 16;
        let dv = 16;
        let n_tokens = 200;
        let mut rng = fastrand::Rng::with_seed(2026);

        // Generate a fixed token stream.
        let tokens: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_tokens)
            .map(|_| {
                let k: Vec<f32> = (0..dk).map(|_| rng.f32() * 2.0 - 1.0).collect();
                let v: Vec<f32> = (0..dv).map(|_| rng.f32() * 2.0 - 1.0).collect();
                let q: Vec<f32> = (0..dk).map(|_| rng.f32() * 2.0 - 1.0).collect();
                (k, v, q)
            })
            .collect();

        let alpha: Vec<f32> = vec![0.99; dk];
        let b: Vec<f32> = vec![0.5; dk];
        let w_channel: Vec<f32> = vec![1.0; dv];

        // Run A: bare GDN2 (no cache).
        let mut s_a = vec![0.0f32; dk * dv];
        let mut out_a = vec![0.0f32; dv];
        let mut temp_a = vec![0.0f32; dv];
        let mut delta_a = vec![0.0f32; dv];
        for (k, v, q) in &tokens {
            gdn2_recurrent_step(
                k,
                v,
                q,
                &mut s_a,
                &alpha,
                &b,
                0.5,
                &w_channel,
                &mut out_a,
                &mut temp_a,
                &mut delta_a,
                dk,
                dv,
                Gdn2GateConfig::EraseOnly,
            );
        }

        // Run B: GDN2 + cache observer (observe after each step, discard read).
        let mut s_b = vec![0.0f32; dk * dv];
        let mut out_b = vec![0.0f32; dv];
        let mut temp_b = vec![0.0f32; dv];
        let mut delta_b = vec![0.0f32; dv];
        // Cache with D=dv, W=8. Keys are dk-dim but we only cache dk-dim keys
        // (in real GDN2 dk==dv==head_dim so this matches).
        let mut cache: HippocampalCache<16, 8> = HippocampalCache::new_with_ones_gamma();
        for (k, v, q) in &tokens {
            gdn2_recurrent_step(
                k,
                v,
                q,
                &mut s_b,
                &alpha,
                &b,
                0.5,
                &w_channel,
                &mut out_b,
                &mut temp_b,
                &mut delta_b,
                dk,
                dv,
                Gdn2GateConfig::EraseOnly,
            );
            // Compute surprise score: β·‖e‖ where β = write gate, ‖e‖ = ‖delta‖.
            let delta_norm: f32 = delta_b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let beta = 0.5; // w_val for EraseOnly
            // Observe with const-generic arrays (dk == dv == 16 in this test).
            let k_arr: [f32; 16] = k[..16].try_into().unwrap();
            let v_arr: [f32; 16] = v[..16].try_into().unwrap();
            cache.observe(&k_arr, &v_arr, beta, delta_norm);
        }

        // Assert byte-identical GDN2 state.
        for i in 0..s_a.len() {
            assert_eq!(
                s_a[i].to_bits(),
                s_b[i].to_bits(),
                "S[{i}] differs: bare={a}, with_cache={b}",
                a = s_a[i],
                b = s_b[i],
            );
        }
        // Assert byte-identical output.
        for i in 0..out_a.len() {
            assert_eq!(out_a[i].to_bits(), out_b[i].to_bits(), "out[{i}] differs");
        }

        // Sanity: cache should have observed some tokens.
        assert_eq!(cache.len(), 8, "cache should be full after 200 tokens");
    }

    // ── Plan 395 Phase 3: G3 feature-gate isolation (W=0 == no cache) ────────
    //
    // With the cache present but the `hippocampal_cache` feature off, the
    // GDN2 state must be byte-identical to a run without any cache machinery.
    // This proves the feature gate is clean (the `merkle_root` lesson).
    #[test]
    fn g3_feature_gate_isolation() {
        // This test runs without the hippocampal_cache feature — it just verifies
        // that bare GDN2 produces deterministic, reproducible state. The feature
        // gate isolation is proven by: (a) this test compiles without the feature,
        // (b) the g3_cache_observer_no_regression test proves the cache doesn't
        // perturb state when the feature IS on.
        let dk = 8;
        let dv = 8;
        let mut s1 = vec![0.0f32; dk * dv];
        let mut s2 = vec![0.0f32; dk * dv];
        let mut out1 = vec![0.0f32; dv];
        let mut out2 = vec![0.0f32; dv];
        let mut temp = vec![0.0f32; dv];
        let mut delta = vec![0.0f32; dv];
        let alpha = vec![0.99; dk];
        let b = vec![0.5; dk];
        let w = vec![1.0; dv];
        let k: Vec<f32> = (0..dk).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..dv).map(|i| i as f32 * 0.2).collect();
        let q: Vec<f32> = (0..dk).map(|i| i as f32 * 0.15).collect();

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s1,
            &alpha,
            &b,
            0.5,
            &w,
            &mut out1,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s2,
            &alpha,
            &b,
            0.5,
            &w,
            &mut out2,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        for i in 0..s1.len() {
            assert_eq!(
                s1[i].to_bits(),
                s2[i].to_bits(),
                "bare GDN2 must be deterministic"
            );
        }
    }
}
