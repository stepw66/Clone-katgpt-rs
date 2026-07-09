//! MAG direction mining: mean-shift extraction, contrast directions, linearity
//! diagnostics, calibrated steering strength, and the 8-operator readout family.
//!
//! These are the **unsupervised acquisition** primitives — they mine direction
//! vectors from the host's own runtime verdicts (`y_M`), not human labels. The
//! mined directions are then consumed by the injection side (`LatentSteering`,
//! `PersonalityWeightedComposition`, `CommittedFieldBlend`, `SphericalSteering`)
//! to close the acquisition loop: NPCs discover their own reasoning directions
//! from experience.

use super::types::{
    check_dim, compute_direction_commitment, normalize_in_place, MagDirection, MagError,
    MagOperator,
};

// ── Mean-shift direction ───────────────────────────────────────────

/// Mine a mean-shift direction `v_Q = normalize(mean(with_prefix) − mean(without_prefix))`.
///
/// Given paired activation readouts — with a prefix/transformation Q applied
/// (`with_prefix`, i.e. `m(Q‖p)`) and without (`without_prefix`, i.e. `m(p)`) —
/// the mean shift is a feature direction (arXiv:2607.04222 §2.1).
///
/// Returns a unit-norm [`MagDirection`] with `recon_error` and `cosine` set to
/// `NaN`; populate them via [`reconstruction_error`] if you need the linearity
/// diagnostic.
///
/// Both sets must share the same per-sample dimensionality `d` (the two sets may
/// have different sample counts — the means are computed with the respective
/// denominators). Returns [`MagError::Empty`] if either set is empty,
/// [`MagError::DimMismatch`] if dimensionalities differ, or
/// [`MagError::ZeroNorm`] if the mean shift is the zero vector.
///
/// # Example
///
/// ```
/// # use katgpt_core::mag::mine_direction;
/// // 3 samples each, d=2. With-prefix adds [+1, 0] to every without-prefix.
/// let with: Vec<[f32; 2]>    = vec![[2.0, 0.0], [3.0, 1.0], [1.0, -1.0]];
/// let without: Vec<[f32; 2]> = vec![[1.0, 0.0], [2.0, 1.0], [0.0, -1.0]];
/// let dir = mine_direction(&with, &without).unwrap();
/// // mean(with) − mean(without) = [1, 0] → normalized [1, 0].
/// assert!((dir.as_slice()[0] - 1.0).abs() < 1e-5);
/// assert!(dir.as_slice()[1].abs() < 1e-5);
/// ```
pub fn mine_direction<S: AsRef<[f32]>>(
    with_prefix: &[S],
    without_prefix: &[S],
) -> Result<MagDirection, MagError> {
    let d = check_dim(with_prefix)?;
    let d2 = check_dim(without_prefix)?;
    if d != d2 {
        return Err(MagError::DimMismatch);
    }

    let mut diff = vec![0.0_f32; d];
    let inv_n_with = 1.0 / with_prefix.len() as f32;
    for s in with_prefix {
        let s = s.as_ref();
        for (acc, &v) in diff.iter_mut().zip(s) {
            *acc += v * inv_n_with;
        }
    }
    let inv_n_without = 1.0 / without_prefix.len() as f32;
    for s in without_prefix {
        let s = s.as_ref();
        for (acc, &v) in diff.iter_mut().zip(s) {
            *acc -= v * inv_n_without;
        }
    }

    finalize_direction(diff)
}

/// Mine a contrast direction `u_Q = normalize(mean(negative) − mean(positive))`.
///
/// The `positive` and `negative` sets are partitioned by the host's own verdict
/// `y_M` (**not** human labels) — the runtime's binary observable (did the NPC
/// succeed? did the claim pass the rubric? did the action hit the target?). The
/// contrast direction points from the positive class toward the negative class.
///
/// This is the **headline unsupervised acquisition** step: the label comes from
/// the runtime itself, making this label-free direction mining
/// (arXiv:2607.04222 §2.3). The GOAT gate G2 (Plan 418) checks that
/// self-labeled classes produce separable directions — if not, the primitive
/// demotes to a research-only Gain.
///
/// Returns the same error variants as [`mine_direction`].
///
/// # Example
///
/// ```
/// # use katgpt_core::mag::mine_contrast_direction;
/// // positive class centered at origin, negative at [+2, 0].
/// let pos: Vec<[f32; 2]> = vec![[0.0, 0.1], [-0.1, 0.0], [0.05, 0.0]];
/// let neg: Vec<[f32; 2]> = vec![[2.0, 0.1], [1.9, 0.0], [2.1, -0.1]];
/// let dir = mine_contrast_direction(&pos, &neg).unwrap();
/// // mean(neg) − mean(pos) ≈ [2, 0] → normalized ≈ [1, 0].
/// assert!((dir.as_slice()[0] - 1.0).abs() < 0.1);
/// ```
pub fn mine_contrast_direction<S: AsRef<[f32]>>(
    positive: &[S],
    negative: &[S],
) -> Result<MagDirection, MagError> {
    // contrast u_Q = mean(negative) − mean(positive) = mine_direction(negative, positive).
    mine_direction(negative, positive)
}

/// Zero-alloc hot-path variant of [`mine_direction`].
///
/// Writes the unit-normalized mean-shift direction into `out[0..d]`. Returns the
/// pre-normalization norm. Does **not** compute the BLAKE3 commitment or create a
/// [`MagDirection`] — for hot paths that reuse a scratch buffer and only need the
/// direction vector. Use [`mine_direction`] when you need the full committed
/// artifact.
///
/// # Errors
///
/// Same as [`mine_direction`]. Additionally returns [`MagError::DimMismatch`] if
/// `out.len() < d`.
pub fn mine_direction_into<S: AsRef<[f32]>>(
    with_prefix: &[S],
    without_prefix: &[S],
    out: &mut [f32],
) -> Result<f32, MagError> {
    let d = check_dim(with_prefix)?;
    let d2 = check_dim(without_prefix)?;
    if d != d2 {
        return Err(MagError::DimMismatch);
    }
    if out.len() < d {
        return Err(MagError::DimMismatch);
    }
    let out = &mut out[..d];

    out.fill(0.0);
    let inv_n_with = 1.0 / with_prefix.len() as f32;
    for s in with_prefix {
        let s = s.as_ref();
        for (acc, &v) in out.iter_mut().zip(s) {
            *acc += v * inv_n_with;
        }
    }
    let inv_n_without = 1.0 / without_prefix.len() as f32;
    for s in without_prefix {
        let s = s.as_ref();
        for (acc, &v) in out.iter_mut().zip(s) {
            *acc -= v * inv_n_without;
        }
    }

    let pre_norm = normalize_in_place(out);
    if pre_norm == 0.0 {
        return Err(MagError::ZeroNorm);
    }
    Ok(pre_norm)
}

/// Build the final [`MagDirection`] from a raw (un-normalized) direction buffer:
/// normalize, BLAKE3-commit, and wrap.
fn finalize_direction(mut diff: Vec<f32>) -> Result<MagDirection, MagError> {
    let pre_norm = normalize_in_place(&mut diff);
    if pre_norm == 0.0 {
        return Err(MagError::ZeroNorm);
    }
    let blake3 = compute_direction_commitment(&diff);
    Ok(MagDirection {
        direction: diff.into_boxed_slice(),
        recon_error: f32::NAN,
        cosine: f32::NAN,
        blake3,
    })
}

// ── Linearity diagnostic ───────────────────────────────────────────

/// Compute the linearity diagnostic (ϵ_Q, mean cosine) for a mined direction.
///
/// For each sample `i`, the actual shift is `Δ_i = m(Q‖p_i) − m(p_i)` and the
/// predicted shift is `α · direction`. The reconstruction error is:
///
/// ```text
/// ϵ_Q = E_i[‖Δ_i − α·direction‖²] / E_i[‖Δ_i‖²]
/// ```
///
/// - `ϵ_Q ≈ 0`: the shift is well-approximated by a single direction (steerable).
/// - `ϵ_Q ≈ 1`: the direction explains none of the shift variance on average
///   (or there is no shift).
/// - `ϵ_Q > 1`: overshoot — the predicted shift exceeds the actual shift.
///
/// The returned `cosine` is the mean per-sample cosine of the predicted shift vs
/// the actual shift (high ⇒ the direction aligns with individual shifts, not
/// just their mean).
///
/// The two sample sets must have the **same length** (paired samples). Returns
/// `(recon_error, cosine)` — use [`MagDirection::with_diagnostics`] to stamp
/// these onto a mined direction.
pub fn reconstruction_error<S: AsRef<[f32]>>(
    with_prefix: &[S],
    without_prefix: &[S],
    direction: &[f32],
    alpha: f32,
) -> Result<(f32, f32), MagError> {
    let d = check_dim(with_prefix)?;
    let d2 = check_dim(without_prefix)?;
    if d != d2 || direction.len() != d {
        return Err(MagError::DimMismatch);
    }
    if with_prefix.len() != without_prefix.len() {
        return Err(MagError::DimMismatch);
    }

    let n = with_prefix.len() as f32;
    let mut sum_num = 0.0_f32; // Σ ‖residual‖²
    let mut sum_denom = 0.0_f32; // Σ ‖Δ‖²
    let mut sum_cos = 0.0_f32;

    for (wp, wo) in with_prefix.iter().zip(without_prefix.iter()) {
        let wp = wp.as_ref();
        let wo = wo.as_ref();
        let mut res_sq = 0.0;
        let mut delta_sq = 0.0;
        let mut dot_pred_delta = 0.0;
        let mut pred_sq = 0.0;
        for j in 0..d {
            let delta = wp[j] - wo[j];
            let pred = alpha * direction[j];
            let res = delta - pred;
            res_sq += res * res;
            delta_sq += delta * delta;
            dot_pred_delta += pred * delta;
            pred_sq += pred * pred;
        }
        sum_num += res_sq;
        sum_denom += delta_sq;
        let denom_cos = (pred_sq * delta_sq).sqrt();
        if denom_cos > 0.0 {
            sum_cos += dot_pred_delta / denom_cos;
        }
    }

    let recon_error = if sum_denom > 0.0 {
        sum_num / sum_denom
    } else {
        1.0 // no actual shift → ϵ = 1.0 by convention (direction explains nothing)
    };
    let mean_cos = sum_cos / n;

    Ok((recon_error, mean_cos))
}

// ── Calibrated steering strength ───────────────────────────────────

/// Calibrate steering strength `α(τ) = τ · ‖mean(with_prefix)‖ / ‖direction‖`.
///
/// The injection magnitude `α · ‖direction‖` is set to `τ` fraction of the mean
/// prefix-activation norm, making the steering strength invariant to the
/// direction's scale and the substrate's activation magnitude
/// (arXiv:2607.04222 §3.2). For a unit-norm direction (the output of
/// [`mine_direction`]), this simplifies to `α = τ · ‖mean(with_prefix)‖`.
///
/// Returns [`MagError::ZeroNorm`] if `direction` is the zero vector.
pub fn calibrate_alpha<S: AsRef<[f32]>>(
    tau: f32,
    with_prefix: &[S],
    direction: &[f32],
) -> Result<f32, MagError> {
    let d = check_dim(with_prefix)?;
    if direction.len() != d {
        return Err(MagError::DimMismatch);
    }

    let mut mean = vec![0.0_f32; d];
    let inv_n = 1.0 / with_prefix.len() as f32;
    for s in with_prefix {
        let s = s.as_ref();
        for (acc, &v) in mean.iter_mut().zip(s) {
            *acc += v * inv_n;
        }
    }
    let prefix_norm = super::types::norm(&mean);
    let dir_norm = super::types::norm(direction);

    if dir_norm == 0.0 {
        return Err(MagError::ZeroNorm);
    }
    Ok(tau * prefix_norm / dir_norm)
}

// ── Operator readouts ──────────────────────────────────────────────

/// Apply a [`MagOperator`] to single-sample activation readouts, writing into
/// `out` (zero-alloc hot path).
///
/// The readout arguments are the per-sample activation vectors under different
/// conditions (see [`MagOperator`] for which each operator uses):
/// - `a_p`: `m(p)` — plain input.
/// - `a_q`: `m(Q)` — prefix/question alone.
/// - `a_qp`: `m(Q‖p)` — prefix + input.
/// - `a_qpy`: `m(Q‖p, y_M)` — prefix + input + verdict.
/// - `a_y`: `m(p, y_M)` — input + verdict (no prefix).
/// - `a_empty`: `m(∅)` — empty/baseline (currently unused by any operator;
///   reserved for future operators).
/// - `a_eqp`: `m(E‖Q‖p)` — few-shot examples + prefix + input.
///
/// Operators that don't use a given readout accept an empty slice `&[]` for it.
/// All non-empty readouts must share the same dimensionality `d`; `out` must be
/// at least `d` elements. Returns `Err(MagError::DimMismatch)` on inconsistency
/// or `Err(MagError::Empty)` if a required readout for the chosen operator is
/// missing.
#[allow(clippy::too_many_arguments, reason = "each arg is a distinct activation readout the operator selects between; see doc comment")]
pub fn apply_operator_into(
    op: MagOperator,
    a_p: &[f32],
    a_q: &[f32],
    a_qp: &[f32],
    a_qpy: &[f32],
    a_y: &[f32],
    _a_empty: &[f32],
    a_eqp: &[f32],
    out: &mut [f32],
) -> Result<(), MagError> {
    let d = a_p.len();
    if d == 0 {
        return Err(MagError::Empty);
    }
    if out.len() < d {
        return Err(MagError::DimMismatch);
    }
    let out = &mut out[..d];

    // Helper: a secondary readout is either empty (unused) or d-wide.
    let check = |s: &[f32]| -> Result<(), MagError> {
        if !s.is_empty() && s.len() != d {
            return Err(MagError::DimMismatch);
        }
        Ok(())
    };
    check(a_q)?;
    check(a_qp)?;
    check(a_qpy)?;
    check(a_y)?;
    check(a_eqp)?;

    match op {
        MagOperator::Direct => {
            out.copy_from_slice(&a_p[..d]);
        }
        MagOperator::Prefixed => {
            if a_qp.is_empty() {
                return Err(MagError::Empty);
            }
            out.copy_from_slice(&a_qp[..d]);
        }
        MagOperator::Answered => {
            if a_qpy.is_empty() {
                return Err(MagError::Empty);
            }
            out.copy_from_slice(&a_qpy[..d]);
        }
        MagOperator::InputDelta => {
            if a_qp.is_empty() {
                return Err(MagError::Empty);
            }
            for j in 0..d {
                out[j] = a_qp[j] - a_p[j];
            }
        }
        MagOperator::QuestionDelta => {
            if a_qp.is_empty() || a_q.is_empty() {
                return Err(MagError::Empty);
            }
            for j in 0..d {
                out[j] = a_qp[j] - a_q[j];
            }
        }
        MagOperator::Interaction => {
            if a_qpy.is_empty() || a_qp.is_empty() || a_y.is_empty() {
                return Err(MagError::Empty);
            }
            for j in 0..d {
                out[j] = a_qpy[j] - a_qp[j] - a_y[j] + a_p[j];
            }
        }
        MagOperator::Verdict => {
            if a_y.is_empty() {
                return Err(MagError::Empty);
            }
            for j in 0..d {
                out[j] = a_y[j] - a_p[j];
            }
        }
        MagOperator::FewShot => {
            if a_eqp.is_empty() {
                return Err(MagError::Empty);
            }
            for j in 0..d {
                out[j] = a_eqp[j] - a_p[j];
            }
        }
    }
    Ok(())
}

/// Allocating convenience wrapper for [`apply_operator_into`]. Returns a fresh
/// `Vec<f32>` of length `d`. Prefer [`apply_operator_into`] on hot paths.
#[allow(clippy::too_many_arguments, reason = "each arg is a distinct activation readout the operator selects between; see doc comment")]
pub fn apply_operator(
    op: MagOperator,
    a_p: &[f32],
    a_q: &[f32],
    a_qp: &[f32],
    a_qpy: &[f32],
    a_y: &[f32],
    a_empty: &[f32],
    a_eqp: &[f32],
) -> Result<Vec<f32>, MagError> {
    let d = a_p.len();
    if d == 0 {
        return Err(MagError::Empty);
    }
    let mut out = vec![0.0; d];
    apply_operator_into(op, a_p, a_q, a_qp, a_qpy, a_y, a_empty, a_eqp, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn mine_direction_recovers_known_shift() {
        // with = without + [1, 0, 0]  →  mean shift = [1, 0, 0] → normalized [1,0,0]
        let with: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![1.0 + i as f32 * 0.01, i as f32 * 0.1, -i as f32 * 0.05])
            .collect();
        let without: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.01, i as f32 * 0.1, -i as f32 * 0.05])
            .collect();
        let dir = mine_direction(&with, &without).unwrap();
        assert_eq!(dir.dim(), 3);
        assert!(approx_eq(dir.as_slice()[0], 1.0, 1e-5));
        assert!(dir.as_slice()[1].abs() < 1e-5);
        assert!(dir.as_slice()[2].abs() < 1e-5);
    }

    #[test]
    fn mine_direction_unequal_counts() {
        // 3 with, 5 without — means should still be correct.
        let with: Vec<[f32; 1]> = vec![[10.0], [10.0], [10.0]];
        let without: Vec<[f32; 1]> = vec![[0.0], [0.0], [0.0], [0.0], [0.0]];
        let dir = mine_direction(&with, &without).unwrap();
        // mean(with)=10, mean(without)=0 → diff=10 → normalized [1.0]
        assert!(approx_eq(dir.as_slice()[0], 1.0, 1e-5));
    }

    #[test]
    fn mine_direction_zero_shift_errors() {
        let with: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0]];
        let without: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(mine_direction(&with, &without).unwrap_err(), MagError::ZeroNorm);
    }

    #[test]
    fn mine_direction_dim_mismatch_errors() {
        let with: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let without: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(mine_direction(&with, &without).unwrap_err(), MagError::DimMismatch);
    }

    #[test]
    fn mine_contrast_points_negative_to_positive() {
        // positive at [0,0], negative at [3,0] → contrast = mean(neg)−mean(pos) = [3,0]
        let pos: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]];
        let neg: Vec<[f32; 2]> = vec![[3.0, 0.0], [3.0, 0.1], [2.9, 0.0]];
        let dir = mine_contrast_direction(&pos, &neg).unwrap();
        assert!(approx_eq(dir.as_slice()[0], 1.0, 0.1));
        assert!(dir.as_slice()[1].abs() < 0.1);
    }

    #[test]
    fn reconstruction_error_linear_shift_is_zero() {
        // with = without + alpha*direction (exact linear shift)
        let direction = [0.6, 0.8]; // unit norm
        let alpha = 0.5;
        let without: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]];
        let with: Vec<[f32; 2]> = without
            .iter()
            .map(|w| [w[0] + alpha * direction[0], w[1] + alpha * direction[1]])
            .collect();
        let (eps, cos) = reconstruction_error(&with, &without, &direction, alpha).unwrap();
        assert!(approx_eq(eps, 0.0, 1e-4), "eps should be ~0, got {eps}");
        assert!(approx_eq(cos, 1.0, 1e-4), "cos should be ~1, got {cos}");
    }

    #[test]
    fn reconstruction_error_no_shift_is_one() {
        // with == without → no shift → eps = 1.0 by convention
        let data: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0]];
        let direction = [1.0, 0.0];
        let (eps, _cos) = reconstruction_error(&data, &data, &direction, 0.5).unwrap();
        assert!(approx_eq(eps, 1.0, 1e-4), "eps should be ~1, got {eps}");
    }

    #[test]
    fn reconstruction_error_overshoot_gt_one() {
        // true shift = [0.5, 0], but we predict 2*[1,0] = [2,0] → overshoot → eps > 1
        let without: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.0, 1.0]];
        let with: Vec<[f32; 2]> = vec![[0.5, 0.0], [0.5, 1.0]];
        let direction = [1.0, 0.0];
        let (eps, _cos) = reconstruction_error(&with, &without, &direction, 2.0).unwrap();
        assert!(eps > 1.0, "eps should be > 1 (overshoot), got {eps}");
    }

    #[test]
    fn calibrate_alpha_unit_direction() {
        // unit direction, prefix mean norm = 5 → alpha = tau * 5 / 1 = tau*5
        let with: Vec<[f32; 2]> = vec![[3.0, 4.0]]; // mean = [3,4], norm = 5
        let direction = [1.0, 0.0]; // unit norm
        let alpha = calibrate_alpha(0.1, &with, &direction).unwrap();
        assert!(approx_eq(alpha, 0.5, 1e-4), "alpha = 0.1 * 5 / 1 = 0.5, got {alpha}");
    }

    #[test]
    fn apply_operator_input_delta() {
        let a_p = [1.0, 2.0, 3.0];
        let a_qp = [4.0, 6.0, 9.0];
        let mut out = [0.0_f32; 3];
        apply_operator_into(
            MagOperator::InputDelta,
            &a_p,
            &[],
            &a_qp,
            &[],
            &[],
            &[],
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [3.0, 4.0, 6.0]);
    }

    #[test]
    fn apply_operator_interaction() {
        let a_p = [1.0, 0.0];
        let a_qp = [2.0, 0.0];
        let a_qpy = [5.0, 0.0];
        let a_y = [3.0, 0.0];
        let mut out = [0.0_f32; 2];
        apply_operator_into(
            MagOperator::Interaction,
            &a_p,
            &[],
            &a_qp,
            &a_qpy,
            &a_y,
            &[],
            &[],
            &mut out,
        )
        .unwrap();
        // 5 - 2 - 3 + 1 = 1
        assert_eq!(out, [1.0, 0.0]);
    }

    #[test]
    fn apply_operator_direct_is_passthrough() {
        let a_p = [7.0, 8.0];
        let mut out = [0.0_f32; 2];
        apply_operator_into(
            MagOperator::Direct,
            &a_p,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [7.0, 8.0]);
    }

    #[test]
    fn apply_operator_missing_readout_errors() {
        let a_p = [1.0, 2.0];
        let mut out = [0.0_f32; 2];
        // Verdict needs a_y but it's empty
        let err = apply_operator_into(
            MagOperator::Verdict,
            &a_p,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut out,
        );
        assert_eq!(err.unwrap_err(), MagError::Empty);
    }

    #[test]
    fn apply_operator_allocating_wrapper() {
        let a_p = [1.0, 2.0];
        let a_eqp = [4.0, 6.0];
        let out = apply_operator(
            MagOperator::FewShot,
            &a_p,
            &[],
            &[],
            &[],
            &[],
            &[],
            &a_eqp,
        )
        .unwrap();
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn with_diagnostics_roundtrip() {
        let with: Vec<[f32; 2]> = vec![[2.0, 0.0], [3.0, 1.0]];
        let without: Vec<[f32; 2]> = vec![[1.0, 0.0], [2.0, 1.0]];
        let dir = mine_direction(&with, &without).unwrap();
        assert!(dir.recon_error.is_nan());
        let (eps, cos) =
            reconstruction_error(&with, &without, dir.as_slice(), 1.0).unwrap();
        let dir = dir.with_diagnostics(eps, cos);
        assert!(!dir.recon_error.is_nan());
        assert!(!dir.cosine.is_nan());
    }
}
