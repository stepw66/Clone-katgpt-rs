//! Adaptive Causal Calibration — cheap-proxy-escalate (Proposal 004).
//!
//! Ships the **open primitive** half of the adaptive thermal path: a cheap
//! OV-circuit proxy detects *bystander suspects* in a single observational
//! pass, and the caller escalates to Plan 358's expensive causal patching only
//! on those `k` suspects instead of all `n_heads`.
//!
//! # Honest caveats — READ BEFORE USE
//!
//! 1. **This scheme is our invention.** HydraHead (arXiv:2606.20097) proposes
//!    causal head-importance scoring. It does **not** propose an OV-circuit
//!    cheap proxy, an adaptive escalate-on-suspects mode, or any two-stage
//!    cheap-then-expensive calibration.
//! 2. **The OV-circuit proxy is an UNVALIDATED hypothesis.** The bystander
//!    signature `attention_mass / ||OV_out|| > τ` is a *reasonable* mech-interp
//!    argument (Elhage et al., "A Mathematical Framework for Transformer
//!    Circuits"), but it has never been measured against ground-truth causal
//!    IE on a real model. If the proxy has low precision (flags non-bystanders),
//!    the escalation fires too often and the cost win evaporates.
//! 3. **Validation requires riir-engine.** Computing `||OV_out[h]||` at the
//!    readout position requires a real transformer forward that exposes per-head
//!    OV outputs. That is explicitly out of scope for katgpt-rs (Plan 358 Risk
//!    #1, Risk #4). The caller supplies both observables; katgpt-rs is the rule.
//! 4. **Promotion to default is blocked on G1 + G2.** `AdaptiveCausal` MUST NOT
//!    become the default `CalibrationMode` until both G1 (proxy precision on a
//!    real model) and G2 (cost reduction holds at production head counts) pass
//!    empirically in riir-engine. Until then `AttentionMass` stays default.
//!
//! # Modelless discipline
//!
//! [`suspect_indices`] is a pure fn over caller-supplied `(attention_mass,
//! ov_output_norm)` pairs — zero allocation, zero deps, `#[inline]`. The caller
//! (riir-engine) supplies the observables from a real forward; katgpt-rs
//! doesn't need to know how they were computed. [`adaptive_partition`] merges
//! the caller-supplied causal scores for the suspects with the non-suspects'
//! attention-mass scores, then delegates to [`partition_by_causal_score`]
//! (Plan 358). Both are leaf-clean.
//!
//! # GOAT gate (this primitive's promotion criteria)
//!
//! | Gate | Criterion | Where |
//! |------|-----------|-------|
//! | G1 | Proxy precision ≥ 0.8 @ recall ≥ 0.9 on a real transformer | riir-engine |
//! | G2 | Cost within ~2× of attention-mass at production head counts | riir-engine + katgpt-rs |
//! | G3 | No-suspect → bit-identical to attention-mass partition | katgpt-rs (this file) |
//! | G4 | `suspect_indices` alloc-free hot path | katgpt-rs (this file) |
//!
//! G3 + G4 are tested here. G1 + G2 are deferred to riir-engine and block
//! promotion — see Proposal 004 §"Phased rollout".

use crate::causal_head_importance::partition_by_causal_score;

/// OV-circuit cheap proxy: flag heads whose `attention_mass / ||ov_output||`
/// ratio exceeds `tau_suspect`.
///
/// These are **suspects** — they attend strongly to the needle (high
/// `attention_mass`) but contribute little to the readout direction (low
/// `||ov_output||`). That high-attend / low-contribute ratio is the defining
/// feature of a *correlated bystander* (HydraHead's term): the head looks
/// important observationally but is overridden downstream. The caller escalates
/// to causal patching on these `k` suspects only, instead of all `n_heads`.
///
/// **Cost:** zero allocation — yields suspect indices in ascending order.
/// **Honesty:** the proxy is an unvalidated hypothesis (see module caveats);
/// its precision is G1's gate, deferred to riir-engine.
///
/// # Arguments
///
/// * `attention_mass` — per-head needle attention-mass `R_h` (observational,
///   from a single forward pass). Length `n_heads`.
/// * `ov_output_norm` — per-head `||OV · attn(·, t_readout)||` norm (the head's
///   output contribution at the readout position). Same length as
///   `attention_mass`. Supplied by the caller (riir-engine) from a real
///   transformer forward.
/// * `tau_suspect` — escalation threshold. A head is a suspect iff
///   `attention_mass / ov_output_norm > tau_suspect`. **No universal default** —
///   its scale depends on model dimensions and must be tuned empirically (G1).
///
/// # Degenerate-head handling
///
/// A head with `ov_output_norm == 0.0` is **not** a suspect — it contributes
/// nothing at all (inert), so it cannot be a *bystander* (a bystander attends
/// strongly but contributes little-but-nonzero). Inert heads keep their
/// attention-mass ranking unchanged, which correctly demotes them when their
/// attention-mass is low.
#[inline]
pub fn suspect_indices<'a>(
    attention_mass: &'a [f32],
    ov_output_norm: &'a [f32],
    tau_suspect: f32,
) -> impl Iterator<Item = usize> + 'a {
    debug_assert_eq!(
        attention_mass.len(),
        ov_output_norm.len(),
        "attention_mass and ov_output_norm must have the same length (n_heads)"
    );
    attention_mass
        .iter()
        .zip(ov_output_norm.iter())
        .enumerate()
        // Bystander signature: attends a lot (high am), contributes little
        // (low ov). am/ov > tau captures this directly. Skip ov == 0 (inert).
        .filter_map(move |(h, (&am, &ov))| {
            if ov > 0.0 && am > 0.0 && am / ov > tau_suspect {
                Some(h)
            } else {
                None
            }
        })
}

/// Merge suspect causal scores with non-suspect attention-mass scores into a
/// single fused ranking, then partition into critical/convertible sets.
///
/// - **Non-suspects** keep their `attention_mass[h]` score (the observational
///   ranking is correct for them — they are not bystanders, so attention-mass
///   does not mislead).
/// - **Suspects** get their caller-supplied causal IE score (the ground truth
///   for them — causal necessity is strictly stronger on bystander-heavy
///   workloads per Plan 358).
///
/// The fused vector is then partitioned by [`partition_by_causal_score`]
/// (Plan 358), keeping the partition logic DRY.
///
/// # G3 degenerate-case guarantee
///
/// When `suspects` is empty, the fused vector equals `attention_mass` verbatim,
/// so the partition is **bit-identical** to
/// `partition_by_causal_score(attention_mass, …)`. This is the G3 gate: the
/// adaptive mode pays zero overhead when there are no bystanders — it collapses
/// to pure attention-mass.
///
/// # Cross-group scale (the G1 question, deferred)
///
/// The raw merge does **not** normalize across the suspect / non-suspect
/// populations — it trusts that attention-mass and causal IE are roughly
/// comparable in scale (both bounded in practice). Whether that trust holds is
/// exactly what G1 measures. If G1 finds a scale mismatch, riir-engine adds a
/// normalization bridge; the open primitive ships the simplest defensible rule
/// and lets measurement decide. Do not over-engineer before measurement.
///
/// # Arguments
///
/// * `attention_mass` — per-head needle attention-mass for **all** `n_heads`.
/// * `suspects` — suspect head indices (ascending), from [`suspect_indices`].
/// * `suspect_causal_scores` — per-suspect causal IE, **parallel to `suspects`**
///   (i.e. `suspect_causal_scores[i]` is the IE of `suspects[i]`). The caller
///   runs patched forwards on the suspects only and supplies the results here.
/// * `critical_ratio` — fraction of heads to retain (delegates to
///   [`partition_by_causal_score`]).
/// * `layer_ids` / `min_one_per_layer` — constrained-global-screening options
///   (delegates to [`partition_by_causal_score`]).
///
/// # Returns
///
/// `(critical_set, convertible_set)` as sorted `Vec<usize>` of head indices,
/// matching [`partition_by_causal_score`]'s shape exactly.
pub fn adaptive_partition(
    attention_mass: &[f32],
    suspects: &[usize],
    suspect_causal_scores: &[f32],
    critical_ratio: f32,
    layer_ids: Option<&[usize]>,
    min_one_per_layer: bool,
) -> (Vec<usize>, Vec<usize>) {
    let n = attention_mass.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    debug_assert_eq!(
        suspects.len(),
        suspect_causal_scores.len(),
        "suspects and suspect_causal_scores must be parallel (same length)"
    );
    debug_assert!(
        suspects.windows(2).all(|w| w[0] < w[1]),
        "suspects must be ascending and deduplicated"
    );
    debug_assert!(
        suspects.iter().all(|&h| h < n),
        "all suspect indices must be < n_heads ({n})"
    );

    // G3 fast path: no suspects → fused == attention_mass verbatim → delegate.
    // This is the "pays zero overhead when no bystanders" guarantee.
    if suspects.is_empty() {
        return partition_by_causal_score(
            attention_mass,
            critical_ratio,
            layer_ids,
            min_one_per_layer,
        );
    }

    // Build the fused score vector: non-suspects keep attention_mass, suspects
    // get their causal IE. One allocation (the fused vec) + one dense bool
    // flag array — mirrors partition_by_causal_score's allocation profile.
    let mut is_suspect = vec![false; n];
    for &h in suspects {
        is_suspect[h] = true;
    }
    let mut fused = attention_mass.to_vec();
    // Parallel walk: suspects and suspect_causal_scores are index-aligned.
    let mut sus_idx = 0usize;
    for h in 0..n {
        if is_suspect[h] {
            fused[h] = suspect_causal_scores[sus_idx];
            sus_idx += 1;
        }
    }
    debug_assert_eq!(sus_idx, suspects.len(), "suspect count mismatch in fused build");

    partition_by_causal_score(&fused, critical_ratio, layer_ids, min_one_per_layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // suspect_indices
    // -----------------------------------------------------------------------

    #[test]
    fn suspect_indices_empty_inputs() {
        let got: Vec<usize> = suspect_indices(&[], &[], 1.0).collect();
        assert!(got.is_empty());
    }

    #[test]
    fn suspect_indices_flags_high_ratio_heads() {
        // Bystander signature: high attention_mass, low ov_output_norm.
        // head0: am=0.8, ov=0.1 → ratio 8.0 (suspect if tau < 8.0)
        // head1: am=0.1, ov=0.9 → ratio 0.11 (not a suspect)
        // head2: am=0.5, ov=0.5 → ratio 1.0 (boundary)
        let am = [0.8, 0.1, 0.5];
        let ov = [0.1, 0.9, 0.5];
        let got: Vec<usize> = suspect_indices(&am, &ov, 1.0).collect();
        // head0 ratio 8.0 > 1.0 → suspect. head1 0.11 < 1.0 → no. head2 1.0 not > 1.0 → no.
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn suspect_indices_strict_inequality_on_tau() {
        // ratio == tau is NOT a suspect (strict >).
        let am = [1.0];
        let ov = [2.0];
        let got: Vec<usize> = suspect_indices(&am, &ov, 0.5).collect();
        // ratio = 0.5 == tau → not flagged.
        assert!(got.is_empty());
    }

    #[test]
    fn suspect_indices_skips_inert_heads() {
        // ov == 0 → inert, not a suspect (cannot be a bystander).
        let am = [0.9, 0.9];
        let ov = [0.0, 0.1];
        let got: Vec<usize> = suspect_indices(&am, &ov, 1.0).collect();
        // head0 inert (ov=0) → skipped. head1 ratio 9.0 > 1.0 → suspect.
        assert_eq!(got, vec![1]);
    }

    #[test]
    fn suspect_indices_skips_zero_attention_heads() {
        // am == 0 → head doesn't attend to the needle, can't be a bystander.
        let am = [0.0, 0.9];
        let ov = [0.1, 0.1];
        let got: Vec<usize> = suspect_indices(&am, &ov, 1.0).collect();
        // head0 am=0 → skipped. head1 ratio 9.0 > 1.0 → suspect.
        assert_eq!(got, vec![1]);
    }

    #[test]
    fn suspect_indices_ascending_order() {
        // Multiple suspects should be yielded in ascending head-index order.
        let am = [0.9, 0.1, 0.9, 0.1, 0.9];
        let ov = [0.1, 0.9, 0.1, 0.9, 0.1];
        let got: Vec<usize> = suspect_indices(&am, &ov, 1.0).collect();
        assert_eq!(got, vec![0, 2, 4]);
    }

    #[test]
    fn suspect_indices_all_below_tau_yields_none() {
        let am = [0.1, 0.2, 0.3];
        let ov = [0.9, 0.9, 0.9];
        let got: Vec<usize> = suspect_indices(&am, &ov, 1.0).collect();
        assert!(got.is_empty());
    }

    // -----------------------------------------------------------------------
    // adaptive_partition — G3 (no-suspect bit-identical to attention-mass)
    // -----------------------------------------------------------------------

    #[test]
    fn g3_no_suspects_bit_identical_to_attention_mass_partition() {
        // G3: when suspects is empty, adaptive_partition must produce EXACTLY
        // the same (critical, convertible) as partition_by_causal_score on the
        // raw attention-mass scores.
        let am = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4];
        let expected = partition_by_causal_score(&am, 0.25, None, false);
        let got = adaptive_partition(&am, &[], &[], 0.25, None, false);
        assert_eq!(got, expected, "G3 violated: no-suspect must equal attention-mass");
    }

    #[test]
    fn g3_no_suspects_bit_identical_with_layer_constraint() {
        // G3 must also hold when min_one_per_layer is active.
        let am = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4];
        let layers = [0, 0, 1, 1, 2, 2, 3, 3];
        let expected = partition_by_causal_score(&am, 0.25, Some(&layers), true);
        let got = adaptive_partition(&am, &[], &[], 0.25, Some(&layers), true);
        assert_eq!(got, expected, "G3 violated with layer constraint");
    }

    #[test]
    fn g3_empty_inputs_match() {
        let expected = partition_by_causal_score(&[], 0.25, None, false);
        let got = adaptive_partition(&[], &[], &[], 0.25, None, false);
        assert_eq!(got, expected);
        assert!(got.0.is_empty() && got.1.is_empty());
    }

    // -----------------------------------------------------------------------
    // adaptive_partition — sanity (suspect-present, unvalidated semantics)
    // -----------------------------------------------------------------------

    #[test]
    fn adaptive_partition_demotes_confirmed_bystander() {
        // A suspect confirmed as a bystander (low IE) should sink below
        // non-suspects in the ranking, dropping out of the critical set.
        // 5 heads, am ranks head0 highest. head0 is a suspect with IE≈0
        // (confirmed bystander). critical_ratio 0.4 → top-2 critical.
        let am = [0.9, 0.8, 0.7, 0.6, 0.5];
        let suspects = [0];
        let suspect_ie = [0.001]; // head0 confirmed bystander
        let (critical, convertible) =
            adaptive_partition(&am, &suspects, &suspect_ie, 0.4, None, false);
        // Fused: [0.001, 0.8, 0.7, 0.6, 0.5]. Top-2 by score: head1 (0.8), head2 (0.7).
        // head0 (bystander, 0.001) is demoted to convertible.
        assert!(!critical.contains(&0), "bystander head0 must be demoted");
        assert!(convertible.contains(&0), "bystander head0 must be in convertible");
        assert_eq!(critical.len(), 2);
    }

    #[test]
    fn adaptive_partition_promotes_loadbearing_suspect() {
        // A suspect confirmed as load-bearing (high IE) should stay critical.
        // 5 heads, am ranks head4 lowest. head4 is a suspect with high IE
        // (the proxy was a false positive — it's actually load-bearing).
        let am = [0.5, 0.6, 0.7, 0.8, 0.1];
        let suspects = [4];
        let suspect_ie = [0.95]; // head4 confirmed load-bearing
        let (critical, _) = adaptive_partition(&am, &suspects, &suspect_ie, 0.4, None, false);
        // Fused: [0.5, 0.6, 0.7, 0.8, 0.95]. Top-2: head4 (0.95), head3 (0.8).
        assert!(critical.contains(&4), "load-bearing head4 must be critical");
    }

    #[test]
    fn adaptive_partition_multiple_suspects_merge_correctly() {
        // Two suspects: one bystander (low IE), one load-bearing (high IE).
        let am = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let suspects = [0, 5];
        let suspect_ie = [0.001, 0.95]; // head0 bystander, head5 load-bearing
        let (critical, convertible) =
            adaptive_partition(&am, &suspects, &suspect_ie, 0.5, None, false);
        // Fused: [0.001, 0.8, 0.7, 0.6, 0.5, 0.95]. Top-3: head5 (0.95), head1 (0.8), head2 (0.7).
        // head0 (bystander) demoted to convertible.
        assert!(critical.contains(&5), "load-bearing head5 must be critical");
        assert!(!critical.contains(&0), "bystander head0 must not be critical");
        assert!(convertible.contains(&0), "bystander head0 in convertible");
        assert_eq!(critical.len(), 3);
        assert_eq!(convertible.len(), 3);
    }

    #[test]
    fn adaptive_partition_sets_are_sorted_ascending() {
        // Output sets must be sorted ascending (matches partition_by_causal_score).
        let am = [0.5, 0.9, 0.1, 0.8, 0.3, 0.7];
        let suspects = [2];
        let suspect_ie = [0.001];
        let (critical, convertible) =
            adaptive_partition(&am, &suspects, &suspect_ie, 0.5, None, false);
        assert!(
            critical.windows(2).all(|w| w[0] < w[1]),
            "critical not sorted: {critical:?}"
        );
        assert!(
            convertible.windows(2).all(|w| w[0] < w[1]),
            "convertible not sorted: {convertible:?}"
        );
    }
}
