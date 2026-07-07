//! GoldShare — content-specific output-fraction diagnostic (Plan 411, Research 392).
//!
//! Distilled from Gollapudi et al., *Can Language Models Actually Retrieve
//! In-Context? Drowning in Documents at Million Token Scale* (arXiv:2607.01538).
//! The paper's deepest finding is the **recall–generation gap**: the per-head
//! pre-softmax retrieval signal (`R^any_L = 1.0` — at least one head ranks the
//! gold document first by MaxSim) persists across corpus sizes N ∈ {500…10k},
//! but the post-normalization mass on gold collapses. The attention output is
//! *rewritten* from a gold-token average to a non-gold-token average at
//! comparable magnitude.
//!
//! # The primitive
//!
//! Given a query's gold token set `G` and the layer's attention output
//! `a_L ∈ ℝ^{d_model}` (the value-weighted sum projected through the output
//! matrix `W_O`), decompose:
//!
//! ```text
//! a_L = a^G_L + a^{Ḡ}_L
//! a^G_L   = (Σ_{t∈G}   α_t · v_t) · W_O    # gold-derived fraction
//! a^{Ḡ}_L = (Σ_{t∉G}   α_t · v_t) · W_O    # distractor-derived fraction
//! ```
//!
//! [`gold_share`] returns `‖a^G_L‖ / ‖a_L‖` ∈ [0, 1]. It is 1.0 at small N
//! (output is gold-dominated) and → 0 at large N (diluted — paper's Table 1
//! shows gold_share drops 0.91 → 0.01 as N grows 500 → 10k, while `‖a_L‖`
//! shrinks only ~36%). This is the **content-specific** diagnostic — it tells
//! you whether the layer's output still carries the gold signal or has been
//! rewritten to carry aggregate noise.
//!
//! # How it complements existing diagnostics
//!
//! - [`super::geometry::effective_rank`] is **content-agnostic**: it detects
//!   *aggregate* collapse of hidden states across tokens, but cannot tell gold
//!   from distractor.
//! - [`super::sink_classify::stable_rank_update_into`] is **per-sink**: it
//!   detects NOP vs Broadcast degeneracy of individual sink columns.
//! - [`gold_share`] is **content-specific**: it tells you the *fraction* of the
//!   layer's output that came from the tokens you actually care about (the gold
//!   set), regardless of whether the aggregate geometry looks healthy.
//!
//! The joint reading (Plan 411 §Phase 3): a sink classifier hit on the gold
//! position with low `gold_share` is a *broadcast that failed* — the signal
//! was in the head per the classifier, but didn't survive normalization into
//! the residual stream.
//!
//! # Multi-head layout
//!
//! `attn_weights` is `(n_heads, n_kv)` row-major — `α[h][t]` lives at
//! `attn_weights[h * n_kv + t]`. `values` is `(n_kv, d_head)` row-major. The
//! per-head attention output `o_h = Σ_t α[h][t] · v[t]` is a `d_head` vector;
//! concatenating across heads gives a `n_heads * d_head` vector that is then
//! projected through `W_O` of shape `(n_heads * d_head, d_model)` to produce
//! the `d_model`-dimensional layer output.
//!
//! # Allocation discipline (G4)
//!
//! All scratch lives in [`GoldShareScratch`], pre-allocated once and reused
//! across calls. [`gold_share`] and [`gold_share_flat`] perform no heap
//! allocation after warmup. The two output buffers (`total_out`, `gold_out`)
//! are `d_model`-length; the per-head accumulator is `n_heads * d_head`-length.
//!
//! Feature-gated behind `#[cfg(feature = "gold_share_probe")]`.

// ──────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────

/// Per-layer GoldShare diagnostic report.
///
/// All fields are plain `f32` so the report is `Copy` and trivially
/// serializable for downstream consumers (sink-aware attention wiring,
/// runtime NPC cognition probes).
#[derive(Debug, Clone, Copy, Default)]
pub struct GoldShareReport {
    /// `‖a^G_L‖` — L2 norm of the gold-derived fraction of the attention output.
    pub gold_norm: f32,
    /// `‖a_L‖` — L2 norm of the full attention output (gold + distractor).
    pub total_norm: f32,
    /// `‖a^G_L‖ / ‖a_L‖` ∈ [0, 1]. The headline number: 1.0 = output is
    /// entirely gold-derived, 0.0 = output carries no gold signal at all.
    /// `f32::NAN` if `total_norm == 0` (degenerate all-zero output).
    pub gold_share: f32,
    /// `max_t∈G (max_h α[h][t])` — the strongest pre-normalization attention
    /// weight on any gold position, across all heads. A high
    /// `gold_pre_softmax_max` paired with a low `gold_share` is the
    /// paper's recall–generation gap signature: the signal was in the heads
    /// but didn't survive normalization into the residual.
    pub gold_pre_softmax_max: f32,
    /// `gold_pre_softmax_max − max_t∉G (max_h α[h][t])` — the gap between the
    /// strongest gold attention weight and the strongest distractor attention
    /// weight. Positive = gold dominates pre-normalization; negative = a
    /// distractor outranks gold. This is the pre-normalization analog of
    /// `gold_share` and disambiguates "gold lost because it was never strong"
    /// from "gold lost despite being strong" (the dilution case).
    pub noise_gap: f32,
}

/// Pre-allocated scratch buffers for GoldShare computation.
///
/// Create once via [`GoldShareScratch::new`] and reuse across calls via
/// [`GoldShareScratch::ensure_capacity`]. The hot path performs no heap
/// allocation when the dimensions match the cache.
///
/// - `head_concat`: per-head attention output concatenated across heads,
///   length `n_heads * d_head`. Holds both the full and gold-restricted sums
///   (computed sequentially, not simultaneously).
/// - `proj_out`: `W_O`-projected output, length `d_model`. Holds both the full
///   and gold-restricted projections.
///
/// This mirrors the [`StableRankScratch`](super::sink_classify::StableRankScratch)
/// convention: caller-owned, lazily grown, zero-alloc hot path.
pub struct GoldShareScratch {
    /// Concatenated per-head attention output `(n_heads * d_head,)`. Reused for
    /// both the full and gold-restricted weighted sums.
    pub head_concat: Vec<f32>,
    /// `W_O` projection output `(d_model,)`. Reused for both the full and
    /// gold-restricted projections.
    pub proj_out: Vec<f32>,
    cached_concat_len: usize,
    cached_d_model: usize,
}

impl GoldShareScratch {
    /// Allocate scratch for a given `n_heads * d_head` concat length and
    /// `d_model` output dimension.
    pub fn new(concat_len: usize, d_model: usize) -> Self {
        Self {
            head_concat: vec![0.0; concat_len],
            proj_out: vec![0.0; d_model],
            cached_concat_len: concat_len,
            cached_d_model: d_model,
        }
    }

    /// Resize if dimensions changed; no-op on the hot path.
    pub fn ensure_capacity(&mut self, concat_len: usize, d_model: usize) {
        if self.cached_concat_len != concat_len {
            self.head_concat.resize(concat_len, 0.0);
            self.cached_concat_len = concat_len;
        }
        if self.cached_d_model != d_model {
            self.proj_out.resize(d_model, 0.0);
            self.cached_d_model = d_model;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Core function (typed &[f32] slice variant)
// ──────────────────────────────────────────────────────────────────────────

/// Compute the GoldShare diagnostic `‖a^G_L‖ / ‖a_L‖`.
///
/// See the [module docs](self) for the full math. `attn_weights` is
/// `(n_heads, n_kv)` row-major, `values` is `(n_kv, d_head)` row-major, and
/// `w_o` is `(n_heads * d_head, d_model)` row-major. `gold_mask` is `(n_kv,)`.
///
/// Returns a [`GoldShareReport`] with the gold/total norms, the headline
/// `gold_share` ratio, the strongest pre-normalization gold attention weight,
/// and the gold-vs-distractor pre-normalization noise gap.
///
/// # Panics (debug only)
///
/// In debug builds, asserts the slice lengths are consistent with the declared
/// dimensions. Release builds skip the checks for hot-path speed.
///
/// # Allocation
///
/// Zero heap allocation when `scratch` is pre-sized via
/// [`GoldShareScratch::ensure_capacity`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn gold_share(
    attn_weights: &[f32],
    values: &[f32],
    gold_mask: &[bool],
    w_o: &[f32],
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    d_model: usize,
    scratch: &mut GoldShareScratch,
) -> GoldShareReport {
    gold_share_flat(
        attn_weights,
        values,
        gold_mask,
        w_o,
        n_heads,
        n_kv,
        d_head,
        d_model,
        scratch,
    )
}

// ──────────────────────────────────────────────────────────────────────────
// Flat-variant (explicit dimensions, matches apply_dual_policy_gate_flat)
// ──────────────────────────────────────────────────────────────────────────

/// Flat-layout variant of [`gold_share`] — identical computation, explicit
/// dimension parameters.
///
/// This is the variant intended for direct integration with the flat
/// `&[f32]` attention outputs produced by `parallax_attn` / `attention.rs`
/// forward paths (mirroring
/// [`apply_dual_policy_gate_flat`](super::sink_classify::apply_dual_policy_gate_flat)).
///
/// `attn_weights` is `(n_heads, n_kv)` flat, `values` is `(n_kv, d_head)` flat,
/// `w_o` is `(n_heads * d_head, d_model)` flat, `gold_mask` is `(n_kv,)`.
///
/// # Allocation
///
/// Zero heap allocation when `scratch` is pre-sized.
#[allow(clippy::too_many_arguments)]
pub fn gold_share_flat(
    attn_weights: &[f32],
    values: &[f32],
    gold_mask: &[bool],
    w_o: &[f32],
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    d_model: usize,
    scratch: &mut GoldShareScratch,
) -> GoldShareReport {
    let concat_len = n_heads * d_head;
    debug_assert_eq!(
        attn_weights.len(),
        n_heads * n_kv,
        "gold_share: attn_weights must be (n_heads={}, n_kv={}) flat",
        n_heads,
        n_kv
    );
    debug_assert_eq!(
        values.len(),
        n_kv * d_head,
        "gold_share: values must be (n_kv={}, d_head={}) flat",
        n_kv,
        d_head
    );
    debug_assert_eq!(
        gold_mask.len(),
        n_kv,
        "gold_share: gold_mask must be (n_kv={},) flat",
        n_kv
    );
    debug_assert_eq!(
        w_o.len(),
        concat_len * d_model,
        "gold_share: w_o must be (n_heads*d_head={}, d_model={}) flat",
        concat_len,
        d_model
    );

    scratch.ensure_capacity(concat_len, d_model);
    let head_concat = &mut scratch.head_concat[..concat_len];
    let proj_out = &mut scratch.proj_out[..d_model];

    // ── Pre-normalization gold / distractor attention maxima ─────────────
    //
    // max_h α[h][t] = strongest attention any head places on position t.
    // Track the max over gold positions and over non-gold positions.
    let mut gold_pre_max = f32::NEG_INFINITY;
    let mut nongold_pre_max = f32::NEG_INFINITY;
    for t in 0..n_kv {
        let mut best_h = f32::NEG_INFINITY;
        for h in 0..n_heads {
            let a = attn_weights[h * n_kv + t];
            if a > best_h {
                best_h = a;
            }
        }
        if gold_mask[t] {
            if best_h > gold_pre_max {
                gold_pre_max = best_h;
            }
        } else if best_h > nongold_pre_max {
            nongold_pre_max = best_h;
        }
    }
    // Handle degenerate cases (all-gold or all-nongold masks).
    if !gold_pre_max.is_finite() {
        gold_pre_max = 0.0;
    }
    if !nongold_pre_max.is_finite() {
        nongold_pre_max = 0.0;
    }
    let noise_gap = gold_pre_max - nongold_pre_max;

    // ── Full attention output: a = (Σ_t α[h][t] · v[t])_h · W_O ──────────
    let total_norm = compute_output_norm(
        attn_weights,
        values,
        w_o,
        n_heads,
        n_kv,
        d_head,
        d_model,
        head_concat,
        proj_out,
        /* restrict_to_gold */ false,
        gold_mask,
    );

    // ── Gold-restricted output: a^G = (Σ_{t∈G} α[h][t] · v[t])_h · W_O ───
    let gold_norm = compute_output_norm(
        attn_weights,
        values,
        w_o,
        n_heads,
        n_kv,
        d_head,
        d_model,
        head_concat,
        proj_out,
        /* restrict_to_gold */ true,
        gold_mask,
    );

    let gold_share = if total_norm > 0.0 {
        gold_norm / total_norm
    } else {
        f32::NAN
    };

    GoldShareReport {
        gold_norm,
        total_norm,
        gold_share,
        gold_pre_softmax_max: gold_pre_max,
        noise_gap,
    }
}

/// Shared inner routine: compute `‖a‖` where `a = (Σ_{t∈R} α[h][t] · v[t])_h · W_O`
/// and `R` is either the full key set or the gold-restricted key set.
///
/// Writes the `W_O` projection into `proj_out` (length `d_model`) and returns
/// its L2 norm. `head_concat` (length `n_heads * d_head`) is used as scratch
/// for the concatenated per-head weighted sum.
///
/// `restrict_to_gold = false` → sum over all `t`; `true` → sum only over
/// `t` where `gold_mask[t]`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_output_norm(
    attn_weights: &[f32],
    values: &[f32],
    w_o: &[f32],
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    d_model: usize,
    head_concat: &mut [f32],
    proj_out: &mut [f32],
    restrict_to_gold: bool,
    gold_mask: &[bool],
) -> f32 {
    let concat_len = n_heads * d_head;
    debug_assert_eq!(head_concat.len(), concat_len);
    debug_assert_eq!(proj_out.len(), d_model);

    // 1. Per-head weighted sum: o_h[d] = Σ_t α[h][t] · v[t][d].
    //    head_concat[h * d_head + d] = o_h[d].
    for h in 0..n_heads {
        let acc = &mut head_concat[h * d_head..(h + 1) * d_head];
        acc.fill(0.0);
        for t in 0..n_kv {
            if restrict_to_gold && !gold_mask[t] {
                continue;
            }
            let a = attn_weights[h * n_kv + t];
            let v = &values[t * d_head..(t + 1) * d_head];
            for d in 0..d_head {
                acc[d] += a * v[d];
            }
        }
    }

    // 2. Project through W_O: a[j] = Σ_i head_concat[i] · W_O[i * d_model + j].
    //    W_O is (concat_len, d_model) row-major.
    proj_out.fill(0.0);
    for i in 0..concat_len {
        let hc = head_concat[i];
        let row = &w_o[i * d_model..(i + 1) * d_model];
        for j in 0..d_model {
            proj_out[j] += hc * row[j];
        }
    }

    // 3. L2 norm of the projected output.
    let mut sum_sq = 0.0_f32;
    for &v in &proj_out[..d_model] {
        sum_sq += v * v;
    }
    sum_sq.sqrt()
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    /// Build a scratch sized for the given dimensions.
    fn make_scratch(n_heads: usize, d_head: usize, d_model: usize) -> GoldShareScratch {
        GoldShareScratch::new(n_heads * d_head, d_model)
    }

    #[test]
    fn all_true_gold_mask_gives_share_one() {
        // 2 heads, 4 keys, d_head=3, d_model=2.
        let n_heads = 2;
        let n_kv = 4;
        let d_head = 3;
        let d_model = 2;
        let attn_weights: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, // head 0
            0.25, 0.25, 0.25, 0.25, // head 1
        ];
        let values: Vec<f32> = (0..n_kv * d_head).map(|i| i as f32).collect();
        let gold_mask = vec![true; n_kv];
        // Identity W_O (concat_len × d_model): first d_model dims pass through.
        let concat_len = n_heads * d_head;
        let mut w_o = vec![0.0_f32; concat_len * d_model];
        for i in 0..d_model.min(concat_len) {
            w_o[i * d_model + i] = 1.0;
        }

        let mut scratch = make_scratch(n_heads, d_head, d_model);
        let report = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch,
        );

        // All-gold → a^G == a → gold_share == 1.0.
        assert!(
            (report.gold_share - 1.0).abs() < TOL,
            "all-gold mask should give gold_share 1.0, got {}",
            report.gold_share
        );
        assert!((report.gold_norm - report.total_norm).abs() < TOL);
    }

    #[test]
    fn all_false_gold_mask_gives_share_zero() {
        let n_heads = 2;
        let n_kv = 4;
        let d_head = 3;
        let d_model = 2;
        let attn_weights: Vec<f32> = vec![0.25; n_heads * n_kv];
        let values: Vec<f32> = (0..n_kv * d_head).map(|i| (i as f32) + 1.0).collect();
        let gold_mask = vec![false; n_kv];
        let concat_len = n_heads * d_head;
        let mut w_o = vec![0.0_f32; concat_len * d_model];
        for i in 0..d_model.min(concat_len) {
            w_o[i * d_model + i] = 1.0;
        }

        let mut scratch = make_scratch(n_heads, d_head, d_model);
        let report = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch,
        );

        // No gold → a^G = 0 → gold_share = 0.
        assert!(
            report.gold_norm.abs() < TOL,
            "all-nongold mask should give gold_norm 0, got {}",
            report.gold_norm
        );
        assert!(report.total_norm > 0.0, "total_norm should be positive");
        assert!(
            report.gold_share.abs() < TOL,
            "all-nongold mask should give gold_share 0, got {}",
            report.gold_share
        );
    }

    #[test]
    fn paper_table1_toy_4head_8key_half_gold() {
        // Paper's Table 1 toy: 4 heads, 8 keys, half gold (positions 0-3 gold).
        // We hand-construct a small case and verify gold_share matches the
        // hand-computed ‖a^G‖ / ‖a‖ to 4 decimals.
        //
        // Layout:
        // - n_heads=4, n_kv=8, d_head=2, d_model=2.
        // - Attention weights: each head attends uniformly (1/8) so the math
        //   is easy to verify by hand.
        // - Values: v[t] = (t+1, t+1) so each key has a distinct vector.
        // - W_O: identity (first 2 rows of the 8×2 matrix are identity, rest 0).
        // - Gold mask: positions 0,1,2,3 are gold; 4,5,6,7 are distractors.
        //
        // Per-head weighted sum with uniform 1/8 attention:
        //   o_h = Σ_t (1/8) · v[t] = (1/8) · Σ_t (t+1, t+1)
        //       = (1/8) · (36, 36) = (4.5, 4.5)   [since Σ(t+1) for t=0..7 = 36]
        // All 4 heads produce the same o_h = (4.5, 4.5), so head_concat =
        //   (4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5)  [8 values]
        // With identity W_O (rows 0,1 are [1,0],[0,1]; rows 2-7 are 0):
        //   a = 4.5 · [1,0] + 4.5 · [0,1] = (4.5, 4.5)
        //   ‖a‖ = sqrt(4.5² + 4.5²) = 4.5 · sqrt(2) ≈ 6.3640
        //
        // Gold-restricted (positions 0-3 only):
        //   o_h^G = Σ_{t=0..3} (1/8) · v[t] = (1/8) · (1+2+3+4) · (1,1)
        //         = (1/8) · 10 · (1,1) = (1.25, 1.25)
        //   head_concat^G = (1.25, 1.25, ...) [8 values, all 1.25]
        //   a^G = 1.25 · [1,0] + 1.25 · [0,1] = (1.25, 1.25)
        //   ‖a^G‖ = 1.25 · sqrt(2) ≈ 1.7678
        //
        // gold_share = 1.7678 / 6.3640 ≈ 0.2778
        //
        // To 4 decimals: 0.2778.
        let n_heads = 4;
        let n_kv = 8;
        let d_head = 2;
        let d_model = 2;

        let attn_weights: Vec<f32> = vec![1.0 / 8.0; n_heads * n_kv]; // uniform 1/8
        let values: Vec<f32> = {
            let mut v = Vec::with_capacity(n_kv * d_head);
            for t in 0..n_kv {
                let val = (t + 1) as f32;
                v.push(val);
                v.push(val); // v[t] = (t+1, t+1)
            }
            v
        };
        let gold_mask: Vec<bool> = vec![true, true, true, true, false, false, false, false];

        // Identity W_O: 8×2 matrix, rows 0,1 are identity, rest zero.
        let concat_len = n_heads * d_head; // 8
        let mut w_o = vec![0.0_f32; concat_len * d_model];
        w_o[0] = 1.0; // row 0 = [1, 0]
        w_o[d_model + 1] = 1.0; // row 1 = [0, 1]
        // rows 2-7 are zero.

        let mut scratch = make_scratch(n_heads, d_head, d_model);
        let report = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch,
        );

        let expected_total = 4.5_f32 * 2.0_f32.sqrt();
        let expected_gold = 1.25_f32 * 2.0_f32.sqrt();
        let expected_share = expected_gold / expected_total;

        assert!(
            (report.total_norm - expected_total).abs() < 1e-3,
            "total_norm: expected {:.4}, got {:.4}",
            expected_total,
            report.total_norm
        );
        assert!(
            (report.gold_norm - expected_gold).abs() < 1e-3,
            "gold_norm: expected {:.4}, got {:.4}",
            expected_gold,
            report.gold_norm
        );
        assert!(
            (report.gold_share - expected_share).abs() < 1e-4,
            "gold_share: expected {:.4}, got {:.4}",
            expected_share,
            report.gold_share
        );
        // Verify the headline to 4 decimals: 0.2778.
        assert!(
            (report.gold_share - 0.2778).abs() < 1e-4,
            "gold_share should be ≈0.2778 to 4 decimals, got {:.6}",
            report.gold_share
        );
    }

    #[test]
    fn gold_share_and_flat_agree_bit_exactly() {
        // Both entry points must produce identical results — gold_share just
        // delegates to gold_share_flat, but we test the contract explicitly
        // in case that delegation ever changes.
        let n_heads = 3;
        let n_kv = 6;
        let d_head = 4;
        let d_model = 5;

        // Non-trivial attention weights (non-uniform, non-symmetric).
        let attn_weights: Vec<f32> = (0..n_heads * n_kv)
            .map(|i| ((i as f32) * 0.13).sin().abs() * 0.5 + 0.01)
            .collect();
        let values: Vec<f32> = (0..n_kv * d_head).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let gold_mask: Vec<bool> = vec![true, false, true, false, true, false];
        // Non-identity W_O: all-ones scaled.
        let concat_len = n_heads * d_head;
        let w_o: Vec<f32> = vec![0.3_f32; concat_len * d_model];

        let mut scratch_a = make_scratch(n_heads, d_head, d_model);
        let mut scratch_b = make_scratch(n_heads, d_head, d_model);

        let report_typed = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch_a,
        );
        let report_flat = gold_share_flat(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch_b,
        );

        // Bit-exact: same input + same algorithm (gold_share delegates to
        // gold_share_flat) → every field must match to the last bit.
        assert!(
            report_typed.gold_norm.to_bits() == report_flat.gold_norm.to_bits(),
            "gold_norm mismatch: typed={} flat={}",
            report_typed.gold_norm,
            report_flat.gold_norm
        );
        assert!(
            report_typed.total_norm.to_bits() == report_flat.total_norm.to_bits(),
            "total_norm mismatch: typed={} flat={}",
            report_typed.total_norm,
            report_flat.total_norm
        );
        assert!(
            report_typed.gold_share.to_bits() == report_flat.gold_share.to_bits(),
            "gold_share mismatch: typed={} flat={}",
            report_typed.gold_share,
            report_flat.gold_share
        );
        assert!(
            report_typed.gold_pre_softmax_max.to_bits()
                == report_flat.gold_pre_softmax_max.to_bits(),
            "gold_pre_softmax_max mismatch"
        );
        assert!(
            report_typed.noise_gap.to_bits() == report_flat.noise_gap.to_bits(),
            "noise_gap mismatch"
        );
    }

    #[test]
    fn gold_pre_softmax_max_and_noise_gap_correct() {
        // Verify the pre-softmax diagnostics independently.
        let n_heads = 2;
        let n_kv = 4;
        let d_head = 2;
        let d_model = 2;

        // Hand-set attention: head 0 puts 0.9 on position 0 (gold), 0.1 on others.
        // head 1 puts 0.8 on position 3 (distractor), 0.2 on others.
        let attn_weights: Vec<f32> = vec![
            0.9, 0.05, 0.03, 0.02, // head 0: gold pos 0 dominates
            0.1, 0.1, 0.0, 0.8, // head 1: distractor pos 3 dominates
        ];
        let values: Vec<f32> = vec![1.0; n_kv * d_head];
        let gold_mask: Vec<bool> = vec![true, false, false, false];
        let concat_len = n_heads * d_head;
        let w_o: Vec<f32> = vec![1.0; concat_len * d_model];

        let mut scratch = make_scratch(n_heads, d_head, d_model);
        let report = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch,
        );

        // max_h α[h][0] (gold) = max(0.9, 0.1) = 0.9.
        // max_h α[h][t] for t∉G: pos1=max(0.05,0.1)=0.1, pos2=max(0.03,0)=0.03,
        //   pos3=max(0.02,0.8)=0.8 → nongold max = 0.8.
        assert!(
            (report.gold_pre_softmax_max - 0.9).abs() < TOL,
            "gold_pre_softmax_max: expected 0.9, got {}",
            report.gold_pre_softmax_max
        );
        assert!(
            (report.noise_gap - (0.9 - 0.8)).abs() < TOL,
            "noise_gap: expected 0.1, got {}",
            report.noise_gap
        );
    }

    #[test]
    fn empty_gold_set_with_nonzero_total_gives_zero_share() {
        // Subset of all-false: verify gold_share = 0 and no NaN.
        let n_heads = 1;
        let n_kv = 3;
        let d_head = 1;
        let d_model = 1;
        let attn_weights = vec![1.0, 1.0, 1.0];
        let values = vec![2.0, 3.0, 4.0];
        let gold_mask = vec![false, false, false];
        let w_o = vec![1.0]; // 1×1 identity
        let mut scratch = make_scratch(n_heads, d_head, d_model);
        let report = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch,
        );
        assert!(report.gold_share.abs() < TOL);
        assert!(!report.gold_share.is_nan());
        assert!(report.total_norm > 0.0);
    }

    #[test]
    fn degenerate_all_zero_output_gives_nan_share() {
        // All-zero values → total_norm = 0 → gold_share = NaN (documented).
        let n_heads = 1;
        let n_kv = 2;
        let d_head = 1;
        let d_model = 1;
        let attn_weights = vec![1.0, 1.0];
        let values = vec![0.0, 0.0];
        let gold_mask = vec![true, false];
        let w_o = vec![1.0];
        let mut scratch = make_scratch(n_heads, d_head, d_model);
        let report = gold_share(
            &attn_weights,
            &values,
            &gold_mask,
            &w_o,
            n_heads,
            n_kv,
            d_head,
            d_model,
            &mut scratch,
        );
        assert!(report.total_norm.abs() < TOL);
        assert!(report.gold_share.is_nan(), "degenerate output → NaN share");
    }

    #[test]
    fn scratch_ensure_capacity_is_noop_on_match() {
        let mut scratch = GoldShareScratch::new(8, 4);
        // Same dims → no realloc (capacity unchanged, content may be stale).
        let ptr_before = scratch.head_concat.as_ptr();
        scratch.ensure_capacity(8, 4);
        assert_eq!(scratch.head_concat.as_ptr(), ptr_before);
        // Different dims → realloc.
        scratch.ensure_capacity(16, 8);
        assert_eq!(scratch.head_concat.len(), 16);
        assert_eq!(scratch.proj_out.len(), 8);
    }
}
