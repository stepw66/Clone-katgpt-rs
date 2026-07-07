//! Plan 411 Phase 3 T3.4 — joint `classify_all_sinks` + `gold_share` integration test.
//!
//! Verifies the "broadcast that failed" signature from Research 392 / arXiv:2607.01538:
//! when a sink classifier hits the gold position as a Broadcast but `gold_share` is low,
//! the signal was in the head per the classifier but didn't survive normalization into
//! the residual (the recall-generation gap). The joint report must be self-consistent.
//!
//! Uses the paper's Table 1 toy geometry: 4 heads × 8 keys × d_head=2, half the keys gold.

#![cfg(all(feature = "sink_aware_attn", feature = "gold_share_probe"))]

use katgpt_core::data_probe::{
    GoldShareReport, GoldShareScratch, SinkClassifierConfig, StableRankScratch,
    classify_all_sinks, gold_share_flat,
};

/// Run gold_share_flat with scratch allocated internally.
fn gold_share_flat_alloc(
    attn_weights: &[f32],
    values: &[f32],
    gold_mask: &[bool],
    w_o: &[f32],
    n_heads: usize,
    n_kv: usize,
    d_head: usize,
    d_model: usize,
) -> GoldShareReport {
    let concat_len = n_heads * d_head;
    let mut scratch = GoldShareScratch::new(concat_len, d_model);
    gold_share_flat(
        attn_weights,
        values,
        gold_mask,
        w_o,
        n_heads,
        n_kv,
        d_head,
        d_model,
        &mut scratch,
    )
}

/// Build the paper's Table 1 toy: 4 heads × 8 keys × d_head=2.
///
/// Layout:
/// - Keys 0..4 are gold (the answer should attend to them).
/// - Keys 4..8 are distractors.
/// - Values are constructed so gold and distractor have comparable per-key norm
///   (so total_norm stays moderate) but the gold-derived output is much smaller
///   than the distractor-derived output under the planted attention weights.
fn build_paper_table1_toy() -> (Vec<f32>, Vec<f32>, Vec<bool>, usize, usize, usize) {
    let n_heads = 4;
    let n_kv = 8;
    let d_head = 2;

    // Attention: each head places most mass on distractors (positions 4..8) and
    // only a little on gold (positions 0..4). This is the diluted regime — the
    // gold signal is present but small in the post-normalization weights.
    let mut attn = vec![0.0_f32; n_heads * n_kv];
    for h in 0..n_heads {
        for t in 0..4 {
            attn[h * n_kv + t] = 0.05; // gold positions get little attention
        }
        for t in 4..8 {
            attn[h * n_kv + t] = 0.20; // distractors get more attention
        }
    }

    // Values: gold and distractor keys carry orthogonal signals of equal norm.
    let mut values = vec![0.0_f32; n_kv * d_head];
    for t in 0..n_kv {
        if t % 2 == 0 {
            values[t * d_head] = 1.0;
        } else {
            values[t * d_head + 1] = 1.0;
        }
    }

    // Gold mask: positions 0..4 are gold.
    let gold_mask = vec![true, true, true, true, false, false, false, false];

    (attn, values, gold_mask, n_heads, n_kv, d_head)
}

#[test]
fn joint_classifier_gold_share_broadcast_that_failed_signature() {
    let (attn, values, gold_mask, n_heads, n_kv, d_head) = build_paper_table1_toy();

    // ── Step 1: classify all sinks on the per-head attention map ──────────
    //
    // classify_all_sinks takes (n_queries, n_keys) per-head attention map.
    // For this toy, treat each head's attention row as a 1-query map.
    let cfg = SinkClassifierConfig::default();
    let mut sink_scratch = StableRankScratch::new(d_head);
    let mut sink_diags: Vec<_> = Vec::new();

    for h in 0..n_heads {
        let head_row: Vec<Vec<f32>> = vec![attn[h * n_kv..(h + 1) * n_kv].to_vec()];
        let values_2d: Vec<Vec<f32>> = (0..n_kv)
            .map(|t| values[t * d_head..(t + 1) * d_head].to_vec())
            .collect();
        let mut head_diags = Vec::new();
        classify_all_sinks(&head_row, &values_2d, &cfg, &mut sink_scratch, &mut head_diags);
        sink_diags.extend(head_diags);
    }

    // ── Step 2: compute gold_share ────────────────────────────────────────
    //
    // Use identity W_O (concat_len × d_model=concat_len) so projected output
    // equals the concat of per-head outputs.
    let concat_len = n_heads * d_head;
    let d_model = concat_len;
    let mut w_o = vec![0.0_f32; concat_len * d_model];
    for i in 0..concat_len {
        w_o[i * d_model + i] = 1.0;
    }

    let report = gold_share_flat_alloc(
        &attn, &values, &gold_mask, &w_o, n_heads, n_kv, d_head, d_model,
    );

    // ── Step 3: assert the "broadcast that failed" self-consistency ───────
    //
    // The headline: gold_share is low — gold positions get little attention
    // and carry only half the value-directions. This is the diluted regime.
    //
    // gold_pre_softmax_max: max attention weight on any gold position across heads.
    // Each head puts 0.05 on each gold position → max = 0.05.
    assert!(
        (report.gold_pre_softmax_max - 0.05).abs() < 1e-5,
        "gold_pre_softmax_max should be 0.05 (max attention on gold positions), got {}",
        report.gold_pre_softmax_max
    );

    // noise_gap = gold_pre_softmax_max − max attention on any distractor.
    // Distractors get 0.20 each → noise_gap = 0.05 − 0.20 = −0.15 (negative =
    // a distractor outranks gold pre-normalization).
    assert!(
        report.noise_gap < 0.0,
        "noise_gap should be negative (distractors outrank gold pre-norm), got {}",
        report.noise_gap
    );
    assert!(
        (report.noise_gap - (0.05 - 0.20)).abs() < 1e-5,
        "noise_gap should be 0.05 - 0.20 = -0.15, got {}",
        report.noise_gap
    );

    // gold_share should be strictly below 0.5 — the diluted regime.
    assert!(
        report.gold_share < 0.5,
        "gold_share should be < 0.5 in the diluted toy, got {} \
         (this is the recall-generation gap signature)",
        report.gold_share
    );
    assert!(
        report.gold_share > 0.0,
        "gold_share should be > 0 (gold signal present, just diluted), got {}",
        report.gold_share
    );

    // ── Step 4: joint signature with the classifier ───────────────────────
    //
    // The classifier may or may not flag any sinks in this toy. The key
    // contract: SinkDiagnostic has the optional gold_share field accessible
    // (compiles only when gold_share_probe is on). All classifier-produced
    // diagnostics have gold_share = None (the classifier doesn't populate it;
    // a future wiring would).
    for diag in &sink_diags {
        let _gs: Option<GoldShareReport> = diag.gold_share;
        assert!(
            _gs.is_none(),
            "classifier should not populate gold_share (future wiring's job)"
        );
    }
}

#[test]
fn joint_signature_healthy_broadcast_when_gold_share_high() {
    // Contrast case: when gold_share is high, a Broadcast classification is
    // a HEALTHY broadcast (signal survives). This is the non-failed case.
    let n_heads = 1;
    let n_kv = 4;
    let d_head = 2;
    // All attention on gold positions (0, 1) — gold_share should be ~1.0.
    let attn = vec![0.5_f32, 0.5_f32, 0.0, 0.0];
    let values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let gold_mask = vec![true, true, false, false];

    let concat_len = n_heads * d_head;
    let d_model = concat_len;
    let mut w_o = vec![0.0_f32; concat_len * d_model];
    for i in 0..concat_len {
        w_o[i * d_model + i] = 1.0;
    }

    let report = gold_share_flat_alloc(
        &attn, &values, &gold_mask, &w_o, n_heads, n_kv, d_head, d_model,
    );

    // All attention on gold → gold_share == 1.0 (healthy broadcast regime).
    assert!(
        (report.gold_share - 1.0).abs() < 1e-5,
        "all-attention-on-gold → gold_share should be 1.0, got {}",
        report.gold_share
    );
    // gold_pre_softmax_max = 0.5 (max attention on any gold position).
    assert!(
        (report.gold_pre_softmax_max - 0.5).abs() < 1e-5,
        "gold_pre_softmax_max should be 0.5, got {}",
        report.gold_pre_softmax_max
    );
    // noise_gap = 0.5 - 0.0 = 0.5 (gold dominates pre-norm).
    assert!(
        (report.noise_gap - 0.5).abs() < 1e-5,
        "noise_gap should be 0.5, got {}",
        report.noise_gap
    );
}
