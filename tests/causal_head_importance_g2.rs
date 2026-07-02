//! G2 (bystander discrimination) gate for CausalHeadImportance (Plan 358 Phase 3).
//!
//! The paper's strongest quality claim: the causal score filters *correlated
//! bystanders* — heads that observational attention-mass wrongly promotes. A
//! correlated bystander attends strongly to the needle (high attention-mass
//! score!) but its output projects to zero in the readout direction (orthogonal
//! — overridden downstream), so its causal IE is exactly 0.
//!
//! This test reproduces the discrimination on a synthetic harness, comparing the
//! **causal partition** (`partition_by_causal_score`, this plan) against the
//! **attention-mass partition** (RTPurbo's real `calibrate_from_scores`,
//! Plan 126) — apples-to-apples, same head counts, same K.
//!
//! Head categories:
//! - **K load-bearing**: attend to needle (moderate) AND project into readout.
//! - **M correlated-bystander**: attend to needle STRONGLY (more than
//!   load-bearing — that's why attention-mass flags them) but project to zero.
//! - **N local**: attend locally (low mass), project to zero.
//!
//! Discrimination verdict:
//! - Causal top-K == load-bearing set (Jaccard 1.0) at every bystander fraction.
//! - Attention-mass top-K includes bystanders (Jaccard < 1.0) once bystanders exist.

use katgpt_core::causal_head_importance::{direct_effect_importance, partition_by_causal_score};
use katgpt_rs::rt_turbo::calibrate_from_scores;
use katgpt_rs::types::RtTurboConfig;

const N_HEADS: usize = 16;
const K_LOAD_BEARING: usize = 4;

/// Per-head profile: how strongly it attends to the needle (attention-mass)
/// and how much its output projects into the readout direction (causal IE).
#[derive(Clone, Copy)]
struct HeadProfile {
    /// Needle attention mass in [0,1]. RTPurbo ranks by this.
    attention_mass: f32,
    /// Readout projection. 1.0 = load-bearing, 0.0 = orthogonal (bystander/local).
    projection: f32,
}

/// Build a head-profile set with `n_bystanders` correlated-bystander heads.
///
/// Layout (deterministic, for stable Jaccard comparisons):
/// - heads `[0, K)`                → load-bearing (moderate attention, projects).
/// - heads `[K, K+n_bystanders)`   → correlated bystanders (HIGH attention, no projection).
/// - heads `[K+n_bystanders, N)`   → local (low attention, no projection).
fn build_profiles(n_bystanders: usize) -> Vec<HeadProfile> {
    assert!(K_LOAD_BEARING + n_bystanders <= N_HEADS);
    let mut profiles = Vec::with_capacity(N_HEADS);
    for h in 0..N_HEADS {
        let (attention, projection) = if h < K_LOAD_BEARING {
            // Load-bearing: attend to needle moderately, project into readout.
            // Lower attention than bystanders — the bystander pathology is that
            // pure attention-mass prefers the bystanders.
            (0.78, 1.0)
        } else if h < K_LOAD_BEARING + n_bystanders {
            // Correlated bystander: attend to needle STRONGLY, but output is
            // orthogonal to the readout (project to zero → causal IE = 0).
            (0.92, 0.0)
        } else {
            // Local: attend locally, no projection.
            (0.08, 0.0)
        };
        profiles.push(HeadProfile {
            attention_mass: attention,
            projection,
        });
    }
    profiles
}

/// Causal IE per head from the linear-map mock forward pass:
/// `m_clean = Σ projections`, `m_corrupt = 0`, `m_patched(h) = m_clean − projection_h`.
fn causal_ie_scores(profiles: &[HeadProfile]) -> Vec<f32> {
    let m_clean: f32 = profiles.iter().map(|p| p.projection).sum();
    let m_corrupt = 0.0f32;
    profiles
        .iter()
        .map(|p| {
            let m_patched = m_clean - p.projection;
            direct_effect_importance(m_clean, m_corrupt, m_patched)
        })
        .collect()
}

/// Attention-mass scores (what RTPurbo ranks by).
fn attention_mass_scores(profiles: &[HeadProfile]) -> Vec<f32> {
    profiles.iter().map(|p| p.attention_mass).collect()
}

/// The ground-truth load-bearing head set.
fn load_bearing_set() -> std::collections::HashSet<usize> {
    (0..K_LOAD_BEARING).collect()
}

/// Jaccard similarity between two sets.
fn jaccard(
    a: &std::collections::HashSet<usize>,
    b: &std::collections::HashSet<usize>,
) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

/// Build an RtTurboConfig with the SAME retrieval_head_ratio as the causal
/// partition's critical_ratio, so both methods select the same K heads — an
/// apples-to-apples comparison. (RTPurbo's default 0.15 would select K=3 on
/// n=16, while the causal partition selects K=4 at ratio 4/16=0.25.)
fn fair_config() -> RtTurboConfig {
    RtTurboConfig {
        retrieval_head_ratio: K_LOAD_BEARING as f32 / N_HEADS as f32,
        ..Default::default()
    }
}

#[test]
fn g2_causal_excludes_bystanders_attention_mass_includes_them() {
    // 25% bystanders (4 of 16): the canonical case.
    let profiles = build_profiles(4);
    let lb_set = load_bearing_set();

    let causal_scores = causal_ie_scores(&profiles);
    let attn_scores = attention_mass_scores(&profiles);

    // Causal partition: top-K by IE. Bystanders have IE=0 → excluded.
    let ratio = K_LOAD_BEARING as f32 / N_HEADS as f32;
    let (causal_critical, _) = partition_by_causal_score(&causal_scores, ratio, None, false);
    let causal_set: std::collections::HashSet<usize> = causal_critical.iter().copied().collect();
    let causal_jaccard = jaccard(&causal_set, &lb_set);

    // Attention-mass partition: top-K by needle attention. Bystanders attend
    // more strongly (0.92 > 0.78) → they displace load-bearing in the top-K.
    // Uses the SAME ratio (0.25 → K=4) as the causal partition for fairness.
    let config = fair_config();
    let attn_cal = calibrate_from_scores(&attn_scores, &config);
    let attn_set: std::collections::HashSet<usize> =
        attn_cal.retrieval_set.iter().copied().collect();
    let attn_jaccard = jaccard(&attn_set, &lb_set);

    // The discrimination: causal perfect (1.0), attention-mass imperfect (< 1.0).
    assert!(
        (causal_jaccard - 1.0).abs() < 1e-6,
        "causal partition should be perfect (Jaccard 1.0), got {causal_jaccard}: {causal_set:?}"
    );
    assert!(
        attn_jaccard < 1.0,
        "attention-mass partition should include bystanders (Jaccard < 1.0), got {attn_jaccard}: {attn_set:?}"
    );
    // Attention-mass top-K must include at least one bystander.
    let bystanders_in_attn = attn_set.iter().filter(|h| **h >= K_LOAD_BEARING && **h < K_LOAD_BEARING + 4).count();
    assert!(
        bystanders_in_attn > 0,
        "attention-mass top-K should include ≥1 bystander, got 0: {attn_set:?}"
    );
    // And causal excludes all bystanders.
    let bystanders_in_causal = causal_set.iter().filter(|h| **h >= K_LOAD_BEARING && **h < K_LOAD_BEARING + 4).count();
    assert_eq!(
        bystanders_in_causal, 0,
        "causal top-K should include 0 bystanders, got {bystanders_in_causal}: {causal_set:?}"
    );
}

#[test]
fn g2_causal_invariant_to_bystander_fraction_attention_mass_degrades() {
    // Vary the bystander fraction {0%, 25%, 50%} and show causal Jaccard stays
    // 1.0 while attention-mass Jaccard drops.
    let lb_set = load_bearing_set();
    let config = fair_config();
    let ratio = K_LOAD_BEARING as f32 / N_HEADS as f32;

    let mut results: Vec<(usize, f32, f32)> = Vec::new(); // (n_bystanders, causal_jac, attn_jac)
    for &n_bystanders in &[0_usize, 4, 8] {
        let profiles = build_profiles(n_bystanders);

        let causal_scores = causal_ie_scores(&profiles);
        let attn_scores = attention_mass_scores(&profiles);

        let (causal_critical, _) =
            partition_by_causal_score(&causal_scores, ratio, None, false);
        let causal_set: std::collections::HashSet<usize> =
            causal_critical.iter().copied().collect();
        let causal_jac = jaccard(&causal_set, &lb_set);

        let attn_cal = calibrate_from_scores(&attn_scores, &config);
        let attn_set: std::collections::HashSet<usize> =
            attn_cal.retrieval_set.iter().copied().collect();
        let attn_jac = jaccard(&attn_set, &lb_set);

        results.push((n_bystanders, causal_jac, attn_jac));
    }

    // Causal Jaccard must be 1.0 at EVERY bystander fraction (invariance).
    for &(n_byst, causal_jac, _) in &results {
        assert!(
            (causal_jac - 1.0).abs() < 1e-6,
            "causal Jaccard != 1.0 at {n_byst} bystanders: {causal_jac}"
        );
    }
    // Attention-mass Jaccard must degrade (non-increasing) as bystander fraction grows.
    // With 0 bystanders, attention-mass finds load-bearing (they're the only
    // high-attention heads) → Jaccard 1.0. With bystanders, it picks them → drops.
    let (_, _causal_0, attn_0) = results[0]; // 0 bystanders
    let (_, causal_4, attn_4) = results[1]; // 4 bystanders
    let (_, causal_8, attn_8) = results[2]; // 8 bystanders

    // At 0 bystanders both methods agree (no bystanders to confuse attention-mass).
    assert!((attn_0 - 1.0).abs() < 1e-6, "attn-mass should be 1.0 at 0 bystanders: {attn_0}");
    // At ≥4 bystanders, attention-mass degrades below causal.
    assert!(attn_4 < causal_4, "attn-mass ({attn_4}) should be < causal ({causal_4}) at 4 bystanders");
    assert!(attn_8 < causal_8, "attn-mass ({attn_8}) should be < causal ({causal_8}) at 8 bystanders");
    // Monotonic degradation (more bystanders → worse or equal attention-mass).
    assert!(
        attn_8 <= attn_4 + 1e-6,
        "attn-mass should degrade with more bystanders: 4b={attn_4}, 8b={attn_8}"
    );

    // Print the table (visible in test output for the benchmark doc).
    eprintln!("\n=== G2 bystander discrimination (n_heads={N_HEADS}, K={K_LOAD_BEARING}) ===");
    eprintln!("{:>12} {:>14} {:>14} {:>10}", "bystanders", "causal_jac", "attn_jac", "verdict");
    for &(n_byst, cj, aj) in &results {
        let verdict = if (cj - 1.0).abs() < 1e-6 && aj < cj { "causal wins" } else { "tie/agree" };
        eprintln!("{:>12} {:>14.3} {:>14.3} {:>10}", n_byst, cj, aj, verdict);
    }
}

#[test]
fn g2_causal_strictly_dominates_on_bystander_workload() {
    // Headline assertion: on a workload with bystanders, causal > attention-mass.
    // This is the promote/demote input for Phase 4: causal strictly dominates
    // on the bystander discrimination task.
    let profiles = build_profiles(6); // 6 bystanders (37.5%)
    let lb_set = load_bearing_set();
    let config = fair_config();
    let ratio = K_LOAD_BEARING as f32 / N_HEADS as f32;

    let causal_scores = causal_ie_scores(&profiles);
    let attn_scores = attention_mass_scores(&profiles);

    let (causal_critical, _) = partition_by_causal_score(&causal_scores, ratio, None, false);
    let causal_set: std::collections::HashSet<usize> = causal_critical.iter().copied().collect();

    let attn_cal = calibrate_from_scores(&attn_scores, &config);
    let attn_set: std::collections::HashSet<usize> = attn_cal.retrieval_set.iter().copied().collect();

    let causal_jac = jaccard(&causal_set, &lb_set);
    let attn_jac = jaccard(&attn_set, &lb_set);

    // Causal recovers the EXACT load-bearing set; attention-mass does not.
    assert_eq!(causal_jac, 1.0, "causal must be perfect");
    assert!(
        attn_jac < causal_jac,
        "attention-mass ({attn_jac}) must be < causal ({causal_jac}) on bystander workload"
    );
}
