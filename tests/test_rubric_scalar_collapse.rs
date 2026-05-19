//! Test: Prove RubricBanditPruner collapses multi-criterion to scalar (the bug)
//!
//! Run: cargo test --features "ropd_rubric,g_zero" --test test_rubric_scalar_collapse -- --nocapture
//!
//! Bug: `compute_reward()` uses `weighted_score()` which collapses all criteria
//! to a single scalar before feeding to bandit. Two RubricVectors with identical
//! weighted_score but different per-criterion profiles produce identical reward.
//! This makes RubricPlayer strategically equivalent to GZeroPlayer (scalar δ).
//!
//! Tests:
//! 1. `test_rubric_scalar_collapse_bandit_reward` — proves same ws → same reward
//! 2. `test_rubric_vs_scalar_delta_equivalence` — proves RubricPruner ≡ DeltaPruner
//! 3. `test_rubric_absorb_uses_per_criterion_but_final_sum_collapses` — absorb filters but reward still scalar

#[cfg(all(feature = "ropd_rubric", feature = "g_zero"))]
#[test]
fn test_rubric_scalar_collapse_bandit_reward() {
    use microgpt_rs::pruners::{
        BanditPruner, BanditStrategy, RubricBanditPruner, RubricTemplate, RubricVector,
    };
    use microgpt_rs::speculative::types::NoScreeningPruner;

    let template = RubricTemplate::bomber();
    let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
    // Bomber weights: [4.0, 2.0, 1.0] (survival, safety, efficiency)
    // Sum = 7.0

    let reference = RubricVector::perfect(weights.clone(), 0);

    // Student A: strong survival (1.0), weak safety (0.0), weak efficiency (0.0)
    // weighted_score = (4*1.0 + 2*0.0 + 1*0.0) / 7.0 = 4/7 ≈ 0.571
    let student_a = RubricVector::new(vec![1.0, 0.0, 0.0], weights.clone(), 0);

    // Student B: moderate survival (0.0), strong safety (1.0), strong efficiency (1.0)
    // weighted_score = (4*0.0 + 2*1.0 + 1*1.0) / 7.0 = 3/7 ≈ 0.429
    let student_b = RubricVector::new(vec![0.0, 1.0, 1.0], weights.clone(), 0);

    // Student C: balanced (4/7 survival contribution, but from different mix)
    // We want SAME weighted_score as A but DIFFERENT per-criterion profile.
    // A: ws = 4/7. Need C with ws = 4/7 but different scores.
    // C: survival=0.5, safety=0.5, efficiency=1.0
    // ws = (4*0.5 + 2*0.5 + 1*1.0) / 7.0 = (2+1+1)/7 = 4/7 ≈ 0.571
    let student_c = RubricVector::new(vec![0.5, 0.5, 1.0], weights.clone(), 0);

    println!("\n🔬 Rubric Scalar Collapse Test");
    println!("{}", "═".repeat(60));
    println!("Reference: perfect (1.0, 1.0, 1.0)");
    println!(
        "  A: scores=(1.0, 0.0, 0.0) ws={:.4}",
        student_a.weighted_score()
    );
    println!(
        "  B: scores=(0.0, 1.0, 1.0) ws={:.4}",
        student_b.weighted_score()
    );
    println!(
        "  C: scores=(0.5, 0.5, 1.0) ws={:.4}",
        student_c.weighted_score()
    );

    // ── BUG PROOF 1: A and C have SAME weighted_score → SAME reward ──
    let ws_a = student_a.weighted_score();
    let ws_c = student_c.weighted_score();

    println!("\n📊 Proof 1: Same weighted_score → same reward");
    println!("  A.weighted_score() = {ws_a:.6}");
    println!("  C.weighted_score() = {ws_c:.6}");

    // reward = reference.ws - student.ws = 1.0 - ws
    let reward_a = reference.weighted_score() - ws_a;
    let reward_c = reference.weighted_score() - ws_c;

    println!("  reward(A) = 1.0 - {ws_a:.6} = {reward_a:.6}");
    println!("  reward(C) = 1.0 - {ws_c:.6} = {reward_c:.6}");

    assert!(
        (ws_a - ws_c).abs() < 1e-6,
        "A and C should have same weighted_score (collapse precondition)"
    );
    assert!(
        (reward_a - reward_c).abs() < 1e-6,
        "BUG: A and C get identical reward despite different per-criterion profiles"
    );
    println!("  ❌ BUG CONFIRMED: A and C get identical reward = {reward_a:.6}");

    // ── BUG PROOF 2: Per-criterion gaps are DIFFERENT but discarded ──
    let gaps_a = student_a.gap_criteria(&reference);
    let gaps_c = student_c.gap_criteria(&reference);

    println!("\n📊 Proof 2: Per-criterion gaps differ but are discarded");
    println!("  A gaps: {:?}", gaps_a);
    println!("  C gaps: {:?}", gaps_c);

    // A has gap only on criteria 1 and 2 (survival is perfect)
    // C has gaps on criteria 0 and 1 (efficiency is perfect)
    let gaps_differ = gaps_a != gaps_c;
    assert!(
        gaps_differ,
        "Per-criterion gaps MUST differ — the information exists but is discarded"
    );
    println!("  ✅ Gaps are different — information EXISTS");
    println!("  ❌ But compute_reward() uses weighted_score() — DISCARDS per-criterion info");

    // ── BUG PROOF 3: Bandit converges identically for A vs C scenarios ──
    let num_arms = 3;
    let num_criteria = 3;

    // Bandit 1: arms observe student_a-style rubrics
    let mut bandit_a: RubricBanditPruner<NoScreeningPruner> = RubricBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
        num_criteria,
    );

    // Bandit 2: arms observe student_c-style rubrics
    let mut bandit_c: RubricBanditPruner<NoScreeningPruner> = RubricBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
        num_criteria,
    );

    // Feed 100 observations of A-style vs C-style to arm 0
    for _ in 0..100 {
        bandit_a.observe_rubric(0, &student_a, &reference);
        bandit_c.observe_rubric(0, &student_c, &reference);
    }

    let reward_a_accum = bandit_a.mean_reward(0);
    let reward_c_accum = bandit_c.mean_reward(0);

    println!("\n📊 Proof 3: Bandit accumulates identical rewards");
    println!("  bandit_a.mean_reward(0) = {reward_a_accum:.6}");
    println!("  bandit_c.mean_reward(0) = {reward_c_accum:.6}");

    assert!(
        (reward_a_accum - reward_c_accum).abs() < 1e-4,
        "BUG: Bandits converge identically — rubric provides no extra signal"
    );
    println!("  ❌ BUG CONFIRMED: Both bandits see identical reward trajectory");

    // ── BUG PROOF 4: A and C have genuinely different profiles ──
    println!("\n📊 Proof 4: Profiles genuinely differ (sanity check)");
    println!("  A: survival=1.0 (perfect), safety=0.0 (terrible), efficiency=0.0 (none)");
    println!("  C: survival=0.5 (ok), safety=0.5 (ok), efficiency=1.0 (perfect)");
    println!("  These are strategically very different game states!");
    println!("  A: alive but in danger, no powerups");
    println!("  C: moderate danger, has powerups, surviving ok");

    // ── Summary ──
    println!("\n{}", "═".repeat(60));
    println!("SUMMARY: Rubric scalar collapse bug");
    println!("{}", "═".repeat(60));
    println!("Root cause: RubricBanditPruner::compute_reward() calls");
    println!("  reference.weighted_score() - student.weighted_score()");
    println!("  which collapses N criteria → 1 scalar before bandit update.");
    println!();
    println!("Impact: RubricPlayer produces identical reward signal as GZeroPlayer");
    println!("  for any two RubricVectors with the same weighted_score.");
    println!("  This explains why bomber_09_rubric_tournament shows");
    println!("  Rubric 8.0% ≈ GZero 8.0% (tied).");
    println!();
    println!("Fix: compute_reward() should use per-criterion gap vector,");
    println!("  not scalar weighted_score(). For example:");
    println!("  - Per-criterion UCB1 arms (one bandit per criterion)");
    println!("  - Weighted gap vector as multi-dimensional reward");
    println!("  - criterion_id as arm selection signal");
}

#[cfg(all(feature = "ropd_rubric", feature = "g_zero"))]
#[test]
fn test_rubric_vs_scalar_delta_equivalence() {
    use microgpt_rs::pruners::{
        BanditPruner, BanditStrategy, DeltaBanditPruner, RubricBanditPruner, RubricTemplate,
        RubricVector,
    };
    use microgpt_rs::speculative::types::NoScreeningPruner;

    // Prove that RubricBanditPruner with perfect reference is equivalent
    // to DeltaBanditPruner with scalar δ = 1.0 - weighted_score.

    let template = RubricTemplate::bomber();
    let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
    let reference = RubricVector::perfect(weights.clone(), 0);

    let num_arms = 5;
    let num_criteria = 3;
    let episodes = 200;

    let mut rubric_bandit: RubricBanditPruner<NoScreeningPruner> = RubricBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
        num_criteria,
    );

    let mut delta_bandit: DeltaBanditPruner<NoScreeningPruner> = DeltaBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
    );

    // Create rubrics that map to specific δ values
    // Arm 0: ws=0.2 → δ=0.8 (best)
    // Arm 1: ws=0.4 → δ=0.6
    // Arm 2: ws=0.6 → δ=0.4
    // Arm 3: ws=0.8 → δ=0.2
    // Arm 4: ws=1.0 → δ=0.0 (worst, perfect student)
    let arm_rubrics: Vec<RubricVector> = [0.2, 0.4, 0.6, 0.8, 1.0]
        .iter()
        .map(|&ws| {
            // Create a rubric with weighted_score = ws
            // Simple: all scores = ws
            RubricVector::new(vec![ws, ws, ws], weights.clone(), 0)
        })
        .collect();

    println!("\n🔬 Rubric vs Scalar δ Equivalence Test");
    println!("{}", "═".repeat(60));

    // Feed same episodes to both bandits
    for _ in 0..episodes {
        for arm in 0..num_arms {
            let rubric = &arm_rubrics[arm];
            let delta = 1.0 - rubric.weighted_score();

            rubric_bandit.observe_rubric(arm, rubric, &reference);
            delta_bandit.observe_delta(arm, delta);
        }
    }

    println!("After {episodes} episodes per arm:");
    for arm in 0..num_arms {
        let ws = arm_rubrics[arm].weighted_score();
        let delta = 1.0 - ws;
        let r_reward = rubric_bandit.mean_reward(arm);
        let d_reward = delta_bandit.mean_delta(arm);
        let diff = (r_reward - d_reward).abs();

        println!(
            "  Arm {arm}: ws={ws:.1} δ={delta:.1} rubric_reward={r_reward:.4} delta_reward={d_reward:.4} diff={diff:.6}"
        );

        // Both should produce identical mean rewards
        assert!(
            diff < 0.01,
            "Arm {arm}: rubric reward ({r_reward:.4}) should equal delta reward ({d_reward:.4})"
        );
    }

    println!();
    println!("✅ CONFIRMED: RubricBanditPruner ≡ DeltaBanditPruner");
    println!("   when reference_rubric is perfect (all 1.0).");
    println!("   The multi-criterion structure provides ZERO additional signal.");
    println!();
    println!("This is why bomber_09_rubric_tournament shows Rubric=GZero tied.");
    println!("The rubric vector is constructed, stored, but never actually used");
    println!("for differentiated decision-making. It collapses to scalar δ.");
}

#[cfg(feature = "ropd_rubric")]
#[test]
fn test_rubric_absorb_uses_per_criterion_but_final_sum_collapses() {
    use microgpt_rs::pruners::{
        AbsorbCompressLayer, CompressConfig, RubricGatedAbsorbCompress, RubricGatedConfig,
        RubricTemplate, RubricVector,
    };
    use microgpt_rs::speculative::types::NoScreeningPruner;

    // Prove that RubricGatedAbsorbCompress DOES use per-criterion gaps
    // for filtering (above_threshold check), but the final absorb reward
    // still collapses via compute_absorb_reward() → sum of weighted gaps.

    let template = RubricTemplate::bomber();
    let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
    let reference = RubricVector::perfect(weights.clone(), 0);
    // Need 2 references (default min_references=2)
    let references = vec![reference.clone(), reference.clone()];

    let num_arms = 3;
    let config = CompressConfig::new(50, 0.1, 5, 1000);

    let inner = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config);
    let mut absorb = RubricGatedAbsorbCompress::new(inner, num_arms, RubricGatedConfig::default());

    // Student A: gap only on criterion 0 (high weight 4.0, gap=1.0)
    // scores=(0.0, 1.0, 1.0), gap_threshold=0.3 → criterion 0 gap=1.0 > 0.3 ✅
    let student_a = RubricVector::new(vec![0.0, 1.0, 1.0], weights.clone(), 0);
    // Student B: gap only on criterion 2 (low weight 1.0, gap=1.0)
    // scores=(1.0, 1.0, 0.0), gap_threshold=0.3 → criterion 2 gap=1.0 > 0.3, but weight=1.0 < 2.0 ❌
    let student_b = RubricVector::new(vec![1.0, 1.0, 0.0], weights.clone(), 0);

    println!("\n🔬 Rubric Absorb Per-Criterion Filtering Test");
    println!("{}", "═".repeat(60));

    // Observe both with 2 references (meets min_references=2)
    absorb.observe_rubric(0, &student_a, &references);
    absorb.observe_rubric(1, &student_b, &references);

    let gaps_a = absorb.last_gaps(0);
    let gaps_b = absorb.last_gaps(1);

    println!("Student A (gap on criterion 0, weight=4.0):");
    for (idx, gap, weight) in gaps_a {
        println!(
            "  criterion {idx}: gap={gap:.2}, weight={weight:.1}, w*gap={:.2}",
            gap * weight
        );
    }

    println!("Student B (gap on criterion 2, weight=1.0):");
    for (idx, gap, weight) in gaps_b {
        println!(
            "  criterion {idx}: gap={gap:.2}, weight={weight:.1}, w*gap={:.2}",
            gap * weight
        );
    }

    let above_a = absorb.is_above_threshold(0);
    let above_b = absorb.is_above_threshold(1);

    println!("\nabove_threshold(A) = {above_a} (gap on high-weight criterion)");
    println!("above_threshold(B) = {above_b} (gap on low-weight criterion)");

    // With default config: gap_threshold=0.3, min_weight_for_absorb=2.0
    // A: criterion 0 gap=1.0, weight=4.0 → 4.0 >= 2.0 AND 1.0 >= 0.3 → above ✅
    // B: criterion 2 gap=1.0, weight=1.0 → 1.0 < 2.0 → below ❌
    assert!(above_a, "A should be above threshold (high-weight gap)");
    assert!(!above_b, "B should be below threshold (low-weight gap)");

    println!("\n✅ GOOD: Per-criterion filtering DOES work correctly.");
    println!("   High-weight gaps are detected, low-weight gaps filtered.");
    println!();
    println!("❌ BUT: The absorb reward still collapses:");
    println!("   compute_absorb_reward() = Σ(weight * gap) for filtered criteria");
    println!("   This is a weighted sum → single scalar → same collapse.");
    println!();
    println!("The filtering is useful (blocks low-weight gaps),");
    println!("but the bandit reward signal is still scalar.");
    println!("This is a partial fix — filtering helps, but doesn't enable");
    println!("per-criterion learning or differentiated arm selection.");
}
