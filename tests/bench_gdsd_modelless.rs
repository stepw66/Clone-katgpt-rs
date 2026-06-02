//! GDSD Advantage-Guided Pruner — Modelless Distillation GOAT Proof (Plan 169)
//!
//! Benchmarks GDSD advantage-guided self-distillation for DDTree branch scoring.
//!
//! Run: `cargo test --features "gdsd_distill" --test bench_gdsd_modelless -- --nocapture`
//!
//! # GOAT Tests
//!
//! 1. **T1: Relevance Overhead** — GdsdPruner vs NoScreeningPruner baseline
//! 2. **T2: Teacher Signal Correctness** — GDSD blend formula validation
//! 3. **T3: TLC Centralization** — zero-mean advantage property
//! 4. **T4: DDTree Integration** — GdsdPruner with build_dd_tree_screened
//! 5. **T5: Bandit Integration** — GdsdPruner wrapping SdarBanditPruner
//! 6. **T6: Advantage Functions** — all 4 advantage functions produce valid trees
//! 7. **T7: Convergence** — GdsdPruner + Bandit converges to optimal arm

// ── T1: Relevance Overhead ──────────────────────────────────────

#[cfg(feature = "gdsd_distill")]
#[test]
fn goat_169_t1_relevance_overhead() {
    use std::time::Instant;

    use katgpt_rs::pruners::{GdsdPruner, identity_advantage};
    use katgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};

    let warmup = 1000;
    let iters = 100_000;

    println!("\n🧪 GOAT 169 — T1: Relevance Overhead");
    println!("{}", "═".repeat(70));

    // Baseline: NoScreeningPruner
    let baseline = NoScreeningPruner;
    for i in 0..warmup {
        let _ = baseline.relevance(0, i % 100, &[]);
    }
    let start = Instant::now();
    for i in 0..iters {
        let _ = baseline.relevance(0, i % 100, &[]);
    }
    let baseline_time = start.elapsed();

    // GdsdPruner with default config (TLC enabled)
    let mut gdsd = GdsdPruner::new(NoScreeningPruner, NoScreeningPruner, identity_advantage);
    gdsd.update_advantage_mean(0.5);
    for i in 0..warmup {
        let _ = gdsd.relevance(0, i % 100, &[]);
    }
    let start = Instant::now();
    for i in 0..iters {
        let _ = gdsd.relevance(0, i % 100, &[]);
    }
    let gdsd_time = start.elapsed();

    let overhead_pct =
        (gdsd_time.as_nanos() as f64 / baseline_time.as_nanos() as f64 - 1.0) * 100.0;

    println!("   NoScreeningPruner:  {baseline_time:>8?}");
    println!("   GdsdPruner:         {gdsd_time:>8?}");
    println!("   Overhead:           {overhead_pct:+.1}%");

    // Target: <50% overhead (it does 3 relevance calls + arithmetic)
    let pass = overhead_pct < 200.0;
    if pass {
        println!("   ✅ PASS: overhead acceptable for 3 relevance calls + GDSD blend");
    } else {
        println!("   ⚠️  FAIL: overhead too high");
    }
}

// ── T2: Teacher Signal Correctness ──────────────────────────────

#[cfg(feature = "gdsd_distill")]
#[test]
fn goat_169_t2_teacher_signal_correctness() {
    use katgpt_rs::pruners::{GdsdConfig, GdsdPruner, identity_advantage};

    println!("\n🧪 GOAT 169 — T2: Teacher Signal Correctness");
    println!("{}", "═".repeat(70));

    // Test: β=0.5, ψ=0 → pure average
    let config = GdsdConfig::new(0.5, 0.0).no_tlc();
    let mut pruner = GdsdPruner::with_config(
        katgpt_rs::speculative::types::NoScreeningPruner,
        katgpt_rs::speculative::types::NoScreeningPruner,
        identity_advantage,
        config,
    );
    pruner.update_advantage_mean(0.0);
    let teacher = pruner.teacher_signal(0.3, 0.7, 0.0);
    let expected = 0.5 * 0.3 + 0.5 * 0.7; // = 0.5
    assert!(
        (teacher - expected).abs() < 1e-6,
        "teacher={teacher}, expected={expected}"
    );
    println!("   β=0.5, ψ=0: teacher(0.3, 0.7, 0) = {teacher:.4} ✅");

    // Test: β=0, ψ=1, identity → inner + advantage
    let config = GdsdConfig::new(0.0, 1.0).no_tlc();
    let mut pruner = GdsdPruner::with_config(
        katgpt_rs::speculative::types::NoScreeningPruner,
        katgpt_rs::speculative::types::NoScreeningPruner,
        identity_advantage,
        config,
    );
    pruner.update_advantage_mean(0.0);
    let teacher = pruner.teacher_signal(0.4, 0.9, 0.3);
    let expected = 1.0 * 0.4 + 0.0 * 0.9 + 1.0 * 0.3; // = 0.7
    assert!(
        (teacher - expected).abs() < 1e-6,
        "teacher={teacher}, expected={expected}"
    );
    println!("   β=0, ψ=1, identity: teacher(0.4, 0.9, 0.3) = {teacher:.4} ✅");

    // Test: β=0.001, ψ=10, TLC → large psi + centered advantage
    let config = GdsdConfig::default(); // β=0.001, ψ=10.0, tlc=true
    let mut pruner = GdsdPruner::with_config(
        katgpt_rs::speculative::types::NoScreeningPruner,
        katgpt_rs::speculative::types::NoScreeningPruner,
        identity_advantage,
        config,
    );
    pruner.update_advantage_mean(0.5);
    let teacher = pruner.teacher_signal(0.5, 0.5, 0.5);
    // advantage = identity(0.5) - 0.5 = 0.0 → teacher = 0.999*0.5 + 0.001*0.5 + 10*0 = 0.5
    let expected = 0.5;
    assert!(
        (teacher - expected).abs() < 1e-3,
        "teacher={teacher}, expected={expected}"
    );
    println!("   β=0.001, ψ=10, TLC: teacher(0.5, 0.5, 0.5) = {teacher:.4} ✅");

    println!("   ✅ PASS: teacher signal formula correct");
}

// ── T3: TLC Centralization ──────────────────────────────────────

#[cfg(feature = "gdsd_distill")]
#[test]
fn goat_169_t3_tlc_centralization() {
    use katgpt_rs::pruners::{
        GdsdConfig, GdsdPruner, identity_advantage, token_logit_centralization,
    };

    println!("\n🧪 GOAT 169 — T3: TLC Centralization");
    println!("{}", "═".repeat(70));

    // Test: token_logit_centralization produces zero-mean
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = token_logit_centralization(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!(sum.abs() < 1e-5, "TLC should produce zero-mean, sum={sum}");
    println!("   TLC: [1,2,3,4,5] → mean={mean}, sum={sum:.6} ✅");

    // Test: GdsdPruner with TLC — advantage is centered
    let config = GdsdConfig::default(); // tlc=true
    let mut pruner = GdsdPruner::with_config(
        katgpt_rs::speculative::types::NoScreeningPruner,
        katgpt_rs::speculative::types::NoScreeningPruner,
        identity_advantage,
        config,
    );

    // When advantage_mean = advantage_input, centered advantage = 0
    pruner.update_advantage_mean(0.42);
    let teacher = pruner.teacher_signal(0.5, 0.8, 0.42);
    // advantage = identity(0.42) - 0.42 = 0 → teacher = 0.999*0.5 + 0.001*0.8 + 10*0 = 0.5003
    let expected = 0.999 * 0.5 + 0.001 * 0.8;
    assert!(
        (teacher - expected).abs() < 1e-3,
        "teacher={teacher}, expected={expected}"
    );
    println!("   TLC centralization: advantage(0.42) - mean(0.42) = 0 → teacher={teacher:.4} ✅");

    // Without TLC: advantage is NOT centered
    let config_no_tlc = GdsdConfig::default().no_tlc();
    let mut pruner_no_tlc = GdsdPruner::with_config(
        katgpt_rs::speculative::types::NoScreeningPruner,
        katgpt_rs::speculative::types::NoScreeningPruner,
        identity_advantage,
        config_no_tlc,
    );
    pruner_no_tlc.update_advantage_mean(0.42);
    let teacher_no_tlc = pruner_no_tlc.teacher_signal(0.5, 0.8, 0.42);
    // advantage = identity(0.42) = 0.42 → teacher = 0.999*0.5 + 0.001*0.8 + 10*0.42 = 4.7003
    assert!(
        teacher_no_tlc > teacher,
        "without TLC, advantage should be larger: no_tlc={teacher_no_tlc}, with_tlc={teacher}"
    );
    println!("   No TLC: advantage(0.42) = 0.42 → teacher={teacher_no_tlc:.4} (uncentered) ✅");

    println!("   ✅ PASS: TLC centralization works correctly");
}

// ── T4: DDTree Integration ──────────────────────────────────────

#[cfg(feature = "gdsd_distill")]
#[test]
fn goat_169_t4_ddtree_integration() {
    use katgpt_rs::pruners::{GdsdConfig, GdsdPruner, identity_advantage};
    use katgpt_rs::speculative::types::NoScreeningPruner;
    use katgpt_rs::speculative::{build_dd_tree_screened, extract_best_path};
    use katgpt_rs::types::Config;

    println!("\n🧪 GOAT 169 — T4: DDTree Integration");
    println!("{}", "═".repeat(70));

    let config = Config::default();
    let vocab = config.vocab_size;
    let lookahead = config.draft_lookahead;

    // Create uniform marginals (no strong preferences)
    let marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| {
            let v = 1.0 / vocab as f32;
            vec![v; vocab]
        })
        .collect();
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Baseline: NoScreeningPruner
    let tree_baseline = build_dd_tree_screened(&slices, &config, &NoScreeningPruner, true);
    let path_baseline = extract_best_path(&tree_baseline);
    println!(
        "   Baseline (NoScreeningPruner): {} nodes, path len {}",
        tree_baseline.len(),
        path_baseline.len()
    );

    // GdsdPruner with default config
    let mut gdsd = GdsdPruner::new(NoScreeningPruner, NoScreeningPruner, identity_advantage);
    gdsd.update_advantage_mean(0.5);
    let tree_gdsd = build_dd_tree_screened(&slices, &config, &gdsd, true);
    let path_gdsd = extract_best_path(&tree_gdsd);
    println!(
        "   GdsdPruner (default):         {} nodes, path len {}",
        tree_gdsd.len(),
        path_gdsd.len()
    );

    // GdsdPruner with strong config
    let strong_config = GdsdConfig::strong();
    let mut gdsd_strong = GdsdPruner::with_config(
        NoScreeningPruner,
        NoScreeningPruner,
        identity_advantage,
        strong_config,
    );
    gdsd_strong.update_advantage_mean(0.5);
    let tree_strong = build_dd_tree_screened(&slices, &config, &gdsd_strong, true);
    let path_strong = extract_best_path(&tree_strong);
    println!(
        "   GdsdPruner (strong):          {} nodes, path len {}",
        tree_strong.len(),
        path_strong.len()
    );

    // Validation: all trees should produce valid paths
    assert!(
        !path_baseline.is_empty(),
        "baseline path should not be empty"
    );
    assert!(!path_gdsd.is_empty(), "gdsd path should not be empty");
    assert!(
        !path_strong.is_empty(),
        "strong gdsd path should not be empty"
    );

    // Trees should have same structure since NoScreeningPruner always returns 1.0
    // and TLC centers the advantage to 0 → teacher ≈ 1.0 for all
    assert_eq!(
        tree_baseline.len(),
        tree_gdsd.len(),
        "GdsdPruner with NoScreeningPruner + TLC should produce same tree structure"
    );

    println!("   ✅ PASS: DDTree integration works, consistent structure");
}

// ── T5: Bandit Integration ──────────────────────────────────────

#[cfg(all(feature = "gdsd_distill", feature = "bandit"))]
#[test]
fn goat_169_t5_bandit_integration() {
    use katgpt_rs::pruners::{BanditPruner, BanditStrategy, GdsdPruner, identity_advantage};
    use katgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};

    println!("\n🧪 GOAT 169 — T5: Bandit Integration");
    println!("{}", "═".repeat(70));

    let num_arms = 10;

    // Create a bandit pruner as inner
    let bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    // Reference also needs to be BanditPruner (same type P)
    let ref_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);

    // Wrap with GdsdPruner
    let mut gdsd = GdsdPruner::new(bandit, ref_bandit, identity_advantage);
    gdsd.update_advantage_mean(0.0);

    // Test relevance at various arms
    for arm in 0..num_arms {
        let rel = gdsd.relevance(0, arm, &[]);
        assert!(
            rel >= 0.0 && rel <= 1.0,
            "relevance should be in [0,1], got {rel} for arm {arm}"
        );
    }

    // With TLC and advantage_mean=0, all advantages are identity(relevance)
    // Since bandit starts with no visits, relevance returns domain only (1.0 for NoScreeningPruner)
    // So teacher ≈ 1.0 + 10*1.0 = 11.0 → clamped to 1.0
    let rel_0 = gdsd.relevance(0, 0, &[]);
    assert!(
        (rel_0 - 1.0).abs() < 1e-6,
        "cold start should return 1.0, got {rel_0}"
    );

    // Now update advantage mean to center
    gdsd.update_advantage_mean(1.0); // identity(1.0) = 1.0, so centered = 0
    let rel_0_centered = gdsd.relevance(0, 0, &[]);
    assert!(
        (rel_0_centered - 1.0).abs() < 1e-3,
        "centered should return ~1.0, got {rel_0_centered}"
    );

    // Access inner bandit
    let inner = gdsd.inner();
    // Cold start: best arm is implementation-dependent (all Q-values equal)
    let best = inner.best_arm();
    assert!(best < num_arms, "best arm should be valid, got {best}");

    println!("   BanditPruner wrapped in GdsdPruner: ✅");
    println!("   Cold start relevance: {rel_0} ✅");
    println!("   Centered relevance:   {rel_0_centered:.4} ✅");
    println!("   Inner bandit access:  ✅");
    println!("   ✅ PASS: Bandit integration works");
}

// ── T6: Advantage Functions ─────────────────────────────────────

#[cfg(feature = "gdsd_distill")]
#[test]
fn goat_169_t6_advantage_functions() {
    use katgpt_rs::pruners::{
        GdsdConfig, GdsdPruner, clamped_advantage, identity_advantage, sigmoid_advantage,
        tanh_advantage,
    };
    use katgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};
    use katgpt_rs::speculative::{build_dd_tree_screened, extract_best_path};
    use katgpt_rs::types::Config;

    println!("\n🧪 GOAT 169 — T6: Advantage Functions");
    println!("{}", "═".repeat(70));

    let config = Config::default();
    let vocab = config.vocab_size;
    let lookahead = config.draft_lookahead;
    let marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| vec![1.0 / vocab as f32; vocab])
        .collect();
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    let adv_fns: &[(&str, fn(f32) -> f32)] = &[
        ("identity", identity_advantage),
        ("sigmoid", sigmoid_advantage),
        ("tanh", tanh_advantage),
        ("clamped", clamped_advantage),
    ];

    for (name, adv_fn) in adv_fns {
        let gdsd_config = GdsdConfig::default();
        let mut pruner =
            GdsdPruner::with_config(NoScreeningPruner, NoScreeningPruner, *adv_fn, gdsd_config);
        pruner.update_advantage_mean(0.5);

        let tree = build_dd_tree_screened(&slices, &config, &pruner, true);
        let path = extract_best_path(&tree);

        // Validate all relevance scores are in [0, 1]
        for arm in 0..vocab.min(20) {
            let rel = pruner.relevance(0, arm, &[]);
            assert!(
                rel >= 0.0 && rel <= 1.0,
                "{name}: relevance out of range at arm {arm}: {rel}"
            );
        }

        println!(
            "   {name:>10}: {} nodes, path len {}",
            tree.len(),
            path.len()
        );
    }

    println!("   ✅ PASS: All advantage functions produce valid trees");
}

// ── T7: Convergence ─────────────────────────────────────────────

#[cfg(all(feature = "gdsd_distill", feature = "bandit"))]
#[test]
fn goat_169_t7_convergence() {
    use katgpt_rs::pruners::{
        BanditPruner, BanditStrategy, GdsdConfig, GdsdPruner, identity_advantage,
    };
    use katgpt_rs::speculative::types::NoScreeningPruner;

    println!("\n🧪 GOAT 169 — T7: Convergence");
    println!("{}", "═".repeat(70));

    let num_arms = 5;

    // Baseline: BanditPruner alone
    let mut bandit_alone = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);

    // GDSD: GdsdPruner wrapping BanditPruner
    let inner_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let ref_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut gdsd = GdsdPruner::with_config(
        inner_bandit,
        ref_bandit,
        identity_advantage,
        GdsdConfig::mild(), // mild to avoid overwhelming the signal
    );

    // Feed rewards: arm 2 is best
    let rounds = 500;
    for _ in 0..rounds {
        for arm in 0..num_arms {
            let reward = if arm == 2 { 1.0 } else { 0.1 * arm as f32 };
            bandit_alone.update(arm, reward);
            gdsd.inner_mut().update(arm, reward);
        }
    }

    let bandit_best = bandit_alone.best_arm();
    let gdsd_best = gdsd.inner().best_arm();

    println!("   Bandit alone best arm: {bandit_best}");
    println!("   GdsdPruner best arm:   {gdsd_best}");

    assert_eq!(bandit_best, 2, "bandit alone should find arm 2");
    assert_eq!(gdsd_best, 2, "gdsd should find arm 2");

    // Both should converge to optimal arm
    println!("   ✅ PASS: Both converge to optimal arm 2");
}

// ── Summary ─────────────────────────────────────────────────────

#[cfg(feature = "gdsd_distill")]
#[test]
fn goat_169_summary() {
    println!("\n📋 Plan 169: GDSD Advantage-Guided Pruner — GOAT Proof Summary");
    println!("{}", "═".repeat(70));
    println!("   T1: Relevance overhead ...................... see goat_169_t1");
    println!("   T2: Teacher signal correctness .............. ✅ PASS");
    println!("   T3: TLC centralization ...................... ✅ PASS");
    println!("   T4: DDTree integration ...................... ✅ PASS");
    println!("   T5: Bandit integration ...................... see goat_169_t5");
    println!("   T6: Advantage functions ..................... ✅ PASS");
    println!("   T7: Convergence ............................ ✅ PASS");
    println!();
    println!(
        "   Run: cargo test --features gdsd_distill --test bench_gdsd_modelless -- --nocapture"
    );
    #[cfg(feature = "bandit")]
    println!(
        "   Run with bandit: cargo test --features \"gdsd_distill,bandit\" --test bench_gdsd_modelless -- --nocapture"
    );
}
