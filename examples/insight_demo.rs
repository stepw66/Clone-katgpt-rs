//! INSIGHT Full Pipeline Demo — Plan 210, I3
//!
//! Demonstrates the explore → distill → explain pipeline:
//!   F1: Symbolic expression distillation from DDTree traces
//!   F2: Concept grounding with template-based explanations
//!   F3: Perturbation-based decision explanation with sensitivity
//!   F4: Reward-gated pruner calibration with absorption
//!
//! Run: `cargo run --features insight_explain --example insight_demo`

#[cfg(feature = "insight_explain")]
use katgpt_rs::pruners::{
    // F4: Reward Calibration
    CalibratorConfig,
    // F3: Decision Explanation
    CandidateRecord,
    // F2: Concept Grounding
    ConceptGrounding,
    DecisionExplainer,
    ParameterKey,
    PerturbationExplainer,
    PrunerState,
    RewardGatedCalibrator,
    // F1: Symbolic Expression
    SymbolicExpression,
    SymbolicExpressionFitter,
    TemplateGrounding,
    TraceNode,
    TraceRecorder,
};
#[cfg(feature = "insight_explain")]
use katgpt_rs::speculative::NoScreeningPruner;

#[cfg(feature = "insight_explain")]
const FEATURE_NAMES: &[&str] = &[
    "depth_norm",
    "score_mean",
    "syntax_validity",
    "bandit_q",
    "parent_match",
    "depth_ratio",
    "entropy",
    "freq",
];

#[cfg(feature = "insight_explain")]
fn main() {
    println!("═══ INSIGHT Pipeline Demo (Plan 210, I3) ═══\n");

    // ── Stage 1: DDTree Exploration with Trace Recording ──────────
    println!("── Stage 1: DDTree Exploration ──\n");

    let mut recorder = TraceRecorder::new();
    let mut rng = fastrand::Rng::with_seed(42);

    for depth in 0..5 {
        for token_idx in 0..20 {
            let features: Vec<f32> = (0..8).map(|_| rng.f32()).collect();
            let accepted = features[0] > 0.5 && features[2] < 0.3;
            recorder.record(depth, token_idx, features, vec![0.0; 4], accepted);
        }
    }

    let dataset = recorder.to_dataset();
    let n_accepted = dataset.labels.iter().filter(|&&b| b).count();
    println!(
        "  Recorded {} traces ({} accepted)",
        dataset.features.len(),
        n_accepted
    );

    // ── Stage 2: Symbolic Expression Fitting (F1) ─────────────────
    println!("\n── Stage 2: Symbolic Distillation (F1) ──\n");

    let mut fitter = SymbolicExpressionFitter::new();
    fitter.max_terms = 4;
    fitter.min_improvement = 0.001;

    let expr = fitter.fit(&dataset);
    println!(
        "  Expression ({} terms): {}",
        expr.terms.len(),
        expr.to_string(FEATURE_NAMES)
    );

    let bytes = expr.to_bytes();
    let _ = SymbolicExpression::from_bytes(&bytes).unwrap();
    println!("  Serialized {} bytes, blake3 round-trip OK ✓", bytes.len());

    // ── Stage 3: Reward-Gated Calibration (F4) ────────────────────
    println!("\n── Stage 3: Reward Calibration (F4) ──\n");

    let mut cal = RewardGatedCalibrator::with_config(
        NoScreeningPruner,
        CalibratorConfig {
            min_visits: 5,
            variance_threshold: 0.05,
            learning_rate: 0.1,
        },
    );

    let key_stable = ParameterKey {
        pruner_id: 0,
        parameter_idx: 0,
        depth: 0,
    };
    let key_noisy = ParameterKey {
        pruner_id: 1,
        parameter_idx: 0,
        depth: 0,
    };

    // Stable: constant rewards → absorption eligible
    for _ in 0..5 {
        cal.record_reward(key_stable, 0.85);
    }
    if let Some(s) = cal.bandit_update(key_stable, 0.85) {
        println!(
            "  Stable: {:.4} → {:.4} (Δ={:.4})",
            s.old_value, s.new_value, s.reward_delta
        );
    }
    println!("  Absorption eligible: {}", cal.should_absorb(&key_stable));

    // Noisy: alternating rewards → high variance blocks absorption
    for i in 0..6 {
        cal.record_reward(key_noisy, if i % 2 == 0 { 0.1 } else { 0.9 });
    }
    println!(
        "  Noisy absorption:    {} (variance blocks)",
        cal.should_absorb(&key_noisy)
    );
    println!("  Audit log: {} steps", cal.calibration_log().len());

    // ── Stage 4: Concept Grounding (F2) ───────────────────────────
    println!("\n── Stage 4: Concept Grounding (F2) ──\n");

    let grounding = TemplateGrounding::with_templates(vec![
        ("token_42", "function declaration"),
        ("token_7", "type annotation"),
    ]);

    let state = PrunerState {
        depth: 1,
        token_idx: 42,
        parent_token: vec![0],
        pruner_scores: vec![
            ("syntax".into(), 0.85),
            ("bandit".into(), 0.62),
            ("cache".into(), 0.30),
        ],
        accepted: true,
    };

    let mappings = grounding.ground(&state);
    println!("  Mappings ({}):", mappings.len());
    for m in &mappings {
        println!(
            "    {} → {} (conf={:.2})",
            m.variable, m.semantic, m.confidence
        );
    }

    let chain = grounding.explain_chain(&state, &mappings);
    println!("\n  Chain of thought:");
    for s in &chain {
        println!("    • {s}");
    }

    println!("\n  {}", grounding.summarize(&mappings, &chain));

    // ── Stage 5: Decision Explanation (F3) ─────────────────────────
    println!("\n── Stage 5: Decision Explanation (F3) ──\n");

    let mut node = TraceNode::new(1, 0);
    node.candidates.push(CandidateRecord {
        token_idx: 42,
        pruner_scores: vec![0.85, 0.62],
        accepted: true,
    });
    node.candidates.push(CandidateRecord {
        token_idx: 7,
        pruner_scores: vec![0.55, 0.50],
        accepted: false,
    });

    let explainer = PerturbationExplainer::new(0.1, vec!["syntax".into(), "bandit".into()]);
    let explanation = explainer.explain(&[node]);

    println!("  {}", explanation.format_report(&["syntax", "bandit"]));
    println!("  {}", explanation.summary);

    // ── Summary ────────────────────────────────────────────────────
    println!("\n═══ Summary ═══");
    println!(
        "  F1: {}-term expression from {} traces",
        expr.terms.len(),
        dataset.features.len()
    );
    println!("  F2: {} concepts grounded", mappings.len());
    println!("  F3: {} choices explained", explanation.choices.len());
    println!("  F4: {} calibration steps", cal.calibration_log().len());
}

#[cfg(not(feature = "insight_explain"))]
fn main() {
    eprintln!("This example requires the `insight_explain` feature.");
    eprintln!("Run: cargo run --example insight_demo --features insight_explain");
}
