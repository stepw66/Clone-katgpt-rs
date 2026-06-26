//! Plan 248 Example: OctreeCTC Reconstructive Memory Navigation — GOAT Proof Demo
//!
//! Before/after comparison of single-shot `project_all()` vs multi-step
//! `project_reconstruct()`. Shows evidence accumulation improvement.

#![cfg(feature = "octree_ctc")]

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::sense::reconstruction::ReconstructionConfig;
use katgpt_core::types::SenseKind;

fn main() {
    println!("=== Plan 248: OctreeCTC Reconstructive Navigation GOAT Demo ===\n");

    let builder = SenseOctreeBuilder::new(3);

    let kinds = [
        SenseKind::CommonSense,
        SenseKind::FighterSense,
        SenseKind::GameTheorySense,
        SenseKind::SpatialSense,
        SenseKind::SocialSense,
        SenseKind::SkillSense,
    ];

    let kind_names = [
        "CommonSense",
        "FighterSense",
        "GameTheorySense",
        "SpatialSense",
        "SocialSense",
        "SkillSense",
    ];

    // Each module gets 3 embeddings for multi-dim projection
    let modules = vec![
        builder.build(
            SenseKind::CommonSense,
            &[
                KgEmbedding { entity_hash: 1, relation_hash: 1, embedding: [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 2, relation_hash: 2, embedding: [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 3, relation_hash: 3, embedding: [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            ],
        ),
        builder.build(
            SenseKind::FighterSense,
            &[
                KgEmbedding { entity_hash: 4, relation_hash: 4, embedding: [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 5, relation_hash: 5, embedding: [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 6, relation_hash: 6, embedding: [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            ],
        ),
        builder.build(
            SenseKind::GameTheorySense,
            &[
                KgEmbedding { entity_hash: 7, relation_hash: 7, embedding: [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 8, relation_hash: 8, embedding: [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 9, relation_hash: 9, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0], sign: true, confidence: 1.0 },
            ],
        ),
        builder.build(
            SenseKind::SpatialSense,
            &[
                KgEmbedding { entity_hash: 10, relation_hash: 10, embedding: [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 11, relation_hash: 11, embedding: [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 12, relation_hash: 12, embedding: [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            ],
        ),
        builder.build(
            SenseKind::SocialSense,
            &[
                KgEmbedding { entity_hash: 13, relation_hash: 13, embedding: [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 14, relation_hash: 14, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 15, relation_hash: 15, embedding: [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            ],
        ),
        builder.build(
            SenseKind::SkillSense,
            &[
                KgEmbedding { entity_hash: 16, relation_hash: 16, embedding: [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 17, relation_hash: 17, embedding: [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], sign: true, confidence: 1.0 },
                KgEmbedding { entity_hash: 18, relation_hash: 18, embedding: [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sign: true, confidence: 1.0 },
            ],
        ),
    ];

    let mut brain = NpcBrain::compose(modules);
    brain.hla_state = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0];

    println!("Initial HLA state: {:?}", brain.hla_state);
    println!("Modules loaded: {}\n", brain.modules.len());

    // Step 1: Single-shot passive projection
    println!("--- Single-Shot Passive (project_all) ---");
    let passive = brain.project_all();
    let mut passive_sum = 0.0f32;
    for (i, &val) in passive.iter().enumerate() {
        let name = kind_names.get(i).unwrap_or(&"?");
        println!("  {:>18}: {:.6}", name, val);
        passive_sum += val;
    }
    println!("  Total passive activation: {:.6}\n", passive_sum);

    // Step 2: Default reconstruction (3 steps)
    println!("--- Default Reconstruction (3 steps) ---");
    let result = brain.project_reconstruct();

    let mut active_sum = 0.0f32;
    for (i, (&pv, &av)) in result.passive.iter().zip(result.active.iter()).enumerate() {
        let name = kinds.get(i).map(|k| kind_names[*k as usize]).unwrap_or(&"?");
        let delta = av - pv;
        println!("  {:>18}: passive={:.6} active={:.6} delta={:+.6}", name, pv, av, delta);
        active_sum += av;
    }
    println!("  Steps taken: {}", result.steps);
    println!("  Evidence: count={}, confidence_sum={:.6}",
        result.evidence.count, result.evidence.confidence_sum);

    // Step 3: Aggressive reconstruction (5 steps, higher LR)
    println!("\n--- Aggressive Reconstruction (5 steps, lr=0.3) ---");
    let config = ReconstructionConfig {
        max_steps: 5,
        hla_learning_rate: 0.3,
        entropy_threshold: 0.01,
        ..Default::default()
    };
    let result_aggressive = brain.project_reconstruct_with_config(config);

    let mut aggressive_sum = 0.0f32;
    for (i, &av) in result_aggressive.active.iter().enumerate() {
        let name = kinds.get(i).map(|k| kind_names[*k as usize]).unwrap_or(&"?");
        println!("  {:>18}: active={:.6}", name, av);
        aggressive_sum += av;
    }
    println!("  Steps taken: {}", result_aggressive.steps);
    println!("  Evidence: count={}, confidence_sum={:.6}",
        result_aggressive.evidence.count, result_aggressive.evidence.confidence_sum);

    // Step 4: GOAT verdict
    println!("\n=== GOAT Verdict ===");
    println!("  Passive total:     {:.6}", passive_sum);
    println!("  Default (3-step):  {:.6}", active_sum);
    println!("  Aggressive (5-step): {:.6}", aggressive_sum);

    let evidence_improvement = if result.evidence.confidence_sum > 1e-6 {
        (result_aggressive.evidence.confidence_sum - result.evidence.confidence_sum)
            / result.evidence.confidence_sum * 100.0
    } else {
        0.0
    };
    println!("  Evidence improvement (aggressive vs default): {:.1}%", evidence_improvement);

    let goat_pass = evidence_improvement >= 20.0;
    if goat_pass {
        println!("  GOAT: PASS (≥20% evidence improvement threshold met)");
    } else {
        println!("  GOAT: FAIL (<20% threshold). Keep as opt-in.");
    }

    println!("\nDone.");
}
