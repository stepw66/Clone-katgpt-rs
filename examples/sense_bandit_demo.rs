//! Plan 221 T12: Sense Bandit Self-Learning Demo.
//!
//! Demonstrates self-play loop → sense trials → AbsorbCompress → HotSwap,
//! GM lock preventing bandit swap, and confidence evolution over N episodes.

use katgpt_core::sense::bandit::{SenseTrial, SenseTrialLog, decay_direction};
use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::hotswap::SenseHotSwap;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::SenseKind;

/// Sigmoid: squash raw value to (0, 1). Never softmax.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Simulated reward: higher activation → higher reward with noise.
fn simulated_reward(activation: f32, rng_val: f32) -> f32 {
    sigmoid(activation * 2.0 - 1.0 + rng_val * 0.3)
}

/// Deterministic pseudo-noise from episode index (no rand dep).
fn pseudo_noise(ep: u32, seed: u32) -> f32 {
    ((ep.wrapping_mul(seed)) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn main() {
    println!("=== Plan 221 T12: Sense Bandit Self-Learning Demo ===\n");

    let builder = SenseOctreeBuilder::new(3);
    let common = builder.build(
        SenseKind::CommonSense,
        &[KgEmbedding {
            entity_hash: 1,
            relation_hash: 1,
            embedding: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
        }],
    );
    let fighter = builder.build(
        SenseKind::FighterSense,
        &[KgEmbedding {
            entity_hash: 2,
            relation_hash: 2,
            embedding: [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
        }],
    );
    let spatial = builder.build(
        SenseKind::SpatialSense,
        &[KgEmbedding {
            entity_hash: 3,
            relation_hash: 3,
            embedding: [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
        }],
    );

    let mut brain = NpcBrain::compose(vec![common, fighter, spatial]);
    brain.hla_state = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];

    let kinds = [
        SenseKind::CommonSense,
        SenseKind::FighterSense,
        SenseKind::SpatialSense,
    ];
    let hotswap = SenseHotSwap::new(&kinds);
    let mut trial_log = SenseTrialLog::default();

    let initial_confidence: Vec<f32> = brain.modules.iter().map(|m| m.confidence).collect();
    println!("--- Initial State ---");
    for m in &brain.modules {
        println!(
            "  {:?}: confidence={:.4}, version={}",
            m.kind, m.confidence, m.version
        );
    }

    // ── Phase 1: Self-play (100 episodes) ──────────────────────
    println!("\n--- Phase 1: Self-Play (100 episodes) ---");
    let alpha = 0.3;

    for ep in 0..100u32 {
        let activations = brain.project_all();
        let (best_idx, best_act) = activations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let reward = simulated_reward(*best_act, pseudo_noise(ep, 2654435761));
        let kind = brain.modules[best_idx].kind;
        trial_log.record(SenseTrial {
            npc_id: 1,
            sense_kind: kind,
            activation: *best_act,
            action_taken: best_idx as u32,
            reward,
        });

        // AbsorbCompress: update module confidence via EMA
        decay_direction(
            &mut brain.modules[best_idx],
            trial_log.trials.last().unwrap(),
            alpha,
        );

        // HotSwap attempt every 20 episodes
        if ep % 20 == 19 {
            let _ = hotswap.swap(kind, brain.modules[best_idx].clone());
        }
    }

    println!("\n--- Confidence Evolution (Phase 1) ---");
    for (i, m) in brain.modules.iter().enumerate() {
        println!(
            "  {:?}: {:.4} → {:.4}  (avg_reward={:.4})",
            m.kind,
            initial_confidence[i],
            m.confidence,
            trial_log.average_reward(m.kind)
        );
    }

    // ── Phase 2: GM Lock FighterSense ──────────────────────────
    println!("\n--- Phase 2: GM Lock FighterSense (50 episodes) ---");
    let locked_confidence = brain.modules[1].confidence;
    hotswap.lock(SenseKind::FighterSense);
    println!(
        "  FighterSense LOCKED at confidence={:.4}",
        locked_confidence
    );

    for ep in 0..50u32 {
        let activations = brain.project_all();
        let (best_idx, best_act) = activations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let reward = simulated_reward(*best_act, pseudo_noise(ep, 2246822519));
        let kind = brain.modules[best_idx].kind;
        trial_log.record(SenseTrial {
            npc_id: 1,
            sense_kind: kind,
            activation: *best_act,
            action_taken: best_idx as u32,
            reward,
        });

        decay_direction(
            &mut brain.modules[best_idx],
            trial_log.trials.last().unwrap(),
            alpha,
        );

        // HotSwap attempt — FighterSense should be rejected
        if ep % 10 == 9 {
            let result = hotswap.swap(SenseKind::FighterSense, brain.modules[1].clone());
            println!(
                "    ep {}: FighterSense swap → {}",
                ep,
                if result.is_ok() {
                    "OK"
                } else {
                    "BLOCKED (locked)"
                }
            );
        }
    }

    // ── Summary ────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!(
        "{:<16} {:>12} {:>12} {:>12} {:>10}",
        "Module", "Initial", "After P1", "After P2", "Locked"
    );
    println!("{}", "-".repeat(64));
    for (i, m) in brain.modules.iter().enumerate() {
        println!(
            "{:<16} {:>12.4} {:>12.4} {:>12.4} {:>10}",
            format!("{:?}", m.kind),
            initial_confidence[i],
            if i == 1 {
                locked_confidence
            } else {
                m.confidence
            },
            m.confidence,
            if i == 1 { "YES" } else { "no" }
        );
    }
    println!("\nTotal trials: {}", trial_log.trials.len());
    println!(
        "BLAKE3 verified: {:>3}/3",
        brain.modules.iter().filter(|m| m.verify()).count()
    );
    println!("\nDone.");
}
