//! Plan 284 Phase 5 — Long2Short brevity tiebreak isolated demo.
//!
//! Constructs two clusters whose `total_reliability` is within `eps` of each
//! other, then shows `brevity_tiebreak` picking the one whose representative
//! trajectory has the fewer tokens/steps. This is the "zero-quality-change
//! tiebreak" from Research 255: it NEVER changes *which* cluster wins by
//! reliability; it only breaks ties in favor of brevity.
//!
//! Run with:
//! ```bash
//! cargo run --release --example clr_brevity_tiebreak --features clr
//! ```
//!
//! # Three cases covered
//!
//! 1. **Tied within `eps`** — shorter representative wins.
//! 2. **Not tied** (Δ > `eps`) — higher-reliability cluster wins, even if longer.
//! 3. **Token-tie tiebreak** — first-encountered wins.

#![cfg(feature = "clr")]

use katgpt_claim::clr::{Cluster, Trajectory, brevity_tiebreak};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 284 — Long2Short brevity tiebreak demo");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Case 1: tied within eps → shorter rep wins ─────────────────────
    case_tiebreak_picks_shorter();

    // ── Case 2: not tied → reliability dominates length ────────────────
    case_reliability_overrides_length();

    // ── Case 3: token-tie → first-encountered wins ─────────────────────
    case_token_tie_first_wins();

    println!("═══════════════════════════════════════════════════════════════");
    println!("All cases consistent with the Long2Short zero-sum tiebreak.");
    println!("Brevity only breaks ties — it never overrides reliability.");
    println!("═══════════════════════════════════════════════════════════════");
}

// ──────────────────────────────────────────────────────────────────────────
// Case 1
// ──────────────────────────────────────────────────────────────────────────

fn case_tiebreak_picks_shorter() {
    println!("── Case 1: tied within eps → shorter rep wins ──────────────────");
    // Two clusters, identical reliability (0.80). Representative tokens: 100 vs 50.
    let trajectories: Vec<Trajectory<()>> = vec![
        Trajectory {
            outcome: (),
            tokens_or_steps: 100,
            claims: vec![],
            log_probs: None,
        },
        Trajectory {
            outcome: (),
            tokens_or_steps: 50,
            claims: vec![],
            log_probs: None,
        },
    ];
    let c0 = cluster((), 0.80, 0, &[0]);
    let c1 = cluster((), 0.80, 1, &[1]);
    let candidates: Vec<&Cluster<()>> = vec![&c0, &c1];

    let eps = 1e-3;
    let idx = brevity_tiebreak(&candidates, &trajectories, eps);

    println!("  Cluster A: Σ r_k = 0.800, rep idx 0, tokens = 100");
    println!("  Cluster B: Σ r_k = 0.800, rep idx 1, tokens =  50");
    println!("  eps       = {eps}");
    println!("  winner    : cluster {idx}");
    assert_eq!(
        idx, 1,
        "case 1: brevity should pick cluster B (shorter, 50 tokens)"
    );
    println!("  → brevity picks cluster B (50 tokens) ✅");
    println!();
}

// ──────────────────────────────────────────────────────────────────────────
// Case 2
// ──────────────────────────────────────────────────────────────────────────

fn case_reliability_overrides_length() {
    println!("── Case 2: reliability dominates length when Δ > eps ───────────");
    // Cluster A has higher reliability (0.90 vs 0.50) but is longer (200 vs 50).
    // Since 0.90 - 0.50 = 0.40 >> eps, A wins outright — brevity doesn't matter.
    let trajectories: Vec<Trajectory<()>> = vec![
        Trajectory {
            outcome: (),
            tokens_or_steps: 200,
            claims: vec![],
            log_probs: None,
        },
        Trajectory {
            outcome: (),
            tokens_or_steps: 50,
            claims: vec![],
            log_probs: None,
        },
    ];
    let c0 = cluster((), 0.90, 0, &[0]);
    let c1 = cluster((), 0.50, 1, &[1]);
    let candidates: Vec<&Cluster<()>> = vec![&c0, &c1];

    let eps = 1e-3;
    let idx = brevity_tiebreak(&candidates, &trajectories, eps);

    println!("  Cluster A: Σ r_k = 0.900, rep idx 0, tokens = 200");
    println!("  Cluster B: Σ r_k = 0.500, rep idx 1, tokens =  50");
    println!("  eps       = {eps}");
    println!("  Δ         = {:.3} (>> eps)", 0.90 - 0.50);
    println!("  winner    : cluster {idx}");
    assert_eq!(
        idx, 0,
        "case 2: reliability dominates — cluster A wins despite being longer"
    );
    println!("  → reliability wins, length is irrelevant ✅");
    println!();
}

// ──────────────────────────────────────────────────────────────────────────
// Case 3
// ──────────────────────────────────────────────────────────────────────────

fn case_token_tie_first_wins() {
    println!("── Case 3: tokens tied → first-encountered wins ────────────────");
    // Both clusters have the same reliability AND the same tokens (50 each).
    // Tiebreak rule: first-encountered wins. That's cluster A (index 0).
    let trajectories: Vec<Trajectory<()>> = vec![
        Trajectory {
            outcome: (),
            tokens_or_steps: 50,
            claims: vec![],
            log_probs: None,
        },
        Trajectory {
            outcome: (),
            tokens_or_steps: 50,
            claims: vec![],
            log_probs: None,
        },
    ];
    let c0 = cluster((), 0.70, 0, &[0]);
    let c1 = cluster((), 0.70, 1, &[1]);
    let candidates: Vec<&Cluster<()>> = vec![&c0, &c1];

    let eps = 1e-3;
    let idx = brevity_tiebreak(&candidates, &trajectories, eps);

    println!("  Cluster A: Σ r_k = 0.700, rep idx 0, tokens = 50");
    println!("  Cluster B: Σ r_k = 0.700, rep idx 1, tokens = 50");
    println!("  eps       = {eps}");
    println!("  winner    : cluster {idx}");
    assert_eq!(idx, 0, "case 3: token tie → first-encountered wins");
    println!("  → tie on tokens, first-encountered (cluster A) wins ✅");
    println!();
}

// ──────────────────────────────────────────────────────────────────────────
// Helper
// ──────────────────────────────────────────────────────────────────────────

fn cluster<T: Clone>(
    outcome: T,
    reliability: f32,
    rep_idx: usize,
    members: &[usize],
) -> Cluster<T> {
    Cluster {
        outcome,
        total_reliability: reliability,
        representative_idx: rep_idx,
        member_indices: members.to_vec(),
    }
}
