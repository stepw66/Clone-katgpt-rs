//! Self-Learning Selectivity Router Demo (Plan 204).
//!
//! Demonstrates how per-position excess kurtosis drives adaptive CoT routing.
//! The router starts with no knowledge, observes kurtosis from simulated
//! inference distributions, and converges to optimal direct/CoT decisions
//! — zero training required.
//!
//! Run with:
//!   cargo run --features selectivity_router --example selectivity_router_demo

use katgpt_rs::speculative::kurtosis_gate::excess_kurtosis;
use katgpt_rs::speculative::{ComputeRoute, ProfileError, SelectivityRouter};

// ── Simulated distributions ───────────────────────────────────────
// These represent logit marginals the model would produce at each position.

/// Peaked (confident) — one dominant token, rest near-zero.
/// High excess kurtosis → direct mode.
const HIGH_KURTOSIS: &[f32] = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];

/// Flat (uncertain) — uniform probability over all tokens.
/// Low/negative excess kurtosis → CoT mode.
const LOW_KURTOSIS: &[f32] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Medium (bell-curve) — moderate confidence.
const MEDIUM_KURTOSIS: &[f32] = &[0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1];

const NUM_POSITIONS: usize = 20;
const NUM_REQUESTS: usize = 200;
const COT_TOKENS_PER_POS: usize = 10;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Self-Learning Selectivity Router Demo (Plan 204)         ║");
    println!("║   Adaptive CoT via per-position EMA kurtosis              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let high_k = excess_kurtosis(HIGH_KURTOSIS);
    let low_k = excess_kurtosis(LOW_KURTOSIS);
    let med_k = excess_kurtosis(MEDIUM_KURTOSIS);
    println!("Distribution kurtosis values:");
    println!("  Peaked (high):   {high_k:+.2}");
    println!("  Flat (low):      {low_k:+.2}");
    println!("  Bell-curve (med): {med_k:+.2}\n");

    // ── Part 1: Routing Decisions Over Time ────────────────────────
    part1_routing_decisions();

    // ── Part 2: CoT Token Savings ──────────────────────────────────
    part2_token_savings();

    // ── Part 3: Compute Route Distribution ─────────────────────────
    part3_compute_routes();

    // ── Part 4: Persistence Demo ───────────────────────────────────
    part4_persistence();

    // ── Part 5: Convergence Plot (ASCII) ───────────────────────────
    part5_convergence_plot();
}

// ── Distribution selector for Part 1 ──────────────────────────────

/// Returns which distribution to use for a position at a given request.
///
/// - Positions 0–4: always high kurtosis (peaked)
/// - Positions 5–9: always low kurtosis (flat)
/// - Positions 10–14: start high, transition to low at request 100
/// - Positions 15–19: start low, transition to high at request 100
fn distribution_for(position: usize, request: usize) -> &'static [f32] {
    match position {
        0..=4 => HIGH_KURTOSIS,
        5..=9 => LOW_KURTOSIS,
        10..=14 => {
            if request < 100 {
                HIGH_KURTOSIS
            } else {
                LOW_KURTOSIS
            }
        }
        15..=19 => {
            if request < 100 {
                LOW_KURTOSIS
            } else {
                HIGH_KURTOSIS
            }
        }
        _ => MEDIUM_KURTOSIS,
    }
}

/// Snapshot: count how many positions are Direct (not thinking) vs CoT (thinking).
fn routing_summary(router: &SelectivityRouter) -> (usize, usize) {
    let mut direct = 0;
    let mut cot = 0;
    for pos in 0..NUM_POSITIONS {
        if router.should_think(pos) {
            cot += 1;
        } else {
            direct += 1;
        }
    }
    (direct, cot)
}

fn part1_routing_decisions() {
    println!("━━━ Part 1: Routing Decisions Over Time ━━━\n");
    println!("Simulating {NUM_REQUESTS} inference requests across {NUM_POSITIONS} positions:");
    println!("  Pos  0–4:  always peaked (confident)     → expect Direct");
    println!("  Pos  5–9:  always flat (uncertain)        → expect CoT");
    println!("  Pos 10–14: peaked → flat transition @100  → expect adaptation");
    println!("  Pos 15–19: flat → peaked transition @100  → expect adaptation\n");

    let mut router = SelectivityRouter::with_capacity(NUM_POSITIONS);
    let snapshot_points = [1, 50, 100, 150, 200];

    for req in 1..=NUM_REQUESTS {
        // Simulate: each request touches all 20 positions
        for pos in 0..NUM_POSITIONS {
            let dist = distribution_for(pos, req);
            let k = excess_kurtosis(dist);
            router.observe(pos, k);
        }

        if snapshot_points.contains(&req) {
            let (direct, cot) = routing_summary(&router);
            println!("  Request {:>3}: Direct={direct:>2}  CoT={cot:>2}", req);
        }
    }

    println!();
}

fn part2_token_savings() {
    println!("━━━ Part 2: CoT Token Savings ━━━\n");

    // Simulate a converged router
    let mut router = SelectivityRouter::with_capacity(NUM_POSITIONS);
    for _ in 0..NUM_REQUESTS {
        for pos in 0..NUM_POSITIONS {
            let dist = distribution_for(pos, NUM_REQUESTS);
            let k = excess_kurtosis(dist);
            router.observe(pos, k);
        }
    }

    // Without router: all positions use CoT
    let tokens_without_router = NUM_POSITIONS * COT_TOKENS_PER_POS;

    // With router: direct positions use 0, CoT positions use COT_TOKENS_PER_POS
    let mut tokens_with_router = 0;
    let mut direct_positions = 0;
    for pos in 0..NUM_POSITIONS {
        if router.should_think(pos) {
            tokens_with_router += COT_TOKENS_PER_POS;
        } else {
            direct_positions += 1;
        }
    }

    let savings = tokens_without_router - tokens_with_router;
    let pct = savings as f64 / tokens_without_router as f64 * 100.0;

    println!("  Positions:                {NUM_POSITIONS}");
    println!("  CoT tokens per position:  {COT_TOKENS_PER_POS}");
    println!("  Without router:           {tokens_without_router} tokens (all CoT)");
    println!("  With router:              {tokens_with_router} tokens");
    println!("  Direct positions:         {direct_positions}/{NUM_POSITIONS}");
    println!("  Savings:                  {savings} tokens ({pct:.0}%)\n");
}

fn part3_compute_routes() {
    println!("━━━ Part 3: Compute Route Distribution ━━━\n");

    let mut router = SelectivityRouter::with_capacity(NUM_POSITIONS);
    for _ in 0..NUM_REQUESTS {
        for pos in 0..NUM_POSITIONS {
            let dist = distribution_for(pos, NUM_REQUESTS);
            let k = excess_kurtosis(dist);
            router.observe(pos, k);
        }
    }

    println!("  Position  Route             Kurtosis (EMA)");
    println!("  ───────── ───────────────── ──────────────");
    for pos in 0..NUM_POSITIONS {
        let route = router.recommend_route(pos);
        let route_str = match route {
            ComputeRoute::CpuSpeculative => "CpuSpeculative",
            ComputeRoute::GpuAutoregressive => "GpuAutoregressive",
        };
        let k = router.kurtosis_at(pos).unwrap_or(0.0);
        println!("  {pos:>8}  {route_str:<17} {k:>+8.2}");
    }

    // Summary
    let cpu_count = (0..NUM_POSITIONS)
        .filter(|&p| router.recommend_route(p) == ComputeRoute::CpuSpeculative)
        .count();
    let gpu_count = NUM_POSITIONS - cpu_count;
    println!();
    println!("  CpuSpeculative:  {cpu_count}");
    println!("  GpuAutoregressive: {gpu_count}\n");
}

fn part4_persistence() {
    println!("━━━ Part 4: Persistence (Serialize/Deserialize) ━━━\n");

    // Build a router with observations
    let mut router = SelectivityRouter::with_capacity(NUM_POSITIONS);
    for _ in 0..NUM_REQUESTS {
        for pos in 0..NUM_POSITIONS {
            let dist = distribution_for(pos, NUM_REQUESTS);
            let k = excess_kurtosis(dist);
            router.observe(pos, k);
        }
    }

    // Capture routing decisions before serialization
    let original_decisions: Vec<bool> =
        (0..NUM_POSITIONS).map(|p| router.should_think(p)).collect();

    // Serialize
    let data = router.serialize();
    println!("  Serialized profile size: {} bytes", data.len());
    println!("  Tracked positions:       {}", router.len());

    // Deserialize into new router
    let restored = match SelectivityRouter::deserialize(&data) {
        Ok(r) => r,
        Err(e) => {
            println!("  ERROR: deserialization failed: {e}");
            return;
        }
    };

    // Verify identical routing decisions
    let mut mismatches = 0;
    (0..NUM_POSITIONS).for_each(|pos| {
        if restored.should_think(pos) != original_decisions[pos] {
            mismatches += 1;
        }
    });

    let match_pct = (NUM_POSITIONS - mismatches) as f64 / NUM_POSITIONS as f64 * 100.0;
    println!("  Restored decisions:      {match_pct:.0}% match ({mismatches} mismatches)");

    // Verify kurtosis values match
    let mut kurtosis_mismatches = 0;
    for pos in 0..NUM_POSITIONS {
        let orig_k = router.kurtosis_at(pos);
        let rest_k = restored.kurtosis_at(pos);
        if orig_k != rest_k {
            kurtosis_mismatches += 1;
        }
    }
    println!("  Kurtosis values:         {match_pct:.0}% match ({kurtosis_mismatches} mismatches)");

    // Show error handling
    let bad_magic = SelectivityRouter::deserialize(&[0xFF; 16]);
    match bad_magic {
        Err(ProfileError::InvalidMagic) => println!("  Invalid magic:     correctly rejected"),
        _ => println!("  Invalid magic:     UNEXPECTED result"),
    }

    let truncated = SelectivityRouter::deserialize(&b"SLR4"[..]);
    match truncated {
        Err(ProfileError::TruncatedData) => println!("  Truncated data:    correctly rejected"),
        _ => println!("  Truncated data:    UNEXPECTED result"),
    }

    println!();
}

fn part5_convergence_plot() {
    println!("━━━ Part 5: Convergence Plot (ASCII) ━━━\n");

    // Build router with history tracking
    let mut router = SelectivityRouter::with_capacity(NUM_POSITIONS);

    println!("  Rows = time (requests), Cols = positions 0–19");
    println!("  D = Direct (no thinking), T = CoT (thinking), . = no data\n");

    // Print header
    print!("         ");
    for pos in 0..NUM_POSITIONS {
        print!("{pos:>2}");
    }
    println!();
    print!("         ");
    for _ in 0..NUM_POSITIONS {
        print!("──");
    }
    println!();

    let sample_points: &[usize] = &[1, 5, 10, 20, 50, 75, 100, 125, 150, 175, 200];
    let mut sampled_idx = 0;

    for req in 1..=NUM_REQUESTS {
        for pos in 0..NUM_POSITIONS {
            let dist = distribution_for(pos, req);
            let k = excess_kurtosis(dist);
            router.observe(pos, k);
        }

        if sampled_idx < sample_points.len() && req == sample_points[sampled_idx] {
            print!("  req {:>3}  ", req);
            for pos in 0..NUM_POSITIONS {
                let ch = if router.should_think(pos) { 'T' } else { 'D' };
                print!(" {ch}");
            }
            // Annotate transitions
            if req == 100 {
                print!("  ← transition point");
            }
            println!();
            sampled_idx += 1;
        }
    }

    println!();
    println!(
        "  Legend: D=Direct (high kurtosis/confident)  T=CoT/Thinking (low kurtosis/uncertain)"
    );
    println!("  Positions 0–4:  stable D (always peaked)");
    println!("  Positions 5–9:  stable T (always flat)");
    println!("  Positions 10–14: D→T after request 100 (adaptation)");
    println!("  Positions 15–19: T→D after request 100 (adaptation)");
    println!();
}

// TL;DR: Demo of the Self-Learning Selectivity Router (Plan 204).
// Shows routing convergence, token savings, compute mapping, persistence,
// and ASCII convergence plot. Uses actual excess_kurtosis() on simulated
// distributions — no external dependencies.
