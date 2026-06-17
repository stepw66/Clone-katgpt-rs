//! Plan 284 Phase 5 — CLR learning potential + memory-write gate demo.
//!
//! Demonstrates the curiosity feedback signals distilled from Research 255:
//!
//! - **`learning_potential`** — `S_LP(y) = -(1/|y|) * Σ log π(y_t|...)` — the
//!   per-token surprise under the current frozen brain. Higher = more
//!   surprising = more worth learning.
//! - **`should_write_memory`** — gates memory persistence on BOTH reliability
//!   AND surprise. Persists exactly the "we got it right but didn't expect to"
//!   trajectories, which are the highest-value training signal for the next
//!   freeze/thaw cycle.
//!
//! NO softmax, NO training. Pure modelless arithmetic on caller-supplied
//! per-token log-probs.
//!
//! Run with:
 //! ```bash
//! cargo run --release --example clr_learning_potential --features clr
//! ```

#![cfg(feature = "clr")]

use katgpt_rs::clr::{ClrConfig, learning_potential, should_write_memory};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 284 — Learning potential + memory-write gate");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("S_LP(y) = -(1/|y|) * Σ log π(y_t | y_<t))");
    println!("should_write_memory(r_k, S_LP) := r_k > τ_reliable ∧ S_LP > τ_curiosity");
    println!();

    let config = ClrConfig::default(); // τ_reliable = 0.5, τ_curiosity = 0.7
    println!(
        "Config: τ_reliable = {}, τ_curiosity = {}",
        config.tau_reliable, config.tau_curiosity
    );
    println!();

    // ── Trajectory A: low surprise (model was confident everywhere) ─────
    // All log-probs are mild negatives → S_LP is small. Even with high CLR
    // reliability, this trajectory is NOT worth writing — the model already
    // expected the outcome.
    println!("── Trajectory A: low surprise, high reliability ─────────────────");
    let log_probs_a = [-0.05f32, -0.10, -0.08, -0.12, -0.09, -0.07, -0.11, -0.10];
    let s_lp_a = learning_potential(log_probs_a.len(), |t| log_probs_a[t]);
    let reliability_a = 0.85; // hypothetical CLR r_k
    let write_a = should_write_memory(reliability_a, s_lp_a, &config);
    println!("  log π     = {log_probs_a:?}");
    println!("  |y|       = {}", log_probs_a.len());
    println!("  Σ log π   = {:.4}", log_probs_a.iter().sum::<f32>());
    println!("  S_LP      = {:.6}", s_lp_a);
    println!("  r_k       = {:.2}", reliability_a);
    println!("  write?    = {write_a}");
    assert!(
        !write_a,
        "A: low surprise should NOT trigger a write, even with high reliability"
    );
    println!("  → DON'T write: high reliability but not surprising ✅");
    println!();

    // ── Trajectory B: high surprise but low reliability ────────────────
    // The model was very uncertain (large negative log-probs), but CLR scored
    // the trajectory as unreliable (probably wrong). Don't write — there's no
    // point learning from a trajectory that didn't work.
    println!("── Trajectory B: high surprise, low reliability ─────────────────");
    let log_probs_b = [-3.0f32, -4.0, -2.5, -3.5, -2.8, -3.2, -4.1, -2.9];
    let s_lp_b = learning_potential(log_probs_b.len(), |t| log_probs_b[t]);
    let reliability_b = 0.20;
    let write_b = should_write_memory(reliability_b, s_lp_b, &config);
    println!("  log π     = {log_probs_b:?}");
    println!("  |y|       = {}", log_probs_b.len());
    println!("  Σ log π   = {:.4}", log_probs_b.iter().sum::<f32>());
    println!("  S_LP      = {:.6}", s_lp_b);
    println!("  r_k       = {:.2}", reliability_b);
    println!("  write?    = {write_b}");
    assert!(
        !write_b,
        "B: low reliability should NOT trigger a write, even with high surprise"
    );
    println!("  → DON'T write: surprising but unreliable ✅");
    println!();

    // ── Trajectory C: the sweet spot — high surprise AND high reliability ─
    // The model was uncertain AND the trajectory passed CLR. This is exactly
    // the "we got it right but didn't expect to" signal — the highest-value
    // training data for the next freeze/thaw direction-vector update.
    println!("── Trajectory C: high surprise AND high reliability ─────────────");
    let log_probs_c = [-3.0f32, -2.5, -3.2, -2.8, -3.5, -2.9, -3.1, -3.0];
    let s_lp_c = learning_potential(log_probs_c.len(), |t| log_probs_c[t]);
    let reliability_c = 0.80;
    let write_c = should_write_memory(reliability_c, s_lp_c, &config);
    println!("  log π     = {log_probs_c:?}");
    println!("  |y|       = {}", log_probs_c.len());
    println!("  Σ log π   = {:.4}", log_probs_c.iter().sum::<f32>());
    println!("  S_LP      = {:.6}", s_lp_c);
    println!("  r_k       = {:.2}", reliability_c);
    println!("  write?    = {write_c}");
    assert!(
        write_c,
        "C: high reliability AND high surprise should trigger a write"
    );
    println!("  → WRITE: reliable AND surprising — exact freeze/thaw target ✅");
    println!();

    // ── Boundary: S_LP monotone in surprise ─────────────────────────────
    // Doubling the per-token surprise doubles S_LP (linear in log-probs).
    println!("── Boundary check: S_LP scales with surprise ───────────────────");
    let s_lp_small = learning_potential(4, |_| -0.5);
    let s_lp_big = learning_potential(4, |_| -5.0);
    println!("  S_LP([-0.5; 4])   = {:.4}", s_lp_small);
    println!("  S_LP([-5.0; 4])   = {:.4}", s_lp_big);
    assert!(s_lp_big > s_lp_small);
    assert!(
        (s_lp_big / s_lp_small - 10.0).abs() < 1e-3,
        "10× larger |log π| should give 10× larger S_LP"
    );
    println!("  → 10× surprise → 10× S_LP (linear in |log π|) ✅");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Gate summary: `should_write_memory` selects exactly the");
    println!("reliable + surprising trajectories — the highest-value");
    println!("training signal for the next freeze/thaw cycle.");
    println!("═══════════════════════════════════════════════════════════════");
}
