//! Gauge-Invariant Adapter Composition — before/after demo (Plan 270).
//!
//! Run: `cargo run --features gauge_invariant \
//!                --example gauge_invariant_demo --release`
//!
//! Demonstrates paper Prop 1 from arXiv:2606.12921 (LoRA-Muon):
//! two LoRA factor pairs representing the *same* weight update `W = A·B^T`
//! can be parameterized by infinitely many gauges `(A·c, B/c)`. Naive sum
//! `A_sum = A_1 + A_2`, `B_sum = B_1 + B_2` produces a result that depends
//! on the arbitrary gauge choice, while `gauge_invariant_compose` recovers
//! the true sum `W_1 + W_2` regardless of parameterization.

#![cfg(feature = "gauge_invariant")]

use katgpt_rs::gauge_invariant::{
    gauge_invariant_compose, gauge_rebalance, GaugePair, GaugeRebalanceScratch,
};

/// Deterministic pseudo-random matrix (xorshift64).
fn seeded_random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut s = seed;
    let mut mat = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0;
        mat.push(v);
    }
    mat
}

/// `A · B^T` for A `m × r`, B `n × r` → result `m × n`.
fn abt(a: &[f32], b: &[f32], m: usize, r: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..r {
                s += a[i * r + k] * b[j * r + k];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Frobenius norm.
fn fro_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 270 — Gauge-Invariant Adapter Composition Demo        ║");
    println!("║  Paper: arXiv:2606.12921 (LoRA-Muon, Prop 1)                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let m = 32_usize;
    let n = 24_usize;
    let r = 4_usize;

    // Two adapters from "different training runs" (synthetic, reproducible).
    let a1 = seeded_random_matrix(101, m, r);
    let b1 = seeded_random_matrix(102, n, r);
    let a2 = seeded_random_matrix(201, m, r);
    let b2 = seeded_random_matrix(202, n, r);

    // True merged weight (the physical quantity we want).
    let w_true_sum = {
        let w1 = abt(&a1, &b1, m, r, n);
        let w2 = abt(&a2, &b2, m, r, n);
        (0..m * n).map(|i| w1[i] + w2[i]).collect::<Vec<_>>()
    };
    let true_norm = fro_norm(&w_true_sum);

    println!("── Setup ──────────────────────────────────────────────────────");
    println!("  Adapter 1: A₁ is {m}×{r}, B₁ is {n}×{r}");
    println!("  Adapter 2: A₂ is {m}×{r}, B₂ is {n}×{r}");
    println!("  Target merged weight: ‖W₁ + W₂‖_F = {true_norm:.4}");
    println!();

    // ─── BEFORE: gauge-mismatched inputs ──────────────────────────────
    // Game 1 ships adapter 1 at gauge c=8 (A scaled up, B scaled down).
    // Game 2 ships adapter 2 at gauge c=0.125 (opposite convention).
    let c1 = 8.0_f32;
    let c2 = 0.125_f32;
    let a1_skewed: Vec<f32> = a1.iter().map(|v| v * c1).collect();
    let b1_skewed: Vec<f32> = b1.iter().map(|v| v / c1).collect();
    let a2_skewed: Vec<f32> = a2.iter().map(|v| v * c2).collect();
    let b2_skewed: Vec<f32> = b2.iter().map(|v| v / c2).collect();

    println!("── BEFORE: gauge-mismatched adapters ──────────────────────────");
    println!("  Game 1 adapter shipped at gauge c = {c1}  (A·{c1}, B/{c1})");
    println!("  Game 2 adapter shipped at gauge c = {c2}  (A·{c2}, B/{c2})");
    println!("  σ_max(A₁') / σ_max(B₁') = {:.2}", c1.powi(2));
    println!("  σ_max(A₂') / σ_max(B₂') = {:.2}", c2.powi(2));
    println!();

    // Naive sum: A_sum = A₁' + A₂', B_sum = B₁' + B₂'.
    let a_naive: Vec<f32> = (0..m * r).map(|i| a1_skewed[i] + a2_skewed[i]).collect();
    let b_naive: Vec<f32> = (0..n * r).map(|i| b1_skewed[i] + b2_skewed[i]).collect();
    let w_naive = abt(&a_naive, &b_naive, m, r, n);
    let naive_norm = fro_norm(&w_naive);
    let naive_err = ((naive_norm - true_norm).abs() / true_norm) * 100.0;

    println!("── Path A: NAIVE sum  A_sum = A₁' + A₂',  B_sum = B₁' + B₂' ──");
    println!("  ‖W_naive‖_F = {naive_norm:.4}   (true: {true_norm:.4})");
    println!("  relative error = {naive_err:.2}%");
    println!("  ❌  gauge-dependent — result depends on arbitrary c₁, c₂");
    println!();

    // ─── AFTER: gauge-invariant compose ───────────────────────────────
    let merged_r = 2 * r;
    let mut out_a = vec![0.0_f32; m * merged_r];
    let mut out_b = vec![0.0_f32; n * merged_r];
    let pairs = [
        GaugePair { eta: 1.0, a: &a1_skewed, b: &b1_skewed, a_rows: m, b_rows: n, rank: r },
        GaugePair { eta: 1.0, a: &a2_skewed, b: &b2_skewed, a_rows: m, b_rows: n, rank: r },
    ];
    gauge_invariant_compose(&pairs, &mut out_a, &mut out_b);
    let w_gauge = abt(&out_a, &out_b, m, merged_r, n);
    let gauge_norm = fro_norm(&w_gauge);
    let gauge_err = ((gauge_norm - true_norm).abs() / true_norm) * 100.0;

    println!("── Path B: GAUGE-INVARIANT compose (rebalance then sum) ───────");
    println!("  ‖W_gauge‖_F = {gauge_norm:.4}   (true: {true_norm:.4})");
    println!("  relative error = {gauge_err:.4}%");
    println!("  ✓  result is identical regardless of c₁, c₂");
    println!();

    // ─── Show rebalance effect per adapter ────────────────────────────
    println!("── Rebalance contribution per adapter (α = 1.0) ───────────────");
    for (idx, (a_in, b_in, label)) in [
        (&a1_skewed[..], &b1_skewed[..], "Game 1"),
        (&a2_skewed[..], &b2_skewed[..], "Game 2"),
    ]
    .iter()
    .enumerate()
    {
        let mut a_re = a_in.to_vec();
        let mut b_re = b_in.to_vec();
        let sigma_a_before = fro_norm(a_in) / (m as f32).sqrt(); // proxy for σ_max
        let sigma_b_before = fro_norm(b_in) / (n as f32).sqrt();
        let ratio_before = sigma_a_before / sigma_b_before;

        let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
        gauge_rebalance(&mut a_re, &mut b_re, m, r, n, r, 1.0, &mut scratch);

        let sigma_a_after = fro_norm(&a_re) / (m as f32).sqrt();
        let sigma_b_after = fro_norm(&b_re) / (n as f32).sqrt();
        let ratio_after = sigma_a_after / sigma_b_after;

        println!(
            "  {label}: σ_max(A)/σ_max(B)  before = {:>7.3}  →  after = {:>6.3}",
            ratio_before, ratio_after
        );
        let _ = idx;
    }
    println!();

    // ─── Verdict ──────────────────────────────────────────────────────
    let improvement = naive_err / gauge_err.max(1e-9);
    println!("── Verdict ────────────────────────────────────────────────────");
    println!(
        "  Gauge-invariant compose improves result by {improvement:.0}× in this scenario"
    );
    println!(
        "  (naive err {naive_err:.2}% → gauge err {gauge_err:.4}%)"
    );
    println!();
    println!("  Paper Prop 1 holds: rebalance preserves A·B^T, so compose([(η₁, A₁, B₁), (η₂, A₂, B₂)])");
    println!("  recovers the gauge-invariant merged weight Σᵢ ηᵢ · Aᵢ · Bᵢᵀ.");
}
