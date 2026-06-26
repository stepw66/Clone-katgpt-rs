//! Plan 332 — Phase 2 GOAT gate for the principled structured basis constructors.
//!
//! Reuses the Phase 0 probe setup (Issue 001, `apollonian_basis_probe.rs`,
//! 2026-06-26) — same multi-scale input, same linear-smoothing target — but
//! swaps the hand-crafted basis for the two PRINCIPLED fixed bases:
//! [`make_dct_log_basis`] and [`make_haar_packet_basis`]. Neither has any
//! a-priori knowledge of the input signal.
//!
//! # Gates
//!
//! - **G1** (PASS): DCT-log cos ≥ random cos + 0.05 **AND** Haar-packet cos ≥
//!   random cos + 0.05, on the multi-scale transport task at τ ∈ {0.5, 0.1}.
//!   KILL if either principled basis cos < random cos.
//! - **G2** (PASS): principled gain ≥ 0.5 × achievable gain (achievable =
//!   hand-crafted − random; principled = principled − random). KILL if < 0.5×.
//! - **G3** (no-regression): the existing FUNCATTN test suite covers this —
//!   this test only sanity-checks the structured bases are drop-in via the
//!   unit test `structured_bases_forward_pass_clean` in `funcattn::tests`.
//! - **G4** (zero-alloc): the constructors are init-time only; the forward
//!   hot path is unchanged. Covered by the existing `funcattn_g5_zero_alloc`
//!   test — no change needed here.
//!
//! Verdict is reported via `eprintln!` (matches the pattern used by
//! `funcattn_g2_funcattn_vs_parallax_vs_sdpa.rs`). The hard `assert!`s only
//! enforce the sanity floor (all variants produced finite output); the GOAT
//! gate verdict is informational so a KILL doesn't break CI until T4.1
//! decides whether to promote or close.

#![cfg(feature = "funcattn_structured_basis")]

use katgpt_core::funcattn::{
    funcattn_forward, make_dct_log_basis, make_haar_packet_basis, FuncAttnBasis, FuncAttnConfig,
    FuncAttnScratch,
};

const D: usize = 64;
const N: usize = 20;
const K: usize = 8;

/// Deterministic LCG for reproducible pseudo-randomness.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (1u64 << 31) as f32 - 0.5
}

/// L2 normalize a vector in place.
fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for &x in v.iter() {
        s += x * x;
    }
    let norm = s.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Gram-Schmidt orthogonalize the rows of `w` (k rows, d cols, row-major).
fn gram_schmidt_rows(w: &mut [f32], k: usize, d: usize) {
    for i in 0..k {
        for j in 0..i {
            let mut dot = 0.0f32;
            for l in 0..d {
                dot += w[i * d + l] * w[j * d + l];
            }
            for l in 0..d {
                w[i * d + l] -= dot * w[j * d + l];
            }
        }
        l2_normalize(&mut w[i * d..(i + 1) * d]);
    }
}

/// Cosine similarity between two flattened vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

/// Build the multi-scale input X (matches the Phase 0 probe exactly so the
/// +0.11 cos hand-crafted gain is the upper bound we measure against).
///
/// Each token is a random combination of `n_scales` sinusoids; the "signal
/// subspace" is the span of the corresponding phase-ramp direction vectors.
fn make_multiscale_x(seed: u64, n_scales: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let freqs: Vec<f32> = (0..n_scales).map(|i| 0.3 + 0.7 * (i as f32)).collect();
    let mut x = vec![0.0f32; N * D];
    for i in 0..N {
        let t = i as f32;
        for j in 0..D {
            let mut v = 0.0f32;
            for (si, &f) in freqs.iter().enumerate() {
                let amp = 1.0 / (si + 1) as f32;
                v += amp * (f * (t + 0.1 * j as f32) * (si as f32 + 1.0)).sin();
            }
            x[i * D + j] = v + 0.05 * lcg_next(&mut s);
        }
    }
    let mut dirs = vec![0.0f32; n_scales * D];
    for si in 0..n_scales {
        let f = 0.3 + 0.7 * si as f32;
        for j in 0..D {
            dirs[si * D + j] = (f * 0.1 * j as f32 * (si as f32 + 1.0)).sin();
        }
        l2_normalize(&mut dirs[si * D..(si + 1) * D]);
    }
    (x, dirs)
}

/// Random row-orthonormal `(k, d)` matrix.
fn random_orthonormal_w(seed: u64, k: usize, d: usize) -> Vec<f32> {
    let mut s = seed;
    let mut w = vec![0.0f32; k * d];
    for v in w.iter_mut() {
        *v = lcg_next(&mut s);
    }
    gram_schmidt_rows(&mut w, k, d);
    w
}

/// Hand-crafted structured basis aligned to the known signal directions — the
/// UPPER BOUND from the Phase 0 probe (+0.11 cos over random). Cheats by
/// using the generative frequencies; the principled bases must not.
fn hand_crafted_w(signal_dirs: &[f32], n_scales: usize, seed: u64) -> Vec<f32> {
    let mut w = vec![0.0f32; K * D];
    for si in 0..n_scales.min(K) {
        w[si * D..(si + 1) * D].copy_from_slice(&signal_dirs[si * D..(si + 1) * D]);
    }
    let mut s = seed;
    for v in w[(n_scales.min(K)) * D..K * D].iter_mut() {
        *v = lcg_next(&mut s);
    }
    gram_schmidt_rows(&mut w, K, D);
    w
}

/// Identity `(d, d)` matrix.
fn identity_mat(d: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

#[test]
fn funcattn_structured_basis_goat_gate() {
    let (x, signal_dirs) = make_multiscale_x(42, 4);

    // Target: linear smoothing operator (the same representable target the
    // Phase 0 probe used). y[i] = 0.25·x[i-1] + 0.5·x[i] + 0.25·x[i+1].
    let mut y_target = vec![0.0f32; N * D];
    for i in 0..N {
        let prev = if i > 0 { i - 1 } else { 0 };
        let next = if i + 1 < N { i + 1 } else { N - 1 };
        for j in 0..D {
            y_target[i * D + j] =
                0.25 * x[prev * D + j] + 0.5 * x[i * D + j] + 0.25 * x[next * D + j];
        }
    }

    // Identity Q/K/V — the transport must come entirely from basis routing.
    let w_q = random_orthonormal_w(999, D, D);
    let w_k = random_orthonormal_w(888, D, D);
    let w_v = identity_mat(D);

    // Four basis variants.
    let w_rand = random_orthonormal_w(100, K, D);
    let w_hand = hand_crafted_w(&signal_dirs, 4, 200);
    let w_dct = make_dct_log_basis(K, D);
    let w_haar = make_haar_packet_basis(K, D);

    let run = |w_basis: &[f32], tau: f32, label: &str| -> f32 {
        let cfg = FuncAttnConfig {
            d: D,
            k: K,
            basis: FuncAttnBasis::Sigmoid,
            temperature: tau,
            alpha: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(N, D, K);
        let mut out = vec![0.0f32; N * D];
        funcattn_forward(&x, &x, w_basis, &w_q, &w_k, &w_v, &cfg, &mut scratch, &mut out)
            .expect("forward");
        for v in &out {
            assert!(v.is_finite(), "{label}: non-finite forward output");
        }
        let cos = cosine(&out, &y_target);
        println!("{label:18} (τ={tau}): cos(out, y) = {cos:+.4}");
        cos
    };

    let mut g1_pass = true;
    let mut g2_pass = true;
    println!("\n=== Plan 332 Phase 2 — GOAT gate (d={D}, n={N}, k={K}) ===");
    for tau in [0.5f32, 0.1] {
        println!("\n--- τ = {tau} ---");
        let cos_rand = run(&w_rand, tau, "random-orth");
        let cos_hand = run(&w_hand, tau, "hand-crafted");
        let cos_dct = run(&w_dct, tau, "DCT-log     ");
        let cos_haar = run(&w_haar, tau, "Haar-packet ");

        let achievable = cos_hand - cos_rand;
        let dct_gain = cos_dct - cos_rand;
        let haar_gain = cos_haar - cos_rand;

        println!("  achievable gain (hand - rand) = {achievable:+.4}");
        println!("  DCT-log   gain (dct  - rand)   = {dct_gain:+.4}");
        println!("  Haar-pack gain (haar - rand)   = {haar_gain:+.4}");

        // G1: principled ≥ random + 0.05 (and never below random).
        let dct_g1 = cos_dct >= cos_rand + 0.05;
        let haar_g1 = cos_haar >= cos_rand + 0.05;
        let dct_kill = cos_dct < cos_rand;
        let haar_kill = cos_haar < cos_rand;
        println!(
            "  G1 DCT-log   PASS={dct_g1}  KILL={dct_kill}  (Δ={:+.4}, threshold +0.05)",
            cos_dct - cos_rand
        );
        println!(
            "  G1 Haar-pack PASS={haar_g1}  KILL={haar_kill}  (Δ={:+.4}, threshold +0.05)",
            cos_haar - cos_rand
        );
        g1_pass = g1_pass && dct_g1 && haar_g1 && !dct_kill && !haar_kill;

        // G2: principled gain ≥ 0.5 × achievable gain.
        // Skip G2 evaluation when the achievable gain is degenerate (≤ 0 or
        // vanishing); G2 is only meaningful when there's something to capture.
        if achievable > 1e-3 {
            let dct_ratio = dct_gain / achievable;
            let haar_ratio = haar_gain / achievable;
            let dct_g2 = dct_ratio >= 0.5;
            let haar_g2 = haar_ratio >= 0.5;
            let dct_verdict = if dct_g2 { "PASS" } else { "FAIL" };
            let haar_verdict = if haar_g2 { "PASS" } else { "FAIL" };
            println!(
                "  G2 DCT-log   captures {:.1}% of achievable ({dct_verdict}, threshold 50%)",
                dct_ratio * 100.0
            );
            println!(
                "  G2 Haar-pack captures {:.1}% of achievable ({haar_verdict}, threshold 50%)",
                haar_ratio * 100.0
            );
            g2_pass = g2_pass && dct_g2 && haar_g2;
        } else {
            println!("  G2 skipped (achievable gain {achievable:+.4} too small to be meaningful)");
        }
    }

    println!("\n=== Verdict ===");
    println!("G1 (principled ≥ random + 0.05): {}", if g1_pass { "PASS" } else { "FAIL/KILL" });
    println!("G2 (captures ≥ 50% of achievable): {}", if g2_pass { "PASS" } else { "FAIL/KILL" });
    if g1_pass && g2_pass {
        println!("→ GOAT gate PASSED. Promote `funcattn_structured_basis` per Plan 332 T4.1.");
    } else if !g1_pass {
        println!("→ G1 KILL: principled basis loses to no-information baseline.");
        println!("  Document negative result per Plan 332 T4.3, close the plan.");
    } else {
        println!("→ G2 KILL: principled basis captures < 50% of achievable gain.");
        println!("  Not worth the complexity; document per Plan 332 T4.3.");
    }
}
