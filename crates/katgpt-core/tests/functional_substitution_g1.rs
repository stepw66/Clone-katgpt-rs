//! Plan 353 Phase 2 — G1 correctness gate for `HeadSubstitutionGate`.
//!
//! These are the integration-test-level G1 cases (the inline `#[cfg(test)]`
//! module in `gate.rs` covers the same matrix; this file is the public,
//! feature-gated regression surface). The four required cases per Plan 353 T2.1:
//!
//! 1. **Identity surrogate** (IoU = 1.0, faithfulness delta = 0) → accept.
//! 2. **Disjoint surrogate** (IoU = 0.0) → reject (regardless of faithfulness).
//! 3. **Partial-overlap surrogate** at known IoU → accept iff
//!    `tau_iou ≤ iou AND worst_case_delta ≤ tau_behavior`.
//! 4. **High IoU but high behavior delta** → reject (faithfulness veto).
//!
//! Also covers the `iou` primitive's hand-computed G1 cases (identity → 1.0,
//! disjoint → 0.0, half-overlap → known fraction).

#![cfg(feature = "functional_substitution_gate")]
#![allow(clippy::float_cmp)]

use katgpt_core::faithfulness::types::FaithfulnessProfile;
use katgpt_core::functional_substitution::{HeadSubstitutionGate, iou, worst_case_behavior_delta};

fn profile(empty: f32, shuf: f32, irrel: f32, fill: f32) -> FaithfulnessProfile<f32> {
    FaithfulnessProfile {
        empty_delta: empty,
        shuffle_or_corrupt_delta: shuf,
        irrelevant_delta: irrel,
        filler_delta: fill,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// `iou` primitive — hand-computed G1 cases
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn g1_iou_identity_is_one() {
    let a = [0.5f32, 0.3, 0.2];
    assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
}

#[test]
fn g1_iou_disjoint_is_zero() {
    let a = [1.0f32, 0.0, 0.0, 0.0];
    let b = [0.0f32, 0.0, 1.0, 0.0];
    assert!(iou(&a, &b).abs() < 1e-6);
}

#[test]
fn g1_iou_half_overlap_known_value() {
    // Σ min = 1, Σ max = 3 → IoU = 1/3.
    let a = [1.0f32, 1.0, 0.0, 0.0];
    let b = [1.0f32, 0.0, 1.0, 0.0];
    assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
}

// ──────────────────────────────────────────────────────────────────────────
// `worst_case_behavior_delta` — adaptation of the real FaithfulnessProfile<D>
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn g1_worst_case_excludes_empty_delta() {
    // empty_delta is the graceful-absence baseline (small for faithful
    // consumers). It must NOT count toward the substitution-cost worst case.
    let p = profile(99.0, 0.1, 0.1, 0.1);
    assert!(
        (worst_case_behavior_delta(&p) - 0.1).abs() < 1e-6,
        "empty_delta=99 must not inflate worst_case"
    );
}

#[test]
fn g1_worst_case_picks_max_of_disruptive_interventions() {
    // shuffle/corrupt is the max.
    assert_eq!(worst_case_behavior_delta(&profile(0.0, 5.0, 3.0, 2.0)), 5.0);
    // irrelevant is the max.
    assert_eq!(worst_case_behavior_delta(&profile(0.0, 1.0, 7.0, 2.0)), 7.0);
    // filler is the max.
    assert_eq!(worst_case_behavior_delta(&profile(0.0, 1.0, 2.0, 9.0)), 9.0);
}

// ──────────────────────────────────────────────────────────────────────────
// `HeadSubstitutionGate::should_substitute` — the four required G1 cases
// ──────────────────────────────────────────────────────────────────────────

/// G1 case 1: identity surrogate (IoU=1.0, all-zero deltas) → accept.
#[test]
fn g1_identity_surrogate_accepted() {
    let zero = profile(0.0, 0.0, 0.0, 0.0);
    let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![zero]);
    assert!(gate.should_substitute(0, 1.0));
}

/// G1 case 2: disjoint surrogate (IoU=0.0) → reject regardless of faithfulness.
#[test]
fn g1_disjoint_surrogate_rejected_regardless_of_faithfulness() {
    // Even a fully-zero profile (worst-case delta = 0) cannot rescue a
    // disjoint surrogate.
    let zero = profile(0.0, 0.0, 0.0, 0.0);
    let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![zero]);
    assert!(!gate.should_substitute(0, 0.0));
}

/// G1 case 3: partial-overlap surrogate at known IoU — boundary conditions.
#[test]
fn g1_partial_overlap_boundary_conditions() {
    let small_delta = profile(0.0, 0.1, 0.1, 0.1); // worst = 0.1 ≤ 0.16
    let large_delta = profile(0.0, 0.5, 0.5, 0.5); // worst = 0.5 > 0.16

    // tau_iou = 0.4, tau_behavior = 0.16.
    let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![small_delta, large_delta]);

    // IoU = 0.4 is exactly at the boundary; strict `<` means 0.4 is accepted.
    assert!(
        gate.should_substitute(0, 0.4),
        "boundary IoU accepted for small-delta head"
    );
    // Same IoU, head 1 has a large delta → vetoed.
    assert!(
        !gate.should_substitute(1, 0.4),
        "large-delta head vetoed at boundary IoU"
    );
    // Just below threshold → rejected.
    assert!(!gate.should_substitute(0, 0.399), "below tau_iou rejected");
}

/// G1 case 4: high IoU but high behavior delta → rejected (faithfulness veto).
#[test]
fn g1_high_iou_high_delta_rejected_by_faithfulness_veto() {
    let load_bearing = profile(0.0, 0.9, 0.9, 0.9); // worst = 0.9 ≫ 0.16
    let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![load_bearing]);
    // IoU = 1.0 passes the cheap proxy, but the faithfulness veto fires.
    assert!(!gate.should_substitute(0, 1.0));
}

// ──────────────────────────────────────────────────────────────────────────
// Defensive edge cases (not in the plan's required-4, but worth pinning)
// ──────────────────────────────────────────────────────────────────────────

/// Un-profiled head (beyond cache) → rejected defensively.
#[test]
fn g1_unprofiled_head_rejected() {
    let gate: HeadSubstitutionGate<f32> = HeadSubstitutionGate::empty(0.4, 0.16);
    assert!(!gate.should_substitute(0, 1.0));
    assert_eq!(gate.num_heads(), 0);
}

/// The gate works with `f64` deltas too (generic over `D`).
#[test]
fn g1_gate_compiles_with_f64_delta_metric() {
    let p = FaithfulnessProfile {
        empty_delta: 0.0_f64,
        shuffle_or_corrupt_delta: 0.0,
        irrelevant_delta: 0.0,
        filler_delta: 0.0,
    };
    let gate = HeadSubstitutionGate::new(0.4, 0.16_f64, vec![p]);
    assert!(gate.should_substitute(0, 1.0));
    assert_eq!(gate.tau_behavior(), 0.16_f64);
}

/// Realistic blend: a head with moderate IoU and moderate delta at the
/// acceptance boundary.
#[test]
fn g1_realistic_blend_at_acceptance_boundary() {
    // A head that is "somewhat" causally load-bearing (worst = 0.16 exactly).
    let boundary = profile(0.0, 0.16, 0.10, 0.12);
    let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![boundary]);
    // IoU at boundary AND delta at boundary → accept (both use `<=`/`>=` semantics).
    assert!(gate.should_substitute(0, 0.4));
    // One tick worse on IoU → reject.
    assert!(!gate.should_substitute(0, 0.3999));
}
