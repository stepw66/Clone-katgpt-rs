//! Module-level integration tests for `personality_composition`.
//!
//! These cover the validation gates from Plan 297:
//!
//! - **T1.V1** (compose core): `compose_zero_weights_uniform`,
//!   `compose_extreme_positive_weight_selects_layer`,
//!   `compose_extreme_negative_weight_zeros_layer`,
//!   `compose_belief_confidence_decay_shrinks_contribution`.
//! - **T2.V1** (drift): `drift_positive_surprise_reinforces`,
//!   `drift_negative_surprise_penalizes`, `drift_clamps_to_w_max`,
//!   `drift_ema_tracks_recent_reward`.
//! - **T3.1–T3.3** (snapshot): see `snapshot.rs` tests; this file adds
//!   `snapshot_round_trip_with_kernel` to exercise the full
//!   snapshot/restore/verify cycle through the public API.
//!
//! Snapshot unit tests for the BLAKE3 commitment scheme live next to the
//! code in `snapshot.rs::tests` (matches the `micro_belief::snapshot`
//! convention). Phase 4 G1-G5 perf tests live in
//! `benches/personality_composition_bench.rs` (orchestrator-owned).

#![cfg(test)]

use crate::personality_composition::direction_source::LayerDirectionSource;
use crate::personality_composition::kernel::PersonalityWeightedComposition;
use crate::personality_composition::sigmoid::{sigmoid, sigmoid_into};
use crate::personality_composition::snapshot::PersonalitySnapshot;
use crate::personality_composition::types::{ArchetypeLabel, PersonalityConfig};

// ─── Mock layer ─────────────────────────────────────────────────────────
//
// Test-only `LayerDirectionSource` with explicit per-axis control. Each
// instance has:
//   - `dir` — what `direction()` returns (the "current" direction).
//   - `recent` — what `recent_direction()` returns (what drift updates on).
//   - `belief` — the `belief_confidence()` value.
//
// Implements `Clone + Copy` so we can build `[&dyn LayerDirectionSource; N]`
// from a `Vec<MockLayer>` by reference without juggling lifetimes.

#[derive(Clone, Copy)]
struct MockLayer<const D: usize> {
    dir: [f32; D],
    recent: [f32; D],
    belief: f32,
}

impl<const D: usize> LayerDirectionSource for MockLayer<D> {
    fn direction(&self, scratch: &mut [f32]) -> &[f32] {
        let n = D.min(scratch.len());
        scratch[..n].copy_from_slice(&self.dir[..n]);
        &scratch[..n]
    }

    fn recent_direction(&self) -> &[f32] {
        &self.recent
    }

    fn belief_confidence(&self) -> f32 {
        self.belief
    }
}

#[allow(dead_code)]
fn make_layers<const N: usize, const D: usize>(
    layers: [MockLayer<D>; N],
) -> [&dyn LayerDirectionSource; N] {
    let mut out: [&dyn LayerDirectionSource; N] = [&layers[0]; N]; // placeholder init
    for (i, l) in layers.iter().enumerate() {
        out[i] = l as &dyn LayerDirectionSource;
    }
    out
}

// The above doesn't quite work because lifetimes get tangled when we try to
// return a borrowed array. The idiomatic pattern is to keep the owned array
// alive in the test body and build the trait-array inline. We keep
// `make_layers` around only to avoid `unused` warnings during refactors —
// the actual tests inline the array construction.

// ─── T1.V1: compose core ────────────────────────────────────────────────

#[test]
fn compose_zero_weights_uniform() {
    // When all w[i] = 0, sigmoid(0/tau) = 0.5 for every layer, so the
    // output is `0.5 * Σ_i belief_i * d_i`. With belief=1.0 (plasma tier),
    // this reduces to `0.5 * Σ_i d_i`.
    type K = PersonalityWeightedComposition<2, 3>;
    let kernel = K::new(PersonalityConfig::default(), [0.0, 0.0]);
    let l1 = MockLayer::<3> {
        dir: [1.0, 0.0, 0.0],
        recent: [0.0; 3],
        belief: 1.0,
    };
    let l2 = MockLayer::<3> {
        dir: [0.0, 2.0, 0.0],
        recent: [0.0; 3],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 2] = [&l1, &l2];
    let mut scratch = [0.0f32; 3];
    let mut out = [0.0f32; 3];
    kernel.compose_into(&layers, &mut scratch, &mut out);
    // Expected: 0.5 * (1, 0, 0) + 0.5 * (0, 2, 0) = (0.5, 1.0, 0.0)
    assert!((out[0] - 0.5).abs() < 1e-6, "out[0] = {}", out[0]);
    assert!((out[1] - 1.0).abs() < 1e-6, "out[1] = {}", out[1]);
    assert!(out[2].abs() < 1e-6, "out[2] = {}", out[2]);
}

#[test]
fn compose_extreme_positive_weight_selects_layer() {
    // With w[i] very large positive and tau=1.0, sigmoid(w[i]/tau) → 1.0.
    // The layer's full direction passes through.
    type K = PersonalityWeightedComposition<1, 3>;
    let kernel = K::new(PersonalityConfig::default(), [20.0]);
    let l1 = MockLayer::<3> {
        dir: [0.7, -0.3, 0.4],
        recent: [0.0; 3],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&l1];
    let mut scratch = [0.0f32; 3];
    let mut out = [0.0f32; 3];
    kernel.compose_into(&layers, &mut scratch, &mut out);
    // Expected: 1.0 * 1.0 * d = d
    assert!((out[0] - 0.7).abs() < 1e-5, "out[0] = {}", out[0]);
    assert!((out[1] - (-0.3)).abs() < 1e-5, "out[1] = {}", out[1]);
    assert!((out[2] - 0.4).abs() < 1e-5, "out[2] = {}", out[2]);
}

#[test]
fn compose_extreme_negative_weight_zeros_layer() {
    // With w[i] very large negative, sigmoid(w[i]/tau) → 0.0.
    // The layer's contribution vanishes.
    type K = PersonalityWeightedComposition<1, 3>;
    let kernel = K::new(PersonalityConfig::default(), [-20.0]);
    let l1 = MockLayer::<3> {
        dir: [0.7, -0.3, 0.4],
        recent: [0.0; 3],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&l1];
    let mut scratch = [0.0f32; 3];
    let mut out = [0.0f32; 3];
    kernel.compose_into(&layers, &mut scratch, &mut out);
    for j in 0..3 {
        assert!(out[j].abs() < 1e-5, "out[{j}] = {} (should be ~0)", out[j]);
    }
}

#[test]
fn compose_belief_confidence_decay_shrinks_contribution() {
    // Two layers with identical direction, identical w=0 (sigmoid = 0.5),
    // but different belief_confidence. The output should scale linearly
    // with belief.
    type K = PersonalityWeightedComposition<2, 2>;
    let kernel = K::new(PersonalityConfig::default(), [0.0, 0.0]);
    let full = MockLayer::<2> {
        dir: [1.0, 1.0],
        recent: [0.0; 2],
        belief: 1.0,
    };
    let half = MockLayer::<2> {
        dir: [1.0, 1.0],
        recent: [0.0; 2],
        belief: 0.5,
    };
    let layers: [&dyn LayerDirectionSource; 2] = [&full, &half];
    let mut scratch = [0.0f32; 2];
    let mut out = [0.0f32; 2];
    kernel.compose_into(&layers, &mut scratch, &mut out);
    // Expected: 0.5*1.0*(1,1) + 0.5*0.5*(1,1) = (0.75, 0.75)
    assert!((out[0] - 0.75).abs() < 1e-6, "out[0] = {}", out[0]);
    assert!((out[1] - 0.75).abs() < 1e-6, "out[1] = {}", out[1]);
}

#[test]
fn compose_is_additive_not_overwriting() {
    // The kernel should ADD into out, not overwrite. Verify by pre-loading
    // a baseline.
    type K = PersonalityWeightedComposition<1, 2>;
    let kernel = K::new(PersonalityConfig::default(), [20.0]); // sigmoid → 1.0
    let l1 = MockLayer::<2> {
        dir: [1.0, 1.0],
        recent: [0.0; 2],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&l1];
    let mut scratch = [0.0f32; 2];
    let mut out = [10.0f32, -5.0f32]; // baseline
    kernel.compose_into(&layers, &mut scratch, &mut out);
    assert!((out[0] - 11.0).abs() < 1e-6, "out[0] = {}", out[0]);
    assert!((out[1] - (-4.0)).abs() < 1e-6, "out[1] = {}", out[1]);
}

// ─── T2.V1: drift ───────────────────────────────────────────────────────

#[test]
fn drift_positive_surprise_reinforces() {
    // r_observed > r_expected → surprise > 0 → w[i] increases when
    // Σ_j d_recent[i][j] > 0.
    type K = PersonalityWeightedComposition<1, 2>;
    let mut kernel = K::new(PersonalityConfig::default(), [0.0]);
    // Set r_expected to a known baseline so surprise is deterministic.
    kernel.r_expected = [0.0];
    let layer = MockLayer::<2> {
        dir: [0.0; 2],
        recent: [1.0, 1.0], // Σ = 2.0
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&layer];
    let w_before = kernel.w[0];
    kernel.drift(&layers, 1.0); // surprise = 1.0
    // Δw = alpha * surprise * Σ_j d_recent[j] = 0.01 * 1.0 * 2.0 = 0.02
    let expected = w_before + 0.02;
    assert!(
        (kernel.w[0] - expected).abs() < 1e-6,
        "w[0] = {} (expected {expected})",
        kernel.w[0]
    );
    assert!(kernel.w[0] > w_before, "drift should reinforce");
}

#[test]
fn drift_negative_surprise_penalizes() {
    // r_observed < r_expected → surprise < 0 → w[i] decreases.
    type K = PersonalityWeightedComposition<1, 2>;
    let mut kernel = K::new(PersonalityConfig::default(), [0.0]);
    kernel.r_expected = [1.0]; // baseline
    let layer = MockLayer::<2> {
        dir: [0.0; 2],
        recent: [1.0, 1.0],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&layer];
    let w_before = kernel.w[0];
    kernel.drift(&layers, 0.0); // surprise = -1.0
    let expected = w_before - 0.02; // 0.01 * (-1.0) * 2.0
    assert!(
        (kernel.w[0] - expected).abs() < 1e-6,
        "w[0] = {} (expected {expected})",
        kernel.w[0]
    );
    assert!(kernel.w[0] < w_before, "drift should penalize");
}

#[test]
fn drift_clamps_to_w_max() {
    // A huge positive surprise with strong d_recent should saturate at
    // w_max, not run off to infinity.
    type K = PersonalityWeightedComposition<1, 2>;
    let config = PersonalityConfig {
        alpha: 1.0, // giant learning rate to force saturation in one step
        w_max: 5.0,
        ..Default::default()
    };
    let mut kernel = K::new(config, [0.0]);
    let layer = MockLayer::<2> {
        dir: [0.0; 2],
        recent: [1.0, 1.0],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&layer];
    kernel.drift(&layers, 100.0); // surprise = 100, delta = 1*100*2 = 200
    assert!(
        (kernel.w[0] - 5.0).abs() < 1e-6,
        "w[0] should clamp at +w_max, got {}",
        kernel.w[0]
    );

    // And the negative direction.
    kernel.r_expected = [100.0];
    kernel.drift(&layers, -100.0); // surprise = -200, delta = -400
    assert!(
        (kernel.w[0] - (-5.0)).abs() < 1e-6,
        "w[0] should clamp at -w_max, got {}",
        kernel.w[0]
    );
}

#[test]
fn drift_ema_tracks_recent_reward() {
    // r_expected should track r_observed via EMA. With ema_decay=0.95,
    // after one drift: r_expected' = 0.95 * r_expected + 0.05 * r_observed.
    type K = PersonalityWeightedComposition<1, 1>;
    let mut kernel = K::new(PersonalityConfig::default(), [0.0]);
    kernel.r_expected = [0.0];
    let layer = MockLayer::<1> {
        dir: [0.0],
        recent: [0.0],
        belief: 1.0,
    };
    let layers: [&dyn LayerDirectionSource; 1] = [&layer];
    kernel.drift(&layers, 1.0);
    let expected = 0.95 * 0.0 + 0.05 * 1.0;
    assert!(
        (kernel.r_expected[0] - expected).abs() < 1e-6,
        "r_expected[0] = {} (expected {expected})",
        kernel.r_expected[0]
    );

    // Many drift steps with constant r_observed should converge to it.
    for _ in 0..1000 {
        kernel.drift(&layers, 1.0);
    }
    assert!(
        (kernel.r_expected[0] - 1.0).abs() < 1e-3,
        "r_expected should converge to 1.0 after many drifts, got {}",
        kernel.r_expected[0]
    );
}

#[test]
fn drift_with_empty_recent_leaves_w_unchanged() {
    // Default-impl LayerDirectionSource (no recent_direction override)
    // should leave w unchanged — only the EMA fires.
    type K = PersonalityWeightedComposition<1, 2>;
    let mut kernel = K::new(PersonalityConfig::default(), [0.42]);

    // A layer that DOESN'T override recent_direction.
    struct NoRecent {
        dir: [f32; 2],
    }
    impl LayerDirectionSource for NoRecent {
        fn direction(&self, scratch: &mut [f32]) -> &[f32] {
            scratch[..2].copy_from_slice(&self.dir);
            &scratch[..2]
        }
        // recent_direction falls back to default → empty slice
    }

    let l = NoRecent { dir: [1.0, 1.0] };
    let layers: [&dyn LayerDirectionSource; 1] = [&l];
    let w_before = kernel.w[0];
    kernel.drift(&layers, 1.0);
    assert!(
        (kernel.w[0] - w_before).abs() < 1e-9,
        "drift with empty recent should not move w"
    );
}

// ─── T3.x snapshot via public API ───────────────────────────────────────

#[test]
fn snapshot_round_trip_with_kernel() {
    // T3.2: snapshot → mutate → verify mismatch → restore → verify match.
    type K = PersonalityWeightedComposition<2, 3>;
    let mut kernel = K::new(PersonalityConfig::default(), [0.5, -0.5]);
    let snap = PersonalitySnapshot::<2>::from_composition(&kernel, ArchetypeLabel::new(11), 1);
    assert!(snap.verify_blake3());

    // Mutate.
    kernel.w[0] = 99.0;
    let mutated = PersonalitySnapshot::<2>::from_composition(&kernel, ArchetypeLabel::new(11), 1);
    assert_ne!(
        mutated.blake3, snap.blake3,
        "mutated w must produce a different blake3"
    );

    // Restore.
    kernel.restore_w(snap.w);
    let restored = PersonalitySnapshot::<2>::from_composition(&kernel, ArchetypeLabel::new(11), 1);
    assert_eq!(
        restored.blake3, snap.blake3,
        "restored w must match the original snapshot's blake3"
    );
}

#[test]
fn w_snapshot_returns_borrowed_array() {
    type K = PersonalityWeightedComposition<3, 4>;
    let kernel = K::new(PersonalityConfig::default(), [1.0, 2.0, 3.0]);
    let view = kernel.w_snapshot();
    // Type assertion: must be &[f32; 3], not &[f32].
    let _: &[f32; 3] = view;
    assert_eq!(view, &[1.0, 2.0, 3.0]);
}

// ─── Sigmoid helpers (re-exposed smoke check) ────────────────────────────

#[test]
fn module_sigmoid_into_matches_scalar() {
    let x = [0.0f32, 1.0, -1.0, 5.0, -5.0];
    let mut out = [0.0f32; 5];
    sigmoid_into(&x, &mut out);
    for (i, xi) in x.iter().enumerate() {
        assert!((out[i] - sigmoid(*xi)).abs() < 1e-7);
    }
}
