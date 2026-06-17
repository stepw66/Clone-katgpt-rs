//! Plan 284 GOAT Gate — G1–G5 benchmark stubs (Phase 4).
//!
//! These are `#[ignore]`-marked placeholder tests. The full criterion
//! benchmarks + synthetic suites are implemented in Phase 4 (deferred until
//! the GOAT gate is run). Each stub documents what the real test must assert.
//!
//! ## GOAT Criteria (Plan 284 §GOAT Criteria)
//!
//! - **G1**: CLR-vote ≥ +3pp over best-of-N majority on a synthetic suite.
//! - **G2**: Verifier sigmoid ECE ≤ 0.10.
//! - **G3**: ≤200µs/call at K=32, M=5, 8-dim direction vectors (target ≤50µs).
//! - **G4**: Zero heap allocation on the vote path (after `ClrScratch::new()`).
//! - **G5**: Feature isolation — compiles with/without `clr`, zero overhead when disabled.
//!
//! Run a single gate with: `cargo test --features clr --test bench_284_clr_goat -- g1 --ignored`
//! (once implemented).

#![cfg(feature = "clr")]

use katgpt_rs::clr::{
    allocate_budget, brevity_tiebreak, ClrConfig, ClrScratch, Cluster, Trajectory, learning_potential,
    mgpo_sampling_weight, should_write_memory,
};

// ─── G1: CLR beats best-of-N majority ───────────────────────────────

/// G1 — CLR-vote beats best-of-N majority by ≥3pp on a synthetic suite.
///
/// Synthetic suite: 50 trajectory-groups, each with 5 clusters of 10
/// trajectories. In each cluster, exactly 1 trajectory has a ground-truth
/// flawed claim (its `embedding[m_flaw]` is orthogonal to
/// `direction_vec[m_flaw]`, forcing `v < 0.5` for that claim).
///
/// Run `clr_vote` (K=50, M=5) vs best-of-N majority (pick cluster with most
/// members). Assert CLR picks the flawless cluster ≥3pp more often than
/// majority, over 100 random seeds (report mean + stddev).
#[test]
#[ignore = "Phase 4 — full criterion benchmark not yet implemented"]
fn g1_clr_beats_best_of_n_majority() {
    let _ = ClrConfig::default();
    // TODO: Phase 4 implements this with the synthetic suite described above.
}

// ─── G2: Calibration ECE ────────────────────────────────────────────

/// G2 — `SigmoidProjectionVerifier` Expected Calibration Error ≤ 0.10.
///
/// Ground-truth binary verdicts constructed so `v_k,m` is calibrated: random
/// `embedding` projections, true verdict is `Bernoulli(sigmoid(dot))`. Compute
/// ECE of `SigmoidProjectionVerifier::verify` outputs over 10K samples.
#[test]
#[ignore = "Phase 4 — calibration benchmark not yet implemented"]
fn g2_calibration_ece() {
    let _ = ClrConfig::default();
    // TODO: Phase 4 constructs a SigmoidProjectionVerifier + the calibration suite.
}

// ─── G3: Hot-path latency ───────────────────────────────────────────

/// G3 — `clr_vote_minimal()` ≤ 200µs/call at K=32, M=5, 8-dim direction
/// vectors. Stretch target ≤50µs.
///
/// This should be a `cargo bench` criterion group, not a `#[test]`. The stub
/// here is a placeholder; the real benchmark lives in a `benches/` target
/// (added in Phase 4).
#[test]
#[ignore = "Phase 4 — criterion latency benchmark not yet implemented"]
fn g3_hot_path_under_200us() {
    let _ = ClrConfig::default();
    // TODO: Phase 4 implements this as a criterion benchmark.
}

// ─── G4: Zero allocation ────────────────────────────────────────────

/// G4 — Zero heap allocation on the vote path after `ClrScratch::new()`.
///
/// Installs a counting global allocator, warms up `ClrScratch::new(32, 5)`
/// once, then calls `clr_vote_minimal()` 1000× and asserts 0 net allocations
/// (excluding the extractor's own allocations — those are caller-domain).
#[test]
#[ignore = "Phase 4 — allocation-counting benchmark not yet implemented"]
fn g4_zero_allocation() {
    let _ = ClrScratch::new(32, 5);
    // TODO: Phase 4 implements this with a counting allocator.
}

// ─── G5: Feature isolation ──────────────────────────────────────────

/// G5 — Feature isolation. This is verified at the *build* level, not runtime:
///
/// 1. `cargo build --no-default-features --features clr` compiles cleanly.
/// 2. `cargo build --no-default-features` (no `clr`) compiles cleanly and `clr`
///    symbols are absent from the binary.
/// 3. Zero overhead when disabled — no `clr` code paths reachable from the
///    default-features build.
///
/// The runtime stub here just confirms the `clr` surface is importable when
/// the feature is on.
#[test]
#[ignore = "Phase 4 — feature-isolation gate verified via build matrix, not runtime"]
fn g5_feature_isolation() {
    // When this test compiles, the `clr` feature surface is reachable.
    // The G5 gate itself is: `cargo build --no-default-features` must also
    // compile (verified separately, not via this test).
    let _ = ClrConfig::default();
}

// ─── Smoke test (not ignored) ───────────────────────────────────────

/// Smoke test — confirm the public CLR surface compiles + links when the
/// `clr` feature is on. Not part of G1–G5; just a build sanity check.
#[test]
fn clr_public_surface_links() {
    let _ = ClrConfig::default();
    let _ = ClrScratch::new(4, 2);

    // Touch the free functions to ensure they link + resolve.
    let _: fn(f32, f32) -> f32 = mgpo_sampling_weight;
    let _: fn(f32, f32, &ClrConfig) -> bool = should_write_memory;
    let _: fn(&[&Cluster<()>], &[Trajectory<()>], f32) -> usize = brevity_tiebreak::<()>;

    // allocate_budget / learning_potential are generic / take slices; just
    // confirm they're callable.
    let alloc = allocate_budget(&[1.0, 1.0], 10);
    assert_eq!(alloc.iter().sum::<usize>(), 10);
    let lp = learning_potential(2, |_| -1.0);
    assert!((lp - 1.0).abs() < 1e-6);
}
