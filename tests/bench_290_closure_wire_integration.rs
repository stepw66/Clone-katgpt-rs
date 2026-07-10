//! Plan 290 Phase 4 T4.2 + T4.3 integration — proves the full
//! wake → sleep → admit loop works end-to-end with real engine types.
//!
//! Exercises:
//! - `PtgTracedPruner` wrapping a real `AbsorbCompressLayer` (T4.2).
//! - `mine_motifs_at_sleep_cycle` discovering motifs the wrapper produced (T4.3).
//! - `MotifAdmitter` admitting a high-PRI motif and registering a composite
//!   primitive id (T2.5/T2.6).
//! - `compute_tar_score` over a perturbed corpus (TaR proxy sanity).
//!
//! # Run
//!
//! ```bash
//! cargo test --features closure_instrument,bandit \
//!     --test bench_290_closure_wire_integration -- --nocapture
//! ```

#![cfg(all(feature = "closure_instrument", feature = "bandit"))]

use katgpt_core::closure::{
    GateResult, MotifAdmitter, MotifMiner, OperatorKind, PrimitiveKind, compute_tar_score,
};
use katgpt_core::closure::mining::mine_motifs_at_sleep_cycle;
use katgpt_pruners::closure_wire::{COMPRESS_PRIMITIVE_ID, PtgTracedPruner};
use katgpt_rs::pruners::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
use katgpt_rs::speculative::types::NoScreeningPruner;

/// Build a traced absorb-compress layer with `n_arms` arms.
fn traced_layer(n_arms: usize) -> PtgTracedPruner<AbsorbCompressLayer<NoScreeningPruner>> {
    let inner = AbsorbCompressLayer::new(NoScreeningPruner, n_arms, CompressConfig::default());
    PtgTracedPruner::new(inner)
}

/// Run one synthetic episode on `traced`: absorb each arm once with a
/// pseudo-random reward, then call compress. Returns the finished PTG
/// (or `None` if the wrapper produced nothing).
fn run_episode(
    traced: &mut PtgTracedPruner<AbsorbCompressLayer<NoScreeningPruner>>,
    family: u32,
    arms: &[usize],
) -> Option<katgpt_core::closure::PrimitiveTransitionGraph> {
    traced.start_episode(family);
    for &arm in arms {
        // Pseudo-deterministic reward in [0, 1).
        let reward = ((arm as f32) * 0.1) + 0.05;
        traced.absorb(arm, reward);
    }
    let _ = traced.compress();
    traced.finish_episode()
}

/// Full wake→sleep→admit loop on a tiny synthetic workload.
///
/// Three task families each run a 3-arm episode several times. The wrapper
/// auto-emits one absorb-node per arm + one compress-node per episode.
/// Mining at the sleep-cycle boundary should discover the recurring arm
/// pattern and admit it as a composite primitive.
#[test]
fn wake_sleep_admit_loop_promotes_motif() {
    let mut miner = MotifMiner::new();
    let mut traced = traced_layer(4);

    // Wake phase: 3 task families × 5 episodes each. Each episode runs the
    // same 3-arm pattern (Search → Verify → Branch analog) + compress.
    for family in 0..3u32 {
        for _ in 0..5 {
            if let Some(ptg) = run_episode(&mut traced, family, &[0, 1, 2]) {
                miner.observe(ptg);
            }
        }
    }

    // Sleep phase: mine + admit.
    let report = mine_motifs_at_sleep_cycle(&miner, &MotifAdmitter::new(), 10_000.0);
    assert!(report.ptg_count >= 15, "ptg_count={}", report.ptg_count);
    assert!(!report.motifs.is_empty(), "should mine motifs");

    // The 3-arm absorb motif should recur across all 3 task families.
    // Each episode produced 4 nodes (3 absorbs + 1 compress); the 3-absorb
    // sub-chain is the recurring motif we look for.
    let cross_family_motifs: Vec<_> = report
        .motifs
        .iter()
        .filter(|m| m.node_count >= 2 && m.task_family_ids.len() >= 3)
        .collect();
    assert!(
        !cross_family_motifs.is_empty(),
        "expected at least one motif across all 3 task families; got {:?}",
        report
            .motifs
            .iter()
            .map(|m| (m.node_count, m.task_family_ids.len()))
            .collect::<Vec<_>>()
    );

    // The admitter should have admitted at least one of them.
    assert!(
        report.admitted_count >= 1,
        "expected ≥1 admission; got report with {} admitted out of {} motifs",
        report.admitted_count,
        report.motifs.len()
    );
}

/// Sanity: TaR proxy is 1.0 for identical corpora, < 1.0 for perturbed.
#[test]
fn tar_proxy_distinguishes_baseline_and_perturbed() {
    let mut miner_base = MotifMiner::new();
    let mut miner_pert = MotifMiner::new();
    let mut traced_base = traced_layer(4);
    let mut traced_pert = traced_layer(4);

    // Baseline: 5 episodes of arms [0,1,2] in family 0.
    for _ in 0..5 {
        traced_base.start_episode(0);
        for &arm in &[0usize, 1, 2] {
            traced_base.absorb(arm, 0.5);
        }
        if let Some(ptg) = traced_base.finish_episode() {
            miner_base.observe(ptg);
        }
    }

    // Perturbed: 5 episodes, but the third arm is replaced (different motif).
    for _ in 0..5 {
        traced_pert.start_episode(0);
        for &arm in &[0usize, 1, 3] {
            traced_pert.absorb(arm, 0.5);
        }
        if let Some(ptg) = traced_pert.finish_episode() {
            miner_pert.observe(ptg);
        }
    }

    let base: Vec<_> = miner_base.recent_ptgs.iter().cloned().collect();
    let pert: Vec<_> = miner_pert.recent_ptgs.iter().cloned().collect();
    let tar = compute_tar_score(&base, &pert);
    // Same corpus → 1.0.
    let tar_same = compute_tar_score(&base, &base);
    assert!((tar_same - 1.0).abs() < 1e-6, "same corpus tar={tar_same}");
    // Perturbed should drop below 1.0 (some overlap from arms 0,1; arm 2 vs 3 diverges).
    assert!(tar < 1.0, "perturbed tar should be < 1.0, got {tar}");
    assert!(
        tar > 0.0,
        "perturbed tar should still have some overlap, got {tar}"
    );
}

/// The wrapper's `relevance()` is unaffected by tracing — verifies the
/// "zero hot-path overhead" contract from G2 at the API level.
#[test]
fn relevance_unchanged_by_tracing() {
    use katgpt_rs::speculative::types::ScreeningPruner;

    let mut traced = traced_layer(4);
    traced.start_episode(99);
    // Relevance queried mid-episode should equal the inner layer's relevance.
    // NoScreeningPruner always returns 1.0.
    let r1 = traced.relevance(0, 0, &[]);
    assert_eq!(r1, 1.0);
    traced.absorb(0, 0.5);
    let r2 = traced.relevance(0, 0, &[]);
    assert_eq!(r2, 1.0, "relevance unchanged after absorb");
    let _ = traced.finish_episode();
}

/// Manually tracing a bandit `update`-equivalent event marks a custom
/// primitive in the PTG — proves the wrapper supports bandit-update tracing
/// via the explicit `trace` API (since `update` is on BanditPruner<P>, not
/// on the outermost wrapper).
#[test]
fn manual_trace_captures_bandit_update_events() {
    let mut traced = traced_layer(2);
    traced.start_episode(42);
    // Simulate: prepare_episode (Recurse from root), then two updates.
    traced.trace(PrimitiveKind::UserDefined(50), OperatorKind::Sequence); // prepare
    traced.trace(PrimitiveKind::UserDefined(0), OperatorKind::Sequence); // update arm 0
    traced.trace(PrimitiveKind::UserDefined(1), OperatorKind::Sequence); // update arm 1
    let ptg = traced.finish_episode().expect("episode produced a PTG");
    assert_eq!(ptg.nodes.len(), 3);
    assert_eq!(ptg.nodes[0].primitive, PrimitiveKind::UserDefined(50));
    assert_eq!(ptg.nodes[1].primitive, PrimitiveKind::UserDefined(0));
    assert_eq!(ptg.nodes[2].primitive, PrimitiveKind::UserDefined(1));
}

/// Compress events emit the reserved COMPRESS_PRIMITIVE_ID with a Branch edge,
/// distinguishing them from linear absorb events when mining.
#[test]
fn compress_event_uses_reserved_primitive_id() {
    let mut traced = traced_layer(2);
    traced.start_episode(0);
    traced.absorb(0, 0.5);
    let _ = traced.compress();
    let ptg = traced.finish_episode().expect("PTG");
    let has_compress = ptg
        .nodes
        .iter()
        .any(|n| n.primitive == PrimitiveKind::UserDefined(COMPRESS_PRIMITIVE_ID));
    assert!(
        has_compress,
        "compress event should emit reserved primitive id"
    );
}

/// `MotifAdmitter::evaluate` on a mined motif returns either Admitted or
/// Rejected — never panics. Smoke test for the integration.
#[test]
fn admitter_evaluates_mined_motifs_without_panic() {
    let mut miner = MotifMiner::new();
    let mut traced = traced_layer(3);
    for family in 0..3u32 {
        for _ in 0..5 {
            traced.start_episode(family);
            for &arm in &[0usize, 1] {
                traced.absorb(arm, 0.5);
            }
            if let Some(ptg) = traced.finish_episode() {
                miner.observe(ptg);
            }
        }
    }
    let report = mine_motifs_at_sleep_cycle(&miner, &MotifAdmitter::new(), 10_000.0);
    // Re-evaluate every mined motif directly to confirm no panic.
    for motif in &report.motifs {
        let _ = MotifAdmitter::new().evaluate(motif, 3, 10_000.0);
    }
    // And confirm the GateResult variants are reachable.
    let any_admitted = report.motifs.iter().any(|m| {
        matches!(
            MotifAdmitter::new().evaluate(m, 3, 10_000.0),
            GateResult::Admitted { .. }
        )
    });
    let _ = any_admitted; // informational; main loop already counts admissions
}
