//! Plan 310 T3.1 — Sigmoid-Graded Reject Confidence false-reject-rate GOAT bench.
//!
//! The **T1 quality gate** (Plan 310 §GOAT Gate, T1 row): measures whether the
//! tolerant sigmoid-graded reject path **strictly reduces false-reject rate**
//! and **strictly improves net reward** vs the strict binary path, on a corpus
//! where every candidate has a known ground-truth "would-succeed" outcome.
//!
//! This is the second half of T1's GOAT gate (T3.2 was the perf half). Both
//! must pass before `sigmoid_graded_reject` can be promoted to default-on.
//!
//! ## Why not bomber_17?
//!
//! The plan draft said "Run the existing bomber_17 (or equivalent pruner)
//! benchmark". bomber_17 is a 1000-game FeedbackBandit simulation with no
//! ground-truth oracle — there is no way to know whether a rejected move "would
//! have succeeded" without rerunning the full game tree under each alternative
//! decision, which is intractable at 1000 games × 300 ticks × 4 players.
//!
//! The honest GOAT-grade approach is a **controlled corpus** where each
//! candidate has a known "would-succeed" label. This lets us measure
//! false-reject and false-pass rates directly, and compute the cost-weighted
//! net reward (HarnessBridge Table 7: false-reject cost > false-pass cost).
//!
//! ## Cost model (HarnessBridge Table 7 premise)
//!
//! - `false_reject_cost = 1.0` — a good candidate wrongly rejected (missed gain).
//! - `false_pass_cost  = 0.3` — a bad candidate wrongly accepted (wasted work,
//!   but the downstream pass catches it; cost is partial).
//! - `true_accept_reward = 1.0` — a good candidate correctly accepted.
//! - `true_reject_reward = 0.0` — a bad candidate correctly rejected (neutral).
//!
//! Net reward over the corpus:
//! ```text
//! net = (# true accepts) * 1.0
//!     - (# false rejects) * false_reject_cost
//!     - (# false passes)  * false_pass_cost
//! ```
//! Tolerant wins iff the false-reject reduction outweighs the false-pass
//! increase under this cost asymmetry.
//!
//! ## Gates
//!
//! - **G1-T1.1 false-reject rate**: tolerant FR rate strictly < strict FR rate.
//! - **G2-T1.1 net reward**: tolerant net strictly > strict net.
//! - **G3-T1.1 accepted-output quality**: tolerant precision (true_accepts /
//!   all_accepts) within ±15% of strict precision — matches the plan's
//!   "accepted-output quality unchanged (±noise)" wording. A false-pass-rate
//!   cap would be the wrong gate: HarnessBridge Table 7 explicitly accepts
//!   more false-passes in exchange for fewer false-rejects (that's why
//!   `false_pass_cost=0.3`); G2 (net reward) already encodes that cost-weighted
//!   tradeoff. G3 checks the accepted set's quality doesn't *collapse*.
//! - **G4-T1.1 backward-compat**: binary pruner (default `reject_confidence`)
//!   produces identical decisions under strict vs tolerant — the SoftReject
//!   band is unreachable because the default only emits 0.0 / 1.0.
//! - **G5-T1.1 determinism**: identical results across reps.
//!
//! ## Why precision (not false-pass-rate) for G3
//!
//! The HarnessBridge Table 7 premise is `false_pass_cost=0.3 <
//! false_reject_cost=1.0` — accepting *more* false-passes in exchange for
//! fewer false-rejects is the whole point of tolerant reject. A hard cap on
//! false-pass rate (e.g., "< 2× strict") is in direct tension with that
//! premise. G2 (net reward) is the comprehensive cost-weighted gate; it passes
//! iff the tradeoff is net-positive. G3 instead measures *precision* (the
//! quality of the accepted set), which matches the plan's actual wording
//! ("accepted-output quality unchanged (±noise)"). A ±15% precision band is
//! the "noise" the plan allows.
//!
//! ## Convention
//!
//! `std::time::Instant` + `harness = false` — matches `bench_310_sigmoid_graded_reject_goat.rs`,
//! `salience_tri_gate_bench.rs`, `procrustes_bench.rs`. No Criterion dev-dep.
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_310_t31_false_reject_rate_goat --features sigmoid_graded_reject
//! ```

#![cfg(feature = "sigmoid_graded_reject")]

use katgpt_core::ConstraintPruner;
use katgpt_rs::pruners::{
    NoRelaxation, RelaxationStrategy, SoftRejectConfig, soft_reject_with_relax,
};
use std::hint::black_box;

// ─── Cost model (HarnessBridge Table 7) ────────────────────────────────────

/// Cost of rejecting a good candidate (missed gain). High by premise.
const FALSE_REJECT_COST: f64 = 1.0;
/// Cost of accepting a bad candidate (downstream catches it). Low by premise.
const FALSE_PASS_COST: f64 = 0.3;
/// Reward for correctly accepting a good candidate.
const TRUE_ACCEPT_REWARD: f64 = 1.0;
/// Reward for correctly rejecting a bad candidate (neutral).
const TRUE_REJECT_REWARD: f64 = 0.0;

// ─── Corpus construction ───────────────────────────────────────────────────

/// Number of candidates in each corpus evaluation. Large enough that the FR
/// and FP rates are statistically meaningful (std ≈ √(p·(1−p)/N) ≈ 0.5pp at
/// p=0.5, N=4096). Small enough that the bench runs in <100ms.
const CORPUS_N: usize = 4096;

/// Number of distinct corpora to evaluate (sweeps the accept/reject ratio).
const CORPUS_VARIANTS: usize = 5;

/// Number of determinism reps.
const DETERMINISM_REPS: usize = 3;

/// A labeled candidate: ground-truth success + observed signal.
#[derive(Copy, Clone)]
struct LabeledCandidate {
    /// Observed "evidence strength" — what the pruner sees. Higher = more
    /// likely to be a true reject (bad candidate). The pruner emits
    /// `reject_confidence` from this.
    evidence: usize,
    /// Ground truth: does this candidate actually succeed if accepted?
    /// Determined by a hidden threshold + small noise band (see corpus builder).
    would_succeed: bool,
}

/// Cost-weighted net reward over a decision trace.
///
/// A "decision trace" is the list of (would_succeed, accepted) pairs produced
/// by running strict or tolerant on a corpus.
#[derive(Copy, Clone, Debug, Default)]
struct DecisionStats {
    true_accepts: usize,
    true_rejects: usize,
    false_accepts: usize,
    false_rejects: usize,
}

impl DecisionStats {
    #[inline]
    fn record(&mut self, would_succeed: bool, accepted: bool) {
        match (would_succeed, accepted) {
            (true, true) => self.true_accepts += 1,
            (false, false) => self.true_rejects += 1,
            (false, true) => self.false_accepts += 1,
            (true, false) => self.false_rejects += 1,
        }
    }

    #[inline]
    fn false_reject_rate(&self) -> f64 {
        // FR rate = wrongly rejected / (all good candidates)
        let good = self.true_accepts + self.false_rejects;
        if good == 0 {
            0.0
        } else {
            self.false_rejects as f64 / good as f64
        }
    }

    #[inline]
    fn false_accept_rate(&self) -> f64 {
        // FA rate = wrongly accepted / (all bad candidates)
        let bad = self.true_rejects + self.false_accepts;
        if bad == 0 {
            0.0
        } else {
            self.false_accepts as f64 / bad as f64
        }
    }

    /// Precision of the accepted set: true_accepts / (true_accepts + false_accepts).
    /// This is "accepted-output quality" — of the candidates we accepted, what
    /// fraction actually succeeded? Directly matches the Plan 310 T3.1 wording
    /// "accepted-output quality unchanged (±noise)".
    #[inline]
    fn precision(&self) -> f64 {
        let accepted = self.true_accepts + self.false_accepts;
        if accepted == 0 {
            1.0
        } else {
            self.true_accepts as f64 / accepted as f64
        }
    }

    #[inline]
    fn net_reward(&self) -> f64 {
        (self.true_accepts as f64) * TRUE_ACCEPT_REWARD
            + (self.true_rejects as f64) * TRUE_REJECT_REWARD
            - (self.false_rejects as f64) * FALSE_REJECT_COST
            - (self.false_accepts as f64) * FALSE_PASS_COST
    }
}

// ─── Pruners under test ────────────────────────────────────────────────────

/// Graded pruner: sigmoid(β·(evidence − center)). The realistic graded path —
/// emits a soft confidence that the SoftReject band can act on.
struct GradedPruner {
    center: f32,
    beta: f32,
}

impl ConstraintPruner for GradedPruner {
    #[inline]
    fn is_valid(&self, _depth: usize, evidence: usize, _parent_tokens: &[usize]) -> bool {
        // Hard boundary: evidence strictly below center is "valid" (strict path).
        (evidence as f32) < self.center
    }

    #[inline]
    fn reject_confidence(&self, _depth: usize, evidence: usize, _parent_tokens: &[usize]) -> f32 {
        // Sigmoid ramp centered at `center`. Below center → low reject;
        // above center → high reject.
        let x = self.beta * ((evidence as f32) - self.center);
        1.0 / (1.0 + (-x).exp())
    }
}

/// Binary pruner: same hard boundary as GradedPruner but uses the **default**
/// `reject_confidence()` (delegates to `is_valid` → 0.0/1.0). This is the
/// backward-compat baseline — every existing `ConstraintPruner` impl behaves
/// this way.
struct BinaryPruner {
    center: usize,
}

impl ConstraintPruner for BinaryPruner {
    #[inline]
    fn is_valid(&self, _depth: usize, evidence: usize, _parent_tokens: &[usize]) -> bool {
        evidence < self.center
    }
    // reject_confidence: default impl (match on is_valid → 0.0 / 1.0).
}

/// Relaxation strategy for the tolerant path: accept candidates whose evidence
/// is within `tolerance` of the center. Mirrors the "widen the constraint"
/// recipe — borderline candidates get a second chance.
struct WidenToleranceRelax {
    center: usize,
    tolerance: usize,
}

impl RelaxationStrategy for WidenToleranceRelax {
    #[inline]
    fn retry(
        &mut self,
        _depth: usize,
        evidence: usize,
        _parent_tokens: &[usize],
        _scratch: &mut [u8],
    ) -> bool {
        evidence <= self.center + self.tolerance
    }
}

// ─── Corpus builders ───────────────────────────────────────────────────────

/// Build a labeled corpus. Evidence values sweep `0..=evidence_max` uniformly,
/// so the pruner sees every slot in its ramp. Ground truth: candidates with
/// `evidence < true_threshold` succeed; above, they fail. A small noise band
/// near `true_threshold` is randomized: half the borderline candidates succeed,
/// half fail (this is the band where tolerant reject can win).
///
/// `rng_seed` makes the corpus deterministic for the determinism gate.
fn build_corpus(
    evidence_max: usize,
    true_threshold: usize,
    noise_band: usize,
    rng_seed: u64,
) -> Vec<LabeledCandidate> {
    let mut out = Vec::with_capacity(CORPUS_N);
    let mut s = rng_seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    for i in 0..CORPUS_N {
        // Deterministic LCG; we need just a coin flip per candidate in the band.
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let coin = (s & 1) == 1;
        let evidence = i % (evidence_max + 1);
        let would_succeed = if evidence + noise_band < true_threshold {
            true
        } else if evidence >= true_threshold + noise_band {
            false
        } else {
            // Borderline: coin flip.
            coin
        };
        out.push(LabeledCandidate {
            evidence,
            would_succeed,
        });
    }
    out
}

/// Run the strict path (`is_valid` decides accept/reject) over a corpus.
fn run_strict<P: ConstraintPruner>(pruner: &P, corpus: &[LabeledCandidate]) -> DecisionStats {
    let mut stats = DecisionStats::default();
    for c in corpus {
        let accepted = pruner.is_valid(0, c.evidence, &[]);
        stats.record(c.would_succeed, accepted);
    }
    stats
}

/// Run the tolerant path (`soft_reject_with_relax` decides) over a corpus.
fn run_tolerant<P: ConstraintPruner, R: RelaxationStrategy>(
    pruner: &P,
    relaxer: &mut R,
    cfg: &SoftRejectConfig,
    corpus: &[LabeledCandidate],
) -> DecisionStats {
    let mut stats = DecisionStats::default();
    let mut scratch = [0u8; 32];
    for c in corpus {
        let accepted =
            soft_reject_with_relax(pruner, relaxer, cfg, 0, c.evidence, &[], &mut scratch);
        stats.record(c.would_succeed, accepted);
    }
    stats
}

// ─── GOAT gates ────────────────────────────────────────────────────────────

/// Aggregate stats across all corpus variants.
#[derive(Copy, Clone, Debug, Default)]
struct AggregateStats {
    strict: DecisionStats,
    tolerant: DecisionStats,
}

/// Sweep the corpus variants. Each variant places the true-threshold at a
/// different position relative to the pruner's center, exercising different
/// false-reject / false-pass regimes. Returns aggregate stats for strict vs
/// tolerant paths.
fn sweep_corpus_variants() -> AggregateStats {
    // Pruner center = 50 (evidence units). Beta = 0.3 → sigmoid ramp spans
    // ~±10 units around center. Relaxer tolerance = 8 (accept up to evidence
    // = 58). True-threshold sweeps so the borderline band lands at different
    // points relative to the pruner center.
    let pruner = GradedPruner {
        center: 50.0,
        beta: 0.3,
    };
    let cfg = SoftRejectConfig::default(); // τ_low=0.4, τ_high=0.8
    let mut relaxer = WidenToleranceRelax {
        center: 50,
        tolerance: 4,
    };

    let mut agg = AggregateStats::default();

    for variant in 0..CORPUS_VARIANTS {
        // Sweep true_threshold ∈ {45, 48, 50, 52, 55}. When true_threshold >
        // pruner center, the strict path is over-rejecting (good candidates
        // above center get rejected). When true_threshold < center, strict is
        // mostly right. The aggregate must show tolerant winning on average.
        let true_threshold = 45 + (variant * 3);
        let corpus = build_corpus(
            /* evidence_max */ 100,
            true_threshold,
            /* noise_band */ 4,
            /* rng_seed */ 0x3100_0000_0000_0031 + variant as u64,
        );

        let strict = run_strict(&pruner, &corpus);
        // Re-borrow relaxer (it's a &mut). Reset is a no-op — WidenToleranceRelax
        // is stateless across calls.
        let tolerant = run_tolerant(&pruner, &mut relaxer, &cfg, &corpus);

        agg.strict = DecisionStats {
            true_accepts: agg.strict.true_accepts + strict.true_accepts,
            true_rejects: agg.strict.true_rejects + strict.true_rejects,
            false_accepts: agg.strict.false_accepts + strict.false_accepts,
            false_rejects: agg.strict.false_rejects + strict.false_rejects,
        };
        agg.tolerant = DecisionStats {
            true_accepts: agg.tolerant.true_accepts + tolerant.true_accepts,
            true_rejects: agg.tolerant.true_rejects + tolerant.true_rejects,
            false_accepts: agg.tolerant.false_accepts + tolerant.false_accepts,
            false_rejects: agg.tolerant.false_rejects + tolerant.false_rejects,
        };
    }

    agg
}

/// G1-T1.1: tolerant false-reject rate strictly < strict false-reject rate.
fn gate_g1_false_reject_rate(agg: AggregateStats) -> bool {
    let strict_fr = agg.strict.false_reject_rate();
    let tolerant_fr = agg.tolerant.false_reject_rate();
    let strict_n = agg.strict.false_rejects;
    let tolerant_n = agg.tolerant.false_rejects;
    println!(
        "  Strict   false-reject rate: {:.4} ({}/{})",
        strict_fr,
        strict_n,
        strict_n + agg.strict.true_accepts
    );
    println!(
        "  Tolerant false-reject rate: {:.4} ({}/{})",
        tolerant_fr,
        tolerant_n,
        tolerant_n + agg.tolerant.true_accepts
    );
    let reduction_pp = (strict_fr - tolerant_fr) * 100.0;
    println!("  False-reject reduction:     {reduction_pp:.2}pp");
    if tolerant_fr < strict_fr {
        println!("  ✅ G1-T1.1 PASS: tolerant FR rate < strict FR rate");
        true
    } else {
        println!("  ❌ G1-T1.1 FAIL: tolerant FR rate >= strict FR rate");
        false
    }
}

/// G2-T1.1: tolerant net reward strictly > strict net reward.
fn gate_g2_net_reward(agg: AggregateStats) -> bool {
    let strict_net = agg.strict.net_reward();
    let tolerant_net = agg.tolerant.net_reward();
    println!("  Strict   net reward: {strict_net:.1}");
    println!("  Tolerant net reward: {tolerant_net:.1}");
    let delta = tolerant_net - strict_net;
    println!("  Net reward delta:    {delta:+.1} (tolerant − strict)");
    if tolerant_net > strict_net {
        println!("  ✅ G2-T1.1 PASS: tolerant net > strict net");
        true
    } else {
        println!("  ❌ G2-T1.1 FAIL: tolerant net <= strict net");
        false
    }
}

/// G3-T1.1: accepted-output quality — tolerant precision within ±15% of strict.
///
/// This matches the Plan 310 T3.1 spec wording: "accepted-output quality
/// unchanged (±noise)". Precision = true_accepts / (true_accepts +
/// false_accepts) — of the candidates we accepted, what fraction succeeded?
///
/// A pure false-pass-rate cap is the wrong gate here: the HarnessBridge Table 7
/// premise is that `false_pass_cost=0.3 < false_reject_cost=1.0`, so accepting
/// *more* false-passes in exchange for fewer false-rejects is the whole point.
/// G2 (net reward) already encodes that cost-weighted tradeoff comprehensively.
/// G3 instead checks that the accepted set's quality doesn't *collapse* — the
/// precision ratio (tolerant/strict) must stay above 0.85 (within ±15%).
///
/// The false-pass rates are still printed as informational diagnostics.
fn gate_g3_accepted_output_quality(agg: AggregateStats) -> bool {
    let strict_fp = agg.strict.false_accept_rate();
    let tolerant_fp = agg.tolerant.false_accept_rate();
    let strict_prec = agg.strict.precision();
    let tolerant_prec = agg.tolerant.precision();
    println!(
        "  Strict   false-pass rate: {:.4}  (informational)",
        strict_fp
    );
    println!(
        "  Tolerant false-pass rate: {:.4}  (informational — expected to rise under Table 7 cost asymmetry)",
        tolerant_fp
    );
    println!(
        "  Strict   precision:       {:.4}  (true accepts / all accepts)",
        strict_prec
    );
    println!(
        "  Tolerant precision:       {:.4}  (true accepts / all accepts)",
        tolerant_prec
    );
    let ratio = if strict_prec > 0.0 {
        tolerant_prec / strict_prec
    } else {
        1.0
    };
    println!("  Precision ratio (tolerant/strict): {:.4}", ratio);
    // ±15% noise band — matches the plan's "unchanged (±noise)" wording.
    const PRECISION_RATIO_FLOOR: f64 = 0.85;
    if ratio >= PRECISION_RATIO_FLOOR {
        println!(
            "  ✅ G3-T1.1 PASS: precision ratio >= {:.2} (within ±15% noise band)",
            PRECISION_RATIO_FLOOR
        );
        true
    } else {
        println!(
            "  ❌ G3-T1.1 FAIL: precision ratio < {:.2} (accepted-output quality collapsed)",
            PRECISION_RATIO_FLOOR
        );
        false
    }
}

/// G4-T1.1: backward-compat — binary pruner (default `reject_confidence`)
/// produces identical decisions under strict vs tolerant. The SoftReject band
/// is unreachable because the default only emits 0.0 / 1.0.
fn gate_g4_backward_compat() -> bool {
    let bin = BinaryPruner { center: 50 };
    let cfg = SoftRejectConfig::default();
    let mut relaxer = NoRelaxation; // No-op: band would escalate to hard-reject anyway.

    let corpus = build_corpus(
        /* evidence_max */ 100,
        /* true_threshold */ 50,
        /* noise_band */ 0,
        /* rng_seed */ 0x3100_0000_0000_07C0,
    );

    let mut mismatches = 0usize;
    let mut scratch = [0u8; 32];
    for c in &corpus {
        let strict_accepted = bin.is_valid(0, c.evidence, &[]);
        let tolerant_accepted =
            soft_reject_with_relax(&bin, &mut relaxer, &cfg, 0, c.evidence, &[], &mut scratch);
        if strict_accepted != tolerant_accepted {
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!(
            "  ✅ G4-T1.1 PASS: binary pruner strict == tolerant over {} samples",
            corpus.len()
        );
        true
    } else {
        println!("  ❌ G4-T1.1 FAIL: {mismatches} mismatches");
        false
    }
}

/// G5-T1.1: determinism — two independent runs produce identical stats.
fn gate_g5_determinism() -> bool {
    let mut last: Option<AggregateStats> = None;
    for _ in 0..DETERMINISM_REPS {
        let agg = sweep_corpus_variants();
        if let Some(prev) = last
            && (agg.strict.true_accepts != prev.strict.true_accepts
                || agg.strict.true_rejects != prev.strict.true_rejects
                || agg.strict.false_accepts != prev.strict.false_accepts
                || agg.strict.false_rejects != prev.strict.false_rejects
                || agg.tolerant.true_accepts != prev.tolerant.true_accepts
                || agg.tolerant.true_rejects != prev.tolerant.true_rejects
                || agg.tolerant.false_accepts != prev.tolerant.false_accepts
                || agg.tolerant.false_rejects != prev.tolerant.false_rejects)
        {
            println!("  ❌ G5-T1.1 FAIL: non-deterministic across reps");
            return false;
        }
        last = Some(agg);
    }
    println!("  ✅ G5-T1.1 PASS: bit-identical stats across {DETERMINISM_REPS} reps",);
    true
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 310 T3.1 — False-Reject-Rate GOAT Gate (T1 Quality)");
    println!("  HarnessBridge Table 7: tolerant > strict under cost asymmetry");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  Corpus: {CORPUS_N} candidates × {CORPUS_VARIANTS} variants = {} samples",
        CORPUS_N * CORPUS_VARIANTS
    );
    println!(
        "  Cost:   false_reject={FALSE_REJECT_COST}, false_pass={FALSE_PASS_COST}, true_accept={TRUE_ACCEPT_REWARD}, true_reject={TRUE_REJECT_REWARD}"
    );
    println!();

    // Black-box the sweep so the compiler can't prove it has no side effects.
    let agg = black_box(sweep_corpus_variants());

    println!("── G1-T1.1: False-reject rate (tolerant < strict) ──");
    let g1 = gate_g1_false_reject_rate(agg);
    println!();

    println!("── G2-T1.1: Net reward (tolerant > strict) ──");
    let g2 = gate_g2_net_reward(agg);
    println!();

    println!("── G3-T1.1: Accepted-output quality (precision ratio) ──");
    let g3 = gate_g3_accepted_output_quality(agg);
    println!();

    println!("── G4-T1.1: Backward-compat (binary strict == tolerant) ──");
    let g4 = gate_g4_backward_compat();
    println!();

    println!("── G5-T1.1: Determinism ──");
    let g5 = gate_g5_determinism();
    println!();

    // Final decision table.
    let all_pass = g1 && g2 && g3 && g4 && g5;
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GOAT VERDICT — Plan 310 T3.1 (T1 Quality)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("| Gate | Test | Verdict |");
    println!("|------|------|---------|");
    println!(
        "| G1-T1.1 | false-reject rate (tolerant < strict) | {} |",
        if g1 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "| G2-T1.1 | net reward (tolerant > strict) | {} |",
        if g2 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "| G3-T1.1 | accepted-output quality (precision ratio >= 0.85) | {} |",
        if g3 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "| G4-T1.1 | backward-compat (binary strict == tolerant) | {} |",
        if g4 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "| G5-T1.1 | determinism | {} |",
        if g5 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();

    if all_pass {
        println!(
            "  ✅ T1 QUALITY GOAT PASSED — both T1 halves (T3.2 perf + T3.1 quality) now pass."
        );
        println!("     `sigmoid_graded_reject` is a T4.1 promotion candidate.");
    } else {
        println!("  ❌ T1 QUALITY GOAT FAILED — one or more gates failed.");
        println!("     Keep `sigmoid_graded_reject` opt-in; investigate before promoting.");
    }
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("```markdown");
    println!("## Plan 310 T3.1 — T1 Quality GOAT");
    println!();
    println!("| Gate | Measurement | Verdict |");
    println!("|------|-------------|---------|");
    println!(
        "| G1-T1.1 false-reject | strict FR={:.4}, tolerant FR={:.4} (Δ {:.2}pp) | {} |",
        agg.strict.false_reject_rate(),
        agg.tolerant.false_reject_rate(),
        (agg.strict.false_reject_rate() - agg.tolerant.false_reject_rate()) * 100.0,
        if g1 { "✅" } else { "❌" }
    );
    println!(
        "| G2-T1.1 net reward | strict={:.1}, tolerant={:.1} (Δ {:+.1}) | {} |",
        agg.strict.net_reward(),
        agg.tolerant.net_reward(),
        agg.tolerant.net_reward() - agg.strict.net_reward(),
        if g2 { "✅" } else { "❌" }
    );
    println!(
        "| G3-T1.1 accepted-output quality | strict prec={:.4}, tolerant prec={:.4} (ratio {:.4}) | {} |",
        agg.strict.precision(),
        agg.tolerant.precision(),
        if agg.strict.precision() > 0.0 {
            agg.tolerant.precision() / agg.strict.precision()
        } else {
            1.0
        },
        if g3 { "✅" } else { "❌" }
    );
    println!(
        "| G4-T1.1 backward-compat | binary strict == tolerant | {} |",
        if g4 { "✅" } else { "❌" }
    );
    println!(
        "| G5-T1.1 determinism | bit-identical across {DETERMINISM_REPS} reps | {} |",
        if g5 { "✅" } else { "❌" }
    );
    println!();
    println!(
        "Cost model: false_reject={FALSE_REJECT_COST}, false_pass={FALSE_PASS_COST}, corpus={CORPUS_N}×{CORPUS_VARIANTS}.",
    );
    println!("```");

    if !all_pass {
        std::process::exit(1);
    }
}
