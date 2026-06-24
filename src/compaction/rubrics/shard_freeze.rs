//! Phase 5 — the **shard-freeze rubric** (P0/P1), the cross-domain
//! isomorphism target for G7.
//!
//! # The isomorphism (G7 — the Super-GOAT claim)
//!
//! `riir-neuron-db::ConsolidationPipeline::can_freeze` is structurally
//! isomorphic to a CUCG `ClosedUnitCompactionGate<ShardFreezeRubric, 2>`:
//!
//! | `can_freeze` (riir-neuron-db `phase_gate.rs`) | CUCG `ShardFreezeRubric` |
//! |-----------------------------------------------|--------------------------|
//! | `input_sufficient = n_wake_events >= intrinsic_dim` | P0 (predicate 0): Yes iff `n >= d` |
//! | `output_converged = spectral_flatness < 0.3`       | P1 (predicate 1): Yes iff `flatness < 0.3` |
//! | `can_freeze = input_sufficient && output_converged` | Fire rule `And(0b0011)` = `P0 ∧ P1` |
//!
//! The two gates produce **the same decision** on the same inputs because
//! they are the same Boolean formula. This is not a coincidence — it is the
//! recognition (Research 007 cross-ref) that *trajectory compaction* (decide
//! when to summarize an agent's context) and *shard consolidation freeze*
//! (decide when a neuron shard's weights are stable enough to commit) are
//! instances of one primitive: a rubric-gated structural-safety check.
//!
//! # Why decoupled (not a cross-repo dependency)
//!
//! `katgpt-rs` does not depend on `riir-neuron-db`. The isomorphism is proved
//! **structurally** (same thresholds, same Boolean formula) rather than by
//! runtime cross-crate calls. The G7 test exhaustively verifies all 4
//! combinations of (P0, P1) and confirms the CUCG decision matches the
//! `can_freeze` formula. This keeps the open primitive (`katgpt-rs`) free
//! of private-runtime (`riir-neuron-db`) coupling — per AGENTS.md
//! SOLID/DRY/Decouple and the 5-repo commercial strategy.
//!
//! # Predicates
//!
//! | Idx | Paper-equivalent | Latent feature | Gate | "Yes" means |
//! |-----|------------------|----------------|------|-------------|
//! | 0   | P0 input-sufficient | `n_wake_events`, `intrinsic_dim` | `n >= d`               | enough samples to recover the subspace (Wang et al. Thm 4) |
//! | 1   | P1 output-converged | `spectral_flatness`              | `flatness < τ` (τ=0.3) | weights have a single dominant mode (converged) |
//!
//! Fire rule: [`FireRule::shard_freeze_rule_2`](super::super::fire_rule::FireRule::shard_freeze_rule_2)
//! = `And(0b0011)` = `P0 ∧ P1`.

use super::super::rubric::{
    PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict,
};

/// The output-convergence threshold, mirroring
/// `riir-neuron-db::phase_gate::FREEZE_FLATNESS_THRESHOLD`. A shard's
/// `style_weights` are "converged" (single dominant mode) when
/// `spectral_flatness < SHARD_FREEZE_FLATNESS_THRESHOLD`.
///
/// **This constant MUST stay in sync with the riir-neuron-db value.** The G7
/// isomorphism breaks if the two thresholds diverge.
pub const SHARD_FREEZE_FLATNESS_THRESHOLD: f32 = 0.3;

/// Caller-supplied scalar features for the [`ShardFreezeRubric`].
///
/// These mirror the inputs to
/// `riir-neuron-db::ConsolidationPipeline::can_freeze`:
/// - `n_wake_events` — wake-event count (raw or compressed post-sleep)
/// - `intrinsic_dim` — participation ratio / numerical rank of `style_weights`
/// - `spectral_flatness` — spectral flatness of `style_weights`
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShardFreezeFeatures {
    /// P0 source — number of wake events feeding consolidation.
    pub n_wake_events: usize,
    /// P0 source — intrinsic dimension `d` of `style_weights` (Wang et al.
    /// Thm 4: subspace recovery needs `n >= d` samples).
    pub intrinsic_dim: usize,
    /// P1 source — spectral flatness of `style_weights`. Lower = more
    /// converged (single dominant mode).
    pub spectral_flatness: f32,
}

impl ShardFreezeFeatures {
    /// Construct a feature triple.
    #[inline]
    #[must_use]
    pub const fn new(n_wake_events: usize, intrinsic_dim: usize, spectral_flatness: f32) -> Self {
        Self {
            n_wake_events,
            intrinsic_dim,
            spectral_flatness,
        }
    }
}

/// Configuration for the [`ShardFreezeRubric`]. The flatness threshold
/// defaults to [`SHARD_FREEZE_FLATNESS_THRESHOLD`] (0.3), mirroring
/// `riir-neuron-db`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShardFreezeRubricConfig {
    /// P1 threshold: `spectral_flatness < flatness_threshold` → converged.
    /// Defaults to 0.3 (the riir-neuron-db value).
    pub flatness_threshold: f32,
}

impl ShardFreezeRubricConfig {
    /// Default config mirroring riir-neuron-db's `can_freeze` thresholds.
    #[must_use]
    pub const fn paper_defaults() -> Self {
        Self {
            flatness_threshold: SHARD_FREEZE_FLATNESS_THRESHOLD,
        }
    }
}

impl Default for ShardFreezeRubricConfig {
    #[inline]
    fn default() -> Self {
        Self::paper_defaults()
    }
}

/// The shard-freeze rubric — P0/P1 mirroring
/// `riir-neuron-db::ConsolidationPipeline::can_freeze`.
///
/// Arity 2. Default fire rule:
/// [`FireRule::shard_freeze_rule_2`](super::super::fire_rule::FireRule::shard_freeze_rule_2)
/// = `And(0b0011)` = `P0 ∧ P1`.
///
/// # Isomorphism with `can_freeze`
///
/// ```text
/// can_freeze = input_sufficient && output_converged
///            = (n_wake_events >= intrinsic_dim) && (spectral_flatness < 0.3)
///            = P0 && P1
///            = FireRule::shard_freeze_rule_2().evaluate(verdict)
/// ```
///
/// The CUCG `Compress` decision corresponds to `can_freeze = true`; the
/// `Continue` decision corresponds to `can_freeze = false`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShardFreezeRubric {
    /// Threshold config.
    pub config: ShardFreezeRubricConfig,
}

impl ShardFreezeRubric {
    /// Construct with paper-default config (flatness threshold 0.3).
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with custom config.
    #[inline]
    #[must_use]
    pub const fn with_config(config: ShardFreezeRubricConfig) -> Self {
        Self { config }
    }

    /// Evaluate the rubric against caller-supplied features, producing the
    /// 2-predicate verdict (P0, P1).
    #[inline]
    #[must_use]
    pub fn evaluate_features(&self, f: &ShardFreezeFeatures) -> RubricVerdict<2> {
        // P0: input-sufficient. Yes iff n_wake_events >= intrinsic_dim.
        // This mirrors subspace_phase_gate::phase_transition_gate(n, d) and
        // can_freeze's input_sufficient field.
        let p0 = if f.n_wake_events >= f.intrinsic_dim {
            PredicateResult::Yes {
                quote_start: f.n_wake_events as u32,
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::Custom(20), // P0: input insufficient
            }
        };
        // P1: output-converged. Yes iff spectral_flatness < threshold.
        // This mirrors can_freeze's output_converged field.
        // NaN-safe: NaN flatness → No (don't freeze on unknown data).
        let p1 = if !f.spectral_flatness.is_nan()
            && f.spectral_flatness < self.config.flatness_threshold
        {
            PredicateResult::Yes {
                quote_start: 0,
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::Custom(21), // P1: not converged
            }
        };
        RubricVerdict::new([p0, p1])
    }

    /// Compute the `FreezeGateReport`-equivalent fields from a verdict, for
    /// audit-trail parity with riir-neuron-db. This is the bridge that makes
    /// the CUCG audit record carry the same information as
    /// `FreezeGateReport`.
    #[inline]
    #[must_use]
    pub fn freeze_report_fields(
        &self,
        f: &ShardFreezeFeatures,
        verdict: &RubricVerdict<2>,
    ) -> FreezeReportFields {
        FreezeReportFields {
            n_wake_events: f.n_wake_events,
            intrinsic_dim: f.intrinsic_dim,
            input_sufficient: verdict.is_yes(0),
            output_flatness: f.spectral_flatness,
            output_converged: verdict.is_yes(1),
            can_freeze: verdict.is_yes(0) && verdict.is_yes(1),
        }
    }
}

/// A field-for-field mirror of
/// `riir-neuron-db::phase_gate::FreezeGateReport`, produced by
/// [`ShardFreezeRubric::freeze_report_fields`]. Used to prove the G7
/// isomorphism: the same fields, the same semantics, the same decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreezeReportFields {
    /// Number of wake events (mirrors `FreezeGateReport::n_wake_events`).
    pub n_wake_events: usize,
    /// Intrinsic dimension `d` (mirrors `FreezeGateReport::intrinsic_dim`).
    pub intrinsic_dim: usize,
    /// Input-side gate: `n >= d` (mirrors `FreezeGateReport::input_sufficient`).
    pub input_sufficient: bool,
    /// Spectral flatness (mirrors `FreezeGateReport::output_flatness`).
    pub output_flatness: f32,
    /// Output-side gate: `flatness < 0.3` (mirrors `FreezeGateReport::output_converged`).
    pub output_converged: bool,
    /// Two-sided decision: `input_sufficient && output_converged`
    /// (mirrors `FreezeGateReport::can_freeze`).
    pub can_freeze: bool,
}

/// `Rubric<2>` impl. Reads features from `scratch.f32_buf` and
/// `scratch.usize_buf` in canonical order:
/// - `usize_buf[0]` = `n_wake_events`
/// - `usize_buf[1]` = `intrinsic_dim`
/// - `f32_buf[0]` = `spectral_flatness`
impl Rubric<2> for ShardFreezeRubric {
    #[inline]
    fn evaluate(&self, _trajectory: &[u8], scratch: &mut RubricScratch) -> RubricVerdict<2> {
        let n_wake_events = scratch.usize_buf.first().copied().unwrap_or(0);
        let intrinsic_dim = scratch.usize_buf.get(1).copied().unwrap_or(0);
        let spectral_flatness = scratch.f32_buf.first().copied().unwrap_or(f32::NAN);
        let features = ShardFreezeFeatures::new(n_wake_events, intrinsic_dim, spectral_flatness);
        self.evaluate_features(&features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::Backstop;
    use crate::compaction::audit::DecisionKind;
    use crate::compaction::fire_rule::FireRule;
    use crate::compaction::gate::ClosedUnitCompactionGate;

    // ─── Unit: predicate wiring ──────────────────────────────────────────

    #[test]
    fn p0_yes_when_n_wake_events_ge_intrinsic_dim() {
        let rubric = ShardFreezeRubric::new();
        let f = ShardFreezeFeatures::new(10, 8, 0.1);
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0), "P0: 10 >= 8 → Yes (input sufficient)");
        assert!(v.is_yes(1), "P1: 0.1 < 0.3 → Yes (converged)");
    }

    #[test]
    fn p0_no_when_n_wake_events_lt_intrinsic_dim() {
        let rubric = ShardFreezeRubric::new();
        let f = ShardFreezeFeatures::new(5, 8, 0.1);
        let v = rubric.evaluate_features(&f);
        assert!(!v.is_yes(0), "P0: 5 < 8 → No (input insufficient)");
        assert!(v.is_yes(1));
    }

    #[test]
    fn p1_no_when_spectral_flatness_ge_threshold() {
        let rubric = ShardFreezeRubric::new();
        let f = ShardFreezeFeatures::new(10, 8, 0.5);
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0));
        assert!(!v.is_yes(1), "P1: 0.5 >= 0.3 → No (not converged)");
    }

    #[test]
    fn p1_no_on_nan_flatness() {
        let rubric = ShardFreezeRubric::new();
        let f = ShardFreezeFeatures::new(10, 8, f32::NAN);
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0));
        assert!(!v.is_yes(1), "P1: NaN → No (safe default)");
    }

    #[test]
    fn p1_boundary_strict_less_than() {
        let rubric = ShardFreezeRubric::new();
        // flatness == threshold (0.3) → No (strict <).
        let f = ShardFreezeFeatures::new(10, 8, 0.3);
        let v = rubric.evaluate_features(&f);
        assert!(!v.is_yes(1), "P1: 0.3 == 0.3 → No (strict <)");
        // flatness just below → Yes.
        let f2 = ShardFreezeFeatures::new(10, 8, 0.299);
        let v2 = rubric.evaluate_features(&f2);
        assert!(v2.is_yes(1), "P1: 0.299 < 0.3 → Yes");
    }

    // ─── G7: exhaustive isomorphism with can_freeze ──────────────────────
    //
    // The Super-GOAT claim: CUCG's shard-freeze gate and riir-neuron-db's
    // `can_freeze` produce the SAME decision on the same inputs because they
    // are the same Boolean formula:
    //
    //   can_freeze = input_sufficient && output_converged
    //              = (n >= d) && (flatness < 0.3)
    //              = P0 && P1
    //
    // We test all 4 combinations of (P0, P1) and verify the CUCG decision
    // (Compress/Continue) matches the can_freeze formula exactly. This is
    // the structural isomorphism proof — no cross-repo dependency needed.

    /// The `can_freeze` formula as a pure function (mirrors
    /// `riir-neuron-db::ConsolidationPipeline::can_freeze`'s decision logic).
    fn can_freeze_formula(n: usize, d: usize, flatness: f32) -> bool {
        let input_sufficient = n >= d;
        let output_converged = flatness < SHARD_FREEZE_FLATNESS_THRESHOLD;
        input_sufficient && output_converged
    }

    #[test]
    fn g7_isomorphism_all_four_combinations() {
        let rubric = ShardFreezeRubric::new();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::shard_freeze_rule_2())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();

        // Test cases covering all 4 combinations of (input_sufficient, output_converged).
        let cases = [
            // (n, d, flatness, label)
            (10, 8, 0.1, "P0=Yes P1=Yes → can_freeze=true"),
            (10, 8, 0.5, "P0=Yes P1=No  → can_freeze=false"),
            (5, 8, 0.1, "P0=No  P1=Yes → can_freeze=false"),
            (5, 8, 0.5, "P0=No  P1=No  → can_freeze=false"),
        ];

        for (n, d, flatness, label) in cases {
            scratch.clear();
            scratch.usize_buf.push(n);
            scratch.usize_buf.push(d);
            scratch.f32_buf.push(flatness);

            let decision = gate.evaluate(b"shard", 0, 1_000_000, None, &mut scratch);
            let expected_freeze = can_freeze_formula(n, d, flatness);

            // CUCG Compress ↔ can_freeze = true.
            // CUCG Continue ↔ can_freeze = false.
            match (&decision, expected_freeze) {
                (d, true) if d.is_compress() => { /* correct */ }
                (d, false) if d.is_continue() => { /* correct */ }
                _ => panic!(
                    "{label}: CUCG decision {:?} != can_freeze={expected_freeze}",
                    DecisionKind::from_byte(decision.audit().decision)
                ),
            }
            eprintln!(
                "{label}: CUCG={:?} can_freeze={expected_freeze} ✓",
                DecisionKind::from_byte(decision.audit().decision)
            );
        }
    }

    #[test]
    fn g7_freeze_report_fields_match_can_freeze_semantics() {
        // The FreezeReportFields produced by ShardFreezeRubric must have the
        // same field semantics as riir-neuron-db's FreezeGateReport:
        //   can_freeze = input_sufficient && output_converged
        let rubric = ShardFreezeRubric::new();

        let cases = [
            (10, 8, 0.1), // both yes
            (10, 8, 0.5), // P0 yes, P1 no
            (5, 8, 0.1),  // P0 no, P1 yes
            (5, 8, 0.5),  // both no
        ];

        for (n, d, flatness) in cases {
            let f = ShardFreezeFeatures::new(n, d, flatness);
            let v = rubric.evaluate_features(&f);
            let report = rubric.freeze_report_fields(&f, &v);

            // Field-by-field semantic checks (mirror FreezeGateReport).
            assert_eq!(report.n_wake_events, n);
            assert_eq!(report.intrinsic_dim, d);
            assert_eq!(report.input_sufficient, n >= d, "input_sufficient = n >= d");
            assert_eq!(report.output_flatness, flatness);
            assert_eq!(
                report.output_converged,
                flatness < SHARD_FREEZE_FLATNESS_THRESHOLD,
                "output_converged = flatness < 0.3"
            );
            assert_eq!(
                report.can_freeze,
                report.input_sufficient && report.output_converged,
                "can_freeze = input_sufficient && output_converged"
            );
        }
    }

    #[test]
    fn g7_bit_identical_decision_across_repeated_evaluations() {
        // The sync-boundary contract: the same inputs MUST produce the same
        // decision every time (deterministic audit). This is what makes two
        // honest nodes agree.
        let rubric = ShardFreezeRubric::new();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::shard_freeze_rule_2())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();

        let mut decisions = Vec::new();
        for _ in 0..10 {
            scratch.clear();
            scratch.usize_buf.push(10);
            scratch.usize_buf.push(8);
            scratch.f32_buf.push(0.15);
            let d = gate.evaluate(b"shard", 0, 1_000_000, None, &mut scratch);
            let audit = d.audit();
            // Serialize to bytes (the sync crossing).
            let bytes: &[u8] = bytemuck::bytes_of(audit);
            decisions.push(bytes.to_vec());
        }
        // All 10 byte vectors must be bit-identical.
        let first = &decisions[0];
        for (i, d) in decisions.iter().enumerate() {
            assert_eq!(d, first, "iteration {i} not bit-identical to iteration 0");
        }
    }

    // ─── Integration: gate + rubric ──────────────────────────────────────

    #[test]
    fn gate_shard_freeze_compresses_when_both_predicates_pass() {
        let rubric = ShardFreezeRubric::new();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::shard_freeze_rule_2())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.usize_buf.push(12);
        scratch.usize_buf.push(8);
        scratch.f32_buf.push(0.1);
        let d = gate.evaluate(b"shard", 0, 1_000_000, None, &mut scratch);
        assert!(
            d.is_compress(),
            "n=12>=d=8, flatness=0.1<0.3 → freeze (Compress)"
        );
    }

    #[test]
    fn gate_shard_freeze_continues_when_input_insufficient() {
        let rubric = ShardFreezeRubric::new();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::shard_freeze_rule_2())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.usize_buf.push(3);
        scratch.usize_buf.push(8);
        scratch.f32_buf.push(0.1);
        let d = gate.evaluate(b"shard", 0, 1_000_000, None, &mut scratch);
        assert!(d.is_continue(), "n=3<d=8 → no freeze (Continue)");
    }
}
