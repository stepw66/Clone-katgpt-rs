//! Phase 4 — the paper's **math rubric** (Q1/Q2/Q3), latent-reframed per
//! Research 300 §1.2 as pure sigmoid projections over caller-supplied scalar
//! features.
//!
//! # Predicates
//!
//! | Idx | Paper | Latent feature | Sigmoid gate | "Yes" means |
//! |-----|-------|----------------|--------------|-------------|
//! | 0   | Q1    | `answer_confidence`  | `σ(β·(conf − τ))`, β>0  | a specific final answer is stated |
//! | 1   | Q2    | `fact_novelty_rate`  | `σ(β·(τ − rate))`, β>0  | last rounds added NO non-trivial fact (stuck) |
//! | 2   | Q3    | `next_step_certainty`| `σ(β·(cert − τ))`, β>0  | can write the exact next step |
//!
//! The fire rule is the paper's `Q1 ∨ (Q2 ∧ Q3)`
//! ([`FireRule::math_rule_3`](super::super::fire_rule::FireRule::math_rule_3)).
//!
//! - **Branch A** (`Q1 = Yes`): lock-in re-prompt preserving the boxed
//!   answer. Compaction is safe because the final answer is already on the
//!   record.
//! - **Branch B** (`Q2 ∧ Q3`): the derivation is stuck (Q2) BUT a clear next
//!   step exists (Q3). Compaction is safe because the next step survives the
//!   summary; the dead paraphrase rounds are discarded.
//!
//! # Caller responsibilities
//!
//! Like [`SearchRubric`](super::search::SearchRubric), the `MathRubric` is
//! agnostic to where the scalars come from. Suggested sources:
//!
//! - **Q1 answer_confidence** — a regex / structural detector for `\boxed{}`
//!   or "Final Answer:" (deterministic, modelless), or a learned-answer-head
//!   logit. Range `[0, 1]`.
//! - **Q2 fact_novelty_rate** — `katgpt_core::cgsp::derivative_curiosity`
//!   rate over the last 2 rounds, or any fact-novelty probe. Range
//!   `[0, +∞)`. LOW = stuck (few new facts); the predicate is inverted
//!   (Yes iff rate LOW).
//! - **Q3 next_step_certainty** — a structural "can I name the next step?"
//!   signal: 1.0 if a case split / substitution / lemma is detectable, 0.0
//!   otherwise. Range `[0, 1]`.

use super::super::rubric::{
    PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict,
};
use super::search::PredicateParams;

/// Caller-supplied scalar features for the [`MathRubric`].
///
/// Field order matches the paper's predicate order (Q1, Q2, Q3).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MathTrajectoryFeatures {
    /// Q1 source — confidence that a specific final answer is stated.
    /// Higher = more likely an answer is present. Range `[0, 1]`.
    pub answer_confidence: f32,
    /// Q2 source — rate of non-trivial fact production in recent rounds.
    /// LOWER = more stuck. Range `[0, +∞)`. The predicate is inverted:
    /// "Yes (stuck)" iff rate is below threshold.
    pub fact_novelty_rate: f32,
    /// Q3 source — certainty that a clear next step can be written.
    /// Higher = more certain. Range `[0, 1]`.
    pub next_step_certainty: f32,
}

impl MathTrajectoryFeatures {
    /// Construct a feature vector. Field order matches Q1, Q2, Q3.
    #[inline]
    #[must_use]
    pub const fn new(
        answer_confidence: f32,
        fact_novelty_rate: f32,
        next_step_certainty: f32,
    ) -> Self {
        Self {
            answer_confidence,
            fact_novelty_rate,
            next_step_certainty,
        }
    }
}

/// Configuration for the three [`MathRubric`] predicates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MathRubricConfig {
    /// Q1 answer: `σ(β·(conf − τ))`, β > 0. "Yes" iff confidence high.
    pub q1: PredicateParams,
    /// Q2 stuck: `σ(β·(τ − rate))` — encoded as negative-β PredicateParams
    /// so `fires()` returns "Yes iff rate LOW". "Yes" means stuck.
    pub q2: PredicateParams,
    /// Q3 has-next: `σ(β·(cert − τ))`, β > 0. "Yes" iff certainty high.
    pub q3: PredicateParams,
}

impl MathRubricConfig {
    /// Paper-calibrated defaults (Research 300 §1.2).
    ///
    /// - **Q1**: answer_confidence ≥ 0.7 (β = 4.0). A `\boxed{}` or
    ///   structural detector typically outputs 0.0 or 1.0; 0.7 leaves
    ///   margin for soft detectors.
    /// - **Q2**: fact_novelty_rate ≤ 0.5 (β = -2.0 inverted). "Fewer than
    ///   0.5 new facts per round in the last 2 rounds = stuck."
    /// - **Q3**: next_step_certainty ≥ 0.6 (β = 4.0). A clear next step
    ///   (case split, substitution, lemma) scores high.
    #[must_use]
    pub const fn paper_defaults() -> Self {
        Self {
            q1: PredicateParams::new(4.0, 0.7),
            q2: PredicateParams::new(-2.0, 0.5),
            q3: PredicateParams::new(4.0, 0.6),
        }
    }
}

impl Default for MathRubricConfig {
    #[inline]
    fn default() -> Self {
        Self::paper_defaults()
    }
}

/// The paper's math rubric — Q1/Q2/Q3 over caller-supplied scalars.
///
/// Arity 3. Default fire rule: [`FireRule::math_rule_3`](super::super::fire_rule::FireRule::math_rule_3).
/// Default config: [`MathRubricConfig::paper_defaults`].
///
/// Stateless like [`SearchRubric`](super::search::SearchRubric); per-probe
/// features live in the caller's `SearchFeatures` (reused as a generic
/// carrier — the math rubric reads its 3 features from the first 3 slots).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MathRubric {
    /// Sigmoid β/τ parameters per predicate.
    pub config: MathRubricConfig,
}

impl MathRubric {
    /// Construct with paper-default config.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with custom config.
    #[inline]
    #[must_use]
    pub const fn with_config(config: MathRubricConfig) -> Self {
        Self { config }
    }

    /// Evaluate the rubric against caller-supplied features, producing the
    /// 3-predicate verdict (Q1, Q2, Q3).
    #[inline]
    #[must_use]
    pub fn evaluate_features(&self, f: &MathFeatures) -> RubricVerdict<3> {
        let span = f.span_end;
        let feat = f.features;
        let yes_span = || PredicateResult::Yes {
            quote_start: span.saturating_sub(1),
            quote_len: 1,
        };
        let q1 = if self.config.q1.fires(feat.answer_confidence) {
            yes_span()
        } else {
            PredicateResult::No {
                reason: PredicateReason::Custom(10), // Q1: no answer detected
            }
        };
        // Q2: stuck. Yes iff rate LOW (β negative).
        let q2 = if self.config.q2.fires(feat.fact_novelty_rate) {
            PredicateResult::Yes {
                quote_start: span.saturating_sub(1),
                quote_len: 1,
            }
        } else {
            PredicateResult::No {
                reason: PredicateReason::Custom(11), // Q2: not stuck (still producing facts)
            }
        };
        let q3 = if self.config.q3.fires(feat.next_step_certainty) {
            yes_span()
        } else {
            PredicateResult::No {
                reason: PredicateReason::Custom(12), // Q3: no clear next step
            }
        };
        RubricVerdict::new([q1, q2, q3])
    }
}

/// Carrier for the [`MathRubric`]'s caller-supplied features (mirrors
/// [`SearchFeatures`] for the math domain).
#[derive(Clone, Debug, Default)]
pub struct MathFeatures {
    /// The current probe's scalar features.
    pub features: MathTrajectoryFeatures,
    /// Trajectory byte-offset for the Yes audit span.
    pub span_end: u32,
}

impl MathFeatures {
    /// Construct with initial features and `span_end = 0`.
    #[inline]
    #[must_use]
    pub fn new(features: MathTrajectoryFeatures) -> Self {
        Self {
            features,
            span_end: 0,
        }
    }
}

/// `Rubric<3>` impl. Reads features from `scratch.f32_buf[0..3]` in
/// canonical order `[answer_confidence, fact_novelty_rate,
/// next_step_certainty]` and span from `usize_buf[0]` or trajectory length.
impl Rubric<3> for MathRubric {
    #[inline]
    fn evaluate(&self, trajectory: &[u8], scratch: &mut RubricScratch) -> RubricVerdict<3> {
        let span = scratch
            .usize_buf
            .first()
            .copied()
            .unwrap_or(trajectory.len()) as u32;
        let answer_confidence = scratch.f32_buf.first().copied().unwrap_or(0.0);
        let fact_novelty_rate = scratch.f32_buf.get(1).copied().unwrap_or(0.0);
        let next_step_certainty = scratch.f32_buf.get(2).copied().unwrap_or(0.0);
        let features = MathFeatures {
            features: MathTrajectoryFeatures::new(
                answer_confidence,
                fact_novelty_rate,
                next_step_certainty,
            ),
            span_end: span,
        };
        self.evaluate_features(&features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::Backstop;
    use crate::compaction::fire_rule::FireRule;
    use crate::compaction::gate::ClosedUnitCompactionGate;

    #[test]
    fn math_rubric_branch_a_q1_yes_fires() {
        // Q1 Yes (answer stated) → fires regardless of Q2, Q3.
        let rubric = MathRubric::default();
        let f = MathFeatures::new(MathTrajectoryFeatures::new(0.9, 2.0, 0.1));
        let v = rubric.evaluate_features(&f);
        assert!(v.is_yes(0), "Q1: conf 0.9 > 0.7 → Yes");
        assert!(!v.is_yes(1), "Q2: rate 2.0 > 0.5 → No (not stuck)");
        assert!(!v.is_yes(2), "Q3: cert 0.1 < 0.6 → No");
        assert_eq!(v.yes_mask(), 0b001);
        assert!(
            FireRule::math_rule_3().evaluate(&v),
            "Q1 → fires (Branch A)"
        );
    }

    #[test]
    fn math_rubric_branch_b_q2_q3_fires() {
        // Q2 Yes (stuck) AND Q3 Yes (has next step) → fires.
        let rubric = MathRubric::default();
        let f = MathFeatures::new(MathTrajectoryFeatures::new(0.1, 0.1, 0.9));
        let v = rubric.evaluate_features(&f);
        assert!(!v.is_yes(0), "Q1: conf 0.1 < 0.7 → No");
        assert!(v.is_yes(1), "Q2: rate 0.1 < 0.5 → Yes (stuck)");
        assert!(v.is_yes(2), "Q3: cert 0.9 > 0.6 → Yes");
        assert_eq!(v.yes_mask(), 0b110);
        assert!(
            FireRule::math_rule_3().evaluate(&v),
            "Q2 ∧ Q3 → fires (Branch B)"
        );
    }

    #[test]
    fn math_rubric_stuck_no_next_step_does_not_fire() {
        // Q2 Yes (stuck) but Q3 No (no next step) → does NOT fire.
        // This is the "dead end" case: compacting here loses the thread.
        let rubric = MathRubric::default();
        let f = MathFeatures::new(MathTrajectoryFeatures::new(0.1, 0.1, 0.1));
        let v = rubric.evaluate_features(&f);
        assert!(!v.is_yes(0));
        assert!(v.is_yes(1));
        assert!(!v.is_yes(2));
        assert_eq!(v.yes_mask(), 0b010);
        assert!(
            !FireRule::math_rule_3().evaluate(&v),
            "Q2 only, no Q3 → dead end → no fire"
        );
    }

    #[test]
    fn math_rubric_not_stuck_no_answer_does_not_fire() {
        // Q1 No, Q2 No (still producing facts) → does NOT fire regardless of Q3.
        let rubric = MathRubric::default();
        let f = MathFeatures::new(MathTrajectoryFeatures::new(0.1, 2.0, 0.9));
        let v = rubric.evaluate_features(&f);
        assert!(!v.is_yes(0));
        assert!(!v.is_yes(1));
        assert!(v.is_yes(2));
        assert_eq!(v.yes_mask(), 0b100);
        assert!(
            !FireRule::math_rule_3().evaluate(&v),
            "No Q1, no Q2 → no fire (still making progress)"
        );
    }

    #[test]
    fn gate_with_math_rubric_branch_a_compresses() {
        let rubric = MathRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::math_rule_3())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.f32_buf.extend_from_slice(&[0.9, 2.0, 0.1]);
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        assert!(d.is_compress(), "Branch A → Compress");
    }

    #[test]
    fn gate_with_math_rubric_branch_b_compresses() {
        let rubric = MathRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::math_rule_3())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.f32_buf.extend_from_slice(&[0.1, 0.1, 0.9]);
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        assert!(d.is_compress(), "Branch B → Compress");
    }

    #[test]
    fn gate_with_math_rubric_dead_end_continues() {
        let rubric = MathRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .fire_rule(FireRule::math_rule_3())
            .backstop(Backstop::None)
            .build();
        let mut scratch = RubricScratch::new();
        scratch.f32_buf.extend_from_slice(&[0.1, 0.1, 0.1]); // stuck, no next step
        let d = gate.evaluate(b"traj", 0, 10_000, None, &mut scratch);
        assert!(d.is_continue(), "dead end → Continue");
    }
}
