//! Step-Attribution Δ-Qualification Primitive (Plan 381, Research 381)
//!
//! Generic modelless instantiation of SkillAdaptor's eq. 8 (Yu et al. 2026,
//! arXiv:2606.01311):
//!
//! ```text
//! Δ = E_q[M(q; K+)] - E_q[M(q; K)]
//! ```
//!
//! Commit candidate state `K+` iff `Δ ≥ threshold` (SkillAdaptor uses `≥ 0`).
//! This is the explicit "did this update actually help?" gate that sits between
//! a proposed mutation and the actual state commit. Sibling to
//! [`crate::trajectory_doctor::TrajectoryDoctor`] (Plan 223):
//! TrajectoryDoctor *localizes* the fault (token-index); this gate decides
//! whether to *commit* the proposed fix (tick-index, generic).
//!
//! # Why modelless
//!
//! SkillAdaptor trains the Qualifier via supervised learning on
//! (prompt, K+, K, reward) tuples. The modelless distillation: the Qualifier's
//! only job is to compare aggregate scores under two states on a fixed replay
//! window. A deterministic `ReplayExecutor` + `ScoreAggregator` reproduces the
//! *decision structure* (the variance-reduction discipline the paper's ablation
//! attributes the Qualifier to — ±8.1→±5.2 on PinchBench) without any LLM calls
//! or learned weights. The consumer (riir-ai Plan 313) plugs in its own
//! cognitive-stack replay + CLR reward aggregator.
//!
//! # Stack slot
//!
//! This primitive does NOT fit a transformer stack slot. It sits in the
//! **cognitive-branch / skill-lifecycle stack** alongside `TrajectoryDoctor`,
//! `FailureEpisodeStore`, and `ClosedUnitCompactionGate`. Promotion/demotion is
//! tracked against the shipped `TrajectoryDoctor` family — if
//! `StepAttributionQualifier` wins the variance-reduction comparison in
//! riir-ai Plan 313 G6, it becomes the recommended commit gate for
//! cognitive-branch / skill-lifecycle consumers; `TrajectoryDoctor` alone
//! remains for localization-only use cases.
//!
//! # Canonical usage
//!
//! ```
//! use katgpt_pruners::step_attribution_qualifier::*;
//!
//! // Toy: state is a scalar f32, replay input is (), score is f32.
//! struct IdentityExecutor;
//! impl ReplayExecutor<f32, (), f32> for IdentityExecutor {
//!     fn replay(&self, k: &f32, inputs: &[()]) -> Vec<f32> {
//!         inputs.iter().map(|_| *k).collect()
//!     }
//! }
//! struct SumAggregator;
//! impl ScoreAggregator<f32> for SumAggregator {
//!     fn aggregate(&self, scores: &[f32]) -> f32 { scores.iter().sum() }
//! }
//! struct AddConst(f32);
//! impl CandidateMutation<f32> for AddConst {
//!     fn apply_to(&self, baseline: &f32) -> f32 { baseline + self.0 }
//! }
//!
//! let qualifier = StepAttributionQualifier::new(IdentityExecutor, SumAggregator, 0.0);
//! let verdict = qualifier.qualify(&1.0, &AddConst(2.0), &[(), (), ()]);
//! assert_eq!(verdict, QualificationVerdict::Commit { delta_above_threshold: true });
//! ```
//!
//! # Prior art: R172 ITSE `WasmTestGate`
//!
//! R172 (MUSE/ITSE, arXiv:2605.27366) proposed `avg_reward_delta`-gated
//! registration for new pruner arms: a candidate arm enters the bandit bank
//! only if its mean reward on the test suite beats the existing best arm's
//! (`avg_reward_delta >= 0`). The shipped [`crate::skill_test::WasmTestGate`]
//! simplified this to a coverage-only gate (no Δ field).
//! [`WasmTestGateAdapter`] restores the R172-proposed Δ acceptance as a special
//! case of this generic primitive — the R172 rule is exactly
//! `StepAttributionQualifier` with `threshold = 0.0` over scalar mean-reward
//! states. This primitive generalizes from "new-pruner registration" to "any
//! candidate/baseline pair" (cognitive-branch updates, freeze acceptance,
//! CommittedFieldBlend gates, etc.).

use core::marker::PhantomData;

// ─────────────────────────────────────────────────────────────────────────
// Traits
// ─────────────────────────────────────────────────────────────────────────

/// A proposed mutation to a baseline state, producing a candidate state `K+`.
///
/// Implementors hold the diff (e.g. failure-counter bump, anti-pattern append,
/// direction delta) so `apply_to` can construct `K+` without mutating the
/// baseline. For hot-path consumers that need in-place mutation, see the
/// Failure Mode note in Plan 381 — ship a parallel `InPlaceCandidateMutation`.
///
/// Generic over the opaque state `K` (e.g. `BranchBank`, a pruner config, a
/// frozen weight snapshot).
pub trait CandidateMutation<K> {
    /// Apply this mutation to the baseline, producing the candidate state `K+`.
    ///
    /// MUST be deterministic: `apply_to(baseline)` always returns the same `K+`
    /// for the same baseline. Non-determinism breaks the Δ comparison.
    fn apply_to(&self, baseline: &K) -> K;
}

/// Deterministic re-execution of a replay window under a given state.
///
/// Implementors guarantee **bit-identical** results for the same `(state, inputs)`
/// pair. If the underlying runtime has non-deterministic ordering (e.g. rayon
/// parallelism in the cognition stack), the consumer MUST force single-threaded
/// execution for the replay window or fall back to a statistical Δ (re-execute
/// N times, compare mean ± stddev). See Plan 313 Failure Mode.
///
/// Generic over:
/// - `K` — the state being qualified (e.g. `BranchBank`).
/// - `I` — the per-step replay input (e.g. `CognitiveBranchTickRecord`).
/// - `S` — the per-step score (e.g. CLR `r_k` as `f32`).
pub trait ReplayExecutor<K, I, S> {
    /// Re-execute the replay window `inputs` under state `k`, returning
    /// per-step scores. The output length MUST equal `inputs.len()`.
    fn replay(&self, k: &K, inputs: &[I]) -> Vec<S>;
}

/// Aggregate a per-step score slice into a single comparable metric.
///
/// Typically `Sum` for CLR reward (more reward = better); may be `Mean` for
/// normalized metrics. The aggregator is applied identically to both the
/// baseline and candidate score vectors — only the state differs.
pub trait ScoreAggregator<S> {
    /// Reduce `scores` to a single metric. MUST be deterministic.
    fn aggregate(&self, scores: &[S]) -> S;
}

/// SkillAdaptor's Localize (eq. 5) + Link (eq. 6) fused into one modelless call.
///
/// Generalizes
/// [`crate::trajectory_doctor::TrajectoryDoctor::localize_failure`] from
/// token-index (DDTree paths) to a generic tick-index (cognitive-branch replay
/// records, pruner step traces, etc.).
///
/// Implementors find the first fault tick in a trajectory, then project the
/// state-delta at that tick onto a direction set to attribute responsibility
/// (SkillAdaptor's Linker, eq. 6).
pub trait StepLocalizer<Dir, W> {
    /// Given a trajectory of per-tick state-deltas + CLR scores + a direction
    /// set, return the first actionable fault + responsibility weights.
    ///
    /// - `trajectory_deltas[t]` — the state-delta at tick `t` (e.g. HLA-delta).
    /// - `trajectory_scores[t]` — the CLR `r_k` score at tick `t`.
    /// - `directions[j]` — the `j`-th direction to project onto (e.g. branch
    ///   direction vectors).
    /// - `tau_reliable` — the CLR reliability threshold; a tick is a candidate
    ///   fault iff `r_k < tau_reliable`.
    ///
    /// Returns `None` if no tick in the window falls below `tau_reliable`.
    /// Returns `Some(TickFaultSite)` for the first actionable fault: the tick
    /// where `r_k < tau_reliable` AND the state-delta is large enough to
    /// attribute to a direction (above the consumer's branch-magnitude floor).
    fn localize_and_link(
        &self,
        trajectory_deltas: &[Dir],
        trajectory_scores: &[W],
        directions: &[Dir],
        tau_reliable: W,
    ) -> Option<TickFaultSite<Dir, W>>
    where
        Dir: AsRef<[W]>,
        W: Copy + PartialOrd + core::ops::Sub<Output = W>;
}

// ─────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────

/// The verdict returned by [`StepAttributionQualifier::qualify`].
///
/// Mirrors the Commit/Rollback decision in SkillAdaptor eq. 8: commit iff
/// `Δ ≥ threshold`. The `delta_above_threshold` / `delta_below_threshold`
/// fields are diagnostic booleans (always `true` for the matching variant) that
/// make the verdict self-documenting in logs without requiring the caller to
/// re-derive the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationVerdict {
    /// `Δ ≥ threshold` — commit the candidate state `K+`.
    Commit {
        /// Diagnostic: the Δ exceeded the acceptance threshold.
        delta_above_threshold: bool,
    },
    /// `Δ < threshold` — rollback to baseline `K`.
    Rollback {
        /// Diagnostic: the Δ fell below the acceptance threshold.
        delta_below_threshold: bool,
    },
}

/// A localized fault: which tick, which "responsible direction", improvement hint.
///
/// Generalizes [`crate::trajectory_doctor::FailureSite`] from token-index to
/// tick-index + adds the SkillAdaptor Linker's responsibility weight.
#[derive(Debug, Clone)]
pub struct TickFaultSite<Dir, W> {
    /// The first actionable fault tick in the replay window.
    pub tick_idx: usize,
    /// The violated predicate / observation (mirrors
    /// [`FailureSite::violated_predicate`](crate::trajectory_doctor::FailureSite::violated_predicate)).
    pub violated: String,
    /// Per-direction responsibility weights (SkillAdaptor eq. 6 output).
    ///
    /// `weights[j] = sigmoid(dot(hla_delta_at_t_star, direction_j))`.
    /// Higher weight = more responsible for the fault.
    pub responsibility: Vec<W>,
    /// Argmax direction index — the "responsible skill/branch".
    ///
    /// When ties occur, the consumer picks the higher-priority direction
    /// (e.g. lower branch_id per R161 §2.2 spawn order).
    pub responsible_idx: usize,
    /// Marker for the generic direction type (e.g. HLA direction vector).
    pub _marker: PhantomData<Dir>,
}

impl<Dir, W> TickFaultSite<Dir, W> {
    /// Construct a fault site. `responsible_idx` MUST be a valid index into
    /// `responsibility`.
    pub fn new(
        tick_idx: usize,
        violated: impl Into<String>,
        responsibility: Vec<W>,
        responsible_idx: usize,
    ) -> Self {
        debug_assert!(
            responsible_idx < responsibility.len(),
            "responsible_idx out of bounds"
        );
        Self {
            tick_idx,
            violated: violated.into(),
            responsibility,
            responsible_idx,
            _marker: PhantomData,
        }
    }
}

/// The Δ≥0 commit gate. Wraps a [`ReplayExecutor`] + [`ScoreAggregator`].
///
/// Generic over:
/// - `K` — the state being qualified.
/// - `I` — the per-step replay input.
/// - `S` — the per-step score (must support subtraction + ordering).
/// - `E` — the [`ReplayExecutor`].
/// - `A` — the [`ScoreAggregator`].
///
/// Construct via [`StepAttributionQualifier::new`] (threshold = 0.0,
/// SkillAdaptor's default) or [`StepAttributionQualifier::with_threshold`]
/// for a strictly-better requirement.
pub struct StepAttributionQualifier<K, I, S, E, A> {
    /// The deterministic re-execution backend.
    pub executor: E,
    /// The score aggregator (applied to both baseline + candidate).
    pub aggregator: A,
    /// Acceptance threshold (default `0.0` = SkillAdaptor's `Δ ≥ 0`).
    ///
    /// Positive values encode a "strictly better" requirement (reject ties).
    pub threshold: S,
    _marker: PhantomData<(K, I)>,
}

// Manual Clone/Debug: PhantomData<(K, I)> is always Clone/Debug, but the
// derived impl would require K: Clone, I: Clone which is needlessly strict
// (the marker doesn't actually hold a value).
impl<K, I, S: Clone, E: Clone, A: Clone> Clone for StepAttributionQualifier<K, I, S, E, A> {
    fn clone(&self) -> Self {
        Self {
            executor: self.executor.clone(),
            aggregator: self.aggregator.clone(),
            threshold: self.threshold.clone(),
            _marker: PhantomData,
        }
    }
}

impl<K, I, S: core::fmt::Debug, E: core::fmt::Debug, A: core::fmt::Debug> core::fmt::Debug
    for StepAttributionQualifier<K, I, S, E, A>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StepAttributionQualifier")
            .field("executor", &self.executor)
            .field("aggregator", &self.aggregator)
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl<K, I, S, E, A> StepAttributionQualifier<K, I, S, E, A>
where
    E: ReplayExecutor<K, I, S>,
    A: ScoreAggregator<S>,
    S: PartialOrd + Copy + core::ops::Sub<Output = S>,
{
    /// Construct with `threshold = 0.0` (SkillAdaptor's `Δ ≥ 0` default).
    ///
    /// Requires `S: Default` so the zero threshold can be constructed. For
    /// non-Default score types, use [`with_threshold`](Self::with_threshold).
    pub fn new(executor: E, aggregator: A, threshold: S) -> Self {
        Self {
            executor,
            aggregator,
            threshold,
            _marker: PhantomData,
        }
    }

    /// Construct with an explicit acceptance threshold.
    ///
    /// `threshold = S::default()` mirrors [`new`](Self::new) for Default types;
    /// a positive threshold encodes a "strictly better" requirement.
    pub fn with_threshold(executor: E, aggregator: A, threshold: S) -> Self {
        Self::new(executor, aggregator, threshold)
    }

    /// Run the full Δ≥0 qualification on a candidate mutation.
    ///
    /// 1. Apply mutation to baseline → `K+`.
    /// 2. Replay window under `K` → `baseline_scores`.
    /// 3. Replay window under `K+` → `candidate_scores`.
    /// 4. Aggregate both.
    /// 5. `Δ = aggregate(K+) - aggregate(K)`; `Commit` iff `Δ ≥ threshold`.
    ///
    /// Returns the verdict. The `Δ` value itself is not returned (callers that
    /// need it for logging can re-derive it by replaying + aggregating
    /// separately, or extend this struct to capture it).
    pub fn qualify(
        &self,
        baseline: &K,
        mutation: &dyn CandidateMutation<K>,
        replay_inputs: &[I],
    ) -> QualificationVerdict {
        // 1. K+ = mutation(baseline)
        let candidate = mutation.apply_to(baseline);

        // 2-3. Deterministic re-execution under both states.
        let baseline_scores = self.executor.replay(baseline, replay_inputs);
        let candidate_scores = self.executor.replay(&candidate, replay_inputs);

        // 4. Aggregate.
        let baseline_metric = self.aggregator.aggregate(&baseline_scores);
        let candidate_metric = self.aggregator.aggregate(&candidate_scores);

        // 5. Δ ≥ threshold ?
        let delta = candidate_metric - baseline_metric;
        if delta >= self.threshold {
            QualificationVerdict::Commit {
                delta_above_threshold: true,
            }
        } else {
            QualificationVerdict::Rollback {
                delta_below_threshold: true,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in aggregators (zero-dep, modelless)
// ─────────────────────────────────────────────────────────────────────────

/// `ScoreAggregator` that sums per-step scores. The CLR-reward default.
#[derive(Debug, Clone, Copy, Default)]
pub struct SumAggregator;

impl ScoreAggregator<f32> for SumAggregator {
    #[inline]
    fn aggregate(&self, scores: &[f32]) -> f32 {
        scores.iter().sum()
    }
}

/// `ScoreAggregator` that returns the mean per-step score. Use for normalized
/// metrics where the window length varies.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanAggregator;

impl ScoreAggregator<f32> for MeanAggregator {
    #[inline]
    fn aggregate(&self, scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in StepLocalizer: dot-product responsibility projection
// ─────────────────────────────────────────────────────────────────────────

/// Default [`StepLocalizer`] implementing the SkillAdaptor Localize+Link
/// modelless distillation:
///
/// - **Localize**: first tick `t*` where `r_k < tau_reliable`.
/// - **Link**: at `t*`, project `delta[t*]` onto each direction via dot-product,
///   then apply a numerically-stable sigmoid to get responsibility weights
///   `∈ (0, 1)`. The argmax is the responsible direction.
///
/// This is the generic consumer-facing localizer. Game-specific localizers
/// (e.g. riir-ai's `HlaDeltaStepLocalizer`, Plan 313 T1.6) can wrap this or
/// implement `StepLocalizer` directly with domain-specific fault predicates.
#[derive(Debug, Clone, Copy, Default)]
pub struct DotProductLocalizer {
    /// Optional floor on `|delta · direction|` — a tick is only actionable if
    /// the projection magnitude exceeds this floor. Default `0.0` (any
    /// non-trivial projection qualifies). Set higher to skip weak signals.
    pub projection_floor: f32,
}

impl DotProductLocalizer {
    /// Construct with `projection_floor = 0.0` (any non-trivial projection).
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with an explicit projection floor.
    pub fn with_floor(projection_floor: f32) -> Self {
        Self { projection_floor }
    }
}

impl StepLocalizer<Vec<f32>, f32> for DotProductLocalizer {
    fn localize_and_link(
        &self,
        trajectory_deltas: &[Vec<f32>],
        trajectory_scores: &[f32],
        directions: &[Vec<f32>],
        tau_reliable: f32,
    ) -> Option<TickFaultSite<Vec<f32>, f32>> {
        assert_eq!(
            trajectory_deltas.len(),
            trajectory_scores.len(),
            "deltas and scores must have the same length"
        );
        if directions.is_empty() {
            return None;
        }

        // Localize: first tick where r_k < tau_reliable AND the delta projects
        // above the floor onto at least one direction.
        for (tick_idx, &score) in trajectory_scores.iter().enumerate() {
            if score >= tau_reliable {
                continue;
            }
            let delta = &trajectory_deltas[tick_idx];

            // Link: project delta onto each direction.
            let mut responsibility = Vec::with_capacity(directions.len());
            let mut best_idx = 0;
            let mut best_weight = f32::NEG_INFINITY;
            for (j, dir) in directions.iter().enumerate() {
                let dot = dot(delta, dir);
                // Numerically stable sigmoid (per AGENTS.md: sigmoid not softmax).
                let weight = stable_sigmoid(dot);
                responsibility.push(weight);
                if weight > best_weight {
                    best_weight = weight;
                    best_idx = j;
                }
            }

            // Floor check: is the best projection above the floor?
            let best_dot = dot(delta, &directions[best_idx]);
            if best_dot.abs() < self.projection_floor {
                // Weak signal — not actionable, continue searching.
                continue;
            }

            return Some(TickFaultSite::new(
                tick_idx,
                format!("r_k={score:.4} < tau_reliable={tau_reliable:.4} at tick {tick_idx}"),
                responsibility,
                best_idx,
            ));
        }

        None
    }
}

/// Dot product of two equal-length f32 slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Numerically stable sigmoid. Per AGENTS.md: sigmoid not softmax.
#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 4 T4.1 — R172 ITSE `WasmTestGate` adapter (prior-art subsumption)
// ─────────────────────────────────────────────────────────────────────────

/// Adapter showing [`StepAttributionQualifier`] subsumes R172 ITSE's proposed
/// `WasmTestGate::avg_reward_delta` acceptance pattern for new-pruner
/// registration (Plan 381 Phase 4 T4.1).
///
/// **Prior art (R172 / MUSE ITSE, arXiv:2605.27366, §3 "Mechanism 3: WASM
/// Test Gates"):** before a candidate pruner arm enters the bandit bank,
/// compute its mean reward on the test suite vs the existing best arm's
/// (`avg_reward_delta`). Register iff `avg_reward_delta >= 0`. The shipped
/// [`crate::skill_test::WasmTestGate`] simplified R172's proposal to a
/// coverage-only gate (no Δ field) — this adapter restores the R172-proposed
/// Δ acceptance as a special case of the generic primitive.
///
/// **Subsumption mapping:**
/// - State `K` = `f32` (the arm's mean reward on the test suite)
/// - [`ReplayExecutor`] = [`ScalarStateExecutor`] (echoes the scalar state;
///   the "replay" is trivial because the state IS the pre-aggregated reward)
/// - [`ScoreAggregator`] = [`SumAggregator`] (identity on a single-element
///   window)
/// - [`CandidateMutation`] = [`ReplaceScalar`] (swap to candidate arm's mean)
/// - `Δ = candidate_mean - baseline_mean = avg_reward_delta`
///
/// This proves [`StepAttributionQualifier`] generalizes from "new-pruner
/// registration" to "any candidate/baseline pair" (cognitive-branch updates,
/// freeze acceptance, CommittedFieldBlend gates, etc.).
#[derive(Debug, Clone, Copy)]
pub struct WasmTestGateAdapter {
    /// Δ threshold for acceptance. R172 default: `0.0` (candidate must not
    /// regress vs existing best arm). Positive values enforce "strictly
    /// better" registration.
    pub threshold: f32,
}

impl Default for WasmTestGateAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmTestGateAdapter {
    /// Construct with R172's default threshold (`Δ ≥ 0.0`).
    pub fn new() -> Self {
        Self { threshold: 0.0 }
    }

    /// Construct with a strict threshold (candidate must beat baseline by
    /// `threshold`). Mirrors
    /// [`StepAttributionQualifier::with_threshold`].
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }

    /// R172 acceptance rule via the generic primitive.
    ///
    /// Delegates to [`StepAttributionQualifier::qualify`] with
    /// [`ScalarStateExecutor`] + [`SumAggregator`] + a single-element replay
    /// window, proving the R172 `avg_reward_delta >= 0` rule is a special
    /// case of the generic Δ≥threshold gate.
    ///
    /// - `baseline_avg_reward` — existing best arm's mean reward on the suite.
    /// - `candidate_avg_reward` — candidate arm's mean reward on the suite.
    /// - Returns [`QualificationVerdict::Commit`] iff
    ///   `candidate - baseline >= threshold`.
    #[inline]
    pub fn qualify(
        &self,
        baseline_avg_reward: f32,
        candidate_avg_reward: f32,
    ) -> QualificationVerdict {
        let qualifier =
            StepAttributionQualifier::new(ScalarStateExecutor, SumAggregator, self.threshold);
        qualifier.qualify(
            &baseline_avg_reward,
            &ReplaceScalar(candidate_avg_reward),
            &[()],
        )
    }
}

/// [`ReplayExecutor`] that echoes a scalar state — the "replay" is trivial
/// because the state IS the pre-aggregated reward. Used by
/// [`WasmTestGateAdapter`] to map R172's pre-computed `avg_reward_delta`
/// pattern onto the generic primitive.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarStateExecutor;

impl ReplayExecutor<f32, (), f32> for ScalarStateExecutor {
    /// Returns one score per input, each equal to the state scalar `k`.
    /// [`SumAggregator`] then reduces to `k * inputs.len()`; with a
    /// single-element window this is just `k`.
    #[inline]
    fn replay(&self, k: &f32, inputs: &[()]) -> Vec<f32> {
        // Length MUST equal inputs.len() per the trait contract.
        let mut out = Vec::with_capacity(inputs.len());
        for _ in inputs {
            out.push(*k);
        }
        out
    }
}

/// [`CandidateMutation`] that replaces the scalar state — models "swap to
/// the candidate arm's mean reward". Used by [`WasmTestGateAdapter`].
#[derive(Debug, Clone, Copy)]
pub struct ReplaceScalar(pub f32);

impl CandidateMutation<f32> for ReplaceScalar {
    #[inline]
    fn apply_to(&self, _baseline: &f32) -> f32 {
        self.0
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tests (Phase 2: T2.1–T2.5)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Toy fixtures ──────────────────────────────────────────────────────
    //
    // State = f32 (a scalar "quality"). Replay input = (). Score = the state
    // itself (identity executor). Sum aggregator = total reward.

    struct IdentityExecutor;
    impl ReplayExecutor<f32, (), f32> for IdentityExecutor {
        fn replay(&self, k: &f32, inputs: &[()]) -> Vec<f32> {
            inputs.iter().map(|_| *k).collect()
        }
    }

    struct AddConst(f32);
    impl CandidateMutation<f32> for AddConst {
        fn apply_to(&self, baseline: &f32) -> f32 {
            baseline + self.0
        }
    }

    // ── T2.1: Δ≥0 gate ────────────────────────────────────────────────────

    #[test]
    fn t21_commit_when_candidate_beats_baseline() {
        // baseline=1.0, candidate=1.0+2.0=3.0, Δ = 3·3 - 3·1 = 6 ≥ 0 → Commit
        let q = StepAttributionQualifier::new(IdentityExecutor, SumAggregator, 0.0);
        let verdict = q.qualify(&1.0, &AddConst(2.0), &[(), (), ()]);
        assert_eq!(
            verdict,
            QualificationVerdict::Commit {
                delta_above_threshold: true
            }
        );
    }

    #[test]
    fn t21_commit_when_candidate_equals_baseline() {
        // baseline=5.0, candidate=5.0+0.0=5.0, Δ = 0 ≥ 0 → Commit (tie allowed)
        let q = StepAttributionQualifier::new(IdentityExecutor, SumAggregator, 0.0);
        let verdict = q.qualify(&5.0, &AddConst(0.0), &[(), ()]);
        assert_eq!(
            verdict,
            QualificationVerdict::Commit {
                delta_above_threshold: true
            }
        );
    }

    #[test]
    fn t21_rollback_when_candidate_worse() {
        // baseline=4.0, candidate=4.0-1.0=3.0, Δ = 3·3 - 3·4 = -3 < 0 → Rollback
        let q = StepAttributionQualifier::new(IdentityExecutor, SumAggregator, 0.0);
        let verdict = q.qualify(&4.0, &AddConst(-1.0), &[(), (), ()]);
        assert_eq!(
            verdict,
            QualificationVerdict::Rollback {
                delta_below_threshold: true
            }
        );
    }

    // ── T2.2: threshold variant ───────────────────────────────────────────

    #[test]
    fn t22_strict_threshold_rejects_marginal_gain() {
        // baseline=1.0, candidate=1.0+0.5=1.5 over W=1, Δ = 0.5.
        // threshold=0.7 → 0.5 < 0.7 → Rollback (strictly-better requirement).
        let q = StepAttributionQualifier::with_threshold(IdentityExecutor, SumAggregator, 0.7);
        let verdict = q.qualify(&1.0, &AddConst(0.5), &[()]);
        assert_eq!(
            verdict,
            QualificationVerdict::Rollback {
                delta_below_threshold: true
            }
        );
    }

    #[test]
    fn t22_strict_threshold_accepts_clear_gain() {
        // baseline=1.0, candidate=1.0+0.5=1.5 over W=2, Δ = 1.0.
        // threshold=0.7 → 1.0 ≥ 0.7 → Commit.
        let q = StepAttributionQualifier::with_threshold(IdentityExecutor, SumAggregator, 0.7);
        let verdict = q.qualify(&1.0, &AddConst(0.5), &[(), ()]);
        assert_eq!(
            verdict,
            QualificationVerdict::Commit {
                delta_above_threshold: true
            }
        );
    }

    // ── T2.3: StepLocalizer ───────────────────────────────────────────────

    #[test]
    fn t23_localizer_finds_first_fault_tick() {
        // 5-tick trajectory. r_k drops below tau=0.7 at tick 2.
        let deltas = vec![
            vec![0.0, 0.0], // tick 0
            vec![0.0, 0.0], // tick 1
            vec![1.0, 0.0], // tick 2 — delta points along direction 0
            vec![0.0, 0.0], // tick 3
            vec![0.0, 0.0], // tick 4
        ];
        let scores = vec![0.9, 0.8, 0.3, 0.9, 0.9];
        let directions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let site = DotProductLocalizer::new()
            .localize_and_link(&deltas, &scores, &directions, 0.7)
            .expect("should find a fault");

        assert_eq!(site.tick_idx, 2, "should localize to tick 2");
        assert_eq!(site.responsible_idx, 0, "direction 0 is responsible");
        // sigmoid(dot([1,0],[1,0])) = sigmoid(1) ≈ 0.7311
        assert!((site.responsibility[0] - stable_sigmoid(1.0)).abs() < 1e-5);
        assert!((site.responsibility[1] - stable_sigmoid(0.0)).abs() < 1e-5); // sigmoid(0)=0.5
    }

    #[test]
    fn t23_localizer_returns_none_when_all_reliable() {
        let deltas = vec![vec![1.0], vec![1.0], vec![1.0]];
        let scores = vec![0.9, 0.9, 0.9]; // all above tau=0.7
        let directions = vec![vec![1.0]];

        let result =
            DotProductLocalizer::new().localize_and_link(&deltas, &scores, &directions, 0.7);
        assert!(result.is_none(), "no fault when all ticks reliable");
    }

    #[test]
    fn t23_localizer_projection_floor_skips_weak_signals() {
        // Tick 2 has a fault (r_k < 0.7) but a weak projection (0.01 < floor 0.5).
        // Tick 3 has the same fault with a strong projection.
        let deltas = vec![
            vec![0.0],
            vec![0.0],
            vec![0.01], // weak
            vec![1.0],  // strong
        ];
        let scores = vec![0.9, 0.9, 0.3, 0.3];
        let directions = vec![vec![1.0]];

        let site = DotProductLocalizer::with_floor(0.5)
            .localize_and_link(&deltas, &scores, &directions, 0.7)
            .expect("should skip weak tick 2 and find tick 3");
        assert_eq!(site.tick_idx, 3, "should skip weak tick 2");
    }

    #[test]
    fn t23_localizer_responsibility_weights_sum_convention() {
        // Two orthogonal directions: delta = [1, 1] projects equally.
        let deltas = vec![vec![1.0, 1.0]];
        let scores = vec![0.3];
        let directions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let site = DotProductLocalizer::new()
            .localize_and_link(&deltas, &scores, &directions, 0.7)
            .expect("should find a fault");

        // Equal projections → equal weights → argmax picks the FIRST (index 0),
        // which matches the "higher-priority (lower branch_id)" convention.
        assert_eq!(site.responsible_idx, 0);
        assert!((site.responsibility[0] - site.responsibility[1]).abs() < 1e-5);
    }

    // ── T2.4: doc-test (covered by the module-level example) ─────────────
    //
    // The canonical usage example is in the module doc. This test exercises
    // the same path to catch regressions.

    #[test]
    fn t24_canonical_usage_example() {
        struct IdentityExecutor2;
        impl ReplayExecutor<f32, (), f32> for IdentityExecutor2 {
            fn replay(&self, k: &f32, inputs: &[()]) -> Vec<f32> {
                inputs.iter().map(|_| *k).collect()
            }
        }
        struct SumAgg;
        impl ScoreAggregator<f32> for SumAgg {
            fn aggregate(&self, scores: &[f32]) -> f32 {
                scores.iter().sum()
            }
        }
        struct AddC(f32);
        impl CandidateMutation<f32> for AddC {
            fn apply_to(&self, baseline: &f32) -> f32 {
                baseline + self.0
            }
        }

        let qualifier = StepAttributionQualifier::new(IdentityExecutor2, SumAgg, 0.0);
        let verdict = qualifier.qualify(&1.0, &AddC(2.0), &[(), (), ()]);
        assert_eq!(
            verdict,
            QualificationVerdict::Commit {
                delta_above_threshold: true
            }
        );
    }

    // ── Built-in aggregators ──────────────────────────────────────────────

    #[test]
    fn mean_aggregator_empty_returns_zero() {
        assert_eq!(MeanAggregator.aggregate(&[]), 0.0);
    }

    #[test]
    fn mean_aggregator_basic() {
        assert!((MeanAggregator.aggregate(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn sum_aggregator_basic() {
        assert!((SumAggregator.aggregate(&[1.0, 2.0, 3.0]) - 6.0).abs() < 1e-6);
    }

    // ── TickFaultSite constructor ─────────────────────────────────────────

    #[test]
    #[should_panic(expected = "responsible_idx out of bounds")]
    fn tick_fault_site_rejects_out_of_bounds_idx() {
        TickFaultSite::<Vec<f32>, f32>::new(0, "x", vec![0.5], 5);
    }

    // ── Phase 4 T4.1: WasmTestGateAdapter subsumption tests ──────────────
    //
    // Proves the adapter delegates correctly to StepAttributionQualifier and
    // reproduces R172's `avg_reward_delta >= 0` acceptance as a special case.

    #[test]
    fn t41_adapter_commit_when_candidate_beats_baseline() {
        let adapter = WasmTestGateAdapter::new();
        let verdict = adapter.qualify(1.0, 1.5); // Δ = +0.5
        assert_eq!(
            verdict,
            QualificationVerdict::Commit {
                delta_above_threshold: true
            }
        );
    }

    #[test]
    fn t41_adapter_commit_when_candidate_equals_baseline() {
        // R172 default threshold is 0.0; Δ=0 qualifies (≥ not >).
        let adapter = WasmTestGateAdapter::new();
        let verdict = adapter.qualify(1.0, 1.0); // Δ = 0.0
        assert_eq!(
            verdict,
            QualificationVerdict::Commit {
                delta_above_threshold: true
            }
        );
    }

    #[test]
    fn t41_adapter_rollback_when_candidate_regresses() {
        let adapter = WasmTestGateAdapter::new();
        let verdict = adapter.qualify(1.0, 0.5); // Δ = -0.5
        assert_eq!(
            verdict,
            QualificationVerdict::Rollback {
                delta_below_threshold: true
            }
        );
    }

    #[test]
    fn t41_adapter_strict_threshold_enforces_strictly_better() {
        // threshold = 0.1: candidate must beat baseline by ≥ 0.1.
        let adapter = WasmTestGateAdapter::with_threshold(0.1);
        let verdict = adapter.qualify(1.0, 1.05); // Δ = +0.05 < 0.1
        assert_eq!(
            verdict,
            QualificationVerdict::Rollback {
                delta_below_threshold: true
            }
        );
    }

    #[test]
    fn t41_adapter_subsumption_matches_direct_qualifier() {
        // The headline proof: the adapter produces bit-identical verdicts to
        // a hand-constructed StepAttributionQualifier on the same data, across
        // a sweep of (baseline, candidate) pairs covering all verdict regimes.
        let adapter = WasmTestGateAdapter::new();
        let direct = StepAttributionQualifier::new(ScalarStateExecutor, SumAggregator, 0.0);
        let cases = [
            (1.0_f32, 1.5_f32), // commit (Δ=+0.5)
            (1.0, 1.0),         // commit (Δ=0, ≥ threshold)
            (1.0, 0.5),         // rollback (Δ=-0.5)
            (0.3, 0.3),         // commit (Δ=0)
            (0.9, 0.8),         // rollback (Δ=-0.1)
            (2.0, 3.0),         // commit (Δ=+1.0)
        ];
        for (baseline, candidate) in cases {
            let adapter_verdict = adapter.qualify(baseline, candidate);
            let direct_verdict = direct.qualify(&baseline, &ReplaceScalar(candidate), &[()]);
            assert_eq!(
                adapter_verdict, direct_verdict,
                "adapter vs direct mismatch at baseline={baseline}, candidate={candidate}"
            );
        }
    }

    #[test]
    fn t41_adapter_default_matches_new() {
        // Default trait impl must agree with `new()` (R172 threshold = 0.0).
        assert_eq!(
            WasmTestGateAdapter::default().threshold,
            WasmTestGateAdapter::new().threshold
        );
        assert_eq!(WasmTestGateAdapter::default().threshold, 0.0);
    }
}
