# Plan 381: Step-Attribution Δ-Qualification Primitive

**Date:** 2026-07-06
**Research:** [katgpt-rs/.research/381_SkillAdaptor_Step_Level_Fault_Attribution_Delta_Qualification.md](../.research/381_SkillAdaptor_Step_Level_Fault_Attribution_Delta_Qualification.md)
**Private guide:** [riir-ai/.research/313_Step_Level_Fault_Attribution_Commit_Gate_Guide.md](../../riir-ai/.research/313_Step_Level_Fault_Attribution_Commit_Gate_Guide.md)
**Runtime wiring:** [riir-ai/.plans/313_step_attribution_branch_wiring.md](../../riir-ai/.plans/313_step_attribution_branch_wiring.md)
**Source paper:** [arXiv:2606.01311](https://arxiv.org/abs/2606.01311) — SkillAdaptor, Yu et al. 2026
**Target:** `crates/katgpt-pruners/src/step_attribution_qualifier.rs` (new module) + Cargo feature `step_attribution_qualifier`
**Status:** Active — Phase 1 next

---

## Goal

Ship a generic modelless primitive that takes a failed trajectory, a candidate state `K+`, a baseline state `K`, and a replay window, and returns an accept/reject verdict based on a Δ-based re-execution comparison (SkillAdaptor eq. 8). This is the explicit "did this update actually help?" gate that TrajectoryDoctor (Plan 223, ships `localize_failure`) and the existing WasmTestGate Δ-field do not unify. Sibling to TrajectoryDoctor in `katgpt-pruners`. Opt-in feature flag; **promote to default ONLY after the riir-ai Plan 313 G6 quality-parity PoC passes** (per §3.6 defend-wrong rule).

The primitive is generic (no game semantics): it works on any candidate/baseline pair where (a) the candidate is a deterministic function of the baseline + a proposed mutation, and (b) the trajectory can be re-executed deterministically given the candidate/baseline state.

## Stack slot (per the per-stack promote/demote discipline)

This primitive does NOT fit a transformer stack slot (attention/KV/sampling/speculative/pruning). It sits in the **cognitive-branch / skill-lifecycle stack** alongside `TrajectoryDoctor`, `FailureEpisodeStore`, `ClosedUnitCompactionGate`. Promotion/demotion is tracked against the shipped `TrajectoryDoctor` family — if `StepAttributionQualifier` wins the variance-reduction comparison in G6, it becomes the recommended commit gate for cognitive-branch / skill-lifecycle consumers; `TrajectoryDoctor` alone remains for localization-only use cases.

## Phase 1 — Skeleton + Trait (CORE)

### Tasks

- [ ] **T1.1** Create `crates/katgpt-pruners/src/step_attribution_qualifier.rs` behind feature `step_attribution_qualifier`. Module doc references Plan 381 + Research 381 + SkillAdaptor arXiv:2606.01311 eq. 8.
- [ ] **T1.2** Define the core trait + types (generic over state `K`, replay input `I`, score `S`):

```rust
//! Step-Attribution Δ-Qualification Primitive (Plan 381, Research 381)
//!
//! Generic modelless instantiation of SkillAdaptor's eq. 8:
//!   Δ = E_q[M(q; K+)] - E_q[M(q; K)]
//! Commit candidate state K+ iff Δ >= 0.
//!
//! Sibling to [`crate::trajectory_doctor::TrajectoryDoctor`] (Plan 223).
//! TrajectoryDoctor localizes the fault; this gate decides whether to commit
//! the proposed fix.

use crate::trajectory_doctor::FailureSite;

/// A proposed mutation to a baseline state, producing a candidate state.
pub trait CandidateMutation<K> {
    /// Apply this mutation to the baseline, producing the candidate state K+.
    fn apply_to(&self, baseline: &K) -> K;
}

/// Deterministic re-execution of a replay window under a given state.
/// Implementors guarantee bit-identical results for the same (state, inputs) pair.
pub trait ReplayExecutor<K, I, S> {
    /// Re-execute the replay window `inputs` under state `k`, returning per-step scores.
    fn replay(&self, k: &K, inputs: &[I]) -> Vec<S>;
}

/// Aggregate a per-step score slice into a single comparable metric.
/// Typically `Sum` for CLR reward; may be `Mean` for normalized metrics.
pub trait ScoreAggregator<S> {
    fn aggregate(&self, scores: &[S]) -> S;
}

/// The Δ≥0 commit gate. Wraps a ReplayExecutor + ScoreAggregator.
pub struct StepAttributionQualifier<K, I, S, E, A> {
    executor: E,
    aggregator: A,
    /// Acceptance threshold (default 0.0 = SkillAdaptor's Δ ≥ 0).
    /// Positive values encode a "strictly better" requirement.
    pub threshold: S,
    _marker: core::marker::PhantomData<(K, I)>,
}

/// The verdict returned by qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationVerdict {
    /// Δ ≥ threshold — commit the candidate.
    Commit { delta_above_threshold: bool },
    /// Δ < threshold — rollback to baseline.
    Rollback { delta_below_threshold: bool },
}

impl<K, I, S, E, A> StepAttributionQualifier<K, I, S, E, A>
where
    E: ReplayExecutor<K, I, S>,
    A: ScoreAggregator<S>,
    S: PartialOrd + Copy + core::ops::Sub<Output = S>,
{
    /// Run the full Δ≥0 qualification on a candidate mutation.
    ///
    /// 1. Apply mutation to baseline → K+
    /// 2. Replay window under K → baseline_scores
    /// 3. Replay window under K+ → candidate_scores
    /// 4. Aggregate both
    /// 5. Δ = aggregate(K+) - aggregate(K); Commit iff Δ ≥ threshold
    pub fn qualify(
        &self,
        baseline: &K,
        mutation: &dyn CandidateMutation<K>,
        replay_inputs: &[I],
    ) -> QualificationVerdict {
        let candidate = mutation.apply_to(baseline);
        let baseline_scores = self.executor.replay(baseline, replay_inputs);
        let candidate_scores = self.executor.replay(&candidate, replay_inputs);
        let delta = self.aggregator.aggregate(&candidate_scores)
            - self.aggregator.aggregate(&baseline_scores);
        if delta >= self.threshold {
            QualificationVerdict::Commit { delta_above_threshold: true }
        } else {
            QualificationVerdict::Rollback { delta_below_threshold: true }
        }
    }
}
```

- [ ] **T1.3** Define the **tick-attribution** extension trait (the SkillAdaptor Localize+Link fused into one modelless call, generalizing `TrajectoryDoctor::localize_failure` from token-index to a generic tick-index):

```rust
/// A localized fault: which tick, which "responsible direction", improvement hint.
/// Generalizes [`crate::trajectory_doctor::FailureSite`] from token-index to tick-index
/// + adds the SkillAdaptor Linker's responsibility weight.
pub struct TickFaultSite<Dir, W> {
    /// The first actionable fault tick in the replay window.
    pub tick_idx: usize,
    /// The violated predicate / observation (mirrors FailureSite::violated_predicate).
    pub violated: String,
    /// Per-direction responsibility weights (SkillAdaptor eq. 6 output).
    /// `weights[j] = sigmoid(dot(hla_delta_at_t_star, direction_j))`.
    pub responsibility: Vec<W>,
    /// Argmax direction index — the "responsible skill/branch".
    pub responsible_idx: usize,
    /// Marker for the generic direction type (e.g. HLA direction vector).
    pub _marker: core::marker::PhantomData<Dir>,
}

/// SkillAdaptor's Localize (eq. 5) + Link (eq. 6) fused into one modelless call.
/// Implementors find the first fault tick in a trajectory, then project the
/// state-delta at that tick onto a direction set to attribute responsibility.
pub trait StepLocalizer<Dir, W> {
    /// Given a trajectory of per-tick state-deltas + CLR scores + a direction set,
    /// return the first actionable fault + responsibility weights.
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
```

- [ ] **T1.4** Wire the feature flag into `crates/katgpt-pruners/Cargo.toml` + `lib.rs` module re-export. Verify `cargo check -p katgpt-pruners --features step_attribution_qualifier` is clean.
- [ ] **T1.5** Update root `Cargo.toml` passthrough feature (mirror the `attention_matching` / `closed_unit_compaction` pattern). Verify `cargo check --features step_attribution_qualifier` is clean.

## Phase 2 — Tests + Doc Example

### Tasks

- [ ] **T2.1** Unit tests for the Δ≥0 gate: candidate beats baseline → `Commit`; candidate equals baseline → `Commit` (Δ=0 ≥ 0); candidate worse → `Rollback`. Use a deterministic toy `ReplayExecutor` (e.g. `SumExecutor` over `&[f32]`).
- [ ] **T2.2** Unit tests for the threshold variant: with `threshold = 0.1`, candidate-with-Δ=0.05 → `Rollback` (strictly-better requirement).
- [ ] **T2.3** Unit tests for `StepLocalizer`: synthetic trajectory with a known fault tick + responsibility weights; assert `localize_and_link` returns the correct `tick_idx` + `responsible_idx`.
- [ ] **T2.4** Doc-test example in the module doc showing the canonical usage: `let verdict = qualifier.qualify(&baseline_k, &mutation, &replay_window); match verdict { Commit => commit(), Rollback => rollback() }`.
- [ ] **T2.5** Run `cargo test -p katgpt-pruners --features step_attribution_qualifier --lib` — all green.

## Phase 3 — Latency Bench (G4)

### Tasks

- [ ] **T3.1** Add a criterion bench `crates/katgpt-pruners/benches/step_attribution_qualifier.rs` measuring `qualify()` latency for W=16/32/64/128 replay windows with a no-op `ReplayExecutor` (isolates gate overhead from executor overhead).
- [ ] **T3.2** Document the sub-ms gate-overhead target (excluding executor). Gate overhead = aggregate + compare + branch; should be < 1µs for W=64.
- [ ] **T3.3** Record results in `katgpt-rs/.benchmarks/381_step_attribution_qualifier_goat.md` (defer creation until bench runs).

## Phase 4 — ClosedUnitCompactionGate + WasmTestGate Adapter Shims (optional)

### Tasks

- [ ] **T4.1** Ship an adapter impl showing `StepAttributionQualifier` subsumes the `WasmTestGate::avg_reward_delta` pattern from R172 ITSE — a `WasmTestGateAdapter` that wraps the existing `WasmTestGate` field as a `StepAttributionQualifier` with `threshold = 0.0`. Document in module doc that R172's gate is the prior art for new-pruner registration; this primitive generalizes to all candidate/baseline pairs.
- [ ] **T4.2** **DEFERRED** — adapter for `ClosedUnitCompactionGate` (Plan 333). CUCG's rubric is structurally different (predicate-list + FireRule, not Δ-comparison); the adapter would translate a rubric verdict into a ScoreAggregator output. Out of scope for Phase 1-3; track in `.issues/` if the G6 PoC (riir-ai Plan 313) shows CUCG-style rubrics would benefit from Δ unification.

## Phase 5 — Promotion Gate (BLOCKED on riir-ai Plan 313 G6)

### Tasks

- [ ] **T5.1** **BLOCKED** — await riir-ai Plan 313 Phase 5 G6 quality-parity PoC. Per §3.6, the open primitive's promotion to default-on requires the private runtime PoC to confirm the modelless instantiation reproduces SkillAdaptor's variance reduction. Until then, `step_attribution_qualifier` stays opt-in.
- [ ] **T5.2** If G6 PASSES: open katgpt-rs Issue for promotion (mirror the `committed_field_blend` Issue 005 pattern), flip Cargo.toml default, update README + research note + guide.
- [ ] **T5.3** If G6 REFUTES parity: record raw PoC numbers as a §"PoC Addendum" in Research 381; the primitive stays opt-in; the architectural + latency claims stand; the quality claim is a tracked follow-up.

## GOAT Gate Summary

| Gate | Evidence | Status |
|------|----------|--------|
| **G1** Correctness (Δ≥0 logic; localize_and_link returns correct fault) | Phase 2 unit tests | ⬜ pending Phase 2 |
| **G2** Quality-parity (reproduces SkillAdaptor ±8.1→±5.2 variance reduction) | riir-ai Plan 313 Phase 5 G6 PoC | ⬜ BLOCKED — mandatory before default-on |
| **G3** No-regression (feature off = byte-identical to develop) | Phase 2 + 4 isolation tests | ⬜ pending |
| **G4** Perf (gate overhead < 1µs at W=64, excluding executor) | Phase 3 bench | ⬜ pending Phase 3 |
| **G5** Modelless (no riir-train/riir_gpu/backprop dep) | Cargo.toml dep audit | ✅ by construction (no new deps) |
| **G6** Feature-isolation (single-feature + all-features check clean) | `cargo check --features step_attribution_qualifier` + `--all-features` | ⬜ pending Phase 1 |

## Failure Mode

If the trait shape (especially `CandidateMutation::apply_to` returning `K` by value) proves too rigid for the riir-ai consumer (which needs `&mut BranchBank` in-place mutation for hot-path compatibility):
1. Add a parallel `InPlaceCandidateMutation<K>` trait with `fn apply_in_place(&self, baseline: &mut K)`.
2. Add `qualify_in_place()` method mirroring `qualify()` but using the in-place trait.
3. Do NOT remove the by-value trait — it's the cleaner API for generic consumers.

Track in `.issues/` if this surfaces during riir-ai Plan 313 Phase 1 wiring.
