# Plan 409: Jacobian-SVD Concept Readout PoC — Faithfulness Pre-Filter Defend-Wrong

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/388_jacobian_lens_single_layer_concept_readout.md](../.research/388_jacobian_lens_single_layer_concept_readout.md)
**Source paper:** [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html) — Gurnee, Sofroniew, Lindsey et al., "Verbalizable Representations Form a Global Workspace in Language Models" (Anthropic, 2026-07-06)
**Target:** `riir-ai/crates/riir-poc/` (defend-wrong PoC crate per research skill §3.6)
**Status:** Active — Phase 1 (PoC scaffolding)

---

## Goal

Settle the **Gain → GOAT-conditional** verdict from Research 388 with a **defend-wrong PoC**. The verdict claims that a ~455 ns Jacobian-SVD "concept readout" can serve as a representational pre-filter before the existing ~1 ms `FaithfulnessProbe` causal probe (Research 388 §2.5 Fusion A), giving a ~2000× speedup on the rejection path.

Per research skill §3.6, this is exactly the claim type that needs a PoC, not architectural reasoning:

| Claim type | Example | Proof required |
|---|---|---|
| Architectural ("the runtime analog exists") | "the Jacobian SVD math ships via Plan 301" | grep + read the code — ✓ done in Research 388 |
| Latency / resource ("modelless, sub-µs") | "pre-filter cost is ~455 ns" | criterion bench — Phase 2 below |
| **Quality ("no missed faithfulness violations")** | "the pre-filter has false-negative rate < τ" | **head-to-head PoC on a controlled toy benchmark — Phase 3 below** |

The PoC's job is to **defend or refute** the quality claim. If the PoC refutes it (false-negative rate too high at any useful pre-filter threshold), the verdict is honestly revised per §3.6: latency claim stands, quality claim becomes a tracked follow-up in `.issues/`.

## What ships

Nothing in `katgpt-rs`. **No production code, no feature flag, no katgpt-rs changes.** The PoC lives entirely in `riir-ai/crates/riir-poc/` and imports `jacobian_svd_at_into` + `FaithfulnessProbe` from `katgpt-core` (already public). The PoC becomes a permanent regression check in `riir-poc` — if Research 388's primitive ever ships in `katgpt-core`, the PoC keeps settling whether the production version still matches the bench.

If the PoC **passes** the quality gate, a separate implementation plan (next slot after this PoC's verdict) opens to land the primitive in `katgpt-core/faithfulness/concept_readout.rs` behind the `concept_readout` feature flag.

## Defend-wrong design

Three competitors, head-to-head on a controlled toy domain. Same domain, same ground truth, same decision criterion.

### Competitors

| Strategy | Cost | Description |
|---|---|---|
| **A. `FaithfulnessProbe` only** (baseline / ground truth) | ~1 ms | The shipped causal intervention probe. Calls `faithfulness_profile(memory, &mut rng)` and reads `is_faithfully_used(threshold)`. This IS the ground truth — by construction in Plan 278, a memory that fails this probe is one the consumer structurally ignores. |
| **B. Jacobian-SVD pre-filter only** (the PoC's target) | ~455 ns | Compute `jacobian_svd_at_into(consumer_as_map, x, eps, scratch)`. Project the target memory direction onto the top-k right singular vectors. Verdict: faithful iff projection magnitude ≥ ρ·‖memory‖ (or iff target direction is inside the top-k row-space at all). |
| **C. Pre-filter + probe (the production wiring)** | ~455 ns + (~1 ms only on accept) | Strategy B as a pre-filter; if B accepts, run Strategy A to confirm. Measures end-to-end speedup vs Strategy A alone AND end-to-end false-negative rate vs Strategy A alone. |

### Toy domain

A controlled synthetic consumer that is **provably either faithful or unfaithful** to a target memory direction, so the ground truth is bit-identical to Strategy A's verdict by construction. Three regimes:

1. **Linear-faithful** — `consumer(x) = W·x` where `W` has known rank r; target memory direction `m` is faithful iff `m` is in the row-space of `W`. Ground truth trivially true. This is the easy case — Strategy B should be exact.
2. **Nonlinear-faithful** — `consumer(x) = σ(W·x)` (sigmoid nonlinearity). Target direction `m` is faithful iff `m` is in the row-space of the *local* Jacobian at the current point `x`. Ground truth: Strategy A on the same `x`. This is the realistic case — Strategy B's local linearization may miss routes that only activate under perturbation.
3. **Adversarial unfaithful** — `consumer(x) = σ(W·x)` where `m` is *deliberately* in the row-space of `W` at low singular value (small but nonzero route). Tests whether Strategy B's threshold ρ catches small-but-real routes. The risk case from Research 388 §2.5 Fusion A.

### Metrics

| Metric | Target (PoC passes) | Refutes |
|---|---|---|
| **Per-decision latency** (Strategy B vs A) | B ≤ 1 µs, A ≥ 100 µs (i.e. ≥100× speedup, ideally ~2000×) | B > 10 µs (no meaningful speedup) |
| **False-negative rate** (Strategy B vs A ground truth) | < 1% on regime 1; < 5% on regime 2; < 10% on regime 3 | > 5% on regime 1 OR > 20% on regime 2 (pre-filter misses too many real violations) |
| **End-to-end speedup** (Strategy C vs A) | ≥ 10× at the target ρ that hits the false-negative targets above | < 2× (pre-filter rarely rejects, so probe always runs) |
| **Latency floor** (Strategy B on n=8, m=8) | < 1 µs (matches Plan 301's ~455 ns baseline + projection overhead) | > 5 µs |

The thresholds above are the PoC's **defend-wrong gates**. They are deliberately chosen so the pre-filter must be both *fast* and *accurate enough* to be worth wiring in production. A pre-filter that is fast but misses 30% of violations is useless; a pre-filter that is accurate but only 2× faster than the probe is not worth the complexity.

### Threshold sweep

The PoC must sweep the pre-filter threshold ρ ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} and the top-k cutoff k ∈ {1, 2, 4, 8} (where k ≤ min(n, m)) to find the operating point that minimizes false-negative rate at acceptable speedup. The verdict table reports the best operating point, not just one fixed threshold — defending the wrong case means showing there exists *no* operating point that hits the gates, not just that one cherry-picked threshold fails.

---

## Phase 1 — PoC scaffolding (unblock)

Minimal skeleton that imports both primitives and runs one trial of each strategy on the linear-faithful regime. Goal: prove the wiring compiles and the ground truth is what we think it is.

### Tasks

- [ ] **T1.1** Add new module `riir-ai/crates/riir-poc/src/jlens_poc.rs` exposing:
  - `LinearFaithfulConsumer { w: Vec<Vec<f32>> }` — implements the consumer as a closure `Fn(&[f32], &mut [f32])` so `jacobian_svd_at_into` can call it.
  - `SigmoidFaithfulConsumer { w: Vec<Vec<f32>> }` — same, with per-output sigmoid.
  - `GroundTruthVerdict` enum (`Faithful | Unfaithful`) computed by construction (known row-space membership for regime 1) or by `FaithfulnessProbe` (regimes 2 and 3).
  - `PrefilterVerdict` enum (`Accept | Reject`) computed by the Jacobian-SVD projection.
  - `run_trial(consumer, x, memory, k, rho, scratch, probe, rng) -> TrialResult` — runs all three strategies on one trial and returns per-strategy latency + verdict.
- [ ] **T1.2** Wire the module into `riir-ai/crates/riir-poc/src/lib.rs` with `pub mod jlens_poc;`.
- [ ] **T1.3** Add a `#[test]` in `jlens_poc.rs` that runs one trial of each strategy on a linear-faithful consumer with `n=8, m=8, rank=4`, target memory in the row-space. Assert: Strategy A says Faithful, Strategy B says Accept, latencies are in expected ranges (A ≥ 10 µs, B ≤ 2 µs on this small case).
- [ ] **T1.4** Isolated build check: `CARGO_TARGET_DIR=/tmp/plan409_p1 cargo check -p riir-poc --features faithfulness_probe` (the FaithfulnessProbe feature must be enabled). Clean up `/tmp/plan409_p1` when done per AGENTS.md.

**Phase 1 unblocks Phase 2** by confirming the two public primitives compose. If composition fails (e.g. `FaithfulnessProbe::faithfulness_profile` signature mismatch, or `jacobian_svd_at_into` requires a specific closure shape the consumer cannot provide), Phase 1 surfaces it before any bench harness is written.

---

## Phase 2 — Latency bench (architectural + latency claim)

Criterion bench measuring per-strategy latency on a grid of `(n, m)` sizes. Confirms the architectural claim (the wiring works at sub-µs for Strategy B) and the latency claim (B is ~100×–2000× cheaper than A).

### Tasks

- [ ] **T2.1** Add `riir-ai/crates/riir-poc/benches/jlens_concept_readout_goat.rs` modeled on `adajepa_modelless_goat.rs`:
  - A `latency_comparison` group: bench Strategy A (`FaithfulnessProbe::faithfulness_profile`), Strategy B (`jacobian_svd_at_into` + projection), Strategy C (B then conditional A) on `(n, m) ∈ {(4,4), (8,8), (16,16), (8,16), (16,8)}`.
  - Use `JacobianSvdScratch::with_capacity(n, m)` reused across iterations (one scratch per bench iteration batch, not per call).
  - Sample size and measurement time tuned per `adajepa_modelless_goat.rs` precedent.
- [ ] **T2.2** Print a latency table at end of bench: per `(n, m)`, the per-call ns for each strategy and the speedup ratio B/A and C/A.
- [ ] **T2.3** Run: `CARGO_TARGET_DIR=/tmp/plan409_p2 cargo bench -p riir-poc --bench jlens_concept_readout_goat --features faithfulness_probe -- --quiet`. Capture the latency table output. Clean up `/tmp/plan409_p2`.
- [ ] **T2.4** Record the latency numbers in this plan's §"Phase 2 results" section below (fill in after T2.3). **Latency gate:** Strategy B must be ≤ 1 µs at `(8, 8)`; speedup ratio B/A must be ≥ 100×. If latency gate fails, halt — the architectural coverage is real but the speedup claim is wrong; revise Research 388's verdict down to "Gain (no GOAT path)" and open `.issues/043_*` to track the gap.

### Phase 2 results

_(filled in after T2.3 runs)_

| (n, m) | Strategy A (ns) | Strategy B (ns) | Strategy C (ns) | B/A speedup | C/A speedup |
|---|---|---|---|---|---|
| (4, 4) | TBD | TBD | TBD | TBD | TBD |
| (8, 8) | TBD | TBD | TBD | TBD | TBD |
| (16, 16) | TBD | TBD | TBD | TBD | TBD |
| (8, 16) | TBD | TBD | TBD | TBD | TBD |
| (16, 8) | TBD | TBD | TBD | TBD | TBD |

---

## Phase 3 — Quality bench (defend-wrong, the gate that decides the verdict)

Head-to-head false-negative rate measurement on the three regimes. This is the phase that **defends or refutes** Research 388's GOAT-conditional verdict.

### Tasks

- [ ] **T3.1** Extend `jlens_poc.rs` with `Regime::{LinearFaithful, SigmoidFaithful, AdversarialUnfaithful}` and `make_trials(regime, n_trials, seed) -> Vec<TrialSpec>`. Each `TrialSpec` carries: the consumer, the input `x`, the target memory direction, and the ground-truth verdict (by construction for regime 1; by `FaithfulnessProbe` for regimes 2 and 3, run once at PoC-init time and cached).
- [ ] **T3.2** Implement the threshold sweep: for each `(regime, k, ρ)` in the grid, run `n_trials = 1000` trials and record:
  - `false_negatives`: trials where Strategy B said Reject but Strategy A said Faithful.
  - `false_positives`: trials where Strategy B said Accept but Strategy A said Unfaithful (less critical — false positives just trigger the probe, which catches them; but report for completeness).
  - `mean_strategy_b_latency_ns`, `mean_strategy_c_latency_ns`.
- [ ] **T3.3** Print a verdict table per regime: rows = `(k, ρ)`, columns = `false_negative_rate`, `false_positive_rate`, `mean_b_latency_ns`, `mean_c_latency_ns`, `verdict` (one of: `PASS` if FN rate below target AND C/A speedup ≥ 10×; `FAIL_FN` if FN rate above target; `FAIL_SPEEDUP` if C/A speedup < 10×).
- [ ] **T3.4** Run: `CARGO_TARGET_DIR=/tmp/plan409_p3 cargo bench -p riir-poc --bench jlens_concept_readout_goat --features faithfulness_probe -- --quiet`. Capture the verdict tables. Clean up `/tmp/plan409_p3`.
- [ ] **T3.5** Record the verdict tables in this plan's §"Phase 3 results" section below.

### Phase 3 results

_(filled in after T3.4 runs)_

#### Regime 1: Linear-faithful (target FN rate < 1%)

| k | ρ | FN rate | FP rate | B latency (ns) | C latency (ns) | C/A speedup | Verdict |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

#### Regime 2: Sigmoid-faithful (target FN rate < 5%)

| k | ρ | FN rate | FP rate | B latency (ns) | C latency (ns) | C/A speedup | Verdict |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

#### Regime 3: Adversarial unfaithful (target FN rate < 10%)

| k | ρ | FN rate | FP rate | B latency (ns) | C latency (ns) | C/A speedup | Verdict |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Phase 4 — Verdict and follow-up

Synthesize Phase 2 + Phase 3 results into the final PoC verdict. Per skill §3.6, the PoC's job is to defend OR refute — both outcomes are valid; the §3.6 lesson (AdaJEPA Research 360) is that an honest refutation is more valuable than a confirmation.

### Tasks

- [ ] **T4.1** Write the §"PoC Addendum" to `katgpt-rs/.research/388_*.md` recording:
  - Which claim types were confirmed (architectural, latency) and which were defended or refuted (quality).
  - The raw numbers from Phase 2 and Phase 3.
  - The best operating point `(k*, ρ*)` if one exists.
- [ ] **T4.2** Update Research 388's verdict per the PoC outcome:
  - **If at least one `(k, ρ)` hits all three regimes' FN targets AND C/A speedup ≥ 10×:** promote Research 388 from "Gain → GOAT-conditional" to **GOAT**. Open the implementation plan (next `.plans/` slot) to land `concept_readout.rs` in `katgpt-core/faithfulness/` behind the `concept_readout` feature flag, with the PoC's `(k*, ρ*)` as default and a benchmark mirroring Phase 2 + Phase 3 as the GOAT gate.
  - **If no `(k, ρ)` hits the targets (PoC refutes the quality claim):** revise Research 388's verdict down to **Gain** (latency-only — the pre-filter is fast but not accurate enough to skip the probe). Open `.issues/043_jlens_prefilter_quality_gap.md` tracking the follow-up: investigate (a) corpus-averaged Jacobian (the paper's actual recipe, ~4-7 µs — does averaging recover the accuracy?), (b) hybrid pre-filter that accepts iff target is in top-k AND singular value ≥ ρ_abs (not just projection ratio), (c) regime-specific operating points.
- [ ] **T4.3** Leave the PoC in place as a permanent regression check. If the primitive ever ships in `katgpt-core`, the PoC will catch any regression that re-opens the quality gap.
- [ ] **T4.4** Commit all PoC code, the updated Research 388, and this plan on `develop` per global AGENTS.md (commit prefix `docs:` for the research note + plan, `feat:` if Phase 4.2 promotes to GOAT and lands code). Do NOT push.

---

## Notes

- **No feature flag in katgpt-rs yet.** This plan does not add `concept_readout` to `katgpt-core`'s `Cargo.toml`. The primitive lands only if Phase 4.2 promotes.
- **Isolated builds.** Each phase uses its own `CARGO_TARGET_DIR=/tmp/plan409_pN` per global AGENTS.md to avoid locking the main `target/` during iteration. Clean up after each phase.
- **No `git stash` of unrelated files.** Per global AGENTS.md, only commit files this plan touched. If a parallel agent has WIP in `riir-poc`, commit this plan's PoC files specifically (`git add riir-ai/crates/riir-poc/src/jlens_poc.rs riir-ai/crates/riir-poc/benches/jlens_concept_readout_goat.rs riir-ai/crates/riir-poc/src/lib.rs katgpt-rs/.research/388_*.md katgpt-rs/.plans/409_*.md`), not `git add -A`.
- **riir-train routing note.** The paper's counterfactual reflection training is deferred to `riir-train/.research/` (next slot) as a one-line note — it is not needed for this PoC (the PoC is fully modelless). Tracked in Research 388 §4.

---

## TL;DR

Defend-wrong PoC for Research 388's "Jacobian-SVD concept readout as `FaithfulnessProbe` pre-filter" claim. Three competitors (probe-only / prefilter-only / prefilter+probe) on three regimes (linear-faithful / sigmoid-faithful / adversarial-unfaithful), threshold-swept over `(k, ρ)`. Latency gate (Phase 2) confirms the ~2000× speedup. Quality gate (Phase 3) defends or refutes the false-negative claim. Verdict (Phase 4) either promotes Research 388 to GOAT (and opens the implementation plan) or honestly downgrades it to Gain (latency-only) and opens an issue tracking the accuracy gap. PoC stays in `riir-poc/` as a permanent regression check.
