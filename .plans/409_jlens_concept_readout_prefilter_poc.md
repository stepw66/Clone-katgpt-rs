# Plan 409: Jacobian-SVD Concept Readout PoC — Faithfulness Pre-Filter Defend-Wrong

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/388_jacobian_lens_single_layer_concept_readout.md](../.research/388_jacobian_lens_single_layer_concept_readout.md)
**Source paper:** [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html) — Gurnee, Sofroniew, Lindsey et al., "Verbalizable Representations Form a Global Workspace in Language Models" (Anthropic, 2026-07-06)
**Target:** `riir-ai/crates/riir-poc/` (defend-wrong PoC crate per research skill §3.6)
**Status:** Active — Phase 1 COMPLETE; Phase 2 (latency bench) next.

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

- [x] **T1.1** Add new module `riir-ai/crates/riir-poc/src/jlens_poc.rs` exposing:
  - `LinearFaithfulConsumer { w: Vec<Vec<f32>> }` — implements the consumer as a closure `Fn(&[f32], &mut [f32])` so `jacobian_svd_at_into` can call it.
  - `SigmoidFaithfulConsumer { w: Vec<Vec<f32>> }` — same, with per-output sigmoid.
  - `GroundTruthVerdict` enum (`Faithful | Unfaithful`) computed by construction (known row-space membership for regime 1) or by `FaithfulnessProbe` (regimes 2 and 3).
  - `PrefilterVerdict` enum (`Accept | Reject`) computed by the Jacobian-SVD projection.
  - `run_trial(consumer, x, memory, k, rho, scratch, probe, rng) -> TrialResult` — runs all three strategies on one trial and returns per-strategy latency + verdict.
- [x] **T1.2** Wire the module into `riir-ai/crates/riir-poc/src/lib.rs` with `pub mod jlens_poc;`.
- [x] **T1.3** Add a `#[test]` in `jlens_poc.rs` that runs one trial of each strategy on a linear-faithful consumer with `n=8, m=8, rank=4`, target memory in the row-space. Assert: Strategy A says Faithful, Strategy B says Accept, latencies are in expected ranges (A ≥ 10 µs, B ≤ 2 µs on this small case).
- [x] **T1.4** Isolated build check: `CARGO_TARGET_DIR=/tmp/plan409_p1 cargo check -p riir-poc --features faithfulness_probe` (the FaithfulnessProbe feature must be enabled). Clean up `/tmp/plan409_p1` when done per AGENTS.md.

### Phase 1 results

**All 5 tests pass.** Phase 1 surfaced three findings that adjust Phase 3's design:

1. **Gram-Schmidt bug fixed.** `make_memory_orthogonal_to_rowspace` was over-subtracting (used `dot * row` instead of `(dot / ‖row‖²) * row` since rows are scaled by singular values). Fixed; the orthogonal memory now genuinely projects to ~0 onto the top-k right singular vectors.

2. **KEY FINDING — nullspace memories are Faithful per the probe, NOT Unfaithful.** A memory in the nullspace of W produces `W·mem = 0` at the current point, but the probe's shuffle/corrupt interventions break the nullspace structure (shuffling moves components into the rowspace), so the consumer reacts and the probe says Faithful. The pre-filter correctly says Reject (projection ~0), which is a **false negative** vs the probe. This is exactly the risk Research 388 §2.5 Fusion A flagged. The test `phase1_finding_nullspace_memory_is_faithful_per_probe` records this.

3. **Implication: the pre-filter and the probe measure DIFFERENT things.** The pre-filter asks "is this direction in the local principal subspace?" (representational); the probe asks "does perturbing this memory change behavior?" (causal). For any consumer that READS its memory, the probe considers almost every non-degenerate memory Faithful. Genuinely Unfaithful verdicts require a consumer that structurally ignores the memory (the new `ConstantConsumer`). This narrows Fusion A's value proposition: the pre-filter's speedup only materializes on the `ConstantConsumer` class (structurally ignored memories), not on nullspace-of-active-consumer memories.

4. **Latency via the generic bridge path is ~890 µs** (allocations per Jacobian column eval). Phase 2's criterion bench must use the native `eval_into` path to measure the true ~455 ns hot-path cost. The `run_trial` bridge is for correctness, not perf.

5. **Degenerate-case handling added.** `prefilter_verdict` now rejects when all top-k singular values are negligible (zero Jacobian → constant consumer), so the `ConstantConsumer` case is correctly rejected.

These findings do NOT refute Fusion A — they sharpen it. The pre-filter is useful for the `ConstantConsumer` class (structurally ignored memories, where the probe wastes ~1 ms to return Unfaithful). Phase 3's quality bench must include a regime that produces a meaningful fraction of such memories to quantify the speedup. The nullspace-of-active-consumer regime produces false negatives and should NOT be counted as a speedup win.

**Phase 1 unblocks Phase 2** by confirming the two public primitives compose. If composition fails (e.g. `FaithfulnessProbe::faithfulness_profile` signature mismatch, or `jacobian_svd_at_into` requires a specific closure shape the consumer cannot provide), Phase 1 surfaces it before any bench harness is written.

---

## Phase 2 — Latency bench (architectural + latency claim)

Criterion bench measuring per-strategy latency on a grid of `(n, m)` sizes. Confirms the architectural claim (the wiring works at sub-µs for Strategy B) and the latency claim (B is ~100×–2000× cheaper than A).

### Tasks

- [x] **T2.1** Add `riir-ai/crates/riir-poc/benches/jlens_concept_readout_goat.rs` modeled on `adajepa_modelless_goat.rs`:
  - A `latency_comparison` group: bench Strategy A (`FaithfulnessProbe::faithfulness_profile`), Strategy B (`jacobian_svd_at_into` + projection), Strategy C (B then conditional A) on `(n, m) ∈ {(4,4), (8,8), (16,16), (8,16), (16,8)}`.
  - Use `JacobianSvdScratch::with_capacity(n, m)` reused across iterations (one scratch per bench iteration batch, not per call).
  - Sample size and measurement time tuned per `adajepa_modelless_goat.rs` precedent.
- [x] **T2.2** Print a latency table at end of bench: per `(n, m)`, the per-call ns for each strategy and the speedup ratio B/A and C/A.
- [x] **T2.3** Run: `CARGO_TARGET_DIR=/tmp/plan409_p2 cargo bench -p riir-poc --bench jlens_concept_readout_goat --features katgpt-core/faithfulness_probe -- --quiet`. Capture the latency table output. Clean up `/tmp/plan409_p2`.
- [x] **T2.4** Record the latency numbers below. **LATENCY GATE: FAILED.** Strategy B is 10-70× SLOWER than Strategy A (not faster); B/A "speedup" is < 0.1× at every size. Per the gate-fail protocol: halt — Phase 3 will NOT run (no point measuring quality when the prefilter is slower than the probe it's supposed to shortcut). Research 388 revised to **Refuted (no GOAT path)**; `.issues/043_*` opened to track the SVD-perf root cause.

### Phase 2 results — LATENCY GATE FAILED

**Verdict: HALT.** The latency gate fails decisively. Strategy B (Jacobian-SVD prefilter) is **10–70× slower** than Strategy A (the causal probe) at every size — the complete inversion of the ~2000× speedup claim. Phase 3 (quality bench) does not run; there is no point measuring false-negative rates on a prefilter that costs more than the probe it gates.

#### Criterion medians (per-strategy, sample_size=100, measurement_time=3s)

| (n, m) | Strategy A (probe) | Strategy B (prefilter) | Strategy C (B+A) | B vs A |
|---|---|---|---|---|
| (4, 4) | 281 ns | 736 ns | 1.02 µs | **2.6× slower** |
| (8, 8) | 445 ns | 31.3 µs | 31.5 µs | **70× slower** |
| (16, 16) | 782 ns | 181 µs | 190 µs | **232× slower** |
| (8, 16) | 585 ns | 48.1 µs | 49.8 µs | **82× slower** |
| (16, 8) | 570 ns | 126 µs | 131 µs | **220× slower** |

#### Gate check (T2.4)

| Gate | Criterion | Actual | Result |
|---|---|---|---|
| B ≤ 1 µs at (8, 8) | ≤ 1000 ns | **31,254 ns** | **FAIL** (31× over) |
| B/A speedup ≥ 100× | ≥ 100× | **0.014×** (B is 70× slower) | **FAIL** |

#### Root cause analysis (diagnostic bench, since deleted)

A diagnostic bench isolated the latency gap's components by calling `jacobian_svd_at_into` at (8, 8) with three different `f` closures:

| `f` closure | Latency | What it measures |
|---|---|---|
| Identity (`f(x)=x`) | **417 ns** | SVD cost only (identity matrix converges immediately) — matches the Plan 301 docstring's ~455 ns claim |
| Flat `Vec<f32>` linear map | **3.9 µs** | Jacobian forward-diff (9 eval calls) + SVD of full-rank 8×8 matrix |
| `Vec<Vec<f32>>` linear map (PoC layout) | **4.0 µs** | Same as flat — layout is NOT the bottleneck |
| Rank-4 linear map via `prefilter_verdict` | **31 µs** | Jacobian + SVD of rank-deficient matrix (null-space convergence is slow) |

**Three findings, each independently fatal to the latency claim:**

1. **The Plan 301 docstring's ~455 ns claim was measured on a trivial `f` (identity), not a realistic linear map.** The SVD of the identity matrix converges in one Jacobi sweep (~417 ns). For a realistic linear map `f(x) = W·x`, the Jacobian computation (n+1 = 9 eval calls at n=8) plus the SVD of the resulting 8×8 matrix costs **~3.9 µs** — 8.6× the docstring claim. The docstring (`subspace_phase_gate.rs:426-429`) is misleading and should be corrected. **Issue 043 tracks this.**

2. **Rank-deficient matrices make the one-sided Jacobi SVD dramatically slower.** The rank-4 matrix produced by `make_rank_r_matrix` (which mirrors real-world consumers — HLA has rank 5 in a 64-dim space; NeuronShard has rank ≪ ambient dimension) triggers 8× more SVD work than the full-rank case. Root cause: null-space column pairs with norms hovering just above the `col_floor_sq` threshold (`frob_sq · tol² ≈ 3e-13`) pass the convergence check, apply spurious noise rotations every sweep, and prevent early termination — potentially hitting `max_sweeps = 60`. **Issue 043 tracks this as a perf bug in the shipped primitive.**

3. **Strategy A (the causal probe) is cheaper than expected for linear consumers.** The original analysis assumed ~1 ms (a complex neural-network forward pass per intervention). For `LinearFaithfulConsumer` at (8, 8), each of the probe's 5 interventions is one 64-MAC matrix-vector multiply — total ~445 ns. The probe is fast because the consumer is simple. For a complex consumer (real NN), the probe would indeed be ~1 ms, but the SVD would also be expensive (n+1 forward passes ≈ 9 × NN_forward_pass), so the prefilter never wins.

**The fundamental latency math refutes the claim:** Strategy B requires n+1 eval calls (forward-difference Jacobian) + an SVD. Strategy A requires 5 eval calls (interventions). For any n ≥ 4, Strategy B does MORE eval calls than Strategy A, plus an SVD on top. The prefilter is structurally incapable of being cheaper than the probe for n ≥ 4, regardless of consumer complexity. The ~2000× speedup claim compared the wrong two numbers (455 ns trivial-SVD vs 1 ms complex-probe) — apples to oranges.

#### What this means for Fusion A

Fusion A (Jacobian-SVD pre-filter for `FaithfulnessProbe`) is **refuted as a latency optimization**. The prefilter costs more than the probe it gates, at every size tested. The architectural insight (SVD right singular vectors represent principal directions) remains true but has no latency value as a pre-filter.

Fusion B (Percepta weight verification) and Fusion C (adaptive HLA readout) are unaffected by this result — they do not claim a latency win. They remain documented in Research 388 as deferred / not-pursued respectively.

---

## Phase 3 — Quality bench (defend-wrong, the gate that decides the verdict)

Head-to-head false-negative rate measurement on the three regimes. This is the phase that **defends or refutes** Research 388's GOAT-conditional verdict.

**STATUS: NOT RUN.** Phase 2's latency gate failed (the prefilter is 10-70× slower than the probe). There is no point measuring false-negative rates on a prefilter that costs more than the probe it gates. Phase 3 is cancelled per the T2.4 halt protocol.

### Tasks

- [-] **T3.1** Extend `jlens_poc.rs` with `Regime::{LinearFaithful, SigmoidFaithful, AdversarialUnfaithful}` and `make_trials(regime, n_trials, seed) -> Vec<TrialSpec>`. Each `TrialSpec` carries: the consumer, the input `x`, the target memory direction, and the ground-truth verdict (by construction for regime 1; by `FaithfulnessProbe` for regimes 2 and 3, run once at PoC-init time and cached).
- [-] **T3.2** Implement the threshold sweep: for each `(regime, k, ρ)` in the grid, run `n_trials = 1000` trials and record:
  - `false_negatives`: trials where Strategy B said Reject but Strategy A said Faithful.
  - `false_positives`: trials where Strategy B said Accept but Strategy A said Unfaithful (less critical — false positives just trigger the probe, which catches them; but report for completeness).
  - `mean_strategy_b_latency_ns`, `mean_strategy_c_latency_ns`.
- [-] **T3.3** Print a verdict table per regime: rows = `(k, ρ)`, columns = `false_negative_rate`, `false_positive_rate`, `mean_b_latency_ns`, `mean_c_latency_ns`, `verdict` (one of: `PASS` if FN rate below target AND C/A speedup ≥ 10×; `FAIL_FN` if FN rate above target; `FAIL_SPEEDUP` if C/A speedup < 10×).
- [-] **T3.4** Run: `CARGO_TARGET_DIR=/tmp/plan409_p3 cargo bench -p riir-poc --bench jlens_concept_readout_goat --features faithfulness_probe -- --quiet`. Capture the verdict tables. Clean up `/tmp/plan409_p3`.
- [-] **T3.5** Record the verdict tables in this plan's §"Phase 3 results" section below.

### Phase 3 results

**NOT RUN — Phase 2 latency gate halted the plan.** See Phase 2 results for the root cause.

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

- [x] **T4.1** Write the §"PoC Addendum" to `katgpt-rs/.research/388_*.md` recording:
  - Which claim types were confirmed (architectural) and which were refuted (latency).
  - The raw numbers from Phase 2 (Phase 3 did not run).
  - No best operating point `(k*, ρ*)` exists — the latency math is structurally unfavorable for n ≥ 4.
- [x] **T4.2** Update Research 388's verdict per the PoC outcome:
  - **Phase 2 latency gate failed.** Revised Research 388's verdict from "Gain → GOAT-conditional" to **Refuted (Fusion A latency path)**. The latency claim (B ~2000× cheaper than A) is false; the architectural insight (SVD principal directions) remains true but has no latency value as a pre-filter.
  - Opened `.issues/043_jacobian_svd_perf_docstring_and_rank_deficient_regression.md` tracking: (a) the misleading ~455 ns docstring claim in `subspace_phase_gate.rs`, (b) the rank-deficient matrix SVD perf regression (8× slower than full-rank), (c) whether a better SVD algorithm or a rank-deficiency fast-path could recover Fusion A's latency claim.
- [x] **T4.3** Leave the PoC in place as a permanent regression check. The latency bench (`jlens_concept_readout_goat.rs`) stays in `riir-poc/benches/` and will catch any SVD perf regression that changes the verdict.
- [x] **T4.4** Commit all PoC code, the updated Research 388, and this plan on `develop` per global AGENTS.md (commit prefix `docs:` for the research note + plan, `feat:` if Phase 4.2 promotes to GOAT and lands code). Do NOT push.
  - **DONE.** Substantive commits already landed in prior sessions: `838837f4` (riir-ai, `feat:` — PoC scaffolding code) and `d11edcf0` (katgpt-rs, `docs:` — Plan 409 + updated Research 388 refuting Fusion A + Issue 043). Working tree verified clean for all Plan 409 files in both repos. Phase 4.2 did NOT promote to GOAT (Phase 2 latency gate FAILED), so no `feat:` promotion commit was needed. This checkbox flip closes the plan.

---

## Notes

- **No feature flag in katgpt-rs yet.** This plan does not add `concept_readout` to `katgpt-core`'s `Cargo.toml`. The primitive lands only if Phase 4.2 promotes.
- **Isolated builds.** Each phase uses its own `CARGO_TARGET_DIR=/tmp/plan409_pN` per global AGENTS.md to avoid locking the main `target/` during iteration. Clean up after each phase.
- **No `git stash` of unrelated files.** Per global AGENTS.md, only commit files this plan touched. If a parallel agent has WIP in `riir-poc`, commit this plan's PoC files specifically (`git add riir-ai/crates/riir-poc/src/jlens_poc.rs riir-ai/crates/riir-poc/benches/jlens_concept_readout_goat.rs riir-ai/crates/riir-poc/src/lib.rs katgpt-rs/.research/388_*.md katgpt-rs/.plans/409_*.md`), not `git add -A`.
- **riir-train routing note.** The paper's counterfactual reflection training is deferred to `riir-train/.research/` (next slot) as a one-line note — it is not needed for this PoC (the PoC is fully modelless). Tracked in Research 388 §4.

---

## TL;DR

**HALTED at Phase 2 — latency gate FAILED.** The Jacobian-SVD prefilter (Strategy B) is **10–70× slower** than the causal probe (Strategy A) at every size tested — the complete inversion of the ~2000× speedup claim. The root cause is fundamental: Strategy B requires n+1 eval calls (forward-difference Jacobian) + an SVD, while Strategy A requires only 5 eval calls (interventions). For n ≥ 4, the prefilter structurally cannot be cheaper than the probe. Phase 3 (quality bench) did not run. Research 388 revised to **Refuted (Fusion A latency path)**; Issue 043 opened to track the SVD perf gap and the misleading docstring. The PoC bench stays in `riir-poc/` as a permanent regression check.
