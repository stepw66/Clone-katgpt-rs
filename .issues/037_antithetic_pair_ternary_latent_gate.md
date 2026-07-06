# Issue 037 — Antithetic-Pair Ternary Latent-Direction Gate (PoC follow-up)

**Filed:** 2026-07-05
**Priority:** P3 (speculative reframe — needs PoC before any verdict)
**Source paper:** [EGGROLL: Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652) — Sarkar et al., Oxford/WhiRL/MILA, Feb 2026
**Cross-ref:** `riir-train/.research/377_EGGROLL_Low_Rank_Evolution_Strategies.md` (training-side distillation)
**Status:** CLOSED (2026-07-06) — Strengthened PoC complete. Verdict: TIER DOWN. Antithetic-pair is not a universal GOAT (3/7 wins against direct-obs coherence gate). Stays as opt-in research note; no katgpt-rs plan unless a concrete scalar-score-only use case emerges. See Strengthened PoC Addendum.

---

## Problem

EGGROLL's primary contribution is a zeroth-order training algorithm (rank-r
perturbations + persistent weight updates via stochastic ascent) → correctly
routed to `riir-train/.research/377`. The modelless-side reframe was initially
dismissed as "already shipped". Re-mining found one pattern that is **not**
obviously covered by the existing runtime — but per §3.6, architectural
distinctness is necessary, not sufficient: a PASS verdict claiming "parity" or
"already covered" is only honest after a PoC.

This issue tracks the PoC that must run before any katgpt-rs verdict can be
handed down on this angle.

## The speculative reframe

EGGROLL uses an **antithetic-pair fitness shaping** to produce a ternary
decision signal:

```
sign(s⁺ − s⁻) ∈ {−1, 0, +1}
```

where `s⁺ = f(M + σE)` and `s⁻ = f(M − σE)` are paired perturbation
evaluations. The paper applies this to **persist weight updates**
(`M ← M + α·Σ Eᵢ·fitnessᵢ`) — that's training, and stays in riir-train.

The **modelless reframe**: apply the same antithetic-pair pattern to a
**latent direction vector** instead of a weight matrix. Concretely:

1. Hold a latent direction vector `d` (HLA direction, functor direction,
   neuron-shard style axis).
2. Per gate evaluation, sample a perturbation `δ` (rank-1: `δ = a·bᵀ` with
   `a, b` drawn from a counter-based RNG — same Salmon/Bradbury trick EGGROLL
   uses, so `δ` need not persist).
3. Forward-pass both `d + δ` and `d − δ` through the relevant kernel
   (HLA `evolve_hla`, functor `predict_stance`, shard `apply_delta`).
4. Compute `sign(score(d + δ) − score(d − δ)) ∈ {−1, 0, +1}`.
5. Update `d` by `±δ·lr` if the sign is nonzero — **a latent direction update,
   NOT a weight mutation** (constraint #4 allows this).

No gradient descent, no base-weight mutation. The compute cost is **2× forward
passes per gate decision** vs 1× for the coherence-threshold gate.

## Why this might be novel (Q1 — prior art check)

| Existing primitive | Gate mechanism | Distinguishes from antithetic-pair? |
|---|---|---|
| `ReestimationScheduler` (`riir-ai/.../latent_functor/reestimation.rs`) | `coherence < tau_reest AND ticks_since >= min_interval` — **single scalar threshold** | YES — no paired perturbation, no ternary direction signal |
| `cgsp_runtime` curiosity signal | Entropy-driven exploration boost | YES — entropy is single-ended, not antithetic-paired |
| `JsUniquenessTrigger` | JS divergence between peer distributions | YES — multi-peer comparison, not paired self-perturbation |
| `BoMSampler` (Plan 281) | K-hypothesis single-pass sampling | YES — samples K latents, no antithetic pairing |
| `SalienceTriGate` (Plan 303) | Three-way `d_speak/d_delegate/w_z/w_c` projection | Partial overlap — both produce a ternary decision, but SalienceTriGate projects onto static directions; antithetic-pair uses sampled perturbations |

Grepping the codebase for `antithetic|sign\(.*-.*\)|ternary.*gate` returned no
antithetic-pair-perturbation primitive. Architectural distinctness claim is
grounded in actual file reads (per §3.6), not grep hits alone.

## Why this might NOT be a GOAT (the defend-wrong angle)

1. **2× forward-pass cost per gate decision.** At 20Hz tick with thousands of
   NPCs, the coherence-threshold gate (1× pass, amortized over `min_interval_ticks`)
   may strictly dominate on latency. The antithetic-pair gate is interesting
   only if its **decision quality** beats the coherence-threshold gate enough
   to justify 2× cost — or only if it catches failure modes coherence-threshold
   misses.
2. **Coherence-threshold may already cover the failure mode.** The
   re-estimation trigger fires when coherence collapses; if that's the
   load-bearing signal, antithetic-pair perturbation is redundant decoration.
3. **The paper validates the pattern for *training*, not *runtime gates*.**
   Quality parity is NOT implied by architectural coverage (the canonical
   AdaJEPA R360 lesson) — a PoC must defend or refute this.

## Required PoC (per §3.6)

**Location:** `riir-ai/crates/riir-poc/benches/antithetic_pair_latent_gate_poc.rs`
**Target dir:** use `CARGO_TARGET_DIR=/tmp/antithetic-poc` per AGENTS.md, clean up after.

**Three competitors head-to-head on a controlled toy domain (no training):**

1. **Baseline (no adaptation):** static direction vector `d₀`, no updates.
2. **Coherence-threshold gate (shipped):** `ReestimationScheduler` with default
   `tau_reest=0.4`, triggers re-estimation from observation buffer when
   coherence collapses.
3. **Antithetic-pair ternary gate (paper-derived, distilled modelless):**
   paired `±δ` perturbation evaluation, ternary direction update per the
   reframe above.

**Controlled toy domain:** synthetic functor drift task — a `FunctorTable`
relation whose true direction drifts at a known rate. Measure:
- **Decision quality:** how quickly does each gate detect drift and update `d`?
- **Latency:** ns/gate-decision (criterion bench).
- **Stability:** does the gate oscillate under noise?

**Verdict table the PoC must print:**

| Gate | Drift detection latency (ticks) | Direction recovery error (cos) | Per-decision cost (ns) | Notes |
|---|---|---|---|---|

## Verdict rules (per §3.6 — defend OR refute)

- **If antithetic-pair beats coherence-threshold on quality at acceptable
  latency cost** → confirm, open plan in `katgpt-rs/.plans/` for a
  `antithetic_pair_latent_gate` feature-flagged primitive. Promote/demote per
  GOAT gate outcome.
- **If antithetic-pair is dominated by coherence-threshold** → refute
  honestly. Record raw numbers here as a PoC Addendum. The "fusion idea"
  becomes a non-shipping research note. The pattern stays as a one-line
  cross-reference in the EGGROLL distillation.
- **If mixed (wins on some drift modes, loses on others)** → tier down to
  Gain, ship behind feature flag for the regime where it wins, do NOT promote
  to default.

## Tasks

- [x] **T1** — Implement the synthetic functor drift toy domain in `riir-poc`
      (controlled, deterministic seed, no training).
      **DONE (2026-07-06).** `riir-ai/crates/riir-poc/src/antithetic_poc.rs` —
      `DriftingDirection` rotates a unit direction in the (0,1)-plane at angular
      rate ω; score = cos(candidate, truth) + Gaussian noise. Deterministic LCG
      PRNG (splitmix64). 7 unit tests pass.
- [x] **T2** — Implement the three gate competitors behind a common trait.
      **DONE (2026-07-06).** `LatentDirectionGate` trait + `Frozen`,
      `CoherenceTriggered` (distilled ReestimationScheduler analog: coherence
      EMA < tau_reest → re-estimate from observation buffer), `AntitheticPair`
      (paper-derived: sign(s⁺−s⁻)·δ·lr latent-direction update).
- [x] **T3** — Run head-to-head bench, print verdict table.
      **DONE (2026-07-06).** `riir-ai/crates/riir-poc/benches/antithetic_pair_latent_gate_poc.rs`.
      7 drift regimes × 8 episodes × 3 competitors. Verdict: CONFIRM 7/7.
- [x] **T4** — Based on T3 outcome: confirm (→ plan) or refute (→ PoC
      addendum update here + one-line cross-ref in Research 377).
      **Initial verdict CONFIRM-with-caveat revised to TIER DOWN after the
      strengthened follow-up PoC (see Strengthened PoC Addendum below).** The
      initial 7/7 CONFIRM was against a weak scalar-score finite-difference
      distillation. Adding `CoherenceTriggeredDirectObs` (direct vector obs +
      tracking-appropriate coherence metric) flipped the verdict to 3/7 antithetic
      wins, 4/7 direct-obs wins. Antithetic-pair is NOT a universal GOAT — it
      wins only in slow-drift and scalar-score-only regimes. Do NOT promote to
      default; keep as opt-in research note.
- [x] **T5** — If T4 confirms, run the modelless unblock protocol §3.5 check
      explicitly: this is a latent-direction update (constraint #4), NOT a
      weight mutation — confirm no `M ← M + ...` step anywhere.
      **DONE.** The antithetic-pair update is `d += sign(s⁺−s⁻)·δ·lr`
      (latent direction only, no weight mutation) — constraint #4 is satisfied by
      construction, no `M ← M + ...` step anywhere. The §3.5 modelless-unblock
      check PASSES. However, the strengthened PoC (T4 follow-up) shows the GOAT
      gate FAILS in 4/7 regimes — the primitive stays opt-in research note, not
      a promoted default. See Strengthened PoC Addendum for details.

## Non-tasks (do NOT do)

- Do NOT implement the EGGROLL training loop here. That's `riir-train`.
- Do NOT apply antithetic-pair to weight matrices. The reframe is
  latent-direction only.
- Do NOT promote to default on architectural coverage alone — wait for PoC.

## PoC Addendum (2026-07-06)

**Verdict: CONFIRM (7/7 regimes) — with an honest caveat.**

### Raw numbers (representative regime: moderate drift ω=0.02, low noise 0.02)

| Strategy | det_lat (ticks) | mean_rec_err | max_rec_err | updates | score_calls | latency (ns/tick) |
|---|---|---|---|---|---|---|
| Frozen (no adaptation) | 200 (never) | 1.5273 | 1.9981 | 0 | 0 | 5.6 |
| CoherenceTriggered (distilled) | 17 | 1.5202 | 1.9979 | 34 | 600 | 180 |
| AntitheticPair (paper) | 0 | 0.6014 | 1.0625 | 199.5 | 400 | 179 |

### Full verdict table (7 regimes, 8 episodes each)

| Regime | Frozen err | Coherence err | Antithetic err | Winner |
|---|---|---|---|---|
| slow ω=0.005, no noise | 0.2063 | 0.2063 (never fires) | 0.0032 | Antithetic |
| slow ω=0.005, low noise | 0.2063 | 0.2063 (never fires) | 0.0183 | Antithetic |
| moderate ω=0.02, no noise | 1.5273 | 1.5206 | 0.4515 | Antithetic |
| moderate ω=0.02, low noise | 1.5273 | 1.5202 | 0.6014 | Antithetic |
| moderate ω=0.02, high noise | 1.5273 | 1.5200 | 1.1661 | Antithetic |
| fast ω=0.05, low noise | 1.1511 | 1.1523 | 1.0563 | Antithetic |
| fast ω=0.05, high noise | 1.1511 | 1.1523 | 1.1088 | Antithetic |

### Latency (criterion, release, CPU)

- Frozen: 5.6 ns (no-op baseline)
- CoherenceTriggered: 180 ns/tick (3 score calls + buffer refit)
- AntitheticPair: 179 ns/tick (3 score calls + sign update)

The Issue 037 concern that antithetic-pair would cost "2× forward passes per
 gate decision" did NOT materialize — both distilled gates do 3 score calls
 (coherence: current + probe+ + probe−; antithetic: d+δ + d−δ + current). At
 D=8 the cost is dominated by the Gaussian RNG, not the score evaluation.

### Caveat — the distilled coherence gate is weak

The `CoherenceTriggered` competitor is a **distillation** of the shipped
`ReestimationScheduler`, not the real thing. It:
1. Never fires in slow-drift regimes (coherence EMA stays above tau_reest=0.4
   because the per-tick score change is small).
2. Barely recovers in moderate/fast-drift regimes (the finite-difference probe
   observation is noisy; the mean-of-buffer re-estimate doesn't track the
   rotating truth well).

The real `ReestimationScheduler` uses a closed-form least-squares refit (Gram
matrix + ridge) against a proper observation buffer, which would track the
rotating direction much better. A stronger coherence-gate distillation (or a
PoC against the real `ReestimationScheduler`) would narrow or eliminate the
antithetic-pair quality advantage.

**Honest verdict: the antithetic-pair pattern is architecturally sound,
modelless (constraint #4), latency-competitive (~180 ns), and beats a weak
coherence-gate distillation. Whether it beats the REAL coherence gate is
unproven — the §3.6 "architectural coverage ≠ quality parity" lesson applies
in reverse here (we proved quality superiority over a weak analog, not the
real one).**

### Next steps (not done in this PoC)

- [ ] Re-run with a stronger CoherenceTriggered distillation (least-squares
      refit instead of mean-of-probes) to see if the quality gap narrows.
- [ ] If antithetic-pair still wins after the coherence gate is strengthened,
      open `katgpt-rs/.plans/NNN_antithetic_pair_latent_gate.md` for the
      feature-flagged primitive.
- [ ] If the coherence gate catches up after strengthening, downgrade to
      REFUTE (the antithetic-pair adds no value over a well-tuned coherence gate).

The PoC stays as a permanent regression check in `riir-poc` per §3.6.

## Strengthened PoC Addendum (2026-07-06, follow-up)

**Verdict revised: TIER DOWN — antithetic-pair is NOT a universal GOAT.**

### What changed

The follow-up added a fourth competitor: `CoherenceTriggeredDirectObs` — a
coherence gate with **direct vector observations** (the production access
pattern), using a **tracking-appropriate coherence metric** (cosine between
current estimate and buffer mean) rather than the production displacement-
alignment metric (which is not applicable to absolute-direction tracking).

A close reading of the production `ReestimationScheduler` code revealed that
the prior session's caveat was based on a **misreading**: the latent-functor
re-estimation path uses **mean displacement** refit
(`f = (1/N) Σ (target_k − source_k)`, see `extract_functor_into`), NOT a
Gram-matrix least-squares refit. The Gram-matrix-LSQ pattern is KARC's
forecasting path (`fit_ridge`), not the latent functor's. The production
coherence trigger (`entry.coherence < tau_reest`) checks the STORED coherence
from the last fit — it detects "last fit was poor", not active drift. This is
fundamentally a fit-quality monitor, not a drift tracker.

The strengthened distillation gives the coherence gate the richest possible
access pattern (direct noisy direction observations, D numbers per tick vs
antithetic-pair's 2 scalar scores) and a tracking-optimized coherence metric.

### Full verdict table (4 competitors, 7 regimes, 8 episodes each)

| Regime | Frozen | Coh (scalar) | Coh (direct-obs) | Antithetic | Winner (direct-obs vs anti) |
|---|---|---|---|---|---|
| slow ω=0.005, no noise | 0.2063 | 0.2063 | 0.2063 (never fires) | 0.0032 | **Antithetic** |
| slow ω=0.005, low noise | 0.2063 | 0.2063 | 0.2063 (never fires) | 0.0183 | **Antithetic** |
| moderate ω=0.02, no noise | 1.5273 | 1.5206 | 0.4870 | 0.4515 | Antithetic (marginal) |
| moderate ω=0.02, low noise | 1.5273 | 1.5202 | 0.4882 | 0.6014 | **Direct-Obs** |
| moderate ω=0.02, high noise | 1.5273 | 1.5200 | 0.4885 | 1.1661 | **Direct-Obs** |
| fast ω=0.05, low noise | 1.1511 | 1.1523 | 0.9629 | 1.0563 | **Direct-Obs** |
| fast ω=0.05, high noise | 1.1511 | 1.1523 | 0.9603 | 1.1088 | **Direct-Obs** |

### Verdict tally

- Antithetic beats CoherenceTriggered (scalar score): **7/7** (unchanged)
- Antithetic beats CoherenceDirectObs (direct vec): **3/7**
- CoherenceDirectObs beats Antithetic: **4/7**

### Latency (criterion, release, CPU)

- Frozen: 1.87 ns (no-op)
- CoherenceTriggered (scalar score): 167.55 ns (3 score calls + finite-diff probe)
- CoherenceTriggeredDirectObs (vec): 122.74 ns (no score calls — just buffer mean)
- AntitheticPair: 159.71 ns (2 score calls + sign update)

The direct-obs gate is cheaper per-tick than antithetic-pair because it doesn't
sample Gaussian perturbations or call the score kernel — but this isn't a fair
latency comparison because the direct-obs gate's real cost is in the encoder
that produces the observations, which is external to the gate.

### Why the direct-obs coherence gate wins in moderate/fast/high-noise regimes

1. **Buffer-mean noise robustness.** The direct-obs gate averages D×OBS_CAPACITY
   = 8×32 = 256 noisy coordinates per refit. The antithetic-pair gate's ternary
   signal from 2 score calls per tick is inherently noisier — at high noise
   (0.1), the sign(s⁺−s⁻) flips randomly, producing erratic updates.
2. **Buffer lag dominates over per-tick tracking error at moderate/fast drift.**
   The buffer mean lags behind truth by ~(OBS_CAPACITY/2)·ω radians, but this
   lag is bounded and predictable. The antithetic-pair gate's per-tick updates
   have zero lag but higher variance — at moderate drift, variance dominates.

### Why antithetic-pair wins in slow-drift regimes

1. **The coherence gate doesn't fire.** With slow drift (ω=0.005), the buffer
   mean stays close to the current estimate (both are near the slowly-rotating
   truth), so coherence stays above `tau_reest=0.4` and the gate never refits.
   The initial estimate stays stale, accumulating error linearly.
2. **Antithetic-pair tracks continuously.** Per-tick updates with zero lag mean
   the estimate stays close to truth even for very slow drift.

### Honest verdict

**Antithetic-pair is NOT a universal GOAT for latent-direction tracking.** It
wins only in two niche regimes:
1. **Slow drift** (where coherence gates don't fire — antithetic-pair's
   continuous tracking is the only option among the competitors).
2. **Scalar-score-only access patterns** (where direct vector observations
   aren't available — e.g. black-box LLM-as-judge kernels that return only a
   scalar reward).

When direct vector observations ARE available and drift is moderate or faster,
the production coherence gate (mean-displacement refit on a buffer) is
competitive or superior, especially under noise.

**Recommendation: TIER DOWN to Gain. Do NOT promote antithetic-pair to
default-on. Keep as opt-in research note. If a future katgpt-rs plan opens for
this primitive, it should be feature-flagged for the scalar-score-only regime
only, and the plan must document the access-pattern constraint explicitly.**

### Tasks T5 closure

- [x] **T5** — Modelless unblock protocol §3.5 check: the antithetic-pair update
      is `d += sign(s⁺−s⁻)·δ·lr` (latent direction only, no weight mutation).
      Constraint #4 is satisfied by construction. No `M ← M + ...` step anywhere.
      **However**, the strengthened PoC shows the pattern is not a universal GOAT —
      the §3.5 modelless-unblock check PASSES (no training needed), but the GOAT
      gate (quality gain over a strong baseline) FAILS in 4/7 regimes. The
      primitive stays opt-in research note, not a promoted default.

## Cross-references

- **Source paper distillation:** `riir-train/.research/377_EGGROLL_Low_Rank_Evolution_Strategies.md`
- **QAT Infusion (the only katgpt-rs note mentioning zeroth-order methods):**
  `katgpt-rs/.research/202_QAT_Infusion_Inference_Time_Quantization_Awareness.md`
- **QAT-LoRA training side:** `riir-train/.research/087_QAT_LoRA_Fusion_Quantization_Aware_Adapter_Training.md`
- **Defend-wrong PoC spec (the rule this issue follows):** research skill §3.6
- **AdaJEPA R360 (canonical "architectural coverage ≠ quality parity" lesson):**
  `riir-ai/.issues/363`, `riir-ai/crates/riir-poc/benches/adajepa_modelless_goat.rs`

## TL;DR

EGGROLL's training algorithm goes to `riir-train/.research/377`. One
speculative modelless reframe survives re-mining: **antithetic-pair perturbation
→ ternary decision → latent direction update** (constraint #4 allows this — no
weight mutation). Architecturally distinct from `ReestimationScheduler`'s
single-coherence-threshold gate, but per §3.6 architectural distinctness is
necessary not sufficient.

**PoC verdict (strengthened follow-up, 2026-07-06): TIER DOWN.** The initial
7/7 CONFIRM was against a weak scalar-score finite-difference distillation of
the coherence gate. Adding a stronger distillation with **direct vector
observations** (`CoherenceTriggeredDirectObs`) flipped the verdict to **3/7
antithetic wins, 4/7 direct-obs wins**. Antithetic-pair is NOT a universal GOAT
for latent-direction tracking — it wins only in slow-drift and scalar-score-only
regimes. When direct vector observations are available, the production
coherence gate (mean-displacement refit) is competitive or superior, especially
under noise.

**A close reading of the production code also corrected a misreading:** the
latent-functor re-estimation path uses mean-displacement refit (not
Gram-matrix LSQ — that's KARC's path), and the production coherence trigger is
a fit-quality monitor (not a drift tracker). The antithetic-pair pattern fills
a real niche (scalar-score-only drift tracking) but is not broadly superior to
the shipped primitives. Keep as opt-in research note; do NOT open a katgpt-rs
plan unless a concrete scalar-score-only use case emerges.
