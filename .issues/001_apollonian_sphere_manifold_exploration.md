# Issue 001: Apollonian Sphere Manifold Geometry — Exploration

**Date:** 2026-06-23
**Status:** Active — FUNCATTN basis selection (#4) VALIDATED by probe (2026-06-26): structured basis beats random by +0.11 cos. T5.1 "invariance" claim falsified. Promoting to plan. MMORPG use cases (#1-#3) still rejected on domain-shape mismatch. **UPDATE 2026-06-26: Plan 332 executed. Principled bases (DCT-log, Haar-packet) tested head-to-head against the hand-crafted bound. MIXED verdict on the probe signal: Haar-packet captures 77% of the achievable gain at τ=0.5/k≤8 (Apollonian-surrogate hypothesis CONFIRMED for localized bases); DCT-log hurts on the narrow-low-freq probe signal BUT is vindicated on both DCT-aligned (+0.34) and realistic broadband PDE-like (+0.34) signals — outperforming even the hand-crafted bound on broadband. Feature ships opt-in; Phase 5 (true Apollonian harmonics) DEFERRED — narrow gain window doesn't justify the implementation cost. See [Plan 332 benchmark](../.benchmarks/332_structured_basis_goat_and_k_sweep.md).**
**Origin:** Gemini "Functional Attention + Relational Functor" reframing (2026-06-23)
**Related Research:** katgpt-rs/.research/257 (FUNCATTN), katgpt-rs/.research/219 (TNO/DEC), katgpt-rs/.research/291 (cross-resolution transport), katgpt-rs/.research/100 (EGA — fixed<learned precedent)

## Context

The Gemini reframing of our latent-to-latent pipeline proposed "nested Apollonian
topologies" as the manifold geometry underlying our latent space. Apollonian
sphere packings (Graham–Lagarias–Mallows–Wilks 2003) have real mathematical
properties that flat `R^d` lacks:

- **Hierarchical metric structure** — natural parent–child relationships between
  packed spheres at multiple scales.
- **Self-similarity** — same packing structure recurs at every scale.
- **Multi-resolution decomposition** — coarse packings are limit approximations
  of fine packings by construction.
- **Known harmonic decompositions** — Apollonian group structures connect to
  spherical harmonic analysis (relevant to FUNCATTN basis selection).

Grep across all 5 repos confirms **zero hits** for "Apollonian" — genuinely
unexplored in our corpus. Not present in `katgpt-rs/.research/`,
`riir-ai/.research/`, `riir-chain/.research/`, `riir-neuron-db/.research/`, or
shipped code.

## The Question

**What concrete game-AI or shard-retrieval use case does Apollonian geometry
enable that our current flat `R^d` + dot-product + sigmoid projection does not?**

Candidate use cases (each needs validation before this can become a plan):

1. **Hierarchical shard retrieval** — Apollonian packing gives natural
   parent–child metric relationships. Could `ShardIndex` use this for
   multi-resolution zone→shard lookup? *Baseline to beat:* lock-free
   `papaya::HashMap` O(1) lookup at current zone count.
2. **Cross-resolution personality transfer** — if shards live on an Apollonian
   manifold, small-dim shards are "coarse approximations" of large-dim shards
   by construction. *Related:* Research 291 (cross-resolution spectral transport).
3. **NPC social hierarchy** — Apollonian packings have a natural
   "center vs periphery" structure. Could this model faction hierarchies or
   attention allocation? *Baseline to beat:* current zone-density gating
   (`latent_functor/zone_gating.rs`).
4. **Spectral basis selection for FUNCATTN** — Apollonian packings have known
   harmonic decompositions. Could this give a better basis than spherical
   harmonics for FUNCATTN? *Baseline to beat:* sigmoid-normalized learned basis
   at k=4..16 (Research 257 §5.5).

## Why This Is an Issue, Not a Plan

We cannot run the novelty gate (Q1–Q4) honestly without a concrete use case.
"Nice geometry" alone fails Q3 (product selling point) and likely Q2 (new class
of behavior — it's not obviously a new capability vs an optimization). Once a
use case is proposed where Apollonian geometry beats flat `R^d` on a measurable
metric, this promotes to a plan with a real GOAT gate.

## Success Criteria (to close this issue)

- [x] Propose a concrete use case with a measurable metric — **DONE 2026-06-26**:
      use case #4 (FUNCATTN basis selection) evaluated with concrete task W
      (multi-scale synthetic transport, d=64, n=20, k∈{4,8,16}), metric
      (reconstruction cos ≥ 0.85), baselines (random-orthogonal, PCA, learned).
- [x] Sketch the minimal prototype — **DONE 2026-06-26**: replace `W_basis` with
      pre-computed Apollonian harmonics, benchmark vs random-orthogonal on the
      multi-scale transport task. See §"Evaluation" below.
- [x] Identify a kill condition — **DONE 2026-06-26**: hard kill = structured
      cos < random-orthogonal at any k. **Kill NOT triggered** — structured
      basis WINS (+0.11 cos at τ=0.5). T5.1 invariance premise falsified by
      probe. See §"Evaluation" below.

If no concrete use case is proposed within 30 days (by 2026-07-23), close as
"shelved — no concrete payoff identified". Do not let this linger as
perpetually-open speculative math.

## Evaluation (2026-06-26, REVISED after code probe)

### ⚠ CORRECTION: initial rejection was based on a falsified premise

The first evaluation (earlier 2026-06-26) rejected FUNCATTN basis selection
based on three precedents, with the T5.1 null result (Plan 286 L145-146) as the
load-bearing claim: *"the adaptive basis's row-normalization is invariant to
basis direction"*. A code probe (`tests/apollonian_basis_probe.rs`) was written
and run to test this claim empirically. **The claim is FALSE.** Structured bases
DO produce materially different Φ, and they produce measurably better transport
output. The rejection is retracted; use case #4 is validated and promoting to a
plan.

### MMORPG use cases (#1-#3) — rejected on domain-shape mismatch (unchanged)

Apollonian geometry answers "have metric, want hierarchy". MMORPG domains are
the inverse: factions/zones/social are explicit trees/graphs (structure known,
metric wanted); positions must stay raw flat-R² by anti-cheat rule; emotions use
flat dot-product+sigmoid by rule. Every candidate either loses to an existing
flat baseline (`papaya::HashMap` O(1), `latent_functor/zone_gating.rs`) or isn't
gameplay-native. This rejection stands — it was based on domain analysis, not on
the T5.1 premise.

### FUNCATTN basis selection (#4) — VALIDATED by probe

**Probe design** (`crates/katgpt-core/tests/apollonian_basis_probe.rs`, 3 tests):
- Input: multi-scale synthetic signal (4 sinusoidal scales, d=64, n=20, k=8)
- Three `w_basis` variants: random-orthogonal, signal-aligned structured,
  second random-orthogonal (noise floor)
- Metrics: Φ cosine similarity, effective rank, sharpness, transport output cos

**Result 1 — T5.1 invariance claim is FALSE:**
```
cos(Φ_rand1, Φ_rand2)  = 0.8613  ← noise floor (two random bases)
cos(Φ_rand1, Φ_struct) = 0.7779  ← structured basis
Δ = 0.0834 > 0.05 threshold  → H_structure HOLDS, H_invariance REJECTED
```
A structured basis produces a Φ that differs from random by MORE than two
random bases differ from each other. The T5.1 claim that "row-normalization is
invariant to basis direction" is empirically false. (Sharpness also differs:
structured 0.2103 vs random 0.1653 — structured is more discriminative.)

The likely explanation for T5.1's null result: PCA pre-rotation of a
random-orthogonal `w_basis` by an orthogonal eigenvector matrix `V` produces
`W·V^T`, which is ALSO random-orthogonal (product of two orthogonal matrices).
So T5.1 was comparing random-vs-random, not random-vs-structured. The
"invariance" was an artifact of the PCA-rotation experimental design, not a
property of the basis normalization.

**Result 2 — temperature amplifies the effect:**
```
τ=0.5: cos(rand,struct) gap = 0.0834
τ=0.1: cos(rand,struct) gap = 0.1407  ← basis choice matters MORE when sharp
```

**Result 3 — structured basis WINS on transport quality (the real test):**
```
Target: linear smoothing operator (representable by FUNCATTN)
τ=0.5:  random cos=+0.4806  structured cos=+0.5900  Δ=+0.1093 (structured +23%)
τ=0.1:  random cos=+0.5526  structured cos=+0.5772  Δ=+0.0245
```
A signal-aligned structured basis produces materially better transport output
than random-orthogonal: +0.11 cos at τ=0.5 (23% relative improvement). This is
on a representable target (linear smoothing across the sequence).

### What this means for Apollonian specifically

The probe used a HAND-CRAFTED structured basis (signal directions from the
generative model), not Apollonian harmonics. So the probe proves:
- **General claim**: structured bases CAN beat random for FUNCATTN (door open)
- **NOT yet proven**: Apollonian harmonics specifically beat random

The question for the plan: can Apollonian harmonics (or any principled
multi-scale basis) match or beat a hand-crafted signal-aligned basis WITHOUT
knowing the signal structure a priori? The hand-crafted basis cheated by using
the generative frequencies. A real basis-selection mechanism must work without
that prior knowledge.

### Remaining concerns (now weaker, not killers)

1. **EGA fixed<learned (Research 100)** — still applies but is now a softer
   concern. EGA showed learned data-adaptive beats fixed wavelets. But the probe
   shows fixed structured beats fixed random. The question is whether Apollonian
   (fixed) can land in the "good structured" region without learning. Open.
2. **FUNCATTN G6 failed** — still true, but the probe shows basis choice matters
   for the transport-quality regime (not the LLM-token regime). FUNCATTN may
   find its niche in transport/smoothing tasks where structured bases help, even
   if it lost the LLM benchmark.
3. **Cross-resolution (Plan 310) already handles multi-scale** — true, but via
   LEARNED bases. The probe suggests even fixed structured bases help; the
   question is whether a principled fixed basis (Apollonian) can avoid the
   learning cost while capturing most of the gain.

### Verdict (revised)

**Use case #4 HOLDS UP.** The T5.1 premise that killed the earlier evaluation
was empirically false. A structured basis produces +0.11 cos improvement on
transport quality. Promoting to a plan with a real GOAT gate: can Apollonian
harmonics (or a principled multi-scale basis) match the hand-crafted structured
basis without a-priori signal knowledge?

## Related Work (external, TBD)

- Graham, Lagarias, Mallows, Wilks, Yan — *Apollonian Circle Packings: Number
  Theory* (2003). J. Number Theory 100:1–45.
- Spherical harmonics on Apollonian packings — analysis literature (needs
  targeted arxiv search before promoting to plan).
- Hyperbolic embeddings (Poincaré, Nickel–Kiela) — different non-Euclidean
  geometry but same "geometry-as-inductive-bias" thesis; relevant prior art for
  the "is non-flat geometry worth it?" question.

## Cross-Refs

- `katgpt-rs/.research/257_Functional_Attention_Spectral_Transport_Operator.md`
  — FUNCATTN basis selection (use case 4).
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md`
  — DEC operators on cell complexes (different topology, same
  "geometry-as-routing" idea — shows we're already open to non-flat geometry).
- `katgpt-rs/.research/280_Resolution_Tiered_Deterministic_Commitment.md`
  — resolution tiering (related to use case 2).
- `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md`
  — F3 fusion target (Apollonian as the natural multi-resolution basis).
- `riir-neuron-db/src/index.rs` — `ShardIndex` baseline for use case 1.

## TL;DR

Apollonian sphere packings proposed as latent manifold geometry. MMORPG use
cases (#1-#3) rejected on domain-shape mismatch. FUNCATTN basis selection
(#4) VALIDATED by code probe (2026-06-26): the T5.1 "invariance" premise was
empirically false (Δ=0.0834 vs 0.05 threshold), and a structured basis beats
random-orthogonal by +0.11 cos on transport quality. Promoting to plan.
Lesson: don't reject on documented claims alone — read the code, run the test.
The T5.1 "random-vs-random" experimental design was the bug, not the basis
normalization.
