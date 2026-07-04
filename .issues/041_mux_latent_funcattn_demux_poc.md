# Issue 041 — Mux-Latent × FUNCATTN demux-on-edge: PoC required before claiming a gain

**Filed:** 2026-07-04
**Priority:** P3 (deferred until PoC lands — may be noise, may be gain, currently undetermined)
**Origin:** Evaluation of Gemini's "Continuous Neuro-Symbolic DAG" proposal (2026-07-04). The proposal suggested composing `mux_latent` (multi-hypothesis packing) with `FUNCATTN` (functional attention) to demultiplex alternative destination hypotheses on a single PTG edge. Both ship; the composition does not. Whether the composition is a gain is unknown.
**Blocks:** Nothing. **Blocked by:** A defend-wrong PoC (this issue's deliverable).
**Type:** PoC / benchmark issue. NOT an implementation issue — the goal is to determine whether the composition is worth implementing.

---

## Problem

Two primitives ship independently:

| Primitive | Location | Role |
|---|---|---|
| `mux_latent` / `mux/demux.rs::MuxDemuxVerifier` | katgpt-rs | Multiplex multiple latent states into one packed representation; demultiplex on demand |
| `FUNCATTN` (Plan 286) | katgpt-rs `katgpt-attn` | Functional attention — attention with learnable/functional Q/K/V projections (gated `funcattn`, opt-in) |

Gemini's proposal: combine them — pack K alternative destination hypotheses into one mux'd latent vector on a PTG edge, then use FUNCATTN to demultiplex ("which of the K hypotheses best matches the query?") at the edge destination. The pitch: "one edge carries K alternatives, attention picks the winner" — cheaper than K parallel edges.

**The gain is unverified.** Three failure modes are plausible:

1. **FUNCATTN already does this.** Functional attention's Q/K/V projections may already induce an implicit mux/demux over alternative targets — in which case explicit `mux_latent` is redundant overhead.
2. **mux_latent destroys the signal FUNCATTN needs.** If packing K hypotheses into one vector loses per-hypothesis locality (e.g., via averaging), FUNCATTN cannot recover the per-hypothesis signal and the composition is strictly worse than either primitive alone.
3. **The composition is a real gain.** mux'd packing is cheaper than K parallel edges, and FUNCATTN can still discriminate — net win on perf with no quality loss.

We don't know which. The proposal asserted (3) without proof.

## Scope

A defend-wrong PoC benchmark that **tries to falsify** the gain hypothesis. PoC lives in `riir-ai/crates/riir-engine/benches/` (not katgpt-rs) because it needs both primitives composed in a runtime context — katgpt-rs ships them separately; riir-ai is where runtime composition is tested.

### PoC design

Three competitors on a fixed workload (e.g., 8 candidate next-states, query vector Q, score each candidate by relevance to Q):

| Competitor | Description |
|---|---|
| **(a) K parallel edges (baseline)** | Score = FUNCATTN(Q, candidate_i) for each i ∈ [0..K). K forward passes. |
| **(b) mux_latent + FUNCATTN demux (proposed)** | Pack K candidates via `mux_latent`, single FUNCATTN pass, demux the resulting scores. 1 forward pass + mux/demux overhead. |
| **(c) FUNCATTN-only (no mux)** | FUNCATTN with K-expanded Q/K/V (FUNCATTN may already do this implicitly). 1 forward pass. |

### Metrics

- **Quality:** top-1 accuracy (does the winner match baseline (a)'s winner?), mean-rank correlation with (a), MSE on soft scores.
- **Perf:** wall-clock μs per "score K candidates" call. Memory: peak allocations.
- **Scaling:** sweep K ∈ {2, 4, 8, 16, 32}. If (b) doesn't beat (a) at K=32, it's not a gain.

### Decision rule

- **(b) loses on quality** at any K → composition is noise. Close this issue with verdict "DO NOT IMPLEMENT".
- **(b) wins on perf but ties on quality** at K ≥ 8 → gain confirmed. File a Plan (not an issue) to implement the composition in katgpt-rs as a first-class primitive.
- **(b) wins on perf but loses on quality** → composition is a perf-for-quality tradeoff. Document the regime (which K, which quality bound) and let consumers decide. Probably stays opt-in.

## Tasks

- [ ] **T1** Build the bench harness in `riir-ai/crates/riir-engine/benches/bench_041_mux_funcattn_demux.rs`. Three competitors, K sweep, quality + perf metrics.
- [ ] **T2** Run on a synthetic dataset (deterministic seed, reproducible) — K=8 candidate vectors sampled from a known distribution, Q sampled independently. Capture results in `.benchmarks/041_mux_funcattn_demux_poc.md`.
- [ ] **T3** Verdict per the decision rule above. If "noise" → close this issue with the bench artifact retained for posterity. If "gain" → file a Plan in katgpt-rs to implement the composition primitive (with a GOAT gate).
- [ ] **T4** If verdict is "tradeoff" → document the regime in the PoC report and close this issue with a recommendation that consumers opt-in via feature flag (no default-on promotion).

## Non-Goals

- ❌ Implementing the mux-FUNCATTN composition as a primitive. This issue is the PoC; the implementation (if any) is a separate Plan gated on the PoC verdict.
- ❌ Quality improvements to either primitive in isolation.
- ❌ Generalized "compose any two katgpt-rs primitives" tooling.

## Honest caveats

1. **My prior is "noise or tradeoff", not "gain".** FUNCATTN's Q/K/V projections already induce per-candidate attention scores; mux_latent's packing is designed for transport efficiency, not attention locality. The composition plausibly doesn't help. The PoC exists to confirm or falsify, not to advocate.
2. **Picking the workload is the hardest part.** A synthetic workload may not exercise the regime where the composition helps (or hurts). If the verdict is "noise" on synthetic but the proposal's use case (real NPC destination hypotheses) has structure the synthetic lacks, the PoC may give a false negative. Note this in the report.
3. **No training.** Per AGENTS.md modelless-first mandate, the PoC must use deterministic (non-learned) FUNCATTN projections and pre-extracted direction sets. If the gain only materializes with trained projections, that's a riir-train dependency and the composition is not modelless-promotable.

## Cross-References

- **`mux_latent`:** `katgpt-rs/crates/katgpt-core/src/mux/demux.rs::MuxDemuxVerifier` (search for module location in T1).
- **FUNCATTN:** `katgpt-rs/crates/katgpt-attn/` (Plan 286, feature `funcattn`, opt-in).
- **Origin evaluation:** Gemini "Continuous Neuro-Symbolic DAG" proposal review (2026-07-04). The composition suggestion is the only part of the proposal that is neither (a) already shipped under a different name nor (b) obviously implementable as a small primitive — it needs a PoC first.

## TL;DR

Both `mux_latent` and `FUNCATTN` ship independently in katgpt-rs. Gemini's proposal suggested composing them: pack K alternative destination hypotheses via mux, demultiplex via FUNCATTN on a single PTG edge. The gain is unverified and my prior is "noise" — FUNCATTN may already do this implicitly. This issue tracks a defend-wrong PoC bench in riir-ai: three competitors (K-parallel-edges baseline, mux+FUNCATTN demux, FUNCATTN-only), quality + perf metrics, K ∈ {2,4,8,16,32}. Verdict per a decision rule: noise → close; gain → file a Plan; tradeoff → document regime. P3, deferred until somebody picks it up.
