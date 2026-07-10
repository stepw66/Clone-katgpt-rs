# Issue 122: Canvas × Functor Topology Modelless Behavioral PoC (Super-GOAT Re-evaluation Gate)

**Created:** 2026-07-10
**Research:** [katgpt-rs/.research/403_Topology_Constrained_Latent_Functor_Composition.md](../.research/403_Topology_Constrained_Latent_Functor_Composition.md) §2.5 (fusion) + §3.1 (Q2 unproven)
**Related:** [R398 §7](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md) (the Issue 043 lesson — ad-hoc metric was wrong tool)
**Status:** Open

---

## The question

Research 403 distills Canvas Engineering onto the latent-functor substrate (gated functor graph instead of attention mask on a DiT), dissolving R398's "needs training" conclusion. The structural value is clear (declared topology + reachability guarantee + isolated swap + attribution). **The open question is Q2 (new behavioral class): does gating the topology (perception→affect→action, no shortcut) produce measurably better NPC behavior than free composition (today's flat FunctorTable)?**

This is the same question Issue 043 tried to answer — but on the WRONG substrate (DiT attention mask) with the WRONG tool (ad-hoc flee metric). This PoC corrects both: latent-functor substrate + FaithfulnessProbe attribution.

## Why this PoC, not architectural reasoning

Per research skill §3.6: architectural coverage ≠ quality parity. R403 §2.4 proves the pieces ship; it does NOT prove the gated graph behaves better than free composition. A PoC is mandatory for the Super-GOAT re-evaluation.

## PoC design (per §3.6 — three competitors minimum, FaithfulnessProbe-attributed)

**Domain:** `riir-ai/crates/riir-poc/` — the defend-wrong R&D crate. Reuse the existing toy NPC cognition task, but rewire it onto the latent-functor substrate.

**Three competitors:**
1. **Floor (no functor graph):** HLA-only, no region producers, no topology. Degenerate baseline.
2. **Free composition (today's FunctorTable):** region producers (region_subspace, DEC, trajectory) composed by hardcoded call order, no gating. This is the shipped state.
3. **Gated graph (R403 fusion):** same region producers, but wired under a declared `CanvasSchema` topology with `compile_functor_graph` + sigmoid subspace gates. perception→affect→action enforced; no perception→action shortcut.

**The confound control (Issue 043's lesson, NOT repeated):**
- Region producers are the **shipped bridges** (region_subspace for affect, DEC codifferential for perception, spectral_trajectory for trajectory), NOT a designer-tuned classifier. This is the critical fix — Issue 043's "behavioral gain" was an artifact of a hand-tuned discriminator; this PoC uses only shipped modelless producers.
- Attribution tool is **`FaithfulnessProbe`** (Plan 278, already wired in riir-poc via `jlens_concept_readout_goat.rs`), NOT a displacement-alignment flee metric. Causal intervention (empty/shuffle/corrupt a region's `FunctorEntry` output, measure downstream NPC-scalar delta) is the scientifically correct isolation.

**Metrics (per region, gated vs free):**
- `FaithfulnessProfile` per region: `empty_delta`, `shuffle_or_corrupt_delta`, `irrelevant_delta`, `filler_delta`.
- `is_faithfully_used(threshold)` verdict per region.
- Reachability sanity: `can_influence(perception, action)` = false at horizon 1, true at horizon 2 (gated variant only).
- Latency: per-tick overhead of gated graph vs free composition (budget: sub-µs gate lookups, canvas compile is one-time).

## Verdict protocol (§3.6 honesty)

- **If the gated graph's `empty_delta` on affect-dependent regions is LARGER than free-composition's** (proving the gate is load-bearing — removing a region's output actually changes behavior) AND free-composition's is SMALLER (proving redundancy — the shortcut compensates) → the topology adds causal structure that free composition lacks → **re-evaluate R403 for Super-GOAT.** Create the riir-ai guide + plan.
- **If the deltas MATCH** → the gating is architectural-only (no behavioral delta) → R403 stays **GOAT**, ships on structural merits (isolated swap + attribution + guarantees). Close with negative result recorded.
- **If the gated graph is WORSE** → gating overhead hurts without payoff → downgrade R403 or keep the gated graph opt-in-only. Record refutation honestly.

**Do NOT silently revise the verdict to match the PoC.** Record raw `FaithfulnessProfile` numbers, then revise explicitly.

## Where the PoC lives

`riir-ai/crates/riir-poc/benches/canvas_functor_topology_modelless.rs` (new). Reuses the `jlens_concept_readout_goat.rs` pattern (`DefaultFaithfulnessProbe` + `faithfulness_profile` at 0.5 threshold). Run:

```bash
CARGO_TARGET_DIR=/tmp/canvas_functor_122 cargo bench \
  -p riir-poc --bench canvas_functor_topology_modelless -- --nocapture
```

Clean up `/tmp/canvas_functor_122` when done. The PoC stays as a permanent regression check (§3.6).

## Blockers

- R403's `compile_functor_graph` sketch (§4) must be implemented as a minimal riir-poc helper (not a full riir-engine module) for the PoC. The PoC does NOT require Plan 419's canvas_schema to be promoted — it can construct the gated graph inline.
- The `ConsumerContext` impl for the NPC action function must wrap the latent-functor bridges (the affect region's output = the action region's input via the gate).

## Out of scope

- Wiring `compile_functor_graph` into production `npc_integration.rs` (that's the riir-ai plan IF the PoC passes).
- Training region content (the producers are shipped modelless bridges; no GD).
- R398's DiT-substrate canvas (superseded by R403 for our codebase; R398's conclusion still holds for literal DiTs).

## TL;DR

R403 says gating the latent-functor topology (canvas reachability over the flat FunctorTable) is a structural GOAT. This PoC tests whether it's a behavioral Super-GOAT. Three competitors (floor / free composition / gated graph), FaithfulnessProbe-attributed (NOT the ad-hoc flee metric that botched Issue 043), shipped-bridge region producers (NOT a designer-tuned classifier). If the gated graph's `empty_delta` proves load-bearing where free-composition is redundant → Super-GOAT. Else → stays GOAT on structural merits.
