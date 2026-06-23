# Issue 001: Apollonian Sphere Manifold Geometry — Exploration

**Date:** 2026-06-23
**Status:** Open — awaiting concrete use case
**Origin:** Gemini "Functional Attention + Relational Functor" reframing (2026-06-23)
**Related Research:** katgpt-rs/.research/257 (FUNCATTN), katgpt-rs/.research/219 (TNO/DEC), katgpt-rs/.research/291 (cross-resolution transport)

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

- [ ] Propose a concrete use case with a measurable metric (e.g., "Apollonian
      ShardIndex retrieval is X% faster than flat papaya HashMap at Y zones",
      or "Apollonian FUNCATTN basis achieves cos ≥ Z where spherical harmonics
      achieve cos < Z on task W").
- [ ] Sketch the minimal prototype: what code, what benchmark, what baseline.
- [ ] Identify a kill condition: what result abandons the idea.

If no concrete use case is proposed within 30 days (by 2026-07-23), close as
"shelved — no concrete payoff identified". Do not let this linger as
perpetually-open speculative math.

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

Apollonian sphere packings proposed as latent manifold geometry. Zero hits in
our 5-repo corpus — genuinely unexplored. Cannot run the novelty gate without a
concrete use case. File as exploration; promote to plan only when someone
proposes a measurable win over flat `R^d`. Close in 30 days if no use case
emerges.
