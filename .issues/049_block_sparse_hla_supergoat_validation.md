# Issue 049: Block-Sparse HLA — Super-GOAT Fusion Candidate Validation

**Date:** 2026-07-08
**Origin:** Research 393 §3 (Block-Sparse Featurizers distillation)
**Status:** Open — Q1–Q4 validation NOT yet done
**Related:** Research 393, Plan 412 (the open primitive that unblocks this), Plan 301 (subspace_phase_gate)

## Context

Research 393 distilled Goodfire's Block-Sparse Featurizers (BSF) into a GOAT open primitive (`SubspaceSteeringField`, Plan 412 — k-dim manifold steering). During the novelty gate, a **Super-GOAT fusion candidate** emerged but was NOT claimed because Q1–Q4 confidence was insufficient:

> **Block-Sparse HLA** — reframe HLA's 8-dim per-NPC latent state as a union of concept subspaces (e.g., 2 blocks of 4: {valence, arousal, desperation, calm} ⊕ {fear, curiosity, ...}) with a block-sparsity prior (a few blocks active per tick). Each block is multidimensional and steerable within-region (via Plan 412). The "active block set" (top-k by energy) is the NPC's current "emotional posture".

This would be a Super-GOAT IF it produces a new capability class ("NPCs whose emotional posture is a sparse set of active concept-blocks, each multidimensional, steerable within-region") — but that claim needs empirical validation on real HLA data, not just architectural reasoning (per the §3.6 defend-wrong PoC rule).

Per the research workflow's "no candidate escape hatch" rule, the verdict was **GOAT** (open primitive only), and this issue tracks the Super-GOAT validation follow-up. **No riir-ai guide is created until Q1–Q4 pass.**

## The 4 questions to answer (Q1–Q4 novelty gate)

### Q1 — No prior art?

**Uncertain.** The existing HLA 8-dim treatment (`riir-ai/crates/riir-engine/src/hla/`) projects to 5 scalar affective axes (valence/arousal/desperation/calm/fear) + 3 reserved. Questions:
- Do the 3 reserved dims already implicitly carry block structure?
- Is the 5-scalar projection actually 1D-per-emotion, or is there hidden multidim structure in how `evolve_hla` populates them?
- Does any existing riir-ai code already group HLA axes into blocks?

**To resolve:** `read_file` `riir-ai/crates/riir-engine/src/hla/kernel.rs` + `forward.rs` + `types.rs`. Grep for any block/subspace/grouping structure in the HLA evolution kernel. If the kernel already treats subsets of the 8 dims as coupled, the block structure partially ships and Q1 → NO (not novel enough).

### Q2 — New class of behavior?

**Uncertain.** "Block-sparse emotional posture" could be:
- A **new capability** — NPCs whose affective state is a sparse set of active concept-blocks, each multidim, enabling finer-grained emotional modeling than 5 scalars.
- A **re-interpretation** — the existing 8-dim state already captures this implicitly; the block framing is just a lens.

**To resolve:** Construct a toy PoC in `riir-ai/crates/riir-poc/` (per §3.6 defend-wrong protocol). Two competitors: (a) current 5-scalar HLA projection, (b) block-sparse HLA (2 blocks of 4, top-1 block active). Run both on a controlled toy emotional-shift benchmark. If (b) produces behaviors (a) cannot — e.g., distinguishing predator-fear from starvation-fear — Q2 → YES. If they produce equivalent behaviors, Q2 → NO.

### Q3 — Product selling point?

**Needs real game data.** Candidate selling point: *"Our NPCs have multidimensional emotional posture — a 'fearful' NPC can be predator-fearful or starvation-fearful, and the designer steers within the fear region to produce distinct behaviors."* This is sellable IF the block structure validates on real NPC behavior traces.

**To resolve:** Run the Q2 PoC on real Seal Online NPC traces (if available) or a representative game-AI benchmark. If the block structure produces meaningfully distinct NPC behaviors, Q3 → YES.

### Q4 — Force multiplier?

**Likely YES.** Connects Plan 412 (subspace steering) × Plan 301 (basis discovery) × Plan 297 (personality composition) × Plan 320 (indicator bank) × Plan 319 (Clifford wedge) × HLA kernel × KarcShard × Plan 251 (DEC hodge). ≥8 cousins. But force multiplication alone doesn't make a Super-GOAT — Q1–Q3 must also pass.

## Blockers

- **Plan 412 must ship first.** The Super-GOAT claim depends on consuming subspace steering fields at runtime; without the open primitive, the HLA fusion is theoretical.
- **Q1 requires reading the HLA kernel** (`riir-ai/crates/riir-engine/src/hla/`) — this is riir-ai private code; the validation happens there.
- **Q2/Q3 require a PoC** in `riir-ai/crates/riir-poc/` per the §3.6 defend-wrong rule. Architectural coverage ≠ quality parity.

## Acceptance

When Q1–Q4 are all answered YES with evidence:
1. Create `riir-ai/.research/NNN_block_sparse_hla_supergoat_guide.md` with the full Super-GOAT mandatory outputs (TL;DR, distilled primitive, connection map, latent-vs-raw boundary, what stays private, validation protocol, implementation priority P0–P3).
2. Open `riir-ai/.plans/MMM_block_sparse_hla_wiring.md` for the runtime integration.
3. Close this issue with a pointer to the guide.

When any of Q1–Q4 is NO:
1. Document the negative result in this issue.
2. The open primitive (Plan 412) stands on its own GOAT merits; the HLA fusion does not elevate to Super-GOAT.
3. Close this issue as "validated — not Super-GOAT".

## Anti-deferral note

This is a **validation task**, not an implementation task. It does not block Plan 412 (the open primitive ships regardless). It blocks ONLY the Super-GOAT guide creation. Do not let it stall — if Q1–Q4 cannot be answered within one focused session after Plan 412 ships, default to "not Super-GOAT" and close.
