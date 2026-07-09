# Issue 049: Block-Sparse HLA — Super-GOAT Fusion Candidate Validation

**Date:** 2026-07-08
**Origin:** Research 393 §3 (Block-Sparse Featurizers distillation)
**Status:** Q1=YES + Q2=YES (PoC PASSED 2026-07-09) + Q4=YES; Q3 (real-game-data) is the FINAL gate. Plan 412 blocker RESOLVED (shipped DEFAULT-ON). **Related:** Research 393, Plan 412 (shipped, DEFAULT-ON — the open primitive), Plan 301 (subspace_phase_gate)

## Context

Research 393 distilled Goodfire's Block-Sparse Featurizers (BSF) into a GOAT open primitive (`SubspaceSteeringField`, Plan 412 — k-dim manifold steering). During the novelty gate, a **Super-GOAT fusion candidate** emerged but was NOT claimed because Q1–Q4 confidence was insufficient:

> **Block-Sparse HLA** — reframe HLA's 8-dim per-NPC latent state as a union of concept subspaces (e.g., 2 blocks of 4: {valence, arousal, desperation, calm} ⊕ {fear, curiosity, ...}) with a block-sparsity prior (a few blocks active per tick). Each block is multidimensional and steerable within-region (via Plan 412). The "active block set" (top-k by energy) is the NPC's current "emotional posture".

This would be a Super-GOAT IF it produces a new capability class ("NPCs whose emotional posture is a sparse set of active concept-blocks, each multidimensional, steerable within-region") — but that claim needs empirical validation on real HLA data, not just architectural reasoning (per the §3.6 defend-wrong PoC rule).

Per the research workflow's "no candidate escape hatch" rule, the verdict was **GOAT** (open primitive only), and this issue tracks the Super-GOAT validation follow-up. **No riir-ai guide is created until Q1–Q4 pass.**

## The 4 questions to answer (Q1–Q4 novelty gate)

### Q1 — No prior art?

**✅ ANSWERED = YES (novel) — 2026-07-09.** See "Q1 verdict" section below.

**⚠️ Correction to the original Q1 instructions:** the issue originally pointed at `riir-engine/src/hla/{kernel,forward,types}.rs` — those are the **wrong HLA**. They implement **Higher-order Linear Attention** (a transformer attention cache: `kernel.rs` maintains rank-1 SK/CQV/mQ/G/h moments). The **emotion HLA** (the 8-dim per-NPC affective state this issue is about) lives in different files. Correct files to read:
- `riir-ai/crates/riir-engine/src/cgsp_runtime/types.rs` — `NpcEmotionScalars`, `HlaCuriosityDirection`
- `riir-ai/crates/riir-engine/src/cwm_runtime/hla_projection.rs` — `HlaDirectionTable` (the 1D-per-emotion projection)
- `riir-ai/crates/riir-engine/src/committed_blend/archetypes.rs` — reserved-dim zeroing
- `riir-ai/crates/riir-engine/src/cce_runtime/state_space.rs`
- `riir-ai/crates/riir-engine/src/crowd_attention.rs`

**Original (now-resolved) questions:**
- Do the 3 reserved dims already implicitly carry block structure? **NO** — they are zeroed padding everywhere.
- Is the 5-scalar projection actually 1D-per-emotion, or is there hidden multidim structure in how `evolve_hla` populates them? **Genuinely 1D-per-emotion** — `HlaDirectionTable` is a `[5][4]` array, one independent direction vector per emotion channel, zero cross-channel coupling.
- Does any existing riir-ai code already group HLA axes into blocks? **NO** — grep for `block_sparse|active_block|partition.*subspace|subspace.*partition|emotion.*block` across `riir-engine/src/**/*.rs` returns zero hits in any emotion-HLA file.

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

- **~~Plan 412 must ship first.~~** ✅ RESOLVED — Plan 412 (`SubspaceSteeringField`) shipped and is **DEFAULT-ON** as of commit `7bff9cc7` (Phase 5 promote). The fusion primitive is available for the Q2 PoC to consume.
- **Q1 (resolved 2026-07-09):** read the emotion-HLA files listed in the Q1 section above — the original `hla/kernel.rs` pointer was the wrong HLA. Verdict: novel, no prior art.
- **Q2/Q3 require a PoC** in `riir-ai/crates/riir-poc/` per the §3.6 defend-wrong rule. Architectural coverage ≠ quality parity.

## Q1 verdict (2026-07-09) — YES, novel

The emotion HLA is a **completely flat, independent-axis** representation. No code
groups HLA axes into blocks, partitions, or coupled subspaces. The block-sparse
reframing would introduce genuinely new structure.

- **The 8 dims are flat:** first 5 are `NpcEmotionScalars` (`valence, arousal,
  desperation, calm, fear` — `cgsp_runtime/types.rs:153-165`); last 3 are
  **zeroed reserved padding** (`committed_blend/archetypes.rs:342-349` `unit_z()`
  sets `z[0..5] = 1.0`, `z[5..8]` stay `0.0`; `karc_bridge/anticipation.rs:122-125`
  emits `[0.0, arousal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`).
- **Projection is 1D-per-emotion:** `HlaDirectionTable`
  (`cwm_runtime/hla_projection.rs:164-171`) is a `[5][4]` array — one independent
  `[f32; 4]` direction vector per emotion channel; `project_channel` (L274-278)
  dots belief scalars with one channel's direction → one scalar via sigmoid. No
  cross-channel coupling.
- **The only "structure" is a flat orthogonal Hadamard set:** `HlaCuriosityDirection`
  (`cgsp_runtime/types.rs:40-50, 104-119`) builds 8 mutually-orthogonal directions
  from a flat Hadamard basis (`h(i,j) = (-1)^popcount(i&j)`). Not grouped into
  blocks of 4 or 2.
- **Crowd attention treats HLA as flat** `[n × 8]` (`crowd_attention.rs:67-105`);
  set-attention is applied *across NPCs*, never across HLA-dimension blocks.

**Next step:** Q2 defend-wrong PoC in `riir-ai/crates/riir-poc/` — can block-sparse
HLA distinguish e.g. predator-fear from starvation-fear in a way the flat
5-scalar projection cannot? Plan 412 (`SubspaceSteeringField`) is now available
for the PoC to consume.

## Q2 verdict (2026-07-09) — YES, new capability (PoC PASSED)

The defend-wrong PoC shipped at
`riir-ai/crates/riir-poc/benches/block_sparse_hla_supergoat_poc.rs` (consumes
the real Plan 412 `SubspaceSteeringField<D=8,K=2>` + `block_energy` +
`walk_manifold`). Three competitors (flat-5 production / block-8 / floor), three
tests. Run: `cargo bench -p riir-poc --bench block_sparse_hla_supergoat_poc`.

**Result table:**

| Test | Flat-5 | Block-8 | Floor | Verdict |
|---|---|---|---|---|
| T1 separation parity (subtype in arousal/desperation) | 100% | 100% | — | **flat sufficient** for classification |
| T2 subtype @ fixed dim4 (subtype in dim5) | 50% | 100% | 50% | **block represents behavior flat CANNOT** |
| T3 control-surface DOF @ fixed dim4 | 1 posture | 32 postures | — | block exposes within-fear steering |

**Decisive test (T2):** hold `dim4` (flat's only fear knob, read directly per
`HlaDirectionTable`) bit-constant; vary only `dim5` (a reserved dim production
never reads — the Q1 finding). Sanity check confirmed `flat_identical=true`:
the flat-5 vector is provably bit-identical across both behavior classes, so flat
scores the majority-class rate (50%). Block reads `(dim4, dim5)` via Plan 412
`block_energy` → 100%. T3 shows flat produces 1 distinct fear-posture at fixed
dim4 vs block's 32, plus a block-only intensity-preserving rotation (norm drift
5.96e-8, other-channel drift 0).

**Q2 = YES.** Block-sparse HLA produces a behavior (a decoupled fear-subtype
steerable at fixed flat-fear) that the flat 5-scalar projection provably cannot
represent.

**Honest defend-wrong caveats (the PoC defends OR refutes; these bound the win):**
1. The gap is specifically the **DECOUPLED-subtype regime** — a designer who
   wants fear-subtype orthogonal to the 5 production scalars. If subtype is
   allowed to ride on arousal/desperation, flat suffices (T1 = 100%).
2. Production already HAS 3 reserved dims `{5,6,7}`; block-sparse is partly the
   decision to READ them as emotion sub-blocks rather than leave them zeroed.
   The capability is real but the "novelty" is partly a **read-policy change**
   layered on Plan 412, not a brand-new math primitive.
3. Intensity-preserving rotation (T3b) moves `dim4` (flat fear) — it is a
   block-level op, NOT "invisible to flat". Flat's deficit is having no second
   axis at all.

These caveats are why Q2 is answered YES but the Super-GOAT guide is NOT yet
created — Q3 (product selling point validated on real game data) remains open,
and caveat #2 may bound the Super-GOAT claim to "novel composition of Plan 412 +
the reserved dims" rather than "novel primitive".

## Progress tracker

- [x] **Q1** — No prior art? **YES** (novel, 2026-07-09)
- [x] **Q2** — New class of behavior? **YES** (PoC PASSED 2026-07-09; defend-wrong caveats documented)
- [ ] **Q3** — Product selling point? (needs real-game-data PoC) — NEXT, the final gate
- [x] **Q4** — Force multiplier? **Likely YES** (≥8 cousin plans — 412 × 301 × 297 × 320 × 319 × HLA kernel × KarcShard × 251)

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
