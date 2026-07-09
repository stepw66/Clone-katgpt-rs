# Issue 049: Block-Sparse HLA — Super-GOAT Fusion Candidate Validation

**Date:** 2026-07-08
**Origin:** Research 393 §3 (Block-Sparse Featurizers distillation)
**Status:** Q1=YES + Q2=YES (PoC PASSED 2026-07-09) + Q4=YES; **Q3=NO (bounded)** — M1 ceiling failure on the real-sim validation (Proposal 001 T4, 2026-07-09). **CLOSED — validated, NOT Super-GOAT.** Plan 412 ships DEFAULT-ON on its own GOAT merits. **Related:** Research 393, Plan 412 (shipped, DEFAULT-ON — the open primitive), Plan 301 (subspace_phase_gate), Proposal 001 (`riir-ai/proposals/001_block_sparse_hla_q3_real_game_validation.md`)

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

**❌ ANSWERED = NO (bounded) — 2026-07-09.** See "Q3 verdict" section below. The pre-registered
M1 decision rule fired its negative branch: on real civ-sim traces, the block-sparse read
adds no separability the existing flat-5 production projection doesn't already have (all
competitors hit AUC=1.0 ceiling on S1-vs-S2; `AUC(C2)−AUC(C0)=0.0 < 0.15`). The confounded
(S3) regime was degenerate. Architectural analysis confirms the behavior selectors M2/M3
would measure are emotion-blind. Per acceptance "when any of Q1–Q4 is NO" → closed as
"validated — not Super-GOAT".

**Original (now-resolved) question:** Candidate selling point: *"Our NPCs have multidimensional
emotional posture — a 'fearful' NPC can be predator-fearful or starvation-fearful, and the
designer steers within the fear region to produce distinct behaviors."* This is sellable IF
the block structure validates on real NPC behavior traces.

**Resolution:** ran the Proposal 001 validation on real civ predator/prey/hunger sim traces
(no recorded Seal Online emotion traces exist — only quest/item content). The selling point
was NOT validated: the natural-sim traces are trivially separable by flat representations,
and the observable behavior layer is emotion-blind at the decisive selectors.

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

## Q3 verdict (2026-07-09) — NO (bounded)

The real-game-data validation ran via [Proposal 001](../../riir-ai/proposals/001_block_sparse_hla_q3_real_game_validation.md):
T1–T3 built a production-faithful trace harness over the real `riir-games::civ`
predator/prey/hunger sim (`compute_force` + `ForceScratch::fear_accum` + per-tick
hunger evolution), 18,000 labeled frames across 3 scenarios (S1 predator, S2
starvation, S3 confounded), with four competitors (C0 floor-6, C1 flat-5, C2
block-2 via Plan 412, C3 oracle). T4 ran the pre-registered M1 classification.

**The M1 result fired the negative branch of the pre-registered decision rule.**

| Task | C0 floor-6 | C1 flat-5 | C2 block-2 | C3 oracle | Verdict |
|---|---|---|---|---|---|
| S1-vs-S2 AUC | 1.0000 | **1.0000** | 1.0000 | 1.0000 | ceiling — all tied |
| S3 contrib AUC | 0.5000 | 0.5000 | 0.5000 | 0.5000 | degenerate (chance) |

- **Pre-registered threshold `AUC(C2)−AUC(C0) ≥ 0.15` → FAIL (delta = 0.0000).**
- **Flat-6 suffices (C0 AUC = 1.0 ≥ 0.85) → YES**, and stronger: **C1 flat-5
  (the existing production single-scalar projection) also AUC = 1.0**. The
  block adds zero separability a representation already shipping in production
  doesn't already have.
- **Why the ceiling:** the scenarios contrast predator-PRESENT (S1, fear_fx ≠ 0)
  vs predator-ABSENT (S2, fear_fx = 0), which is trivially separable by ANY
  representation. The hard case the block was designed for — predator-fear vs
  starvation-fear at *matched* flat-fear — is not what these scenarios produce.
- **S3 (confounded) was degenerate:** `pos_test = 0` — the prey flees the
  predator within a few ticks, so predator-fear and starvation-fear are
  temporally separated (predator early, starvation late), never co-active. The
  sim does not sustain a confounded regime, so contribution separation is
  untestable on these traces (oracle also at 0.5 → fixture cannot test it).

**T5 (M2 designer steerability) + T6 (M3 downstream observability) were deferred**
as moot: the pre-registered decision rule makes M1 failure → Q3 = NO (the M2/M3-
dependent rows all require M1 to PASS first). Architectural corroboration: the
observable behaviors M2/M3 measure are emotion-blind at the decisive layers —
`CivAction` selection (`leo_act.rs` reads Q-values + physical positions; grep for
`fear|hunger|emotion|curiosity|lambda` returns ZERO hits in that file) and flee
physics (`compute_force`, predator-position-driven). The one emotion→behavior
path that exists (emotion scalars → `motivation.rs` stat_vec idx 7 →
`MotivationState`) is indirect and does not reach `leo_act`'s action selector.
Block steering changes internal projections but cannot produce distinct
*observable* NPC behaviors — designer-*visible*, not designer-*valuable*.

**This is the honest "designer-visible not designer-valuable" outcome the
proposal's decision rule explicitly allowed** (the M3-fail row), reached via the
M1 ceiling rather than M3. Running T5/T6 would confirm it but cannot overturn a
Q3=NO from M1 failure per the pre-registered contract.

**What this does NOT invalidate:**
- **Q2 stands** — block-sparse HLA CAN represent a decoupled fear-subtype that
  flat-5 provably cannot, at the *representation* level (synthetic PoC, T2
  decisive test, bit-identical flat inputs). That is a real information-
  theoretic capability.
- **Plan 412 stands** — `SubspaceSteeringField` is a GOAT open primitive
  (DEFAULT-ON) on its own merits (k-dim manifold steering, K=1 parity with Plan
  309, zero-alloc, orthonormality gate). It is unaffected by this fusion
  verdict.
- **What Q3=NO bounds:** the *fusion* — reframing HLA as block-sparse concept
  subspaces — does not elevate to Super-GOAT because (a) natural game traces
  don't require it (flat-5 suffices) and (b) no behavior consumer currently
  turns the block's representational advantage into observable NPC behavior.

**Re-elevation path (explicit, not pursued):** Q3 could be re-tested if (a) the
scenario harness is redesigned to construct S1-vs-S2 at *matched flat-fear* (so
flat-5 cannot trivially separate), AND (b) a behavior consumer is wired to read
the block (a fear-type → flee/forage router in `leo_act` or the motivation→action
path). Both are implementation tasks (a new plan) that presuppose the capability
this validation was meant to justify — so they are out of scope for this issue.

## Progress tracker

- [x] **Q1** — No prior art? **YES** (novel, 2026-07-09)
- [x] **Q2** — New class of behavior? **YES** (PoC PASSED 2026-07-09; defend-wrong caveats documented)
- [x] **Q3** — Product selling point? **NO (bounded)** (M1 ceiling failure on real-sim validation, 2026-07-09)
- [x] **Q4** — Force multiplier? **Likely YES** (≥8 cousin plans — 412 × 301 × 297 × 320 × 319 × HLA kernel × KarcShard × 251)

**Q1–Q4 NOT all YES → per acceptance, the Super-GOAT guide is NOT created and the issue is
closed as "validated — not Super-GOAT".** Plan 412 stands on its own GOAT merits (DEFAULT-ON).

## Acceptance

When Q1–Q4 are all answered YES with evidence:
1. Create `riir-ai/.research/NNN_block_sparse_hla_supergoat_guide.md` with the full Super-GOAT mandatory outputs (TL;DR, distilled primitive, connection map, latent-vs-raw boundary, what stays private, validation protocol, implementation priority P0–P3).
2. Open `riir-ai/.plans/MMM_block_sparse_hla_wiring.md` for the runtime integration.
3. Close this issue with a pointer to the guide.

When any of Q1–Q4 is NO:
1. Document the negative result in this issue.
2. The open primitive (Plan 412) stands on its own GOAT merits; the HLA fusion does not elevate to Super-GOAT.
3. Close this issue as "validated — not Super-GOAT".

**▶ Resolution (2026-07-09): the NO-branch executed.** Q3 = NO (bounded). The negative
result is documented in the Q3 verdict section above. Plan 412 stands on its own GOAT
merits (DEFAULT-ON, commit `7bff9cc7`); the block-sparse HLA fusion does NOT elevate to
Super-GOAT. This issue is **CLOSED — validated, NOT Super-GOAT**. No Super-GOAT guide is
created; no wiring plan is opened. (The anti-deferral note's one-session budget was honored:
Q1+Q2 resolved 2026-07-09, Q3 resolved 2026-07-09 in the same window after Plan 412 shipped.)

## Anti-deferral note

This is a **validation task**, not an implementation task. It does not block Plan 412 (the open primitive ships regardless). It blocks ONLY the Super-GOAT guide creation. Do not let it stall — if Q1–Q4 cannot be answered within one focused session after Plan 412 ships, default to "not Super-GOAT" and close.
