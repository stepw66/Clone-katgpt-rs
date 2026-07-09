# Issue 049: Block-Sparse HLA — Super-GOAT Fusion Candidate Validation

**Date:** 2026-07-08
**Origin:** Research 393 §3 (Block-Sparse Featurizers distillation)
**Status:** Q1=YES + Q2=YES (PoC PASSED 2026-07-09) + Q4=YES; **Q3=NO (bounded)** — M1 ceiling failure + M2/M3 **measured** failures on the real-sim validation (Proposal 001 T4–T6, 2026-07-09). **CLOSED — validated, NOT Super-GOAT.** Plan 412 ships DEFAULT-ON on its own GOAT merits. **Related:** Research 393, Plan 412 (shipped, DEFAULT-ON — the open primitive), Plan 301 (subspace_phase_gate), Proposal 001 (`riir-ai/proposals/001_block_sparse_hla_q3_real_game_validation.md`)

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
(S3) regime was degenerate. T5 (M2) + T6 (M3) were subsequently **measured** (not just
predicted) and also failed decisively: M2 found C2=1 cluster (threshold ≥3), M3 found
0% displacement variation (threshold ≥30%). The Q3=NO verdict now rests on three
independent measured failures. Per acceptance "when any of Q1–Q4 is NO" → closed as
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
| S3 contrib AUC | **0.9987** | 0.7800 | **0.9987** | **0.9987** | non-degenerate (z-score) — block ties flat-6 |

- **Pre-registered threshold `AUC(C2)−AUC(C0) ≥ 0.15` → FAIL (delta = 0.0000).**
- **Flat-6 suffices (C0 AUC = 1.0 ≥ 0.85) → YES**, and stronger: **C1 flat-5
  (the existing production single-scalar projection) also AUC = 1.0**. The
  block adds zero separability a representation already shipping in production
  doesn't already have.
- **Why the ceiling:** the scenarios contrast predator-PRESENT (S1, fear_fx ≠ 0)
  vs predator-ABSENT (S2, fear_fx = 0), which is trivially separable by ANY
  representation. The hard case the block was designed for — predator-fear vs
  starvation-fear at *matched* flat-fear — is not what these scenarios produce.
- **S3 (confounded) contribution separation — non-degenerate after label fix,
  block STILL ties the trivial fix.** The raw `|fear_fx| > hunger` label is
  degenerate (predator force ~0.005–0.33 vs hunger ~0.3–1.0 → only 1/6000 S3
  frames satisfy it; the prey escapes the predator within ~15 ticks so
  predator-fear is transient while hunger accumulates). The committed T4 code
  (commit `eecb6425`) uses a **scale-comparable z-score label** instead: each
  contribution axis is z-scored within the S3 population and a frame is
  "predator-elevated" iff `z(|fear_fx|) > z(hunger)` (~45% positive,
  non-degenerate). Result: **C1 flat-5 = 0.7800** (single fear scalar cannot
  see the hunger axis at all — the flat-5 collapse), but **C0 floor-6 = C2
  block-2 = C3 oracle = 0.9987**. **C2 block advantage over C0 = 0.0000** —
  the block ties the trivial fix (add a hunger scalar) even on the
  properly-constructed contribution-separation task. Predator and starvation
  signals already live on orthogonal raw dims (`fear_fx` vs `hunger`) in this
  sim, so the block-sparse confound thesis has nothing to separate by
  construction.
- **This *strengthens* the NOT-Super-GOAT finding.** The earlier `pos_test = 0`
  degenerate read (commit `65991228`, now superseded) was a label artifact —
  the corrected non-degenerate task confirms the block adds nothing even when
  the test is fair.

**T5 (M2 designer steerability) + T6 (M3 downstream observability) — MEASURED.**
Originally deferred as moot (M1 failure → Q3=NO per the pre-registered sequential gate).
Promoted from deferred to done because (a) the repo rule "Dont defer benchmark task"
explicitly forbids deferring benchmarks, (b) M1 was uninformative-at-ceiling rather than
cleanly negative — the verdict rested on architectural *prediction* not measurement, and
(c) M2 sidesteps the M1 ceiling by construction (matched flat-fear = 0.7, not the flawed
S1/S2 predator-present-vs-absent scenarios). T5/T6 were run as actual measurements
(2026-07-09, same session) and **confirm the architectural prediction decisively**:

**M2 (designer steerability) — MEASURED FAIL.** At matched flat-fear = 0.7, 3 hunger
postures × alpha grid (25 for 2-DOF competitors, 5 for C1). DBSCAN eps=0.05, min_pts=5.
| Competitor | Samples | Clusters | MaxFréchet | Fréchet σ |
|---|---|---|---|---|
| C0 floor-6 | 75 | **1** | 0.000000 | 0.000000 |
| C1 flat-5 | 15 | **1** | 0.000000 | 0.000000 |
| C2 block-2 | 75 | **1** | 0.000000 | 0.000000 |
| C3 oracle | 75 | **1** | 0.000000 | 0.000000 |
Both pre-registered thresholds FAIL: cluster `C2 ≥ 3` scored `C2 = 1`; Fréchet
`max(C2) > 2σ(C1)` scored `0.0 > 0.0` (false). C2 ties C0 and C1 — the block adds zero
behavioral clusters. **Root cause (mechanism, measured):** `SubspaceSteeringField::apply`
does `state[j] += Σ_k α_k · block[k][j]`; the committed predator axis `[-1,0,0,0,0,0]`
modifies `fear_fx` only, the starvation axis `[0,0,1,0,0,0]` modifies `hunger` only —
neither touches `fear_fy` (dim 1). At matched flat-fear, `fear_fy = 0` and `fear_fx < 0`,
so `atan2(0, negative) = π` always; the block structure cannot produce angular variation
through alpha-steering.

**M3 (downstream observability) — MEASURED FAIL.** All 75 steered C2 postures were fed
into the REAL civ `compute_force` simulation (50 ticks each). **All 75 trajectories are
bit-identical** (verified tick-by-tick via `f32::to_bits()`). Displacement variation =
0.0% (threshold ≥30% → FAIL). **Root cause (mechanism, measured):** `compute_force` takes
positions as input and produces `fear_fx` as output; alpha-steering modifies the output,
not the input — the sim recomputes `fear_fx` from predator position each tick, overwriting
any perturbation. The steered emotion state cannot feed back into the trajectory.

**Verdict strengthened.** The Q3=NO conclusion now rests on **three independent measured
failures** (M1 ceiling + M2 cluster/Fréchet + M3 displacement), not on a single
uninformative M1 plus architectural prediction.

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
path), AND (c) the block basis is redesigned to include a `fear_fy` component so
alpha-steering can produce flee-angle variation (M2 measured the committed T1
predator axis `[-1,0,0,0,0,0]` is purely x-axis → flee-angle trapped at π). All
three are implementation tasks (a new plan) that presuppose the capability this
validation was meant to justify — so they are out of scope for this issue.

## Progress tracker

- [x] **Q1** — No prior art? **YES** (novel, 2026-07-09)
- [x] **Q2** — New class of behavior? **YES** (PoC PASSED 2026-07-09; defend-wrong caveats documented)
- [x] **Q3** — Product selling point? **NO (bounded)** (M1 ceiling + M2/M3 measured failures on real-sim validation, 2026-07-09)
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
