# Research 408: TILR — Trajectory-Invariant Latent Refinement

> **Source:** [TILR: Trajectory-Invariant Latent Refinement](https://arxiv.org/abs/2606.29164) — Malarkkan, et al. (Arizona State University), ICML 2026 Mech Interp Workshop, arXiv:2606.29164
> **Date:** 2026-07-10
> **Status:** Active — GOAT verdict, plan filed
> **Related Research:** 406 (Spectral Rewiring — SVD projection onto base subspace, the closest cousin), 357 (Neural Procedural Memory — PASS, contrastive PCA already shipped), 397 (MAG — unsupervised direction mining), 393 (Block-Sparse Featurizer → Plan 412 Subspace Steering), 389 (CHaRS — cluster-aware steering via OT), 053 (CNA — contrastive neuron attribution), 114 (river-valley diagnostics — `subspace_ratios` ships the γ ratio)
> **Related Plans:** 423 (spectral_rewire — weight-delta SVD projection), 412 (subspace_steering — k-dim block), 416 (region_subspace — MFA local subspaces), 152 (river_valley — `subspace_ratios` diagnostic), 087 (CNA), 418 (MAG), 309 (latent_field_steering), 301 (subspace_phase_gate — the SVD primitive), 425 (this primitive)
> **Classification:** Public

---

## TL;DR

TILR is a **100% training-free, forward-pass-only** refinement that (1) collects contrastive differences `δ_t = h_good_t − h_bad_t` from a frozen reference pair (two checkpoints, or a freeze/thaw pair), (2) runs truncated SVD on the stacked differences to discover a low-rank **invariant subspace** `S_r` (the paper finds r≈12 at τ=0.90 variance, often r=1 suffices), then (3) at inference replaces unconstrained contrastive updates with a **subspace-projected** correction `h_t = h_tilde + η_t · Π d_t` whose step size is gated by an **adaptive alignment gate** `γ_t = ‖Πd_t‖ / ‖d_t‖ ∈ [0,1]` so that `η_t = η_base · γ_t`. When `γ→0` (correction signal outside the invariant subspace), the correction vanishes and the output bit-recovers the uncorrected backbone — a strict **no-harm guarantee**. Validated on GPT-2 base/medium and Qwen2.5-Math-1.5B: ~10% answer-consistency improvement under paraphrase, 25–53% latent trajectory variance reduction, +2.1% accuracy over unconstrained refinement, 72–74% checkpoint-pair sensitivity reduction.

**Distilled for katgpt-rs (modelless, inference-time):**

The three TILR operations decompose into primitives we already ship, plus one integration that we do not:

| TILR operation | Shipped equivalent | Gap |
|---|---|---|
| Contrastive difference `δ_t = h_good − h_bad` | **CNA** (Plan 087), **NPM** (R357 — PASS), **MAG** (Plan 418) | None — already shipped, in pure latent space |
| Truncated SVD on differences → invariant subspace `U_r` | `thin_svd_into` + `SvdResult` in **`subspace_phase_gate.rs`** (Plan 301) | None — the SVD primitive ships, used by Plan 423 |
| Subspace projection `Π d_t = U_r U_r^T d_t` | **`spectral_rewire.rs`** (Plan 423) projects `ΔW` onto `U_r V_r^T`; **`subspace_steering.rs`** (Plan 412) applies `s + Σ α_j u_j` | **GAP** — both project onto a subspace, neither *gates the step size by the alignment ratio* |
| Alignment gate `γ_t = ‖Πd_t‖ / ‖d_t‖` | **`river_valley::subspace_ratios`** (Plan 152) computes `r_dom = ‖U_k^T g‖ / ‖g‖` — **the identical metric**; `spectral_rewire::on_manifold_fraction` is the same ratio for weight deltas | **GAP** — both ship the ratio as a *diagnostic output*, neither uses it to *modulate the step size* `η_t = η_base · γ_t` |
| Adaptive step `η_t = η_base · γ_t` (graceful degradation) | — | **NOVEL COMBINATION** — no shipped primitive modulates correction strength by the subspace-alignment ratio with a strict `γ→0 ⇒ uncorrected` no-harm guarantee |

The transferable insight that is NOT yet shipped: **an alignment-gated projected correction primitive** `tilr_refine_into(state, direction, basis, eta_base, scratch)` that (a) projects the direction onto a frozen SVD basis, (b) computes the alignment fraction, (c) applies `state += eta_base * gamma * projected`, and (d) guarantees `gamma→0` bit-recovers the input. Every piece exists in isolation; the integration as a single zero-alloc runtime gate does not.

---

## 1. Paper Core Findings

### 1.1 The thesis

Contrastive steering (CAA / ActAdd / difference-in-means) applies a correction direction at inference, but the correction is **unconstrained** — it lives in whatever subspace the contrastive differences happen to span, including input-specific noise. TILR's claim: contrastive differences between a *good* and *bad* reference model are **highly low-rank** (the reasoning-quality variance concentrates in r≈12 directions, often r=1), and projecting the per-instance correction onto this **invariant subspace** both improves quality (removes off-subspace noise) and flattens sensitivity to the choice of reference pair.

### 1.2 The five-step mechanism

1. **Contrastive difference collection.** Frozen backbone `f`, reference pair `(f_good, f_bad)` (e.g. two checkpoints from different training epochs). For each calibration input `x_i`, run `f` with residual blending `α` to get intermediate states `h_tilde_t`, then pass each through both reference models to get `δ_t = h_good_t − h_bad_t`. N=200 calibration inputs, T steps each. No labels needed.

2. **Invariant subspace identification.** Stack all `N·T` differences column-wise into `Δ ∈ ℝ^(d×NT)`. Truncated SVD: `Δ ≈ U_r Σ_r V_r^T`, where `r` retains `τ=0.90` of variance. The invariant subspace `S_r` is defined by projection `Π = U_r U_r^T`. **Key empirical finding:** r_0.90 averages 12; many datasets need only 1 direction. Principal-angle analysis vs input PCA confirms `S_r` captures *reasoning-quality* variance, not generic input variance.

3. **Subspace-constrained update.** At inference, replace unconstrained contrastive update with projected version: `h_t = h_tilde_t + η_t · Π d_t`, where `d_t` is the per-instance contrastive direction.

4. **Adaptive alignment gate.** `γ_t = ‖Πd_t‖ / (‖d_t‖ + ε) ∈ [0,1]`. Measures fraction of correction signal within `S_r`. Step size: `η_t = η_base · γ_t`. When `γ→1`, full projected correction applied. When `γ→0`, correction vanishes — **graceful degradation bit-recovers the uncorrected backbone exactly** (no-harm guarantee). Note: `α` (residual blend) is NOT gated — it controls backbone memory-vs-computation, independent of correction reliability.

5. **Computational cost.** One matrix-vector product `O(dr)` per step (d=768, r≤20 → negligible vs `O(d²)` transformer forward). One-time calibration: `NT` reference forward passes + truncated SVD (<30s on H100). <3% wall-clock overhead.

### 1.3 Results (GPT-2 base/medium, Qwen2.5-Math-1.5B)

- ~10% answer consistency improvement under paraphrase
- 25–53% latent trajectory variance reduction (mean 39%)
- +2.1% average accuracy over unconstrained refinement
- 72–74% reduction in checkpoint-pair sensitivity (on 3/4 datasets)
- The SVD is on **contrastive differences**, not raw embeddings — captures reasoning-quality variance, not lexical/style variance

### 1.4 What is genuinely novel (vs already-shipped)

- **Graceful degradation as a first-class contract.** The `η_t = η_base · γ_t` gate makes "no-harm" a *bit-identical* guarantee: at `γ=0`, the correction is exactly zero, not approximately zero. Our `subspace_ratios` computes this ratio but does not wire it into a step-size gate.
- **Subspace-mediated input invariance.** Two inputs whose contrastive directions differ only outside `S_r` get *identical* projected corrections — a form of trajectory invariance that raw contrastive steering cannot offer.
- **Sensitivity flattening.** Low-`γ` inputs become insensitive to `η_base` choice (quadratic suppression `O(γ²)`), so a single `η_base` suffices across diverse inputs.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | ≥2 codebase equivalents |
|---|---|
| contrastive difference `δ_t` | **direction vector** (HLA), `EmotionDirections` delta, `ContrastivePairProvider` output (Plan 087), MAG `mine_contrast_direction` (Plan 418) |
| invariant subspace `U_r` | **`style_weights[64]`** basis (NeuronShard), `SvdResult.right_singular_vectors` ("semantic axes", Plan 301), `SubspaceSteeringField.block` (Plan 412) |
| subspace projection `Π = U_r U_r^T` | `spectral_rewire::project_core` (Plan 423), `SubspaceSteeringField::apply` (Plan 412), dot-product projection onto `EmotionDirections` |
| adaptive alignment gate `γ_t` | **`river_valley::subspace_ratios` → `r_dom`** (Plan 152, identical metric), `spectral_rewire::on_manifold_fraction` (Plan 423), sigmoid gate (used everywhere but not on this ratio) |
| reference pair `(f_good, f_bad)` | **freeze/thaw pair** (`MerkleFrozenEnvelope`), `KarcShard`/`ArchetypeBlendShard` snapshots, two epoch checkpoints |
| residual blending `α` | **leaky integrator**, `carry-forward`, `LatentSteeringVector.alpha` (Plan 309) |
| trajectory variance | **coherence**, **staleness**, `latent_functor/reestimation.rs` coherence decay |
| graceful degradation / no-harm | "graceful degradation" pattern (shipped in many primitives — Plan 025, 237, 316 — but not as a *bit-identical γ→0 contract* on a correction gate) |
| truncated SVD | `thin_svd_into` + `SvdScratch` + `SvdResultScratch` (**`subspace_phase_gate.rs`**, Plan 301) |

### 2.2 Fusion grep results (both layers, all 5 repos, both vocabularies)

The grep for `invariant.subspace|contrastive.difference|TILR|trajectory.invariant` returned **ZERO** paper-vocabulary hits in `.research/` notes (the only hit was `extract_chars_anchor_bank(trajectory)` in R389 CHaRS, which uses clustering not SVD). The codebase-vocabulary grep, however, surfaced **10 close cousins**:

| Cousin | Mechanism | Overlap with TILR | Difference |
|---|---|---|---|
| **Plan 423 spectral_rewire** (R406) | SVD-project a weight delta `ΔW` onto base `U_r V_r^T` | Subspace projection + `on_manifold_fraction` = γ | Operates on **weight deltas** (offline); TILR on **latent state** (online per-step). Does NOT gate step size by the fraction. |
| **Plan 152 river_valley `subspace_ratios`** | `r_dom = ‖U_k^T g‖ / ‖g‖` — **the identical γ metric** | Computes γ exactly | **Diagnostic only** — never wired into a step-size gate `η_t = η_base · γ_t`. |
| **Plan 412 subspace_steering** (R393) | k-dim block steering `s + Σ α_j u_j` | Subspace-projected correction on latent state | `α_j` are **fixed per-axis strengths**, NOT adaptive to the per-instance alignment fraction. No γ gate. |
| **Plan 416 region_subspace** (R396) | MFA region-conditioned subspaces | Per-region subspace projection | Sigmoid membership gates `g_k(x)`, not alignment-fraction γ. Region-conditioned, not correction-gated. |
| **R357 NPM** (Neural Procedural Memory) | PCA first-PC of contrastive differences → steering vector | Contrastive difference + PCA (= truncated SVD) | **PASS verdict** — already shipped, stronger in latent space (HLA 8-dim not 4096-dim). No γ gate. |
| **Plan 418 MAG** (R397) | Unsupervised direction mining via contrast | Contrastive direction extraction | Direction *mining*, not *gated correction*. No subspace, no γ. |
| **Plan 087 CNA** (R053) | Contrastive neuron attribution + modulation | Contrastive difference → sparse circuit | Modulates *neurons* (top-0.1%), not a *subspace-projected latent correction*. No γ. |
| **R389 CHaRS** | Cluster-aware steering via Sinkhorn OT | Input-adaptive steering vector | Uses **clustering + OT**, not **SVD + γ gate**. Different mechanism. |
| **Plan 309 latent_field_steering** | 1D `s + α·v` injection | Additive latent correction | Single direction, fixed `α`, no subspace, no γ. |
| **Plan 405 spherical_steering** | Slerp toward single target | Additive correction | Single-target Slerp, no subspace, no γ. |

**Conclusion of the fusion grep:** every TILR component ships in isolation, but the specific integration — *SVD-discovered subspace + alignment-fraction γ gate + bit-identical γ→0 graceful degradation, applied as an online per-step latent-state correction* — is **not shipped**. This is a Gain-to-GOAT level contribution: a principled integration of existing pieces into a single primitive with a provable no-harm contract.

### 2.3 Latent-space reframing (mandatory per workflow §1 step 3)

The paper operates on GPT-2 / Qwen2.5 residual streams (768-d / 2048-d). Re-cast each TILR operation on the codebase's latent-state kernels:

**(a) HLA per-NPC latent state (8-dim, `riir-ai/crates/riir-engine/src/hla/`)**

The reference pair becomes two HLA snapshots — e.g. the NPC's `style_weights` at two different archetype commitments, or two freeze/thaw versions of the NPC's personality shard. The contrastive difference `δ = h_good − h_bad` is an 8-d vector. SVD over many such differences discovers the 1–3 dominant "reasoning-quality" axes within the 8-d affect space (valence/arousal/desperation/calm/fear + 3 reserved). The γ gate then says: "only apply the personality correction if the current contrast direction aligns with the NPC's invariant personality subspace; otherwise leave the HLA state untouched." This is a **per-NPC no-harm personality refinement** — the NPC's behavior is only steered when the steering signal is "on-personality", and bit-recovers the unsteered state otherwise. This composes naturally with `CommittedFieldBlend` (Plan 321) and `EmotionDirections::project`.

**(b) `latent_functor/` operations (`riir-ai/crates/riir-engine/src/latent_functor/`)**

The invariant subspace `U_r` becomes a per-functor "convergence basis" — the directions along which the functor's coherence actually improves. The γ gate modulates the `reestimation.rs` step size: `η_t = η_base · γ_t` means the re-estimation only fires when the correction is within the functor's well-behaved subspace. This is a **principled replacement** for the scalar coherence threshold `tau_reest` — instead of a binary "coherence < tau → re-estimate", it's a continuous "re-estimate proportional to the subspace alignment of the drift". Composes with `reestimation.rs` and `reestimation_steerer.rs`.

**(c) `cgsp_runtime/` curiosity signals (`riir-ai/crates/riir-engine/src/cgsp_runtime/`)**

Curiosity = prediction error. TILR reframing: the invariant subspace `U_r` is the set of directions in which the NPC's world model is *reliably predictive*. A prediction error `d_t` that lies mostly within `U_r` (high γ) is a "trustworthy surprise" — the NPC should update aggressively. A prediction error outside `U_r` (low γ) is "off-model noise" — the NPC should ignore it (no-harm). This is a **curiosity reliability gate**: `curiosity_signal *= γ_t`. Composes with `curiosity_class_router.rs` and the `pulse_bridge.rs`.

**(d) `NeuronShard` style_weights / freeze envelope / consolidation (`riir-neuron-db/src/`)**

The reference pair is two shard versions — e.g. pre-consolidation and post-consolidation `style_weights[64]`, or two `KarcShard` freeze snapshots. The contrastive difference `δ ∈ ℝ^64` captures what the consolidation/freeze cycle *changed*. SVD over many such deltas discovers the "consolidation-invariant subspace" — the directions that consolidation reliably affects. The γ gate then says: "only apply a shard correction if it lies within the consolidation-invariant subspace." This is a **freeze/thaw no-harm refinement** — shard corrections that would drift outside the consolidated subspace are suppressed. Composes with `MerkleFrozenEnvelope` (`freeze.rs`), `consolidation.rs`, and the `can_freeze` gate (`phase_gate.rs`).

**(e) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-dec/`)**

The invariant subspace `U_r` is the **harmonic subspace** of a `hodge_decompose` — the divergence-free, curl-free component that represents the "stable" part of a flow field. A correction `d_t` to a `DecFlowField` is gated by its alignment with the harmonic subspace: `η_t = η_base · γ_harmonic`. This is a **manifold-stable flow correction** — only corrections that preserve the harmonic structure are applied. **Curse-of-dimensionality caveat (AGENTS.md):** this DEC reframing is valid only for d ≤ 3 (2D maps, 3D belief regions). Do NOT apply to high-dim shards (d=64) or HLA (d=8 is borderline). Composes with `flow.rs::DecFlowField` and `hodge.rs::harmonic_projector`.

### 2.4 Fusion opportunities (the highest-value combinations)

- **F1: TILR × `spectral_rewire` (Plan 423) × freeze/thaw.** Use SAR to discover the invariant subspace from freeze/thaw weight deltas, then apply TILR's γ-gated correction to the latent state at runtime. SAR gives the subspace; TILR gives the gated application. This closes the loop from offline weight-space SVD to online latent-space correction.
- **F2: TILR × `reestimation.rs` (coherence-driven re-estimation).** Replace the scalar `tau_reest` threshold with `η_t = η_base · γ_t` — re-estimate proportional to subspace alignment, not binary threshold. This is a continuous, principled replacement for the current binary gate.
- **F3: TILR × `CommittedFieldBlend` (Plan 321).** The γ gate becomes a per-NPC "personality alignment gate" — only apply personality blend corrections that align with the NPC's committed archetype subspace. No-harm = "don't drift the NPC off-personality."

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. Architectural guide → riir-ai/.research/. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | One-line note. No files created in this session. |

### Verdict: **GOAT** (not Super-GOAT)

**One-line reasoning:** Every TILR component (contrastive SVD, subspace projection, γ alignment ratio) already ships in isolation; TILR's contribution is the *principled integration* as an alignment-gated correction with a bit-identical no-harm contract — a provable quality/robustness gain over unconstrained contrastive steering, but not a new capability class.

**Novelty gate (Q1–Q4):**

1. **No prior art?** ⚠️ **PARTIAL.** The specific *combination* (SVD-on-contrastive-trajectory-differences + alignment-gated subspace projection + bit-identical γ→0 graceful degradation) is not shipped as a single primitive. BUT every component ships: `spectral_rewire` (Plan 423) does the projection, `river_valley::subspace_ratios` (Plan 152) computes the γ ratio, `subspace_steering` (Plan 412) does subspace correction on latent state. The gap is the *integration as a step-size gate*, not a new mechanism. → Q1 = **NO** (not a clean "no prior art").
2. **New class of behavior?** **NO.** It's a more principled, more robust version of existing contrastive steering (CNA/NPM/MAG). It does not enable a capability no incumbent has — it makes the existing capability safer (no-harm) and less sensitive (checkpoint-pair flattening).
3. **Product selling point?** **WEAK.** "NPC latent reasoning invariant to input paraphrases" is a quality improvement, not a capability. Cannot finish "our NPCs do X no competitor can" — competitors with CNA + subspace steering have approximately the same capability.
4. **Force multiplier?** **YES.** Connects freeze/thaw (reference pairs), HLA (latent state), direction vectors (contrastive differences), sigmoid gating (the γ gate), and SVD primitives (Plan 301). But force multiplication alone does not make a Super-GOAT — it makes a strong GOAT.

**Q1=NO → verdict is GOAT (not Super-GOAT).** The user's preliminary assessment is **confirmed**. No private architectural guide needed; no Super-GOAT mandatory outputs triggered.

### MOAT gate per domain

| Domain | In scope? | MOAT contribution |
|--------|-----------|-------------------|
| **katgpt-rs** (public engine) | ✅ YES | Generic alignment-gated subspace projection primitive — a principled integration of existing SVD + subspace-ratio + steering pieces. Research-grade primitive for the adoption funnel. **Correct home for the open primitive.** |
| **riir-ai** (private runtime) | Conditional | The latent-state application (HLA no-harm personality refinement, functor γ-gated re-estimation) is a *consumer* of the open primitive — land in riir-ai only if a Super-GOAT guide is later warranted (needs GOAT-gate validation first). |
| **riir-neuron-db** (private shards) | Conditional | The freeze/thaw reference-pair integration is a *consumer* — land only if the shard-level GOAT gate validates. |
| **riir-chain** (private chain) | NO | No chain/LatCal/sync-boundary angle. |
| **riir-train** | NO | 100% modelless — no training dependency. |

### Why not Super-GOAT (the honest demotion)

The closest the codebase comes to TILR is the trio **Plan 423 (spectral_rewire) + Plan 152 (river_valley subspace_ratios) + Plan 412 (subspace_steering)**. A developer who reads those three plans and wires `η_t = eta_base * subspace_ratios(d, U_r).0` into a `SubspaceSteeringField::apply` call has built TILR in 5 lines. The *idea* is not moat-worthy; the *validated integration with a bit-identical no-harm contract* is a GOAT-tier quality gate, not a Super-GOAT moat. Claiming Super-GOAT here would be claiming novelty over a combination that is 90% assembled from shipped pieces — exactly the false-Super-GOAT failure mode the research skill §1.5 warns against.

---

## 4. Implementation routing

- **Open primitive** → `katgpt-rs/crates/katgpt-core/src/` (new module `tilr.rs` or fold into `subspace_steering.rs` as an extension). Feature flag `tilr_invariant_subspace`. Generic math, no game/chain/shard semantics.
- **Plan** → `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md`.
- **GOAT gate:** G1 (correctness — γ→0 bit-recovers input, projection preserves ranking), G2 (perf — overhead <3%, one `O(dr)` matvec per step), G3 (no regression), G4 (alloc-free projection path). **Not UQ-bearing** → no conformal floor needed.
- **Consumer wiring (deferred to follow-up issues, not this plan):**
  - riir-ai: HLA no-harm personality refinement (consumes the open primitive on 8-d HLA state).
  - riir-neuron-db: freeze/thaw reference-pair shard refinement (consumes on 64-d `style_weights`).
  - riir-ai: `reestimation.rs` γ-gated step size (replaces scalar `tau_reest`).

---

## 5. Modelless-unblock protocol check (§3.5)

TILR is 100% modelless (forward passes + SVD + projection + gate). No §3.5 deferral needed. The three modelless paths:
1. **Freeze/thaw** — the reference pair IS a freeze/thaw pair. ✅
2. **Raw/lora hot-swap** — the correction is a latent-space projection, not a weight mutation. ✅ (not needed, but compatible)
3. **Latent-space correction** — the γ-gated projection IS a latent-space correction. ✅

No riir-train dependency. No PoC required (no "already ships" parity claim — we're building the integration, not claiming coverage).

---

## TL;DR

TILR = alignment-gated subspace-projected contrastive correction with a bit-identical no-harm contract. Every piece ships (SVD via Plan 301, projection via Plan 423, γ ratio via Plan 152, subspace steering via Plan 412); the integration as a single γ-gated runtime correction primitive does not. **GOAT** — plan filed at `.plans/425_*.md`, feature flag `tilr_invariant_subspace`, open primitive in katgpt-rs. Not Super-GOAT because Q1 (no prior art) is NO — the combination is 90% assembled from shipped pieces, and a 5-line wiring of existing primitives reproduces the mechanism.
