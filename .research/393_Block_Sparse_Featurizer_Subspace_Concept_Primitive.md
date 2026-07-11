# Research 393: Block-Sparse Featurizers — Subspace Concept Primitive (Manifold Steering)

> **Source:** [Uncovering Neural Geometry in Vision Models With Block-Sparse Featurizers](https://www.goodfire.ai/research/bsf-vision) — Goodfire, Jun 2026. Full paper: [arXiv:2606.25234](https://arxiv.org/abs/2606.25234). Code: [goodfire-ai/block-sparse-featurizer](https://github.com/goodfire-ai/block-sparse-featurizer).
> **Date:** 2026-07-08
> **Status:** Active — GOAT verdict, plan filed
> **Related Research:** 290 (Latent Field Steering — the 1D cousin this generalizes), 279 (Diffusion ≡ Subspace Clustering — Jacobian SVD discovers the basis), 299 (Clifford wedge — pairwise bivector, not block-featurizer), 276 (PersonalityWeightedComposition — sigmoid blend of N direction vectors), 144 (Functional Emotions — read-side direction vectors), 039 (SpectralQuant — offline eigenbasis), 143 (Latent Terms SAE — the SAE-rejection precedent), 053 (CNA — "don't implement SAE"), 312 (Viable Manifold Graph — walks samples, not within a subspace)
> **Related Plans:** 309 (Latent Field Steering — 1D), 301 (subspace_phase_gate — discovers basis, reduces to scalars), 320 (Indicator Probe Bank — multi-direction with block similarity), 319 (Clifford geometric product), 322 (Phase-Modulated Coupling — 2-subspace rotation), 412 (this primitive's plan), 162 (Emotion Vector)
> **Classification:** Public

---

## TL;DR

Goodfire's **Block-Sparse Featurizers (BSF)** decompose model activations into multidimensional **subspaces** (blocks) rather than 1D directions, enforcing sparsity at the *block* level (a few blocks fire per input, not a few individual features). Empirically, vision-model concepts are **2–4 dimensional manifolds** (stable rank rarely exceeds 4 even with block size 16), and keeping the block as the unit of meaning enables **manifold steering** — walking *within* a concept region to generate controlled variations, instead of turning a 1D knob up/down.

**Distilled for katgpt-rs (modelless, inference-time):** the BSF *training* (encoder/decoder fit to minimize reconstruction error) is riir-train territory — we do not train featurizers here. The transferable modelless primitives are: (1) **subspace steering** — generalize `LatentSteeringVector` (Plan 309, 1D `direction: Vec<f32>` + scalar `α`) into a k-dim block form `s' = s + Σ_j α_j · u_j` where `{u_1..u_k}` span the concept subspace, enabling walk-within-region steering; (2) **block-wise TopK activation consumption** — generalize `BlockTopKRouter` (VortexFlow, currently attention-token selection) into an activation-featurizer consumption pattern where the *block as a whole* is the feature; (3) **block-dimensionality diagnostic** — already ships as `effective_rank` / `stable_rank_update_into` (Plan 287, Roy-Vetterli standard, identical metric to BSF's "stable rank of a block").

**Why this matters here:** our entire latent-state substrate (HLA 8-dim, `LatentSteeringVector`, `EmotionDirections::project`, `PersonalityWeightedComposition`, `IndicatorProbeBank`, `analytic_lattice::direction_vector_decode`) is built on **1D direction vectors + dot-product + sigmoid**. BSF's empirical claim — "concepts are 2–4 dimensional manifolds, not lines" — implies these 1D projections are **systematically lossy** for any concept whose true geometry is ≥2D. A `SubspaceSteeringField` that carries a k-dim orthonormal block (discovered offline via Plan 301 Jacobian SVD, or constructed deterministically per §3.5) instead of a single direction is a strict generalization that subsumes the 1D case at `k=1` and unlocks manifold-walk steering at `k≥2`.

---

## 1. Paper Core Findings

### 1.1 The core thesis — concepts are manifolds, not lines

The most popular interpretability methods (SAEs, transcoders) treat each internal concept as a **single straight line** (one direction vector). Goodfire argues this is a limiting assumption: concepts like "tree" (sapling → oak → redwood → bonsai, green → red → bare, birch → pine → willow) vary along **many axes simultaneously** — they are regions in a multidimensional space, not points on a line. A toy model (left column: ground-truth manifolds stacked in superposition) shows a standard SAE fragments the manifolds across many redundant 1D features (middle), while BSF recovers them coherently (right).

### 1.2 Block-Sparse Featurizers — the mechanism

A BSF decomposes activations into **blocks**, where each block is the set of directions spanning one subspace:

```
encoder: x ∈ R^n  →  [B_1, B_2, ..., B_K]   where B_k ∈ R^g (block size g)
sparsity: block-wise TopK — select the k blocks with largest L2 norm ‖B_k‖
decoder: x_hat = Σ_{k ∈ topK} W_dec_k · B_k
loss: reconstruction error ‖x − x_hat‖²
```

Three variants share the generic architecture, differing in encoder/decoder/activation:
- **vanilla BSF** — linear encoder `xW + b`, block-wise TopK on L2 norm.
- **Grassmannian BSF** — encoder produces points on the Grassmannian (subspace manifold).
- **group-lasso BSF** — group-lasso regularization (`Σ_k ‖B_k‖₂`) as the structured-sparsity penalty.

The paper's central empirical finding: **the principle of block-sparsity matters more than the implementation** — all three variants recover concept manifolds, and the same principle improves concurrent methods (MFA, SMixAE).

### 1.3 What BSF finds in vision models (DINOv3, SDXL)

- **Visual concept manifolds** — block activations form dense connected clouds, not lines. Example: the "arch" block has interpretable internal structure (red = bottom of arch → yellow = middle → green = top), smoothly tracing the concept's variation.
- **Manifold steering** — because a block gives a *region* (not a point), you can move around inside it. The "pretzel manifold" in SDXL: walking point-by-point through one subspace generates pretzels twisted/braided/knotted differently. A 2D grid over the region sweeps hat brim angle × crown shape, or coffee milk pattern × spoon location.
- **Curve detectors reveal Fourier harmonics** — in InceptionV1, BSF shows the classic "curve detector" neurons + SAE features are fragmented views of a continuous rotation subspace. The block also surfaces **higher-order Fourier harmonics** as symmetries: 1st harmonic = full 360°, 2nd = 180°-periodic (peaks at 0° and 180°), etc. This explains Olah et al.'s observation that some curve neurons wrap every 180° — they read off the 2nd harmonic.
- **Stable rank distribution** — across 32,000 concepts in DINOv3, the stable rank of blocks rarely exceeds 4 (block size was 16). **Most vision concepts are 2–4 dimensional.** This is "perhaps unsurprising: visual concepts are 2D projections of a three-dimensional scene."

### 1.4 What is training-only (→ riir-train, do NOT distill here)

- The encoder/decoder **training** (minimizing reconstruction error over a corpus) is gradient descent through weights. This is the dictionary-learning training loop — out of scope for this workflow.
- The three BSF **variants** (vanilla, Grassmannian, group-lasso) are training-method distinctions. → riir-train if we ever train our own featurizers.
- The DINOv3 / SDXL **empirical findings** (stable rank distribution, curve detector harmonics) are properties of *trained vision models* — not directly transferable to our HLA/functor substrate without an analogous trained featurizer.

**The modelless-unblock question (§3.5):** can the block structure be **deterministically constructed** rather than trained? Yes, partially — see §2.2. The block *basis* can come from (a) Plan 301 Jacobian SVD (runtime-discovered, modelless), (b) SpectralQuant offline eigenbasis (R039, modelless), or (c) hand-constructed orthogonal sets (e.g., the 5 HLA axes already form a partial block). What cannot be constructed modellessly is the **reconstruction-optimal** block partition — that requires the training loop. We consume pre-discovered blocks; we do not train the featurizer.

---

## 2. Distillation

### 2.1 Transferable primitives (stripped of training setup)

| # | Primitive | Transferable insight | Existing closest cousin |
|---|-----------|---------------------|------------------------|
| **A** | **Subspace steering field** | Generalize `LatentSteeringVector` from 1D `direction: Vec<f32>` + scalar `α` to a k-dim orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`. Steer by `s' = s + Σ_j α_j · u_j`. At `k=1` this is identical to Plan 309; at `k≥2` it enables **manifold walking** (sweep `α` across a grid to generate concept variations). | `LatentSteeringVector` (Plan 309, shipped, **1D only**). `Phase-Modulated Coupling` (Plan 322, rotates within a 2-subspace `(a,b)` plane — closest existing 2D case, but uses cos/sin weights not per-axis `α_j`). `Spherical Steering` (Plan 405, Slerp toward a single target — single-target, not walk-within-region). |
| **B** | **Block-wise TopK activation consumption** | Treat the block as the feature unit. Given `K` pre-discovered blocks `{U_1..U_K}` (each `n×g`), project activation `x` onto each, compute block energy `‖U_k^T x‖`, select top-k blocks. The *set of active blocks* is the sparse concept code. | `BlockTopKRouter` (VortexFlow/dash_attn, Plan 196) — **same block-TopK mechanism, different domain** (attention token selection, not activation featurization). `IndicatorProbeBank` (Plan 320) — multiple direction vectors with similarity structure, but each member is 1D + dot-product + sigmoid, OR-fused (not block-reduced). |
| **C** | **Block-dimensionality diagnostic** | Stable rank of a block = effective number of dimensions it uses. Already shipped as `effective_rank` / `stable_rank_update_into` (Roy-Vetterli standard `‖O‖_F² / ‖O‖_op²`). | `data_probe/geometry.rs::effective_rank`, `data_probe/sink_classify.rs::stable_rank_update_into` (Plan 287). **Identical metric, already shipped.** `participation_ratio` (Plan 301) — continuous analog. |
| **D** | **Block-sparsity prior on NPC latent state** | Hypothesis: at any tick, an NPC's HLA state is well-explained by a **small number of active concept-blocks** (e.g., "in combat" block + "afraid" block), not a dense 8-dim vector. This is a structural reframe of HLA, not a new computation. | None shipped as a *prior*. `PersonalityWeightedComposition` (Plan 297) blends N layers with sigmoid weights — closest to "which blocks are active", but over designer-defined layers, not discovered concept subspaces. |

### 2.2 Latent-space reframing (mandatory per workflow §1 step 3)

The paper operates on vision-model activations (DINOv3, SDXL, InceptionV1). Re-cast each mechanism on the codebase's latent-state kernels:

**(a) HLA per-NPC latent state (8-dim, `riir-ai/crates/riir-engine/src/hla/`)**

HLA's 8-dim state is currently treated as **5 scalar affective axes** (valence/arousal/desperation/calm/fear) + 3 reserved. BSF's reframing: each "emotion" is not a 1D scalar readout but a **concept subspace** within the 8-dim state. "Fear" might be a 2-3 dim region (fear-of-predator, fear-of-starvation, social-fear) that the scalar projection collapses. A subspace steering field for "make this NPC afraid" would carry a 2-3 dim block and walk *within* the fear region (predator-fear vs starvation-fear produce different behaviors). This generalizes both `EmotionDirections::project` (read-side, 1D) and `LatentSteeringVector` (write-side, 1D).

**(b) `latent_functor/` operations (`riir-ai/crates/riir-engine/src/latent_functor/`)**

Each functor application currently projects onto scalar coherence. BSF reframing: the functor's *active subspace* (the block of directions it's currently sensitive to) is the meaningful unit. `reestimation.rs`'s coherence threshold could become a **block-stable-rank threshold** — "the functor has converged when its active block's stable rank drops below τ" (it's operating in a low-dim region), complementing the existing scalar coherence gate.

**(c) `cgsp_runtime/` curiosity (`riir-ai/crates/riir-engine/src/cgsp_runtime/`)**

Curiosity = prediction error. BSF reframing: a query is "novel" iff it activates a **block that was previously off** (block-sparsity prior) — not just high scalar prediction error. This is a *structural* novelty signal: "this is a concept the NPC hasn't engaged with recently" (new block fires) vs "this is more of the same concept" (same block, higher energy).

**(d) LatCal fixed-point commitment (`riir-chain/src/encoding/latcal*.rs`)**

A k-dim orthonormal block `{u_1..u_k}` commits as `k·d` raw f32 scalars (the block matrix) + `k` strength scalars `{α_j}`. At `k=2, d=8` that's 18 f32 = 72 bytes — trivially LatCal-committable. The per-axis strengths `{α_j}` are the synced raw scalars (deterministic, bit-identical across quorum); the block basis `{u_j}` is also raw (orthonormal, deterministic). **No latent embedding crosses the sync boundary** — the block is a fixed-size raw matrix, not a variable-length embedding. This is cleaner than syncing a full HLA embedding.

**(e) `NeuronShard` `style_weights[64]` / freeze envelope / consolidation (`riir-neuron-db/src/`)**

`style_weights[64]` is already a 64-dim basis. BSF reframing: it should be interpretable as a **set of g-dim blocks** (e.g., 4 blocks of 16, or 8 blocks of 8), where each block is one "play-style concept" (aggressive, defensive, economic, social, ...). The freeze envelope commits the block partition + per-block stable rank. Consolidation (Raven/δ-Mem) selects which blocks to keep based on stable rank (low-stable-rank blocks are "saturated concepts"; high-stable-rank are "still exploring"). AnyRAG escalation: a query that activates a high-stable-rank block (under-explored concept) escalates.

**(f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-core/src/dec/`)**

DEC `hodge_decompose` splits a flow field into exact ⊕ harmonic ⊕ coexact channels — a **3-block decomposition** of the flow. BSF reframing: each DEC channel IS a concept block, and the stable rank of each channel's flow tells you whether that concept is 1D (exact = gradient = rank 1) or multidim (harmonic/coexact can be higher rank). The DEC substrate already does block-structured decomposition; BSF adds the "treat each block as a steerable concept region" interpretation. **Curse-of-dimensionality caveat (AGENTS.md):** boundary-vs-volume wins only for d ≤ 3 — DEC blocks (2D maps, 3D belief regions) are in the winning regime; HLA/shard blocks (d=8, d=64) are NOT. Do not apply boundary-flux BSF reasoning to high-dim shards.

### 2.3 Fusion

The closest cousins across all five repos, and what fusing each with BSF's block-as-unit idea produces:

1. **× Latent Field Steering (Plan 309, R290) → Subspace Steering Field (PRIMARY FUSION, katgpt-rs).** Generalize `LatentSteeringVector { direction: Vec<f32>, alpha: f32 }` to `SubspaceSteeringField { block: [[f32; D]; K], alphas: [f32; K] }`. At `K=1` it's bit-identical to Plan 309 (the 1D case). At `K≥2` it enables **manifold walking** — sweep `alphas` over a grid to generate concept variations (the "pretzel manifold" pattern). Zero-alloc, SIMD, BLAKE3-committed (commit the flattened block + alphas). This is the open primitive for Plan 412. **Novel capability**: steer within a concept region, not just along one direction.

2. **× subspace_phase_gate (Plan 301, R279) → Block Discovery Pipeline.** Plan 301 discovers the subspace *basis* `{u_1..u_k}` via Jacobian SVD; Plan 412 *consumes* that basis as a steerable block. Together: runtime-discovers the concept subspaces (301) → freezes them into a BLAKE3-committed block → steers/walks within them (412). This closes the loop from "discover axes" to "use axes for control". The Jacobian SVD output is the block; the subspace steering field is the consumer.

   **CONSUMER LIVE (2026-07-09):** `riir-neuron-db` ships `NeuronShard::steerable_axes::<K>()` (`src/phase_gate.rs`, default-on) — the first downstream consumer of this fusion. It wraps `semantic_axes` (Layer 2 SVD audit) as the block-discovery backend and builds a `SubspaceSteeringField<8, K>` from the top-K right singular vectors (singular values → sigmoid-bounded alphas). Issue 001 GOAT gate G1-G5 all PASS; promoted to default-on. See `riir-neuron-db/.benchmarks/001_subspace_steering_consumer_goat.md`.

3. **× Indicator Probe Bank (Plan 320, R301) → Block-Structured Probe Bank.** Currently the bank is N 1D direction vectors with a similarity matrix that *reveals* block structure post-hoc. BSF reframing: make the block structure **primary** — the bank holds K blocks of g indicators each, and the "active block" (block-wise TopK) is the detection signal, not per-indicator dot-product. This unifies detection (320) with steering (412) under one block-structured substrate.

4. **× PersonalityWeightedComposition (Plan 297, R276) → Block-Active Personality.** Currently blends N designer-defined layers with sigmoid weights. BSF reframing: the layers become *discovered concept blocks*, and the personality weights `{w_i}` become **block activation gates** (which concept-blocks is this NPC embodying?). "Aggressive personality" = combat-block + dominance-block active; "merchant personality" = economic-block + social-block active. The drift rule (R276 §1.2) already updates weights from reward surprise — it would now update *block activation*, not layer weight.

5. **× Clifford Geometric Product (Plan 319, R299) → Block-Internal Structure Detection.** The Clifford wedge `u∧v` detects rotational structure between two vectors. Within a BSF block, the pairwise wedges `{u_i ∧ u_j : i<j}` characterize the block's **internal geometry** (is it a flat disk? a curved manifold? a rotation orbit?). This is the *diagnostic* for "is this block a meaningful concept manifold or just a random subspace?". Complementary: BSF block = the unit; Clifford wedge = the intra-block structure sensor.

6. **× HLA kernel (`evolve_hla`, `riir-ai/crates/riir-engine/src/hla/`) → Block-Sparse HLA.** Reframe HLA's 8-dim state as a union of concept subspaces (e.g., 2 blocks of 4: {valence, arousal, desperation, calm} ⊕ {fear, curiosity, ..., ...}). The evolution kernel updates block energies; the *active block set* (top-k by energy) is the NPC's current "emotional posture". This is the private Super-GOAT angle (riir-ai) — see §3.

7. **× KarcShard delay basis (`riir-neuron-db/src/karc_shard.rs`, K=4, M=8) → Delay-Block Duality.** KarcShard's delay embedding IS a block structure (K delays × M basis functions). BSF reframing: each delay-basis column is one "temporal concept block"; the forecaster's `Wout` matrix projects block activations to future state. The block-stable-rank tells you how many distinct temporal modes the NPC's recent trajectory spans.

8. **× DEC `hodge_decompose` (Plan 251) → DEC Channel as Concept Block.** The 3 DEC channels (exact/harmonic/coexact) are a block decomposition of the flow field. Each channel's stable rank = that concept's dimensionality. Enables "steer the harmonic flow" as a block-level control on terrain cochains.

**Strongest fusion candidates**: #1 (Subspace Steering Field — the open primitive) and #6 (Block-Sparse HLA — the private riir-ai angle). #1 is the katgpt-rs GOAT; #6 is a Super-GOAT *candidate* that needs Q1–Q4 validation before claiming (see §3).

---

## 3. Verdict

### Tier: **GOAT** (open primitive) — with a **Super-GOAT fusion candidate tracked in Issue 049 (CLOSED 2026-07-09, NOT Super-GOAT)**

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **PARTIAL.** The 1D case fully ships (`LatentSteeringVector` Plan 309, `EmotionDirections` Plan 162, `PersonalityWeightedComposition` Plan 297, `IndicatorProbeBank` Plan 320). The **subspace (k-dim) case** does not ship as a steering primitive — `Phase-Modulated Coupling` (P322) is the closest 2D case but uses cos/sin weights, not per-axis `α_j`, and is single-pair not walk-within-region. The block-featurizer *consumption* pattern (block-wise TopK on activations) does not ship — `BlockTopKRouter` is attention-token selection, not activation featurization. The **stable-rank diagnostic ships** (Plan 287, identical Roy-Vetterli metric). So: the diagnostic is covered, the 1D steering is covered, the k-dim steering + block-featurizer-consumption is genuinely missing. | Vocabulary translation: "Block-Sparse Featurizer" → "subspace steering field" / "block-wise TopK activation"; "manifold" → "subspace" / "concept block"; "stable rank" → "effective_rank" (HIT — Plan 287); "SAE" → rejected per R143/R053. Three-layer check (notes + code + vocab) done. |
| Q2 New class of behavior? | **PARTIAL.** "Walk within a concept region to generate variations" is a new capability vs 1D knob steering (P309). But it's a **generalization** of existing steering to higher dim, not a fundamentally new mechanism class. The block-featurizer consumption pattern is a new *composition* of existing pieces (block TopK + activation projection), not a new primitive kind. | |
| Q3 Product selling point? | **PARTIAL.** "NPC personality traits are multidimensional regions you can walk through, not 1D knobs — a 'fearful' NPC can be predator-fearful or starvation-fearful, and the designer steers within the fear region." Sellable, but the game-AI payoff is speculative until the block structure is validated on real HLA data. | |
| Q4 Force multiplier? | **YES.** Connects LatentSteering (P309) × subspace_phase_gate (P301) × IndicatorProbeBank (P320) × PersonalityWeightedComposition (P297) × Clifford wedge (P319) × HLA kernel × KarcShard × DEC hodge_decompose (P251). ≥8 cousins. | |

**Not all-4-YES → not Super-GOAT.** The open primitive (subspace steering field) is a clean GOAT: provable generalization of 1D to k-dim, subsumes Plan 309 at `k=1`, enables manifold walking at `k≥2`. The GOAT gate (Plan 412) proves: (G1) `k=1` is bit-identical to Plan 309; (G2) `k≥2` preserves behavior rank while enabling within-region walks; (G3) zero-alloc; (G4) latency within budget.

### Super-GOAT fusion candidate (NOT claimed — tracked in Issue 049, CLOSED 2026-07-09)

> **Update 2026-07-09:** Issue 049 was **CLOSED with a negative Q3 result** — the Block-Sparse HLA claim was validated as NOT Super-GOAT (three independent measured failures). The issue file was deleted per the reduce-noise rule. See `riir-ai/.proposals/010_block_sparse_hla_q3_real_game_validation.md` for the full validation record. The text below is preserved as the original candidate framing.

The **Block-Sparse HLA** fusion (#6 above) — reframing HLA's 8-dim state as a union of concept subspaces with block-sparsity prior — is a Super-GOAT *candidate* if it produces a new capability class ("NPCs whose emotional posture is a sparse set of active concept-blocks, each multidimensional, steerable within-region"). But the novelty gate (Q1–Q4) is **not yet confident enough to commit**:
- Q1 is uncertain: does the existing HLA 8-dim treatment already implicitly capture this? The 5-scalar projection is 1D-per-axis, but the 3 reserved dims might already be a latent block structure.
- Q2 is uncertain: is "block-sparse emotional posture" a new capability or just a re-interpretation?
- Q3 needs real game data to validate the selling point.

Per the workflow's "no candidate escape hatch" rule, I do **not** write "Super-GOAT candidate" in the verdict. Instead, **Issue 049** tracks the Q1–Q4 validation follow-up. If the validation passes, the riir-ai guide (`riir-ai/.research/NNN_block_sparse_hla_*.md`) gets created at that point with the full Super-GOAT mandatory outputs.

### One-line reasoning

BSF's training method (encoder/decoder fit) is riir-train territory; its modelless transferable primitive is the **k-dim subspace steering field** that generalizes our 1D `LatentSteeringVector` (Plan 309) to enable manifold-walk steering, plus the **block-featurizer consumption pattern** (block-wise TopK on activations, reusing the shipped `effective_rank` diagnostic). GOAT: strict generalization, subsumes 1D at `k=1`, unlocks within-region steering at `k≥2`.

### Routing

- **katgpt-rs/.plans/412_subspace_steering_field_primitive.md** — open primitive. `SubspaceSteeringField { block: [[f32; D]; K], alphas: [f32; K] }` + `apply_subspace_steering` (SIMD, zero-alloc) + `walk_manifold` (grid sweep over `alphas`) + BLAKE3 commitment. Feature flag `subspace_steering`. GOAT gate G1–G5.
- **katgpt-rs/.issues/049_block_sparse_hla_supergoat_validation.md** — tracks the Super-GOAT fusion candidate (Block-Sparse HLA). No guide created until Q1–Q4 pass.
- **riir-ai** — deferred. The private Super-GOAT guide + HLA wiring plan opens only if Issue 049 validation passes.
- **riir-train** — the BSF training itself (encoder/decoder fit, three variants) is a training-method note, out of scope for this workflow.

### MOAT gate (per domain, §1.6)

| Domain | In scope? | MOAT contribution |
|--------|-----------|-------------------|
| `katgpt-rs` (public engine) | **YES** | Subspace steering field is a generic math primitive (k-dim orthonormal block + per-axis strengths + manifold walk). No game/chain/shard semantics. Ships behind `subspace_steering` feature flag; GOAT gate decides promote-to-default. **Per-stack tracking**: this occupies the "steering" slot alongside Plan 309 (1D) and Plan 322 (2D phase-rotation) and Plan 405 (Slerp). Demote Plan 309 to opt-in only if `k=1` parity holds AND `k≥2` shows measurable gain — otherwise they coexist (1D for simple steering, k-dim for manifold walking). |
| `riir-ai` (private runtime) | Candidate (Issue 049) | Block-Sparse HLA is a pillar-level reframe IF validated. Not claimed yet. |
| `riir-chain` | No | Block commitment is trivial raw-matrix LatCal (§2.2d) — no novel chain mechanism. |
| `riir-neuron-db` | Indirect | `style_weights[64]` block interpretation (§2.2e) is a future neuron-db angle, not this primitive's scope. |
| `riir-train` | Redirect | BSF training → riir-train. |

---

## 4. What stays open vs private

| Artifact | Repo | Visibility | Why |
|----------|------|-----------|-----|
| `SubspaceSteeringField` primitive | katgpt-rs | **Open (MIT)** | Generic k-dim math. No game IP. Reusable beyond our stack. |
| Plan 412 | katgpt-rs | **Open** | Generic primitive plan. |
| Research 393 (this note) | katgpt-rs | **Open** | Distillation of a public paper. |
| Block-Sparse HLA Super-GOAT guide | riir-ai | **Private — NOT YET CREATED** | Only opens if Issue 049 Q1–Q4 validation passes. |
| BSF training (3 variants) | riir-train | **Private — out of scope** | Training method. Note "→ riir-train" and stop. |

---

## 5. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ The subspace steering field consumes **pre-discovered** blocks (from Plan 301 Jacobian SVD, SpectralQuant offline eigenbasis, or hand-constructed orthogonal sets). No gradient descent at inference. The BSF *training* is riir-train; the *consumption* is modelless. |
| Latent-to-latent preferred | ✅ Operates entirely in latent space (HLA 8-dim, functor state). Never crosses to tokens. |
| Use sigmoid not softmax | ✅ Per-axis strengths `α_j` are sigmoid-bounded. The block-wise TopK selection uses L2 norm (not softmax) — consistent with `BlockTopKRouter`'s existing pattern. |
| Freeze/thaw over fine-tuning | ✅ Blocks are BLAKE3-committed frozen artifacts; atomic Arc swap for hot-swap. Steering is an additive overlay, not a mutation of frozen state (same pattern as Plan 309). |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; HLA wiring → riir-ai (deferred); BSF training → riir-train (out of scope). |
| Raw scalars at sync boundary | ✅ The block matrix `{u_1..u_k}` + strengths `{α_j}` are fixed-size raw f32 arrays — deterministic, bit-identical across quorum. No variable-length embedding crosses sync. (§2.2d) |
| Zero-alloc hot path | ✅ `apply_subspace_steering` is SIMD SAXPY over `K·D` elements; `walk_manifold` reuses a pre-allocated output grid. |
| CPU/GPU/ANE auto-route | ✅ At `K·D ≤ 64` (HLA scale), SIMD CPU. Larger blocks (shard-scale `D=64`) may benefit from GPU matmul — threshold-adaptive dispatch. |

---

## 6. Modelless-first check (§3.5 protocol)

Before any riir-train deferral, the three modelless unblock paths for "where do the blocks come from?":

1. **Freeze/thaw snapshot correction** — can a frozen snapshot carry pre-discovered blocks? **YES.** The block basis `{u_1..u_k}` is a fixed-size raw matrix, freezable as a `MerkleFrozenEnvelope` payload. Thaw loads the block; no training needed at inference. **Path 1 PASSES.**

2. **Raw/lora reader-writer hot-swap** — can a deterministically-constructed LoRA produce the block? **PARTIAL.** A reader-LoRA that projects activations onto a pre-specified orthogonal set (e.g., the 5 HLA axes + 3 reserved, or a DCT/DFT basis) is deterministic. But the **reconstruction-optimal** block partition (which the BSF training discovers) cannot be constructed in closed form — it depends on the activation distribution. **Path 2 PASSES for hand-constructed blocks, FAILS for reconstruction-optimal blocks.**

3. **Latent-space correction** — can the block be derived via dot-product projection? **YES, via Plan 301.** `jacobian_svd_at` discovers the leading singular vectors of any map's Jacobian at a point — these ARE candidate block bases. The block is derived from runtime data (modelless Jacobian estimation), not trained. **Path 3 PASSES.**

**Verdict: MODELLESS-VALIDABLE.** Blocks can come from (a) hand-construction, (b) Plan 301 Jacobian SVD, or (c) SpectralQuant offline eigenbasis. The reconstruction-optimal BSF partition requires training (→ riir-train), but the *consumption* of any pre-discovered block is modelless. No riir-train deferral needed for the steering primitive itself.

---

## 7. Open questions / risks

1. **Does `k≥2` steering preserve behavior rank?** Same headline risk as Plan 309. If walking within a concept region changes which action the NPC selects (beyond the intended affect shift), the primitive is dangerous. **Mitigation:** G2 gate measures cosine similarity of action rankings pre/post steering across a grid of `alphas`; gate requires ≥0.95 over the walked region.

2. **Is the block basis orthonormal?** BSF's Grassmannian variant enforces this; vanilla BSF does not. For steering, orthonormality is desirable (clean per-axis control) but not required (non-orthogonal blocks still steer, just with cross-axis coupling). **Mitigation:** construct with `newton_schulz_orthogonalize` (Plan 152, shipped) at freeze time; commit the orthonormalized block.

3. **Stable rank ≠ true concept dimensionality on HLA.** BSF's "2–4 dim" finding is for *vision* models (2D projections of 3D scenes). HLA's 8-dim affective space may have different intrinsic structure — possibly 1D per emotion (validating the current 1D treatment) or possibly multidim. **Mitigation:** the primitive is agnostic to `k`; the empirical question of "what's the right `k` for HLA emotions?" is deferred to Issue 049 / riir-ai validation.

4. **Block-sparsity prior may not hold for HLA.** BSF assumes "a few blocks fire per input". HLA's 8-dim state might be dense (all 5 emotions always somewhat active). If so, block-wise TopK is the wrong consumption pattern for HLA (though subspace steering still works). **Mitigation:** the primitive ships both block-TopK consumption AND subspace steering independently; consumers pick the pattern that fits their domain.

5. **Manifold walking may produce semantically meaningless intermediate points.** Walking a 2D grid over a "fear" block produces points that are mathematically in-span but may not correspond to coherent emotional states (linear interpolation in latent space ≠ semantic interpolation). BSF's vision examples (pretzel manifold) work because the *generative model* decodes the latent point to a meaningful image. For HLA, there's no decoder — the latent point IS the state. **Mitigation:** G2 (behavior rank preservation) catches incoherent intermediate states; the walked region may be narrower than the full block span.

---

## 8. References

- **Source paper**: [arXiv:2606.25234](https://arxiv.org/abs/2606.25234) — Goodfire, "Block-Sparse Featurizers".
- **Blog**: [goodfire.ai/research/bsf-vision](https://www.goodfire.ai/research/bsf-vision).
- **Code**: [github.com/goodfire-ai/block-sparse-featurizer](https://github.com/goodfire-ai/block-sparse-featurizer).
- **Prior Goodfire work**: [Can SAEs Capture Neural Geometry?](https://www.goodfire.ai/research/can-saes-capture-neural-geometry) (the SAE-fragments-manifolds finding), [The World Inside Neural Networks](https://www.goodfire.ai/research/the-world-inside-neural-networks).
- **Closest internal cousin (steering, 1D)**: `katgpt-rs/.research/290_latent_field_steering_open_primitive.md` + `katgpt-rs/crates/katgpt-core/src/latent_steering.rs` (Plan 309).
- **Closest internal cousin (basis discovery)**: `katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md` + `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs` (Plan 301).
- **Closest internal cousin (2D rotation)**: `katgpt-rs/.benchmarks/322_phase_rotation_goat.md` (Phase-Modulated Coupling).
- **Closest internal cousin (multi-direction bank)**: `katgpt-rs/crates/katgpt-core/src/pruners/indicator_probe_bank.rs` (Plan 320).
- **Stable rank (shipped)**: `katgpt-rs/crates/katgpt-core/src/data_probe/sink_classify.rs::stable_rank_update_into` + `data_probe/geometry.rs::effective_rank` (Plan 287).
- **SAE rejection precedent**: `katgpt-rs/.research/143_Latent_Terms_SAE_BM25_Retrieval.md` (NO GAIN) + `katgpt-rs/.research/053_CNA_Contrastive_Neuron_Attribution.md` ("don't implement SAE").
- **Canonical plan example**: `katgpt-rs/.plans/309_latent_field_steering_primitive.md` (the 1D sibling this generalizes).
- **Super-GOAT fusion tracker**: `katgpt-rs/.issues/049_block_sparse_hla_supergoat_validation.md`.

---

## TL;DR

Goodfire's Block-Sparse Featurizers decompose activations into multidimensional concept subspaces (blocks) via block-sparsity, finding that vision concepts are 2–4 dimensional manifolds (stable rank ≤ 4). The BSF *training* is riir-train territory; the modelless transferable primitive is the **subspace steering field** — a k-dim orthonormal block `{u_1..u_k}` + per-axis strengths `{α_j}` that generalizes our 1D `LatentSteeringVector` (Plan 309) to enable manifold walking (steer within a concept region, not along one line). At `k=1` it's bit-identical to Plan 309; at `k≥2` it unlocks within-region steering. The stable-rank diagnostic already ships (Plan 287, identical Roy-Vetterli metric). **Verdict: GOAT** — open primitive in katgpt-rs (Plan 412), strict generalization of 1D steering, subsumes Plan 309 at `k=1`. A Super-GOAT fusion candidate (Block-Sparse HLA — reframe HLA's 8-dim state as a union of concept subspaces) was tracked in Issue 049 **but CLOSED 2026-07-09 with negative Q3 result (NOT Super-GOAT)** — see `riir-ai/.proposals/010_block_sparse_hla_q3_real_game_validation.md`.
