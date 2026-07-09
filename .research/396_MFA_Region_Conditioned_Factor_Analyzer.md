# Research 396: MFA — Region-Conditioned Factor Analyzers for Local-Geometry Steering

> **Source:** "From Directions to Regions: Decomposing Activations in Language Models via Local Geometry" — Shafran, Ronen, Fahn, Ravfogel, Geiger, Geva. [arXiv:2602.02464](https://arxiv.org/abs/2602.02464). Feb 2026. Code + trained MFAs: https://github.com/ordavid-s/decomposing-activations-local-geometry
> **Date:** 2026-07-09
> **Status:** Active
> **Related Research:** 393 (BSF — closest cousin, same "concepts are manifolds not lines" thesis), 389 (CHaRS — cluster-aware steering), 302 (FAME — per-entity MoE), 276 (PersonalityWeightedComposition), 290 (Latent Field Steering)
> **Related Plans:** 412 (SubspaceSteeringField — the within-region primitive this generalizes), 409 (CharsAnchorBank — the region+centroid cousin), 321 (CommittedFieldBlend — the per-entity MoE cousin)
> **Classification:** Public

---

## TL;DR

The paper trains large-scale **Mixtures of Factor Analyzers (MFA)** on residual-stream activations of Llama-3.1-8B / Gemma-2-2B and shows that decomposing activations into **(region centroid μ_k + within-region low-rank subspace W_k)** — rather than a global dictionary of 1D directions (SAEs) — yields dramatically higher interpretability (IF 0.96 vs 0.29), better causal localization, and stronger steering. This is the **third paper in our corpus** (after R393 BSF and R389 CHaRS) to argue that concepts are multidimensional regions, not lines — but it is the first to give a **principled region-conditioned structure**: each region has BOTH an absolute position (centroid) AND its own local geometry (factor-analyzer subspace).

**Distilled for katgpt-rs (modelless, inference-time):**
A frozen MFA artifact `{μ_k, W_k, Ψ, π}` (trained offline → freeze/thaw) is consumed modellessly via four closed-form operations: (1) **per-region sigmoid membership gates** `g_k(x) = sigmoid(a_k(x))` where `a_k` is the Gaussian log-likelihood (reformulated from the paper's softmax responsibilities to sigmoid per the AGENTS.md mandate — this is *more* expressive: an NPC can be partially in multiple regions simultaneously, not winner-take-all); (2) **local coordinates** `ẑ_k = Z_k(x − μ_k)` (eq. 9-10, closed-form matrix-vector); (3) **centroid steering** `f_μ(x) = (1−α)x + αμ_k` (interpolation toward absolute position); (4) **local subspace steering** `f_w(x) = x + W_k·v` (additive offset within the region). The MFA training itself (GD on negative log-likelihood) is riir-train; everything shipped here is the modelless consumer.

**This is the unification of Plan 412 (within-region subspace) × Plan 409 (region centroids + routing) that neither alone provides.** Plan 412 ships a single subspace block with no region structure; Plan 409 ships region centroids + translation vectors but no per-region subspaces; MFA gives both — K regions, each with its own centroid AND local subspace.

---

## 1. Paper Core Findings

### 1.1 The thesis

Activation decomposition methods (SAEs, dictionary learning) are tightly coupled to the geometric assumption that concepts are **single global directions** (linear separability). The paper shows this overlooks concepts with nonlinear or multidimensional structure, and proposes modeling activation space as a **collection of Gaussian regions**, each equipped with its own local low-rank covariance structure. The unit of analysis shifts from "isolated directions" to "local regions with their own low-rank geometry."

### 1.2 The mechanism — Mixture of Factor Analyzers (MFA)

MFA (Ghahramani & Hinton 1996) models activation space as K components. Conditioned on component `ω = k`:

```
x = μ_k + W_k·z_k + ε        (generative)
C_k = W_k·W_k^T + Ψ          (component covariance)
```

where `μ_k` is the region centroid, `W_k ∈ R^{d×R}` are the factor-analyzer loadings (R-dim local subspace), `z_k ~ N(0, I)` are latent factors, and `Ψ` is diagonal noise. Each activation is decomposed into two compositional geometric objects:

| Object | What it captures | Steering mode |
|--------|-----------------|---------------|
| **Region centroid `μ_k`** | Absolute position in activation space — "which broad concept neighborhood" | **Centroid interpolation** `f_μ(x) = (1−α)x + αμ_k` — move toward a region |
| **Local loadings `W_k`** | Within-region variation — "how does the concept vary inside this region" | **Local offset** `f_w(x) = x + W_k·v` — walk within the region |

### 1.3 Responsibilities and decomposition

An activation `x` is assigned to component `k` via posterior responsibilities:
```
R_k(x) = p(k|x) = π_k·N(x|μ_k, C_k) / Σ_i π_i·N(x|μ_i, C_i)
```
and the local coordinates within region `k` are the posterior-mean latent vector:
```
ẑ_k = Z_k·(x − μ_k),   Z_k := (I_R + W_k^T Ψ^{-1} W_k)^{-1} W_k^T Ψ^{-1}
```
The full decomposition is a single linear product `x ≈ A·b(x)` where `A = [μ_1 | W_1 | … | μ_K | W_K]` is the shared dictionary and `b(x)` stacks `[R_k(x); R_k(x)·ẑ_k(x)]` per region.

### 1.4 Empirical headline

- **Interpretability Fraction (IF):** MFA achieves **0.96 ± 0.2** vs SAEs **0.29 ± 0.2** — most high-contribution MFA features are interpretable; most SAE features are not.
- **Causal localization (RAVEL/MCQA, Table 2):** MFA beats PCA/SAEs by 3–16 points, beats the supervised DBM baseline on 5/8 tasks, competitive with SOTA DAS.
- **Causal steering (Figure 5):** MFA centroids steer better than SAE features and DiffMeans in the majority of settings — roughly **2× median score** on Gemma-2-2B.
- **Two-mode steering (Table 1):** centroid interpolation promotes a broad theme (e.g., "genres"); local offsets refine into subthemes (e.g., "fantasy" vs "spy thriller"). This two-level structure is the key empirical finding.
- **Ablation (§6.1):** RAVEL variables are captured primarily by centroids (centroids-only ≈ full performance); MCQA positional variables need the local loadings (centroids-only drops 80%→39%). Both modes matter for different concept types.

### 1.5 What is training-only (→ riir-train, do NOT distill here)

- The MFA **fitting** (GD/EM on negative log-likelihood over 100M activations) is gradient descent through the `{μ_k, W_k, Ψ, π}` parameters. → riir-train.
- The K-means initialization (4M-activation sample, K-means seed for centroids). → offline, but produces the frozen artifact.
- The reconstruction-error benchmarking (Table 4) against SAEs. → measurement, not mechanism.

**The modelless-unblock question (§3.5):** can the MFA structure be **consumed modellessly** given a frozen artifact? **YES** — see §2.1 and §3. The artifact `{μ_k, W_k, Ψ, π}` is a frozen, BLAKE3-committable Pod; all inference-time operations (responsibilities, local coords, steering) are closed-form linear algebra.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent |
|---|---|
| residual stream activation | HLA 8-dim state, `evolve_hla` output, `style_weights[64]` |
| region / Gaussian component | **latent region** — centroid-anchored subspace cluster in HLA/functor/shard space |
| factor analyzer / loadings `W_k` | **local subspace block** — k-dim orthonormal-ish basis within one region (cf. `SubspaceSteeringField` Plan 412, but region-conditioned) |
| centroid `μ_k` | **region anchor** — absolute position (cf. `CharsAnchorBank.src_centroids` Plan 409, `ArchetypeBlendShard`) |
| responsibilities `R_k(x)` (softmax posterior) | **per-region sigmoid membership gate** `g_k(x) = sigmoid(a_k(x))` — reformulated per AGENTS.md sigmoid mandate; MORE expressive (multi-region membership, not winner-take-all) |
| local coordinates `ẑ_k` | **within-region projection** (cf. `block_energy` Plan 412, but posterior-mean adjusted) |
| centroid steering `(1−α)x + αμ_k` | **region interpolation** (cf. `slerp_steering_into` Plan 405 toward centroid target, but linear blend not Slerp) |
| loading steering `x + W_k·v` | **subspace offset** (cf. `apply_subspace_steering` Plan 412, but region-conditioned) |
| SAE (global dictionary) | `IndicatorProbeBank` (Plan 320) — N 1D direction vectors, the incumbent this paper beats |
| "concepts are regions not directions" | **identical thesis to R393 (BSF) and R389 (CHaRS)** — this is the third paper making it |

**Standing vocabulary blocks added for future local-geometry papers:**
- "factor analyzer" / "loadings" / "within-region variation" → **local subspace block**, **region-conditioned basis**, `SubspaceSteeringField` (region-conditioned variant)
- "responsibilities" / "posterior assignment" → **sigmoid membership gate**, **per-region confidence** (NOT softmax routing)
- "centroid interpolation" / "move toward region" → **region interpolation**, anchor blend
- "local geometry" / "regions" → **latent region bank**, **centroid-anchored subspace cluster**

### 2.2 What we already ship (prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan | Diff vs MFA |
|---|---|---|---|
| k-dim subspace steering (within-region) | **`SubspaceSteeringField`** | Plan 412, `crates/katgpt-core/src/subspace_steering.rs` — `block: [[f32;D];K]`, `alphas`, `walk_manifold`. DEFAULT-ON. | Plan 412 is a **single** subspace block with K axes — **NO regions, NO centroids, NO responsibilities routing**. MFA gives K regions each with its OWN subspace. |
| Region centroids + input-adaptive routing | **`CharsAnchorBank`** | Plan 409, R389 — `src_centroids`, `ot_plan`, `v_ij`. Planned. | CHaRS has centroids + RBF-gated routing, but the steering vectors are **inter-region translations** `v_ij = b_j − a_i`, NOT **within-region subspaces** (factor analyzers). MFA's loadings `W_k` are intra-region. |
| Per-entity MoE blend of K archetype fields | **`CommittedFieldBlend`** | Plan 321, R302 — `pi: [f32;N]`, `apply_blended`. DEFAULT-ON. | Per-entity FIXED routing (π computed once), no regions. MFA routes **per-input** via responsibilities. |
| 1D latent steering | **`LatentSteeringVector`** | Plan 309, R290 — `direction: Vec<f32>`, `alpha`. DEFAULT-ON. | 1D; MFA generalizes to region-conditioned multi-dim. |
| Spherical Slerp toward target | **`spherical_steering`** | Plan 405, R382 — `slerp_steering_into`, `vmf_confidence_gate`. DEFAULT-ON. | Single target, norm-preserving; MFA centroid steering is linear blend toward one of K centroids. |
| Phase-modulated 2D coupling | **`Phase-Modulated Coupling`** | Plan 322 — cos/sin rotation in `(a,b)` plane. DEFAULT-ON. | 2D single-pair rotation; MFA is R-dim per region. |
| Subspace basis discovery (runtime SVD) | **`subspace_phase_gate`** | Plan 301, R279 — `semantic_axes`, Jacobian SVD. | Discovers ONE basis; MFA needs K region-conditioned bases (discovered offline, consumed here). |
| Frozen artifact, BLAKE3 commitment | **`NeuronShard` / `MerkleFrozenEnvelope`** | riir-neuron-db/src/shard.rs, freeze.rs | MFA artifact `{μ_k, W_k, Ψ, π}` is a new shard subtype candidate. |
| Contrastive direction construction | **`EmotionDirections::project`** | Plan 162 — read-only 1D projection | MFA regions are unsupervised (no contrastive pair needed); EmotionDirections are supervised 1D. |

**Critical distinction from Plan 412 (the closest cousin):** Plan 412's `SubspaceSteeringField` carries ONE block `{u_1..u_k}` that applies globally — every activation is steered by the same K axes. MFA carries K **region-conditioned** blocks `{W_1, ..., W_K}`, each active only when the activation is assigned to region k. At the degenerate limit `K=1` (single region, centroid = 0, loadings = identity), MFA's local-coordinate steering reduces to Plan 412's subspace steering — but MFA's region structure (multiple centroids + per-region routing) is what Plan 412 entirely lacks.

**Critical distinction from Plan 409 (CHaRS):** CHaRS models **inter-region** geometry (how to translate activations from source region `a_i` to target region `b_j` via `v_ij`). MFA models **intra-region** geometry (how activations vary *within* a single region via its local loadings `W_k`). The two compose: CHaRS tells you how to *move between* regions; MFA tells you how to *move within* a region. Both have centroids; only MFA has per-region subspaces.

### 2.3 Transferable primitive

A region-conditioned subspace field — K regions, each with a centroid + local subspace — consumed via four modelless operations:

```rust
/// A frozen MFA-like artifact: K regions, each with a centroid μ_k and a
/// local R-dim subspace (factor-analyzer loadings W_k). BLAKE3-committed.
///
/// Region-conditioned generalization of Plan 412's `SubspaceSteeringField`.
/// Plan 412 = single block, no regions (the K=1, centroid=0, loadings=I limit).
/// Plan 409 (CHaRS) = regions + centroids, but translations not subspaces.
/// This primitive = regions + centroids + per-region subspaces (the MFA structure).
///
/// The artifact is TRAINED OFFLINE (riir-train: GD on negative log-likelihood,
/// or K-means + per-region PCA as a deterministic constructor). Once frozen,
/// all consumption is closed-form linear algebra — no gradients at inference.
pub struct RegionSubspaceField<const D: usize, const K: usize, const R: usize> {
    /// Region centroids `μ_k ∈ R^D`. K rows.
    pub centroids: [[f32; D]; K],
    /// Per-region factor-analyzer loadings `W_k ∈ R^{D×R}`. K blocks of R rows.
    /// Each `loadings[k]` is a `[[f32; D]; R]` — R local axes for region k.
    pub loadings: [[[[f32; D]; R]; K]; 1],  // or flattened [K][R][D]
    /// Per-region mixture log-weights `log π_k` (pre-computed at freeze).
    pub log_pi: [f32; K],
    /// Diagonal noise precision (inverse variance) per dimension.
    pub psi_inv: [f32; D],
    /// Pre-computed posterior-mean projector `Z_k ∈ R^{R×D}` per region.
    /// `Z_k = (I_R + W_k^T Ψ^{-1} W_k)^{-1} W_k^T Ψ^{-1}` — closed-form at freeze.
    pub projectors: [[[[f32; D]; R]; K]; 1],  // or flattened
    /// `BLAKE3(centroids || loadings || log_pi || psi_inv)` — content commitment.
    pub commitment: [u8; 32],
}
```

**Four modelless operations (all closed-form, zero-alloc after field construction):**

1. **Per-region sigmoid membership gates** (reformulated from softmax responsibilities):
   ```
   a_k(x) = log_pi[k] − 0.5 · Mahalanobis(x, μ_k, C_k) − 0.5 · log|C_k|
   g_k(x) = sigmoid(a_k(x) − τ)              // ∈ (0,1), independent per region
   ```
   Per-region independent sigmoid gates (NOT winner-take-all softmax). An NPC can be partially in the "combat" region AND the "fear" region simultaneously — more expressive than the paper's categorical responsibilities, consistent with CommittedFieldBlend's sigmoid gates (Plan 321).

2. **Local coordinates within region k** (posterior mean, eq. 9-10):
   ```
   ẑ_k = Z_k · (x − μ_k)    // closed-form matrix-vector, R-dim output
   ```

3. **Centroid steering** (eq. 14, move toward region):
   ```
   f_μ(x, k, α) = (1 − α)·x + α·μ_k
   ```
   Linear interpolation toward centroid k. At α=0 identity, α=1 full region replacement.

4. **Local subspace steering** (eq. 15, walk within region):
   ```
   f_w(x, k, v) = x + W_k · v    // v ∈ R^R, additive offset within region k
   ```
   Identical math to Plan 412's `apply_subspace_steering`, but **region-conditioned** — `W_k` is selected by the active region.

### 2.4 Fusion

The closest cousins across all five repos, and what fusing each with MFA's region-conditioned structure produces:

1. **× Plan 412 (`SubspaceSteeringField`) → Region-Conditioned Subspace Steering (PRIMARY, katgpt-rs).** Plan 412 is the within-region primitive (single block). MFA generalizes it: instead of one global block, K region-conditioned blocks, each active when the activation is in that region. At `K=1, μ_1=0, W_1=I` it reduces to Plan 412. **This is the open primitive for Plan 416.** Novel capability: steer within the *correct local geometry* for the activation's current region, not a global subspace.

2. **× Plan 409 (`CharsAnchorBank`, CHaRS) → Inter+Intra Region Steering.** CHaRS steers *between* regions (translation vectors `v_ij`); MFA steers *within* regions (loadings `W_k`). Fusing: a steering field that can both move an NPC to a different emotional region (CHaRS translation) AND walk within the current region (MFA local offset). The two steering modes compose: first translate to the target region, then refine within it.

3. **× Plan 321 (`CommittedFieldBlend`) → Per-Entity Region-Conditioned Personality.** CommittedFieldBlend blends K archetype operator fields with per-entity FIXED weights π. MFA adds region structure: each archetype field could be a region-conditioned factor analyzer, and the NPC's personality determines *which regions it tends to occupy* (region priors). The committed π becomes region-membership priors, not just field weights.

4. **× HLA kernel (`evolve_hla`) → Region-Structured HLA (Super-GOAT candidate, extends Issue 049).** R393's "Block-Sparse HLA" candidate (Issue 049) speculated about interpreting HLA's 8-dim state as a union of concept subspaces. MFA gives the concrete recipe: fit K factor analyzers on HLA trajectories, decompose each NPC's HLA state into (region assignment + local coordinates). Each NPC's emotional state lives in one of K regions (e.g., "calm," "combat," "social," "fleeing"), each with its own local subspace. Steering "make this NPC afraid" = centroid interpolation toward the fear region; steering "afraid in a specific way" = local subspace offset within the fear region. **This makes Issue 049's speculative candidate concrete — but the Q1–Q4 validation still needs real game data, so it remains a tracked candidate, not a committed Super-GOAT.**

5. **× `latent_functor/` (`riir-engine`) → Region-Conditioned Functors.** Each functor application projects onto scalar coherence. MFA reframing: the functor's behavior depends on which region the source/target states occupy. `reestimation.rs`'s coherence threshold could become a **region-membership-aware** threshold — re-estimate when the state moves to a region the current functor wasn't fit for.

6. **× `NeuronShard` / `MerkleFrozenEnvelope` (`riir-neuron-db`) → MFA Shard.** The frozen MFA artifact `{μ_k, W_k, Ψ, π}` is a natural shard subtype — fixed-size Pod (K·D + K·R·D + K + D + 32 bytes), BLAKE3-committed, freezable/thawable. `style_weights[64]` could be reinterpreted as region loadings. Consolidation (Raven/δ-Mem) selects which regions to keep based on membership entropy (high-entropy regions = "still exploring this concept").

7. **× LatCal (`riir-chain`) → Region-Conditioned Commitment.** The MFA parameters cross the sync boundary as raw scalars: K·D centroid floats + K·R·D loading floats + K mixture weights. At K=8, D=8, R=2 that's 8·8 + 8·2·8 + 8 = 200 f32 = 800 bytes — trivially LatCal-committable. The per-region sigmoid membership gates are the synced raw scalars (the NPC's current region memberships), not the full field definitions.

8. **× DEC `hodge_decompose` (Plan 251) → Region = Cell, Subspace = Local Cochain.** A DEC cell complex over the activation manifold: each cell IS a region, and the local cochain structure within each cell IS the factor-analyzer subspace. `hodge_decompose` splits the flow into exact/harmonic/coexact — a 3-region decomposition. MFA generalizes this to K regions with learned boundaries. **Curse-of-dimensionality caveat:** boundary-vs-volume wins only for d ≤ 3; HLA (d=8) and shards (d=64) do NOT benefit from boundary-only computation. The DEC mapping is conceptual, not for perf.

**Strongest fusion candidates:** #1 (Region-Conditioned Subspace Steering — the open primitive, Plan 416) and #4 (Region-Structured HLA — the Super-GOAT candidate extending Issue 049). #1 is the katgpt-rs GOAT; #4 is a tracked candidate needing real-game validation.

### 2.5 Latent-space reframing (mandatory per fusion protocol §1)

**(a) HLA per-NPC latent state (8-dim, `katgpt-core/src/sense/`, `riir-engine/src/hla/`):**
HLA's 8-dim state is currently read via 5 scalar projections (valence/arousal/desperation/calm/fear). MFA reframing: the 8-dim space is partitioned into K regions (e.g., "combat-fear," "social-calm," "exploration-curiosity"), each with its own local subspace. An NPC's HLA state at any tick has a **region-membership profile** `{g_k(hla_t)}` (which regions it's currently in) + **local coordinates** `{ẑ_k(hla_t)}` (where it sits within each region). This generalizes both the 5-scalar projection (1D-per-axis read) and Plan 412's single-block subspace (global block). **Sigmoid gates (not softmax) mean the NPC can be in multiple regions at once** — e.g., "70% combat, 30% fear" — which is richer than the paper's categorical assignment.

**(b) `latent_functor/` operations (`riir-engine/src/latent_functor/`):**
The functor currently projects onto scalar coherence per (source, target) relation. MFA reframing: the functor's active subspace depends on which region the source/target states occupy. `reestimation.rs`'s coherence trigger becomes region-aware: re-fit the functor when the state crosses into a region the current functor wasn't fit for (region-boundary crossing as the re-estimation trigger, complementing the existing coherence-decay trigger).

**(c) `cgsp_runtime/` curiosity (`riir-engine/src/cgsp_runtime/`):**
Curiosity = prediction error. MFA reframing: a query is "novel" iff it activates a **region with low membership history** (the NPC hasn't been in this region recently) — a structural novelty signal. This complements the existing coherence-decay + JS-uniqueness curiosity signals. "New region fires" = structural novelty; "same region, high local-coordinate displacement" = within-region novelty.

**(d) LatCal fixed-point commitment (`riir-chain/src/encoding/`):**
The region-membership profile `{g_k(x)}` (K sigmoid gates) crosses the sync boundary as K raw f32 scalars. The MFA field itself (centroids + loadings) is a frozen library artifact referenced by shard hash, not per-NPC state. At K=8, the synced artifact is 8 floats per NPC — cleaner than syncing the full HLA embedding.

**(e) `NeuronShard` (`riir-neuron-db/src/`):**
The MFA artifact is a `RegionSubspaceShard` subtype. Layout: `[zone_hash(32) | centroids(K·D·4) | loadings(K·R·D·4) | log_pi(K·4) | psi_inv(D·4) | version(4) | blake3(32)]`. At K=8, D=8, R=2: 32 + 256 + 512 + 32 + 32 + 4 + 32 = 900 bytes, padded to 1024. `MerkleFrozenEnvelope` wraps it. The archetype library (K frozen NeuronShards, one per region centroid+loadings pair) is a separate frozen artifact referenced by hash.

**(f) DEC Stokes-calculus (`katgpt-core/src/dec/`):**
DEC `hodge_decompose` splits a flow into 3 channels (exact/harmonic/coexact) — a 3-region decomposition with fixed regions. MFA generalizes to K *learned* regions with data-driven boundaries. The DEC substrate provides the *infrastructure* for region-structured cochains; MFA provides the *learned region partition*. **Curse-of-dimensionality caveat:** DEC boundary-flux wins only for d ≤ 3 (game maps); MFA on HLA (d=8) or shards (d=64) does not benefit from boundary-only computation.

---

## 3. §3.5 Modelless Unblock Protocol (MANDATORY — passed)

The MFA training (fitting `{μ_k, W_k, Ψ, π}` via GD on negative log-likelihood) appears to need gradient descent. Before any riir-train deferral, exhaust the three modelless paths:

**Path 1 — Freeze/thaw snapshot correction:** ✅ **PASSES.** The trained MFA `{μ_k, W_k, Ψ, π}` is a frozen, BLAKE3-committable artifact. It is thawed at inference and consumed via closed-form operations (responsibilities, local coords, steering). No gradient descent at inference. The MFA is a freeze/thaw artifact by construction — exactly Path 1.

**Path 2 — Raw/lora reader-writer hot-swap:** N/A. The MFA is not a LoRA overlay; it's a standalone artifact. Path 1 covers it.

**Path 3 — Latent-space correction:** ✅ **PASSES for the consumption side.** All four inference-time operations are closed-form:
- Sigmoid membership gates: `g_k(x) = sigmoid(log π_k − 0.5·Mahalanobis(x, μ_k, C_k) − τ)` — dot-product + sigmoid, zero-alloc.
- Local coordinates: `ẑ_k = Z_k·(x − μ_k)` — matrix-vector multiply, pre-computed projector.
- Centroid steering: `(1−α)x + αμ_k` — SAXPY.
- Local steering: `x + W_k·v` — matrix-vector multiply.

**Deterministic constructor (§3.5 path 2 analog):** even without GD training, a usable MFA-like artifact can be **deterministically constructed** via K-means (offline, no GD) on a corpus of activations, followed by per-region PCA (closed-form eigendecomposition, no GD) to get the local loadings `W_k`. This is a modelless constructor that produces a region-conditioned artifact without any gradient descent. The GD-trained version (paper's method) will have better likelihood, but the K-means + PCA constructor is a modelless baseline that ships the same consumption interface.

**Verdict: MODELLESS-VALIDABLE.** The MFA is consumed modellessly (Path 1 freeze/thaw + Path 3 closed-form consumption). The training of the artifact itself is riir-train (or the deterministic K-means+PCA constructor for a modelless baseline). No riir-train deferral for the consumption primitive.

---

## 4. §3.6 Defend-wrong PoC — NOT REQUIRED

This verdict makes **no quality-parity claim** ("our modelless version matches MFA's steering performance" would require a PoC). The claim is purely architectural: the region-conditioned factor-analyzer structure is a clean generalization that fills a gap between Plan 412 (subspace, no regions) and Plan 409 (regions, no subspace). Per §3.6, "pure architectural redirects" and "no quality-parity claim" verdicts do not require a PoC.

If a future plan claims the modelless K-means+PCA constructor matches the GD-trained MFA's steering quality, THAT claim requires a head-to-head PoC in `riir-poc/`.

---

## 5. Verdict

### Tier: **GOAT** (open primitive)

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **YES (for the region-conditioned factor-analyzer).** Vocabulary translation done. Plan 412 (single block, no regions), Plan 409 (regions + translations, no per-region subspaces), Plan 321 (per-entity fixed, no regions) each cover ONE axis — none combines all three (K regions + per-region centroids + per-region subspaces + per-input routing). Grep for `factor analy|responsibilit|MFA|mixture of factor|local geometry|region.condit|centroid interpolation` returned ZERO codebase hits. Three-layer check (notes + code + vocab) done. | "factor analyzer" → "local subspace block"; "responsibilities" → "sigmoid membership gate"; "centroid" → "region anchor"; "local geometry" → "region-conditioned subspace". |
| Q2 New class of behavior? | **PARTIAL.** The operation "blend K region-conditioned subspace offsets by membership gates" is the CommittedFieldBlend / CHaRS operation class with a different routing signal (per-region Gaussian membership vs per-entity trajectory summary vs RBF-on-centroids). It's a **refinement + unification** of Plan 412 (within-region) and Plan 409 (between-region), not a new operation class. Consistent with R389 (GOAT) and R393 (GOAT) precedent — both were GOAT for the same Q2 reason. | |
| Q3 Product selling point? | **PARTIAL.** "NPCs have region-structured latent geometry — each NPC's HLA state lives in one of K regions, each with its own local subspace; steer by moving the region (centroid interpolation) OR walking within it (local offset)." Concrete and demoable, but refines the existing per-NPC personality/steering story (R290 / R297 / R321 / R393). | |
| Q4 Force multiplier? | **YES.** Connects Plan 412 (subspace), Plan 409 (centroids), Plan 321 (per-entity blend), HLA kernel, latent_functor, NeuronShard (MFA shard subtype), LatCal (commitment), DEC hodge_decompose (region = cell). ≥8 cousins. | |

**Not all-4-YES → not Super-GOAT.** The open primitive (region-conditioned subspace field) is a clean GOAT: provable generalization that unifies Plan 412 (within-region) × Plan 409 (between-region), subsumes Plan 412 at `K=1, μ=0, W=I`, enables two-mode steering (centroid + local) at `K≥2`. The GOAT gate (Plan 416) proves: (G1) degenerate `K=1` parity with Plan 412; (G2) `K≥2` two-mode steering produces distinct region/local effects; (G3) zero-alloc; (G4) latency within budget; (G5) BLAKE3 commitment determinism.

### Super-GOAT fusion candidate (NOT claimed — extends Issue 049)

The **Region-Structured HLA** fusion (#4 above) — fitting K factor analyzers on HLA trajectories, decomposing each NPC's emotional state into region assignment + local coordinates — is a Super-GOAT *candidate* that **makes Issue 049's "Block-Sparse HLA" speculative candidate concrete** (MFA gives the recipe: K-means + per-region PCA on HLA trajectories). But the novelty gate (Q1–Q4) is **not yet confident enough to commit**:
- Q1 is uncertain: does the existing HLA 5-scalar projection already implicitly capture region structure?
- Q2 is uncertain: is "region-structured emotional posture" a new capability or a re-interpretation?
- Q3 needs real game data to validate the selling point.

Per the workflow's "no candidate escape hatch" rule, this is **not** written as "Super-GOAT candidate" in the verdict. Issue 049 (already tracking Block-Sparse HLA) is extended with the MFA-recipe concrete construction; no new guide created until Q1–Q4 pass.

### One-line reasoning

MFA's training (GD fit) is riir-train; its modelless transferable primitive is the **region-conditioned factor-analyzer field** — K regions, each with a centroid (absolute position) and a local subspace (within-region variation) — consumed via per-region sigmoid membership gates + posterior-mean local coordinates + two-mode steering (centroid interpolation + local offset). GOAT: strict unification of Plan 412 (within-region subspace) × Plan 409 (region centroids), subsumes Plan 412 at the single-region degenerate limit, unlocks two-mode region/local steering.

### Routing

- **`katgpt-rs/.plans/416_region_subspace_field_primitive.md`** — open primitive. `RegionSubspaceField<D, K, R>` + `membership_gates` + `local_coordinates` + `steer_centroid` + `steer_local` + `decompose`. Feature flag `region_subspace_steering`. GOAT gate G1–G5.
- **`katgpt-rs/.issues/049_block_sparse_hla_supergoat_validation.md`** — extended: the MFA-recipe (K-means + per-region PCA on HLA trajectories) makes the Block-Sparse HLA candidate concrete. No new issue; the existing Issue 049 now references this research note as the construction recipe.
- **No private guide (riir-ai / riir-chain / riir-neuron-db) at this verdict tier.** GOAT does not trigger the mandatory-guide rule (§1.5).
- **No riir-train deferral for the consumption primitive.** The MFA training itself (GD fit) is riir-train; the modelless K-means+PCA constructor is a modelless baseline.

### MOAT gate per domain (§1.6)

- **`katgpt-rs` (public engine):** in-scope. Paper-derived fundamental primitive (region-conditioned factor-analyzer field for local-geometry decomposition and steering). Ships behind feature flag `region_subspace_steering`; GOAT gate decides promote-to-default vs demote. **Per-stack ledger:** this primitive occupies the "steering" slot alongside Plan 309 (1D), Plan 322 (2D phase-rotation), Plan 405 (Slerp), and Plan 412 (k-dim single block). Region-conditioned is the strict superset of Plan 412 — if the GOAT gate shows it subsumes Plan 412's use cases at acceptable overhead, Plan 412 stays as the K=1 degenerate case; both coexist (single-block for simple steering, region-conditioned for local-geometry steering).
- **`riir-ai` (private runtime):** the Region-Structured HLA fusion (#4) is pillar-adjacent (touches P2 neuron-db substrate, self-learn NPCs) — track via Issue 049. If it promotes to Super-GOAT, the private guide lands in `riir-ai/.research/`.
- **`riir-neuron-db` (private shards):** `RegionSubspaceShard` subtype is a neutral-Gain addition to the shard family (extends the freeze-envelope family with K-centroid + K-loadings layout). Not pillar-level on its own.
- **`riir-chain` (private chain):** LatCal commitment of MFA parameters (#7) is speculative P3.
- **`riir-train`:** the MFA training (GD/EM fit on activations) is a training-method note. The deterministic K-means+PCA constructor is modelless and stays in katgpt-rs.

---

## 6. Constraints check

| Constraint | Compliance |
|---|---|
| Modelless first | ✅ Consumption is closed-form (sigmoid gates, matrix-vector, SAXPY). Training is offline → frozen artifact (freeze/thaw). No GD at inference. |
| Latent-to-latent preferred | ✅ All operations are on latent state (HLA 8-dim, shard style_weights). Sigmoid gates (not softmax) for region membership. |
| Freeze/thaw over fine-tuning | ✅ The MFA artifact is frozen + BLAKE3-committed + atomic-swappable. |
| Self-learn welcome | ✅ Region memberships can drift at runtime (an NPC moves between emotional regions); the field itself stays frozen, the membership profile updates. |
| 5-repo discipline | ✅ Open primitive in katgpt-rs; private fusion (Region-Structured HLA) tracked in riir-ai via Issue 049; shard subtype in riir-neuron-db. |
| SOLID, DRY | ✅ Reuses Plan 412's subspace steering math for the local-offset operation; reuses Plan 409's centroid concept. New code is the region-conditioning + membership gates + two-mode dispatch. |
| Tests/examples | ✅ GOAT gate: G1 degenerate K=1 parity with Plan 412; G2 two-mode steering distinct effects; G3 zero-alloc; G4 latency; G5 commitment determinism. |
| CPU/GPU/ANE auto-route | ✅ At D=8 (HLA), K=8, R=2: ~200 FLOPs per membership gate, ~130 FLOPs per local-coordinate — plasma-tier (sub-µs), stays on SIMD. |
| Plasma → Hot → Warm → Cold → Freeze tiering | ✅ Frozen MFA artifact = Cold/Freeze tier; membership gates + local coords = Plasma tier; steering = Plasma tier. Sync boundary: K membership scalars per NPC (raw, deterministic). |

---

## 7. Open questions / risks

1. **K-means+PCA constructor quality.** The deterministic constructor (K-means + per-region PCA, no GD) produces a usable but likely lower-quality MFA than the GD-trained version. The GOAT gate should benchmark the constructor's reconstruction error against the paper's reported numbers (Table 4) to set expectations. If the gap is large, the GD-trained artifact (riir-train) is the production path; the constructor is the modelless baseline.

2. **Sigmoid vs softmax responsibilities.** The paper uses softmax (categorical, winner-take-all). We reformulate to per-region sigmoid (independent, multi-label). This is more expressive (multi-region membership) but changes the decomposition math — the reconstruction `x ≈ Σ_k g_k·[μ_k + W_k·ẑ_k]` needs a normalization (divide by `Σ_k g_k`) that the softmax version gets for free. The GOAT gate should verify reconstruction quality is acceptable with sigmoid gates.

3. **Region count K.** The paper uses K ∈ {1K, 8K, 32K} for LLM activations. For HLA (8-dim), K is bounded by 2^8 = 256 max, but practically K ∈ {4, 8, 16} (comparable to archetype counts in Plan 321, where K=3 is the default). The right K for game-scale HLA needs empirical validation.

4. **Noise covariance Ψ.** The paper uses a shared diagonal Ψ across all regions. For HLA, per-dimension noise might vary (the "reserved" 3 HLA dims might have higher noise). The constructor should support per-dimension Ψ but default to shared for simplicity.

5. **Curse of dimensionality.** MFA's local-geometry advantage is strongest in low-dim (d ≤ 16). For high-dim shards (d=64), the per-region subspace R must be kept small (R ≤ 4) or the projector `Z_k` becomes expensive. The DEC curse-of-dimensionality caveat applies: boundary-flux reasoning wins only for d ≤ 3.

---

## 8. References

- **Source paper:** [arXiv:2602.02464](https://arxiv.org/abs/2602.02464) — Shafran et al., "From Directions to Regions"
- **Code + trained MFAs:** https://github.com/ordavid-s/decomposing-activations-local-geometry
- **MFA origin:** Ghahramani & Hinton (1996), "The EM Algorithm for Mixtures of Factor Analyzers"
- **Closest cousin (same thesis):** R393 (Block-Sparse Featurizers, Goodfire) — "concepts are manifolds not lines"; Plan 412 ships the within-region primitive
- **Cluster-aware steering cousin:** R389 (CHaRS) — region centroids + OT routing; Plan 409
- **Per-entity MoE cousin:** R302 (FAME) — committed archetype blend; Plan 321
- **1D steering baseline:** R290, Plan 309 (`LatentSteeringVector`)
- **Spherical steering:** R382, Plan 405
- **Subspace discovery:** R279, Plan 301 (`subspace_phase_gate`)

## TL;DR (one-line)

MFA's region-conditioned factor-analyzer (K regions, each with centroid + local subspace) is the modelless unification of Plan 412 (within-region subspace) × Plan 409 (region centroids) — a clean GOAT open primitive consumed via per-region sigmoid membership gates + two-mode steering (centroid interpolation + local offset), with the Super-GOAT fusion (Region-Structured HLA) extending Issue 049.
