# Research 314: Group Invariance of f-Divergences and the Fisher–Rao Distance

> **Source:** [Group invariance of f-divergences and the Fisher–Rao distance](https://arxiv.org/pdf/2606.25790) — Frank Nielsen (Sony CSL) & Kazuki Okamura (Shizuoka Univ.), arXiv:2606.25790 [math.ST], 25 Jun 2026
> **Date:** 2026-06-26
> **Status:** Done
> **Related Research:** 219 (DEC operators — adjacent geometry, different object), 046 (Symmetry-Compatible Equivariant Optimizers → riir-train), 270/Plan 270 (Gauge-Invariant Adapter Composition — the closest cousin, narrower gauge)
> **Related Plans:** 270 (gauge-invariant adapter compose — already ships the R+ scaling case)
> **Classification:** Public

---

## TL;DR

A pure information-geometry result: every f-divergence (KL, Hellinger, TV, χ²) and the Fisher–Rao geodesic distance between two members of a transformation model is **invariant under the diagonal group action**, and therefore depends only on a **maximal invariant** of the parameter pair. For transitive actions the maximal invariant is a **double coset** `H g₁⁻¹ g₂ H`; for multidimensional location-scale families it reduces to an explicit, finite signature: the singular values of `V₂⁻¹V₁` (with multiplicities) plus the block norms of `UᵀV₂⁻¹(μ₁−μ₂)` (Proposition 3.3 / 4.2).

**Distilled for katgpt-rs (modelless, inference-time):** the transferable primitive is *"reduce a pair of parameters to its symmetry-canonical form, then evaluate any invariant divergence as a function of that form."* It is a generic, training-free, deterministic mathematical reduction — exactly the shape of a katgpt-rs primitive.

**Verdict: GAIN — deferred.** The framework is mathematically novel relative to our corpus (zero prior hits on `maximal_invariant`, `double_coset`, `fisher_rao`, `bregman`, `fenchel`), but it does **not** map onto our current latent-state shapes. Our vector latent states (HLA 8-dim affect, `NeuronShard::style_weights[64]`) carry **trivial symmetry groups** — dimensions are either semantically distinct (valence ≠ arousal) or BLAKE3-committed (fixing the basis) — so the maximal invariant collapses to the full state and the machinery adds nothing. The one non-trivial symmetry we actually have (LoRA `(A,B)` factorization gauge under R+ scaling) is **already shipped** by Plan 270. The framework becomes directly actionable only if/when we introduce `(μ, Σ)` Gaussian-belief latent states.

---

## 1. Paper Core Findings

### 1.1 The invariance theorem (Theorem 2.1)

Given a *transformation model* — a group `G` acting measurably on sample space `X` and on parameter space `Θ`, with a multiplier `χ` satisfying `χ(g)·p(gx | gθ) = p(x | θ)` — **every** f-divergence is invariant:

```
D_f(gθ₁ : gθ₂) = D_f(θ₁ : θ₂)   for all g ∈ G.
```

This subsumes KL, reverse-KL, Hellinger, TV, χ², α-divergences, etc. in one stroke.

### 1.2 Maximal invariant (Theorem 2.2)

A map `m(θ₁, θ₂)` that is constant on each diagonal orbit `O(θ₁,θ₂)` and separates distinct orbits is a **maximal invariant**. Every invariant divergence is necessarily a function of `m`:

```
D_f(θ₁ : θ₂) = F_f(m(θ₁, θ₂)).
```

So `m` is a *sufficient statistic* for the entire class of invariant divergences — compute it once, evaluate any f-divergence on it.

### 1.3 Double-coset form for transitive actions (Theorem 2.3)

If `G` acts transitively on `Θ`, identify `Θ ≅ G/H` for the stabilizer `H` of a base point. Then the maximal invariant for pairs is the **double coset**:

```
m(g₁H, g₂H) := H g₁⁻¹ g₂ H ∈ H\G/H.
```

Two parameter pairs have the same double coset **iff** they lie on the same diagonal orbit. This is the geometric "relative position" of two points in a homogeneous space.

### 1.4 Explicit SVD form for location-scale families (Proposition 3.3 — the paper's main constructive contribution)

For the multidimensional location-scale family with `(μ, V) ∈ ℝᵈ × GL(d,ℝ)/O(d,ℝ)` (the quotient kills the orthogonal-rotation gauge so identifiability holds), the maximal invariant of `((μ₁,V₁),(μ₂,V₂))` is given by:

1. `S := V₂⁻¹ V₁`, `ν := V₂⁻¹(μ₁ − μ₂)`.
2. SVD: `S = U Σ Wᵀ`, `Σ = diag(τ₁ I_{m₁}, …, τ_k I_{m_k})`, `τ₁ > … > τ_k > 0`.
3. Decompose `z := Uᵀν = (z⁽¹⁾, …, z⁽ᵏ⁾)` by singular-value multiplicity blocks.
4. **Maximal invariant** = `(τⱼ, mⱼ, rⱼ := ‖z⁽ʲ⁾‖)_{j=1..k}`.

So every f-divergence between two location-scale densities depends **only** on these singular values (with multiplicities) and block norms. The naive `Σ₂^{-1/2}Σ₁^{1/2}` form fails to be a maximal invariant for `d ≥ 2` because it can be asymmetric / non-PD; the SVD-block-norm fix is the paper's repair.

### 1.5 Fisher–Rao distance reduction (Proposition 4.2)

The **same** invariant reduction applies to the Fisher–Rao Riemannian geodesic distance — it is a function of the same `(τⱼ, mⱼ, rⱼ)` signature. Affine invariance lets you translate the pair to canonical form `((ν, [S]), (0, [I_d]))` before computing.

### 1.6 Dually-flat / Bregman side result (Example 3.4)

For the centered matrix scale family under `GL(d,ℝ)` congruence, the log-det Bregman/Fenchel–Young divergence is also invariant, and the maximal invariant reduces to the **eigenvalue spectrum** of `Σ₂⁻¹Σ₁`. Connects to V-potential geometry on the SPD cone (Ohara–Eguchi).

---

## 2. Distillation

### 2.1 Why it does NOT directly map to our codebase (the honest assessment)

The framework's constructive power lives in §1.4 — the SVD-block-norm signature for **(μ, V) location-scale parameters**. Auditing our latent-state shapes against this requirement:

| Latent state | Shape | Natural symmetry group | Maximal invariant | Paper applies? |
|---|---|---|---|---|
| HLA affect (`HlaQHeadState`: valence, arousal, desperation, calm, fear + 3) | 8-dim **vector** (semantic axes) | **trivial** — axes are semantically distinct; rotating valence↔arousing is meaningless | the vector itself | ✗ adds nothing |
| HLA `sk = Σ k_i k_iᵀ` (`HlaLayerState`) | d×d unnormalized **activation covariance** | orthogonal conjugation `sk ↦ Q sk Qᵀ` *if* the key basis is gauge — but basis is fixed by `W_k`, not free | full `sk` (basis is pinned) | ✗ adds nothing |
| `NeuronShard::style_weights[64]` | fixed-size Pod **vector** | **trivial** — BLAKE3 commitment fixes the byte basis; any permutation breaks the hash | the vector itself | ✗ adds nothing |
| Frozen LoRA pair `(A, B)` | rank-r factors | **R+ scaling** `(A,b) ↦ (A·c, B/c)` — leaves `ABᵀ` invariant | `ABᵀ` (the merged weight) | ✓ **already shipped** as Plan 270 |
| Attention distribution over tokens | categorical (discrete) | — | — | ✗ not location-scale |
| DEC cochains (`CochainField`) | k-cell features on a spatial complex | — | — | ✗ different object (forms on a mesh, not params on a manifold) |

**The structural mismatch:** Nielsen–Okamura's machinery is for *parametric distribution families with a rich continuous symmetry* (affine group on `(μ,Σ)`). Our latent states are either (a) semantic vectors with no non-trivial symmetry, (b) BLAKE3-committed blobs where the basis is pinned by the hash, or (c) the one gauge case (LoRA R+ scaling) that Plan 270 already covers. There is **no shipped `(μ, Σ)` latent state** for the SVD-block-norm signature to reduce.

### 2.2 What IS transferable (the conceptual primitive)

Even though the constructive §1.4 algorithm has no current target, two conceptual results are worth keeping in the toolbox:

1. **"Every invariant divergence is a function of the maximal invariant"** (Theorem 2.2). This is the information-geometric analog of *"find the sufficient statistic before comparing."* It tells us: if we ever introduce a symmetry on a latent space, compute the maximal invariant once and reuse it for KL *and* Hellinger *and* TV *and* Fisher–Rao — don't recompute each from scratch.
2. **The double-coset `H g₁⁻¹ g₂ H` as "relative position"** (Theorem 2.3). A deterministic, low-dimensional, basis-independent signature for a *pair*. This is the conceptual cousin of BLAKE3 (which signs a *single* blob) — the double coset signs a *pair* up to symmetry.

### 2.3 Why not Super-GOAT (novelty gate, all four asked)

- **Q1 No prior art?** PARTIAL. The general framework (maximal invariants, double cosets, f-divergence invariance) has **zero** hits across notes + code in all five repos (confirmed by grep on `fisher_rao|fisher_metric|maximal_invariant|double_coset|bregman|fenchel|PositiveDefinite`). BUT the *practical instance* we'd actually use — LoRA factorization gauge invariance — is already shipped by Plan 270 (17/17 GOAT pass, default-on). So the un-shipped part is the *general theory*, not a capability gap.
- **Q2 New class of behavior?** NO. It is a mathematical *refinement* of distribution/snapshot comparison. We already compare distributions (dot-product + sigmoid, KL in distillation, BLAKE3 equality on shards). This gives a canonical symmetry-reduced form, not a new capability class.
- **Q3 Product selling point?** Cannot finish the sentence. "Our NPCs compare belief states invariantly" fails because our belief states have trivial symmetry — there's nothing to quotient out.
- **Q4 Force multiplier?** Weak/theoretical only. Connections to HLA `sk` covariance, `NeuronShard` comparison, LatCal commitment of relative position, and DEC quotient cochains all exist on paper but require a `(μ, Σ)` latent-state redesign to materialize.

→ Fails Q2/Q3/Q4. **Not Super-GOAT.**

### 2.4 Fusion (speculative — documented for the record, no plan this session)

| Fusion | What it would produce | Blocker |
|---|---|---|
| Maximal invariant × **HLA `sk` as Gaussian belief covariance** | Invariant NPC-to-NPC affect comparison via §1.4 signature → symmetry-aware social clustering, symmetry-aware KG triple emission | Requires reinterpreting `sk` (activation covariance) as belief covariance — a design change to `hla/`, not a primitive extraction |
| Double-coset × **LatCal commitment** | Chain-transportable "relative position of two shards/beliefs" assertion: `(τⱼ, mⱼ, rⱼ)` → LatCal fixed-point → BLAKE3 | Requires `(μ, Σ)` shard geometry we don't have; current shards commit *single* blobs, not *pairs* |
| Maximal invariant × **DEC quotient cochain** | Symmetry-reduced Hodge decomposition on a quotient cell complex | DEC operates on spatial meshes, not parameter manifolds; the bridge is tenuous |
| §1.4 SVD signature × **Plan 270 gauge-invariant compose** | Generalize Plan 270 from R+ scaling gauge to *affine* gauge on `(A,B)` pairs | Speculative — would need a concrete multi-group LoRA scenario to justify |

None of these are actionable today. The first one (HLA `sk` reinterpretation) is the most likely to become real, but it is a riir-ai design decision, not a katgpt-rs primitive.

---

## 3. Verdict

**GAIN — deferred (no plan this session).**

| Criterion | Result |
|---|---|
| Modelless? | ✅ pure math, no training, deterministic reductions |
| Latent-to-latent? | △ would be, *if* we had `(μ, Σ)` latent states |
| Novel vs corpus? | ✅ zero prior hits on the general framework |
| Novel vs Plan 270? | ✗ the practical gauge case is already shipped |
| Maps to current data shapes? | ✗ HLA/shards have trivial symmetry; no `(μ, Σ)` latent state |
| New capability class? | ✗ refinement of existing comparison primitives |
| Product selling point? | ✗ cannot articulate one on current shapes |
| Force multiplier (≥2 pillars)? | ✗ connections are theoretical, require redesign |

**One-line reasoning:** Beautiful and technically un-shipped as a general framework, but our vector/Blob latent states carry trivial symmetry groups — the maximal invariant collapses to the state itself — and the one non-trivial gauge we have (LoRA R+) is already covered by Plan 270. Reopen if we introduce Gaussian `(μ, Σ)` belief states (e.g. a future Plan that gives HLA a covariance-interpretable belief layer, or a shard layout that stores `(μ, Σ)` pairs).

**Trigger condition for re-evaluation (→ upgrade to GOAT + plan):**
- A future plan introduces a `(μ, V)` or `(μ, Σ)` latent state (Gaussian belief, distributional embedding, second-order personality representation).
- At that point, the §1.4 SVD-block-norm signature becomes the canonical comparison primitive for that state, and a `katgpt-rs/src/info_geometry/` module (gated `invariant_divergence`) implementing `location_scale_maximal_invariant((μ₁,V₁),(μ₂,V₂)) -> (τ[], m[], r[])` + `eval_f_divergence(F, signature)` becomes a concrete GOAT candidate.

Until then, the value of this note is **conceptual**: it documents that (a) "invariant comparison = function of maximal invariant" is the right mental model when we *do* introduce symmetries, and (b) Plan 270 is the R+-gauge instance of this broader theorem.
