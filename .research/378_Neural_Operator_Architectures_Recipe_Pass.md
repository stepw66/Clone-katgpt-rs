# Research 378: Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning — PASS

> **Source:** Julius Berner, Miguel Liu-Schiaffini, Jean Kossaifi, Valentin Duruisseaux, Boris Bonev, Kamyar Azizzadenesheli, Anima Anandkumar. *Principled approaches for extending neural architectures to function spaces for operator learning.* **Nature Machine Intelligence** (2026). [doi:10.1038/s42256-026-01267-z](https://doi.org/10.1038/s42256-026-01267-z)
> **Date:** 2026-07-04
> **Status:** Done — PASS
> **Related Research:** 219 (TNO→DEC), 280 (RTDC), 291 (cross-resolution transport, **DEFAULT-ON**), 296 (Stokes/DEC vocabulary crosswalk), 303 (Transolver→FUNCATTN), 306 (Galerkin→FUNCATTN), 307 (FNO practical perspective, **same authors**, Gain)
> **Classification:** Public

---

## TL;DR

This Nature Machine Intelligence perspective is the **canonical unifying reference** for the neural-operator line (FNO + GNO + OFORMER + integral-transform + encoder-decoder layers). It is a **training-architecture design paper** — FNO/OFORMER/GNO still require gradient descent on PDE data to be useful — and the **principles** it distills (discretization-agnostic operators on coordinates; quadrature-weighted aggregation; fixed-parameter latent interfaces; coordinate-aware receptive fields) **are already captured across our corpus AND shipped as code**. Every modelless principle the paper names maps 1:1 onto a shipped primitive. Verdict: **PASS** — the companion FNO paper by the **same authors** was already evaluated as **Gain** in Research 307; this broader perspective adds no new modelless primitive beyond what 307 covered.

**Distilled for katgpt-rs (modelless, inference-time):** nothing net-new. The paper's value is its unified vocabulary and Fig. 2 recipe (NN layer → identify continuous operator → discretize with quadrature weights) — useful pedagogically, but every primitive it motivates already ships.

---

## 1. Paper Core Findings (for cross-reference)

The paper distills four **defining principles** for neural operators and gives a recipe for converting NN layers into operator layers:

| Principle | Paper statement | Our shipped analog |
|---|---|---|
| **Discretization-agnostic** | The learned map is applicable at any discretization; outputs differ only by vanishing discretization error. | DEC substrate (`katgpt-dec`): `exterior_derivative`, `codifferential`, `hodge_laplacian`, `hodge_decompose` operate on a `CellComplex` keyed by **coordinates, not indices** (Plan 251, Research 219). `cross_resolution_transport` (DEFAULT-ON since 2026-06-23, Plan 310) is strictly stronger than FNO — composes cross-resolution with cross-domain in one 4-matrix product. |
| **Fixed number of parameters** | Parameter count independent of discretization. | Every `NeuronShard` is a fixed-layout Pod (`style_weights[64]`, `#[repr(C)]`); HLA is a fixed 8-dim latent state; `ShardIndex` is a lock-free map with fixed-size values. The principle is **structural** in our codebase. |
| **Universal approximation** | Family can approximate any sufficiently regular operator. | N/A at inference (we don't train); the modelless analog is "the operator set is rich enough to express the runtime behavior" — covered by composing DEC + functor + HLA. |
| **Discretization convergence** | Output converges as discretization refines. | `cross_resolution_transport` ships exactly this contract (Research 291). RTDC (Research 280) commits the resolution-tiered result across the sync boundary. |

The paper's **layer-correspondence table** (FC → integral transform; conv → spectral/FNO; GNN → GNO; transformer → OFORMER; encoder-decoder → latent interface) maps onto shipped primitives:

| Paper's operator layer | Our shipped analog |
|---|---|
| **Integral-transform layer** (FC → `∫ K(x,y)f(x)dx`) | `latent_functor/arithmetic.rs` functor applications; `SenseModule::project`; `funcattn` integral-kernel attention. |
| **Pointwise layer** (`g(x)=K(f(x))`) | Every sigmoid-gated projection in HLA, `phase_rotation`, `dirichlet`, `dendritic_gate`. |
| **Convolutional / spectral layer** (FNO) | `cross_resolution_transport` (DEFAULT-ON); DEC `exterior_derivative`; `katgpt-rs/crates/katgpt-core/src/spectral/`; `LatCalSpectralFixed` for sync-boundary Fourier commitment. |
| **Graph neural operator** (GNO) | DEC on a `CellComplex`; `viable_manifold_graph`; `zone_manifold`; `InterestCohain` lattice edges. |
| **Attention operator** (OFORMER) | `funcattn` (Research 303 Transolver + 306 Galerkin = FUNCATTN grandparent/predecessor line). |
| **Encoder-decoder latent interface** | `ShardIndex::query` (cosine retrieval → fixed-dim latent); `ItemEmbedIndex`; HLA scalar projection (latent → 5 synced scalars). |

The paper's **quadrature weights Δi** mechanism (Fig. 5 — irregular point clouds need per-point weights for sums to converge to a unique integral) is the one mechanism with the narrowest prior-art coverage. It already ships in two narrow contexts: Gauss-Legendre quadrature in `katgpt-dec/src/nonlinear_heat_kernel.rs` (Plan 359 Phase 3) and Gauss-Hermite quadrature in DMFT mean-field integration (Plan 371). It is **not** wired as a *general* aggregation principle for irregular point clouds (fog-of-war visible sets, crowd NPCs in zones) — see §3.

---

## 2. Why PASS (not Gain, not Super-GOAT)

### 2.1 The novelty gate fails on every axis (Q1–Q4)

| Q | Answer | Notes |
|---|--------|-------|
| **Q1 No prior art?** | ❌ NO | The **companion paper** by the **same author team** (Duruisseaux/Kossaifi/Anandkumar, arxiv 2512.01421, "FNO explained: a practical perspective") was already evaluated as **Gain** in Research 307 with explicit reasoning: "the FNO headline inference primitive (resolution-invariant spectral transport) **already ships as our Super-GOAT `cross_resolution_transport`**". This Nature paper is the *broader perspective* that subsumes that companion. The principle of "coordinate-aware operators on a cell complex" was distilled even earlier in Research 219 (TNO→DEC). Research 296 documented the vocabulary-translation lesson that applies here too. |
| **Q2 New capability class?** | ❌ NO | Every modelless principle the paper names is already a shipped capability class (DEC, cross-resolution transport, FUNCATTN, fixed-dim shard latent interfaces). |
| **Q3 Product selling point?** | ❌ NO | Cannot finish "our NPCs do X that no competitor can" with anything from this paper — the relevant X (cross-resolution latent transport, coordinate-aware operators) already shipped and is already a selling point per `riir-ai/.docs/pillars/`. |
| **Q4 Force multiplier (≥2 pillars)?** | ❌ NO | Doesn't multiply any pillar beyond what `cross_resolution_transport` + DEC already multiply. |

**Verdict: 0/4 YES → not Super-GOAT.**

### 2.2 Why not even Gain

Research 307 already enumerated the **three narrow Gain-tier gaps** in this paper family (Fourier continuation for non-periodic latent fields; standalone FFT-based spectral differentiation; Tucker/HOSVD for `NeuronShard` compaction — the latter already promoted to default-on as `tucker_factorization` per `riir-neuron-db/README.md`). Those gaps are tracked there. **This broader perspective adds no fourth gap** — every principle it generalizes is already covered.

### 2.3 §3.6 defend-wrong PoC consideration

Per the §3.6 rule, a PASS verdict backed only by architectural reasoning is the #1 false-PASS failure mode. **This PASS is not architectural-only** — it rests on:
1. **Shipped code**: DEC operators (`katgpt-dec`), `cross_resolution_transport` (DEFAULT-ON), quadrature in `nonlinear_heat_kernel.rs` (Plan 359 Phase 3 GOAT gate passed).
2. **Prior verdict on the companion paper**: Research 307 evaluated the **same author team's** FNO paper as Gain with explicit Q1–Q4 reasoning. This Nature paper is the strict superset of that companion; a superset of a Gain-tier paper cannot become Super-GOAT without a new primitive, and none is identified.
3. **No quality-parity claim is being made** — we are not asserting "our runtime matches the paper's numbers on Navier-Stokes" (that would be a training-time claim → riir-train). We are asserting only that the **modelless principles** the paper distills are already shipped, which is a grep-verifiable architectural claim that does not require a PoC.

The PASS is grounded.

---

## 3. The one near-Gap worth noting (not opening a plan)

**General quadrature-weighted aggregation on irregular point clouds.** The paper's Fig. 5 makes the case crisply: aggregating function values at irregularly spaced points with uniform weights (1/n) does not converge to a unique integral as the discretization refines — dense regions dominate. Quadrature weights (Δi = local cell volume, e.g. via Voronoi/Delaunay or Monte-Carlo density) fix this.

**Status in our codebase:**
- ✅ Regular grids (game maps): quadrature weights collapse to uniform → standard aggregation is correct. **No gap.**
- ✅ Gauss-Legendre quadrature for time integration: ships in `nonlinear_heat_kernel.rs` (Plan 359). **No gap.**
- ✅ Gauss-Hermite quadrature for mean-field DMFT integrals: ships in Plan 371. **No gap.**
- ⚠️ General irregular point clouds (fog-of-war visible sets, crowd NPCs in a zone with non-uniform spatial density): aggregation currently uses uniform weights or pre-bucketed zone densities. A `quadrature_weighted_aggregate(points, values, weights)` primitive would make these aggregations integral-consistent.

**Why not opening a plan:** the irregular-point-cloud case is a narrow refinement of already-working aggregations (`zone_density`, `crowd_joint_inference`, `SalienceTriGate::decide_batch`). Game-map grids are regular; fog-of-war visible sets are bounded and the bias from uniform weights is small and already absorbed by HLA's sigmoid projection. The cost-benefit doesn't clear the GOAT gate. **If a future zone-attention or crowd-aggregation feature shows resolution-dependent bias, this is the lever to pull** — track via an `.issues/` entry at that point, not pre-emptively.

---

## 4. Routing

- **No katgpt-rs plan** (PASS, no new primitive).
- **No Super-GOAT guide** (novelty gate fails Q1–Q4).
- **Training parts → riir-train** (one-line note): the paper's empirical content (training FNO/GNO/OFORMER on Navier-Stokes, multi-resolution curriculum, physics-informed residual loss, Adam with model-specific schedules) is training procedure → riir-train.
- **Canonical reference value**: this Nature paper is the **single best citation** for the neural-operator design principles that our DEC + `cross_resolution_transport` + FUNCATTN primitives instantiate. Future research notes that need to motivate coordinate-aware operators or quadrature-weighted aggregation should cite this paper (and Research 219 + 307) rather than re-deriving the principles.

---

## 5. Cross-references

- `katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md` — **the companion paper by the same authors, already Gain.** This PASS rests primarily on 307's prior verdict.
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — neural operators → DEC substrate distillation (the coordinate-aware-operator principle).
- `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md` — the FNO headline primitive, **already shipped DEFAULT-ON**.
- `katgpt-rs/.research/280_Resolution_Tiered_Deterministic_Commitment.md` — RTDC: cross-resolution commitment across the sync boundary.
- `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — vocabulary-translation lesson (paper vocab ↔ codebase vocab) that applies here.
- `katgpt-rs/.research/303_Transolver_Physics_Attention_FUNCATTN_Predecessor.md` + `306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md` — the OFORMER / physics-attention line.
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` + `252` — DEC operators (the shipped coordinate-aware substrate).
- `katgpt-rs/.plans/310_cross_resolution_spectral_transport_primitive.md` — execution record for the DEFAULT-ON cross-resolution primitive.
- `katgpt-rs/.plans/359_dec_heat_kernel_trajectory.md` — quadrature (Gauss-Legendre) in the nonlinear heat kernel.
- → **riir-train** for FNO/GNO/OFORMER training, multi-resolution curriculum, physics-informed residual loss.

## TL;DR

Nature MI 2026 perspective unifying FNO/GNO/OFORMER under common neural-operator design principles. The principles (discretization-agnostic coordinate-aware operators, quadrature-weighted aggregation, fixed-dim latent interfaces) **already ship** as DEC + `cross_resolution_transport` (DEFAULT-ON) + fixed-layout `NeuronShard` Pods. The companion FNO paper by the **same authors** was already evaluated as Gain in Research 307; this broader perspective adds no new modelless primitive. **PASS** — no files created in this session beyond this note; no plan opened; training content → riir-train. The paper's value going forward is as the **canonical citation** for the design principles our DEC/cross-resolution/FUNCATTN primitives instantiate.
