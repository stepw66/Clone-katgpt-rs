# Research 395: NNs → NOs — Principled Recipe for Extending Neural Architectures to Function Spaces

> **Source:** [Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning](https://arxiv.org/pdf/2506.10973) — Berner, Liu-Schiaffini, Kossaifi, Duruisseaux, Bonev, Azizzadenesheli, Anandkumar (NVIDIA + Caltech), arXiv:2506.10973v1, 12 Jun 2025, 38pp
> **Date:** 2026-07-09
> **Status:** Done
> **Related Research:** 307 (FNO Practical Perspective — same group's follow-up, the headline distillation), 219 (TNO → DEC operators), 257 (FUNCATTN), 291 (Cross-Resolution Spectral Transport — **the Super-GOAT that emerged from this paper family**), 303/306 (FUNCATTN predecessors — Transolver, Galerkin)
> **Related Plans:** 251 (DEC), 286 (FUNCATTN), 310 (Cross-Resolution Spectral Transport — DEFAULT-ON), 323 (Fourier continuation), 325 (spectral differentiation), 326 (Tucker/HOSVD)
> **Classification:** Public

---

## TL;DR

This is the **progenitor recipe paper** for the NVIDIA/Caltech neural-operator framework (FNO, GNO, SFNO, UNO, DeepONet, Transformer-NO). It distills the general pattern: *identify the continuous analog of an NN layer → parametrize its weights as learnable functions → discretize with quadrature weights → fix the receptive field w.r.t. the underlying domain*. Nearly the entire recipe is **training-side** (the kernel function `K`, basis `Φ`, encoder/decoder are all learned via backprop) → **riir-train**. The narrow modelless subset — quadrature-weighted aggregation, fixed-radius neighborhoods, encoder-decoder inner products, pointwise operators — is **fully subsumed by already-shipped primitives**. **Research 307 already distilled the more specific follow-up paper (FNO Practical Perspective); all three of its Gain-tier follow-up plans (323, 325, 326) shipped.** No new primitive. No new capability class. No Super-GOAT.

**Distilled for katgpt-rs (modelless, inference-time):** Nothing new. Every modelless piece maps to existing code (see §2.2 table). Training-side recipe → riir-train.

---

## 1. Paper Core Findings

### 1.1 What the paper IS

A **recipe / survey paper**. It does not introduce a new architecture. It unifies and motivates the conversion pattern shared by FNO/GNO/SFNO/UNO/DeepONet/Transformer-NO, framing them all as instances of:

```
g(yⱼ) = ∫_{x ∈ D(yⱼ)} K(x, yⱼ, f(x), f(yⱼ)) dx   (the graph neural operator, Eq 11/18/19)
```

Five conversion instances are demonstrated:

| NN Layer | → Neural Operator Layer | Core Modification |
|---|---|---|
| Fully-connected (§3.1) | Integral transform `g(yⱼ) = Σᵢ K(xᵢ,yⱼ)f(xᵢ)Δᵢ + b(yⱼ)` | Parametrize `K_{ji}`, `b_j` as evaluations of learnable functions; add quadrature weights `Δᵢ` |
| Convolution (§3.3) | Spectral conv (FNO) or local integral kernel | Parametrize kernel as learnable function `K(yⱼ−x)` with fixed support `2r` |
| Graph NN (§3.4) | Graph neural operator (GNO) | Quadrature weights for aggregation; neighborhoods defined by domain subsets `D(yⱼ) = B_r(yⱼ)` (radius graphs), not nearest-neighbor counts |
| Self-attention (§3.5) | Transformer neural operator | Quadrature weights in numerator AND denominator (normalization) of softmax |
| Encoder-decoder (§3.6) | Inner-product encoder + linear latent map + dictionary decoder (DeepONet, NOMAD, FNO as Fourier-basis special case) | Encoder uses inner products `vⱼ = ⟨bⱼ, f⟩_{L²}`; decoder uses linear combinations `g(y) = Σⱼ wⱼ bⱼ(y)` |

### 1.2 The four pillars of the recipe (§3.8)

1. **Identify the continuous analog** — what operator does the NN layer roughly discretize?
2. **Parametrize weights as learnable functions** — `K_{ji} → K(xᵢ, yⱼ)` etc.
3. **Quadrature-weighted aggregation** — `Δᵢ` ensures convergence to a unique integral as resolution refines.
4. **Fixed receptive field w.r.t. the underlying domain** — define neighborhoods by domain subsets `D(yⱼ)`, not by neighbor counts.

### 1.3 Discretization convergence (§2.1, App A.4)

The unifying theoretical property: a *discretization-convergent* operator satisfies

```
sup_{f∈K} ‖NO_θ(f|Xₙ) − NO_θ(f)‖_G → 0   as n → ∞
```

for any discrete refinement `(Xₙ)`. This is what makes outputs "consistent across resolutions" — the headline neural-operator capability. The paper proves a triangle-inequality bound (App B, Eq 41): `ε(f, X̃) ≤ ε_approx + ε_opt + 2ε_discr(f,X) + ε_discr(f,X̃)`.

### 1.4 Transferable modelless primitives (the parts not requiring backprop)

Almost everything in the recipe is training-side. The modelless residue:

| Primitive | Math | Modelless? |
|---|---|---|
| Quadrature weights from point cloud | `Δᵢ = volume of partition cell containing xᵢ` (Riemann), or `1/n` (Monte Carlo), or Delaunay volume share (Alg 1) | ✅ Closed-form |
| Radius-graph neighborhood | `D(yⱼ) = B_r(yⱼ)` — fixed-radius spatial query | ✅ Free (spatial partitioning) |
| Positional encoding (coordinate concatenation) | `f(xᵢ) → (f(xᵢ), xᵢ)` (optionally via sin/cos frequencies) | ✅ Free |
| Pointwise operators (Nemytskii) | `g(xᵢ) = K(f(xᵢ))` — activations, 1×1 convs | ✅ Free, resolution-agnostic by construction |
| Function-space mean/variance (normalization) | `μ = (Σᵢ f(xᵢ)Δᵢ) / (Σᵢ Δᵢ)` (Eq 16) | ✅ Free |
| Function-space L²/Sobolev loss | `Σⱼ |g(yⱼ) − g*(yⱼ)|² Δⱼ` | ✅ Free |
| Inner-product encoder (Eq 15) | `vⱼ = Σᵢ [bⱼ(xᵢ)]* f(xᵢ) Δᵢ ≈ ⟨bⱼ, f⟩_{L²}` | ✅ Free **given** a fixed basis `bⱼ` |
| Linear-combination decoder (Eq 33) | `g(y) = Σⱼ wⱼ bⱼ(y)` | ✅ Free **given** a fixed dictionary `bⱼ` |

The kicker is in **"given a fixed basis"** — in the paper, `bⱼ` (and `K`, `Φ`, the encoder/decoder networks) are *learned end-to-end via backprop*. That is genuinely a training procedure.

### 1.5 Training-side content → riir-train

Per skill §3.5 modelless-unblock check: NONE of the following can be unblocked via freeze/thaw or raw/lora hot-swap or latent correction. They are genuinely training procedures.

- **Learning the kernel `K`** — the universal-approximation guarantee (App B) requires the kernel to be expressive enough; that expressivity comes from training.
- **Learning the basis `bⱼ`** (DeepONet trunk net, FNO Fourier-truncation schedule, PCA-Net).
- **Multi-resolution curriculum training**, **autoregressive rollout training** (pushforward noise injection).
- **Physics-informed (PINN-style) loss** as supervision-free regularizer (requires differentiating the output function w.r.t. coordinates).
- **Sobolev loss** (requires ground-truth derivatives).
- **Multi-objective loss balancing** (SoftAdapt / ReLoBRaLo).

→ riir-train.

---

## 2. Distillation

### 2.1 Transferable primitive (modelless subset)

The modelless residue of this paper reduces to **three operations on a discretized function** with a **pre-chosen basis**:

```
// Inner-product encode (Eq 15)
v[j] = Σ_i basis[j](x_i)* · f(x_i) · Δ_i      // ≈ ⟨basis[j], f⟩_L²

// Latent linear map (the only "learning" — but if K is fixed, it's free)
w = K · v                                       // K ∈ R^{k×k}

// Linear-combination decode (Eq 33)
g(y) = Σ_j basis[j](y) · w[j]
```

This is **exactly** the FUNCATTN mechanism (Research 257, Plan 286): `Φ^T Q` (encode) → `C ∈ R^{k×k}` (latent map) → `Ψ w` (decode). The paper itself confirms this in Appendix A.8 — the spectral convolution, GNO, integral transform, and DeepONet are all special cases of the encoder-decoder pattern.

### 2.2 Where the pieces already live (BOTH layers, ALL repos)

| Paper concept | Shipped primitive | Match |
|---|---|---|
| Cross-resolution spectral transport (the headline FNO capability) | `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs::transport_cross_resolution_into` (Plan 310, **DEFAULT-ON**) | **Strictly stronger** — composes cross-resolution × cross-domain transport in one 4-matrix product |
| SpectralConv (FNO §3.3) | `katgpt-rs/crates/katgpt-core/src/funcattn.rs` (Plan 286) + `spectralquant/spectral_kv_cache.rs` (Plan 039) | FUNCATTN = SpectralConv with frozen bases |
| Encoder-decoder inner-product operator (paper §3.6, App A.8) | `funcattn` + `cross_resolution_transport` (same machinery, different vocabulary) | **Exact** — see Research 257 §1.1, 307 §2.2 |
| Integral operator / GNO (§3.1, §3.4) | `katgpt-rs/crates/katgpt-dec/src/operators.rs::exterior_derivative` (Plan 251) | **DEC formulation** — `d` on a cell complex IS a GNO with kernel = incidence matrix |
| Spectral differentiation `F{∂^m f} = (ik)^m f̂` (App A.6) | `katgpt-rs/crates/katgpt-core/src/spectral/differentiation.rs` (Plan 325, **DEFAULT-ON**) + DEC `exterior_derivative` | **Shipped** as the specialized periodic-1D case where DEC is overkill |
| Non-periodic boundary handling (Fourier continuation) | `katgpt-rs/crates/katgpt-core/src/spectral/continuation.rs` (Plan 323) | **Shipped** — the gap Research 307 §3 identified, now closed |
| Tucker / HOSVD weight factorization (TFNO §6.1) | `katgpt-rs/crates/katgpt-core/src/linalg/tucker.rs` (Plan 326) | **Shipped** |
| Quadrature weights for irregular point clouds (Alg 1) | Implicit in DEC (incidence matrices encode partition volumes) + `papaya`/spatial-hash partitioning in `ShardIndex`/`ItemEmbedIndex` | Covered |
| Fixed-radius neighborhood definition (§3.4) | `katgpt-rs/crates/katgpt-core/src/zone_density.rs` + spatial partitioning in `riir-games` | Covered |
| Positional encoding (coordinate concatenation, §3.7) | Standard pattern across HLA / CGSP / Fourier MCTS (`encode_offset(dx,dy)` in `riir-engine::fourier`) | Covered |
| Pointwise operators (Nemytskii, §3.2) | Trivial — all activations ship as pointwise by construction | Covered |
| Function-space mean/variance (Eq 16) | DEC-weighted aggregations; HLA scalar projections | Covered |
| Spectral commitment crossing sync boundary | `riir-chain/src/encoding/latcal_fixed.rs::LatCalSpectralFixed` (Plan 265) | **Shipped** — `(freq × 10⁶, amp × 10⁶, phase × 10⁶)` fixed-point Fourier coefficients for chain commitment |
| Per-NPC HLA field over a zone grid as FNO input | `apply_field_to_crowd` (Plan 309 latent steering) | **Shipped** — crowd-scale HLA field is a 2D field of 8-ch latents, natural FNO input |

**Notes-layer coverage:** Research 307 (FNO Practical Perspective), 219 (TNO → DEC), 257 (FUNCATTN), 291 (Cross-Resolution Spectral Transport), 303/306 (FUNCATTN predecessors — Transolver, Galerkin). **This paper is the architectural progenitor of all five.**

### 2.3 Latent-space reframing (mandatory §3)

Re-cast the recipe against each Super-GOAT factory module:

- **HLA per-NPC latent state** (`riir-ai/crates/riir-engine/src/hla/`): the paper's "channels" (codomain dim `c`) = HLA's 8 affect channels (valence/arousal/desperation/calm/fear + 3). The encoder-decoder pattern (§3.6) IS what `funcattn` does on HLA — project the 8-ch state onto k direction vectors, apply a `k×k` map, reconstruct. Crowd-scale HLA over a zone grid is the 2D-function-on-domain case the paper's FNO targets. **Already wired.**
- **`latent_functor/`**: the encoder-decoder layer (paper §3.6) is exactly the latent-functor application. `transport_cross_resolution_into` IS the asymmetric-basis functor. **Shipped.**
- **`cgsp_runtime/`**: spectral-band curiosity (which Fourier band is the NPC exploring?) — niche, not wired.
- **LatCal fixed-point commitment** (`riir-chain/src/encoding/`): `LatCalSpectralFixed` already commits Fourier `(freq, amp, phase)` as i64 × 10⁶ — FNO coefficients cross the sync boundary as raw fixed-point scalars. **Shipped.**
- **`NeuronShard` `style_weights[64]`** (`riir-neuron-db/src/shard.rs`): TFNO Tucker factorization of the 8×8 reshaped weight matrix — **shipped as Plan 326** (`linalg/tucker.rs`).
- **DEC Stokes-calculus** (`katgpt-rs/crates/katgpt-dec/src/`): the paper's integral operators are DEC `exterior_derivative`/`codifferential` on a periodic grid; spectral differentiation is `d` in Fourier vocabulary. **Shipped as Plans 251, 325.**

No Super-GOAT reframing emerges — every axis reduces to a shipped primitive.

### 2.4 Closest cousins (fusion candidates)

The three closest existing distillations, ranked:

1. **Research 307** (FNO Practical Perspective) — **strictly more specific** follow-up paper from the same group, distilled 2026-06-25, all three Gain-tier follow-up plans shipped.
2. **Research 291 + Plan 310** (Cross-Resolution Spectral Transport) — the Super-GOAT that emerged from this paper family; **strictly stronger** than the paper's standalone super-resolution.
3. **Research 219 + Plan 251** (TNO → DEC) — the topological-operator instance; ships the integral-operator substrate in DEC vocabulary.

**No fusion idea.** Every combination of (this paper × note A × note B) reduces to something Research 307 already considered and either shipped or routed to riir-train.

---

## 3. Verdict

### **Pass** — progenitor survey paper of already-distilled + already-shipped work; training-side recipe → riir-train.

**One-line reasoning:** This paper (2506.10973) is the architectural progenitor of the NVIDIA/Caltech neural-operator framework. Its more specific follow-up (FNO Practical Perspective, 2512.01421) was already distilled as Research 307 → Gain, with all three Gain-tier follow-up plans (323 Fourier continuation, 325 spectral differentiation, 326 Tucker/HOSVD) shipped. The modelless subset of this paper — quadrature weights, radius-graph neighborhoods, encoder-decoder inner products, pointwise operators — is **fully subsumed** by `cross_resolution_transport` (DEFAULT-ON), `funcattn`, DEC operators, and the recently-shipped `spectral/{continuation,differentiation}` + `linalg/tucker` modules. The training-side content (learning the kernel `K`, basis `Φ`, encoder/decoder) → riir-train per §3.5.

### Routing

- **No Super-GOAT guide** (novelty gate fails Q1: prior art is overwhelming — see §2.2).
- **No katgpt-rs plan opened in this session** — every modelless piece is shipped; no new capability class.
- **Training side → riir-train** (one-line note): the recipe's "parametrize weights as learnable functions" step (kernel `K`, basis `bⱼ`, FNO mode-mixing matrix `R(k)`, DeepONet trunk/branch nets, NOMAD decoder MLP) all require gradient descent. Physics-informed loss, Sobolev loss, multi-resolution curriculum, autoregressive pushforward loss, multi-objective SoftAdapt/ReLoBRaLo balancing — all training procedures.
- **One DRY observation** (not a plan — an issue): the encoder-decoder pattern (paper §3.6, App A.8) explicitly unifies `funcattn`, `cross_resolution_transport`, DEC integral operators, and DeepONet as instances of `inner-product encode → latent linear map → linear-combination decode`. We ship all four instances but have no single trait abstracting them. A `FunctionSpaceEncoderDecoder` trait in `katgpt-rs/crates/katgpt-core/src/` could be a future DRY refactor — **deferred to `.issues/` per the global rule "Create issue at .issues for optimization or refactor task, do not create plan"**. See `katgpt-rs/.issues/395_*` (to be created if user wants the refactor).

### Novelty gate (Q1–Q4)

| Q | Answer | Notes |
|---|--------|-------|
| **Q1 No prior art?** | ❌ NO | Research 307 (sibling paper), 219 (TNO→DEC), 257 (FUNCATTN), 291 (cross-res transport) collectively cover every modelless piece. Plans 251, 286, 310, 323, 325, 326 all shipped. |
| **Q2 New capability class?** | ❌ NO | The paper's contributions are (a) the recipe (training-side), (b) unification of existing neural-operator variants (architectural survey). No new capability. |
| **Q3 Product selling point?** | ❌ NO | Cannot finish "our NPCs do X that no competitor can" with anything from this paper — the relevant X (cross-resolution latent transport, function-space projection, fixed-receptive-field aggregation) already shipped and is already a selling point. |
| **Q4 Force multiplier (≥2 pillars)?** | ❌ NO | No new pillar connection — the existing Fourier Spatial pillar (P4) already absorbs the spectral content. |

**Verdict: 0/4 YES → Pass.** No guide, no plan, no Super-GOAT.

### Why this is Pass (not Gain)

Research 307 was Gain because it identified three narrow *unshipped* gaps (Fourier continuation, spectral differentiation, Tucker). All three have since shipped (Plans 323/325/326). This paper is the *progenitor* — it adds nothing the more specific follow-up didn't already cover. There is no unshipped gap left to plan. The DRY trait refactor opportunity (§3 Routing) is an `.issues/`-class refactor, not a Gain-tier primitive.

---

## 4. Cross-references

- `katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md` — **the sibling paper (same group, more specific), already distilled → Gain, all 3 follow-up plans shipped**
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — TNO → DEC operators (the integral-operator substrate)
- `katgpt-rs/.research/257_Functional_Attention_Spectral_Transport_Operator.md` — FUNCATTN (encoder-decoder with frozen bases — the modelless subset of this paper)
- `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md` — the Super-GOAT that emerged from this paper family
- `katgpt-rs/.research/303_Transolver_Physics_Attention_FUNCATTN_Predecessor.md` + `306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md` — FUNCATTN predecessors
- `katgpt-rs/.research/039_SpectralQuant_Calibrated_Eigenbasis_KV_Compression.md` — SpectralConv applied to KV cache
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` — DEC (the integral-operator substrate, COMPLETE)
- `katgpt-rs/.plans/286_functional_attention_spectral_transport.md` — FUNCATTN open primitive
- `katgpt-rs/.plans/310_cross_resolution_spectral_transport_primitive.md` — Cross-resolution transport (**DEFAULT-ON**)
- `katgpt-rs/.plans/323_fourier_continuation_primitive.md` — Fourier continuation (the Research 307 gap, now closed)
- `katgpt-rs/.plans/325_spectral_differentiation_primitive.md` — Standalone spectral differentiation (**DEFAULT-ON**)
- `katgpt-rs/.plans/326_tucker_hosvd_factorization.md` — Tucker/HOSVD tensor factorization
- `riir-chain/.research/004_LatCal_Committed_Karc_Readout.md` + `riir-chain/src/encoding/latcal_fixed.rs::LatCalSpectralFixed` — fixed-point Fourier commitment across the sync boundary
- → **riir-train** for the entire training-side recipe: kernel `K` learning, basis `bⱼ` learning, FNO mode-mixing `R(k)` learning, DeepONet trunk/branch training, NOMAD decoder MLP training, physics-informed loss, Sobolev loss, multi-resolution curriculum, autoregressive pushforward loss, multi-objective SoftAdapt/ReLoBRaLo loss balancing.

## TL;DR

This paper (2506.10973, Berner/Liu-Schiaffini/Kossaifi/Anandkumar) is the **progenitor recipe paper** for the NVIDIA/Caltech neural-operator framework — it unifies FNO/GNO/SFNO/UNO/DeepONet/Transformer-NO as instances of "parametrize NN weights as learnable functions, aggregate with quadrature weights, fix the receptive field w.r.t. the underlying domain". Research 307 already distilled the more specific sibling paper (FNO Practical Perspective, 2512.01421) → Gain, and all three of its Gain-tier follow-up plans (323 Fourier continuation, 325 spectral differentiation, 326 Tucker/HOSVD) **shipped**. The modelless subset of this paper is **fully subsumed** by `cross_resolution_transport` (DEFAULT-ON), `funcattn`, DEC `exterior_derivative`, `spectral/{continuation,differentiation}`, and `linalg/tucker`. The training-side recipe (learning the kernel `K`, basis `Φ`, encoder/decoder) → **riir-train** per §3.5. **Verdict: Pass** — 0/4 on the novelty gate; no new primitive, no new capability class, no Super-GOAT, no plan opened. One DRY observation (unified `FunctionSpaceEncoderDecoder` trait abstracting `funcattn` × `cross_resolution_transport` × DEC × DeepONet-style) is deferred to `.issues/` per the global rule on refactor-class work.
