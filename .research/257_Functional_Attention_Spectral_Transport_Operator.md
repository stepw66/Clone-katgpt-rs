# Research 257: Functional Attention — Spectral Transport Operator

> **Source:** "Functional Attention: From Pairwise Affinities to Functional Correspondences" (Xiao, Gao, Weber, Yang, Cremers — ICML 2026, PMLR 306)
> **arxiv:** [2605.31559](https://arxiv.org/pdf/2605.31559) · **code:** https://github.com/xjffff/FUNCATTN
> **Date:** 2026-06-17
> **Status:** Active
> **Related Research:** 135 (Parallax — closest shipped cousin), 077 (SpectralQuant — eigenbasis toolkit), 057 (HLA), 237 (CHIAR operator routing)
> **Related Plans:** 286 (open primitive, Gain), riir-ai 318 (rank-k latent_functor upgrade, primary value)
> **Cross-ref (riir-ai):** Plan 303 (latent_functor rank-1 — what this upgrades to rank-k), Research 123 (Latent Functor Runtime Guide — Super-GOAT this extends)
> **Classification:** Public

---

## TL;DR

FUNCATTN reinterprets attention not as softmax over pairwise token affinities but as a **closed-form Tikhonov-regularized k×k linear operator C between learned adaptive bases Φ (query side) and Ψ (key-value side)** — directly imported from the functional-maps framework in geometry processing. The operator is solved in closed form:

$$C^* = \tilde Q \tilde K^\top (\tilde K \tilde K^\top + \lambda I_k)^{-1}, \quad \text{where } \tilde Q = \Phi^\top Q,\ \tilde K = \Psi^\top K,\ \tilde V = \Psi^\top V$$

$$\text{FUNCATTN}(Q,K,V) = \Phi \, C^* \, \tilde V$$

Complexity is **linear in n** (`O(ndk + dk·min(k,d) + min(k,d)³)`, k≪n), the operator is **resolution-invariant** (train at n=2048, test at n=8192), and **Lipschitz continuity is bounded by λ** (Prop 4.5: ‖∂A‖ ≤ (C₁/λ + C₂/λ²)·‖ΔX‖). The paper shows SOTA on 6 PDE benchmarks + RNA segmentation + OOD AirfRANS, beating Transolver 6–26%.

**Distilled for katgpt-rs (modelless, inference-time):** the *closed-form Tikhonov solve in spectral space* is a fully inference-time primitive — given pre-trained basis matrices Φ, Ψ (small `d→k` projections), the attention output is pure linear algebra (one small Cholesky/Schur solve + three matmuls). No gradient, no in-place weight mutation. The mechanism **recovers Intention (Garnelo & Czarnecki 2023) as a special case** when Φ=Ψ=orthonormal full basis. It also **recovers the Schur-complement least-squares solver already shipped in `riir-gpu/schur.rs`** (Plan 067) — meaning the math is already in our stack, just framed as a *training* primitive rather than as an *attention operator*.

**Verdict: GOAT.** Reasoning below (§3).

---

## 1. Paper Core Findings

### 1.1 The reinterpretation

Standard scaled dot-product attention treats `Attention(Q,K,V) = Softmax(QK^T/√d_k) V` as a **pointwise affinity matrix** mapping `n` value tokens to `n` output tokens — O(n²) in sequence length, with no principled discretization invariance.

FUNCATTN observes: *the attention matrix is just the matrix representation of a linear operator between function spaces*. If we equip query space `F(X)` and key-value space `F(Y)` with **k-dimensional adaptive bases** Φ ∈ R^{n×k}, Ψ ∈ R^{n×k}, then the linear operator T admits a k×k matrix C — and recovering it is a **convex problem** (regularized least-squares, closed form).

This is the **functional maps** framework from geometry processing (Ovsjanikov 2012): instead of seeking point-to-point correspondences between 3D shapes (combinatorial, O(n²)), one seeks a small k×k operator between truncated Laplace-Beltrami eigenbases (k≪n).

### 1.2 The mechanism

**Basis** (Eq. 9): learned, input-adaptive, soft piecewise-constant (P0):

$$\Phi = \text{Softmax}(\text{Linear}_\Phi(X)) \in \mathbb{R}^{n \times k}, \quad \Psi = \text{Softmax}(\text{Linear}_\Psi(X)) \in \mathbb{R}^{n \times k}$$

Prop 4.3: this is a *generalization of P0 elements* — partition-of-unity for all τ > 0; as τ→0, recovers hard piecewise-constant basis `1_{Λ_j(x)}` where `Λ_j = {x : s_j(x) > s_l(x) ∀ l≠j}`.

**Spectral projection** (Remark 4.1): they use `Φ^T` rather than the Moore-Penrose pseudo-inverse `Φ^† = (Φ^TΦ)^{-1}Φ^T` because the latter destabilizes training and inflates the inverted matrix's condition number by >10× (Fig 6). The two coincide when Φ is orthonormal; in general, `Φ^T Q` returns the inner products `⟨Φ_{:,j}, Q⟩`, a legitimate function-space representation.

**Operator C** (Eq. 6–7): Tikhon-regularized least-squares in spectral space:

$$\min_C \|\tilde Q - C\tilde K\|_F^2 + \lambda\|C\|_F^2 \quad\Longrightarrow\quad C^* = \tilde Q \tilde K^\top (\tilde K \tilde K^\top + \lambda I_k)^{-1}$$

Woodbury identity gives the dual form `C^* = \tilde Q (\tilde K^\top \tilde K + \lambda I_d)^{-1} \tilde K^\top` — pick whichever of k or d is smaller.

**Full attention** (Eq. 8): `FUNCATTN(Q,K,V) = Φ · C^* · Ṽ` where `Ṽ = Ψ^T V`.

### 1.3 Continuity, complexity, special cases

**Prop 4.5 (Local Lipschitz):** for input X with ‖X‖≤B, `‖∂A‖_F ≤ (C_1/λ + C_2/λ²)·‖ΔX‖_F` where C₁,C₂ are polynomial in B,n,W*. **λ controls the Lipschitz constant** — formalizing the role of the Tikhonov term as a numerical safeguard, not a hyperparameter (sensitivity ablation Tab 13: test error varies <0.02 across 10× change in α_init).

**Complexity:** `O(ndk + dk·min(k,d) + min(k,d)³)` — linear in n, cubic in k (typically k=64, robust default; 32 for smooth fields, 128–256 for high-frequency). The paper benchmarks linear scaling in n vs softmax-quadratic at n=2¹⁴ (Fig 5).

**Prop A.4 (Intention as special case):** with Φ=Ψ=orthonormal full basis `I_n`, FUNCATTN reduces to `Q(K^TK+λI_d)^{-1}K^TV` = Intention (Garnelo & Czarnecki 2023). FUNCATTN strictly generalizes by learning **a non-orthonormal adaptive basis**.

**Theorem A.3:** FUNCATTN is a Monte-Carlo discretization of a regularized integral operator with kernel `κ(g_i,g_j) = (Φ C Ψ^T)_{ij}` — i.e., a **learnable integral neural operator**.

### 1.4 Empirical results (selected)

| Benchmark | Transolver | **FUNCATTN** | Δ |
|---|---|---|---|
| Elasticity (rel L2 ×100) | 0.64 | **0.50** | -22% |
| Darcy | 0.57 | **0.42** | -26% |
| Airfoil | 0.53 | **0.43** | -19% |
| Navier-Stokes | 9.44 | **8.00** | -15% |
| AirfRANS OOD Reynolds (CL err %) | 32.2 | **23.4** | -27% |
| RNA 3D segmentation acc | 87.5% | **89.0%** | +1.5pp |
| Burgers super-res 2048→8192 | 1.243 | **1.081** | -13% |

Ablations: k=64 robust default within 5% of best across all benchmarks (Tab 6); learnable basis (no orthogonality constraint) wins (Tab 7); transpose projection > pseudo-inverse (Tab 12).

### 1.5 What the paper does NOT claim

- **NLP unverified.** §6 future work: *"investigating functional attention in domains with less direct function-space interpretations, such as natural language processing, remains a promising future task."* The functional-map inductive bias is strongest where the underlying signal has low intrinsic complexity relative to its discretization — exactly PDE solution fields, point-cloud geometry, regression. **Token sequences may or may not have this property.**
- **Training is required** for basis matrices W_Φ, W_Ψ (standard transformer training). The closed-form C solve is inference-time; the basis is not.

---

## 2. Distillation

### 2.1 Transferable primitive

The pure-inference kernel is **three matmuls + one small linear solve**:

```text
// Inference-time, given trained W_Φ, W_Ψ ∈ R^{d×k}:
Φ ← softmax_rows(X · W_Φ)              // R^{n×k}, partition-of-unity rows
Ψ ← softmax_rows(X · W_Ψ)              // R^{n×k}
Q̃ ← Φᵀ · Q                              // R^{k×d}  (NOT Φᵀ⁺ — transpose, see Rem 4.1)
K̃ ← Ψᵀ · K                              // R^{k×d}
Ṽ ← Ψᵀ · V                              // R^{k×d}
C ← Q̃ · K̃ᵀ · (K̃ · K̃ᵀ + λ I_k)⁻¹         // R^{k×k}  closed-form ridge solve
                                          //         (Woodbury dual when d<k)
out ← Φ · C · Ṽ                          // R^{n×d}
```

All steps are matmuls or a single k×k Cholesky/Schur solve (k=64 typically). No autograd needed at inference. **Zero-allocation hot path**: Φ, Ψ, Q̃, K̃, Ṽ, C reuse pre-allocated scratch; `Φ · C · Ṽ` is two FMA matmuls; the solve is `O(k³)` ≈ 256K flops at k=64.

### 2.2 Where the pieces already live in our stack

| Paper piece | Already shipped? | Where | Notes |
|---|---|---|---|
| Closed-form ridge solve `M(M^TM+λI)^{-1}` | ✅ | `riir-ai/crates/riir-gpu/src/schur.rs` (Plan 067, riir-train) | SchurSolver::solve_unconstrained solves `Qz*=-p` with `Q=X^TX+λI` via Cholesky. Same math, framed as training primitive. |
| Eigenbasis / spectral basis | ✅ | `katgpt-rs/src/spectralquant/spectral.rs` (Plan 077) | `calibrate_eigenbasis` from sample covariance. SpectralQuant's per-dim eigenbasis rotation IS the "fixed basis" ablation row in Tab 7. |
| Linear attention + sigmoid basis | ✅ | `katgpt-rs/crates/katgpt-core/src/parallax_attn.rs` (Plan 135) | `ParallaxActivation::Sigmoid` is the default — partition-of-unity kernel `K(x,y)=σ(x·y·s)`. Different operator (NW correction), same sigmoid-partition-of-unity idea. |
| Streaming second-order state | ✅ | `katgpt-rs/src/hla/` (Plan 057) | O(1) outer-product accumulator. Different math (no closed-form solve) but solves the same problem (linear attention with bounded state). |
| Latent operator between spaces | ✅ (rank-1) | `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs` (Plan 303) | `extract_functor`: `f = mean_k(target_k - source_k)`, coherence `mean_k cos(...)`. **This is the rank-1 special case of FUNCATTN's k×k operator C.** Apply via `out = source + f` (additive). |
| Per-NPC recurrent belief kernel | ✅ | `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` (`evolve_hla`) | No research note framing it as such — per the workflow's canonical failure mode. |
| Freeze/thaw snapshot of direction vectors | ✅ | `riir-ai/crates/riir-engine/src/latent_functor/table.rs` | `FunctorEntry { direction, coherence, version: Uuid::now_v7(), commitment: [u8;32] }`. Atomic Arc-swap. Versioned. BLAKE3-committed. |

### 2.3 Closest cousins (3)

1. **Plan 135 / Research 140 — Parallax Sigmoid Attention.** The closest *attention* cousin. Difference: Parallax corrects softmax/sigmoid attention via `o_LL = o_NW - Σ_KV·ρ` (Nadaraya-Watson local-linear upgrade), while FUNCATTN replaces attention *with* the linear solve itself. Parallax is a *correction on top of* an attention kernel; FUNCATTN *is* the kernel. Both use sigmoid-based partition-of-unity (Parallax by default per Plan 161, FUNCATTN via softmax-with-τ that → P0 as τ→0).

2. **Plan 303 / Research 123 — Latent Functor Runtime.** The closest *runtime* cousin. `extract_functor` is exactly `C ← Q̃K̃ᵀ(K̃K̃ᵀ+λI)^{-1}` with **k=1, λ=0, Φ=I, Ψ=I** — i.e., the rank-1, unregularized, basis-free special case. FUNCATTN is the rank-k, Tikhonov-regularized, basis-projected generalization. This is the **primary fusion target** (§2.4).

3. **Plan 077 — SpectralQuant eigenbasis.** The closest *basis-toolkit* cousin. `calibrate_eigenbasis` (PCA from sample covariance, eigendecomposition) is the *fixed* basis ablation in Tab 7 (FUNCATTN with Fourier basis). The paper's *learnable* basis (Eq. 9) is a runtime-computed, input-adaptive generalization of SpectralQuant's offline-calibrated rotation.

### 2.4 Fusion

#### Fusion F1 (PRIMARY — riir-ai): Latent Functor rank-1 → rank-k

**The combination:** `latent_functor/arithmetic.rs` × FUNCATTN × SchurSolver × `latent_functor/reestimation.rs` (Plan 303's coherence-driven scheduler).

Today `extract_functor` learns `f = mean_k(target_k - source_k)` — a single displacement vector per (NPC, relation). This captures only **monotonic translational** relations: "if A fears B, A's embedding shifts by f". It cannot represent **rotational / multi-axis** relations like "A's fear of B is high-arousal-low-valence but A's admiration of C is low-arousal-high-valence" — those bend multiple semantic axes simultaneously and need a k×k operator, not a single direction.

**The upgrade:** generalize `FunctorEntry.direction: Vec<f32>` (length `dim`) to `FunctorEntry.operator: MatMxK` (k×k in spectral coords), plus `basis_source: MatDxK`, `basis_target: MatDxK`. Apply via `out = Φ_target · C · Ψ_source^T · source` instead of `out = source + f`. Extraction is the closed-form Tikhonov solve ( reuse SchurSolver). Re-estimation coherence check generalizes from `mean cos(displacement, f)` to **Frobenius residual** `‖Q̃ - C·K̃‖_F / ‖Q̃‖_F` — same scheduler, different quality metric.

**Why this is novel:** nobody has shipped a rank-k relational operator between latent spaces with closed-form Tikhonov re-estimation for game NPCs. The combination unifies three pieces already in our stack (SchurSolver math + latent_functor scaffolding + freeze/thaw commitment) into a single new capability: NPCs learn **non-translational relational transformations** ("fear" as a rank-k rotation in latent space, not a single shift).

**Capability unblocked:** multi-axis NPC relations. Betrayal is not "shift embedding by f" — it's "rotate embedding through operator C that bends valence, arousal, desperation simultaneously". This is the difference between a linear-affine NPC mind and a truly **manifold-aware** NPC mind.

#### Fusion F2 (SECONDARY — katgpt-rs open primitive): Functional Attention as a new attention operator

**The combination:** FUNCATTN × sigmoid basis (per AGENTS.md "never softmax") × SchurSolver × freeze/thaw-versioned basis snapshots.

The paper uses softmax for its basis (Eq. 9). AGENTS.md mandates sigmoid. **The fusion hypothesis:** replace `Softmax(Linear(X))` with `Sigmoid(Linear(X))` normalized to partition-of-unity (same trick Parallax uses for sigmoid attention). The partition-of-unity property (Prop 4.3) holds for any row-normalized non-negative kernel; sigmoid-normalized is a valid P0 generalization. Closed-form C solve is unchanged.

**Why fuse with freeze/thaw:** the basis matrices W_Φ, W_Ψ (small `d×k`) are perfect freeze/thaw candidates — atomic Arc-swap, BLAKE3-committed, per-entity personality divergence via snapshot versioning. Different NPCs / different domains get different basis snapshots, hot-swapped at runtime. This is the **private-runtime extension** of the open primitive.

**Capability unblocked:** a new attention class in katgpt-rs that is *closed-form linear-algebra* (not iterative, not softmax-based, not outer-product accumulator). Different inductive bias than all 6 shipped attention variants (SDPA, HLA/AHLA, Parallax, GDN2, DashAttn, Lighthouse). Composes with SpectralQuant (eigenbasis pre-rotation of basis) and CHIAR (operator-level routing between FUNCATTN and Parallax by spectral entropy).

#### Fusion F3 (TERTIARY — speculative): Functional Attention × Latent Functor × Dirichlet Energy

**Speculative combination:** use FUNCATTN's C operator as the **transport map** that role_transport.rs (Plan 151) currently approximates with diagonal/orthogonal RoPE extensions. Dirichlet Energy (Plan 149, Research 111) measures alignment quality — it generalizes from "is f a good rank-1 fit?" to "is C a good rank-k fit?" via `E(C) = Σ_ij A_ij ‖h_i - C h_j‖²`. This would let NPCs learn **operator-valued role transport** (rank-k rotation between role embeddings), not just diagonal gates. Out of scope for the immediate plan; flagged for follow-up if F1 ships and proves the operator-valued relational primitive is useful.

---

## 3. Verdict

### Tier: **GOAT**

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **Partial.** Math pieces all shipped (Schur ridge solve, SpectralQuant eigenbasis, Parallax sigmoid partition-of-unity, latent_functor rank-1 operator). The *combination* as an attention operator + as a rank-k functor upgrade is novel. | Vocabulary translation was essential: paper's "functional maps" ↔ codebase's "latent functor"; paper's "Tikhonov operator" ↔ codebase's "Schur complement solver". |
| Q2 New class of behavior? | **Yes for riir-ai** (rank-k relational operator — NPCs model non-translational relations). **No for katgpt-rs** (just another attention variant in an already-crowded field; paper itself hasn't shown NLP gain). | |
| Q3 Product selling point? | **Yes for riir-ai** ("NPCs learn rank-k relational operators between latent spaces, re-estimated at runtime via closed-form ridge regression"). **No for katgpt-rs** (public paper, anyone can implement). | |
| Q4 Force multiplier? | **Yes** — connects latent_functor + Schur + SpectralQuant + Parallax + freeze/thaw + Dirichlet Energy. ≥3 pillars. | |

**Not Super-GOAT** because: (a) the math pieces are all already in our stack — this is a *combination*, not a new primitive; (b) the primary selling point (rank-k functor) is an **upgrade** of the existing latent_functor Super-GOAT (Research 123 / Plan 303), not a new pillar; (c) the open katgpt-rs primitive's NLP value is unverified by the paper itself.

**Why GOAT and not Gain:** the riir-ai side has a concrete, measurable, in-domain gain (rank-k functor coherence > rank-1 on multi-axis relations), backed by an existing benchmark harness (Bench 263 T4.1 betrayal-prediction GOAT already proves rank-1 works at cos=0.9999 — the rank-k extension has a clear target). The katgpt-rs open primitive is Gain-tier (plan only, behind feature flag, await LLM-token-prediction GOAT proof).

**One-line verdict reasoning:** FUNCATTN's transferable primitive — closed-form Tikhonov k×k operator between learned sigmoid-bases — upgrades our rank-1 latent_functor to rank-k (riir-ai primary value, GOAT) and adds a new closed-form attention class to katgpt-rs (Gain-tier open primitive behind feature flag). Not Super-GOAT because the math is already distributed across Schur/SpectralQuant/latent_functor and the rank-k upgrade extends an existing Super-GOAT rather than creating a new pillar.

### Routing

- **riir-ai/.plans/318_latent_functor_rank_k_upgrade.md** — primary plan, private runtime. P0: extend `FunctorEntry` to operator-valued, swap mean-displacement extraction with Tikhonov ridge solve, generalize coherence metric. GOAT gate on multi-axis relation benchmarks.
- **katgpt-rs/.plans/286_functional_attention_spectral_transport.md** — open primitive, Gain-tier. Feature flag `funcattn`. Skeleton + sigmoid-basis variant + GOAT proof on token-prediction proxy (random-token HLA-distillation style per Plan 059). Promote to opt-in only if it beats Parallax; do NOT promote to default until LLM-domain evidence exists.
- **No riir-ai Super-GOAT guide required** (verdict is GOAT, not Super-GOAT).
- **No katgpt-rs implementation this session** — plan only, await prioritization.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ C solve is closed-form given trained W_Φ,W_Ψ. No backprop. |
| Latent-to-latent preferred | ✅ Operates entirely in spectral/latent space; Φ,Ψ project raw → latent; out is raw reconstruction. |
| Use sigmoid not softmax | ⚠️ Paper uses softmax (Eq. 9). **Fusion F2 mandates sigmoid-normalized basis** to comply with AGENTS.md. Partition-of-unity property (Prop 4.3) holds for any row-normalized non-negative kernel. |
| Freeze/thaw over fine-tuning | ✅ W_Φ,W_Ψ (and the rank-k C in F1) are perfect freeze/thaw candidates — small matrices, atomic Arc-swap, BLAKE3-committed. |
| 4-repo discipline | ✅ Open primitive → katgpt-rs; rank-k game runtime → riir-ai; no chain IP; no training know-how leaks. |
| Raw scalars at sync boundary | ✅ In F1, the *operator* stays local; only its scalar outputs (valence/arousal/desperation/calm/fear projections of `out`) cross sync. Same boundary discipline as rank-1 latent_functor. |
| Zero-alloc hot path | ✅ All matmuls + one small k×k solve. `Vec::with_capacity` once, `clear()`+reuse across calls per optimization.md. |

---

## 5. Open questions / risks

1. **Does the sigmoid basis preserve FUNCATTN's accuracy?** Paper ablates softmax-vs-orthogonal-vs-free in Tab 7 but not sigmoid. The partition-of-unity proof (Prop 4.3) is basis-agnostic, but the P0 limit (τ→0) is softmax-specific. **Mitigation:** sigmoid-normalized-rows still satisfy partition-of-unity; the τ-anneal becomes a β-anneal on the sigmoid slope. Needs empirical check in Plan 286 G3.

2. **Does FUNCATTN help LLM token prediction at all?** Paper §6 explicitly defers NLP. **Risk:** we ship the open primitive, run GOAT gate, find no gain over Parallax/SDPA on real LM data, demote. This is the expected outcome for the katgpt-rs side. **The riir-ai side does not depend on this** — game-runtime functor upgrade has its own benchmark (multi-axis NPC relations).

3. **Rank-k functor vs rank-1: when does k>1 actually matter?** Rank-1 already achieves cos=0.9999 on the T4.1 betrayal benchmark (Bench 263). The hypothesis is that **multi-axis** relations (fear + admiration + curiosity simultaneously) break rank-1. **Validation:** construct a synthetic multi-axis benchmark (e.g., NPC observes a target that simultaneously inspires fear in valence-axis AND admiration in dominance-axis — rank-1 displacement cannot fit, rank-k can). This is G2 in Plan 318.

4. **Numerical stability of `Φ^T` vs `Φ^†`.** Paper uses transpose (Remark 4.1) — unregularized pseudo-inverse diverges. We must do the same. SchurSolver's `eps_reg=1e-8` is the right stabilization. Document in plan.

5. **k selection.** Paper's k=64 default is for PDE meshes with n~10⁴. For NPC latent dim ~64 and observation sets ~20, k should be much smaller — likely k=4 to k=16. **FILLED 2026-06-26 by Plan 332 k-sweep** (`.benchmarks/332_structured_basis_goat_and_k_sweep.md`): swept k ∈ {4, 8, 16, 32} with four basis variants (random-orthogonal, hand-crafted, DCT-log, Haar-packet) on multi-scale transport at d=64, n=20, τ=0.5. **Elbow at k=16** — principled bases (Haar-packet) beat random by +0.08 at k∈{4,8} (the NPC regime) but lose to random at k≥16 (rank saturation: random-orthogonal at k=16 has enough rank to approximate any direction in d=64, so the structural advantage of a fixed basis evaporates). Practical guidance: k∈{4,8} with Haar-packet for localized transport tasks; random-orthogonal fine at k≥16. The strict GOAT gate FAILS (DCT-log loses on this probe signal due to frequency mismatch, Haar loses at sharp τ=0.1) so `funcattn_structured_basis` stays opt-in — see Plan 332 verdict.

---

## TL;DR

FUNCATTN is the **functional-maps framework applied to attention**: replace softmax pairwise affinities with a closed-form ridge-regularized k×k operator between learned adaptive bases. Linear-in-n, resolution-invariant, Lipschitz-bounded by λ. The math (ridge solve, eigenbasis, sigmoid partition-of-unity) is already distributed across our stack (Schur/SpectralQuant/Parallax/latent_functor). **GOAT verdict**: primary value is the riir-ai rank-1 → rank-k latent_functor upgrade (concrete game-domain gain, extends existing Super-GOAT 123/303); katgpt-rs open primitive is Gain-tier behind feature flag (paper itself hasn't shown NLP gain). Not Super-GOAT because (a) no novel math, (b) extends an existing pillar rather than creating one. **Fusion F1 is the headline** — rank-k functor with closed-form ridge re-estimation unblocks multi-axis NPC relations, the difference between linear-affine and manifold-aware NPC minds.

**Issue 363 Update (2026-07-02):** Fusion F1's rank-k operator has been extended to **n-ary coalitions** via `HyperKgFunctorEdge` (`riir-ai/crates/riir-engine/src/kg_hyperedge.rs` Phase 5). The operator maps the mean-pool of participant states to coalition-level state, generalizing from pairwise `(A→B)` to coalition `({A,B,C,...}→state)`. This is the functional edge × hyperedge fusion that the original F1 framing gestured at ("non-translational relations") but didn't wire for n-ary. O(N) coalition prediction vs O(N²) pairwise. See Research 123 §Issue 363 Update for the full gain table.

---

## 6. Code Verification Addendum (2026-06-17, post-distillation)

Reference implementation reviewed at `.raw/FUNCATTN/` (official code from xjffff/FUNCATTN). **Three material discrepancies between paper text and actual shipped code** — Plans 286 and 318 must follow the CODE, not the paper formulas, where they differ.

### Discrepancy 1: Regularization form — convex combo, NOT additive Tikhonov

- **Paper Eq. 7**: `C* = Q̃K̃ᵀ(K̃K̃ᵀ + λI_k)⁻¹` (additive λ on k×k primal)
- **Code** (`PDE-StandardBenchmark/model/Functional_attention.py` L73-76, `Few-Shot-Regression/models.py` L172-175):
  ```python
  alpha = self.sigmoid(self.alpha)                    # α ∈ (0,1), learnable
  reg_dual = (1 - alpha) * kTk + alpha * self.I_d      # (1-α)·K̃ᵀK̃ + α·I_d (d×d DUAL)
  Z = torch.linalg.solve(reg_dual, kH)                # solves reg_dual · Z = K̃ᵀ
  C = torch.matmul(q_slice_token, Z)                  # C = Q̃ · Z
  ```

**The actual regularization is a convex combination** `(1-α)·K̃ᵀK̃ + α·I_d` where `α = sigmoid(parameter)` ∈ (0,1), in the **dual form** (d×d, not k×k). This is strictly better-conditioned than additive λ:
- Eigenvalues of `(1-α)·K̃ᵀK̃ + α·I` lie in `[α, α + (1-α)·λ_max(K̃ᵀK̃)]` — bounded spectrum.
- Additive `K̃ᵀK̃ + λI` has unbounded λ_max — needs careful λ tuning.
- The convex combo guarantees well-posedness for any α ∈ (0,1) — this is why the paper reports λ insensitivity (Tab 13: <0.02 test error variation across 10× α_init change).

**Implementation implication for Plans 286/318**: use `reg = (1-α)·M + α·I` with `α = sigmoid(learnable_param)` or `α = fixed_const ∈ (0.01, 0.5)`. SchurSolver's `eps_reg` parameter already does additive — we need a sibling `solve_convex_combo(M, alpha)` or wrap it.

### Discrepancy 2: Learnable temperature on basis softmax

- **Paper Eq. 9**: `Φ = Softmax(Linear(X))` (no temperature mentioned in main text)
- **Code** (`Functional_attention.py` L13, L60-61):
  ```python
  self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)  # learnable τ
  slice_weights = self.softmax(self.in_project_basis(x_mid) / torch.clamp(self.temperature, min=0.1, max=5.0))
  ```

The temperature IS the τ from Prop 4.3 (P0 limit as τ→0). It's **learnable per-head**, clamped to [0.1, 5.0]. My note §2.2 mentioned this in passing but Plans missed it.

**Implementation implication**: `FuncAttnConfig` must carry `temperature: f32` (or per-head vector) clamped to [0.1, 5.0]. For our sigmoid-basis variant, this becomes a sigmoid-slope β.

### Discrepancy 3: Two input projections (Φ ≠ Ψ)

- **Paper §4**: treats Φ and Ψ as if potentially the same matrix (Remark 4.1 discusses using transpose for both)
- **Code** (`Functional_attention.py` L17-24, L55-58):
  ```python
  self.in_project_x = nn.Conv2d(embed_dim, embed_dim, ...)   # for basis weights
  self.in_project_fx = nn.Conv2d(embed_dim, embed_dim, ...)  # for slice tokens (values being projected)
  # basis weights from x_mid, slice tokens from fx_mid — DIFFERENT projections
  slice_weights = softmax(in_project_basis(x_mid) / temp)
  slice_token = einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)  # fx_mid, not x_mid
  ```

So Φ uses one projection (`in_project_x` → `in_project_basis`), Ψ uses a different one. They are NOT the same matrix in general. Plus the basis projection itself uses **orthogonal weight initialization** (`torch.nn.init.orthogonal_`).

**Implementation implication**: Plan 286 T1.2's `FuncAttnConfig` needs separate `w_phi` and `w_psi` weight matrices (already planned) PLUS a separate `w_value_proj` for the value-side input projection. Orthogonal init for the basis projection is important for training stability — document.

### Bonus: Intention uses a DIFFERENT regularization form

**Intention** (`models.py` L66-71) uses the **N×N primal additive** form:
```python
KKT = torch.bmm(K, Kt)              # N×N
reg = KKT + (self.ridge + 1e-5) * I_N  # ADDITIVE ridge
```

While FUNCATTN uses **d×d dual convex combo**. The paper's Prop A.4 ("FUNCATTN reduces to Intention when Φ=Ψ=I_n") is about the *operator form* — it does NOT mean the regularization forms match. A faithful Intention baseline must use additive ridge on N×N; a faithful FUNCATTN must use convex combo on d×d.

### Summary of code-vs-paper deltas

| Aspect | Paper text | Actual code | Plan impact |
|---|---|---|---|
| Regularization | `K̃K̃ᵀ + λI` (additive, k×k primal) | `(1-α)·K̃ᵀK̃ + α·I` (convex combo, d×d dual) | Plans 286 T1.4, 318 T2.2 must use convex combo |
| Basis temperature | not in main text | learnable τ∈[0.1,5.0] per head | Plan 286 T1.2 must add temperature field |
| Φ vs Ψ | "possibly same" | different projections, orthogonal init | Plan 286 T1.4 already has w_phi/w_psi split — add value-side w_fx |
| Intention regularization | "reduces to" | actually additive N×N primal | Note for Plan 286 G2 baseline reproduction |

**Net effect on verdict:** unchanged. GOAT, primary value still Plan 318. But implementation correctness now depends on matching the code's convex-combo regularization — getting this wrong would silently produce an unstable solve and a false negative on the GOAT gate.
