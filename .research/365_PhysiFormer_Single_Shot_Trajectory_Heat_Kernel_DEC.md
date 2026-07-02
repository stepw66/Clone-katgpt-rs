# Research 365: PhysiFormer → Single-Shot Trajectory Prediction via DEC Heat Kernel

> **Source:** Yiming Chen, Yushi Lan, Andrea Vedaldi, *PhysiFormer: Learning to Simulate Mechanics in World Space* — [arXiv:2606.27364](https://arxiv.org/abs/2606.27364) (VGG Oxford, 25 Jun 2026). Code: [github.com/yimingc9/PhysiFormer](https://github.com/yimingc9/PhysiFormer).
> **Date:** 2026-07-02
> **Status:** Done — verdict locked (**GOAT** — provable gain, modelless DEC analog identified)
> **Classification:** Public (this note). Training recipe (JiT DiT-L, diffusion loss) → riir-train.
> **Related Research:** 359 (Motor-Gated DEC World Model — the step-by-step version this replaces for long horizons), 208 (SLoD — heat kernel on KG graph Laplacian, the precedent for heat kernel ops), 288 (KARC — delay-basis trajectory forecasting, step-by-step rollout), 360 (AdaJEPA — same "world model adaptation" domain, PASS), 358 (SMWM — same domain, PASS), 219 (TNO → DEC substrate), 296 (Stokes vocabulary crosswalk)
> **Related Plans:** 251 (DEC operators), 357 (Motor-Gated DEC Field — the step-by-step primitive), 235 (SLoD heat kernel on KG), 308 (KARC forecaster), 332 (KARC runtime), 341 (Sleep-Time anticipation)

---

## TL;DR

PhysiFormer is a trained diffusion transformer (DiT-L) that predicts full 3D mesh vertex trajectories in a **single denoising pass** directly in world coordinates, conditioned only on initial positions + velocities + material type. Its headline empirical result: single-shot joint trajectory diffusion **beats autoregressive baselines by 10–100×** on rigidity preservation and momentum consistency at 49-frame horizons, because AR suffers irreducible error accumulation that gradually deforms rigid objects. The paper explicitly argues against learned latent autoencoders (JiT principle: "let denoising models denoise" in raw coordinate space) and against hand-coded inductive biases (rigidity, causality).

**The distilled principle (mechanism-agnostic, the actual contribution):**

> *Predicting the entire trajectory jointly from the initial state in a single operation avoids the compounding error of step-by-step (autoregressive) rollout — for any structured prediction domain where per-step errors are not exactly self-correcting.*

This is NOT about diffusion, NOT about DiT, NOT about 3D meshes. It is a prediction-strategy principle: **single-shot joint prediction > step-by-step rollout** when the prediction target has long-range structure that per-step approximation errors corrupt. The paper's evidence (rigidity loss 100× lower at 49 frames, momentum drift ratio 1.91 vs 11.9) is the empirical signature; the principle holds for any mechanism.

**Distilled for katgpt-rs (modelless, inference-time):**

Our DEC substrate ships `evolve_motor_gated_field` (Plan 357) which advances a cochain field by ONE timestep per call — the **autoregressive** approach. The single-shot analog is the **heat kernel / operator exponential** `h(t) = exp(t·A)·h₀` where `A = -I + Δ + diag(motor)` is the DEC propagation operator. For the **linear** case (no ReLU gate), `exp(t·A)·h₀` is the **exact** trajectory — zero error accumulation, exact Hodge-decomposition preservation — while step-by-step Euler `(I + dt·A)^T·h₀` accumulates `O(T·dt²)` global error and slowly mixes the exact/coexact/harmonic split. For the **nonlinear** case (with ReLU), a Krylov-subspace exponential integrator gives higher-order accuracy per matrix-vector product than Euler.

We already ship `heat_kernel_weights` in `slod.rs` (Research 208 / Plan 235) — but only for **KG graph Laplacians** (Poincaré-ball kNN), NOT for **DEC cochain fields**. The DEC heat kernel trajectory prediction primitive does NOT ship. This note distills the principle, the modelless analog, and the fusion with our existing substrate.

---

## 1. Paper Core Findings

### 1.1 The single-shot vs autoregressive comparison (the load-bearing result)

PhysiFormer trains a DiT-L on 10k–60k synthetic 3D mesh physics trajectories (rigid + elastic objects, collisions, Genesis simulator). At inference, it generates the full 49-frame vertex trajectory in one denoising pass (50 Heun steps). It compares against three AR baselines: ΦAR (custom transformer, context 1 or 4 frames, with/without train-time noise injection) and TIE (Transformer with Implicit Edges, r=0.4 and r=1.0).

| Method | MSE (49f) | Rigidity Loss (49f) | Momentum Drift (49f) |
|---|---|---|---|
| **PhysiFormer** | **9.55e-3** | **1.85e-4** | **1.91** |
| ΦAR_ctx1 | 217e-3 | 143e-4 | 11.9 |
| ΦAR_ctx4 | 101e-3 | 27.6e-4 | 8.37 |
| ΦAR_ctx4_noised | 117e-3 | 18.5e-4 | 7.95 |
| TIE r=1.0 | 14.8e-3 | 20.6e-4 | 2.29 |

The pattern: at 10 frames, AR baselines are competitive (rigidity loss within 10×). At 49 frames, AR baselines degrade by 10–150× while PhysiFormer stays stable. The failure mode: **stationary objects fail to remain at rest, rigid objects deform, objects escape the bounding box** — classic error accumulation.

### 1.2 Why single-shot wins (§3, §4.5)

The paper identifies two causes:

1. **Train-test distribution mismatch (exposure bias).** AR models train on ground-truth context but test on their own outputs. Noise injection (ΦAR_ctx4_noised) and longer context windows mitigate but do not eliminate this.

2. **Irreducible error accumulation.** Even with perfect in-distribution training (Table 5: GT-conditioned AR has rigidity loss ~1e-7), self-conditioned rollout compounds per-step error multiplicatively. This is structural, not fixable by training. Diffusion Forcing and Self Forcing (cited) partially address it but don't remove it.

PhysiFormer sidesteps both by predicting the full trajectory jointly: no rollout → no exposure bias → no error accumulation. The cost is a fixed inference budget (50 denoising steps regardless of trajectory length), which the paper shows is competitive with physics simulators for elastic objects (6.4s on H100 vs 20–36s for Genesis on CPU).

### 1.3 The generative uncertainty property

Because mass, friction, and restitution are NOT provided as conditioning inputs, PhysiFormer samples **diverse plausible futures** from the same initial conditions. Five generations have non-zero MSE variance (σ = 0.293) but near-identical rigidity loss (σ = 0.021). Interpretation: the model captures the uncertainty over unobserved physical parameters as trajectory-level diversity. This is a property of the generative formulation, not the diffusion specifically — any trajectory-level sampler would inherit it.

### 1.4 Factorized attention (architectural, secondary)

The DiT backbone uses alternating spatio-temporal attention, with the novelty of factoring spatial attention into **full-spatial** (all vertices in a frame) and **object-level** (vertices within each object). This gives permutation-invariant multi-object reasoning without object-ID embeddings, and generalizes to unseen object counts (trained on ≤5 objects, tested on 15). Cost: O(TN² + NT²) instead of O(T²N²). This is an engineering contribution — alternating attention is prior art (SVD, Align-Your-Latents), and the object-level factorization is structured masking.

### 1.5 What is NOT the contribution

- The DiT-L architecture (from Peebles & Xie).
- The JiT x-prediction-with-v-loss objective (from Li & He).
- Coordinate-conditioned RoPE (from RenderFormer).
- Register tokens (from Darcet et al.).
- Operating in world space vs pixel space (the representation choice, motivated by RenderFormer).

These are all composition-of-known-parts. The contribution is: **(a) the demonstration that single-shot joint diffusion works for physics and beats AR, and (b) the factorized attention pattern.** Of these, (a) is the transferable principle; (b) is engineering.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase)

| Paper term | DEC / codebase equivalent | Where it ships |
|---|---|---|
| Vertex trajectory `X ∈ R^{T×N×3}` | Cochain field evolution `h(t) ∈ CochainField` over T ticks | `katgpt-core::dec::CochainField` |
| Autoregressive rollout `X_{t+1} = f(X_t)` | Step-by-step Euler `h_{t+1} = h_t + dt·A·h_t` via `evolve_motor_gated_field` | Plan 357, `dec::evolve_motor_gated_field` |
| Single-shot diffusion (50 Heun steps) | Heat kernel / operator exponential `h(t) = exp(t·A)·h₀` | **NOT SHIPPED on DEC** — `heat_kernel_weights` ships only on KG Laplacian (slod.rs) |
| Rigidity preservation | Hodge decomposition preservation (exact ⊕ harmonic ⊕ coexact) | `dec::hodge_decompose`, Research 219/296 |
| Momentum drift | `belief_mass_divergence` (DEC flux conservation) | Plan 314, `dec::belief_mass_divergence` |
| Error accumulation | Per-step Euler error compounding `O(T·dt²)` | Well-known numerical analysis; our `tau_reest` re-estimation trigger exists BECAUSE of it |
| Material conditioning (rigid/elastic) | Motor-gated channels (multiplicative gain on specific channels) | `evolve_motor_gated_field`, Plan 357 |
| Generative uncertainty (5 diverse futures) | K-hypothesis trajectory sampling (BoMSampler extended to trajectories) | `best_belief.rs` (single-step); trajectory-level NOT shipped |
| Factorized attention (time/space/object) | Alternating attention (SHINE M2P transformer) | `riir-gpu/src/hypernet/m2p_transformer.rs` (hypernet, not game NPC) |
| Register tokens (global context) | Standard ViT technique | N/A for DEC (DEC has no attention) |

**Grep verification:** paper-vocabulary grep (`PhysiFormer|mesh.*diffusion|vertex.*trajectory|single.shot.*diffusion`) returned **zero hits** across all five repos. Codebase-vocabulary grep (`evolve_motor_gated|heat_kernel|hodge_laplacian|expm`) hit: `evolve_motor_gated_field` (Plan 357, step-by-step), `heat_kernel_weights` (slod.rs, KG only), no DEC trajectory exponential. **The single-shot DEC heat kernel trajectory primitive does not ship.**

### 2.2 The modelless fusion (the actual contribution)

**The PhysiFormer principle in DEC language:**

Our `evolve_motor_gated_field` advances the cochain field by one step:
```
h_{t+1} = h_t + dt · (-h_t + Δ·h_t + motor ⊙ h_t)
        = (I + dt·A) · h_t        where A = -I + Δ + diag(motor)
```
For a T-step trajectory, we call it T times: `h_T = (I + dt·A)^T · h₀`. This is **explicit Euler** — the autoregressive approach PhysiFormer argues against. Per-step local error `O(dt²)`, global error `O(T·dt²)`.

**The single-shot analog (modelless, the missing primitive):**

```
h(t) = exp(t · A) · h₀       — the DEC heat kernel / operator exponential
```

This is the **exact** solution to the linear DEC propagation equation. It is computable via:
- **Eigendecomposition** (offline, once): `A = VΛV⁻¹`, then `h(t) = V·exp(t·Λ)·V⁻¹·h₀`. Each eigenvector damped by `exp(t·λ_k)` independently. Harmonic components (λ=0) preserved exactly; exact/coexact components damped by their eigenvalues. **The Hodge decomposition is preserved exactly.**
- **Krylov subspace** (online, per query): `exp(t·A)·h₀ ≈ V_k · exp(t·H_k) · V_kᵀ · h₀` where `V_k` is the k-dimensional Krylov basis (k≈20–50). Cost: O(k·nnz(A)) = O(k·n) for a sparse grid Laplacian. For T > k, this is **cheaper than T Euler steps** AND more accurate.

**Why this is the PhysiFormer principle, modellessly:**

| PhysiFormer (trained diffusion) | DEC heat kernel (modelless) |
|---|---|
| Single denoising pass → full trajectory | Single matrix-explicit → full trajectory |
| Avoids AR error accumulation | Avoids Euler error accumulation (exact for linear) |
| Preserves rigidity (learned) | Preserves Hodge decomposition (analytic) |
| 50 Heun steps, O(50·forward) | k Krylov iterations, O(k·nnz) |
| Captures uncertainty via stochasticity | Deterministic; uncertainty via BoM extension |
| Needs trained DiT-L weights | Needs only the DEC operator (already shipped) |

### 2.3 The nonlinear case (with ReLU gate)

`evolve_motor_gated_field` includes a `relu_gate_into` step (the non-negative lateral propagation from Research 359 §1.3). This makes the propagation nonlinear: `h_{t+1} = (I + dt·A)·ReLU(h_t)`.

The heat kernel doesn't directly apply to nonlinear operators. But **exponential integrators** (Hochbruck & Ostermann 2010) handle this: split the operator into linear `L` (the Δ part, where exp applies) and nonlinear `N(h)` (the ReLU part, treated as a source term):

```
h(t) = exp(t·L)·h₀ + ∫₀ᵗ exp((t-s)·L)·N(h(s))ds
```

Approximated by Krylov on `L` + quadrature on the integral. This is a standard higher-order method — still modelless, still no training, and still avoids the per-step Euler error accumulation that PhysiFormer identifies as the failure mode.

**§3.5 modelless-unblock check:** all three paths are satisfied trivially — there is nothing to "unblock" because the primitive IS the modelless solution. The freeze/thaw path (Path 1) applies to the motor-direction vectors (frozen `LatentSteeringVector`s, already shipped). The latent-correction path (Path 3) is the heat kernel itself (analytic correction of the Euler approximation error). No riir-train dependency.

### 2.4 Connection to existing primitives (force-multiplier map)

| Existing primitive | Relationship | What the heat kernel adds |
|---|---|---|
| **DEC operators** (Plan 251) | The operator `A` is built from `hodge_laplacian` | The trajectory-level operator `exp(t·A)` |
| **Motor-Gated DEC Field** (Plan 357, R359) | The step-by-step version | The single-shot version (this note) |
| **SLoD heat kernel** (Plan 235, R208) | Precedent: heat kernel on KG graph Laplacian | Extension: heat kernel on DEC cochain Laplacian for trajectory prediction |
| **KARC forecaster** (Plan 308/332) | Step-by-step delay-basis ridge forecasting | Single-shot trajectory via operator exponential |
| **BoMSampler** (Plan 281) | K-hypothesis sampling at single timestep | Extension: K-hypothesis trajectory sampling (diverse futures) |
| **ReestimationScheduler** (R123, Plan 303) | Exists BECAUSE of step-by-step error accumulation | Single-shot avoids the need for re-estimation triggers |
| **Sleep-Time anticipation** (Plan 341) | Single-step projection of belief | Full-trajectory projection from initial belief |
| **belief_mass_divergence** (Plan 314) | Per-tick flux conservation check | Trajectory-level conservation bound |

---

## 3. Latent-Space Reframing (mandatory per workflow §1.5 step 3)

Re-cast the single-shot trajectory prediction on each Super-GOAT factory module:

### 3.1 HLA per-NPC latent state

HLA's 8-dim per-NPC state is a vector, not a cochain. To apply the heat kernel, construct a cell complex on the NPC's belief manifold (e.g., discretize ℝ⁸ via `SafeManifoldGraph` from Plan 312). The heat kernel then predicts the full HLA trajectory from initial state — useful for sleep-time anticipation (predict the NPC's emotional trajectory over the next planning horizon without per-tick simulation). The `evolve_hla` kernel (step-by-step) is the Euler approximation; the heat kernel is the exact linear evolution.

### 3.2 latent_functor operations

`apply_functor` (rank-1 additive) is one Euler step. A trajectory-level functor would be `exp(T·F)·z₀` where `F` is the functor's linearization — predicting where the latent state ends up after T applications without iterating. This connects to KARC's delay-basis: the delay embedding IS a trajectory-level representation.

### 3.3 DEC cochain fields (the primary target)

This is where the heat kernel lands most naturally. The game map IS a cell complex (`CellComplex::grid_2d`). A threat/occupancy/safety field IS a cochain (`CochainField`). The motor-gated Hodge-Laplacian IS the propagation operator. The heat kernel `exp(t·A)·h₀` predicts the full field evolution — where the threat field will be in 5 seconds, computed in one pass. **This is the DEC analog of PhysiFormer's mesh trajectory diffusion.**

### 3.4 KARC delay-basis

KARC's delay embedding `[z_{t-d}, ..., z_{t-1}]` is already a trajectory representation. The ridge regression maps delay-embedding → next-step. A trajectory-level KARC would map delay-embedding → full-future-trajectory via a single operator (the ridge solution extended to multi-step output). This is the linear-algebra version of PhysiFormer's "predict the whole trajectory jointly."

### 3.5 SLoD heat kernel (the direct precedent)

SLoD's `heat_kernel_weights(eigenvalues, eigenvectors, query, sigma)` computes `w_i = Σ_k exp(-λ_k·σ)·v_k[i]²·query_coeffs[k]` — exactly the spectral form of the heat kernel on a graph Laplacian. The DEC extension replaces the graph Laplacian with the Hodge-Laplacian, and `sigma` (diffusion time) becomes `t` (prediction horizon). The implementation pattern (precompute eigendecomposition offline, query heat kernel online) transfers directly.

---

## 4. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|---|---|---|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | Open primitive + private guide |
| **GOAT** | Provable gain over existing approach, not a new class | Plan + implement, feature flag + benchmark |
| **Gain** | Incremental improvement | Plan only, behind feature flag |
| **Pass** | Not relevant, or training-only | One-line note |

### Verdict: **GOAT** — provable gain, modelless DEC analog

**One-line reasoning:** The single-shot heat kernel `exp(t·A)·h₀` is **provably exact** for linear DEC propagation (vs `O(T·dt²)` error for step-by-step Euler), is **modelless** (Krylov subspace, no training), is **novel for our codebase** (heat kernel ships for KG not DEC trajectory), and **connects ≥4 existing primitives** (DEC, Motor-Gated Field, SLoD, KARC). PhysiFormer's empirical evidence (100× rigidity improvement at 49 frames) validates the principle in a different domain.

**Why GOAT and not Super-GOAT:**
- The mechanism (matrix exponential / heat kernel) is well-established in numerical analysis (Hochbruck & Ostermann 2010). The novelty is in the **application** to DEC cochain fields for game AI trajectory prediction, not in the math itself.
- The product selling point ("zone-level crowd trajectory prediction in one pass") is incremental over the existing step-by-step `evolve_motor_gated_field` — it's an accuracy + latency improvement, not a new capability class. We already predict field evolution; this predicts it more accurately and in one pass.
- The force-multiplier map is strong (≥4 primitives) but the connection is "this replaces the step-by-step version of X" rather than "this composes X + Y into something neither does alone."

**Honest assessment of gain magnitude:**
- For **short horizons** (T < k ≈ 20–50 Krylov dimensions): step-by-step Euler is cheaper and the error is negligible. The heat kernel provides no benefit. At 20Hz tick, 1-second prediction = 20 steps ≈ Krylov dimension — break-even.
- For **long horizons** (T > 50, multi-second): the heat kernel is both cheaper (O(k·n) vs O(T·n)) and dramatically more accurate. Sleep-time anticipation, zone-level crowd flow over 5+ seconds, KARC multi-step forecasting benefit.
- For **nonlinear** (with ReLU gate): the exponential integrator is higher-order but the gain over Euler depends on the stiffness of the nonlinearity. Needs benchmarking.

**Path to Super-GOAT (if GOAT gate passes strongly):** if the benchmark shows ≥10× accuracy improvement at T=100 with ≤2× latency cost, AND the multi-hypothesis trajectory extension (BoM over trajectories) produces genuinely diverse plausible futures (the PhysiFormer generative uncertainty property), this could upgrade to Super-GOAT — "NPCs that sample diverse plausible crowd-flow futures for the full planning horizon in one pass, with exact topological invariant preservation." That would be a new capability class.

### MOAT gate (per domain §1.6)

| Domain | Fit | Rationale |
|---|---|---|
| **katgpt-rs** (public engine) | ✅ **Strong fit** | This is a **paper-derived fundamental/principle primitive** (single-shot trajectory prediction principle from PhysiFormer) fused with the DEC substrate (existing pillar). The heat kernel on DEC cochain fields is generic math with no game semantics. Feature flag + benchmark + GOAT gate. Promote/demote tracked per the DEC stack slot. |
| riir-ai (private runtime) | Application layer | Game AI wiring (crowd flow, NPC anticipation, anti-cheat) is private. But the primitive itself is public math. |
| riir-chain | N/A | No commitment/sync-boundary angle. |
| riir-neuron-db | N/A | No shard/freeze angle (motor directions are already frozen via existing mechanisms). |
| riir-train | Training recipe only | The DiT-L diffusion training (JiT objective, material conditioning, noise scale) → riir-train. Not this note. |

---

## 5. Plan (sketch — full plan in `.plans/365_`)

**Target:** `katgpt-rs/crates/katgpt-core/src/dec/heat_kernel_trajectory.rs` + feature `dec_heat_kernel_trajectory`

**Phases:**
1. **Linear heat kernel** — `heat_kernel_trajectory_linear(cx, h0, motor, t, eigendecomposition)` using precomputed DEC eigendecomposition. Exact for linear propagation. Benchmark vs T-step `evolve_motor_gated_field` at T=20, 50, 100, 200.
2. **Krylov online** — `heat_kernel_trajectory_krylov(cx, h0, motor, t, k)` for sparse grids where eigendecomposition is too expensive. O(k·nnz) per query.
3. **Nonlinear exponential integrator** — split linear (Δ) + nonlinear (ReLU), Krylov on linear + quadrature on nonlinear.
4. **Multi-hypothesis trajectory** — extend BoMSampler to trajectory-level: K diverse `h(t)` samples via perturbed initial states or perturbed motor vectors. The modelless analog of PhysiFormer's generative uncertainty.
5. **GOAT gate** — G1 (exact vs Euler: error = 0 for linear at any T), G2 (latency: Krylov O(k·n) vs Euler O(T·n) at T=100), G3 (Hodge decomposition preserved: check `hodge_decompose` components unchanged), G4 (zero-alloc after eigendecomposition precompute), G5 (no-regression on existing DEC tests).

**Feature flag:** `dec_heat_kernel_trajectory` (opt-in). Promote to default if G1+G2+G3 pass at T≥50.

---

## 6. What stays public vs private

| Component | Visibility | Rationale |
|---|---|---|
| Heat kernel on DEC cochain Laplacian (generic math) | **Public** (katgpt-rs) | Generic inference substrate; no game semantics |
| Krylov subspace exponential integrator | **Public** (katgpt-rs) | Standard numerical method, no IP |
| Motor-gated heat kernel `exp(t·(-I+Δ+diag(motor)))` | **Public** (katgpt-rs) | The motor vectors are inputs; the operator is generic |
| Game AI wiring (crowd flow prediction, NPC anticipation, anti-cheat bounds) | **Private** (riir-ai) | Product-specific tuning and integration |
| Multi-hypothesis trajectory BoM extension | **Public** (katgpt-rs) if generic, **Private** if game-specific | Depends on implementation |

---

## 7. Limitations and honest risks

1. **The linear case is the strong claim; the nonlinear case is weaker.** For propagation without ReLU, the heat kernel is exact — an unconditional win. With ReLU, it's an exponential integrator — higher-order but the gain depends on nonlinearity stiffness. The GOAT gate must test BOTH.

2. **Eigendecomposition cost.** For a 256×256 grid (65536 vertices), the full eigendecomposition is O(n³) — prohibitive. The Krylov online path avoids this but has per-query cost. The right approach: precompute top-k eigenvectors offline (like SLoD does), use truncated spectral heat kernel for long-horizon, Krylov for ad-hoc queries.

3. **The break-even horizon is uncertain.** At 20Hz tick, 1 second = 20 steps ≈ typical Krylov dimension. For sub-second prediction, Euler is fine. The gain is for multi-second horizons (sleep-time, zone-level planning). Whether game AI actually needs multi-second field trajectory prediction is a product question.

4. **PhysiFormer's domain (3D mesh physics) is not our domain (2D game maps + HLA).** The principle transfers, but the magnitude of the gain may differ. Our DEC propagation is already topology-preserving by construction (d∘d=0); PhysiFormer's AR baselines had NO such guarantee. The "error accumulation breaks structure" failure mode may be less severe for us because our structure is enforced per-step.

5. **The multi-hypothesis trajectory extension is speculative.** PhysiFormer's generative uncertainty comes from diffusion's stochasticity. Our modelless analog (perturbed initial states or motor vectors) may not produce meaningfully diverse futures — it depends on the sensitivity of the heat kernel to perturbations, which is governed by the eigenvalue spectrum.

---

## TL;DR

PhysiFormer's fundamental contribution is NOT the trained DiT-L — it's the **prediction-strategy principle**: single-shot joint trajectory prediction avoids the compounding error of step-by-step autoregressive rollout. The modelless DEC analog is the **heat kernel / operator exponential** `exp(t·A)·h₀`, which is **exact** for linear DEC propagation (zero error accumulation, exact Hodge-decomposition preservation) and computable in O(k·n) via Krylov subspace. We already ship `evolve_motor_gated_field` (step-by-step Euler, Plan 357) and `heat_kernel_weights` (on KG Laplacian only, Plan 235) — the DEC cochain trajectory heat kernel is the missing fusion. **Verdict: GOAT** — provable gain for linear case, modelless, novel application, connects ≥4 primitives. Promote to default if the GOAT gate passes at T≥50. The training recipe (JiT DiT-L diffusion) → riir-train. The factorized attention and register-token findings are engineering, not fundamental.
