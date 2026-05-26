# Research 115: PEIRA — Predictive Encoders through Inter-View Regressor Alignment

> **Source:** [PEIRA: Learning Predictive Encoders through Inter-View Regressor Alignment](https://arxiv.org/pdf/2605.17671) — Michael Arbel, Basile Terver, Jean Ponce (Inria / ENS), 2026
> **Date:** 2026-05-26
> **Related Research:** 037 (REAP Model-Based/Modelless), 039 (SpectralQuant), 051 (Deep Manifold), 054 (ASFT), 070 (GDN2), 080 (VPD)
> **Related Plans:** 153 (PEIRA modelless distillation)
> **Related MMO Pillars:** Pillar 3 (NPC Dialog — representation alignment), cross-cutting (LoRA training stability)

---

## TL;DR

PEIRA is a **non-contrastive self-supervised** method that learns representations by training a **regularized linear regressor** to predict representations of one data view from another. Its explicit objective — EPEIRA = −½ Tr(P_{U,V}) + λ/2 (‖U‖² + ‖V‖²) — has a clean closed form: the only stable equilibria are non-trivial global minimizers spanning **nonlinear CCA subspaces**. Unlike SimSiam, BYOL, and I-JEPA (which can collapse), PEIRA **provably does not collapse**. The auxiliary loss L_aux avoids differentiating through the matrix inverse, and the gradient is 4/λ-Lipschitz (well-conditioned optimization).

**Why it matters to us:** Our model-based/modelless framework (Research 037) needs a principled way to align representations across the spectrum. PEIRA's trace maximization between views maps directly: U = model-based representation, V = modelless representation. The collapse-free guarantee and CCA subspace recovery provide a theoretically grounded alternative to our existing heuristic distillation losses (SDAR Plan 038, ROPD Plan 036, VPD Plan 080), with a measurable quality signal (spectral alignment between Σ and N eigenvectors) that can feed into `ScreeningPruner::relevance()`.

---

## 1. Key Ideas

### 1.1 The PEIRA Objective

Given two encoder networks f_φ (producing view U) and f_ψ (producing view V), PEIRA trains a **regularized linear regressor** to predict V from U and vice versa. The objective is:

```
EPEIRA(φ, ψ) = -½ Tr(P_{U,V}) + λ/2 (‖U‖² + ‖V‖²)
```

where:
- P_{U,V} = Σ_{U,V} (N_{U,V} + λI)⁻¹
- Σ_{U,V} = cross-covariance matrix between the two views
- N_{U,V} = average of the two auto-covariance matrices
- λ > 0 is the regularization parameter

The trace term Tr(P_{U,V}) captures **how much of the cross-view structure** the regularized regressor can explain. Maximizing it drives both encoders to produce representations that are mutually predictive while the λ penalty prevents degenerate solutions.

### 1.2 Collapse-Free Guarantee

Self-distillation methods (SimSiam, BYOL, I-JEPA) require architectural tricks (stop-gradients, momentum encoders, exponential moving average targets) to avoid representational collapse. PEIRA proves that:

1. **Trivial solutions (constant representations) are unstable equilibria** — any perturbation away from collapse is amplified by the gradient.
2. **The only stable equilibria are non-trivial global minimizers** that span the nonlinear CCA subspace.
3. This is a **property of the objective**, not of the optimizer or architecture.

This means no stop-gradients, no momentum encoders, no predictor networks with asymmetric architectures. The regularization λ alone ensures collapse-free learning.

### 1.3 CCA Subspace Recovery

PEIRA recovers the **nonlinear canonical correlation analysis (CCA)** subspaces of the two views. At convergence:

- The learned representations span the same subspace as the top-k CCA directions
- k is controlled by the regularization λ (larger λ → fewer CCA directions recovered)
- The eigenvalues of P_{U,V} at convergence equal the canonical correlations ρ₁, ρ₂, ..., ρ_k

This provides a natural **quality metric**: the canonical correlations directly measure how much shared structure the two views capture.

### 1.4 Auxiliary Loss and Stochastic Optimization

Computing Tr(P_{U,V}) requires a matrix inverse — differentiating through it is expensive and numerically unstable. PEIRA uses an **auxiliary loss**:

```
L_aux = -½ Tr(Σ_{U,V} A) + ¼ Tr(A (N_{U,V} + λI) A^T)
```

where A is a learnable auxiliary matrix. At A* = (N_{U,V} + λI)⁻¹ Σ_{U,V}^T, L_aux equals L_PEIRA.

**Stochastic Compositional Optimization (SC-PEIRA, Algorithm 1):**
- Maintain EMA estimates of Σ and N matrices (no per-batch matrix inverse)
- Update A via gradient step on L_aux
- Update encoders via gradient through L_aux
- The gradient is **4/λ-Lipschitz** — well-conditioned regardless of encoder architecture

### 1.5 Spectral Alignment Dynamics

During training, the eigenvectors of Σ_{U,V} (signal cross-covariance) and N_{U,V} (noise auto-covariance) **gradually align**. This spectral alignment:

- Starts random (no correlation between eigenbases)
- Progressively concentrates the signal into the top eigenvectors
- Provides a **training health metric**: alignment progress directly reflects representation quality

---

## 2. Relevance to katgpt-rs

### 2.1 Model-Based/Modelless Representation Alignment

In our framework (Research 037), representations span a spectrum from **modelless** (router counts, bandit Q-values, static rules) to **model-based** (full forward pass, LoRA gradients). PEIRA maps directly:

| PEIRA Concept | Our Analog |
|---------------|-----------|
| View U encoder | Model-based encoder (full forward pass with LoRA) |
| View V encoder | Modelless encoder (bandit/routing/heuristic) |
| Cross-covariance Σ_{U,V} | Correlation between model-based and modelless quality signals |
| Auto-covariance N_{U,V} | Signal variance within each mode |
| Canonical correlations ρ₁...ρ_k | How well modelless predicts model-based quality |

The trace maximization gives a **principled alignment loss** between the two ends of our spectrum — not a heuristic KL or L2, but an information-theoretically grounded measure.

### 2.2 Distillation Loss Alternative

Our existing distillation losses each have tradeoffs:

| Method | Plan | Mechanism | Weakness |
|--------|------|-----------|----------|
| GFlowNet | 052 | Flow-based trajectory scoring | Complex, requires careful temperature tuning |
| δ-Mem | 053 | Delta signal from log-prob differences | Requires on-policy rollouts |
| ROPD | 071 | Rubric-conditioned on-policy distillation | Needs hand-crafted rubrics |
| SDAR | 072 | Self-distilled agentic RL with gated absorption | Heuristic gating |
| VPD | 080 | Variational EM co-evolution of teacher/student | Complex EM loop, expensive |
| **PEIRA** | **153** | **Regularized trace maximization, collapse-free** | **Trail VICReg by ~2.3pp on ImageNet** |

PEIRA's advantages:
- **Provably collapse-free** — no architectural tricks needed
- **Closed-form quality metric** — canonical correlations ρ directly measure alignment quality
- **Simple implementation** — EMA covariance matrices + auxiliary matrix A
- **4/λ-Lipschitz gradient** — well-conditioned, compatible with any optimizer (including AMUSE, Research 114)

### 2.3 ScreeningPruner Quality Signal

The spectral alignment between Σ_{U,V} and N_{U,V} eigenvectors provides a measurable indicator of representation quality. This can feed directly into our existing pruning architecture:

```rust
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}
```

A `PEIRAPruner` could implement `relevance()` as the canonical correlation between model-based and modelless representations of the current token — a theoretically grounded quality score that replaces heuristic relevance functions.

### 2.4 CCA Subspace as Quality Metric

The canonical correlation structure provides a natural "quality metric" for distillation: measure how well the student (modelless) spans the teacher's (model-based) CCA subspace. This is:

- **Interpretable** — ρ = 1 means perfect alignment, ρ = 0 means orthogonal
- **Differentiable** — gradients flow through the EMA covariance estimates
- **Efficient** — only requires maintaining running covariance matrices

---

## 3. Connections to Existing Work

### 3.1 Direct Connections

| Research | Connection | PEIRA's Contribution |
|----------|------------|----------------------|
| **037 (REAP)** | Same model-based/modelless spectrum | Theoretical bridge: trace maximization formalizes alignment between modes |
| **039 (SpectralQuant)** | Eigenbasis alignment is central to both | SpectralQuant calibrates eigenbasis for KV compression; PEIRA aligns eigenbases for representation quality. Same spectral toolkit, different target. |
| **051 (Deep Manifold)** | Fixed-point boundary conditions | PEIRA's equilibrium analysis (stable = non-trivial, unstable = collapse) provides explicit fixed-point characterization for the alignment boundary |
| **054 (ASFT)** | Both prevent drift | ASFT anchors via forward KL against base model; PEIRA anchors via regularized trace. Different mechanisms, same goal. |
| **070 (GDN2)** | Signal/noise decomposition | GDN2's erase/write gate decomposition separates signal from noise in attention state; PEIRA's Σ/N decomposition separates cross-view signal from per-view noise. Same structural insight at different granularities. |
| **080 (VPD)** | Both do teacher/student co-evolution | VPD uses variational EM (E-step trains teacher, M-step distills to student); PEIRA uses regularized regression (simultaneous, no EM loop). PEIRA is simpler with stronger theoretical guarantees but less expressive. |
| **114 (AMUSE)** | Optimization compatibility | PEIRA's 4/λ-Lipschitz gradient means well-conditioned updates — compatible with AMUSE's Schedule-Free + Muon orthogonalization. They compose. |

### 3.2 Distinction from Contrastive Methods

| Method | Collapse Prevention | Objective | Quality Metric |
|--------|--------------------|-----------|---------------|
| SimCLR | Negative samples | InfoNCE | None explicit |
| BYOL | Momentum encoder + stop-gradient | Prediction | None explicit |
| SimSiam | Stop-gradient + predictor asymmetry | Cosine similarity | None explicit |
| VICReg | Variance/Invariance/Covariance regularizer | Heuristic balance | Variance threshold |
| I-JEPA | Predict in representation space | L2 prediction | None explicit |
| **PEIRA** | **Regularization λ (proven)** | **Tr(Σ(N+λI)⁻¹)** | **Canonical correlations ρ** |

PEIRA is the only method with both **provable collapse-freedom** and an **explicit, interpretable quality metric** derived from the objective itself.

---

## 4. Algorithm Pseudocode (Rust Translation Target)

```
PEIRA State:
  - sigma_ema: EMA cross-covariance matrix Σ_{U,V}  (d_u × d_v)
  - n_ema: EMA auto-covariance matrix N_{U,V}        (d × d)
  - A: auxiliary matrix                               (d_v × d_u)
  - ema_momentum: decay rate for covariance estimates (e.g. 0.999)

Per step:
  1. x1, x2 = augment(batch)                    // two views
  2. u = f_φ(x1)                                 // encoder 1 output (d_u)
  3. v = f_ψ(x2)                                 // encoder 2 output (d_v)
  4. sigma_batch = u^T v / batch_size             // cross-covariance estimate
  5. n_batch = (u^T u + v^T v) / (2 * batch_size) // auto-covariance estimate
  6. sigma_ema = ema_momentum * sigma_ema + (1 - ema_momentum) * sigma_batch
  7. n_ema = ema_momentum * n_ema + (1 - ema_momentum) * n_batch
  8. // Auxiliary loss (avoids matrix inverse in backward pass)
  9. L_aux = -½ Tr(sigma_ema @ A)
            + ¼ Tr(A @ (n_ema + λ * I) @ A^T)
 10. // Update A via gradient step on L_aux
 11. A_grad = -½ sigma_ema^T + ½ A @ (n_ema + λ * I)
 12. A = A - lr_A * A_grad
 13. // Update encoders via gradient through L_aux
 14. φ, ψ = optimizer.step(∇_{φ,ψ} L_aux)
```

Key implementation notes:
- Σ and N are maintained as EMA — no per-batch eigendecomposition
- A update is a simple matrix multiply — O(d²) per step
- No stop-gradients, no momentum encoder, no predictor network
- The λ parameter controls the effective rank of recovered CCA subspace

---

## 5. Verdict

**CONDITIONAL ADOPT** — PEIRA's theoretical framework (collapse-free equilibria, CCA subspace recovery, spectral alignment) directly addresses our model-based/modelless alignment problem. The auxiliary loss is simple to implement (EMA covariance matrices + closed-form predictor).

However:

| Concern | Severity | Mitigation |
|---------|----------|-----------|
| ImageNet-1K trails VICReg by ~2.3pp (66.50% vs 68.81%) | Medium | We're not doing vision; LLM representation alignment may differ. Benchmark against our existing losses. |
| Method is new (May 2026) | Medium | Start as feature-gated experiment (`peira_distill`), not default. |
| No LLM-scale validation yet | Medium | Our use case (model-based/modelless alignment) is simpler than full pretraining. Lower risk. |
| Best-case is equal to VICReg on vision | Low | Our modelless representations are simpler than augmented image views. PEIRA's CCA structure may be more appropriate for our setting. |

**Best ROI:** Implement as an alternative distillation loss, benchmark against existing SDAR/VPD/ROPD losses. If canonical correlations provide useful quality signals, wire into `ScreeningPruner::relevance()`.

### Feature Gate Proposal

| Component | Feature Gate | Type | Dependency |
|-----------|-------------|------|-----------|
| PEIRA auxiliary loss | `peira_distill` | Model-based | `bandit` (BanditPruner integration) |
| EMA covariance tracker | `peira_distill` | Infrastructure | None |
| Canonical correlation quality metric | `peira_distill` | Diagnostic | None |
| PEIRAPruner (ScreeningPruner impl) | `peira_distill` | Modelless→model-based | `peira_distill` + `bandit` |

**NOT default-on initially** — experimental, needs GOAT proof.

---

## 6. Open Questions

1. **LLM representation alignment vs. vision:** PEIRA is validated on ImageNet. Do the same collapse-free guarantees hold for token-level representations in autoregressive LLMs where the two "views" are model-based vs. modelless quality signals?
2. **Effective rank for distillation:** What λ gives the right CCA subspace rank for our model-based/modelless alignment? Too small λ → recovers noise dimensions; too large λ → misses important correlations.
3. **Interaction with AMUSE (Research 114):** PEIRA's 4/λ-Lipschitz gradient should compose well with AMUSE's Schedule-Free averaging. But does AMUSE's bulk-oriented update conflict with PEIRA's spectral alignment dynamics?
4. **Online covariance estimation stability:** Our inference path processes one token at a time. Is the EMA covariance estimate stable at batch_size=1, or do we need to accumulate covariance over a sequence window?
5. **Comparison to VPD for our use case:** VPD (Research 080) uses a more expressive variational EM formulation. For our specific model-based/modelless alignment problem, does the simpler PEIRA formulation lose critical expressiveness, or is the CCA subspace sufficient?
6. **Canonical correlations as relevance score:** If we use ρ as a `relevance()` signal in ScreeningPruner, what's the right normalization? Raw ρ ∈ [0, 1] or something scaled by effective rank?
