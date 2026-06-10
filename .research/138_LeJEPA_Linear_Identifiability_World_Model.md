# Research 138: LeJEPA — When Does LeJEPA Learn a World Model?

> **Source:** [When Does LeJEPA Learn a World Model?](https://www.alphaxiv.org/abs/2605.26379) — David Klindt, Yann LeCun, Randall Balestriero, 2026-05
> **Date:** 2026-05-30
> **Related Research:** 115 (PEIRA), 136 (Latent Prediction), 121 (Spectral Hierarchy), 070 (GDN2), 034 (D2F), 035 (Attractor Fixed-Point), 051 (Deep Manifold)
> **Related Plans:** 156 (Spectral Hierarchy ✅), 149 (Dirichlet Energy ✅), 150 (PEIRA ⏳), 154 (Sleep ⏳)
> **Domain:** katgpt-rs (open, representation quality diagnostics)

---

## TL;DR

LeJEPA proves that a JEPA learns a **linearly identifiable** representation (recovers the world's true latent variables up to linear transform) **if and only if the latent variables are Gaussian**. The proof uses a Hermite spectral decomposition where each degree of non-linearity is strictly penalized by alignment, making the linear map the optimum. Identifiability degrades gracefully when objectives are only approximately satisfied (approximate bound). Linear identifiability is sufficient for optimal latent-space planning.

**Verdict: LOW-MODERATE GAIN — Three distillable diagnostics: (1) Hermite spectral decomposition as representation quality metric (non-linearity penalty per degree), (2) approximate identifiability bound as graceful degradation test for our PEIRA/spectral alignment, (3) linear identifiability validates that linear probing is the correct evaluation tool. No new algorithm — the paper's contribution is theoretical proof for training-time representation learning. We don't train JEPAs in Rust. The spectral diagnostic is the only transferable code, and our existing `eigenspace_alignment` + `cauchy_interlacing_check` already cover related ground. No feature gate needed — add Hermite non-linearity score as optional output to existing `spectral_hierarchy` module if demand arises. No plan.**

---

## Key Contributions

| # | Contribution | Formal Statement | Transferability |
|---|-------------|-------------------|-----------------|
| 1 | **Linear identifiability (⇔)** | LeJEPA learns linearly identifiable rep ↔ latents are Gaussian | Theoretical — validates linear probing as correct evaluation |
| 2 | **Spectral decomposition proof** | Each Hermite degree k > 0 is strictly penalized by alignment weight λ_k | Diagnostic: Hermite coefficient spectrum = representation quality |
| 3 | **Approximate bound** | If alignment + Gaussianity satisfied to ε, identifiability error = O(ε) | Graceful degradation test for our spectral diagnostics |
| 4 | **Optimal planning** | Linear orthogonal identifiability ⟹ optimal LQR planning in latent space | Validates our MCTS + forward model operating in learned coordinates |

---

## Core Architecture

### LeJEPA Objective

```
L = L_align + λ_SIG · L_SIGReg

L_align = E[‖encoder(x_t) - predictor(encoder(x_{t+1}))‖²]
L_SIGReg = D_SKL(N(0,I) ‖ embedding_distribution)  // sketched isotropic Gaussian
```

The SIGReg component forces embeddings toward isotropic Gaussian. The alignment component forces temporal prediction to match. Together, they guarantee linear identifiability.

### Hermite Spectral Decomposition (The Key Proof Technique)

The encoder f = Σ_k c_k H_k where H_k are Hermite polynomials:
- k=0: constant (trivial, removed by centering)
- k=1: linear ← **this is what identifiability requires**
- k>1: non-linear ← **alignment strictly penalizes each degree**

The alignment objective decomposes into:
```
L_align = Σ_{k≥2} λ_k · ‖c_k‖²    where λ_k > 0 strictly
```

So non-linear components are ALWAYS penalized, and the linear component (k=1) is the unique optimum.

### Approximate Bound

If the LeJEPA objectives are only ε-approximately satisfied:
```
‖W_true - A·W_learned‖_F ≤ C · ε^{1/2}
```

This means even imperfect training still recovers approximately correct latent structure.

---

## Mapping to Our Architecture

### What We Already Have

| Paper Concept | Our Equivalent | Status |
|---------------|---------------|--------|
| Linear identifiability | `eigenspace_alignment(g(k))` — measures eigenvector alignment | ✅ Plan 156 COMPLETE |
| Spectral decomposition | `cauchy_interlacing_check` — validates eigenvalue ordering | ✅ Plan 156 COMPLETE |
| Gaussian regularization | `GDN2` gated erase/write (Gaussian recurrence) | ✅ Plan 105 COMPLETE |
| Latent-space planning | `MCTS` + `GameState` forward model | ✅ STRATEGA distilled |
| Collapse prevention | PEIRA `Tr(Σ(N+λI)⁻¹)` — proven collapse-free | ✅ Research 115 |
| Hermite non-linearity | `dirichlet_energy` — measures smoothness (related but different) | ✅ Plan 149 COMPLETE |

### What's New (Potential Adds)

| Concept | Transfer | Value | Effort |
|---------|----------|-------|--------|
| Hermite coefficient spectrum as non-linearity diagnostic | Diagnostic only | LOW — we already have eigenspace + Dirichlet | 1-2 days |
| Approximate bound as graceful degradation test | Test only | LOW — theoretical, not code | 0.5 day |
| SIGReg as training regularizer | Training-time only — katgpt-rs is inference | NONE — we don't train | N/A |

---

## Distillation Verdict

### Tier 1: Already Covered (No Action)

1. **Linear probing is correct evaluation** — Our `eigenspace_alignment` already does this. The paper proves it's theoretically justified. No code change.
2. **Gaussian latent structure** — Our `GDN2` recurrence is already Gaussian-gated. Research 070 distilled this.
3. **Spectral analysis** — Our `spectral_hierarchy` module already decomposes co-occurrence matrices into eigenvectors + checks alignment. Research 121 proved this.
4. **Latent-space planning** — Our MCTS already plans in learned coordinates. STRATEGA (Research 027) distilled this.

### Tier 2: Marginal Diagnostic (Optional Future)

5. **Hermite non-linearity spectrum** — Could add to `spectral_hierarchy` as `hermite_nonlinearity_score()`. But `eigenspace_alignment` + `dirichlet_energy` already cover the "is the representation good?" question. The Hermite decomposition is more precise (degree-by-degree) but the marginal value over existing diagnostics is small.
6. **Approximate bound test** — Could add as a test assertion: "if eigenspace alignment > 0.9, then representation approximately identifies latents." But this is already implicit in our GOAT proof thresholds.

### Tier 3: Out of Scope

7. **SIGReg training regularizer** — We don't train in katgpt-rs. This is riir-ai territory (training pipeline).
8. **JEPA architecture** — We don't implement JEPA encoders. Our representation learning is through GDN2 fast weights, not predictive coding.

---

## Connection to Optimization.md

The paper's spectral decomposition is an **offline diagnostic** (computed once on saved representations), not a hot-path operation. No optimization concerns:
- Hermite coefficient computation is O(n × d × K) where K = max degree (typically 5-10)
- Not in the inference hot path
- Can be computed in debug builds only

---

## Civilization Engine (Plan 168) Relevance

**None.** LeJEPA is about representation learning theory. Civilization Engine is about game design composition (conflict, economy, aging). No overlap. The paper's "linear identifiability → optimal planning" result theoretically validates MCTS in learned coordinates, but our MCTS already works in explicit game state space, not latent space.

---

## Verdict

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Research file | ✅ This file | Theoretical validation of existing approach |
| Plan | ❌ No plan needed | No new code to write — existing diagnostics cover it |
| Feature gate | ❌ Not needed | No new functionality |
| GOAT proof | ❌ Not applicable | Paper proves theory, not our code |
| Civilization Engine | ❌ Not related | Different domain entirely |

**Bottom line:** LeJEPA proves that our existing approach (spectral analysis + Gaussian fast weights + linear probing) is theoretically sound. The paper is a **validation** of choices we already made, not a source of new algorithms. The Hermite decomposition is a potentially more precise diagnostic than what we have, but the marginal value is small compared to existing `eigenspace_alignment` + `dirichlet_energy`.

---

## References

- Klindt, LeCun, Balestriero. "When Does LeJEPA Learn a World Model?" arXiv:2605.26379, 2026.
- Research 115: PEIRA (collapse-free alignment, our closest analog)
- Research 136: Latent Prediction Sample Complexity (JEPA hierarchy proof)
- Research 121 → Plan 156: Spectral Hierarchy (eigenspace alignment, our implementation)
- Research 070 → Plan 105: GDN2 (Gaussian-gated recurrence, our JEPA-like component)
