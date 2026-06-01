# Research 150: RecFM — Recursive Cross-Scale Consistency for Modelless Inference

**Paper:** [arXiv:2605.26535v1](https://arxiv.org/abs/2605.26535v1) — Recursive Flow Matching: A Wall-Bouncing Pendulum Perspective
**Date:** 2026-06, distilled 2026-06
**Related Research:** 034 (D2F Discrete Diffusion Forcing), 073 (LT2 Looped Transformers), 091 (SpecHop Multi-Hop), 131 (DiffusionBlocks)
**Related Plans:** 066 (D2F), 108 (LT2 Pipeline), 131 (SpecHop), 136 (Training-Free Loop Wrapper), 168 (RecFM Recursive Consistency — proposed)
**Domain:** katgpt-rs (open, general-purpose inference infrastructure)

---

## TL;DR

RecFM observes that a single interpolated state x_t lies on infinitely many trajectories indexed by scale α. Vanilla flow matching exploits only one. RecFM uses every (τ, α) pair as independent supervision for the same directional quantity at the same spatial point. The cross-scale consistency constraint directly reduces trajectory acceleration ∂t_v, tightening Euler discretization error bound.

We extract the structural principle — **cross-scale consistency at shared spatial points reduces truncation error** — and apply it modellessly to DDTree branch pruning, D2F denoising, LT2 sub-stepping, and SpecHop multi-hop speculation. No training required.

**Verdict: 🟢 GAIN — Three modelless improvements (DDTree, LT2, SpecHop) ship immediately. D2F recursive denoising deferred pending benchmark (2× forward passes per step).**

---

## 1. Paper Core (Distilled in Our Terms)

### 1.1 The Key Insight

A point x_t on the interpolation path between x_0 (noise) and x_1 (data) is not unique to one trajectory. It lies on a family of trajectories, each indexed by a different scale parameter α. Vanilla flow matching picks one trajectory and optimizes along it. RecFM uses all of them.

**Physics intuition**: wall-bouncing pendulum. After each bounce, velocity attenuates: v^(i+1) = α · v^(i). This creates a family of trajectories that share spatial points but traverse them at different speeds.

### 1.2 Loss Function

```text
L_total = Σ_i ||v̂^(i) - α^(i)·v*||² + λ · Σ_i ||v̂^(i) - α^(i)·v̂^(1)||²
```

Where v̂^(i) = v_θ(x_t, τ^(i), α^(i)) is the predicted velocity at scale i.

The first term matches each scale's velocity to its ground truth. The second term enforces cross-scale consistency: predicted velocities at different scales should be related by the scale factor.

### 1.3 Key Results

| Metric | RecFM | Vanilla FM | Improvement |
|--------|-------|------------|-------------|
| Inference speed | 20× vs diffusion emulators | Baseline | 20× speedup |
| MSE | 15% lower | Baseline | 15% reduction |
| Inference steps | 1-2 | 5+ | 3-5× fewer steps |
| Optimal depth D | 2 (primary + 1 secondary) | N/A | Minimal overhead |

### 1.4 Why It Works (Theorem 3.1)

RecFM's cross-scale consistency constraint directly reduces ∂t_v (temporal acceleration along the trajectory). Lower acceleration means the Euler discretization error bound is tighter:

```text
Euler error ≤ C · Δt² · sup ||∂t_v||
```

By constraining ∂t_v through cross-scale supervision, the ODE integration becomes more accurate at the same step count — or equivalently, achieves the same accuracy with fewer steps.

---

## 2. Fusion Ideas for katgpt-rs (Modelless Extractions)

### Fusion Idea 1: Recursive DDTree — Cross-Scale Branch Consistency

**What**: Apply RecFM's cross-scale consistency to DDTree branch expansion.

Current DDTree: each rollout independently explores marginal space. Branches that share a prefix have no mutual constraint.

RecFM fusion: When two branches share prefix up to depth d, their remaining marginals should satisfy a velocity-scaling constraint. Concretely:

```text
For branch i with marginal p_i at depth d, and branch j with marginal p_j at the same depth:
  v_i = p_i[d+1] - p_i[d]   (discrete velocity at depth d)
  Require: v_j ≈ α · v_i     for some scale factor α
```

This is purely modelless — it constrains the DDTree search space without any training. The consistency constraint prunes branches that violate cross-scale coherence, similar to how RecFM prunes high-acceleration trajectories.

**Why it's not a direct mapping**: RecFM operates on continuous ODE trajectories. We adapt the structural principle (cross-scale consistency at shared spatial points) to discrete marginal distributions in DDTree. The "trajectory" is the sequence of marginal distributions at increasing DDTree depths.

**Alignment with optimization.md**:
- O(1) consistency check per branch pair (fixed-size array comparison)
- Zero allocation: reuse existing `parent_path` bitfield
- Pre-compute scale factors once per depth level

**Verdict by 003 strategy**: MIT open-source engine. Pure modelless inference improvement — strengthens the engine without touching lora.bin. Direct gain, ship it.

---

### Fusion Idea 2: Recursive D2F Denoising — Multi-Scale Velocity Consistency

**What**: Add RecFM's secondary trajectory supervision to D2F denoising iterations.

Current D2F (Plan 066): iterative denoising with block-causal attention. Each step predicts clean tokens from noisy input. No cross-step consistency constraint.

RecFM fusion: During denoising step t, also predict what the clean tokens would be at "time" τ = t/α (a rescaled denoising step). The predicted velocity (change in token distribution) should satisfy:

```text
v_denoise(τ, α) ≈ α · v_denoise(t, 1)
```

Concretely: run two forward passes per denoising step — one at the actual noise level, one at a scaled noise level. Enforce that their predicted clean distributions are related by the scale factor.

This is the most direct analogue because:
1. D2F already has a noise schedule (Plan 066) — this adds a secondary schedule at scale α
2. The denoising "trajectory" is the sequence of noise levels → clean tokens
3. RecFM's Theorem 3.1 directly applies: reducing ∂t_v (trajectory acceleration in denoising space) tightens the Euler error bound on the denoising ODE

**Why it's not a direct mapping**: RecFM trains a neural network with the consistency loss. We apply the consistency as a modelless inference-time constraint: check if the two forward passes agree, and if they don't, blend them using the scale factor.

**Alignment with optimization.md**:
- Two forward passes share the same KV cache (same input, different noise schedule)
- Pre-allocate velocity buffers once, reuse across steps
- SIMD-friendly: element-wise comparison of velocity vectors

**Verdict by 003 strategy**: Modelless improvement to D2F, which is part of the MIT engine. Strengthens tri_mode inference quality without training changes. However, 2× forward passes per step is a real cost. **Benchmark first before committing.**

---

### Fusion Idea 3: Recursive LT2 ODE Refinement — Acceleration-Bounded Sub-Stepping

**What**: Use RecFM's acceleration constraint to tighten LT2's damped Euler sub-steps.

Current LT2 (Plan 136): damped Euler sub-stepping with anchor blend. The damping factor (1/K) provides stability but doesn't guarantee trajectory curvature reduction.

RecFM fusion: RecFM's Theorem 3.1 proves that minimizing ∂t_v (temporal acceleration) directly reduces Euler discretization error. Apply this to LT2:

After each sub-step, compute the acceleration a = ∂t_v ≈ (v_{k+1} - v_k) / Δt. If ||a|| exceeds a threshold, apply additional damping. This is the discrete analogue of RecFM's consistency loss, which constrains ∂t_v.

Concretely:

```rust
// After damped Euler sub-step
let v_k = y_k - x_prev;  // velocity at step k
let v_k1 = y_k1 - x;     // velocity at step k+1
let accel = (v_k1 - v_k); // approximate acceleration
let accel_norm = simd_l2_norm(accel);
if accel_norm > accel_threshold {
    // Extra damping proportional to excess acceleration
    let extra_damp = accel_threshold / accel_norm;
    simd_scale_inplace(&mut x, extra_damp);
}
```

**Why it's not a direct mapping**: LT2 is not a flow matching system — it's a training-free loop wrapper. But the mathematical principle (constraining trajectory acceleration improves Euler integration) is universal. We're extracting the fundamental insight, not the specific algorithm.

**Alignment with optimization.md**:
- SIMD l2 norm already available
- No allocation: operate on existing residual buffers
- O(1) per sub-step: one norm computation + conditional scale

**Verdict by 003 strategy**: Pure modelless improvement to LT2, which is already GOAT-proven. Strengthens inference quality at zero training cost. Ship it.

---

### Fusion Idea 4: Recursive SpecHop — Cross-Hop Velocity Consistency

**What**: Apply RecFM's cross-scale consistency to SpecHop's multi-hop speculation.

Current SpecHop (Plan 131): speculates on observations ahead of time, verifies when target returns. Each hop is independent.

RecFM fusion: When speculating on hop k+1, check consistency with hop k's speculated observation. If the "velocity" of observation change (string diff or embedding distance) violates a scaling constraint, reduce confidence.

This is the most creative fusion: treating the sequence of speculated observations as a "trajectory" and applying cross-scale consistency. Hops that are mutually consistent get higher confidence; contradictory hops get penalized.

**Why it's not a direct mapping**: SpecHop operates on discrete observations (strings), not continuous vector fields. We compute observation "velocity" as the diff between consecutive observations and enforce that hops deeper in the chain show diminishing changes (consistent with converging toward a fixed point).

**Alignment with optimization.md**:
- Observation velocity is a scalar comparison (token overlap ratio) — O(1) per hop
- No allocation: operate on existing observation buffers
- Confidence adjustment is a scalar multiply — trivial cost

**Verdict by 003 strategy**: Speculative pipeline improvement, MIT engine. Strengthens spec quality without training. **Keep gated** — confidence calibration is subtle and needs real-world validation.

---

## 3. GOAT Verdict

| Fusion Idea | Target | Gain Mechanism | Perf Impact | Verdict |
|-------------|--------|----------------|-------------|---------|
| Recursive DDTree | DDTree | Branch consistency pruning | Negligible (bitfield ops) | ✅ Ship modelless |
| Recursive D2F | D2F denoising | Cross-scale velocity blend | 2× forward passes per step | ⚠️ Benchmark first |
| Recursive LT2 | LT2 sub-steps | Acceleration-bounded damping | Negligible (SIMD norm) | ✅ Ship modelless |
| Recursive SpecHop | SpecHop pipeline | Cross-hop consistency scoring | Negligible (string diff) | ⚠️ Keep gated |

---

## 4. What NOT to Apply

**RecFM's neural network training** (the v_θ network with α conditioning) is model-based. That belongs in riir-ai, not katgpt-rs. The modelless extractions above capture the mathematical principle without the training requirement.

**The paper's image generation results** (Appendix I) are irrelevant to our domain.

**The paper's physics-specific evaluation** (PDE residuals, kinetic energy) validates the framework but doesn't directly apply — we're not solving PDEs. We extract the structural principle (cross-scale consistency reduces truncation error) and apply it to our discrete systems.

---

## 5. Open Questions

1. **D2F cost-benefit**: Does the quality improvement from recursive denoising justify 2× forward passes? Need benchmark on real inference workloads.
2. **Scale factor selection**: What α values work best for DDTree branch consistency? Paper suggests α ∈ [0.3, 0.7] for continuous systems. Our discrete setting may differ.
3. **SpecHop convergence assumption**: Are speculated observations actually converging? If the domain has oscillatory behavior, the diminishing-velocity assumption breaks down.
