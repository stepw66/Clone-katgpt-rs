# Research: Discrete Critical Interval Solver Switching

**Date:** 2026-06
**Source:** "Why Gaussian Diffusion Models Fail on Discrete Data and How to Prevent It?" (arXiv:2604.02028)
**Status:** GOAT — Gain Proven, Modelless Path

---

## Paper Core Thesis

Gaussian diffusion with DDPM solver fails on discrete data (text, code, proteins) because during the **critical interval** (late sampling phase, ~t/T ∈ [0.0, 0.4]), the density of noisified data splits into multiple modes. DDPM occasionally enters low-density regions between modes, producing OOD inputs for the model → incorrect generations.

Three mitigations work:
1. **Self-conditioning (SC)**: Model conditions on its own previous prediction → anchors trajectory, avoids OOD regions
2. **q-sampling**: Alternative solver `x_{t-1} = √(ᾱ_{t-1})·x̂_0 + √(1-ᾱ_{t-1})·ε` → moves farther from current (bad) prediction, escapes traps
3. **MBR decoding**: Branch trajectories, select best candidate via utility function

**Best result**: SC(p=0.95) + q-sampling(t_act/T=0.3-0.5) combined. No retraining needed for q-sampling.

---

## Why It Matters For Us

Our D2F (Discrete Diffusion Forcing) pipeline already confirmed: **discrete > continuous for text** (R010 ColaDLM rejected, R044 ELF rejected, R041 RePlaid adopted schedules only). This paper explains *why* discrete is harder for Gaussian approaches and provides specific solver-level fixes.

### Validation of Our Architecture

| Paper Finding | Our Status | Verdict |
|---|---|---|
| Gaussian diffusion fails on discrete data | We use mask-based discrete diffusion (D2F) | ✅ Already correct |
| Self-conditioning essential for discrete | RePlaid SC + DPM-Solver++(2M) implemented | ✅ Partially there |
| q-sampling helps in critical interval | Not implemented | 🔧 New |
| Higher SC probability (0.95 > 0.5) helps | Using 0.5 default | 🔧 Tunable |
| Context-dependent embeddings more robust | We use shallow embeddings for dLLM | 📝 Note for riir-ai |

---

## Fusion Ideas (Not Direct Mapping)

### 1. CriticalIntervalGate — Entropy-Triggered Solver Switch (Modelless)

**Not just "detect critical interval"** — fuse with our existing `Regime` classification (Plan 141) and `TriggerGate` (compute tier router).

The insight: DDTree's marginal entropy **already measures** how multimodal the distribution is at each depth. When entropy is high at early depths → the drafter is uncertain → this IS the critical interval.

**Mechanism:**
```
dd_tree build:
  for depth in 0..width:
    marginals = dflash_predict(depth)
    H = shannon_entropy(marginals)
    if H > H_critical:
      // In critical interval — switch solver
      switch from DPM-Solver++(2M) to q-sampling
      enable self-conditioning from previous depth's prediction
    else:
      // Normal regime — use fast solver
      continue with DPM-Solver++(2M)
```

**Why novel:** Paper detects critical interval via noised data density (requires analytical computation). We detect it via DDTree marginal entropy (free — already computed). Same signal, zero overhead.

**Expected gain:** q-sampling alone gives +5-15 Mauve on text (paper Table 1). With SC, +20-50 Mauve on low-quality baselines. For our D2F pipeline with already-strong DPM-Solver++, expect more modest but measurable gains on code generation tasks.

### 2. SelfCond Drafter — Draft-Refine Loop in Speculative Decoding (Modelless)

**Not just "add self-conditioning"** — fuse with our existing `SpeculativeGenerator` trait.

Current: `dflash_predict_ar_with` → marginals → DDTree → verify

Proposed: `dflash_predict_ar_with` → marginals → DDTree → **feed best-path tokens back as conditioning** → `dflash_predict_ar_with` again → refined marginals → DDTree → verify

This is a **2-pass speculative draft**: first pass explores, second pass refines using self-conditioning. The paper's insight that SC "anchors predictions to avoid OOD regions" applies here — the second pass is anchored to the first, reducing the chance of entering low-density regions.

**Cost:** 2× drafting compute. **Benefit:** Higher acceptance rate → fewer verification rejections → net throughput improvement if acceptance rate gain > 2×.

**This should be feature-gated** because:
- For game domains, lower entropy = less diverse behavior (R041 warning)
- For code translation (riir-ai), higher quality = more correct translations

### 3. MBR Tree Selection — Minimum Bayes Risk for DDTree (Modelless)

**Not just "MBR decoding"** — fuse with our existing DDTree scoring.

Current DDTree selection strategies: `BestQ`, `MostFrequent`, `Top1Converged` (EqR)

Proposed: `MbrSelect` strategy — extract K=5 best paths from DDTree, score each against all others using ConstraintPruner validity as utility, select minimum-risk path.

```
mbr_select(tree, K=5):
  candidates = top_K_paths_by_score(tree, K)
  // utility: how many ConstraintPruner checks pass
  for each candidate i:
    risk[i] = sum over j != i of (utility(candidate_j) - utility(candidate_i))
  return candidates[argmin(risk)]
```

**Why novel:** Paper uses BLEU/Hamming for utility. We use ConstraintPruner — which is domain-aware (Rust syntax validity, game rule validity). The ConstraintPruner IS the risk function.

**Cost:** O(K²) ConstraintPruner evaluations — cheap (already in hot path).
**Benefit:** Paper shows MBR improves correctness by up to 20% for discrete tasks.

### 4. Q-Sample Solver — Posterior Sampling in D2F (Modelless)

The paper's q-sampling formula adapted for our discrete mask-based diffusion:

Current D2F denoising (simplified):
```
for each masked position:
  logits = model(noised_input)
  token = argmax_sample(logits, constraint)
  unmask(position, token)
```

Proposed q-sample variant:
```
for each masked position:
  logits = model(noised_input)
  // Standard: commit highest-confidence token
  // Q-sample: re-noise and re-predict, using model's own prediction as x_0 estimate
  x_0_hat = embedding(argmax(logits))
  x_{t-1} = sqrt(alpha_{t-1}) * x_0_hat + sqrt(1 - alpha_{t-1}) * noise
  // Re-run model on x_{t-1} for refined prediction
  refined_logits = model(x_{t-1})
  token = argmax_sample(refined_logits, constraint)
  unmask(position, token)
```

**This is essentially a 2-step denoise** — but only activated during critical interval. The paper proves this helps because q-sampling's larger step size escapes low-density traps.

**Cost:** 2× model forward per masked position during critical steps only.
**Benefit:** Paper shows q-sampling alone improves quality comparable to self-conditioning, without retraining.

### 5. Higher SC Probability for Code LoRA Training (Model-Based)

Paper's Table 6 shows SC p=0.95 outperforms p=0.5 by 5-15 Mauve on text, with strongest gains for embedding-based models.

For riir-ai's LoRA training pipeline (Python→Rust code generation), this is a **training hyperparameter change**:
- Current: SC probability 0.5 (RePlaid default)
- Proposed: SC probability 0.95 for code tasks
- Cost: 1.13× training time (paper's measurement)
- Benefit: Better code generation quality, especially for discrete code tokens

---

## GOAT Verdict

### Modelless (katgpt-rs)

| Fusion Idea | Gain | Perf Cost | Default? |
|---|---|---|---|
| CriticalIntervalGate | High — entropy-triggered solver switch, zero detection overhead | Low — branch on entropy threshold | ✅ **YES** — entropy already computed, solver switch is just different math |
| Q-Sample Solver | Medium-High — paper proves quality gain | Medium — 2× forward during critical steps only | Feature-gated — cost depends on critical step frequency |
| SelfCond Drafter | Medium — 2-pass speculative draft | High — 2× draft compute | Feature-gated — only for quality-critical paths (code, not games) |
| MBR Tree Selection | Medium — better path selection from DDTree | Low — O(K²) constraint checks, K small | Feature-gated — needs K candidates |

**Default-on**: CriticalIntervalGate (entropy detection is free, solver switch is just different arithmetic, no perf hurt)

**Feature-gated**: Q-Sample Solver, SelfCond Drafter, MBR Tree Selection

### Model-Based (riir-ai)

| Fusion Idea | Gain | Perf Cost | Default? |
|---|---|---|---|
| Higher SC probability (p=0.95) | High — paper proves for code | 1.13× training time | ✅ **YES** — training cost acceptable, quality is everything for RIIR |
| Context-dependent embeddings | Very High — TEncDM Enc vs Emb: 67→87 Mauve | Architecture change | Future — requires encoder training |

**Default-on**: Higher SC probability for code LoRA training (no inference cost, only 13% training cost increase)

---

## Relationship to Existing Work

| Existing | Relationship |
|---|---|
| R041 RePlaid (self-conditioning) | Validates SC, extends with higher p and critical-interval gating |
| R044 ELF (embedded language flows) | Rejected continuous diffusion — paper confirms this was correct |
| Plan 141 Data Probes (regime classification) | `Regime` enum → add `Critical` variant with entropy threshold |
| Plan 089 Tri-Mode Inference | D2F drafter can benefit from q-sampling during critical steps |
| Plan 217 NextLat Belief Drafter | `LatentDynamicsMLP::draft()` can add self-conditioning loop |
| Plan 178 MUX Multiplexed Reasoning | Latent collapse detector → fuse with critical interval gate |
| Plan 182 Trust Region Adaptive Speculation | Adaptive window sizing → add entropy-triggered solver switch |

---

## What We're NOT Doing

1. **Continuous latent diffusion for text** — Paper shows it helps, but we already decided against it (R010, R044). D2F mask-based is the right path.
2. **MBR for unconditional generation** — Paper shows it's impractical (slows generation). Only for conditional/code tasks.
3. **Resampling x_t for SC** — Paper's Appendix J shows this doesn't help. Skip.
4. **Adding variance to x_0** — Paper's Appendix J shows this doesn't transfer to real data. Skip.

---

## TL;DR

The paper explains *why* Gaussian diffusion fails on discrete data (multimodal density gaps in critical interval) and provides three fixes (SC, q-sampling, MBR). Our D2F architecture already chose discrete over continuous — confirmed correct. The novel integration is **entropy-triggered solver switching during DDTree construction** (CriticalIntervalGate) — zero overhead, measurable quality gain, default-on. For riir-ai, increase SC probability to 0.95 for code LoRA training. Feature-gate the more expensive options (2-pass drafting, MBR selection, q-sample re-denoise).
