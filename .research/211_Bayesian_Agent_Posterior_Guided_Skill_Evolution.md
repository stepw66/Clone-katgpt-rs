# Research: Bayesian-Agent → Posterior-Guided Pruner Evolution

**Date:** 2026-06
**Source:** [Bayesian-Agent: Posterior-Guided Skill Evolution for LLM Agent Harnesses](https://arxiv.org/pdf/2606.08348)
**Status:** Distilled + Fusion Verdict

---

## Paper Summary

Bayesian-Agent treats reusable agent skills and SOPs as Bayesian hypotheses. Instead of LLM self-critique or raw success/failure counts, it:

1. **Records verified trajectory evidence** (external grader, not self-assessment)
2. **Maintains feature-conditioned categorical posterior** over each skill (context, failure mode, token bucket, turn bucket, latency bucket)
3. **Maps posterior → inspectable actions**: explore, patch, split, compress, retire
4. **Separates posterior audit from model-facing instructions** — LLM receives executable patches, not posterior numbers

### Key Results (deepseek-v4-flash)
| Benchmark | GA Baseline | BA-Inc (Repair) | Gain |
|-----------|-------------|-----------------|------|
| SOP-Bench | 80% | 95% | +15pp |
| Lifelong AgentBench | 90% | 100% | +10pp |
| RealFin-Bench | 45% | 65% | +20pp |

### Core Architecture
```
TrajectoryEvidence → SkillBelief → CategoricalBayesState → RewritePolicy → {explore,patch,split,compress,retire}
```

- **Beta-Bernoulli**: α/(α+β) success probability, conservative smoothing
- **Categorical Naive Bayes**: P(class) × ∏ P(feature=value|class), Laplace smoothing, log-space
- **Policy thresholds**: retire β≥4 & p<0.45, patch failure_count≥2, split ≥3 contexts & ≥4 obs, compress ≥3 obs & p≥0.72

---

## Distillation to katgpt-rs (Modelless)

### What Maps Directly

| Paper Concept | katgpt-rs Equivalent | Status |
|---------------|---------------------|--------|
| Skill hypothesis | `ConstraintPruner` arm in BanditPruner | ✅ Exists |
| Trajectory evidence | `PrunerMemory` ring buffer (Plan 192) | ✅ Exists |
| Patch action | AbsorbCompress (Q→hard block promotion) | ✅ Exists |
| Retire action | AbsorbCompress demote loser | ✅ Exists |
| Registry | `SkillCatalog` (papaya HashMap) | ✅ Exists |
| Safe exploration | `SafePhased` bandit (Plan 137) | ✅ GOAT |
| Feature-conditioned posterior | ❌ **Missing** | ❌ Gap |
| Split/Compress actions | ❌ **Missing** (no precision-gated triggers) | ❌ Gap |

### What's Novel: The Precision Gap

The paper's key innovation is **feature-conditioned uncertainty**, not just binary success/failure counts. Our `PrunerMemory` stores flat reward history. The paper's `CategoricalBayesState` stores per-feature likelihoods.

**This is exactly what BAKE (R209) was designed for.** BAKE's `[f32; 8]` precision vectors per embedding dimension = per-feature Bayesian certainty. The fusion:

```
PrunerMemory (flat log) + BAKE precision (per-dim certainty) = PosteriorGuidedPrunerMemory
```

---

## Creative Fusion: Posterior-Guided Pruner Evolution (PGPE)

### Core Idea

**Treat each `ConstraintPruner` arm as a Bayesian hypothesis with per-feature precision.** Use BAKE's precision vectors to drive the five lifecycle actions. This goes beyond the paper by:

1. **Precision-gated actions** (not just threshold counts) — surprise triggers PATCH when posterior shift exceeds precision budget
2. **Continuous latent features** (not just discrete buckets) — BAKE operates in continuous embedding space, paper uses discrete buckets
3. **SIGMOID gating** (not softmax, per project rules) — sigmoid(precision × surprise) as action trigger
4. **Zero-alloc hot path** — all posterior math in fixed-size arrays, SIMD-friendly

### Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         PosteriorGuidedPruner            │
                    │                                         │
  Evidence ──────►│  TrajectoryEvidence (verified outcome)    │
  (verifier)       │         │                               │
                    │         ▼                               │
                    │  BAKE Precision Update                  │
                    │  precision_new = precision_old + obs    │
                    │  posterior = μ × (precision / total)    │
                    │         │                               │
                    │         ▼                               │
                    │  SurpriseComputer                       │
                    │  surprise = |posterior - prior| × λ     │
                    │  gate = sigmoid(surprise)               │
                    │         │                               │
                    │         ▼                               │
                    │  PrecisionPolicy                        │
                    │  ┌─────────────────────────────────┐   │
                    │  │ if gate > 0.7 && fail_count≥2   │───► PATCH
                    │  │ if precision_diverges(skills)    │───► SPLIT
                    │  │ if precision > threshold         │───► COMPRESS
                    │  │ if precision → 0                 │───► RETIRE
                    │  │ else                             │───► EXPLORE
                    │  └─────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
```

### Novel Contributions vs Paper

| Aspect | Paper (Python) | Our Fusion (Rust) |
|--------|---------------|-------------------|
| Posterior model | Categorical Naive Bayes (discrete) | BAKE precision vectors (continuous) |
| Action trigger | Fixed thresholds on counts | Sigmoid-gated surprise |
| Feature space | Pre-bucketed (6 buckets) | Continuous embedding dimensions |
| Memory | Ring buffer (100 items) | PrunerMemory + BAKE precision (fixed-size) |
| Decision | Ordered priority rules | Precision-weighted posterior policy |
| Hot path | N/A (Python) | Zero-alloc, SIMD, feature-gated |
| Cross-harness | Adapter protocol | ConstraintPruner trait + WASM boundary |

### Fusion: 3 Novel Ideas Beyond Direct Mapping

#### 1. **Precision-Gated AbsorbCompress** (replaces Q-threshold)
Current AbsorbCompress uses `q_threshold=0.05` and `min_visits=200`. Replace with:
- `absorb` when `precision > λ_absorb && surprise < ε` (certain and stable)
- `compress` when `precision > λ_compress` across merged arms (merge only if both certain)
- This fixes Plan 192's failure: bomber Q-values clustered at -0.09 ± 0.02 because visits were too low to differentiate. Precision tracking says "we don't know yet" instead of "they're all equally bad."

#### 2. **Surprise-Triggered Patch Generation** (replaces failure_count≥2)
Instead of "same failure mode appears twice → patch," use:
- `surprise = KL(posterior || prior)` per dimension
- `patch_trigger = sigmoid(β × surprise)` where β is configurable sensitivity
- This catches novel failure modes faster (high surprise from single observation) while ignoring noisy single failures (low surprise if prior was already uncertain)

#### 3. **Precision-Gated Safe Exploration** (upgrades PrudentBanker)
Current SafePhased uses phase-gap for α escalation. Replace with:
- `α = sigmoid(λ × (precision_skill - precision_threshold))`
- High-precision skills get aggressive exploration (we know they work)
- Low-precision skills stay conservative (we don't know yet)
- This directly addresses the paper's "full mode is not uniformly better" finding — precision tells you WHEN to be aggressive

---

## Verdict

### GOAT/Gain Analysis

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| **Novelty** | HIGH | Precision-gated lifecycle + surprise triggers + continuous posterior = beyond paper |
| **Feasibility** | HIGH | All primitives exist (BAKE R209 design, PrunerMemory P192, SafePhased P137) |
| **Impact** | HIGH | Fixes P192's bomber failure (sparse Q-values), upgrades bandit precision |
| **Modelless** | YES | Pure inference-time, no LLM training, just posterior math on evidence |
| **Commercial** | YES | Fits engine layer (MIT), precision-gated pruners are fuel (competitive moat) |
| **Performance** | HIGH | Fixed-size arrays, SIMD, zero-alloc — paper is Python, we're Rust |

### Decision: **GAIN — Implement as modelless feature**

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:
- **Engine layer (MIT)**: `PosteriorGuidedPruner` trait + `PrecisionPolicy` + surprise computation
- **Fuel layer (competitive)**: Precision vectors per pruner domain (game-specific, not shipped)
- **The architecture is proven** (BAKE R209 design complete, PrunerMemory exists, SafePhased GOAT)
- **The gap is implementation, not research**

### Why This Is Better Than Direct Paper Mapping

1. The paper uses Python with discrete buckets — we use Rust with continuous precision vectors
2. The paper treats skills as text SOPs — we treat pruners as compiled hypotheses with WASM test gates
3. The paper's lifecycle is post-hoc (collect evidence then decide) — ours is online (every inference updates precision)
4. The paper separates audit from instructions — we separate precision (audit) from pruner code (execution)

---

## Relationship to Existing Work

| Existing | Relationship |
|----------|-------------|
| R209 (BAKE katgpt-rs) | **Provides precision primitive** — PGPE consumes BAKE precision vectors |
| R093 (BAKE riir-ai) | **Provides training pipeline** — trains precision-equipped skills |
| R172/P192 (MUSE/ITSE) | **Provides lifecycle infra** — PGPE upgrades Q-threshold to precision-gated |
| P137 (PrudentBanker) | **Provides safe exploration** — PGPE gates α on precision |
| R062 (SHINE) | **Future: skill creation** — SHINE generates new skills from precision-aware context |
| R059 (MUSE Game Validators) | **riir-ai analog** — per-validator precision in game domain |

---

## TL;DR

**Bayesian-Agent's posterior-guided skill evolution = BAKE precision + MUSE lifecycle + PrudentBanker safe exploration.** The paper proves the concept (80%→95% SOP-Bench, 45%→65% RealFin). We have all the primitives designed but not connected. The fusion goes beyond the paper by using continuous precision vectors instead of discrete buckets, sigmoid-gated surprise instead of fixed thresholds, and zero-alloc Rust instead of Python. **Verdict: GAIN. Implement as `PosteriorGuidedPruner` behind `posterior_evolution` feature gate.**
