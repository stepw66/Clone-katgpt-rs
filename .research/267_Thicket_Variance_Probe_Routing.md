# Research 267: Thicket Variance Probe (TVP) — Decoding-Space Density as Substrate Routing Signal

**Date:** 2026-06-14
**Status:** Active — GOAT/gain verdict below
**Source:** arXiv:2603.12228 — *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* (Gan & Isola, MIT CSAIL, Mar 2026)
**Predecessor:** Research 081 (direct weight-space RandOpt mapping, Plan 121 complete)
**Placement:** Public `katgpt-rs/.research/` — generic inference framework mechanics (per Strategy 003)

---

## TL;DR

RandOpt (Neural Thickets) shows that after sufficient pretraining, the **weight neighborhood** of a model is dense with task-specialists — random Gaussian perturbations land on good solutions with measurable probability δ(m), and the set of specialists is diverse (spectral discordance D). Plan 121 already mapped this **directly** into a weight-space bandit (`src/pruners/randopt.rs`, synthetic-only).

This research proposes the **creative fusion** the user asked for: instead of re-implementing weight-space RandOpt, we lift the paper's *fundamental insight* — **variance structure of perturbation probes reveals loss-landscape geometry** — out of weight-space and into **decoding-config-space**. We sample K cheap probes that perturb decoding knobs (temperature, top-p, drafter seed, KV quantization noise, substrate mask bit-flips), measure the disagreement of their outputs, and feed that disagreement as a *new routing signal* into the existing `InferenceRouter` (CPU↔GPU↔ANE) and `S2FCollapseDetector` (CoT budget expand/contract).

**Name:** Thicket Variance Probe (TVP) — distinct from `parallel_probe.rs` (Plan 133), which probes same-config DDTree branches.

**Verdict:** ✅ **GAIN** — GOAT-gated, opt-in until proven. Distinct from all 7 existing router signals (trust, RV, critical_entropy, lodestar, breakeven, modality, QPS). Public framework primitive; private riir-ai side (LoRA thicket distillation) handled separately.

---

## 1. Paper Summary — What Changed vs Plan 081

Plan 081/121 captured the paper's *algorithm* (sample N perturbations, score, take top-K, ensemble via majority vote, O(1) wall-clock). The published version (v1, 30pp) adds three things 081 did not absorb:

### 1.1 Scaling law for solution density
$$\delta(m) = \mathbb{P}_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}[s(\theta + \epsilon) \geq s(\theta) + m]$$
δ(m) **increases monotonically** with model size (Fig 3a). For Qwen-0.5B → 32B on GSM8K: fraction of perturbations matching/exceeding base grows from 0% → 64%. **Implication:** δ(m) is a *quality barometer* — its value tells you what regime you're in (needle-in-haystack vs thicket vs plateau).

### 1.2 Spectral discordance D as specialist detector
$$D = 1 - \frac{1}{M(M-1)} \sum_{j \neq k} C_{jk}$$
where C is the Pearson correlation of per-task percentile ranks across perturbations. D→0 = generalists, D→1 = specialists. **D also scales with model size** (Fig 3b).

### 1.3 Format-vs-Reasoning Thicket Decomposition (Section 8)
On GSM8K (Qwen2.5-3B, N=3000, K=50), the +9.5pp accuracy gain decomposes as:
- Reasoning thicket: +12.3% (base wrong, perturbed correct — substantive)
- Format thicket: +19.0% (format fixed, then correct — cosmetic)
- Regression: −0.7%
- **Implication:** a chunk of "disagreement" between probes is *cosmetic* (format), not substantive (reasoning). A router that cannot tell them apart over-invests compute on format-only noise.

### 1.4 Distillation recovery (Section 7)
Distilling the K=50 ensemble into a single model via SFT on hard examples recovers ~98% of ensemble accuracy at ~2% of training cost. **Implication for model-based path:** the same recipe works in LoRA-space (paper Section 9.5: low-dim structure of fine-tuning).

---

## 2. The Creative Fusion — Not a Direct Mapping

Plan 121 already did the direct mapping. The user explicitly asks for **fundamental application, not direct mapping**. The fundamental insight of the paper is:

> **The variance structure of perturbation probes is a sufficient statistic for the local geometry of the loss landscape.**

The paper applies this in **weight-space** (perturb θ). But the same principle applies in any space where perturbations explore the local solution manifold:

| Space | Perturbation | What variance reveals |
|-------|--------------|----------------------|
| Weight-space (paper, Plan 121) | θ + σ·ε | density of task-experts near θ |
| **Decoding-config-space (TVP, this plan)** | **temperature, top-p, seed, KV-noise, mask bits** | **density of decoding-specialists for current query** |
| Prompt-space | paraphrase, suffix | density of prompt-experts |
| LoRA-space (riir-ai LTD) | A·B + σ·ε | density of adapter-experts near base LoRA |

TVP is the modelless decoding-space instance. We do not perturb weights (that requires training-context). We perturb **inference-time knobs** that are free to flip.

---

## 3. TVP Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│  Query arrives → cheap pre-decode phase                          │
│                                                                  │
│   1. Sample K probes (default K=4, max 8)                        │
│      Each probe perturbs decoding knobs:                         │
│        • temperature ∈ {T₀, T₀±ΔT}  (free)                       │
│        • top-p ∈ {p₀, p₀±Δp}        (free)                       │
│        • drafter seed = base+i       (free)                       │
│        • KV quantization noise σ_kv·ε (cheap, in-place)          │
│        • substrate mask bit-flips    (only if substrate_gate on) │
│                                                                  │
│   2. Each probe produces ONE token (or short N-token stub)       │
│      Run on CPU/SIMD only — probes must be cheap                 │
│                                                                  │
│   3. Compute variance signals from K probe outputs               │
│      • answer_disagreement: 1 − max_class_share (categorical)    │
│      • logit_kl: mean pairwise KL divergence (continuous)        │
│      • format_disagreement: token-form variance, answer-agnostic │
│      • reasoning_disagreement: answer-disagreement minus format  │
│                                                                  │
│   4. Emit TVP signal → InferenceRouter and CollapseDetector      │
│      • High reasoning_disagreement → promote CPU→GPU/ANE         │
│                                   → expand CoT budget            │
│      • High format_disagreement only → stay on CPU, canonicalize │
│      • Low disagreement → stay on CPU, no extra CoT              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Why this is distinct from existing router signals

The `InferenceRouter` already consumes 7 signals. TVP is signal #8:

| # | Signal | Source | What it measures | Granularity |
|---|--------|--------|------------------|-------------|
| 1 | QPS / p99 | TriggerGate | Load | system |
| 2 | trust | Plan 182 verifier | Speculative accept rate | per-decode |
| 3 | RV (acceptance variance) | Plan 202 | Variance of accept/reject over time | EMA stream |
| 4 | critical_entropy | Plan 222 | Marginal entropy > H_critical | per-token |
| 5 | lodestar | Plan 207 | Completion distance | per-query |
| 6 | breakeven | Plan 250 | Cost-amortization N* | per-tier-pair |
| 7 | modality | Plan 227 | Query classifier | per-query |
| **8** | **TVP (this plan)** | **K perturbed probes** | **Loss-landscape geometry proxy via probe-output disagreement** | **per-query** |

**Key distinction from RV (#3):** RV is the variance of a *boolean stream* (accept/reject) over many decodes. TVP is the variance of *K probe outputs within one query*. RV says "how stable has speculation been lately"; TVP says "how ambiguous is the model about *this specific query*". Different statistical object, different actuator.

**Key distinction from critical_entropy (#4):** Entropy is computed on the *base model's* next-token distribution. TVP samples *counterfactual* distributions under decoding perturbations. A flat entropy can hide thickets (multiple modes of equal mass); TVP exposes them.

### 3.2 Composition (not replacement)

TVP does not replace any existing signal. It composes in the same cascade as trust/RV/critical_entropy:

```
tier_after_trust    = trust_signal gate         (existing)
tier_after_rv       = rv_signal gate             (existing)
tier_after_critical = critical_entropy gate      (existing)
tier_after_tvp      = tvp_signal gate            (NEW)
tier_final          = breakeven amortization     (existing)
```

Order matters: TVP runs *before* breakeven so the cost-aware amortizer can veto a promotion that hasn't paid off yet.

---

## 4. Self-Learning Adaptive CoT (Constraint 4)

TVP feeds two adaptive loops, both modelless (no LLM training):

### 4.1 Substrate EMA
Per query-class (modality bucket), maintain EMA of `reasoning_disagreement` observed vs the *quality outcome* (did the final answer match a verifier / constraint?). If high disagreement correlates with high error → next time disagreement is high, promote more aggressively. Sigmoid-blended, never softmax (per project conventions).

### 4.2 Probe-count bandit
K itself is a bandit arm. Default K=4. If TVP signal at K=4 is ambiguous (disagreement near threshold), escalate to K=8 and re-measure. If TVP signal is decisive at K=2 (very low or very high disagreement), drop to K=2 next time for that query-class. This is the `BanditStrategy::RandOptAdaptive` from Plan 121, repurposed for *probe-count selection* instead of *arm selection*.

### 4.3 Threshold adaptation
The promote/demote thresholds (`tvp_promote_at`, `tvp_demote_at`) self-tune via the same EMA mechanism as `S2FCollapseDetector.threshold()`. No manual tuning.

---

## 5. CPU/GPU/ANE Auto-Route (Constraints 7, 9)

The router cascade already exists. TVP adds one more input:

```rust
// In InferenceRouter::forward, after tier_after_critical:
#[cfg(feature = "thicket_variance_probe")]
let tier_after_tvp = match self.tvp_signal {
    s if s.reasoning_disagreement > self.tvp_promote_at
        && tier_after_critical == ComputeTier::CpuOnly
        && self.gpu.is_some() =>
    {
        ComputeTier::CpuGpu  // thicket is sparse → invest compute
    }
    s if s.reasoning_disagreement < self.tvp_demote_at
        && tier_after_critical == ComputeTier::CpuGpu
        && self.gate.estimated_qps()
            < self.gate.config().gpu_activate_qps * self.gate.config().hysteresis_factor =>
    {
        ComputeTier::CpuOnly  // thicket is dense → cheap substrate suffices
    }
    _ => tier_after_critical,
};
```

**Threshold-adaptive** (constraint 9): `tvp_promote_at` and `tvp_demote_at` are not constants — they are sigmoid-blended functions of (a) recent error rate at each tier, (b) current QPS load, (c) breakeven N* for the tier pair. This means under heavy load, TVP requires *stronger* disagreement to promote (the bar goes up); under light load, even mild disagreement promotes.

**Plasma/hot/warm/cold/freeze path** (constraint 8):
- **Plasma (always-on hot path):** TVP signal extraction is O(K) with K≤8 — fits in hot path if K is small
- **Hot (default-on after GOAT):** probe execution + signal emission
- **Warm (opt-in):** substrate mask bit-flip perturbations (requires `substrate_gate`)
- **Cold (offline):** EMA threshold calibration, query-class statistics
- **Freeze:** `TvpSignalFrozen` (16 bytes, repr(C)) for persistence across restarts

---

## 6. Plasma/HOT/WARM/COLD/FREEZE Mapping (Constraint 8)

| Tier | TVP component | Cost | When |
|------|--------------|------|------|
| **Plasma** | `TvpSignal` struct (16 bytes), `reasoning_disagreement` field read | ~1ns | Every query, hot path |
| **Hot** | Probe launch + variance computation (O(K²) pairwise KL, K≤8 → ≤28 ops) | ~5-20μs for K=4 single-token probes | Every query when feature on |
| **Warm** | Substrate mask bit-flip perturbation source | ~1μs per probe | Only when `substrate_gate` enabled |
| **Cold** | Per-query-class EMA threshold adaptation, bandit arm updates for K | ~100ns, batched | Per query (amortized) |
| **Freeze** | `TvpSignalFrozen` magic+version+thresholds+EMA state, BLAKE3-hashed | disk I/O only | Restart, migration |

---

## 7. Tests / Examples — Before vs After (Constraint 6)

### 7.1 Before/after thinking vs non-thinking
The demo `examples/thicket_variance_probe_01_basic.rs` constructs two synthetic query populations:
- **Easy queries** (high solution density, low disagreement): TVP should keep them on CPU, no CoT
- **Hard queries** (low density, high disagreement): TVP should promote to GPU + expand CoT

Expected outcome:
| Metric | Non-thinking baseline | TVP-routed | Source |
|--------|----------------------|-----------|--------|
| Tokens on easy queries | 100% | 10-30% | TVP detects dense thicket → no CoT |
| Substrate on easy | CPU/GPU mixed | CPU only | TVP demotes |
| Substrate on hard | CPU only | GPU/ANE | TVP promotes |
| Accuracy on hard | baseline | +2-5pp | extra CoT + better substrate |
| Total wall-clock | baseline | ≤ baseline | cheap probes + saved CoT |

### 7.2 GOAT gate (must pass before default-on)
- **G1:** Probe overhead ≤ 30% of single-decode cost (K=4 single-token probes)
- **G2:** TVP-only routing ≥ no-routing baseline on a synthetic disagreement benchmark (no regression)
- **G3:** Zero overhead when feature disabled (`#[cfg]` gate, like all others)
- **G4:** Ablation vs RV (Plan 202): TVP+RV ≥ max(TVP-only, RV-only). If TVP is redundant with RV, demote to research-only (like DFlare, Plan 174).
- **G5:** Format-vs-reasoning decomposition correctness: synthetic format-only disagreement does NOT trigger promote; synthetic reasoning-only disagreement DOES trigger promote.
- **G6:** Freeze/thaw roundtrip preserves thresholds and EMA state.
- **G7:** Threshold self-adaptation: after N queries, promote threshold converges (std < 0.05).

---

## 8. GOAT/Gain Verdict

Per Strategy 003 decision rules:

### 8.1 WHAT vs HOW
- **WHAT (public):** "Inference-time K-probe disagreement as a substrate routing signal." Generic framework primitive. **Public OK** — analogous to trust/RV/critical_entropy, all of which are public in `katgpt-rs`.
- **HOW (private):** Exact perturbation magnitudes (ΔT, Δp, σ_kv), exact thresholds, query-class EMA configs. These stay in `riir-ai` deployment configs, not in the public engine.

### 8.2 Verdict
✅ **GAIN** — pursue as Plan 267.

**Why:**
1. Fills a real gap — no existing signal measures loss-landscape geometry via probe variance
2. Direct fusion of paper's δ(m)/D metrics (currently only diagnostics) into runtime routing
3. Composes cleanly with existing 7 signals — additive, not replacing
4. Modelless — no training, no LoRA, fits the engine-first mandate
5. Self-learning — EMA + bandit satisfy constraint 4 without crossing into training
6. CPU/GPU/ANE adaptive — directly extends existing router cascade (constraint 7, 9)
7. Paper's Section 8 decomposition (format vs reasoning) is novel and unexploited

**Risks (must address in plan):**
1. **Probe latency chicken-and-egg** — probes must run on cheap substrate (CPU single-token), main decode gets routed
2. **Probe cost vs signal value** — must prove K=4 probes save more compute than they spend (G1 gate)
3. **Double-counting with RV** — ablation gate G4; demote if redundant
4. **δ/D metric repurposing** — paper defined them for weight-space; need to validate (or substitute logit-KL) for decoding-space
5. **Naming collision** — use `thicket_variance_probe.rs`, not `parallel_probe.rs` (Plan 133 is different)

### 8.3 Priority
- **Modelless (katgpt-rs):** HIGH. Implement Plan 267.
- **Model-based (riir-ai):** MEDIUM. Separate plan for LoRA Thicket Distillation (LTD) — distill top-K LoRA perturbations into single LoRA via SFT on hard examples. Public research describes WHAT; private plan holds HOW (hyperparameters).

### 8.4 What NOT to do
- Do NOT replace Plan 121's weight-space RandOpt — it's complete and complementary
- Do NOT perturb actual model weights at inference (too expensive without training context)
- Do NOT use softmax anywhere — sigmoid-blended thresholds per project conventions
- Do NOT promote to default-on until G1-G7 all pass (follow DFlare Plan 174 pattern: GOAT-failed → research-only)

---

## 9. Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| **121 RandOpt weight-space** | Predecessor. TVP is the decoding-space analog. Shares δ/D metrics via `bandit::solution_density()` and `bandit::spectral_discordance()`. |
| **133 parallel_probe** | Different. Parallel-Probe probes *same-config* DDTree branches (consensus vote prunes branches). TVP probes *perturbed-config* single tokens (variance routes substrate). |
| **202 RV (acceptance_variance)** | Closest analog. RV = boolean-stream variance over time. TVP = multi-probe-output variance within one query. Ablation gate G4 checks redundancy. |
| **212 CollapseDetector** | Compose. TVP's `reasoning_disagreement` is the *inverse* signal of CollapseDetector's `hesitation_count`. High disagreement → expand CoT; high hesitation → contract CoT. |
| **216 SubstrateGate** | Compose. TVP's format-vs-reasoning decomposition decides whether to route by capability (substrate) or by compute (tier). |
| **222 critical_entropy** | Compose. Entropy is base-model marginal; TVP is perturbed-model disagreement. Flat entropy can hide thickets; TVP exposes them. |
| **250 breakeven** | Compose. TVP runs *before* breakeven so cost-amortization can veto promotions. |
| **174 DFlare** | Precedent for GOAT-fail → research-only demotion. TVP follows same gate discipline. |

---

## 10. Architecture Integration

### 10.1 Feature gate
```toml
thicket_variance_probe = ["rv_gated_routing"]  # composes with RV for ablation
```
Opt-in until G1-G7 pass. Not in `default` until GOAT proved.

### 10.2 Module structure
```
src/pruners/
  thicket_variance_probe.rs   # TvpSignal, TvpConfig, TvpProbeSource, TvpAggregator
                              # TvpSignalFrozen, freeze/thaw
  bandit.rs                   # existing solution_density(), spectral_discordance()
  acceptance_variance.rs      # existing RV — TVP composes here for ablation
  collapse_detector.rs        # existing — TVP feeds disagreement signal

src/
  inference_router.rs         # add tvp_signal field + tier_after_tvp gate
```

### 10.3 Key types (public — generic framework)
```rust
pub struct TvpConfig {
    pub probe_count: u8,            // K, default 4, max 8
    pub temperature_delta: f32,     // ΔT, default 0.1
    pub top_p_delta: f32,           // Δp, default 0.05
    pub kv_noise_sigma: f32,        // σ_kv, default 0.0 (off)
    pub mask_flip_count: u8,        // substrate mask bit-flips, default 0
    pub promote_at: f32,            // disagreement threshold to promote, default 0.6
    pub demote_at: f32,             // disagreement threshold to demote, default 0.2
    pub ema_alpha: f32,             // threshold adaptation rate, default 0.05
}

pub struct TvpSignal {
    pub reasoning_disagreement: f32,   // [0,1], high = needle regime
    pub format_disagreement: f32,      // [0,1], high = cosmetic only
    pub logit_kl: f32,                 // mean pairwise KL across probes
    pub probe_count_used: u8,          // actual K used (bandit may reduce)
}

pub trait TvpProbeSource: Send + Sync {
    /// Run one probe with perturbation `arm` and return its top-1 token id + logits.
    fn probe(&self, arm: u8) -> ProbeOutput;
}

pub struct ProbeOutput {
    pub token_id: u32,
    pub logits: Vec<f32>,        // top-K only, capped at 32 for O(K²) KL
    pub format_hash: u64,        // canonical form hash (format-only disagreement)
}
```

### 10.4 What stays private (riir-ai)
- Exact `promote_at`/`demote_at` values per game domain
- Query-class EMA tuning curves
- LoRA-space thicket distillation recipe (Section 7 of paper, adapted to LoRA)
- Trained substrate masks for bit-flip perturbation source

---

## 11. References

- Paper: https://arxiv.org/pdf/2603.12228
- Project page: https://thickets.mit.edu
- Upstream code: https://github.com/sunrainyg/RandOpt (in `.raw/RandOpt/`)
- Predecessor: `.research/081_RandOpt_Neural_Thickets_Random_Weight_Perturbation.md`
- Predecessor plan: `.plans/121_randopt_weight_perturbation.md` (complete)
- Strategy: `.research/003_Commercial_Open_Source_Strategy_Verdict.md`
- Related signals: Plans 182 (trust), 202 (RV), 222 (critical_entropy), 250 (breakeven)

---

## TL;DR

Plan 121 did the direct weight-space RandOpt mapping. This research proposes the **creative fusion**: lift the paper's fundamental insight (probe-variance reveals loss-landscape geometry) from weight-space to **decoding-config-space**, and feed it as signal #8 into the existing InferenceRouter. Modelless, composable, self-learning, CPU/GPU/ANE adaptive. GOAT-gated with 7 criteria including an ablation vs RV to avoid double-counting. **Verdict: GAIN — pursue as Plan 267.**
