# Research Verdict 41: RePlaid — Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

**Paper:** Continuous Diffusion Scales Competitively with Discrete Diffusion for Language (arXiv:2605.18530)
**Authors:** Zhihan Yang, Wei Guo, Shuibai Zhang, Subham Sekhar Sahoo, Yongxin Chen, Arash Vahdat, Morteza Mardani, John Thickstun (NVIDIA, Cornell, Georgia Tech, UW-Madison, MBZUAI-IFM)
**Date:** May 19, 2026
**Status:** 🟡 SELECTIVE ADOPTION — Theoretical framework adopted (variance-minimized schedules, ELBO regularization). Full continuous diffusion mechanism rejected (incompatible with DDTree speculative decoding, same verdict as Research 10 ColaDLM).

---

## 1. Core Premise

RePlaid revisits **Plaid** (Gulrajani & Hashimoto, NeurIPS 2023) — a likelihood-based continuous diffusion language model — and modernizes it to match the architecture of discrete DLMs (MDLM, Duo). Under this unified setting:

1. **Continuous diffusion scales on par with discrete diffusion** — 20× compute gap vs AR (vs MDLM's 14×, Duo's 22×)
2. **PPL 22.1 on OpenWebText** — SOTA among continuous DLMs, beating MDLM low-var (23.1), Duo (25.2), LangFlow (32.2)
3. **1.8× fewer parameters** than MDLM/Duo at compute-optimal frontier
4. **Over-training advantage** — beats MDLM when trained past compute-optimal (critical for small inference models)

The key enablers are: (a) ELBO training with learnable noise schedule, (b) learnable low-dim embeddings (d_e=16), (c) self-conditioning, (d) output prior logits.

---

## 2. Key Findings

### 2.1 Scaling Laws (Unified Protocol)

| Method | Compute Gap vs AR | PPL (OWT 1M) | Parameters vs MDLM |
|--------|-------------------|---------------|---------------------|
| AR | 1× | 17.5 | — |
| MDLM (low var) | 14× | 23.1 | baseline |
| Duo | 22× | 25.2 | ≈1× MDLM |
| **RePlaid (s.c.)** | **20×** | **22.1** | **1.8× fewer** |
| RePlaid (no s.c.) | 27× | 23.6 | 1.8× fewer |

First unified scaling comparison between continuous and discrete DLMs using Sahoo et al. protocol (SlimPajama, Llama-2 tokenizer, identical DiT architecture).

### 2.2 Ablation Impact (Table 2)

| Component Removed | PPL | Δ |
|---|---|---|
| Full RePlaid (s.c.) | 22.1 | — |
| w/o output prior | 22.5 | +0.4 |
| w/o self-conditioning | 23.6 | +1.5 |
| w/o learnable noise schedule | 24.4 | +2.3 |
| **w/o learnable embeddings** | **39.4** | **+17.3** |

**Learnable embeddings are the dominant factor** — without them, RePlaid becomes the worst DLM. This is the single most important finding for our stack.

### 2.3 Theoretical Results

**Proposition 1 (Constant Per-Timestep Diffusion Loss):** There exists a unique noise schedule γ* such that the per-timestep diffusion loss is constant across time. Training the noise schedule to minimize Monte-Carlo variance converges to this schedule.

**Lemma 3 (Linear Information Decay):** Under Bayes-optimal denoiser with optimal noise schedule, mutual information I(e; z_t) decays linearly in t: I(e; z_t) = I(e; z_0) - κt.

**Proposition 2 (Near-Linear CE):** Per-timestep cross-entropy decomposes into a linear-in-t trend plus a non-negative residual (conditional total correlation that increases in t). Empirically, CE is near-linear when the noise schedule is learned.

### 2.4 Embedding Geometry Discovery

ELBO-trained embeddings at d_e=16 show:
- **POS-structured clustering** (Fig 5a) — discovered without any POS labels
- **Low-rank structure** — 90% variance in 6 principal components (Fig 5b)
- Adding cross-entropy loss **destroys** this structure (Fig 5c) — PCA flattens, PPL degrades from 22.1→26.1

### 2.5 Sampling Results

- **DPM-Solver++(2M)** beats DDPM at low NFEs (< 64 steps)
- **DDPM** beats deterministic solvers at high NFEs (≥ 64 steps)
- RePlaid (no s.c.) outperforms Duo at every T on MAUVE
- Self-conditioning compounds confidence → lower entropy at high T

---

## 3. Relationship to Our Previous Research

### 3.1 vs Research 10 (ColaDLM) — This Paper is Stronger

| Aspect | ColaDLM (Research 10) | RePlaid (This Paper) |
|--------|----------------------|---------------------|
| Architecture | VAE + DiT (2B params) | Modernized Plaid (0.1B) |
| Latent space | d=16-128 VAE latents | d_e=16 learnable embeddings |
| Training | Flow matching (not likelihood) | ELBO (true likelihood bound) |
| PPL | Not reported (generative metrics only) | 22.1 (SOTA continuous) |
| Our verdict | Mechanism rejected | Theoretical framework adopted |
| Actionable insight | Block-causal generation | Variance-minimized schedules |

**Key difference:** ColaDLM required a VAE+DiT rewrite that was fundamentally incompatible with our stack. RePlaid's insights decompose into modular, adoptable pieces (schedules, self-conditioning, embedding geometry) that work within our existing architecture.

### 3.2 vs Research 34 (D2F Discrete Diffusion Forcing)

| Aspect | D2F (Research 34) | RePlaid |
|--------|-------------------|---------|
| Diffusion type | Discrete (mask tokens) | Continuous (Gaussian noise on embeddings) |
| Noise schedule | Fixed monotonic (min→max ratio) | Learned (variance-minimized) |
| Our implementation | ✅ Plan 066 complete | ❌ Not implementing full pipeline |
| What to adapt | **Variance-minimized schedule for D2F** | — |

### 3.3 vs Research 38 (SDAR Sigmoid Gating)

| Aspect | SDAR (Research 38) | RePlaid |
|--------|-------------------|---------|
| Gating mechanism | Sigmoid gate σ(β·Δt), β=5.0 fixed | Learned noise schedule γ̃(t) |
| Self-supervised signal | Teacher-student gap | ELBO variance |
| Our modelless result | ELO 954 Bomber, draws 100% FFT | — |
| What to adapt | **Learnable β via variance minimization** | — |

---

## 4. What We Distill (Adopted)

### 4.1 Variance-Minimized Noise Schedules — ADOPT

**Insight:** Training the noise schedule to minimize Monte-Carlo variance of the per-timestep loss yields a constant-difficulty schedule (Prop 1). This is self-supervised — no teacher needed.

**Adaptation:**
- **D2F noise schedule** (`src/dllm.rs` `NoiseSchedule`): Replace fixed monotonic ratios with variance-minimized ratios that equalize per-step reconstruction difficulty.
- **Bandit exploration schedule** (`src/pruners/bandit.rs`): Make exploration rate adaptive based on per-episode reward variance tracking.
- **SDAR β learning** (`src/pruners/sdar_gate.rs`): Replace fixed β=5.0 with variance-minimized β that equalizes gated signal intensity across episodes.

**Priority:** High. Self-contained, no model needed, directly improves existing infrastructure.

### 4.2 Self-Conditioning Draft-Refine Loop — ADOPT (Model Path)

**Insight:** For 25% of training, run an initial gradient-free forward pass to estimate clean data, then feed it back as conditioning. At inference, always self-condition. Contributes 1.5 PPL improvement.

**Adaptation:**
- In MTP drafter (Plan 055) and speculative decoding, implement a draft-refine loop where the first pass generates a coarse prediction and the second pass refines it conditioned on the draft.
- Compatible with existing `GpuForwardPass` — just run forward twice.

**Priority:** Medium. Requires wgpu kernel changes but compatible with existing MTP infrastructure.

### 4.3 Over-Training Small Models with ELBO Auxiliary — CONSIDER

**Insight:** RePlaid beats MDLM in the over-trained regime (3.1× vs 6.9× past optimal). ELBO regularization prevents overfitting in small models.

**Adaptation:**
- Add an ELBO-style auxiliary loss to wgpu LoRA training (Plan 008) that regularizes embedding geometry alongside cross-entropy.
- Relevant for QLoRA + IA3 PEFT (Plan 071) targeting small, efficient models.

**Priority:** Medium. Requires new wgpu loss kernels but the scaling law justification is strong.

### 4.4 Low-Rank Game State Embeddings — CONSIDER

**Insight:** ELBO training discovers low-rank, linguistically structured embeddings at d_e=16 without supervision. Adding CE loss disrupts this structure.

**Adaptation:**
- Apply variance-minimized schedule optimization to game state representations (`GoState`, `BomberState`) to discover latent game structure without labeled data.
- Relevant to SpectralQuant eigenbasis calibration (Plan 077) — same principle of letting variance minimization discover structure.

**Priority:** Low-Medium. Interesting but would require new infrastructure.

---

## 5. What We Do NOT Distill (Rejected)

### 5.1 Full Continuous Diffusion Pipeline — REJECTED

**Reason:** Same verdict as Research 10 (ColaDLM). Our DDTree speculative decoding branches on discrete token indices. Continuous diffusion trajectories branch on latent vectors. These are fundamentally incompatible.

Our D2F (Plan 066) is discrete diffusion and works well with our stack. Switching to continuous would require rewriting the entire DDTree + ConstraintPruner + ScreeningPruner pipeline.

### 5.2 Low-dim Embeddings (d_e=16) Everywhere — REJECTED

**Reason:** RePlaid's 50× FLOP savings at d_e=16 vs d_e=768 is specific to the embedding corruption step in continuous diffusion. Our game states and validator embeddings serve different purposes (classification, reward, constraint checking, not generative modeling).

### 5.3 ELBO as Sole Training Objective — REJECTED

**Reason:** Our SDAR (Plan 072) and ROPD (Plan 071) results show sigmoid gating and rubric vectors work well modellessly. Adding ELBO would be over-engineering for the modelless path. The model path uses cross-entropy + SDAR loss, which our benchmarks show is effective.

### 5.4 ODE Solver Distillation — DEFERRED

**Reason:** Continuous diffusion enables trajectory distillation via PFODE solvers, which is unavailable to discrete diffusion. This is a significant advantage but requires adopting continuous diffusion first (rejected above). Long-term research direction only.

---

## 6. Honest Corrections and Caveats

### 6.1 The 20× Gap is Still Large

RePlaid closes the gap from 64× (original Plaid) to 20×, but 20× more compute than AR is still impractical for production at our scale. The theoretical insights (variance minimization, linear info decay) are more valuable than the mechanism itself.

### 6.2 RePlaid's Gains Are Architecture-Specific

The PPL improvements come from aligning continuous diffusion with discrete DLM architecture + adding Plaid-specific components. The variance-minimized schedule works for any process with per-step losses, but the specific PPL numbers are not transferable.

### 6.3 Self-Conditioning Has Entropy Cost

Self-conditioning improves PPL but reduces output entropy at high sampling steps (Fig 3c, 4a). For our game arenas, lower entropy means less diversity in exploration. This is a trade-off we must monitor.

### 6.4 Theoretical Results Assume Bayes-Optimal Denoiser

Prop 2 (near-linear CE) assumes a Bayes-optimal denoiser (Def 1). Our D2F models are far from Bayes-optimal. The practical benefit of the linear schedule may be smaller than the theory predicts.

---

## 7. Actionable Mappings

### 7.1 Immediate (Modelless — microgpt-rs)

| Code Location | Current | Change | Insight Source |
|---|---|---|---|
| `src/dllm.rs` `NoiseSchedule` | Fixed monotonic min→max | Variance-minimized ratios (track per-step loss, equalize difficulty) | Prop 1 |
| `src/pruners/bandit.rs` `BanditStrategy` | Fixed ε, UCB1 | Add `VarianceEpsilon` that adapts ε to equalize per-episode reward variance | Prop 1 analog |
| `src/pruners/sdar_gate.rs` `SDAR_BETA` | Fixed 5.0 | Add `learned_beta()` that minimizes gated-signal variance across episodes | Sec 5.2 |

### 7.2 Medium-term (Model — riir-ai)

| Code Location | Current | Change | Insight Source |
|---|---|---|---|
| `riir-gpu/src/backward.rs` | Pure CE loss | Add ELBO auxiliary loss (reconstruction + diffusion terms) | Sec 2, Eq 4 |
| `riir-gpu/src/forward.rs` `GpuForwardPass::forward()` | Single forward pass | Add self-conditioning path (forward → estimate → forward with conditioning) | Sec 2 |
| `riir-gpu/src/training_loop.rs` `TrainingConfig` | Fixed LR schedule | Add per-component LR (embeddings: 1e-2, backbone: 2e-4) from Tab 7 | Tab 7, 9 |

### 7.3 Long-term Research

- PFODE distillation path for continuous embeddings (if we ever adopt them)
- Low-rank game state embeddings via variance-minimized optimization
- Linear information decay for MCTS simulation budget allocation

---

## 8. Final Verdict

**Overall: 🟡 SELECTIVE ADOPTION**

RePlaid is significantly stronger than ColaDLM (Research 10) because its insights decompose into modular, adoptable pieces that work within our existing architecture. The paper does not change our fundamental decision to use discrete diffusion (D2F) over continuous diffusion.

**The most valuable contribution is the theoretical framework:**
1. **Prop 1** — Variance minimization yields optimal schedules (self-supervised, no teacher needed)
2. **Lemma 3** — Linear information decay under optimal conditions (principles for budget allocation)
3. **Embedding geometry** — ELBO creates structured low-rank spaces without supervision

**The mechanism (continuous diffusion itself) is rejected** for the same reasons as Research 10: incompatible with DDTree speculative decoding.

**Recommended next steps:**
- Plan 078 (microgpt-rs modelless): Variance-minimized schedules for D2F noise + bandit exploration + SDAR β
- Plan 079 (riir-ai model-based): Self-conditioning draft-refine loop + ELBO auxiliary loss for wgpu LoRA

---

## 9. Paper Metadata

| Field | Value |
|---|---|
| Parameter Scale | 19M–2121M (scaling law), 0.1B (benchmark) |
| Training Data | SlimPajama (scaling law), OpenWebText (benchmark) |
| Tokenizer | Llama-2 (V=32,001) |
| Sequence Length | 2048 (scaling law), 1024 (benchmark) |
| Embedding Dimension | d_e=16 (low-dim, learnable, unit-norm) |
| Key Baselines | AR, MDLM, Duo, FLM, LangFlow |
| Compute | 8× GB200 (scaling), 8× H200 (benchmark) |
| Key Ablation | Learnable embeddings: +17.3 PPL (Table 2) |
| Samplers | DDPM, DDIM, DPM-Solver++(2M), Heun |
| Feature Flags | Self-conditioning (25% train / 100% inference) |