# Research: GDSD — Guided Denoiser Self-Distillation for Diffusion LMs

**Date:** 2026-06-02
**Source:** arXiv:2605.29398 — Tang et al. (UCL, Alibaba, UNIST, Basel)
**Status:** Creative Distillation — Modelless + Model-Based Fusion

---

## 1. Paper Core: What GDSD Actually Proves

**The fundamental insight:** For diffusion LMs (dLLMs), RL alignment via ELBO-based likelihood surrogates is fundamentally broken because of Training-Inference Mismatch (TIM). GDSD shows that the closed-form optimal policy under reverse-KL regularized RL induces an *energy-guided denoiser* — and you can distill this teacher directly via logit matching without ever computing likelihoods.

**The math that matters:**
- Reverse-KL regularized RL → closed-form optimal policy: `π* ∝ π_old^(1-β) · π_ref^β · exp(ψ·A(x₀))`
- This induces a teacher denoiser: `p*(x₀|xₜ) = p_ref_old(x₀|xₜ) · exp(ψ·A(x₀) - Aₜ(xₜ))`
- Token-Level Logit Centralization (TLC) eliminates the partition function: `log p̄*(x₀|xₜ) = (1-β)·log p̄_old + β·log p̄_ref + ψ·A(x₀) - b`
- The practical loss is just squared logit matching: `L = (log p̄_θ - log p̄_old - ψ·A)² + β·(log p̄_θ - log p̄_ref)²`

**Key results:** Up to +19.6% on Sudoku/Countdown with Dream-7B, +5% on LLaDA-8B. Stable training vs ELBO collapse.

---

## 2. Creative Fusion: Beyond Direct Mapping

### 2.1 Fundamental Abstraction: Advantage-Guided Denoising is Universal

The GDSD insight generalizes beyond diffusion LMs. **Any generative process where likelihood is intractable but denoiser logits are available** can use this trick. Our codebase has TWO such processes:

1. **D2F block-parallel decoding** (model-based, riir-ai) — discrete diffusion with block-causal attention
2. **DDTree speculative decoding** (modelless, katgpt-rs) — tree search where "denoising" = logit refinement through pruning

### 2.2 Fusion Idea A: DDTree-as-Diffusion (Modelless)

The DDTree is already a denoising process: marginals get refined at each depth by ScreeningPruner. The "denoiser" is the pruner's relevance function. The "teacher" is a hypothetical optimal pruner that assigns high relevance to high-advantage branches.

**Key insight:** We can apply GDSD-style advantage-guided self-distillation to the DDTree pruner WITHOUT any model training. Instead of matching denoiser logits, we match *pruner relevance scores* to an advantage-weighted teacher:

```
r_teacher(depth, token) = (1-β) · r_old(depth, token) + β · r_ref(depth, token) + ψ · A(action)
```

Where:
- `r_old` = current pruner's relevance score
- `r_ref` = base/reference pruner (e.g., NoScreeningPruner = 1.0 everywhere)
- `A(action)` = advantage from bandit/arena rewards
- `ψ` = guidance coefficient

This creates a **GDSDPruner** — a `ScreeningPruner` that distills advantage-weighted relevance from a self-teacher. It's GDSD for modelless inference: no likelihood, no gradients, just pruner score matching.

**Why this is creative, not direct:** The paper operates on denoiser log-probs. We map it to pruner relevance scores, which are the modelless analog. The "denoiser" in our case IS the pruner — it refines the search tree the same way diffusion refines noise.

### 2.3 Fusion Idea B: GDSD Loss for D2F Training (Model-Based)

Direct but with our twist: Our D2F infrastructure (`dllm.rs`, `GpuD2fDistill`) currently uses ELBO or KL distillation. GDSD replaces both with squared logit matching + TLC.

**Our twist — the WGSL factor:** The GDSD loss is element-wise squared error on centralized logits. This is PERFECT for WGSL compute shaders:
- No softmax (expensive on GPU)
- No log-softmax (even more expensive)
- No partition function estimation
- Just: `(centralized_logit_a - centralized_logit_b)²` per token
- TLC = subtract mean over vocab — one `workgroup_sum` + broadcast

**WGSL kernel sketch:**
```wgsl
// loss_gdsd.wgsl
// Input: student_logits[V], teacher_logits[V], advantage[f32]
// Output: loss[f32]

@group(0) @binding(0) var<storage, read> student: array<f32>;
@group(0) @binding(1) var<storage, read> teacher: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: GdsdParams;

struct GdsdParams { vocab_size: u32, psi: f32, beta: f32, advantage: f32 }

@compute @workgroup_size(256)
fn gdsd_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    // TLC: subtract mean over vocab (done in separate pass)
    // This kernel just computes per-token squared error
    let v = gid.x;
    if (v >= params.vocab_size) { return; }
    let s = student[v];
    let t = teacher[v];
    let diff = s - t;
    output[v] = diff * diff;
}
```

### 2.4 Fusion Idea C: GDSD + GZero Self-Play (Modelless + Model-Based)

The GZero loop already has Hint-δ (log-prob shift as intrinsic reward). GDSD provides a principled way to turn δ into a teacher signal:

**Modelless path (katgpt-rs):**
- GDSDPruner uses `A = δ` (Hint-δ advantage) as the energy guidance
- Creates a self-teacher pruner: `r* = r_old + ψ·δ`
- The pruner now actively guides the DDTree toward high-δ branches

**Model-based path (riir-ai):**
- GZeroLoop gets a new `gdsd_loss` mode alongside DPO/SLIME
- Rollout → reward → advantage → GDSD squared logit matching
- Bypasses ELBO in the D2F training path

### 2.5 Fusion Idea D: TLC as General Stabilization (Cross-Cutting)

Token-Level Logit Centralization isn't just for GDSD. It's a general technique for stabilizing ANY distillation process:
- **SDAR:** Centralize teacher/student log-probs before sigmoid gating → prevents logit drift
- **ASFT:** Centralize forward-KL anchoring → more stable KL
- **GRPO/CISPO:** Centralize importance ratios → prevents TIM-style collapse

TLC should be a standalone utility, not tied to GDSD.

---

## 3. Verdict by Commercial Strategy (003)

### What's Open Source (katgpt-rs, MIT)?

| Component | Verdict | Reasoning |
|-----------|---------|-----------|
| **GDSDPruner** (modelless) | ✅ Open — `ScreeningPruner` impl | Pruner infrastructure is open engine. Advantage-guided relevance is the analog of GDSD for modelless inference. No competitive moat — anyone can implement it. The moat is `lora.bin` + `validator.wasm`. |
| **TLC utility** | ✅ Open — `katgpt-core` | Logit centralization is a math trick, not a trade secret. Open it for community benefit. |
| **DDTree GDSD integration** | ✅ Open | Tree search + advantage weighting is generic. |

### What's Private (riir-ai, SaaS)?

| Component | Verdict | Reasoning |
|-----------|---------|-----------|
| **`loss_gdsd.rs` + `loss_gdsd.wgsl`** | ✅ Private — riir-gpu | GPU training losses are private SaaS intelligence. Part of the training moat. |
| **GZeroLoop GDSD mode** | ✅ Private | Self-play orchestration with GDSD loss is SaaS infrastructure. |
| **D2F + GDSD training pipeline** | ✅ Private | The D2F training infrastructure is already private (`dllm` feature gate). |

### Alignment with Verdict Principles

1. **"Don't broaden the wedge. Go deeper."** — GDSDPruner deepens the DDTree pruner ecosystem. It's a new pruner, not a new product direction.
2. **"Engine without lora.bin produces syntactically-valid-but-semantically-wrong Rust."** — GDSDPruner improves tree search quality, but without trained weights the pruner has no advantage signal to guide with. The engine improves, the moat remains.
3. **"MIT for the engine, SaaS for the intelligence."** — Modelless pruner = engine (MIT). GPU loss kernel = intelligence (private). Clean split.
4. **"The architecture is proven. The gap is one trained model."** — GDSD doesn't change this. It makes both the modelless pruner and the model-based trainer more stable, but the gap remains the same.

---

## 4. Gain Assessment

### Modelless (katgpt-rs)

| Dimension | Gain | Risk |
|-----------|------|------|
| **GDSDPruner** | Medium — advantage-guided tree search could improve DDTree accuracy by prioritizing high-reward branches | Low — it's just a new `ScreeningPruner` impl, no existing code changes |
| **TLC utility** | Low for modelless — centralized log-probs in bandit context may help SDAR stability | Minimal — pure utility function |
| **DDTree GDSD integration** | Low-medium — depends on quality of advantage signal from arena/bandit | Low — additive feature |

### Model-Based (riir-ai)

| Dimension | Gain | Risk |
|-----------|------|------|
| **GDSD loss for D2F** | **High** — directly replaces ELBO in D2F training, proven +5-20% on dLLM benchmarks | Medium — requires WGSL kernel, integration with existing D2F pipeline |
| **TLC stabilization for SDAR/ASFT** | Medium — could stabilize existing losses that already work | Low — additive preprocessing step |
| **GZeroLoop GDSD mode** | Medium — new loss option alongside DPO/SLIME, but DPO already works well | Low — composable, doesn't replace existing paths |

### Overall Verdict: **GAIN for model-based, CONDITIONAL GAIN for modelless**

- **Model-based:** GDSD loss directly improves D2F training stability and accuracy. The paper proves it on LLaDA-8B and Dream-7B — the exact architectures we target. This is a clear win.
- **Modelless:** GDSDPruner is a creative idea that maps the GDSD insight to the pruner domain. It may improve DDTree quality but the advantage signal quality depends on arena/bandit maturity. Worth implementing as an opt-in feature to test.

### Default-On Decision

Per optimization.md principles: "if gain and no perf hurt must be on by default"

- **Modelless GDSDPruner:** NOT default-on until GOAT proven. Advantage signal quality is uncertain.
- **Model-based GDSD loss:** NOT default-on. It's a new loss competing with existing ELBO/SDAR. Needs GOAT proof first.
- **TLC utility:** Default-on as a preprocessing option behind a config flag. Zero perf impact when disabled.

---

## 5. Optimization Alignment

| Optimization Principle | GDSD Compliance |
|------------------------|-----------------|
| **Profile first** | ✅ WGSL kernel is trivial (element-wise squared error). No profiling needed — it's the simplest possible loss. |
| **No allocation in hot loops** | ✅ TLC = subtract scalar from buffer. Zero alloc. |
| **Pre-compute lookup tables** | N/A — GDSD has no lookup tables. |
| **GPU wins only for batched matmul** | ✅ GDSD loss is NOT a matmul — it's element-wise. CPU computation is fine for small batches. The GPU path exists for large V (vocab size > 10K). |
| **No rayon for tiny workloads** | ✅ TLC centralization is O(V) per token — too small for rayon. Serial is correct. |
| **WASM batch API** | ✅ GDSDPruner could use batch relevance scoring for all tokens at a depth. |

---

---

## 6. Actual Implementation Analysis (Source Code)

Source: `.raw/GDSD/gdsd/` — official implementation from authors.

### 6.1 Core Loss (GDSD Direct, `gdsd_trainer_batchll.py` L191-192)

```python
# The ENTIRE GDSD loss is 3 lines:
logits_diff = (per_token_logps - old_per_token_logps).clamp(
    math.log(1 - self.epsilon_low), math.log(1 + self.epsilon_high)
)
loss = ((logits_diff / logits_to_keep - self.psi * advantages.view(-1, 1, 1)) ** 2).sum() / batch_size
```

Plus optional KL regularization (k2 estimator = squared log-ratio / 2):
```python
if self.beta != 0.0:
    kl = compute_approx_kl(per_token_logps, ref_per_token_logps, "k2")  # = (log_ratio)**2 / 2
    mean_kl = kl.sum() / (batch_size * self.num_mc * logits_to_keep)
    loss += self.beta * mean_kl
```

**Key finding:** The loss operates on *sequence-level* log-probs (sum of per-token log-probs), NOT per-token.
The `/logits_to_keep` normalization is critical — it makes the squared error operate on mean log-prob difference.

### 6.2 TLC Variant (`gdsd_trainer_tlc.py` L455-456)

The ONLY difference from non-TLC: one line of centralization in `_compute_denoise_probs_from_masked_batch`:
```python
# TLC: subtract mean over vocab dimension (detached)
logits_kept = logits_kept - logits_kept.mean(dim=-1, keepdim=True).detach()
```

This happens BEFORE cross-entropy computation. The rest of the trainer is identical.

### 6.3 Denoising Probability Computation

"Denoise probs" are actually ELBO estimates: cross-entropy on masked positions → sum → negate.
For GDSD (non-TLC), no centralization. For GDSD-TLC, centralize logits first.

Variance reduction via coupled masking: flip masked/unmasked positions, compute CE on both, average.

### 6.4 Generation Pipeline

Block-wise semi-autoregressive diffusion:
- `gen_length` tokens generated in blocks of `block_length=32`
- Per block: `steps_per_block` diffusion iterations
- Each step: forward pass → Gumbel noise → argmax → low-confidence remasking
- Transfer tokens based on confidence ranking

### 6.5 Actual Hyperparameters (from `train_gdsd.yaml`)

```yaml
# Default config
rl_loss_type: gdsd
beta: 0.001          # KL regularization (very small)
epsilon: 0.0004      # Clipping range (VERY tight, ~0.04%)
psi: 1.0             # Guidance coefficient (NOT 10.0 as in paper Table 4)
num_generations: 6   # Rollouts per prompt
num_iterations: 8    # Old policy refresh iterations
num_mc: 1            # Monte Carlo samples for ELBO estimation
reduce_var: false    # No coupled variance reduction in default
diffusion_steps: 64  # Generation steps (half of completion length)
generation_temperature: 1.0
learning_rate: 3e-6
lora_r: 128, lora_alpha: 64, lora_dropout: 0.05
gradient_accumulation_steps: 12
ref_model_sync_steps: 64  # Refresh old policy every 64 steps
```

**Critical finding:** `psi=1.0` in the yaml, not 10.0 as in paper. The paper Table 4 says psi=10.0 for LLaDA-8B.
This suggests the yaml is a demo config, and the paper results used higher psi. Our GOAT proof should sweep psi.

**Another critical finding:** `epsilon=0.0004` is EXTREMELY tight clipping. This is essentially no clipping.
The paper uses log-space clipping: `clamp(log(0.9996), log(1.0004))` ≈ no clipping at all.
This makes sense — GDSD doesn't need PPO-style clipping because the squared loss is naturally bounded.

### 6.6 Architecture Implications for Our Implementation

1. **Loss is simpler than we thought.** We planned 3 WGSL kernels. We only need 2:
   - `gdsd_loss.wgsl` — squared error on log-prob differences
   - `gdsd_reduce.wgsl` — tree reduction
   TLC can be a Rust-side operation (subtract f32 mean from buffer) — no GPU kernel needed.

2. **Denoise probs are sequence-level sums.** Our D2F infrastructure computes per-token masked CE.
   We need to SUM over tokens (not mean) to match the paper.

3. **Coupled variance reduction is optional.** Default config has `reduce_var: false`.
   We can add it later as an optimization.

4. **psi=1.0 works.** The guidance coefficient doesn't need to be 10.0 for the basic case.
   Paper uses higher psi for specific benchmarks.

5. **No logit centralization in the base GDSD.** Only TLC variant adds it.
   Our base implementation can skip centralization entirely.

---

## 7. Reference

- Tang et al. "GDSD: Reinforcement Learning as Guided Denoiser Self-Distillation for Diffusion Language Models." arXiv:2605.29398, May 2026.
- Code: https://github.com/GaryBall/GDSD
- Local source: `.raw/GDSD/`
