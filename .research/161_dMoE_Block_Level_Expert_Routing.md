# Research 161: dMoE — Block-Level Expert Routing for MoE dLLMs

> **Paper:** [dMoE: dLLMs with Learnable Block Experts](https://arxiv.org/abs/2605.30876)
> **Authors:** Sicheng Feng, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang (NUS)
> **Date:** June 2026
> **Verdict:** ✅ GAIN — Two modelless distillations, one modelless enhancement, one model-based fusion.
> **GOAT Pillar:** ❌ Not a pillar — general inference optimization. Evaluated against [MMO GOAT Pillars](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md): passes LoRA-independent (required) but fails MMO-product (required). Stays in `katgpt-rs` domain for modelless, `riir-ai` for LoRA routing.
> **Domain:** `katgpt-rs` (modelless) + `riir-ai` (LoRA routing fusion)

---

## Paper Summary

### Problem
MoE dLLMs process multiple tokens per forward pass (block parallel decoding), but MoE routing is token-level → each token independently selects experts → too many unique experts activated → **memory-bound** (expert weight loading dominates latency).

**Key numbers**: LLaDA2.0-mini activates ~70 unique experts per layer per block. MoE latency = 60-80% of total inference time.

### Solution: Block-Level Expert Routing (dMoE)

```
1. Token-level routing → per-token expert scores: si = Router(ti)
2. Aggregate → block-level expert scores: Sblock = ⊕i∈B si
3. Normalize + top-p → adaptive coreset: C = Top-P(S̃block, p)
4. Restrict token routing to coreset: Ri = TopK(si|C, k)
```

**Key insight**: Expert concentration varies across denoising steps and blocks (CV=15-18%). Top-p adapts automatically — small coreset when concentrated, larger when dispersed.

### Results
- Unique experts: 69.5 → 14.6 (4.77× reduction)
- Performance: 99.11% retained
- Memory: 76-80% reduction
- Latency: 1.14-1.66× speedup
- At matching compression levels, DES-S/DES-V lose 20-40% performance while dMoE loses <1%

### Two Key Observations

**Observation A**: Token-level expert scores positively correlate with expert importance (r=0.462). Router weights predict which experts matter.

**Observation B**: Unique expert count has high coefficient of variation (15-18% across benchmarks). Fixed-size coreset is suboptimal.

### Self-Distillation Training
- Uses same model to generate training targets (no external data)
- CSE loss between original routing and coreset-constrained routing
- p_train=0.6, but p_test is tunable at inference time (0.4-0.8 range works)

---

## Creative Fusion: Beyond Direct Mapping

Direct mapping would be "add MoE to our stack" — we don't have MoE. Instead, we distill the **fundamental principle**: **aggregate granular scores → coarser unit → adaptive top-p coreset → restrict selection**. This applies everywhere we have per-element routing decisions:

### D1: DDTree Vocab Coreset (Modelless — katgpt-rs)

**Principle**: Aggregate token marginals across speculative batch → identify "vocab coreset" → restrict DDTree branching.

```
dMoE: token expert scores → block expert scores → top-p coreset → restrict expert pool
Us:   token marginals     → batch marginals     → top-p coreset → restrict DDTree vocab
```

**How it works**:
1. During speculative decode, Drafter produces marginals for K+1 draft tokens
2. Aggregate: for each vocab token, take max probability across all K+1 positions
3. Top-p: cumulative sum → threshold at p → vocab coreset C
4. DDTree expansion only considers tokens in C

**Gain**: DDTree branching factor reduces from |V| to |C| (typically 10-50× reduction). Tree construction is O(|V|) per depth level → O(|C|) after coreset.

**Risk**: If p is too aggressive, valid tokens fall outside coreset → pruned by ConstraintPruner → degradation.
**Mitigation**: Top-p is adaptive — high-entropy marginals get larger coreset naturally.

### D2: Adaptive Top-p Bandit Arms (Modelless — katgpt-rs)

**Principle**: dMoE's core insight is that expert concentration varies — use top-p, not top-k. Apply same principle to BanditPruner arm selection.

```
dMoE: expert concentration varies → top-p selects adaptive coreset
Us:   arm concentration varies    → top-p selects adaptive arm budget
```

**How it works**:
1. BanditPruner computes arm scores (Q-values + UCB bonus)
2. Sort descending, cumulative sum
3. Top-p threshold → select arms until cumulative score ≥ p
4. Only selected arms are evaluated by WASM validator

**Gain**: When arm scores are concentrated (clear best action), evaluates fewer arms → faster WASM FFI. When dispersed (uncertain), evaluates more → better exploration.

**Connection to Adaptive CoT (Plan 194)**: The bandit already learns WHEN to think. With top-p, it also learns HOW MANY alternatives to consider. This creates a two-dimensional adaptive budget: (think/direct) × (narrow/wide arms).

**Risk**: Top-p computation overhead (~100ns for sort + cumsum) vs fixed top-k (~10ns).
**Mitigation**: For N arms ≤ 16 (typical game actions), sort is <50ns. The savings from skipping WASM validation on excluded arms (200-500ns each) dominate.

### D3: Delta Sparse Verification Enhancement (Modelless — katgpt-rs)

**Principle**: dMoE aggregates per-token routing to find shared experts → avoids redundant weight loading. Apply to LeviathanVerifier's sparse_matmul during verification.

```
dMoE: aggregate token expert scores → find shared experts → load once, reuse
Us:   aggregate token neuron indices → find shared neurons → compute once, reuse
```

**Status**: Already proposed as Plan 096 T4 (Delta Sparse Matmul). dMoE provides additional theoretical justification: the block-level aggregation pattern and the observation that routing concentration varies, meaning delta sparsity benefits also vary (and should be measured per-block, not averaged).

**New insight from dMoE**: Use top-p to decide WHEN delta sparse is worth it. If the coreset overlap between consecutive tokens > 70% → delta sparse wins. If < 30% → standard sparse is faster (overhead of tracking shared neurons dominates).

### D4: Block-Level LoRA Expert Routing (Model-based — riir-ai)

**Principle**: Apply dMoE's block-level aggregation to LoRA expert routing in game AI.

```
dMoE: token scores → block scores → top-p coreset → restrict MoE experts
Us:   action scores → frame scores → top-p coreset → restrict LoRA experts
```

**Landing**: riir-ai domain. Frame-sampling produces N entities × M actions per game frame. Each (entity, action) pair may activate different LoRA experts. Without aggregation, worst case is N×M unique LoRA switches per frame.

**How it works**:
1. For each entity, LoRA scores each action → per-entity expert scores
2. Aggregate across entities → frame-level expert scores
3. Top-p → coreset of LoRA experts for entire frame
4. Load coreset LoRA weights once, route actions within coreset

**Gain**: LoRA switching cost reduces from O(N×M) to O(|coreset|). For 4 players × 6 actions = 24 evaluations with 8 possible experts → typically reduces to 3-5 unique experts per frame.

**See**: `riir-ai/.research/051_dMoE_Block_Level_LoRA_Routing.md`

---

## GOAT Verdict

Per [003_Commercial_Open_Source_Strategy_Verdict.md](003_Commercial_Open_Source_Strategy_Verdict.md):

| Distillation | Gain | Perf Hurt | Default? | Modelless? | Verdict |
|---|---|---|---|---|---|
| D1: DDTree Vocab Coreset | 10-50× branching reduction | Top-p too small → missed tokens | ⚠️ Opt-in first, default after GOAT | ✅ | ✅ GAIN |
| D2: Adaptive Top-p Bandit | Dynamic arm budget, faster when confident | ~100ns top-p overhead | ✅ Default ON (savings > overhead) | ✅ | ✅ GAIN GOAT |
| D3: Delta Sparse Enhancement | Conditional speedup on shared neurons | Tracking overhead when overlap < 30% | ⚠️ Opt-in diagnostic | ✅ | ⚠️ Enhancement only |
| D4: Block LoRA Routing | LoRA switch reduction N×M → |coreset| | Training change for LoRA routing | ⚠️ Opt-in (riir-gpu) | ❌ LoRA | ✅ GAIN |

**Decision**: D2 is GOAT → **default ON**. D1 is GAIN → implement as opt-in, benchmark, then promote to default if GOAT passes. D3 enhances existing plan. D4 lands in riir-ai.

---

## What We Don't Need

| From Paper | Why Skip |
|---|---|
| MoE architecture | We don't have MoE layers — we have sparse MLP + bandit routing |
| Self-distillation training | Modelless distillation only — no LLM training |
| LLaDA2.0-mini fine-tuning | No model training in katgpt-rs |
| Expert merging/pruning | No experts to merge |
| Full fine-tuning results | LoRA only constraint |
| Block diffusion specifics | Our D2F module has its own block handling |

---

## Related Research in Our Stack

| Ours | Connection |
|---|---|
| Research 059 (MoE+SD CoDesign) | Direct predecessor — temporal routing overlap, Amdahl cost model |
| Research 126 (MoA) | Our "Mixture of Activations" — token-adaptive LoRA mixing, but fixed top-k |
| Research 071 (ROPD) | Self-distillation pattern — same concept (model teaches itself) |
| Plan 096 (MoE SD Distillation) | Delta sparse matmul (T4) — dMoE justifies when it's worth doing |
| Plan 194 (Adaptive CoT) | Bandit learns when to think — D2 adds "how many arms" dimension |
| Plan 176 (Three-Way Compute) | TriggerGate routes CPU/GPU/ANE — D1 coreset could inform routing tier |
| Research 072 (DMax) | Aggressive parallel decode — block-level coreset could reduce DMax expert explosion |

---

## References

- dMoE paper: arXiv:2605.30876
- Code: https://github.com/fscdc/dMoE
- Model: https://huggingface.co/FSCCS/dMoE-16B
- LLaDA2.0-mini: https://huggingface.co/inclusionAI/LLaDA2.0-mini
- Related: DES (arXiv:2602.00879), TEAM (arXiv:2602.08404), EC-DLM (arXiv:2604.01622)
- Block Diffusion: arXiv:2508.15487
