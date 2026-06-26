# Breakeven Complexity: Inference-Time Cost-Aware Routing

**Date:** 2026-06
**Paper:** [Breakeven complexity: A new perspective on neural PDE solvers](https://arxiv.org/pdf/2605.15399)
**Status:** Research Verdict

---

## Paper Summary

The paper proposes **breakeven complexity** N* = B / (C_δB - C_inf), a cost-aware metric that counts forward solves before a learned/approximate method becomes cost-effective vs an error-equivalent traditional method. Key findings:

1. Neural surrogates need hundreds of thousands of calls on easy problems, but breakeven drops sharply as problem difficulty increases
2. Scaling laws can predict optimal data/compute allocation for a given budget
3. Worst-case vs average-case breakeven reveals robustness — tighter ratio = more robust solver
4. Wallclock time is the right cost currency (not FLOPs) because throughput varies 90× between dense tensor ops and stencil computations

## Novel Fusion: Breakeven Complexity for LLM Inference Routing

### The Core Insight

The paper compares **classical PDE solvers** vs **neural PDE surrogates**. We map this to:

| Paper Concept | katgpt-rs Mapping |
|--------------|-------------------|
| Classical solver (high-fidelity) | Full attention forward pass |
| Classical solver (low-fidelity) | Sparse/pruned attention |
| Neural surrogate (trained) | Speculative decoding (draft model + verify) |
| Data generation cost B | Draft model compilation/quantization cost |
| Inference cost C_inf | Per-token speculative decode cost |
| Error-matched classical cost C_δB | Per-token full attention cost at matched quality |
| Breakeven N* | Tokens before speculative decode amortizes its overhead |

### The Equation Applied

```
N* = B_draft / (C_full_attention - C_speculative)
```

Where:
- `B_draft` = upfront cost of loading + quantizing + warming up the draft model
- `C_full_attention` = wallclock time per token for full attention (the "classical solver")
- `C_speculative` = wallclock time per token for speculative decode (the "neural surrogate")

When N* is small (harder inference, longer sequences, higher QPS), speculative decode wins.
When N* is infinite (C_speculative ≥ C_full_attention), speculative decode never wins.

### Multi-Fidelity Inference Stack

Extending the paper's multi-fidelity classical solver concept:

```
Tier 0 (CPU, full attention)     → "High-fidelity classical solver"
Tier 1 (CPU, sparse attention)   → "Low-fidelity classical solver"  
Tier 2 (GPU, full attention)     → "Accelerated classical solver"
Tier 3 (GPU, speculative decode) → "Neural surrogate"
Tier 4 (GPU+ANE, speculative)    → "Optimized neural surrogate"
```

Each tier has a breakeven threshold against the tier below it. The TriggerGate + InferenceRouter already switches tiers based on QPS — breakeven complexity adds a **cost-amortization** dimension.

### Key Novel Ideas

1. **Breakeven Bandit**: A meta-bandit that tracks per-tier breakeven N* and routes inference to the tier that has already amortized its setup cost. Uses existing `FrequencyBandit` infrastructure.

2. **Adaptive Fidelity Matching**: The paper's error-matched classical solver maps to our KV compression. Instead of fixed compression, compute error-matched compression level: "What KV compression gives the same perplexity as full attention at sequence position N?" This is the paper's Eq. 2 applied to KV cache.

3. **Workload Forecasting**: The paper notes that breakeven depends on workload size. We can estimate workload size from the prompt length + predicted output length (using existing `LodestarDistance` infrastructure). If predicted N < N*, use the simpler tier.

4. **Scaling Law for Budget Allocation**: The paper's Section 4.2 uses scaling laws to predict optimal data/compute split. We can use the same approach to predict: given a fixed time budget, how much to spend on KV cache warming vs inference.

### Why This Is Novel

- The paper applies breakeven to PDE solvers. Nobody has applied it to LLM inference routing.
- The multi-fidelity concept (varying resolution classical solver) maps 1:1 to our multi-tier inference stack.
- The scaling law allocation (data vs compute) maps to our KV cache prefill vs decode tradeoff.
- The worst-case vs average-case robustness analysis maps to our speculative decode acceptance rate variance.

### GOAT Verdict: HIGH GAIN

**Score: 8/10**

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| Novelty | 9 | Breakeven complexity for LLM inference is unexplored territory |
| Applicability | 9 | Maps directly to existing InferenceRouter + TriggerGate + FrequencyBandit |
| Performance | 8 | Principled cost-aware routing beats heuristic QPS thresholds |
| Commercial | 7 | Improves SaaS cost-efficiency, but not a differentiator alone |
| Risk | 6 | Requires benchmarking to validate; overhead of breakeven computation must be negligible |

**Decision: PROCEED.** Create plan 250 with BreakevenBandit as the core implementation.

---

## Mapping to Existing Infrastructure

| Component | Role | Status |
|-----------|------|--------|
| `InferenceRouter` | Tier dispatch (already has trust, RV, critical-interval signals) | ✅ Exists |
| `TriggerGate` | QPS-based tier promotion/demotion | ✅ Exists |
| `FrequencyBandit` | Multi-arm bandit for spec decode config | ✅ Exists |
| `ComputeTier` | CPU / CPU+GPU / CPU+GPU+ANE | ✅ Exists |
| `RouterStats` | Runtime metrics (QPS, tier transitions) | ✅ Exists |
| BreakevenBandit | NEW — tracks N* per tier pair, routes to amortized tier | ❌ New |
| FidelityMatcher | NEW — error-matched KV compression level | ❌ New |

## Constraints Check

- [x] Modelless first — all computation is inference-time, no LLM training
- [x] Lands in katgpt-rs domain — pure inference routing
- [x] SOLID, DRY — extends existing traits (uses FrequencyBandit pattern)
- [x] CPU/GPU auto-route — breakeven drives tier selection
- [x] Plasma/hot/warm/cold — breakeven N* determines when each tier is worth it
- [x] Threshold-based — sigmoid thresholds for tier transitions (not softmax)
- [x] Feature-gated — behind `breakeven_routing` feature flag

## TL;DR

Breakeven complexity from PDE solvers maps cleanly to LLM inference tier routing. The key insight — "harder problems have lower breakeven, making approximation more valuable" — explains why GPU speculative decode is worth it for long sequences but not for short prompts. Implement as `BreakevenBandit` that tracks per-tier amortization thresholds and routes accordingly.
