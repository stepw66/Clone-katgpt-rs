# Verdict: ANE Sharding & Residency Patterns → katgpt-rs Modelless Fusion

**Date:** 2026-06
**Source:** [videlalvaro/ane-book](https://github.com/videlalvaro/ane-book) — Apple Neural Engine inference
**Status:** GOAT Verdict

---

## Source Distillation

The ane-book solves: **running LLMs on fixed-function accelerators with hard constraints** (250MB shard limit, no dynamic control flow, INT8-only production quantization). Their solutions are:

| Pattern | What ANE Book Does | Why It Exists |
|---------|-------------------|---------------|
| **Layer-range sharding** | Split transformer into heterogeneous layer-range chunks tuned per weight budget | ANE has 250MB compiled-weight ceiling |
| **Residency validation** | `MLComputePlan` checks every op lands on ANE, not CPU — compile success ≠ execution target | INT4 shards silently fell back to CPU |
| **RangeDim (variable T)** | Single compiled artifact handles T∈[1..4] — 1 for decode, 4 for speculative verify | Avoids duplicate shards per phase |
| **Matmul-as-scatter** | `matmul(write_mask, new_values)` replaces for-loop for KV cache writes | Loop-free, T-agnostic, ANE-friendly |
| **Conv2d(1×1) universal** | Every linear layer → Conv2d(1×1) in NCHW layout | ANE speaks conv, not matmul |
| **Soft-routed MoE** | All experts run, zero-mask non-selected → static graph, ANE-resident | Dynamic branching kills ANE residency |
| **mmap embedding/RoPE** | Separate .bin files, zero-copy lookup | Keeps CoreML package compute-only |
| **INT8 per-tensor only** | INT4 per-block silently falls to CPU in small shards | Verified empirically, not documented by Apple |

---

## Fusion Ideas (Not Direct Mapping)

### Fusion 1: **Residency Audit for DDTree Pruning Paths** (MODELLESS — HIGH GAIN)

**The ANE insight:** "Compile success ≠ execution target." A model that compiles, runs, and produces correct output may be executing on the wrong hardware path — silently 5× slower.

**Our fusion:** Apply the same principle to DDTree + ConstraintPruner paths. Currently we verify tree correctness (100% valid nodes). But we don't verify that pruning paths actually land on the **optimal compute path**. A `ConstraintPruner` that prunes 95% of branches but forces the remaining 5% into expensive verification is silently worse than one that prunes 80% but keeps everything on fast paths.

**Modelless implementation:**
```rust
/// Residency audit — does this pruner actually land on fast paths?
pub trait PrunerResidencyAudit: Send + Sync {
    /// Returns (fast_path_ratio, avg_branch_cost) after a full DDTree build
    fn audit(&self, tree: &DDTree) -> ResidencyReport;
}

pub struct ResidencyReport {
    pub fast_path_ratio: f32,      // fraction of retained nodes on fast verification path
    pub avg_branch_cost_ns: f64,   // nanoseconds per retained node
    pub silent_degradation: bool,  // true if pruning looks good but cost is hidden
}
```

**GOAT test:** Run 1000 DDTree builds with different pruners. If `fast_path_ratio < 0.8`, the pruner is "silently degrading" — pruning branches but forcing expensive verification. Same test as the ANE INT4 bug: functional correctness ≠ performance correctness.

**Verdict:** ✅ GAIN — catches a class of bugs our current 100%-valid-nodes test cannot. Modelless. Zero perf hurt (audit is post-hoc, not in hot path). **Must be on by default** as a test, not runtime.

---

### Fusion 2: **RangeDim Budget Adapter for DDTree** (MODELLESS — HIGH GAIN)

**The ANE insight:** `RangeDim` compiles T∈[1..4] into one artifact. The same compute graph handles both single-token decode and multi-token speculative verification without separate compiled models.

**Our fusion:** DDTree currently uses a fixed `budget` (max tree nodes). But the optimal budget varies wildly:
- **Easy queries:** budget=1 (greedy, no tree needed) — like T=1 decode
- **Hard queries:** budget=64 (deep tree exploration) — like T=4 speculative verify
- **Currently:** We have `budget_adaptation` feature but it's compression-based, not query-difficulty-based

**The ANE-inspired twist:** Use a **single DDTree configuration** with a **RangeDim-style budget slot** that adapts per-query based on entropy. The DDTree already has the mechanism (BinaryHeap, marginal probs). We just need:

```rust
/// RangeDim-inspired budget: single config, variable budget per query
pub struct RangeBudget {
    pub min_budget: usize,  // T=1 equivalent (greedy)
    pub max_budget: usize,  // T=4 equivalent (speculative)
    // Bandit learns the right budget per query type
    bandit: BanditPruner,
}

impl RangeBudget {
    pub fn budget_for_entropy(&self, entropy: f32) -> usize {
        // Low entropy → min_budget (easy, like decode)
        // High entropy → max_budget (hard, like speculative verify)
        let t = (entropy / self.entropy_threshold).clamp(0.0, 1.0);
        let budget = self.min_budget as f32 + t * (self.max_budget - self.min_budget) as f32;
        budget as usize
    }
}
```

**GOAT test:** Run 1000 queries, measure acceptance rate at each budget level. The bandit should learn:
- Low entropy queries: budget=1, acceptance ~100% → no wasted tree exploration
- High entropy queries: budget=32-64, acceptance ~60% → worthwhile exploration

**Verdict:** ✅ GAIN — we already have BanditPruner + budget_adaptation. This fusion makes budget selection **entropy-aware** (modelless, inference-time only). No perf hurt — reduces wasted computation on easy queries. **Must be on by default** as part of the adaptive CoT stack.

**Note:** This is NOT a direct mapping of RangeDim. RangeDim solves variable token count in compiled graphs. We're solving variable **search budget** in inference-time tree search. The fusion is the **pattern** (single-config, variable execution) not the mechanism.

---

### Fusion 3: **Matmul-as-Scatter for KV Cache Compression** (MODELLESS — MEDIUM GAIN)

**The ANE insight:** `matmul(write_mask, new_values)` writes T tokens to KV cache positions in a single operation. Loop-free, T-agnostic.

**Our fusion:** Our Hybrid OCT+PQ KV cache already does compression, but the **write path** still does per-slot updates:
```
for slot in selected_slots:
    cache[slot] = compress(new_kv)
```

The ANE pattern suggests:
```
cache_delta = matmul(slot_mask, compress_batch(new_kv))
cache = apply_delta(cache, cache_delta)
```

This is a batch write — useful when DDTree speculates multiple tokens and we need to write K/V for all of them at once (speculative acceptance). Currently we write one at a time after verification.

**Verdict:** ⚠️ CONDITIONAL GAIN — only helps during speculative acceptance (writing T>1 accepted tokens). Current acceptance rate is ~60-70% for 4 tokens. Benefit = batch write of 2-3 tokens. Marginal. Keep as **optional optimization** behind existing speculative feature flag. Not on by default.

---

### Fusion 4: **Soft-Routed Expert Dispatch for BanditPruner Arms** (MODELLESS — HIGH GAIN)

**The ANE insight:** MoE on ANE uses dense soft routing — run ALL experts but zero-mask non-selected. Avoids dynamic branching. Trades compute for static graph guarantees.

**Our fusion:** BanditPruner currently does **hard selection** — pick one arm (one ScreeningPruner configuration), use it exclusively. The ANE pattern suggests **soft routing** — score ALL arms, blend their relevance scores:

```rust
/// Soft-routed bandit: blend all arms instead of picking one
pub fn soft_route_relevance(
    arms: &[ArmConfig],
    state: &QueryState,
    temperature: f32,
) -> f32 {
    let weights: Vec<f32> = arms.iter()
        .map(|arm| (arm.ucb_score(state) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    
    arms.iter().zip(weights.iter())
        .map(|(arm, w)| arm.relevance(state) * (w / total))
        .sum()
}
```

**Why this is creative (not direct mapping):** The ANE does soft routing because the hardware demands static graphs. We do soft routing because **blending relevance scores from multiple pruners is more robust than picking one**. The math is the same (softmax-weighted mixture), the motivation is different (robustness vs hardware constraint).

**GOAT test:** Run 1000 queries with hard-select vs soft-blend. Measure:
- Valid node ratio (should be ≥100% like current)
- Acceptance rate (should improve, soft blend catches edge cases)
- Latency impact (soft blend is O(arms) per node — arms=4-8, negligible)

**Verdict:** ✅ GAIN — improves pruning quality without adding branching. The cost is O(arms) per relevance check, but arms=4-8 and relevance check is already O(1). No perf hurt. **Must be on by default** as the default bandit strategy.

---

### Fusion 5: **Shard Residency for CPU/GPU Auto-Route** (MODELLESS — INFRASTRUCTURE)

**The ANE insight:** Shard sizing is a first-class design variable. Models are split into heterogeneous chunks tuned per weight budget. Residency validation ensures each shard lands on the right hardware.

**Our fusion:** Our CPU/GPU auto-route currently routes at the **query level** (whole forward pass on CPU or GPU). The ANE pattern suggests **layer-level routing** — some layers always run on CPU (small, latency-sensitive), some always run on GPU (large, throughput-sensitive). 

This is already partially done (Kog CPU Fusion folds RMSNorm into CPU path). The ANE fusion is making this **systematic and validated**:

```rust
/// Layer residency map — which layers go where?
pub struct LayerResidencyMap {
    /// CPU-only layers (RMSNorm, small projections, pruner checks)
    pub cpu_layers: Vec<usize>,
    /// GPU-only layers (matmul, large FFN, attention)  
    pub gpu_layers: Vec<usize>,
    /// Validated at startup via micro-benchmark
    pub validated: bool,
}

impl LayerResidencyMap {
    /// Validate that layers actually land on intended hardware
    /// (ANE analogy: MLComputePlan residency check)
    pub fn validate(&self) -> bool {
        for &layer in &self.cpu_layers {
            let ns = benchmark_layer(layer, Cpu);
            assert!(ns < CPU_LAYER_THRESHOLD_NS, "Layer {} claimed CPU but too slow", layer);
        }
        // Similar for GPU layers
        true
    }
}
```

**Verdict:** ⚠️ CONDITIONAL GAIN — this is infrastructure, not algorithm. It helps when the system runs on heterogeneous hardware (CPU+GPU). Already partially implemented via feature flags. Make it **validated at startup** (like ANE residency check). Not on by default — requires GPU feature flag.

---

## Summary Verdict Table

| Fusion | Target | Gain | Perf Hurt | Default? | Research # |
|--------|--------|------|-----------|----------|------------|
| **1. Residency Audit** | DDTree + Pruners | HIGH — catches silent degradation | Zero (post-hoc test) | **YES** — test-only | This doc |
| **2. RangeBudget** | DDTree budget adaptation | HIGH — saves compute on easy queries | Zero — reduces work | **YES** — adaptive CoT | This doc |
| **3. Matmul Scatter KV** | KV cache batch write | MEDIUM — marginal batch speedup | Near zero | NO — optional | This doc |
| **4. Soft-Route Bandit** | BanditPruner arms | HIGH — more robust pruning | Near zero (O(arms)) | **YES** — default | This doc |
| **5. Layer Residency** | CPU/GPU routing | MEDIUM — systematic routing | Zero | NO — needs GPU | This doc |

## GOAT Decision per Verdict 003

Per [003_Commercial_Open_Source_Strategy_Verdict.md](003_Commercial_Open_Source_Strategy_Verdict.md):

- **Fusions 1, 2, 4** → modelless, inference-time only, no LLM training → ✅ lands in katgpt-rs domain
- **Fusion 3** → optional optimization, behind existing feature flag → ✅ lands in katgpt-rs domain
- **Fusion 5** → infrastructure, requires GPU → lands in riir-ai domain (see riir-ai research)

**The "Ferrari, no gas" test:** These fusions are engine improvements (better pruning, better budget, better routing). They make the engine more efficient without touching the fuel (lora.bin). This is exactly the right layer for katgpt-rs — MIT-licensed engine improvements that benefit everyone.

**Creative vs Direct:** None of these are direct mappings. The ANE solves hardware constraints; we solve inference-time search efficiency. The fusion is the **pattern** (validate what actually runs, single-config variable execution, soft-blend over hard-select), not the mechanism.
