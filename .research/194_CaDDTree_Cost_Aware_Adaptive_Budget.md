# Research 194: CaDDTree — Cost-Aware Adaptive DDTree Budget Selection

## Papers
- **Primary**: arXiv:2606.01813 — "Cost-Aware Diffusion Draft Trees for Speculative Decoding" (Zhang, Qiu, He, Dai, June 2026)
- **Complementary**: arXiv:2605.29727 — "BASTION: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting" (Oh, Cao, Kim, Jung, Ahmad, Bae, Yun, May 2026)

## Key Findings

### CaDDTree (Primary)
1. **Acceptance length is non-decreasing in budget** → always favors bigger trees regardless of verification cost. Fixed budget selection has no principled basis.
2. **Token throughput is unimodal in budget** → has a sweet spot, provably findable by greedy search.
3. **Throughput decomposes into per-round 1-D search** over budget B, enabling efficient greedy stopping rule.
4. **Under convex verification cost, throughput function is unimodal** (Theorem 1) → never worse than fixed budget.
5. **Requires no offline budget search** — adapts budget each round from current per-position distributions and verification cost.
6. **Experiments on Qwen3-4B and Qwen3-8B across 8 benchmarks** — matches or surpasses DDTree with oracle budget selection.

### BASTION (Complementary)
1. **Acceptance surrogate** estimates expected accepted length from path confidence scores (product of top-k probabilities).
2. **Online latency estimator** calibrates hardware-aware roofline model.
3. **Adaptive best-first expansion** grows tree until marginal gains no longer justify incremental verification costs.
4. **Training-free, distribution-preserving** — no per-setting tuning required.
5. **Up to 6.61× speedup** over standard autoregressive decoding, 39% over SOTA block-diffusion baselines.

## Modelless Distillation Strategy

Both papers are **training-free** and **distribution-preserving** — ideal for modelless implementation:

| Component | Source | Our Mapping |
|-----------|--------|-------------|
| Acceptance Surrogate | BASTION §3.1 | `Π(1 - top_k_prob_i)` geometric acceptance from marginals + BFCP region quality (Plan 213) |
| Online Latency Estimator | BASTION §3.2 | Calibrate existing `SpecCostSnapshot` (Plan 096) + `RooflineCost` (Plan 159) into live latency model |
| Adaptive Budget Search | CaDDTree §4 | Replace fixed `Config::tree_budget` with unimodal search: expand greedily, stop when dT/dB < 0 |
| Unimodality Proof | CaDDTree Theorem 1 | Guarantees greedy stopping is optimal under convex verification cost |

## Infrastructure Dependencies

| Existing Feature | What It Provides |
|-----------------|-----------------|
| `bfcf_tree` (Plan 213) | Per-region marginal quality → feeds acceptance surrogate |
| `belief_drafter` (Plan 217) | Variable-length draft → adaptive lookahead |
| `bfcf_lfu_shard` (Plan 218) | Cached region quality → cheap surrogate computation |
| `spec_cost_model` (Plan 096) | `SpecCostSnapshot` with Amdahl decomposition → latency estimator seed |
| `roofline_cost` (Plan 159) | Hardware-aware roofline model → latency estimator baseline |
| `lodestar` (Plan 207) | Completion-distance pruning adapts tree shape — CaDDTree adapts tree *size* |

## Verdict

**GOAT-worthy.** Both papers prove the approach is:
1. Training-free (modelless)
2. Distribution-preserving (lossless)
3. Provably optimal (unimodality theorem)
4. Composable with existing infrastructure
5. Expected gain: +15-39% throughput over fixed budget with zero regression

## Implementation Plan → Plan 219

Feature gate: `caddtree_budget` (auto-enables `bfcf_tree`, `spec_cost_model`)
- Phase 1: Acceptance Surrogate (geometric estimate from marginals + region quality)
- Phase 2: Online Latency Estimator (EMA of draft/verify times, roofline baseline)
- Phase 3: Unimodal Budget Search (greedy stopping rule)
- Phase 4: Integration with DDTree pipeline
- Phase 5: GOAT verification

## TL;DR

CaDDTree proves token throughput is unimodal in tree budget under convex verification cost — greedy stopping is provably optimal, no offline sweep needed. BASTION provides the acceptance surrogate and online latency estimator to make it practical. Together they replace fixed `Config::tree_budget` with adaptive per-round budget selection, composable with existing BFCF/roofline/cost-model infrastructure. Expected +15-39% throughput, zero regression, fully training-free. Feature gate `caddtree_budget`, Plan 219.
