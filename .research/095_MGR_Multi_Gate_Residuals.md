# Research 95: MGR — Multi-Gate Residuals

> **Paper:** [Multi-Gate Residuals](https://arxiv.org/abs/2605.23259)
> **Authors:** Zhizhan Zheng, Feiyun Zhang, Shuchun Liu, Tian Xia, Xi Liu, Dasheng Hu, Hongquan Zhou (Shanghai Yichuang, Fudan University)
> **Date:** May 2026
> **Verdict:** ⚠️ Partial distill — validation only, no new code.
> **GOAT Pillar:** ❌ Not a pillar — general transformer architecture, not game-specific. Evaluated against [MMO GOAT Pillars](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md): fails MMO-product (required), fails LoRA-independent (high). Stays in `katgpt-rs` domain.
> **Domain:** `katgpt-rs` — validates our existing `delta_routing` (Plan 097) design. No game IP, no selling point, no secret.
>
> MGR's AttnPool is structurally identical to our `depth_route()`. The novel parts (multi-stream residuals + gated lerp) are **training-time only**, requiring n× stream memory and weight format changes. Our codebase is inference-focused. Distill only the **convex-combination stability proof** (§3.2) for documentation. No new feature gate needed.

---

## Paper Summary

MGR proposes a residual connection architecture that maintains **n parallel hidden streams** through the network depth. Each stream acts as an independent memory bank. The core mechanism:

### Two-Phase Architecture
1. **Accumulation phase** (layers 0..n-1): Standard residual stream, one new stream initialized per layer.
2. **Lerping phase** (layers n..L): Streams are mixed with layer outputs via **gated interpolation**, then aggregated via **AttnPool** for the next layer's input.

### AttnPool (Aggregator) — Eq. 1
```
α_{i→l} = softmax(w_α · RMSNorm(s_i) / √d)    // per-stream attention weight
h_l = Σ α_{i→l} · s_i                           // weighted aggregation
```
This is essentially **depth attention over streams** — identical in structure to our `depth_route()` (Plan 097).

### Two Gating Variants (Mixers)

**Independent (Sigmoid):**
```
β_{i←l} = σ(w_β · RMSNorm(s_i) / √d + b)      // each stream gated independently
s'_i = (1 - β_i) ⊙ s_i + β_i ⊙ F_l(h_l)       // convex combination update
```

**Competitive (Softmax):**
```
β_{i←l} = softmax([forget_score, stream_scores...])[i+1]  // normalized across all streams + forget gate
s'_i = (1 - β_i) ⊙ s_i + β_i ⊙ F_l(h_l)                  // competitive update
```

### Stability Proof (Key Result — §3.2)
The convex combination `s'_i = (1-β)⊙s_i + β⊙F_l(h_l)` with β∈(0,1) guarantees:
- **Per-layer norm ceiling**: `‖s'_i‖ ≤ max(‖s_i‖, ‖F_l(h_l)‖)`
- **Global depth guarantee**: `‖x_L‖ ≤ max(‖x_0‖, max_l ‖F_l(x_l)‖)` — bounded by worst single layer, not cumulative
- This eliminates the "massive activation" problem (PreNorm dilution) without communication overhead

### Gate Bias Initialization — Eq. 14
```
b_init^(L) = ln(√(L/L_base) · (exp(-b_base) + 1) - n)
```
Scales with depth to prevent O(L) variance explosion. Reference: b_base = -3 at L_base = 21, n = 4.

### Practical Efficiency Strategies
- **Kernel fusion**: Lerp + AttnPool share the same stream memory (3nC reads + nC writes for fused op)
- **Fallback inversion**: Inverse-solve input streams from output streams during backward pass (only store top-p outliers), saving ~n× memory during training

### Results
- **S (0.12B), M (0.35B), L (0.77B)** on FineWeb-10BT, nanoGPT framework
- MGR (both variants) **outperforms** Full AttnRes, mHC-lite, Block AttnRes, PreNorm at all scales
- n=8 competitive variant is best overall
- Massive activations eliminated; gradient dilution resolved
- Block pruning shows **uniform depth utility** (no dead deeper layers)

---

## Mapping to Our Codebase

### What We Already Have (delta_routing, Plan 097)

| MGR Component | Our Equivalent | Status |
|---|---|---|
| AttnPool (Eq. 1) | `depth_route()` | ✅ Implemented — RMSNorm → dot-product query → softmax → weighted sum |
| Per-layer query vectors | `delta_routing_query` | ✅ In `TransformerWeights` |
| Per-layer norm weights | `delta_routing_norm` | ✅ In `TransformerWeights` |
| Block-delta accumulation | `block_deltas` | ✅ In `ForwardContext` |
| Routing logits buffer | `delta_routing_logits` | ✅ Temp buffer |

### What MGR Adds That We Don't Have

| MGR Component | Description | Relevance |
|---|---|---|
| Multi-stream residuals | n parallel hidden states maintained through depth | ⚠️ Training architecture — requires n× memory per layer during forward pass |
| Gated interpolation (lerp) | β-gated convex combination per stream | ⚠️ Training architecture — changes weight layout |
| Competitive forget gate | Softmax across streams + explicit forget bias | ⚠️ Training architecture |
| Bias initialization scaling | Eq. 14 for depth-proportional gate bias | 📐 Useful formula — applies to any gated architecture |
| Convex-combination stability proof | Per-layer norm ceiling, global depth bound | ✅ **Validates our existing `depth_route` design** — our additive routing is a special case |
| Fallback inversion | Training memory optimization | ❌ Training-only, not relevant |

### Modelless Distillation Potential

MGR is fundamentally a **model architecture** change, not a modelless technique. However:
1. The **AttnPool scoring pattern** (RMSNorm → dot-product → softmax → weighted aggregation) can be distilled as a **modelless pruning heuristic**: score accumulated block deltas, only keep top-k for routing.
2. The **stability analysis** (§3.2) gives us a theoretical justification for why our existing `depth_route` works — it's a convex combination at the block level.

---

## GOAT Pillar Evaluation

Evaluated against [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md):

| Criterion | Weight | Score | Reason |
|-----------|--------|-------|--------|
| GOAT passed | Required | ⬜ | No GOAT proof exists yet — Plan 134 proposes empirical norm stability test |
| MMO-product | Required | ❌ | General transformer architecture, no game-specific contribution |
| LoRA-independent | High | ❌ | Requires model weights (multi-stream params, gating biases) — not modelless |
| Defensible | Medium | ❌ | Paper is public, algorithm is straightforward, no game domain knowledge |
| Secret coverage | Medium | ❌ | No A/A2/B/C/D secret is strengthened by this |

**Result:** ❌ **Not a GOAT pillar.** Stays in `katgpt-rs` (open MIT domain) as infrastructure validation.

If multi-stream residuals were ever applied to **game AI** (e.g., separate streams for spatial/tactical/strategic reasoning in MCTS), that game-specific implementation would belong in `riir-ai` (private). But the base architecture is public and not a selling point.

---

## Verdict

**⚠️ PARTIAL DISTILL — infrastructure only, no new feature gate.**

### Reasons:
1. **Our `delta_routing` already implements the inference-time subset of MGR** — the AttnPool aggregator is structurally identical to `depth_route()`.
2. **MGR's novel contribution (multi-stream residuals + gated lerp) is a training-time architecture** — it requires n× stream memory, per-stream gating weights, and changes the weight checkpoint format. Our codebase is inference-only.
3. **The stability proof (§3.2) is valuable** but validates our existing design rather than requiring new code.
4. **No modelless distillation angle** — multi-stream residuals require model weights. The AttnPool scoring pattern is already captured.

### What to Distill:
- ✅ **Stability proof**: Add §3.2 convex-combination argument to our `depth_route` documentation — proves our additive routing can't cause unbounded activation growth
- ✅ **Bias initialization formula (Eq. 14)**: Record for future training infrastructure
- ✅ **Kernel fusion insight**: Lerp + AttnPool fusion pattern (3nC read + nC write) useful if we ever add multi-stream support

### What NOT to Distill:
- ❌ Multi-stream residual topology (training architecture, n× memory)
- ❌ Gated interpolation mixer (training-only)
- ❌ Fallback inversion (backward-pass optimization)
- ❌ Competitive vs independent gate comparison (training ablation)

---

## References
- MGR paper: arXiv:2605.23259
- Related: AttnRes (arXiv:2603.15031) — full depth attention
- Related: Hyper-Connections (arXiv:2409.19606) — multi-stream topology
- Related: mHC (arXiv:2512.24880) — manifold-constrained hyper-connections
- Our existing: Plan 097 (delta_routing), `depth_route()` in transformer.rs
