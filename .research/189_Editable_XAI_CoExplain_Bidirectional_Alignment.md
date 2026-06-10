# Research: Editable XAI — CoExplain Bidirectional Alignment (Modelless)

**Date:** 2026-06
**Status:** Research → Verdict
**Context:** katgpt-rs (MIT) modelless inference engine
**Paper:** "Editable XAI: Toward Bidirectional Human-AI Alignment with Co-Editable Explanations of Interpretable Attributes" (CHI '26, arXiv:2602.12569v1)

---

## 1. Paper Summary

CoExplain is a neuro-symbolic framework for **bidirectional human-AI alignment** through editable decision tree explanations. Three interaction modes:

| Mode | Direction | Mechanism | Analogy |
|------|-----------|-----------|---------|
| **Read** | AI → Human | Distill NN → decision tree | Our DDTree path extraction |
| **Write** | Human → AI | Parse user DT → equivalent NN | Our ConstraintPruner synthesis |
| **Enhance** | Human ↔ AI | Train with TED regularization | Our bandit pruner refinement |

User study (N=43): CoExplain improved user-AI faithfulness (Cohen's d = 0.87–2.01) and reduced editing effort by 53% vs manual editing. 89.74% AI suggestion acceptance rate.

## 2. Core Mechanisms (Distilled for Our Stack)

### 2.1 DT → NN Parsing (Write Mode)

Each DT decision node maps to a **pair of sigmoid neurons** with biases `±τ`:

```
Layer 1 (decision): a₁ = σ(W₁ᵀ x + b₁)  — sigmoid with threshold biases
Layer 2 (conjunction): a₂ = ReLU(a₁_prev + a₁_curr - 1)  — Łukasiewicz AND
Output (disjunction): ŷ = cReLU(Σ a₂)  — clipped ReLU for OR
```

Zero-padding adds unused neurons/connections for future extensibility (training updates them).

**Our mapping**: This IS our `ConstraintPruner` trait. User rules → pruner implementations (WASM or native). The "parsing" is compilation of human-readable rules into executable constraint checks.

### 2.2 NN → DT Distillation (Read Mode)

CART-based tree learning on NN predictions. Tree predicts explanation label `ỹ` matching NN prediction `ŷ`. Hyperparameter tuning on tree depth for faithfulness.

**Our mapping**: This IS our DDTree path extraction + the AND-OR decomposition (Plan 190). We already extract decision paths from marginal distributions.

### 2.3 TED-Regularized Training (Enhance Mode)

Multi-objective loss with **Tree Edit Distance (TED) proxy model**:

```
L = L_data(ŷ, y) + λ_b · L_behavior(ỹ, ỹ') + λ_t · L_topology(d̂, d)
```

- `L_data`: Standard cross-entropy on ground truth
- `L_behavior`: Cross-entropy between distilled DT prediction and user DT prediction
- `L_topology`: MSE between proxy-predicted TED and actual TED
- `F_d(θ)`: Proxy model mapping NN parameters → predicted TED distance

Two enhancement types:
- **Threshold**: Freeze all except layer-1 biases (thresholds). Conservative.
- **Topology**: Unfreeze all + zero-padded neurons. Aggressive. User-locked rules get higher TED penalty.

**Our mapping**: Threshold enhancement = our bandit Q-value refinement on pruner thresholds. Topology enhancement = our AND-OR decomposition restructuring. The TED metric is new — we don't measure structural divergence from user intent.

## 3. Application to katgpt-rs

### 3.1 Direct Mapping Table

| CoExplain Concept | katgpt-rs Equivalent | Status | Gap |
|-------------------|---------------------|--------|-----|
| Decision tree rules | DDTree paths (speculative decoding) | ✅ Exists | — |
| DT → NN parsing | ConstraintPruner synthesis from rules | ✅ Partial | Need rule→WASM compiler |
| NN → DT distillation | DDTree + AND-OR decomposition | ✅ Exists | — |
| Threshold enhancement | BanditPruner Q-value updates | ✅ Exists | — |
| Topology enhancement | AND-OR restructuring (Plan 190) | ✅ Exists | — |
| TED regularization | Structural divergence metric | ❌ Missing | New — proxy model not needed for modelless |
| User-written rules | Curator WASM validators | ✅ Exists | Need curator-facing rule editor |
| Enhancement constraints (λ sliders) | Bandit temperature/exploration rate | ✅ Partial | Need user-controllable params |
| Zero-padding extensibility | Extra DDTree branches | ❌ Missing | Minor — allocate unused capacity |

### 3.2 What We Already Have (Strong Foundation)

1. **ConstraintPruner trait** — the exact interface for "write mode" (user rules → constraint checks)
2. **SynPruner + PartialParser** — two-tier validation pipeline (matches CoExplain's threshold/topology split)
3. **BanditPruner** — self-refining pruner with Q-value updates (matches CoExplain's enhance)
4. **AND-OR DDTree** — structural decomposition with memoized subgoals (matches CoExplain's topology)
5. **CompilerFeedback** — extracts suggestions for self-correction (matches CoExplain's L_behavior)
6. **ScreeningPruner** — relevance-based gating (analogous to CoExplain's prediction similarity)

### 3.3 What's Missing (Gaps)

1. **Bidirectional rule editing** — currently one-way (pruners validate, but developers can't edit rules at inference time)
2. **TED-equivalent divergence metric** — no way to measure how far pruner behavior has drifted from original specification
3. **Curator rule editor** — no UI for writing/editing constraint rules (Web UI planned but not built)
4. **Episode → rule synthesis** — episodes accumulate but don't generate new pruner rules automatically

## 4. Novel Fusion Ideas

### Fusion 1: CoEditable ConstraintPruner — Bidirectional Developer↔Pruner Alignment

**What**: Allow developers/Curators to edit constraint rules and have them take effect immediately in the inference pipeline. Currently, pruners are compiled ahead of time. Adding runtime edit capability enables CoExplain's "Write" mode.

**Mechanism**:
1. Developer writes/edit decision tree rules via rule editor (Web UI or MCP)
2. Rules compile to WASM ConstraintPruner via existing `riir-validator-sdk`
3. Hot-swap into inference pipeline (already supported via papaya lock-free pool)
4. Observe pruning accuracy → feed back to developer (Read mode)
5. Bandit refines thresholds based on acceptance rates (Enhance mode)

**Modelless fit**: ✅ Pure inference-time. No training. WASM compilation is deterministic.

### Fusion 2: Self-Refining Pruner — Inference-Time Constraint Evolution

**What**: CoExplain's enhance loop applied to ConstraintPruner accuracy. Currently BanditPruner updates Q-values. Adding CoExplain's threshold/topology distinction:
- **Threshold mode**: Adjust SynPruner rejection thresholds based on compiler feedback (how often did accepting this token lead to compile failure?)
- **Topology mode**: Restructure DDTree branching patterns based on observed acceptance rates (which branches are productive vs dead ends?)

**Mechanism**:
1. Track per-pruner accuracy: TP (rejected invalid), TN (accepted valid), FP (accepted invalid), FN (rejected valid)
2. Threshold adjustment: sigmoid-scaled acceptance threshold based on FP/FN ratio
3. Topology adjustment: prune low-value DDTree branches, expand high-value ones
4. TED constraint: penalize changes that diverge from developer's original rule specification

**Modelless fit**: ✅ Pure statistics. Bandit updates are O(1) per token. No training.

### Fusion 3: Neuro-Symbolic RIIR Feedback Loop — CoExplain × Curator Marketplace

**What**: Apply CoExplain's Read/Write/Enhance cycle to the full RIIR pipeline, enabling the Curator marketplace (Verdict 003 Phase 5) WITHOUT needing lora.bin first.

**Mechanism**:
1. **Read**: Extract decision rules from successful RIIR translations (which patterns worked for which Python→Rust constructs)
2. **Write**: Curators write translation rules as decision trees via Web UI/MCP agent
3. **Enhance**: Bandit-optimized threshold refinement on translation rules, using compiler feedback as ground truth

This directly enables the Curator model from Verdict 003:
- Method A (GitHub Pick): Platform translates, extracts rules, Curator reviews/refines
- Method B (Link Resource): Curator writes rules from specs, platform validates against code

**Commercial impact**: Unblocks Phase 5 (Curator Beta) without waiting for Phase 4 (Data Flywheel). Curators can start contributing rules immediately, the bandit refines them, and the accumulated rules BECOME the training data for lora.bin later.

**Modelless fit**: ✅ All inference-time. Curator rules → WASM validators. Enhancement via bandit + compiler feedback.

### Fusion 4: TED-Lite — Structural Divergence Metric for Pruners

**What**: CoExplain uses Tree Edit Distance to measure how far enhanced explanations diverge from user rules. We don't need the full TED algorithm (ZSS) — we can compute a cheaper structural divergence metric for DDTree paths.

**Mechanism**:
1. Snapshot developer's original pruner specification as a "golden tree"
2. As bandit refines thresholds/topology, compute lightweight divergence:
   - Threshold divergence: Σ |τ_current - τ_original| / N_thresholds
   - Topology divergence: Hamming distance on branch existence vectors
3. Clamp adjustments when divergence exceeds developer-configurable λ_t
4. Expose as diagnostic metric in inference logs

**Modelless fit**: ✅ Pure arithmetic. O(k) per pruner where k = number of thresholds.

## 5. GOAT Verdict

**Fusion 3 (Neuro-Symbolic RIIR Feedback Loop) is GOAT.**

Why:
1. **Directly enables commercial strategy** — Curator marketplace without lora.bin
2. **Novel combination** — CoExplain's Read/Write/Enhance × RIIR pipeline × bandit refinement is new
3. **No perf hurt** — WASM validators are already in the hot path, bandit is O(1)
4. **Self-reinforcing** — Curator rules → better translations → more rules → flywheel

**Fusion 2 (Self-Refining Pruner) is runner-up.** It's pure modelless improvement to existing infrastructure, but narrower commercial impact. Still worth doing because it's zero-risk.

**Fusion 1 (CoEditable ConstraintPruner) is a prerequisite** for Fusion 3. You can't have the feedback loop without editable pruners.

**Fusion 4 (TED-Lite) is enabling infrastructure** for Fusions 1-3. Without it, there's no guardrail against pruner drift.

## 6. Gains

| Fusion | Gain | Perf Impact | Risk |
|--------|------|-------------|------|
| Fusion 1 (CoEditable) | Enables runtime rule editing | None (WASM hot-swap) | Low |
| Fusion 2 (Self-Refining) | Improves pruner accuracy over time | None (O(1) bandit) | None |
| Fusion 3 (RIIR Loop) | Unlocks Curator marketplace early | None (existing infra) | Medium (new UI) |
| Fusion 4 (TED-Lite) | Prevents pruner drift | Negligible (O(k) metric) | None |

All are modelless. No LLM training. No perf hurt. All can be feature-gated.

## 7. Recommendation

### Default ON (after GOAT proof, no perf hurt)
- **Fusion 2**: Self-Refining Pruner — extends existing BanditPruner, zero risk, pure accuracy improvement
- **Fusion 4**: TED-Lite — small diagnostic metric, enables safe pruner evolution

### Feature-Gated
- **Fusion 1**: `coexplain_pruner` — CoEditable ConstraintPruner (new trait methods + rule editor backend)
- **Fusion 3**: `coexplain_riir` — Full RIIR feedback loop (depends on Fusion 1 + Curator API)

### Execution Order
1. Fusion 4 (TED-Lite) — 1-2 days, enables safe pruner evolution
2. Fusion 2 (Self-Refining) — 3-5 days, extends BanditPruner
3. Fusion 1 (CoEditable) — 1-2 weeks, new trait + rule editor backend
4. Fusion 3 (RIIR Loop) — 2-4 weeks, full Curator integration

### Tests/Examples Required
- Before/after pruner accuracy: show bandit-refined thresholds catch more invalid tokens
- Before/after DDTree quality: show TED-clamped pruners produce more valid branches
- Curator rule → WASM → DDTree integration test
- Self-refining pruner accuracy over N iterations (convergence proof)

---

## 8. Key Quotes from Paper

> "Users can adapt AI reasoning to their domain knowledge, while simultaneously deepening their own understanding of how the AI makes decisions." — Bidirectional alignment thesis

> "Threshold Enhancement will retain the same decision tree structure (topology), but may change the threshold value." — Conservative update pattern matches our BanditPruner

> "CoExplain struck a balance between maintaining alignment with users' initial knowledge and achieving near-optimal model performance." — The exact balance we need for Curator rules

> "Editable XAI should be framed as a dialogic process, where explanations are shaped through iterative contributions from both human and AI." — Our inference-time bandit refinement is this dialogic process

---

TL;DR: CoExplain's Read/Write/Enhance neuro-symbolic cycle maps cleanly to our ConstraintPruner → DDTree → bandit pipeline. The GOAT fusion is applying this to the RIIR Curator marketplace, enabling Curators to write rules NOW without waiting for lora.bin. Default ON: self-refining pruner + TED-Lite divergence metric. Feature-gated: CoEditable pruners + full RIIR feedback loop.
