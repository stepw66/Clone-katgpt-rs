# Research 185: INSIGHT — Neuro-Symbolic RL with Symbolic Distillation & Explanation

**Paper:** [INSIGHT: Neuro-Symbolic RL with Symbolic Policy Distillation](https://arxiv.org/abs/2403.12451)
**Date:** 2026-06-07
**Status:** GOAT verdict: PROCEED — modelless fusions 1-4 are all engine (MIT), inference-time only
**Domain:** Modelless core (`katgpt-rs`) — engine. Concept grounding (F2) + decision traces (F3) → SaaS audit surface.
**Depends on:** DDTree AND-OR decomposition (Plan 190), BanditPruner, AbsorbCompressLayer, ScreeningPruner, TrialLog, RegressionSuite
**Sibling work:** 184 FOL-LNN (logical rule extraction), 206 EGCS (episode-guided constraints), 209 FOL plan (logical pipeline)

---

## TL;DR

INSIGHT proves a three-stage pipeline — **explore with expressive policy → distill into interpretable symbolic form → explain with language** — achieves strong RL performance while remaining human-readable. Our DDTree + BanditPruner + AbsorbCompress stack already performs the first two stages at inference time. The four fusions close the loop: (F1) fit compact symbolic expressions to DDTree accept/reject boundaries as new ScreeningPruner implementations, (F2) add concept grounding that maps pruner internals to human-readable explanations, (F3) add sensitivity-based decision explanation that attributes token choices to individual pruner scores, (F4) formalize reward-gated pruner calibration as the AbsorbCompress pattern extended to continuous parameters. All four are **modelless inference-time computation** — no LLM training, no neural network training. F1 and F4 are GOAT default-on. F2 and F3 are opt-in audit/development surfaces.

---

## 0. One-paragraph thesis

INSIGHT's core pattern is: train a neural policy (PPO) to explore, jointly distill it into an EQL (Equation Learner) symbolic expression with flexible activations (square, cube, identity, multiply, add, constant) and sparsity regularization, then use LLM-based concept grounding and gradient explanation to produce human-readable policy descriptions. Our katgpt-rs is modelless — no training at all — but the **architectural pattern maps perfectly**: DDTree's exploration IS the neural actor, ConstraintPruner/ScreeningPruner rules ARE the symbolic policy, AbsorbCompress IS the distillation (statistical→rule promotion), and what's missing is (1) fitting polynomial expressions to exploration boundaries (EQL analogue, inference-time), (2) mapping pruner states to natural language (concept grounding), (3) sensitivity-based attribution (gradient analogue, pure computation), and (4) formalizing reward-weighted threshold calibration. The result: a self-improving inference engine that explores, distills symbolic knowledge, and explains its decisions — all without ever training a model.

---

## 1. Paper Summary

### Core Method

INSIGHT (arXiv 2403.12451) is a Neuro-Symbolic RL framework with three components:

1. **Vision Distillation → Structured State**: Distill vision foundation models (FastSAM + DeAot) into a lightweight CNN that outputs object coordinates as structured state representations. This replaces raw pixel inputs with discrete symbolic states (positions, categories).

2. **Neural Guidance → Symbolic Policy Distillation**: A neural actor (PPO) explores the environment while an EQL (Equation Learner) actor jointly distills the policy into concise symbolic expressions. EQL uses flexible activation functions: {square, cube, identity, multiply, add, constant}. Sparsity regularization (L1 on expression complexity) ensures interpretability. Joint training: neural actor provides exploration breadth, EQL provides interpretable compactness.

3. **LLM-Based Explanation**:
   - **Concept grounding**: Associate EQL policy variables with task semantics (e.g., `x₁ = "distance to nearest enemy"`)
   - **Chain-of-thought interpretation**: Step-by-step policy explanation
   - **Gradient-based decision explanation**: ∂(action log-likelihood) / ∂(object coordinates) — which input features most influenced the decision

### Key Results

- Competitive performance with pure neural RL baselines on Atari and robotic control
- **10-100× more interpretable**: Symbolic policies are human-readable equations
- Concept grounding accuracy >90% when evaluated by domain experts
- Gradient explanations correctly identify causal input features in >85% of cases

### What Makes It Work

1. **Structured state representation** — discrete object coordinates enable symbolic reasoning
2. **EQL with sparsity** — flexible activations find compact expressions; L1 prevents bloat
3. **Joint neural-symbolic training** — neural explores, symbolic distills, both improve
4. **Three-layer explanation** — concept grounding (what) + chain-of-thought (why) + gradient (how much)

---

## 2. GOAT Verdict

| Criterion | Assessment |
|-----------|------------|
| **Modelless compatibility** | ✅ All four fusions are inference-time. No training. |
| **Existing trait alignment** | ✅ Maps directly to ScreeningPruner, ConstraintPruner, BanditPruner, AbsorbCompress |
| **Code complexity** | ⚠️ F1 (EQL analogue) is the most complex. F2-F4 are straightforward. |
| **Performance risk** | ✅ F1-F3 are post-hoc or off-path. F4 is the existing AbsorbCompress pattern formalized. |
| **Commercial alignment** | ✅ F1+F4 are engine (MIT). F2+F3 create audit SaaS surface. |
| **Sigmoid compliance** | ✅ All scoring uses sigmoid bounds. No softmax anywhere. |
| **GOAT gateable** | ✅ Single `insight_explain` feature flag with sub-gates per fusion. |

**Verdict: PROCEED.** INSIGHT's architecture is not a direct dependency — we don't need their vision module, their PPO trainer, or their LLM. But the **pattern** of explore→distill→explain maps 1:1 to our modelless stack, and the four fusions fill genuine capability gaps (symbolic expression fitting, human-readable explanation, sensitivity attribution, reward-gated calibration).

---

## 3. Architecture Alignment

### What We Already Have (No Changes Needed)

| INSIGHT Component | katgpt-rs Equivalent | Status |
|-------------------|---------------------|--------|
| Neural actor (PPO) | `SpeculativeGenerator` via DDTree exploration | ✅ Exists |
| Symbolic policy | `ConstraintPruner::is_valid()` + `ScreeningPruner::relevance()` | ✅ Exists |
| Joint training (neural↔symbolic) | `AbsorbCompressLayer` (bandit→rule promotion) | ✅ Exists |
| Reward signal | `GameState::reward()` + `ReplayReward` trait | ✅ Exists |
| Episode persistence | `TrialLog` (JSONL) | ✅ Exists |
| Regression testing | `RegressionSuite` (golden episode replay) | ✅ Exists |
| Hot-swap for rules | `HotSwapPruner` (runtime WASM reload) | ✅ Exists |
| Forward model | `GameState` trait (`advance`, `is_terminal`, `reward`, `actions`) | ✅ Exists |
| Non-terminal eval | `StateHeuristic` trait | ✅ Exists |

### What's New (4 Fusions)

| Fusion | INSIGHT Inspiration | katgpt-rs Novelty | Complexity |
|--------|--------------------|--------------------|------------|
| F1: Symbolic Expression Distillation | EQL symbolic policy | Fit polynomial expressions to DDTree accept/reject boundaries | Medium |
| F2: Concept Grounding | LLM concept grounding | Template-based pruner→natural language mapping | Low |
| F3: Decision Explanation | Gradient-based explanation | Sensitivity analysis on pruner scores | Low |
| F4: Reward-Gated Calibration | Vision→perception refinement | Formalize AbsorbCompress for continuous parameters | Low |

---

## 4. Fusion Analysis

### Fusion 1: Symbolic Distillation of Speculative Paths (MODELLESS)

**INSIGHT maps to:** EQL symbolic policy distillation (neural actor → interpretable expressions)

**The modelless twist:** We don't train an EQL. We *fit* compact symbolic expressions to DDTree exploration traces at inference time.

```
DDTree explores many token paths
    │
    ├── Accept/reject decisions form a labeled dataset:
    │   (state_features, pruner_scores) → accept/reject
    │
    ├── SymbolicExpressionFitter (NEW)
    │   ├── Candidate basis functions: {x, x², x³, sin(x), const}
    │   ├── Greedy forward selection with sparsity budget (max_terms: usize)
    │   ├── Fit to accept/reject boundary via least-squares
    │   └── Output: compact polynomial expression as ScreeningPruner
    │
    └── New ScreeningPruner implementation: ExpressionPruner
        ├── expression: SymbolicExpression (serializable, human-readable)
        ├── relevance(depth, token, parents) → evaluate expression
        └── Self-improving: absorb-compress promotes stable expressions to hard constraints
```

**Why modelless works:** The EQL in INSIGHT is trained end-to-end with gradient descent. But our "training data" already exists — it's the DDTree exploration trace. Every token that was accepted or rejected, along with the ScreeningPruner scores at each depth, IS a supervised dataset. Fitting a polynomial to this boundary is pure computation — no gradient descent, no backprop, no training loop. The result is a human-readable expression like:

```
relevance = sigmoid(0.7 × syntax_score + 0.2 × bandit_q - 0.1 × depth_penalty)
```

**Implementation:**
- `SymbolicExpression { terms: Vec<Term>, bias: f32 }` where `Term { basis: BasisFn, coefficient: f32, feature_idx: usize }`
- `BasisFn` enum: `Identity`, `Square`, `Cube`, `Sigmoid` (no softmax)
- `SymbolicExpressionFitter { max_terms: usize, min_improvement: f32 }`
- `ExpressionPruner<P>` wraps inner pruner + fitted expression
- Feature-gated: `insight_explain` → `symbolic_distill` sub-gate

**Performance:** Fitting is post-DDTree, off the hot path. Expression evaluation is O(terms) — negligible vs. inner pruner cost.

---

### Fusion 2: Concept Grounding for Pruner Rules (MODELLESS)

**INSIGHT maps to:** Concept grounding (variable names → task semantics)

**The modelless twist:** No LLM at inference time. Template-based grounding with LLM-optional development mode.

```
ConstraintPruner/ScreeningPruner internals
    │
    ├── ConceptGrounding trait (NEW)
    │   ├── fn ground(pruner_state: &PrunerState) -> Vec<ConceptMapping>
    │   ├── Template-based: pre-defined vocab→semantic mappings
    │   └── LLM-optional: during development, use LLM to validate/refine templates
    │
    ├── ConceptMapping (NEW)
    │   ├── variable: String (e.g., "score_0.8")
    │   ├── semantic: String (e.g., "syntax validity threshold")
    │   ├── confidence: f32 (sigmoid-bounded)
    │   └── source: GroundingSource { Template, Learned }
    │
    └── PolicyExplanation (NEW)
        ├── mappings: Vec<ConceptMapping>
        ├── chain_of_thought: Vec<String> (template-filled reasoning steps)
        └── summary: String (human-readable)
```

**Example output:**
```
Token 'fn' at depth 2 was accepted because:
1. Syntax pruner scored it 0.92 (high syntax validity)
2. Bandit Q-value was 0.71 (historically good arm)
3. No constraint violations detected
→ Concept: "function declaration at expected position"
```

**Implementation:**
- `ConceptGrounding` trait with `fn ground(&self, ...) -> Vec<ConceptMapping>`
- `TemplateGrounding` — static implementation with vocab→semantic lookup tables
- `PolicyExplanation` struct — serializable, JSONL-loggable via TrialLog
- No LLM dependency at runtime. LLM-optional during development for template generation.

**Performance:** Template lookup is O(1) — pre-computed `HashMap<String, String>`. Zero hot-path cost.

---

### Fusion 3: Gradient-Based Decision Explanation for Inference Paths (MODELLESS)

**INSIGHT maps to:** Gradient-based decision explanation (∂action / ∂input)

**The modelless twist:** No gradients (no differentiable model). Instead, perturbation-based sensitivity analysis.

```
DDTree exploration trace
    │
    ├── DecisionExplainer trait (NEW)
    │   ├── fn explain(trace: &DDTreeTrace) -> DecisionExplanation
    │   ├── Records which ScreeningPruner scores influenced each token
    │   └── Computes sensitivity by perturbation
    │
    ├── Sensitivity computation (MODELLESS):
    │   ├── For each token choice, perturb each pruner score ±δ
    │   ├── Re-run accept/reject with perturbed scores
    │   ├── Measure: would the output change?
    │   └── Attribution: "If pruner P scored 10% higher, output changes from X to Y"
    │
    └── DecisionExplanation (NEW)
        ├── token_choices: Vec<TokenChoice>
        ├── alternatives_rejected: Vec<RejectedAlternative>
        └── sensitivity_report: Vec<SensitivityEntry>
```

**Example output:**
```
Token 'struct' was chosen over 'enum' at depth 0:
- Syntax pruner: struct=0.85, enum=0.72 (Δ=0.13)
- Bandit Q-value: struct=0.68, enum=0.51 (Δ=0.17)
- Sensitivity: If syntax pruner had scored 'enum' 0.15 higher, 'enum' would have been selected
→ Primary driver: bandit Q-value (Δ=0.17 > 0.13)
```

**Implementation:**
- `DecisionExplainer` trait with `fn explain(&self, trace: &[TraceNode]) -> DecisionExplanation`
- `PerturbationExplainer` — perturbs scores ±δ and re-evaluates
- `TraceNode` — lightweight DDTree path recording (already partially exists in DDTreeBranchCache)
- Caching: sensitivity is computed once per trace, not per token
- Feature-gated: `insight_explain` → `decision_explain` sub-gate

**Performance:** Sensitivity analysis is O(tokens × pruners × δ_perturbations). Run post-inference or async. Not on hot path.

---

### Fusion 4: End-to-End Reward-Gated Pruner Calibration (MODELLESS)

**INSIGHT maps to:** Vision→perception refinement with reward signals

**The modelless twist:** This IS AbsorbCompress extended to continuous parameters. We already have the pattern.

```
Compilation success/failure (reward signal)
    │
    ├── RewardGatedCalibrator<P> (NEW, formalizes existing pattern)
    │   ├── Tracks which pruner parameter values led to high-reward outputs
    │   ├── Bandit-style updates: adjust thresholds toward high-reward regions
    │   ├── Papaya lock-free HashMap for parameter→reward tracking
    │   └── Promotes stable calibrated parameters to hard constraints via AbsorbCompress
    │
    ├── CalibrationStep (NEW)
    │   ├── parameter_id: usize
    │   ├── old_value: f32
    │   ├── new_value: f32
    │   ├── reward_delta: f32
    │   └── blake3 hash for audit trail
    │
    └── Self-improving cycle:
        Infer → Calibrate → Absorb → Hot-swap → Verify (RegressionSuite) → Repeat
```

**Why this already works:** AbsorbCompress already promotes stable low-Q bandit arms to hard constraints. This fusion formalizes the same pattern for continuous pruner parameters (thresholds, weights, relevance multipliers). The innovation is treating *parameters* as bandit arms — each parameter value is an arm, reward is compilation success, and AbsorbCompress promotes well-calibrated parameter values to fixed constraints.

**Implementation:**
- `RewardGatedCalibrator<P: ScreeningPruner>` — wraps inner pruner with parameter tracking
- Uses papaya lock-free HashMap for `ParameterKey → (reward_sum, visits)` — zero contention
- `CalibrationStep` struct for audit trail — blake3 hashed, JSONL-logged
- Integration with existing `AbsorbCompressLayer` — calibrated parameters get absorbed
- Integration with existing `RegressionSuite` — verify calibration doesn't regress

**Performance:** Parameter updates are O(1) per token via papaya. Absorption check is the existing AbsorbCompress path.

---

## 5. Alignment with Existing Architecture

### Trait-Level Mapping

```
INSIGHT                          katgpt-rs
─────────────────────────────    ─────────────────────────────
Neural Actor (PPO)           →   SpeculativeGenerator (DDTree)
EQL Symbolic Policy          →   ConstraintPruner + ScreeningPruner
Joint Training Loop          →   BanditPruner + AbsorbCompress
Reward Signal                 →   GameState::reward() + ReplayReward
Episode Buffer                →   TrialLog (JSONL)
Regression Tests              →   RegressionSuite (golden replay)
Hot-Swap                      →   HotSwapPruner (WASM reload)
Concept Grounding (LLM)      →   ConceptGrounding trait (template)
Gradient Explanation          →   DecisionExplainer trait (perturbation)
Parameter Calibration         →   RewardGatedCalibrator (formalized)
```

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/pruners/symbolic_expression.rs` | Create | `SymbolicExpression`, `SymbolicExpressionFitter`, `BasisFn` |
| `src/pruners/expression_pruner.rs` | Create | `ExpressionPruner<P>` ScreeningPruner impl |
| `src/pruners/concept_grounding.rs` | Create | `ConceptGrounding` trait, `TemplateGrounding`, `PolicyExplanation` |
| `src/pruners/decision_explainer.rs` | Create | `DecisionExplainer` trait, `PerturbationExplainer` |
| `src/pruners/reward_calibrator.rs` | Create | `RewardGatedCalibrator<P>`, `CalibrationStep` |
| `src/pruners/mod.rs` | Modify | Add new modules behind `insight_explain` feature gate |
| `Cargo.toml` | Modify | Add `insight_explain` feature with sub-gates |
| `examples/insight_demo.rs` | Create | Demonstrates all four fusions end-to-end |

### Existing Files Referenced

| File | Relevance |
|------|-----------|
| `src/pruners/bandit.rs` | `BanditPruner<P>` — F1 fitting uses bandit traces |
| `src/pruners/absorb_compress.rs` | `AbsorbCompressLayer<P>` — F4 extends to continuous params |
| `src/pruners/hot_swap.rs` | `HotSwapPruner<P>` — F4 hot-swaps calibrated parameters |
| `src/pruners/trial_log.rs` | `TrialLog` — F2/F3 explanations logged to JSONL |
| `src/pruners/regression.rs` | `RegressionSuite` — F4 verification |
| `src/pruners/mod.rs` | Module registration |
| `crates/katgpt-core/src/traits.rs` | `ScreeningPruner`, `ConstraintPruner`, `SpeculativeGenerator`, `GameState`, `StateHeuristic` |
| `src/speculative/types.rs` | `DDTreeBranchCache` — F1/F3 trace source |

---

## 6. Decision: Fusion Priority

| Priority | Fusion | Rationale | Effort | Impact |
|----------|--------|-----------|--------|--------|
| **P0** | F4: Reward-Gated Calibration | Formalizes existing pattern. Low risk. Immediate value. | 2 days | High — closes the self-improving loop |
| **P1** | F1: Symbolic Expression Distillation | Core innovation. EQL analogue. Enables F2. | 5 days | High — human-readable pruner expressions |
| **P2** | F2: Concept Grounding | Depends on F1 expressions. Template-based. | 2 days | Medium — audit surface for SaaS |
| **P3** | F3: Decision Explanation | Depends on F1 traces. Perturbation-based. | 3 days | Medium — debugging/audit tool |

**Recommended approach:** Ship F4 first (it's formalizing what we already have). Then F1 (the core novelty). F2 and F3 follow naturally from F1's outputs.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| F1 symbolic fitting quality | GOAT gate: expression accuracy must ≥80% on DDTree boundary before promotion |
| F1 overfitting to recent traces | Sparsity budget (`max_terms`) + L1 regularization analogue |
| F2 template staleness | HotSwapPruner for template updates; LLM-optional during development |
| F3 perturbation cost | Run async/post-inference. Cache results. O(tokens × pruners) is bounded. |
| F4 parameter oscillation | AbsorbCompress's stability criteria (min visits + low variance) prevent premature absorption |
| Feature flag bloat | Single `insight_explain` parent gate with 4 sub-gates. Each independently disableable. |

---

## 8. What Goes to riir-ai (NOT katgpt-rs)

| Component | Why riir-ai |
|-----------|-------------|
| Training EQL networks from scratch | Model-based training — not modelless |
| LLM-based concept grounding refinement | Model-based — template generation is development-time only |
| Neural perception module (FastSAM+DeAot distillation) | Model-based vision training |
| Gradient descent on pruner parameters | Training loop — our F4 uses bandit-style updates, not SGD |

---

## 9. References

- INSIGHT paper: arXiv 2403.12451
- EQL (Equation Learner): Sahoo et al., "Learning Equations for Extrapolation" (ICML 2018)
- `.research/184_FOL_LNN_Inference_Time_Logical_Rules.md` — sibling work on logical rule extraction
- `.plans/209_fol_logical_rule_inference.md` — FOL pipeline implementation plan
- `.plans/190_and_or_dtree_blueprint_decomposition.md` — DDTree AND-OR decomposition
- `.plans/206_episode_guided_constraint_synthesis.md` — EGCS/EpisodePruner
- `.research/037_REAP_Model-Based_Modelless_Duality.md` — model-based/modelless spectrum
- `.research/080_VPD_Variational_Policy_Distillation.md` — variational EM distillation pattern

---

TL;DR: INSIGHT's explore→distill→explain pattern maps 1:1 to katgpt-rs's modelless stack. Four fusions add: (F1) EQL-like symbolic expression fitting from DDTree traces, (F2) template-based concept grounding for pruner rules, (F3) perturbation-based decision explanation, (F4) formalized reward-gated pruner calibration. All modelless, all inference-time, all engine (MIT). Ship F4 first (formalizes existing pattern), then F1 (core novelty), then F2+F3 (audit surface). Feature-gated under `insight_explain`.
