# Research 96: HL-ImageNet — Symbolic Pipeline Overfitting & Code-as-Model Regularization

> **Source:** [Heuristic Learning for Symbolic ImageNet-10 (Phase 2)](https://github.com/xisen-w/hl-imagenet) by Xisen Wang, 2026
> **Reference:** Weng, J. (2026). *Learning Beyond Gradients*. (Research 014)
> **Date:** 2026-05 (paper), distilled 2026-05
> **Verdict:** ⚠️ **Partial distill — conceptual validation + regularization framework only. No new code.**
> **GOAT Pillar:** ❌ Not a pillar — perception-domain HL insights, not game-specific. Evaluated against [MMO GOAT Pillars](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md): fails MMO-product (required), fails LoRA-independent (high — conclusions apply to both model-based and modelless, but the paper itself is about image classification, not games). Stays in `katgpt-rs` domain.
> **Domain:** `katgpt-rs` — validates our existing heuristic learning infrastructure (Plan 032, bandit feature). The key insight (code-as-model overfitting + patch regularization) is a **design principle** for our existing `AbsorbCompress` + `BanditPruner` loop, not a new feature.

---

## Summary

HL-ImageNet Phase 2 is a controlled experiment that treats a **symbolic image classifier as a heuristic learning system** — a coding agent iteratively edits Python rules, evaluates on train/val, and keeps or reverts patches. The key finding is not that symbolic vision works well (it doesn't — 51.9% val on ImageNet-10). The key finding is that **code-as-model exhibits the same overfitting dynamics as neural networks**, and that current HL systems lack the regularization machinery to prevent it.

This directly validates and sharpens our existing heuristic learning design (Research 014, Plan 032, `bandit` feature).

---

## The 5 Findings Distilled

### Finding 1: Fitting Is Doable (Even Symbolically)
- A coding agent can drive train accuracy from random → 84–100% on ImageNet-10 using only hand-coded visual rules + pairwise reranking + verify rules
- **Our mapping:** Our `AbsorbCompress` cycle (Plan 032) already does this — bandit arms that win get absorbed into the pruner. The codebase *is* the model. We already know this works for Bomber (+177 win, Plan 033).

### Finding 2: Generalization Is The Hard Part
- `base_rerank`: 55.4% train / 51.9% val (gap: 3.5pp)
- `full verify`: 84.0% train / 50.5% val (gap: 33.5pp)
- Verify rules memorize training examples; pairwise reranking generalizes best
- **Our mapping:** This is the **critical lesson** for our HL loop. Our `BanditPruner` arms are evaluated on train episodes. If we don't enforce held-out selection, we'll accumulate the same kind of narrow, brittle rules.

### Finding 3: Reranking > Verify Rules (Generalization Hierarchy)
```
base visual statistics  → moderate transfer
pairwise reranking      → best transfer
verify rules (patches)  → high train gain, weak/negative transfer
```
- **Our mapping:** Our `ScreeningPruner` (pairwise relevance scoring) is structurally similar to the reranking layer. Our `AbsorbCompress` patches are structurally similar to verify rules. The lesson: **prefer screening-style improvements over patch-style improvements**. This validates our existing architecture: `ScreeningPruner::relevance()` is a better generalization target than accumulated `AbsorbCompress` patches.

### Finding 4: Code-as-Model ↔ ML Concept Mapping
| ML Concept | Code Equivalent |
|---|---|
| Model | The codebase |
| Parameters | Thresholds, constants, prototypes, rule conditions |
| Update step | A code patch |
| Optimizer | Coding agent + evaluation feedback |
| Reward | Train accuracy |
| Regularization | Patch acceptance rules, held-out checks |
| Memory | Logs, docs, plots, error audits |

- **Our mapping:** This is already our architecture. `TrialLog` = memory, `AbsorbCompress` = update step, `BanditPruner` = optimizer, `bandit` reward = win/loss. The paper's regularization recommendations are what we should adopt.

### Finding 5: Cascade Dynamics → Symbolic Fragility
- Sequential pipeline: early score change → ranking change → different verify path → unexpected regressions
- Higher train accuracy → less slack → sharper local optimum
- **Our mapping:** Our `BanditPruner<P>` arms can interact — absorbing one arm can shift the reward landscape for others. This is why `AbsorbCompress` has a **compress** phase (remove stale arms). The cascade warning reinforces the need for our existing `HotSwapPruner` isolation.

---

## What We Distill (Design Principles, Not Code)

### D1: Patch Regularization Criteria
From the paper's "Remaining Challenges" section, distilled into our context:

| Criterion | Our Equivalent | Already Enforced? |
|-----------|---------------|-------------------|
| **Support** | How many episodes does this bandit arm apply to? | ⬜ Partial — `AbsorbCompress` tracks win count but not support threshold |
| **Precision** | How often does the arm help when it fires? | ⬜ Partial — tracked via bandit Q-value |
| **Transfer** | Does the arm improve held-out episodes? | ❌ No held-out split in current HL loop |
| **Complexity** | How many thresholds/branches does the arm add? | ❌ Not measured |
| **Locality** | Blast radius in the pipeline? | ⬜ Partial — `HotSwapPruner` provides isolation |
| **Cascade risk** | Does absorbing this arm break other arms? | ⬜ Partial — compress phase handles stale, not cascade |

### D2: Generalization-Aware Acceptance Rule
Paper's recommendation:
```
accept patch if held-out transfer improves
AND support is broad enough
AND complexity is justified
AND cascade risk is bounded
```

Our equivalent (for `AbsorbCompress`):
```
absorb arm if:
  - Q-value > threshold (already done)
  - support_count >= min_episodes (NEW — need support gate)
  - held-out win rate improves (NEW — need train/dev split)
  - arm complexity <= budget (NEW — need complexity budget)
```

### D3: The Generalization Hierarchy (Validated)
Paper proves: **reranking > verify rules** for transfer. Our architecture already follows this:
- `ScreeningPruner::relevance()` = reranking (better generalization target)
- `AbsorbCompress` patches = verify rules (train-only optimization)
- **Keep the current ratio:** ScreeningPruner does the heavy lifting; AbsorbCompress is for polishing

---

## Model-Based vs Modelless Implications

| Aspect | Modelless (Bandit-only) | Model-Based (LoRA) |
|--------|------------------------|-------------------|
| **Overfitting risk** | Same as paper — code patches memorize | Higher — weight memorization + code memorization |
| **Regularization needed** | Support threshold, held-out split | Same + weight decay (already have RMSNorm) |
| **Reranking equivalent** | `ScreeningPruner::relevance()` | LoRA-adapted relevance scoring |
| **Verify rule equivalent** | `AbsorbCompress` patches | LoRA fine-tuning steps |
| **Lesson applies?** | ✅ Directly — same dynamics | ✅ Indirectly — same principles, different substrate |

---

## What We Do NOT Distill

1. **No new feature gate** — These are design principles for existing infrastructure, not new capabilities
2. **No image classification code** — We don't do vision
3. **No symbolic feature extraction** — Our domain is game state, not pixels
4. **No CNN comparison** — Irrelevant to our stack

---

## Verdict

**TL;DR:** HL-ImageNet is a **controlled experiment that validates our existing HL architecture** and identifies 3 concrete gaps: (1) no held-out split in `AbsorbCompress`, (2) no support threshold for arm acceptance, (3) no complexity budget for patches. These gaps are real but low-priority — our current HL loop produces defensible results (Bomber +177, GOAT proven) because game domains have much cleaner feedback signals than vision.

**Action:** Document the regularization principles in `.docs/09_heuristic-learning.md` as design guidelines. No new code needed. If HL loop quality degrades in future domains, add support threshold as a Plan.

**Pillar check (per Decision Matrix 27):** ❌ Not a pillar — not game-specific, not MMO-product, not defensible IP. This is infrastructure quality assurance, not a moat component.

**Priority:** Low — nice-to-have validation, no action items beyond documentation.

---

## References

- Wang, X. (2026). *Heuristic Learning for Symbolic ImageNet-10*. https://github.com/xisen-w/hl-imagenet
- Weng, J. (2026). *Learning Beyond Gradients*. https://trinkle23897.github.io/learning-beyond-gradients/ (Research 014)
- Plan 032: Heuristic Learning Infrastructure
- Plan 033: Bomberman HL Arena
- `.docs/09_heuristic-learning.md`: Our HL documentation
- `.docs/27_mmo_goat_pillars_decision_matrix.md`: GOAT pillar decision matrix
