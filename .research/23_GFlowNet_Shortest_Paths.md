# Research: GFlowNet Shortest Paths — Modelless Distillation (23)

> Source: [Learning Shortest Paths with Generative Flow Networks](https://arxiv.org/pdf/2603.01786) by Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov (HSE University, École polytechnique, Université Paris-Saclay)
> Date: 2026-03, distilled 2026-06
> Code: https://github.com/GreatDrake/gfn-pathfinding
> **Verdict: MEDIUM-HIGH VALUE — Modelless distillation to DDTree + ScreeningPruner + BanditPruner, no new training needed**

## TL;DR

The paper proves that minimizing expected trajectory length E[nτ] in a non-acyclic GFlowNet forces the backward policy P_B to assign zero probability to all non-shortest paths. The model outputs logits for ALL neighbors in a single forward pass, then beam search picks the best path. Validated on Swap puzzles (n=15, 20) and Rubik's Cubes (2×2×2, 3×3×3), showing competitive results with 16× smaller beam budgets than CayleyPy.

**Our verdict:** The paper's core theorem is architecture-agnostic. We don't need to train a GFlowNet neural network. We can distill the theoretical insight — flow minimization = shortest paths — into our existing `ScreeningPruner` + `BanditPruner` + `AbsorbCompress` stack using LoRA log-probs we already compute and WASM `relevance()` we already call. Four concrete modelless distillations captured in Plan 052.

---

## Core Theorem (What We Actually Need)

**Theorem 3.4:** Minimizing E[nτ] is equivalent to assigning zero probability to all trajectories that are NOT shortest paths from s₀ to terminal states.

**In English:** If you penalize long trajectories, the policy naturally concentrates on shortest paths.

**Key efficiency result:** The backward policy P_B outputs logits for ALL neighbors in a single forward pass. CayleyPy needs 12 separate forward passes (one per neighbor) for the same beam width. This is why the paper is 3.6× faster at test time (1.74s vs 6.19s on H200 for 3×3×3 Rubik's).

---

## Paper Architecture (What We DON'T Need)

| Component | Paper | Why We Skip |
|-----------|-------|-------------|
| Separate P_F/P_B neural networks | 6-layer residual MLP, 25M params | We use LoRA marginals (P_F) + WASM relevance (P_B) |
| Trajectory Balance loss L_regTB | Equation 8, trained on JAX/GPU | We don't train — we use existing model's logits |
| On-policy trajectory sampling | Batch of partial trajectories N_max | We use DDTree's existing marginal computation |
| Flow parameterization F(s) = R(s)/P_F(s_f\|s) | Learned during training | We compute from LoRA stop-probability |

---

## Mapping to Our Stack

```
Paper (GFlowNet)                     Our Stack (Modelless)
─────────────────────                 ─────────────────────────
P_F (forward policy)          ←→     LoRA model's marginals (already computed)
P_B (backward policy)         ←→     WASM Validator::relevance() (already called)
Reward R(x)                   ←→     Validator::is_valid() = terminal state
Flow F(s) = R(s)/P_F(s_f|s)  ←→     1.0 / model_stop_prob[depth] (new computation)
λ · F(s) regularization       ←→     AbsorbCompress (already promotes high-flow arms)
Trajectory Balance L_TB       ←→     DeltaBanditPruner δ signal (already exists)
Beam search W                 ←→     DDTree tree_budget + draft_lookahead (already exists)
```

### Four Modelless Distillations

**D1: FlowPruner** — Stop-probability regularization as ScreeningPruner wrapper.
Paper adds `λ / P_F(s_f | s, θ)` to the loss. We compute `1.0 - stop_prob[depth]` from LoRA marginals and blend into relevance.

**D2: Balanced DDTree** — Harmonize forward marginals with backward relevance.
Paper proves P_F and P_B should agree on trajectory distribution. Currently DDTree uses only forward marginals, relevance is binary filter. New: `score = ln(P_llm) + λ × ln(R_backward)`.

**D3: Flow-weighted bandit reward** — Trajectory length bonus via `observe_delta_with_flow`.
Paper minimizes trajectory length via flow regularization. We add `λ_length / prefix_len` to δ reward.

**D4: Goal-state replay sampling** — Walk winning replays backward through WASM validator.
Paper constructs backward policy by reversing edges. We walk winning Bomberman/FFT replays backward tick-by-tick, recording which actions pass WASM validation.

### Why This Works for Game MMO Context

- NPC pathfinding at scale: O(1) per query with trained policy vs O(b^d) A* per NPC per tick
- AoE threat avoidance: instant escape routing without BFS per threat
- DDTree solution quality: backward relevance ensures shorter, more direct solutions
- All modelless: uses existing LoRA + WASM + Bandit, no new training infrastructure

### What Won't Transfer

- The paper's actual neural network training (requires JAX/GPU trajectory sampling)
- Trajectory balance loss in WGSL (requires new `riir-gflownet` crate)
- Non-acyclic GFlowNet theory for cyclic environments (our game loops are tick-based, not cyclic graphs)

---

## Experimental Results (Paper)

### Swap Puzzle (n=15, n=20)
- Greedy P_B (W=1) finds exact shortest paths after sufficient training
- Beam search W=4 converges faster
- Generalizes: model sees 10^9 / 2.4×10^18 states

### Rubik's Cube (2×2×2, 3×3×3)
- 2×2×2: Finds optimal solutions (avg 10.64 moves) with W=10 vs CayleyPy W=10
- 3×3×3: Competitive avg length (21.24 vs 21.15) at W=2^18
- 3.6× faster inference (single forward pass vs 12× per neighbor)
- 6× smaller beam needed for equivalent solve rate

### Regularization Coefficient λ
- Larger λ → better solutions but risk of total failure
- Rule of thumb: pick largest λ that still finds valid paths after few iterations

---

## Relationship to Existing Work

| This Paper | Our Existing |
|------------|-------------|
| Flow minimization | AbsorbCompress (promotes good, blocks bad) |
| Backward policy P_B | ScreeningPruner::relevance() |
| Single-pass neighbor evaluation | DDTree marginals (already compute all tokens) |
| Beam search with dedup | DDTree best-first search (W=∞ by default) |
| Trajectory balance | DeltaBanditPruner δ signal |

**See also:**
- Plan 049 (G-Zero self-play) — δ is the reward signal shared between G-Zero and GFlowNet distillation
- Plan 021 (ScreeningPruner) — the P_B slot that GFlowNet would enhance
- Plan 017 (Hierarchical Tactical AI) — the target-sequence architecture where flow-pruned A* applies
- Research 21 (G-Zero) — the Hint-δ foundation reused here