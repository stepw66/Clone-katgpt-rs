# Issue 030: Cost-Aware Reward-Proportional Latent Trajectory Scorer (Research 263 Fusion)

**Date:** 2026-06-18
**Source:** [Research 263 — Latent Thought Flow](../.research/263_Latent_Thought_Flow_Reward_Proportional_Latent_Reasoning.md)
**Paper:** [arxiv:2606.16222](https://arxiv.org/abs/2606.16222) — Latent Thought Flow (Zou et al., 2026)
**Type:** Optimization / fusion candidate — **not a plan** (per AGENTS.md, optimization tasks go to `.issues/`)
**Verdict at filing:** GAIN — every component has shipped prior art; the fusion is incremental synthesis, not a new capability class.
**Status:** ❌ **CLOSED 2026-06-18** — GOAT gate G1a/G1b FAIL. See [Benchmark results](#benchmark-results-run-2026-06-18) below. Confirms Research 263 GAIN verdict; do not promote.

---

## Benchmark Results (RUN 2026-06-18)

Bench: `katgpt-rs/benches/latent_thought_flow_scorer_bench.rs`
Run: `cargo bench --bench latent_thought_flow_scorer_bench --features self_advantage_gate,micro_belief`
Config: DIM=32, VOCAB=8, N_TRAJECTORIES=8, N_QUERIES=2000, λ_c=0.03, K_CANDIDATES=[0,1,2,3,5,8]

### Synthetic task signal (sanity check — kernel DOES solve the task)

| K (iters) | Accuracy |
|---|---|
| 0 (no thought) | 12.45% (chance) |
| 1 | 28.55% |
| 2 | 33.95% |
| 3 | 39.00% |
| 5 | 39.00% (plateaus) |
| 8 | 39.00% |

### G1: Composite vs baselines

| Scorer | Accuracy | Discard % |
|---|---|---|
| first-K1 (no score) | 28.55% | 0.00% |
| **majority vote (no score)** | **39.00%** | 0.00% |
| cost-only (argmax) | 12.45% | 0.00% |
| advantage-only (argmax) | 38.40% | 0.00% |
| entropy-only (argmax) | 33.95% | 0.00% |
| **composite (argmax, paper shape)** | **28.55%** | 45.09% |
| weighted-vote (composite) | 33.95% | 0.00% |
| weighted-vote (advantage) | 39.00% | 0.00% |

### GOAT gate verdict

| Gate | Criterion | Result | Verdict |
|---|---|---|---|
| G1a | Composite ≥3pp over best single | -9.85pp (28.55% vs 38.40%) | ❌ FAIL |
| G1b | Best fusion ≥1pp over majority | +0.00pp (39.00% ties 39.00%) | ❌ FAIL |
| G1c | ≥30% dead-thought discard | 45.09% | ✅ PASS |
| G2 | Interior entropy band peak | flat 28.55% across all bands | ⚠️ WEAK |
| G3 | <1µs per-trajectory scoring | 802.7 ns | ✅ PASS |

### Interpretation

The modelless fusion of LTF's inference-time insights **does not beat simple majority vote** in the synthetic-task setting:

1. **Majority vote (39%) dominates.** With a fixed (untrained) sampler, N=8 trajectories produce diverse votes; plurality aggregates them robustly. The paper's "argmax of reward-proportional score" assumes a *trained* sampler that concentrates on good trajectories — that concentration comes from the GFlowNet training (→ riir-train), not from the scoring formula itself.
2. **Composite scorer (28.55%) actively hurts** — the cost penalty `exp(-λ_c·K)` biases it toward K=0/K=1 trajectories which are below the convergence point (K=3). The paper's λ_c=0.03 is tuned for their trained trajectories (mean length 1-4); in our untrained setup, the convergent K is higher and the cost penalty wrongly penalizes it.
3. **Weighted-vote (advantage) only ties majority vote** — self-advantage weighting is a wash when the underlying voted actions are already majority-correct.
4. **Entropy band has no effect** — once the composite score is dominated by the advantage term, the multiplicative entropy gate cannot change the argmax. G2's flat 28.55% confirms this.
5. **G3 latency is fine** — 802ns per trajectory, 6.4µs per query (N=8). Fits plasma tier.

### Conclusion

This is an **honest negative result** that confirms Research 263's GAIN verdict. The fusion as specified in this issue (paper's argmax-of-composite-score) does not produce a measurable gain over the existing majority-vote aggregation that ships with Plan 260 (MaxProof Population TTS). The paper's accuracy gains (9.5pp) come from the GFlowNet *training* (EW-SubTB + reference-prior + LoRA-on-latent-head), which is correctly redirected to riir-train.

**Do not promote.** Close this issue. If a future paper provides a modelless concentration mechanism (a way to make the fixed sampler concentrate on good trajectories without training), revisit.

---

## Original filing (preserved for context)

## Summary

Unify five existing primitives into one inference-time operator: a **cost-aware reward-proportional scorer over N latent thought trajectories**, with **self-advantage as the teacher-free quality signal** and an **entropy-band gate** to suppress both collapsed and over-stochastic trajectories.

This is the modelless distillation of arxiv:2606.16222 (Latent Thought Flow). The paper's GFlowNet training machinery (EW-SubTB, reference-prior, LoRA-on-latent-head) → riir-train and is out of scope here. Only the inference-time scoring pattern is in scope.

---

## Components (all shipped — this is a wiring task, not a from-scratch build)

| Component | Source (shipped) | Role in fusion |
|---|---|---|
| Trajectory generator | `LatentThoughtKernel` (Plan 276, `katgpt-rs/crates/katgpt-core/src/micro_belief/latent_thought.rs`) | Generate N variable-length latent thought trajectories per query |
| Quality signal V(τ) | Self-Advantage log-ratio `A(a) = log π+(a) − log π̂(a)` (Research 250, Plan 283) | **Teacher-free** — replaces LTF's trained accuracy reward. Pre-recursion vs post-recursion logits of the same model. |
| Cost penalty C(τ) | `lambda_flow × (1 - stop_prob[depth])` (Plan 052, `katgpt-rs/src/speculative/dd_tree.rs:3641-3648`) | Trajectory-length regularization — favors shorter trajectories unless quality improves. LTF uses `exp(-λ_c·T)` with λ_c=0.03. |
| Entropy-band reweighting | `EntropyWeightedJudge` (Plan 121) — `score = magnitude × entropy_weight` | Apply paper §C.2's "effective entropy regime": suppress trajectories below collapse threshold (Ξ < Ξ_low) AND above noise threshold (Ξ > Ξ_high). LTF's sweet spot Ξ ≈ 0.024. |
| Aggregation | Majority vote / BT pairwise (Plan 260, Plan 040) | Pick winner from N scored trajectories |

---

## Composite score (the unified operator)

For each candidate latent trajectory τ_i (i ∈ 1..N):

```
score(τ_i) = sigmoid(A(τ_i))                     // self-advantage quality, bounded
           · exp(-λ_c · len(τ_i))                 // cost penalty (Plan 052 shape)
           · entropy_band_gate(Ξ(τ_i))            // 1 if Ξ_low < Ξ < Ξ_high, else decay
```

Pick `argmax_i score(τ_i)` or majority-vote on the decoded answer.

Where:
- `A(τ_i)` = self-advantage from running the same model pre- and post-latent-thought (Research 250)
- `len(τ_i)` = number of latent thoughts T in trajectory i
- `Ξ(τ_i)` = average differential entropy of the latent thought distributions in trajectory i (paper Eq. 28)
- `entropy_band_gate(Ξ)` = `sigmoid((Ξ - Ξ_low)/τ) · sigmoid((Ξ_high - Ξ)/τ)` — smooth bandpass

Default constants (paper §C.2, table 10): Ξ_low ≈ 0.015 (collapse onset), Ξ_high ≈ 0.028 (noise onset), λ_c = 0.03.

---

## Why file as issue, not plan

Per AGENTS.md: *"Create issue at ./issues for optimization task, do not create plan."* This is an optimization/fusion of existing primitives — every component ships, the work is wiring + benchmark, not new mechanism design.

Per Research 263 verdict: GAIN, not GOAT. No new capability class. Promotion to plan/feature-flag only if a benchmark shows ≥30% wasted-thought-cycle reduction at matched quality.

---

## Validation protocol (G1–G3, run before promoting to plan)

- [ ] **G1 — Composite scorer vs single-component baselines.** On bomber arena (Plan 076) or HLA-driven NPC thought cycles (Plan 276), compare: (a) `LatentThoughtKernel` alone (current default), (b) + self-advantage gate only, (c) + cost penalty only, (d) + entropy-band only, (e) full composite. Metric: thought-cycle utilization (% of thoughts that produce non-zero self-advantage) and end-task quality (win-rate or accuracy).
- [ ] **G2 — Effective entropy band empirically validated.** Sweep Ξ_low, Ξ_high on the same benchmark. Confirm a band exists where quality peaks (paper §C.2 phenomenon). If no band found → fusion fails, close issue.
- [ ] **G3 — Latency budget.** Composite scorer must fit plasma tier (sub-µs per trajectory per NPC at d_belief=32). Self-advantage doubles forward passes — quantify the cost. If > 2× single-thought latency with no quality gain → fusion fails.

**Promotion gate:** G1 shows ≥30% wasted-thought reduction at matched quality AND G2 confirms the entropy band AND G3 fits plasma tier → promote to plan + feature flag. Otherwise close.

---

## Cross-pollination (track for future, do not implement now)

- **NPC crowd-scale curiosity** (riir-ai Research 126, Plan 299) — cost-aware scorer prunes dead thoughts at 20Hz × 1000 NPCs. Massive tick-budget savings if G1 holds at crowd scale.
- **Freeze/thaw** — sampler bias per NPC personality snapshotted as versioned latent-direction vector (BLAKE3-committed). Two same-type NPCs diverge over time.
- **CGSP dual-pool memory** (Plan 282/312) — cost-aware scorer decides "worth committing to long-term memory" vs "discard as dead compute". Bridge to Plan 308 Cognitive Integrity Layer's dead-injection detector.
- **riir-ai Plan 317 Latent Functor Game Theory Wiring** — latent trajectories over game-theoretic moves; reward-proportional scoring with self-advantage could replace the bandit arm selection.

---

## Related

- Research: [263 Latent Thought Flow](../.research/263_Latent_Thought_Flow_Reward_Proportional_Latent_Reasoning.md)
- Research: [250 Latent Recursion Self-Advantage](../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md)
- Research: [242 Topological State Tracking / LatentThoughtKernel](../.research/242_Topological_State_Tracking_Recurrent_Belief.md)
- Research: [204 NFCoT (closest cousin)](../.research/204_NFCoT_Normalizing_Flow_Continuous_CoT.md)
- Plan: [052 GFlowNet Modelless Distillation](../.plans/052_gflownet_modelless_distillation.md)
- Plan: [121 RMSD EntropyWeightedJudge](../.plans/125_rmsd_relevance_masked_self_distillation.md)
- Plan: [276 MicroRecurrentBeliefState](../.plans/276_micro_recurrent_belief_state.md)
- Plan: [283 Self-Advantage Recursion Gate](../.plans/283_self_advantage_recursion_gate.md)
- Shipped code: `katgpt-rs/src/speculative/dd_tree.rs` (`build_dd_tree_balanced`)
- Shipped code: `katgpt-rs/crates/katgpt-core/src/micro_belief/latent_thought.rs` (`LatentThoughtKernel`)
