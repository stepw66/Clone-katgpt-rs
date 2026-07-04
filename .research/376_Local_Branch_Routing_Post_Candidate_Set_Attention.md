# Research 376: Local Branch Routing (LBR) — Post-Candidate Set-Attention Token Router

> **Source:** [Efficient and Trainable Language Model Test-Time Scaling via Local Branch Routing](https://arxiv.org/abs/2606.25354) — Yin, Jin, Pan, Yang, Xia, Pai, Hu, Zhang, Zhao, Zhao, Xu, Li, Wang, McAuley, Wang (Northwestern, Rutgers, UW–Madison, CMU, LMSYS, Tilde, UCSB, Toronto, UBC, UCSD), June 2026, arXiv:2606.25354v2
> **Date:** 2026-07-04
> **Status:** Active
> **Related Research:** 158 (MUX — the paper's soft-token baseline), 177 (Domino — decoupled causal spec), 232/265 (ColliderPruner — hidden-state candidate scoring, partial prior art), 367 (QuasiMoTTo — QMC TTS), 175 (ThoughtFold)
> **Related Plans:** 178 (MUX — modelless distillations shipped), 265 (CCCP ColliderPruner — DEFAULT-ON), 329 (BranchRouter — dot-product routing), 271 (Attention Matching)
> **Classification:** Public

---

## TL;DR

LBR introduces a **token-level test-time scaling** decode loop: at each step it grows a small width-K, depth-L lookahead tree, **forwards every sampled branch through the model**, then uses a lightweight **set-attention router** to select which depth-1 subtree to commit. The unselected subtrees are pruned, the survivor is shifted forward, and a new layer is grown — a rolling "prune–shift–grow" loop. The key empirical finding (Figure 4b): **post-candidate hidden states are substantially more predictive of the correct target than pre-branching hidden states**, so routing *after* forwarding beats both discrete CoT (routes before) and Soft Thinking / Multiplex Thinking (merges candidates into one continuous embedding, blurring concept identity).

**Distilled for katgpt-rs (modelless, inference-time):** The training story (GRPO with a tree-trajectory likelihood, jointly optimizing base model + router) goes to riir-train. The **inference mechanism** is modelless: the prune-shift-grow loop + a set-attention router over forwarded candidate hidden states. Two of the three building blocks **already ship**: `ColliderPruner::batch_is_valid_with_hidden` (Plan 265, DEFAULT-ON) scores candidates by their post-token hidden states, and `set_sigmoid_attention_into` (in `set_attention.rs`) provides the cross-candidate comparison primitive the paper's router uses. What does NOT ship: the specific composition — a decode loop that forwards K candidate subtrees, compares their post-candidate hidden states via set-attention, and commits one via **relative routing** (not binary prune/keep).

**Verdict: GOAT (modelless path).** The architectural pattern partially ships (ColliderPruner + set_attention); the novel contribution is the prune-shift-grow decode loop + relative-routing commit. **Quality parity with the paper's RLVR-trained version is UNPROVEN and requires a PoC** per the §3.6 defend-wrong rule — architectural coverage ≠ quality parity. The modelless insight that does transfer regardless of training: *route after forwarding, not before* (Figure 4b's probe result holds for any model).

---

## 1. Paper Core Findings

### 1.1 The prune-shift-grow decode loop

At each decoding step `t`, LBR maintains a depth-L, width-K local lookahead tree `A_t` rooted at the committed prefix `x_<t`. Four stages:

1. **Grow**: sample K children from each active frontier node via the filtered model distribution `π̃_θ`. **Crucially, every sampled node is also forwarded** — `h_t(v) = f_θ(x_<t, path(v))` is stored. The tree is not just candidate token strings; it's a forwarded tree of local hidden states.
2. **Route**: a set-attention router observes the forwarded tree and samples `k*_t ~ ρ_φ(· | x_<t, A_t; θ)`, then commits `x_t = tok(u_{t,k*_t})`.
3. **Prune**: discard all depth-1 subtrees except the selected one.
4. **Shift & regrow**: the selected subtree's root token is committed; its descendants become the partial lookahead for the next position; one new layer is grown to restore depth L.

The main setting is L=1, K=3 (route among 3 forwarded candidate next-tokens). L=2, K=3 looks one step deeper.

### 1.2 The set-attention router (Figure 2)

Permutation-invariant over the K candidate subtrees:

1. **Subtree encoding**: each depth-1 candidate subtree `u_{t,k}` is encoded into a vector `g_{t,k}`. For L=1, this is just the post-token hidden state `h_t(u_{t,k})`. For L>1, a shared path encoder summarizes the candidate root + its local continuations.
2. **Cross-subtree set attention**: `g̃_{t,1..K} = SetAttn_cand(g_{t,1..K})` — candidates attend to their sibling alternatives. **Ablation (Figure 5): removing this cross-subtree attention consistently hurts** — the gain is not just from exposing post-token hidden states, but from *comparing* them.
3. **Score & normalize**: `s_{t,k} = w_φᵀ g̃_{t,k}`, then softmax over K with temperature τ.

### 1.3 Why post-candidate hidden states help (Figure 4b — the key modelless insight)

On the synthetic radix-translated reachability task, a linear probe predicts the target node id from hidden states at diverging positions:

- **Discrete CoT** must choose the next digit from the *pre-branching* hidden state. The probe shows the pre-branching state is weakly predictive.
- **LBR** forwards each candidate digit first, then routes. The *post-correct-candidate* state is much more predictive than the pre-branching state. There is a large separation between correct and wrong post-candidate states.
- **Soft Thinking** merges candidate embeddings into a continuous mixture before future computation — a concept-identity probe (Figure 7) shows the mixture does not fully identify which concept branch the model is following (0.759 at the final digit vs. 1.0 for discrete CoT and LBR). **Concept ambiguity explains why Soft Thinking underperforms.**

This is the modelless kernel: *the candidate token itself reveals downstream information that is not available before branching.* It holds for any model architecture, with or without training.

### 1.4 Math reasoning results (Table 1)

On six benchmarks (Minerva, AIME'25/'24, MATH500, AMC'23, Olympiad) with DeepSeek-R1-Distill-Qwen-1.5B and 7B:

- LBR (L=1) improves both Pass@1 and Pass@32 over discrete CoT, vanilla discrete-token RLVR, and RLVR with soft tokens (Multiplex Thinking).
- L=2 often further improves, especially on the 7B backbone.
- Example (7B, AIME'25 Pass@1): Discrete CoT 16.0 → RLVR-disc 17.1 → RLVR-soft 19.7 → **LBR L=1 23.4 → LBR L=2 28.0**.

### 1.5 Tree-trajectory likelihood (Eq. 1) — the training hook → riir-train

```
log p_{θ,φ}(F | q) = Σ_t [ Σ_{v∈G_t} log π̃_θ(tok(v) | ctx(v))  +  log ρ_φ(k*_t | x_<t, A_t; θ) ]
```

Only two stochastic operations per step: sampling newly grown nodes `G_t`, and sampling the router choice. Prune/shift/reuse are deterministic. This factorization enables end-to-end GRPO with verifier rewards, jointly training base model (lr 1e-6) and router (lr 1e-4). **This training pipeline → riir-train.** The inference mechanism stays here.

---

## 2. Distillation

### 2.1 The modelless kernel (the part that survives without training)

Strip the GRPO training. What remains is a **decode-step decision structure**:

> *At each token position, sample K candidate next-tokens, forward each through the model, compare the resulting K post-candidate hidden states via set-attention, commit the one whose post-candidate state best matches the routing objective, prune the rest, shift forward, regrow.*

The routing objective is the modelless replaceable part. The paper trains it (set-attention MLP + softmax). Modelless replacements:

| Routing objective | Modelless instantiation | Codebase anchor |
|---|---|---|
| "Which candidate preserves task-relevant structure?" | Collider-preservation score (Fisher-z CI test on post-candidate hidden state) | `ColliderPruner::batch_is_valid_with_hidden` (Plan 265, DEFAULT-ON) |
| "Which candidate's post-state best matches a target direction?" | Dot-product + sigmoid projection onto a frozen direction vector | `latent_steering.rs`, `committed_field_blend.rs` |
| "Which candidate is most coherent with the prefix belief?" | Cosine similarity to HLA belief-state centroid | `sense/`, HLA kernels |
| "Which candidate is most novel / curiosity-inducing?" | Dot-product inverse to recent direction vectors (curiosity signal) | `cgsp_runtime/`, `manifold_bandit.rs` |
| "Which candidate do sibling alternatives suggest is best?" | Set-attention max over cross-candidate comparisons | `set_attention.rs::set_sigmoid_attention_into` |

The last row is the paper's exact router, minus the trained MLP. The `set_sigmoid_attention_into` primitive already provides cross-state attention with sigmoid (not softmax — per the AGENTS.md mandate) and optional top-k reduction. **A modelless router = set-attention over post-candidate hidden states + a frozen scoring direction.**

### 2.2 Prior art audit (three-layer check per §1.5)

| Layer | Closest cousin | What it does | Gap to LBR |
|---|---|---|---|
| **Notes** | Research 158 (MUX) | Packs K token hypotheses into one continuous latent via vocabulary superposition; lossless demux. | MUX **merges** candidates; LBR **preserves** them as discrete forwarded branches and routes. The paper explicitly beats Multiplex Thinking (the MUX-derived baseline). |
| **Notes** | Research 177 (Domino) | Parallel draft backbone + cheap sequential causal correction. | Domino is for **acceleration** (accept more draft tokens); LBR is for **decision quality** (route to the best candidate). Different objective, same "forward candidates then decide" skeleton. |
| **Code** | `collider_pruner.rs::ColliderPruner` (Plan 265, DEFAULT-ON) | `batch_is_valid_with_hidden(depth, parent_hidden, candidates_hidden, results)` — scores each candidate by post-token hidden state via Fisher-z collider preservation; sigmoid-bounded; alloc-free stack buffer. | **Binary prune/keep**, not relative route-and-commit. Specialized to collider-consistency, not general routing. **But the architectural pattern (score candidates by forwarded hidden states) is identical.** |
| **Code** | `set_attention.rs::set_sigmoid_attention_into` | Cross-state sigmoid attention with optional top-k; permutation-aware; alloc-free scratch buffers. | Provides the router's comparison primitive. **No decode loop uses it for token-level branch routing today.** |
| **Code** | `branching/router.rs::BranchRouter` (Plan 329) | Dot-product snap + Jaccard fallback routing among cognitive branches; ~301ns hot path on 64-branch bank D=8. | Routes among **memory branches** (cognitive domain), not **token candidates** (decode domain). Same routing arithmetic, different operand type. |
| **Code** | `dd_tree.rs` / speculative decode | Tree-structured draft + verify; SpecInfer-style tree attention. | For **acceleration** (verify draft tokens in parallel). LBR uses the tree as **decision evidence** (route among forwarded futures). |

**Verdict on prior art:** the *components* ship; the *specific composition* (prune-shift-grow decode loop + relative routing over forwarded candidate subtrees via set-attention) does **not** ship. ColliderPruner is the closest architectural cousin — it proves the post-candidate-hidden-state scoring pattern is sound and alloc-free at production latency.

### 2.3 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent |
|---|---|
| "Local Branch Routing" / "LBR" | Post-candidate set-attention branch router |
| "lookahead tree" / "forwarded branches" | DDTree with forwarded hidden states (speculative decode tree) |
| "router" / "set-attention router" | `set_sigmoid_attention_into` + scoring head; or `ColliderPruner` generalization |
| "prune-shift-grow" | Rolling speculative tree (novel loop) |
| "post-candidate hidden states" | Forwarded candidate hidden states (`ColliderPruner::cand_hidden`) |
| "Soft Thinking" / "Multiplex Thinking" | MUX (Plan 178) — the soft-token alternative LBR beats |
| "tree-trajectory likelihood" / "RLVR" | GRPO with tree trace → riir-train |
| "commit / prune / shift / regrow" | Tree speculative decode + branch commit |
| "depth-L lookahead" | Speculation depth / draft length |
| "width-K branches" | Draft fanout / speculation width |

### 2.4 Latent-space reframing (mandatory per workflow §1.3)

How LBR looks when operating on the seven Super-GOAT factory modules:

- **(a) HLA per-NPC latent state**: each NPC at a decision point forwards K candidate next-action HLA vectors, a set-attention router compares them, commits one. This is **per-NPC token-level test-time scaling** — the HLA state is the "post-candidate hidden state" the router reads.
- **(b) `latent_functor/` operations**: a functor application could expand K candidate continuations as parallel functor applications, then route via set-attention on the resulting latent vectors. Maps to `latent_functor/zone_gating.rs` (zone-gated activation) generalized from "gate on/off" to "route among K".
- **(c) `cgsp_runtime/` curiosity**: curiosity signal drives the width K — more curious = wider exploration. The router's scoring direction could be the curiosity direction vector.
- **(d) LatCal fixed-point**: not directly relevant — this is a local decode decision, not a sync-boundary commitment.
- **(e) `NeuronShard` style_weights**: a dendritic branch view could expose K candidate continuations as different branch selections within the shard's `style_weights[64]`.
- **(f) DEC Stokes operators**: the prune-shift-grow loop has a boundary-flavored structure — the depth-L tree's frontier is a boundary; rolling forward shifts the boundary. Not a strong mapping; the deeper fit is HLA/functor.

**Strongest latent reframing: HLA + per-NPC test-time scaling** — each NPC forwards K candidate actions, observes resulting HLA states, routes via set-attention. This is a riir-ai application of the katgpt-rs primitive (see §Fusion).

---

## 3. Fusion Ideas

### 3.1 LBR × ColliderPruner (katgpt-rs, strongest internal fusion)

**Generalize `ColliderPruner` from binary prune to relative route-and-commit.** The hidden-state scoring API (`batch_is_valid_with_hidden`) already exists; extend it with a `route_with_hidden` method that returns the argmax candidate (or a sigmoid-weighted sample) instead of a binary mask. TheCollider-preservation score becomes one possible routing objective; dot-product-to-direction is another. **This is the minimal primitive**: a `PostCandidateRouter` trait that composes `ColliderPruner`'s scoring with `set_attention`'s cross-candidate comparison.

### 3.2 LBR × MUX (the paper's own contrast — make it composable)

MUX (Plan 178) packs K hypotheses into one continuous token; LBR preserves them as discrete branches. **Fusion**: a hybrid decode mode that uses MUX for cheap width (compress K candidates into one mux'd token for the forward pass) but uses LBR-style set-attention routing for the commit decision (demux the mux'd token to recover K discrete candidates, then route). This gets LBR's decision quality at MUX's forward cost. Maps to Issue 041 (MUX × FUNCATTN demux-on-edge, filed today) — the same "pack K, demux on demand" pattern.

### 3.3 LBR × Domino (parallel forward + cheap sequential route)

Domino (Research 177) decouples parallel drafting from sequential correction. **Fusion**: forward the K candidate subtrees in parallel (Domino's parallel backbone), then apply the set-attention router as the "sequential correction" step — but now the correction is a routing decision, not a logit residual. This gives LBR's routing at Domino's parallel-forward cost.

### 3.4 LBR × Curiosity (riir-ai, per-NPC adaptive width)

CGSP curiosity signal drives the width K: high-curiosity positions (high entropy, low confidence) get wider lookahead (K=5); low-curiosity positions get K=1 (standard decode). This is **per-NPC adaptive test-time scaling** — spend compute where the NPC is uncertain. Maps to riir-ai `.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md` and `.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md`.

### 3.5 LBR × HLA (riir-ai, per-NPC post-action belief routing)

Each NPC forwards K candidate next-actions, computes the resulting HLA state for each (valence/arousal/desperation/calm/fear), and routes to the action whose post-action HLA state best matches the NPC's committed personality direction. **This is LBR applied to NPC cognition, not token decoding.** The "post-candidate hidden state" is the post-action HLA vector; the "router" is a dot-product + sigmoid onto the NPC's personality direction.

---

## 4. Verdict

**Tier: GOAT (modelless path).**

| Criterion | Verdict | Reasoning |
|---|---|---|
| Modelless first (§3.5) | ✅ | Inference mechanism is modelless: prune-shift-grow + set-attention router over forwarded candidate hidden states. The router can be a deterministic scoring function (collider-preservation, dot-product-to-direction, cosine-to-belief-centroid). The training (GRPO with tree-trajectory likelihood) → riir-train. |
| Latent-to-latent preferred | ✅ | Routing operates on post-candidate hidden states (latent vectors). Decode to token only at the commit boundary. Uses sigmoid (via `set_sigmoid_attention_into`), not softmax, per mandate. |
| Freeze/thaw over fine-tuning | ✅ | The routing direction vector can be a frozen snapshot (`MerkleFrozenEnvelope`-committed personality direction in riir-neuron-db). No weight mutation at runtime. |
| Novelty (Q1–Q4) | Partial | Q1 (no prior art for the specific composition) ✅; Q2 (new decode primitive, not just better numbers) ✅; Q3 (product selling point: "NPCs forward candidate actions and route by resulting belief state") ✅; Q4 (force multiplier across HLA, set_attention, ColliderPruner, CGSP) ✅. **But quality gains in the paper depend on RLVR training → riir-train. The modelless version's quality is unproven (needs PoC).** This caps the verdict at GOAT, not Super-GOAT. |
| Force multiplier (≥2 pillars) | ✅ | Connects to: reasoning pack (P8), self-learn NPCs, per-NPC test-time scaling, MUX (the soft-token contrast). |
| 5-repo discipline | ✅ | Open primitive (post-candidate set-attention router) → katgpt-rs. Per-NPC runtime application → riir-ai. Training → riir-train. No shard/chain IP leaked. |

**One-line reasoning:** LBR's inference mechanism (prune-shift-grow + set-attention routing over forwarded candidate subtrees) is a novel composition of shipped primitives (`ColliderPruner` scoring + `set_attention` comparison + DDTree structure) that provides a new modelless decode primitive; the RLVR training goes to riir-train, and the per-NPC application is a riir-ai fusion.

### 4.1 MOAT gate (per domain, §1.6)

| Domain | Verdict |
|---|---|
| `katgpt-rs` (public engine) | **In scope — GOAT.** Post-candidate set-attention branch router is a paper-derived fundamental decode primitive. Feature-flagged, GOAT-gated, promote/demote tracked on the speculative-decode stack slot (alongside Domino, MUX, ThoughtFold). |
| `riir-ai` (private runtime) | **In scope — fusion GOAT.** Per-NPC post-action HLA routing connects to per-NPC test-time scaling (Research 136) and committed personality (Research 158/302). Strengthens the self-learn NPCs pillar. |
| `riir-chain` | Out of scope — no chain/sync-boundary angle. |
| `riir-neuron-db` | Out of scope — the frozen routing direction vector could be stored in a shard, but that's a consumer pattern, not new shard IP. |
| `riir-train` | **In scope (training only).** GRPO with tree-trajectory likelihood, joint base-model + router training. Note the redirect; do not implement here. |

### 4.2 §3.6 PoC requirement (defend-wrong)

**Mandatory PoC before any quality-parity claim.** The architectural coverage is clear (ColliderPruner + set_attention + DDTree provide the building blocks). But:

| Claim type | Status | Proof |
|---|---|---|
| **Architectural** ("the decode loop can be assembled from shipped primitives") | ✅ Proven | ColliderPruner's `batch_is_valid_with_hidden` + `set_sigmoid_attention_into` + DDTree cover the loop. |
| **Latency / resource** ("modelless, sub-µs routing, no GD") | ⚠️ Plausible, unmeasured | ColliderPruner's hot path is alloc-free with stack buffers; set_attention is alloc-free with scratch buffers. Expect <1µs router overhead per decode step. **Needs a criterion bench.** |
| **Quality** ("modelless routing beats discrete CoT at parity with the paper's RLVR version") | ❌ **UNPROVEN — needs PoC** | The paper's gains come substantially from RLVR training of the base model. The modelless insight (post-candidate hidden states are more predictive, Figure 4b) holds for any model, but the *magnitude* of the modelless gain is unknown. **A PoC in `riir-ai/crates/riir-poc/` is mandatory before claiming quality parity.** Three competitors: (1) discrete CoT baseline, (2) modelless LBR (deterministic router), (3) the paper's RLVR-trained LBR (if riir-train lands it). Run on a controlled toy domain (radix-translated reachability or Sudoku). |

**If the PoC refutes quality parity** (likely outcome: modelless version beats discrete CoT modestly but does not match the RLVR-trained version): record raw numbers, keep the verdict at GOAT for the architectural/latency axes, track the quality gap as a riir-train dependency. Do NOT silently revise the verdict.

---

## 5. What Goes Where

### 5.1 katgpt-rs (open primitive)

- **`PostCandidateRouter` trait** (new, in `katgpt-core/src/speculative/` or a new `branch_routing/` module): generalizes `ColliderPruner`'s hidden-state scoring to relative routing. Methods: `route_with_hidden(depth, parent_hidden, candidates_hidden) -> usize` (argmax) and `route_sampled(...) -> usize` (sigmoid-weighted sample).
- **Prune-shift-grow decode loop** (new, in `katgpt-rs/src/`): a rolling lookahead tree that forwards K candidates, calls `PostCandidateRouter`, commits one, prunes, shifts, regrows. Composes with existing DDTree + speculative verify infrastructure.
- **Feature flag**: `local_branch_routing` (opt-in). Promote to default only after the PoC confirms a modelless gain.
- **GOAT gate**: G1 (routing correctness on radix-reachability toy), G2 (router latency <1µs), G3 (no regression vs standard decode when K=1), G4 (alloc-free hot path, reusing ColliderPruner's stack-buffer pattern), G5 (modelless — no GD).

### 5.2 riir-ai (private runtime application)

- **Per-NPC post-action HLA routing**: each NPC forwards K candidate actions, computes resulting HLA states, routes via dot-product + sigmoid onto the NPC's committed personality direction. Composes with `committed_blend/`, `hla/`, `entity_cognition/`.
- **Adaptive width via curiosity**: CGSP curiosity signal drives K. Composes with `cgsp_runtime/`.
- **Cross-NPC set-attention routing**: at crowd scale, route among candidates across NPCs (crowd MCGS). Composes with `crowd_attention.rs`.

### 5.3 riir-train (training only — note the redirect)

- **GRPO with tree-trajectory likelihood** (Eq. 1): joint base-model + router training. The tree-trace likelihood factorization (grow-stochastic + route-stochastic, prune/shift/reuse deterministic) is the training contribution.
- **Router pre-training**: the set-attention router (path encoder + 2 set-attention blocks + scoring head) needs initialization before RLVR. Could be warm-started from a modelless router (ColliderPruner scoring) to reduce training cost.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Modelless routing quality gap vs RLVR-trained version | PoC in `riir-poc` before any parity claim. Honest verdict revision if refuted. |
| Forwarding K candidates is K× the forward cost | L=1 main setting is only K=3 forwards per commit (vs 1 for standard decode). Tree-attention shares KV cache across branches (paper's Appendix A.3). For per-NPC HLA routing, the "forward" is a cheap HLA projection, not a full LM pass — 20Hz tick budget is feasible. |
| Router entropy collapse (paper Figure 6 shows entropy decreases during training) | For the modelless router, entropy is controlled by the sigmoid temperature and the scoring-direction norm. No collapse risk without training. |
| Concept ambiguity (the Soft Thinking failure mode) | LBR preserves discrete branches by construction — no mixture embedding. The modelless version inherits this property. |
| Latency budget (20Hz tick = 50ms for thousands of NPCs) | Per-NPC routing uses HLA projection (µs-scale), not LM forward passes. Only NPC "dialogue" decisions (few concurrent) would use the full LM-forward LBR loop. |

---

## 7. Key References

- **Paper**: [arXiv:2606.25354](https://arxiv.org/abs/2606.25354) — Local Branch Routing.
- **Closest baseline**: MUX / Multiplex Thinking — `katgpt-rs/.research/158_MUX_Multiplexed_Latent_Reasoning.md`, Plan 178.
- **Architectural cousin**: ColliderPruner (CCCP) — `katgpt-rs/src/collider_pruner.rs`, Plan 265, Research 232.
- **Router primitive**: `set_sigmoid_attention_into` — `katgpt-rs/crates/katgpt-core/src/set_attention.rs`.
- **Speculative decode substrate**: Domino — `katgpt-rs/.research/177_Domino_Decoupled_Causal_Speculative_Decoding.md`.
- **Per-NPC TTS context**: `riir-ai/.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md`.
- **Open PoC question**: Issue 041 (MUX × FUNCATTN demux-on-edge) — `katgpt-rs/.issues/041_mux_latent_funcattn_demux_poc.md` — related "pack K, demux on demand" pattern.

---

## TL;DR (revised post-PoC)

**PoC REFUTED the modelless quality claim.** The verdict stands on architectural coverage + latency, but the quality axis is honestly downgraded: the modelless post-candidate routing does NOT robustly beat the pre-branching baseline. See §8 PoC Addendum for raw numbers.

LBR's training (GRPO with tree-trajectory likelihood) → riir-train. Its inference mechanism (prune-shift-grow + set-attention router over forwarded candidate hidden states) has architectural coverage in shipped primitives (`ColliderPruner::batch_is_valid_with_hidden` + `set_sigmoid_attention_into` + DDTree), but the PoC showed: (1) post-candidate routing only beats baseline in the clean-signal regime (+11.8pp at σ_pre=σ_post=0.1) and ties or loses under moderate noise; (2) **set-attention adds ZERO modelless value with identity projections** (SetAttentionRouter ≈ IndependentRouter within ±1.0pp across all cells) — the paper's set-attention gains require trained Q/K → riir-train; (3) ColliderRouter (shipped analog) underperforms at high noise. **Phase 2 of Plan 377 does NOT proceed** — the open primitive is not justified by a modelless gain. The quality gap is a riir-train dependency (RLVR training of the router + base model).

---

## 8. PoC Addendum (Plan 377 Phase 1, executed 2026-07-04)

**Location:** `riir-ai/crates/riir-poc/benches/lbr_modelless_goat.rs` + `riir-ai/crates/riir-poc/src/lbr_poc.rs`.
**Domain:** radix-translated graph reachability (N=16 nodes, D=16 hidden dim, K=3 candidates, 500 tasks per noise cell).

### Raw results

| Strategy | σ_pre=0.1 σ_post=0.1 | σ_pre=0.5 σ_post=0.1 | σ_pre=0.5 σ_post=0.5 | σ_pre=1.0 σ_post=0.5 | σ_pre=1.0 σ_post=1.0 | Latency |
|---|---|---|---|---|---|---|
| **DiscreteCoT (baseline)** | 72.8% | 83.0% | 83.0% | 86.4% | 86.4% | **23.7 ns** |
| **IndependentRouter** | 84.6% (+11.8pp) | 84.6% (+1.6pp) | 83.4% (+0.4pp) | 83.4% (-3.0pp) | 79.6% (-6.8pp) | **53.0 ns** |
| **SetAttentionRouter** | 84.6% (+11.8pp) | 84.6% (+1.6pp) | 83.2% (-0.2pp) | 83.2% (-3.2pp) | 78.6% (-7.8pp) | **367.8 ns** |
| **ColliderRouter** | 82.8% (+10.0pp) | 84.0% (+1.0pp) | 79.0% (-4.0pp) | 76.4% (-10.0pp) | 75.6% (-10.8pp) | **63.4 ns** |

### Findings

1. **Post-candidate routing wins ONLY in the clean-signal regime** (+11.8pp at σ_pre=σ_post=0.1). Under moderate-to-high noise, it ties or loses to the baseline. The modelless quality gain is not robust.

2. **The baseline paradoxically improves with noise** (72.8% → 86.4% as σ_pre goes 0.1 → 1.0). Root cause: the baseline uses a 0.3/0.7 weighted blend of (pre-branching state alignment) + (target embedding alignment). Higher σ_pre noise on the pre-branching component is dominated by the target-alignment component, which is noise-free. The post-candidate strategies rely purely on post-candidate signal, which degrades with σ_post.

3. **SetAttentionRouter ≈ IndependentRouter across ALL cells** (within ±1.0pp). **Set-attention adds ZERO modelless value with identity projections.** The paper's cross-subtree set-attention gain (Figure 5) requires trained Q/K/V projections. This is a genuine riir-train dependency — the set-attention mechanism is architecturally available (`set_sigmoid_attention_into`) but needs learned projections to extract routing signal.

4. **ColliderRouter underperforms at high noise** (-10.8pp at σ_pre=σ_post=1.0). The partial-correlation CI test is noise-sensitive — denominator instability when candidate-parent correlation is high.

5. **Latency**: SetAttentionRouter is 7× slower than IndependentRouter (368 ns vs 53 ns) for zero quality gain. If a modelless router were justified, IndependentRouter (53 ns) would be the right choice, not SetAttentionRouter.

### Verdict revision (honest)

| Claim type | Pre-PoC status | Post-PoC status |
|---|---|---|
| **Architectural** ("decode loop assemblable from shipped primitives") | ✅ Proven | ✅ Confirmed — the loop runs, all primitives compose. |
| **Latency** ("modelless, sub-µs") | ⚠️ Plausible | ✅ Confirmed — IndependentRouter 53 ns, well under 1µs. SetAttentionRouter 368 ns also under 1µs but unjustified. |
| **Quality** ("modelless routing beats discrete CoT") | ❌ Unproven | **❌ REFUTED for the general case.** Only wins in the clean-signal regime. Under moderate noise, the pre-branching baseline's target-alignment boost dominates. The modelless post-candidate routing does NOT robustly beat baseline by ≥5pp. |

### What this means

- **Plan 377 Phase 2 does NOT proceed.** No `PostCandidateRouter` trait, no `local_branch_routing` feature flag, no new module in katgpt-core. The architectural coverage via ColliderPruner + set_attention is sufficient; a dedicated primitive is not justified by a modelless gain.
- **The quality gain is a riir-train dependency.** The paper's gains come from RLVR training of both the base model and the router (including the set-attention Q/K/V projections). The modelless analog does not reproduce these gains.
- **The PoC stays as a permanent regression check** in `riir-poc` per §3.6 — if a future modelless router variant (e.g., with frozen pre-trained projections from riir-train) is proposed, this bench re-runs to test it.
- **Research 376 verdict revised**: GOAT (architectural + latency) → **Gain (architectural coverage only, quality unproven, needs riir-train for the gain)**. The verdict was honestly flagged as needing a PoC; the PoC refuted the quality claim. This is the §3.6 defend-wrong protocol working as designed.

### Surprising finding (baseline target-alignment boost)

The baseline's 0.3/0.7 blend of pre-branching + target-alignment is a strong heuristic that's hard to beat with post-candidate routing alone. This is an artifact of the synthetic task design (embeddings structured by BFS distance from a reference). A real LM's pre-branching distribution would not have this target-alignment boost — the paper's Figure 4b shows the pre-branching state is weakly predictive. The PoC's baseline is stronger than the paper's because the synthetic embeddings leak target information into the pre-branching state. **A follow-up PoC with unstructured embeddings (or a real micro-GPT) would give a fairer test of the post-candidate advantage.** This is tracked as a non-blocking follow-up — the set-attention finding (zero modelless value) holds regardless of the embedding structure.
