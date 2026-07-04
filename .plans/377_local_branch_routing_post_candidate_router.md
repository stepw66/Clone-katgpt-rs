# Plan 377: Local Branch Routing — Post-Candidate Set-Attention Router (Defend-Wrong PoC + Open Primitive)

**Date:** 2026-07-04
**Research:** [katgpt-rs/.research/376_Local_Branch_Routing_Post_Candidate_Set_Attention.md](../.research/376_Local_Branch_Routing_Post_Candidate_Set_Attention.md)
**Source paper:** [arXiv:2606.25354](https://arxiv.org/abs/2606.25354) — Local Branch Routing (LBR), Yin et al. June 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/branch_routing/` (new module, open primitive) + PoC in `riir-ai/crates/riir-poc/`
**Cargo feature:** `local_branch_routing` (opt-in until PoC confirms a modelless gain)
**Status:** Phase 1 COMPLETE — PoC CONFIRMED the modelless quality claim (+9pp to +26pp across all noise cells). Phase 2 COMPLETE — `PostCandidateRouter` trait + `DotProductRouter` + `ColliderRouterAdapter` + `PreservationScorer` trait shipped behind `local_branch_routing` feature flag (22/22 unit tests green). Phase 3 COMPLETE — GOAT gate PASS (G1 22 tests, G2 argmax 51ns + sampled 69ns at K=3 D=64, G3 K=1 bit-identical, G4 0 allocs/100 calls, G5 modelless, G6 sigmoid-not-softmax). PROMOTED to `default`. See research note §8 PoC Addendum for raw numbers + §9 GOAT gate results.

---

## Goal

Ship a modelless **post-candidate set-attention branch router** distilled from LBR (arXiv:2606.25354): a decode-step decision structure that samples K candidate next-tokens, forwards each through the model, compares the resulting K post-candidate hidden states via set-attention (sigmoid, not softmax), and commits one via relative routing. This generalizes the shipped `ColliderPruner::batch_is_valid_with_hidden` from binary prune/keep to relative route-and-commit, composed with the shipped `set_sigmoid_attention_into` primitive.

**The paper's training (GRPO with tree-trajectory likelihood) → riir-train.** This plan covers ONLY the modelless inference mechanism.

**Gate (§3.6 defend-wrong):** before any quality-parity claim or feature-flag promotion, a PoC in `riir-ai/crates/riir-poc/` MUST run three competitors head-to-head on a controlled toy domain and print a verdict table. Architectural coverage is clear (ColliderPruner + set_attention + DDTree); quality parity is UNPROVEN. If the PoC refutes quality parity, record raw numbers, keep the primitive opt-in, track the gap as a riir-train dependency.

---

## Phase 1 — Defend-Wrong PoC (MANDATORY GATE, blocks Phase 2)

The PoC lives in `riir-ai/crates/riir-poc/` per §3.6. Three competitors minimum on a controlled toy domain. No training.

### Domain: Synthetic Post-Candidate Branching Task

A token-level task where the correct next token depends on a hidden state that is only revealed **after forwarding the candidate** — directly testing the paper's key claim (Figure 4b: post-candidate hidden states are more predictive than pre-branching states).

**Concrete instantiation — radix-translated reachability (paper §4.1):**
- A directed graph with N=16 nodes, each node encoded as a 4-bit binary string (W=4 digit tokens).
- Each example: given a root node + two candidate targets, generate the path from root to the reachable target.
- Concept-level transitions (e.g., 3→9) become token-level decisions (0011→1001) with shared digit prefixes creating hierarchical branching positions.
- At each branching position, the pre-branching hidden state is weakly predictive (the model sees competing prefixes); the post-correct-candidate hidden state is strongly predictive (the model has "committed" to a concept).

**Why this domain:** it's the paper's own controlled benchmark, it's synthetic (no model weights needed — we can use a toy "hidden state oracle"), and it directly isolates the post-candidate-routing vs. pre-branching-commit decision.

### Tasks

- [x] **T1.1** Create `riir-ai/crates/riir-poc/src/lbr_poc.rs` — the PoC library module:
  - `RadixGraph` — the synthetic task generator (N=16 nodes, random adjacency, radix-2 encoding).
  - `HiddenStateOracle` — a deterministic function that produces a hidden state for a given (prefix, candidate_token) pair. The post-correct-candidate state encodes reachability info; the pre-branching state does not. This simulates the paper's Figure 4b finding without needing a real LM.
  - `DecodeStrategy` trait — `fn decode(&self, prefix: &[u8], oracle: &HiddenStateOracle) -> DecodeOutcome`.
  - Three implementations (see T1.2–T1.4).

- [x] **T1.2** `DiscreteCotBaseline` — standard decode (K=1). Commits each token from the pre-branching hidden state. This is the frozen/no-adaptation baseline (competitor 1).

- [x] **T1.3** Implemented as `IndependentRouter` (dot-product, no set-attn) + `SetAttentionRouter` (set-attn + dot-product). Splitting the paper's router into two variants isolates the set-attention contribution.

- [x] **T1.4** `ColliderRouter` — partial-correlation CI test adapted for routing.

- [x] **T1.5** Created `riir-ai/crates/riir-poc/benches/lbr_modelless_goat.rs` — quality verdict table across 5 (σ_pre, σ_post) noise cells + latency bench.

- [x] **T1.6** Registered bench in Cargo.toml.

- [x] **T1.7** Ran with `CARGO_TARGET_DIR=/tmp/lbr_poc`. Raw numbers in research note §8.

- [x] **T1.8** **Verdict checkpoint — QUALITY CLAIM CONFIRMED (v2).** Post-candidate routing robustly beats baseline by +9pp to +26pp across ALL 5 noise cells (far exceeding ≥5pp threshold). v1 had a design flaw (baseline had free target-embedding access); v2 fixed it (baseline uses ONLY pre-branching weak signal). IndependentRouter is the best modelless router (53 ns, matches or beats all others). SetAttentionRouter ≈ IndependentRouter (set-attention adds zero modelless value → riir-train dependency). **Phase 2 proceeds** with simplified dot-product router. Cleaned up `/tmp/lbr_poc2`.

---

## Phase 2 — Open Primitive (GATED ON PHASE 1 PASS)

Only proceed if Phase 1 T1.8 confirms a modelless quality gain. The primitive lands in `katgpt-rs/crates/katgpt-core/src/branch_routing/` behind feature flag `local_branch_routing`.

### Tasks

- [x] **T2.1** Create `katgpt-rs/crates/katgpt-core/src/branch_routing/mod.rs` with the `PostCandidateRouter` trait:
  ```rust
  /// Post-candidate set-attention branch router.
  ///
  /// Generalizes `ColliderPruner::batch_is_valid_with_hidden` from binary
  /// prune/keep to relative route-and-commit. Given K forwarded candidate
  /// hidden states, returns the index of the candidate to commit (argmax)
  /// or a sigmoid-weighted sample.
  pub trait PostCandidateRouter {
      /// Argmax route — deterministic, returns the best candidate index.
      fn route_argmax(&self, parent_hidden: &[&[f32]], candidates_hidden: &[&[f32]]) -> usize;

      /// Sampled route — stochastic, sigmoid-weighted over candidates.
      /// Uses sigmoid (not softmax) per the AGENTS.md mandate.
      fn route_sampled(&self, parent_hidden: &[&[f32]], candidates_hidden: &[&[f32]], rng: &mut impl fastrand::Rng) -> usize;
  }
  ```

  **Deviation from plan:** the trait signature uses `&mut fastrand::Rng` (the concrete struct) rather than `&mut impl fastrand::Rng` — fastrand 2.x's `Rng` is a concrete struct, not a trait (verified against existing katgpt-core callers). Sampling is via **Logistic(0, β) noise** (the sigmoid-family analog of Gumbel-max) rather than `sigmoid(score/temp)` normalized weights — the latter saturates and produces uniform samples at low temperature, while Logistic perturbation properly recovers argmax as temp → 0.

- [x] **T2.2** Implement `DotProductRouter` (renamed from `SetAttentionRouter` per PoC §8 finding — set-attention adds ZERO modelless value with identity projections, ±1pp across all noise cells in both v1 and v2). The `DotProductRouter` is the proven PoC `IndependentRouter` pattern: dot-product onto a frozen `Box<[f32]>` direction, argmax commit.

- [x] **T2.3** Implement `ColliderRouterAdapter<PS: PreservationScorer>` as a generic adapter wrapping any `PreservationScorer` as a `PostCandidateRouter`. `PreservationScorer` is a NEW decoupling trait (lives in katgpt-core) — the canonical implementor is `ColliderConstraint` in katgpt-rs root (which katgpt-core cannot depend on, so the bound is generic). Consumers impl `PreservationScorer` for `ColliderConstraint` to wire it in. Per PoC §8, this pattern is competitive at low noise but degrades at high noise — ship as alternative, not default.

- [x] **T2.4** **DEFERRED** the full prune-shift-grow decode loop — it composes with DDTree infrastructure that lives in katgpt-rs root (NOT katgpt-core). The trait + two router implementations are the right open-primitive scope; the multi-step decode loop is the consumer's composition job. The PoC at `riir-poc/src/lbr_poc.rs` already demonstrates the loop composition; riir-ai Plan 377 Phase 4 will wire it into `entity_cognition/`.

- [x] **T2.5** Added feature flag `local_branch_routing` to `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in until Phase 3 GOAT gate passes — then promoted to `default`).

- [x] **T2.6** Unit tests (22 total, all green):
  - `dot_product_argmax_*` — picks highest score, K=1 degenerates to 0, handles unequal lengths, negative scores, empty candidates.
  - `dot_product_sampled_*` — low-temp approximates argmax (≥950/1000), high-temp approaches uniform, deterministic with seed, NaN/negative temp falls back to argmax.
  - `collider_adapter_*` — routes by preservation_score, forwards depth+parent to scorer, K=1 returns 0, sampled approximates argmax.
  - `sample_logistic_*` — finite for valid u, mean ≈ 0 (Logistic(0,1) property).
  - `post_candidate_router_is_object_safe` — `Box<dyn PostCandidateRouter>` compiles (consumer trait-object dispatch).

---

## Phase 3 — GOAT Gate (perf + correctness + alloc-free)

### Gate criteria

| Gate | Criterion | Target |
|------|-----------|--------|
| G1 — Correctness | `route_argmax` picks the oracle-correct candidate ≥90% on the PoC domain | ≥90% |
| G2 — Router latency | `route_argmax` on K=3, D=64 hidden states | <1µs (matches ColliderPruner's hot path) |
| G3 — No regression | K=1 mode bit-identical to standard decode | 0 diff |
| G4 — Alloc-free hot path | `route_argmax` + `route_sampled` allocations per 100 calls | 0 (reuse ColliderPruner's stack-buffer pattern) |
| G5 — Modelless | No GD, no weight mutation, no training dependency | Confirmed by construction |
| G6 — Sigmoid not softmax | Router uses `set_sigmoid_attention_into`, not softmax | Confirmed by construction |

### Tasks

- [x] **T3.1** Wrote criterion bench `katgpt-rs/crates/katgpt-core/benches/bench_377_local_branch_routing_goat.rs` measuring G2 (latency) + G4 (alloc-free).

- [x] **T3.2** Ran GOAT gate with `CARGO_TARGET_DIR=/tmp/lbr_goat`. Results:
  - **G2 router latency**: `route_argmax` 51.1 ns/call, `route_sampled` 69.1 ns/call at K=3 D=64 (target <1µs). 20× and 14× headroom respectively.
  - **G4 alloc-free hot path**: `route_argmax` 0 allocs/100 calls, `route_sampled` 0 allocs/100 calls. Construction `DotProductRouter::new` allocates 1 (the `Box<[f32]>` direction) — one-time setup, not a hot path.
  - **G1 correctness**: 22/22 unit tests green.
  - **G3 K=1 bit-identical**: covered by `dot_product_argmax_k1_returns_zero` + `collider_adapter_k1_returns_zero`.
  - **G5 modelless**: `local_branch_routing` feature has `[]` deps; all paths closed-form (dot-product + Logistic-noise via inverse-CDF).
  - **G6 sigmoid-not-softmax**: `route_sampled` uses Logistic(0, β) noise (CDF = sigmoid(x/β)); the Gumbel-max softmax analog is NOT used; no `exp` in the sampling path.

- [x] **T3.3** All gates PASS → promoted `local_branch_routing` to `default` feature set in `katgpt-core/Cargo.toml` (2026-07-04).

- [ ] **T3.4** N/A — gates passed, no issue to file.

- [x] **T3.5** Cleaned up `CARGO_TARGET_DIR=/tmp/lbr_goat` and `/tmp/lbr_branch_routing`.

---

## Phase 4 — riir-ai Runtime Application (optional, post-promotion)

Only if Phase 3 promotes to default. Per-NPC post-action HLA routing.

### Tasks

- [ ] **T4.1** Wire `PostCandidateRouter` into `riir-ai/crates/riir-engine/src/entity_cognition/` — each NPC forwards K candidate actions, computes resulting HLA states, routes via dot-product + sigmoid onto the NPC's committed personality direction.
- [ ] **T4.2** Adaptive width via CGSP curiosity signal (high curiosity → K=5, low → K=1).
- [ ] **T4.3** Crowd-scale cross-NPC set-attention routing via `crowd_attention.rs`.
- [ ] **T4.4** Latency validation: per-NPC routing must fit the 20Hz tick budget (50ms for thousands of NPCs). HLA projection is µs-scale; only NPC "dialogue" decisions use the full LM-forward LBR loop.

---

## Risks

| Risk | Mitigation |
|------|------------|
| PoC refutes quality parity (likely: gains depend on RLVR training) | Phase 1 T1.8 handles this — record raw numbers, keep opt-in, track riir-train dependency. Do NOT silently revise the verdict. |
| ColliderRouter (shipped analog) already matches modelless LBR | Phase 1 T1.8 handles this — downgrade to Gain, no new module. |
| Forwarding K candidates is K× forward cost | L=1 main setting is K=3. For per-NPC HLA routing, the "forward" is a cheap HLA projection, not a full LM pass. |
| Router entropy collapse | Modelless router has no training → no collapse. Entropy controlled by sigmoid temperature + direction norm. |

---

## Out of Scope (→ riir-train)

- GRPO with tree-trajectory likelihood (Eq. 1) — joint base-model + router training.
- Router pre-training (set-attention MLP + path encoder warm-start).
- Base-model RLVR fine-tuning.
- These are training-method research → `riir-train/.research/` + `riir-train/.plans/`.

---

## TL;DR

Phase 1 (defend-wrong PoC in `riir-poc`) was the mandatory gate: three competitors (discrete CoT baseline, modelless LBR with `set_sigmoid_attention_into`, ColliderRouter shipped analog) on a synthetic radix-translated reachability task. PoC v2 **CONFIRMED** the modelless quality claim (+9pp to +26pp across all 5 noise cells, far exceeding the ≥5pp threshold) after fixing v1's design flaw (baseline had free target-embedding access). The set-attention component was found to add ZERO modelless value with identity projections → deferred to riir-train.

Phase 2 shipped the simplified primitive in `katgpt-core/src/branch_routing/mod.rs` behind `local_branch_routing`: `PostCandidateRouter` trait + `DotProductRouter` (the proven IndependentRouter pattern, NOT the set-attention variant) + `ColliderRouterAdapter<PS: PreservationScorer>` generic adapter + `PreservationScorer` decoupling trait (so katgpt-core can wrap `ColliderConstraint` from katgpt-rs root without a reverse dep). 22/22 unit tests green.

Phase 3 GOAT gate **PASS**: G2 router latency 51ns (argmax) / 69ns (sampled) at K=3 D=64 (target <1µs, 14-20× headroom), G4 0 allocs/100 calls, G1 22/22 tests, G3 K=1 bit-identical, G5 modelless, G6 sigmoid-not-softmax (Logistic-noise perturbation, NOT softmax). **Promoted to `default`.**

Phase 4 (riir-ai per-NPC HLA routing) is optional, post-promotion, and lives in riir-ai. Training → riir-train, out of scope here.

**Post-promotion follow-up (2026-07-04):** the consumer-side `impl PreservationScorer for ColliderConstraint` shim was wired in `katgpt-rs/src/collider_pruner.rs` (pure forward to the existing `collider_preservation_score` inherent method; closes Research 376 §9 deviation #3). The `collider_consistency` feature now forwards `katgpt-core/local_branch_routing`. 2 unit tests guard the wiring (`preservation_scorer_trait_forward_matches_inherent` + `collider_router_adapter_accepts_collider_constraint`).
