# Research 386: UnMaskFork — Deterministic Action Branching MCTS for Test-Time Scaling

> **Source:** [UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching](https://arxiv.org/abs/2602.04344) — Kou Misaki, Takuya Akiba (Sakana AI), arXiv:2602.04344v1, 5 Feb 2026
> **Date:** 2026-07-07
> **Status:** Done — verdict locked
> **Classification:** Public
> **Related Research:** 034 (D2F — our dLLM substrate), 072 (DMax — closest dLLM parallel-decoding cousin), 161 (dMoE — block-level expert routing), 225 (MSA sparse attention), 260 (MaxProof population TTS), 263 (Latent Thought Flow — different TTS axis), 280 (RTDC — deterministic commitment angle), 367 (QuasiMoTTo — different sampler for TTS), 373 (ReMax — different exploration operator), 350 (density-aware compute scheduling)
> **Related Plans:** 066 (D2F inference pipeline — host for UMF's MCTS layer), 061 (Fourier MCTS transposition table — state-only cousin), 388 (ProofGoalCache — state-only cousin), **390 (this research's implementation plan — katgpt-rs/.plans/390_*.md)**

---

## TL;DR

UnMaskFork (UMF) is a **test-time scaling** (TTS) framework for Masked Diffusion Language Models (MDLMs) that replaces **stochastic sampling** (temperature, random remasking) with **deterministic action branching** inside MCTS. Each MCTS *action* is an *inference configuration* `a = (θ_a, T_a, g_a)` — a tuple of (model parameters, temperature, remasking strategy). Because the transitions are deterministic at low temperature, the same `(state, action)` pair always produces the same `(next_state, reward)`, enabling **aggressive state-action pair caching** with zero recompute on cache hits (~55% hit rate at NFE=12288 on LiveCodeBench). UMF beats Best-of-N, DTS*, and AB-MCTS on LiveCodeBench / HumanEval+ / MBPP+ (28.0 / 88.0 / 72.0 pass@1 at NFE=12288 vs next-best 21.0 / 81.0 / 68.0).

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **state-action pair caching for MCTS over a discrete action space of inference configurations**, plus the theoretical framing that makes it provably better than static config selection (Eq. 1: `Σ_t E_zt[min_a ε_a(zt)] ≤ min_a Σ_t E_zt[ε_a(zt)]` — a state-dependent switching policy strictly dominates any single static kernel). Three concrete pieces:

1. **`Action` = `(config_id, temperature, strategy)` tuple** — a finite, discrete action space distinct from "which game move" or "which token". In our codebase, this generalizes to (which frozen shard / archetype blend / solver kind / remasking strategy to apply at this MCTS node).
2. **`StateActionCache`** — `papaya::HashMap<(StateHash, ActionId), (NextState, Reward)>`. Distinct from our shipped `TranspositionTable` (Plan 061) and `ProofGoalCache` (Plan 388), which key on **state only**. UMF's key insight is that the same state under *different* actions yields *different* deterministic transitions; caching the pair captures both axes.
3. **Determinism enables Var[R] ≈ 0** — single rollout per (state, action) is exact. No averaging, no variance penalty. Cache hit probability → 1 on revisits. This is the budget-efficiency argument that distinguishes UMF from stochastic TTS baselines (DTS*, BoN at high temperature).

**Verdict: GOAT (pending benchmark).** Novel state-action caching primitive with **no prior art** in the codebase (3-layer check: notes + code + vocabulary translation; closest cousins `TranspositionTable`/`ProofGoalCache` are state-only; `dllm_solver.rs` is single-axis decode-step switching, not tree-level; `BranchRouter` routes at token/sample level, not MCTS-node level). Not Super-GOAT because Q3 (product selling point) is a perf claim ("cache reuse under fixed NFE"), not a new capability class — you cannot finish "our NPCs do X that no competitor can" with a *caching* primitive. Touches ≥2 pillars (foundation MCTS + dLLM inference substrate + neuron-db shard-as-model potential), so Q4 (force multiplier) holds. Theoretical guarantee (Eq. 1) is the genuine novel insight — provably better than static config selection under any diverse kernel set.

---

## 1. Paper Core Findings

### 1.1 The problem — stochastic TTS hurts MDLMs

Standard AR-LLM TTS (Best-of-N, tree search) increases temperature to encourage diversity. UMF §1 + §6.2.2 show this **degrades** MDLM generation: high-temperature sampling disrupts the iterative global refinement that masked diffusion relies on. Early stochastic errors propagate through subsequent denoising steps, breaking global consistency. Empirically (Table 4): temperature-TTS at T=(0.1, 1.0) gets 27.0 pass@1 on LiveCodeBench vs UMF's 28.0; the gain from diversity is outweighed by the quality loss from stochasticity.

### 1.2 UMF — deterministic action branching + MCTS

UMF reformulates the unmasking trajectory as a **search tree**:

- **Nodes** = partially-masked states at fixed mask-ratio schedule `ρ ∈ [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2]` (denser early for trajectory diversity)
- **Branches** = selected **inference actions** `a = (θ_a, T_a, g_a)`:
  - `θ_a` = model parameters (one of multiple pre-trained MDLMs: Dream-Coder, LLaDA, etc.)
  - `T_a` = temperature (~0 for greedy determinism)
  - `g_a` = remasking strategy (entropy-based for Dream, low-confidence for LLaDA)
- **Transition** `F_a : z_t ↦ z_{t-1}` advances state to next mask-ratio while holding `a` fixed
- **MCTS** (UCT selection, expansion, backup) optimizes the path through the tree

The action space is **discrete and finite** (the available MDLMs × strategies × temperature settings). Crucially, transitions become **fully deterministic** when `T_a ≈ 0` + `g_a` deterministic + fixed tie-breaking.

### 1.3 State-action pair caching (the budget win)

Algorithm 1 line 11-12: when expanding node `N` with action `a`, if `(N, a)` is already cached, return the cached `(next_state, reward)` with **zero NFE cost** (no forward pass). Determinism guarantees the cached value is exact.

Empirical cache hit rates (Table 3, LiveCodeBench):
- NFE=3072: 47.8% hit rate, +4.76% pass@1 over no-cache
- NFE=6144: 54.5% hit rate, +4.35% pass@1
- NFE=12288: 55.8% hit rate, +7.69% pass@1

The cache converts "wasted NFE on redundant rollouts" into "deeper tree exploration at the same NFE budget."

### 1.4 Theoretical motivation (§5.1, Eq. 1) — the genuinely transferable insight

Inference = sequence of reverse-kernel transitions `p_θ(z_s | z_t)`. Each action `a` selects a kernel `K_a`. Negative ELBO decomposes as sum of per-step KL divergences `ε_a(z_t)` between true posterior and model kernel.

**The switching-policy dominance inequality:**

```
Σ_t E_zt[ min_{a ∈ A} ε_a(z_t) ]  ≤  min_{a ∈ A} Σ_t E_zt[ ε_a(z_t) ]      (Eq. 1)
```

A state-dependent switching policy `π_t(a | z_t)` that picks the best kernel *per step* strictly outperforms any single static model. **This holds for ANY diverse kernel set** — stochastic perturbations OR structural differences (multi-model). UMF's empirical contribution is showing the structural (multi-model) version dominates the stochastic version in practice because it gets the diversity without the variance penalty.

### 1.5 Budget efficiency: deterministic vs stochastic diversity (§5.2)

MCTS value estimation: `Q(z, a) = E[R(τ) | z, a]`. To estimate within error `ε` with high confidence, required rollouts `m ∝ Var[R] / ε²`.

- **Stochastic actions** (high T): `Var[R] > 0` → need `m` large → NFE spent on averaging, not exploration
- **Deterministic multi-model actions**: `Var[R] ≈ 0` → `m = 1` is exact → all NFE goes to tree expansion + cache reuse

### 1.6 Multi-model collaboration within one trajectory (§6.3.3, §6.4)

"Pair" baseline (independent Dream-Coder + LLaDA, take better): 24.0/78.0/69.0. Multi-model UMF (interleaved within one trajectory): **28.0/88.0/72.0**. The structured interleaving — not just access to multiple models — is what unlocks the gain. Case study (Figure 3-4): Dream-Coder outlines, LLaDA fills requirements, LLaDA begins core impl, Dream-Coder refines. Heterogeneous tokenizers handled via special-token direct map + text re-encode (only ~6 tokenizer swaps per 768-step generation).

### 1.7 Results summary

| Benchmark | BoN best | DTS* best | AB-MCTS best | **UMF** |
|---|---|---|---|---|
| LiveCodeBench (NFE=12288) | 19.0 | 18.0 | 21.0 | **28.0** |
| HumanEval+ | 75.0 | 75.0 | 81.0 | **88.0** |
| MBPP+ | 66.0 | 68.0 | 68.0 | **72.0** |
| MATH (NFE=12288) | — | — | — | **60.95** (vs 49.52 at NFE=768) |

Scaling continues to NFE=24576 (LiveCodeBench 30.0) — no saturation.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalents |
|---|---|
| test-time scaling (TTS) | test-time scaling, `mcts_search`, `cognitive_branches_runtime`, CLR voting |
| MCTS over unmasking trajectory | `mcts_search` (generic MCTS), `mcts_collapse_bridge` |
| **action = inference configuration (θ_a, T_a, g_a)** | `SolverKind` (DPM/QSample/DDPM in `dllm_solver.rs`), `BranchRouter` (token/sample level), `polytope_router`, `Dynamic Pair` (Plan 260) — but NONE at MCTS-node granularity |
| **state-action pair caching** | **NEW** — closest cousins `TranspositionTable` (Plan 061, state-only) and `ProofGoalCache` (Plan 388, state-only) |
| deterministic transition / Var[R]≈0 | `SolverKind` deterministic switching, `LatCalFixed::det()` (i128 bit-identical) |
| node cache / rollout cache | `proof_cache.rs`, `policy_cache.rs` (both state-only) |
| multi-model / heterogeneous MDLMs | `Dynamic Pair`, `dMoE`, `ArchetypeBlendShard`, `KarcShard`, `BranchBank` (multiple frozen shards as the "model" axis) |
| adaptive kernel selection (§5.1) | "switching policy", "router", `BranchRouter::route` |
| unmask trajectory / partial state | d2f pipeline, `D2fBlockState`, `set_diffusion_schedule` |
| reward / NFE budget | `StateHeuristic`, `gain_cost_halt`, `BreakevenComplexityRouter` |

### 2.2 The transferable primitive (modelless)

```rust
/// A single discrete action in the MCTS-over-configurations search.
/// Generalizes UMF's `(θ_a, T_a, g_a)` to any finite set of inference
/// configurations (solver kind + strategy + temperature + adapter id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InferenceAction {
    pub config_id: u16,      // index into a caller-supplied config table
    pub strategy_id: u8,     // remasking / sampling strategy enum
    // (temperature is part of config_id; adapter/shard id is part of config_id)
}

/// Cache key: (state hash, action). Distinct from state-only transposition.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateActionKey {
    pub state: blake3::Hash,
    pub action: InferenceAction,
}

/// Cache entry: the deterministic (next_state_hash, reward) for this pair.
/// Populated on first rollout; zero-cost on subsequent visits.
pub struct StateActionCache<R> {
    inner: papaya::HashMap<StateActionKey, (blake3::Hash, R)>,
}

impl<R: Copy> StateActionCache<R> {
    /// O(1) lock-free lookup. Returns None on miss.
    pub fn get(&self, state: blake3::Hash, action: InferenceAction)
        -> Option<(blake3::Hash, R)> { ... }
    /// Insert after a deterministic rollout. Caller MUST guarantee the
    /// transition is deterministic given (state, action) — i.e. low-T greedy
    /// decoding with a deterministic strategy and fixed tie-breaking.
    pub fn insert(&self, state: blake3::Hash, action: InferenceAction,
                  next: blake3::Hash, reward: R) { ... }
}
```

The MCTS loop change vs shipped `mcts_search`:

```
Expand(N, T, C):
    a ← select_unexplored_action(N)
    if C.has(state_hash(N), a):              // ← NEW: state-action cache hit
        return C.get(state_hash(N), a)       // zero NFE cost
    N_new ← apply_action(N, a, T)            // deterministic rollout to next ρ
    C.set(state_hash(N), a, hash(N_new))     // ← NEW: cache the pair
    r ← rollout_to_terminal(N_new, a, T, C)  // also caches intermediate pairs
    C.set_score(hash(N_new), a, r)
    return (N_new, r)
```

### 2.3 Latent-space reframing (mandatory before verdict)

Re-cast UMF's "action = inference configuration" as a discrete selection over each of the seven Super-GOAT factory substrates:

| Substrate | "Action = ?" reframing | Strength |
|---|---|---|
| **HLA per-NPC state** (riir-engine/hla) | action = which direction-vector projection to apply at this NPC-decision MCTS node; state-action cache = per-(NPC-state, direction-set) cache, reused across thousands of NPCs with similar belief states | **Strong** — crowd-scale cache reuse, the per-NPC test-time-scaling substrate (riir-ai R136) |
| **latent_functor** | action = which functor composition (`zone_gating` × `reestimation` × `arithmetic`) to apply at this tree node; Eq. 1 proves interleaving beats any single static composition | **Strong** — direct application of the switching-policy theorem |
| **cgsp_runtime** | action = which curiosity-class router / conjecturer arm to expand with; state-action cache reuses explored conjectures | Medium — CGSP already has bandit arms; the MCTS framing adds tree structure |
| **LatCal commitment** (riir-chain/encoding) | deterministic transitions = committable per-branch; reward-backed trajectory can be LatCal-committed at each (state, action) boundary | Medium — deterministic → raw-fixed-point bookkeeping per branch; riir-chain/.research/009 (DP) is the closest cousin |
| **NeuronShard** (riir-neuron-db) | action = which frozen shard (`KarcShard` / `ArchetypeBlendShard` / `BranchBank` snapshot) to apply at this node; **multi-shard-as-action** is the direct analog of UMF's multi-MDLM | **Strong** — `KarcShard` is the freeze/thaw Pod substrate; multi-shard within one trajectory is UMF's "collaborative generation" mapped to crowd NPCs |
| **DEC Stokes operators** | (N/A — DEC operators are continuous cochain algebra; UMF is discrete action selection. No direct mapping.) | Weak |
| **MCTS substrate** (`katgpt-core/src/mcts.rs`) | action already abstract; the new piece is the *cache axis* — state-action pair vs state-only | **Direct** — this is where the primitive lands |

**Primary reframing**: UMF's "multi-model collaboration within one trajectory" generalizes to **multi-shard collaboration within one NPC's decision trajectory** — at each MCTS node, pick which frozen shard (personality snapshot) advances the belief state. Crowd scale: thousands of NPCs reuse the same `(belief-state-class, shard-id)` cache entries. The theoretical guarantee (Eq. 1) says interleaving shards per-step beats committing to one shard for the whole trajectory.

### 2.4 Fusion (the closest 2-3 cousins)

| Cousin | Repo | What it ships | Fusion with UMF produces |
|---|---|---|---|
| **D2F** (R34, Plan 066) | katgpt-rs | dLLM block-causal inference pipeline + `SolverKind` decode-step switching | **UMF-over-D2F** — MCTS over the dLLM's unmasking trajectory, with `SolverKind` × remasking strategy as the action axis. The dLLM is the host; UMF is the search layer on top. |
| **TranspositionTable** (Plan 061) + **ProofGoalCache** (Plan 388) | katgpt-rs | State-only MCTS caches (BLAKE3-keyed) | **State-action pair cache** — generalizes both to the (state, action) key, capturing the "same state, different action, different transition" axis neither covers |
| **`dllm_solver.rs`** (Critical Interval Solver Switching, Plan 222) | katgpt-core | Entropy-triggered single-axis switching (DPM↔QSample↔DDPM) at decode-step level | **Tree-level switching** — UMF promotes decode-step switching to MCTS-node switching, with Eq. 1 as the theoretical justification |
| **dMoE block expert routing** (R161) | katgpt-rs | Per-block expert coreset (token-level) | **Per-MCTS-node config coreset** — analogous aggregation pattern at a different granularity |
| **Per-NPC CLR runtime** (riir-ai R136) | riir-ai | K-candidate voting with claim-level reliability, no MCTS | **CLR-vote-as-reward** for UMF over NPC decision trees — CLR's `(mean(v))^M` becomes the MCTS leaf reward, the multi-shard action space becomes UMF's multi-MDLM analog |
| **Dynamic Pair LoRA routing** (Plan 260) | riir-ai | 2-adapter routing at token/sample level | **N-adapter routing at MCTS-node level** — UMF generalizes Dynamic Pair from binary to N-way with search |
| **RTDC** (R280) | katgpt-rs + riir-chain | Deterministic multi-resolution Merkle commitment | **Per-branch LatCal commitment** — deterministic UMF transitions are committable; the reward-backed trajectory becomes a deterministic ledger entry |

---

## 3. Verdict

**Tier: Gain (revised from GOAT-pending-benchmark, 2026-07-07).**

**Revision rationale (Plan 390 Phase 3 GOAT gate, 2026-07-07):** The G2
budget-expansion gate FAILED on the synthetic domain (1.00× vs 1.4×
target) because the domain was too small for the cache's cumulative
savings to manifest as reward-convergence speedup. The cache works
(G1 42% hit rate, G3 no-regression, G5 bounded), but the headline GOAT
metric could not be validated. See `.issues/044_*` for the full gap
analysis and re-gate conditions. Verdict revised GOAT → Gain: the
primitive is a correct modelless caching improvement, but does not
meet the GOAT threshold on the available benchmark. Re-gate contingent
on a larger domain or real dLLM PoC (Plan 5).

| Question | Answer |
|---|---|
| Q1 — No prior art? | **YES.** 3-layer check: no `.research/` note, no shipped code, vocabulary translation found zero hits for state-action pair caching in MCTS. Closest shipped cousins (`TranspositionTable`, `ProofGoalCache`) are state-only. `dllm_solver.rs` is single-axis decode-step switching, not tree-level. `BranchRouter` is token/sample level, not MCTS-node level. |
| Q2 — New class of behavior? | **PARTIAL.** State-action pair caching for deterministic transitions IS a new caching primitive. But MCTS-over-configurations is structurally close to existing MCTS + branch routing — it's "better numbers under fixed budget", not a capability no incumbent can match. |
| Q3 — Product selling point? | **NO.** "Cache reuse under fixed NFE" is a perf claim, not a capability claim. Cannot finish "our NPCs do X that no competitor can" with a *caching* primitive. The dLLM substrate (D2F/DMax) carries the capability story; UMF is a search efficiency layer on top. |
| Q4 — Force multiplier? | **YES.** Connects ≥2 pillars: foundation MCTS (`katgpt-core/src/mcts.rs`) + dLLM inference substrate (D2F Plan 066) + neuron-db shard-as-model potential (`KarcShard`/`ArchetypeBlendShard` as UMF's "multi-MDLM" axis). |

3/4 → **GOAT, not Super-GOAT.** The modelless constraint is decisively satisfied (MCTS + caching + multi-config routing, no training). Theoretical guarantee (Eq. 1) is the genuine novel insight.

**MOAT gate per domain (§1.6):** The open primitive (state-action cache + MCTS-over-configurations) is generic search/caching math, no game/chain/shard semantics → **`katgpt-rs` (public)**. A potential riir-ai follow-up (multi-shard-as-action for crowd NPCs) is mentioned in §2.3 but is contingent on the GOAT gate passing first; not opened in this session.

**UQ-bearing primitive floor check (the "Report the Floor" rule, §2 of AGENTS.md):** **N/A.** UMF is NOT UQ-bearing — it produces a single best trajectory by reward (terminal reward = proportion of passed tests), no probability distribution / predictive interval / quantile / coverage guarantee / confidence score / calibrated uncertainty. The conformal-naive floor does not apply.

**§3.5 modelless-unblock check:** Not invoked (no riir-train deferral was contemplated). UMF is modelless by construction.

**§3.6 defend-wrong PoC:** NOT required for the verdict itself (verdict is GOAT with explicit "pending benchmark", not a PASS claiming parity, and not a Super-GOAT claiming quality). The benchmark IS the GOAT gate (Plan 390) — a controlled toy benchmark comparing state-action cache vs state-only transposition vs no-cache under fixed NFE, on a deterministic-decoding domain. PoC is the gate, not a pre-gate requirement.

### One-line reasoning

State-action pair caching for MCTS over a discrete inference-configuration action space is a novel modelless primitive with a provable dominance guarantee (Eq. 1); it promotes to default only if the GOAT gate shows ≥1.4× effective-budget expansion at matched reward on our dLLM substrate, otherwise stays opt-in behind a feature flag.

---

## 4. What ships (open primitive, katgpt-rs)

**Target:** `katgpt-rs/crates/katgpt-core/src/mcts.rs` (extend existing generic MCTS) + new `katgpt-rs/crates/katgpt-core/src/mcts_state_action_cache.rs` (the cache primitive) + feature flag `mcts_state_action_cache`.

**The minimal open primitive** (no dLLM dependency, no game IP, no chain IP):

1. `InferenceAction { config_id, strategy_id }` — opaque action handle (caller-defined semantics)
2. `StateActionCache<R>` — `papaya::HashMap<(blake3::Hash, InferenceAction), (blake3::Hash, R)>`, lock-free, BLAKE3-keyed (per AGENTS.md: blake3 over SHA256)
3. `mcts_search_with_state_action_cache<S, P, H, C>` — extension of `mcts_search` that consults/inserts the cache at Expand time; falls back to standard rollout on miss
4. `DeterministicTransition` marker trait — caller asserts the (state, action) → next_state map is deterministic; violated determinism invalidates the cache entry (debug-mode check via BLAKE3 of recomputed transition)
5. Theoretical-test: a property test proving Eq. 1 holds empirically on a controlled toy domain (sum-of-mins ≤ min-of-sums over a synthetic kernel-error landscape)

**Composition with shipped substrate** (not in scope for the open primitive, noted as fusion):

- D2F host: `mcts_search_with_state_action_cache` driven by `D2fPipeline` with `SolverKind` × remasking strategy as the action axis
- riir-ai follow-up (contingent on GOAT gate): `ArchetypeBlendShard` / `KarcShard` as the "multi-model" axis for per-NPC runtime test-time scaling

---

## 5. What does NOT change

- Existing `mcts_search` API is unchanged — the cache is an opt-in extension via a new function, not a modification.
- Existing `TranspositionTable` (Plan 061) and `ProofGoalCache` (Plan 388) are NOT deprecated — they remain the right tool for state-only caching. UMF's state-action cache is for the case where the action axis matters (deterministic multi-config search).
- No riir-train dependency. No training. No gradient updates.
- No sync-boundary crossing — the cache is local to the search instance, latent, never synced.

---

## 6. Risks and unknowns

- **Cache invalidation under non-determinism**: if the caller violates the `DeterministicTransition` contract (e.g., temperature > 0, RNG in the strategy), the cache returns stale results. Debug-mode BLAKE3 re-check catches this; release-mode assumes the contract holds. Mitigation: document the contract loudly; the feature is opt-in.
- **Memory growth**: the cache is unbounded by default. Mitigation: optional LRU bound (papaya supports size-bounded eviction); the existing `proof_cache.rs` per-decode-step scope pattern (created fresh per search, dropped at end) is the default.
- **Hashing cost**: BLAKE3 of the full state per (state, action) lookup is the hot-path cost. Mitigation: caller can supply an incremental hash (BLAKE3 supports incremental hashing across the rollout), or use a cheaper 64-bit fingerprint for the first-level probe and BLAKE3 only on fingerprint collision.
- **Benefit depends on cache hit rate**: UMF reports ~55% hit rate at NFE=12288 on coding tasks. At low NFE (768), hit rate is 0% and the cache is pure overhead. The GOAT gate must measure hit rate vs NFE on our domain to find the breakeven point.
- **Multi-shard tokenizer mismatch** (riir-ai fusion only): UMF handles heterogeneous MDLM tokenizers via special-token direct map + text re-encode. The shard analog is "different `style_weights[64]` layouts across shards" — solvable via the existing dendritic branch view (`shard::dendritic`), but not free.

---

## 7. References

- Misaki, Akiba. *UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching.* arXiv:2602.04344, Feb 2026.
- Jain et al. *Diffusion Tree Sampling.* NeurIPS 2025 (DTS*, the closest stochastic-MCTS-on-MDLM baseline).
- Inoue et al. *Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search.* NeurIPS 2025 (AB-MCTS, the generic adaptive-branching baseline UMF beats).
- Kocsis, Szepesvári. *Bandit Based Monte-Carlo Planning.* ECML 2006 (UCT).

---

## TL;DR

**Pre-flight done.** UMF is modelless (MCTS + caching + multi-config routing, no training). Verdict **GOAT** — novel state-action pair caching primitive for MCTS over discrete inference-configuration actions, with a provable dominance guarantee (Eq. 1: sum-of-mins ≤ min-of-sums → state-dependent switching policy beats any single static kernel). Lands in `katgpt-rs` (generic search/caching math, no game/chain/shard semantics). Plan 390 opens the GOAT gate benchmark. Not Super-GOAT (Q3 selling point is perf, not capability). No UQ-bearing floor check (UMF produces a single best trajectory, not a probability distribution). No §3.5 modelless-unblock needed (no riir-train deferral contemplated). No §3.6 PoC needed pre-gate (the benchmark IS the gate).
