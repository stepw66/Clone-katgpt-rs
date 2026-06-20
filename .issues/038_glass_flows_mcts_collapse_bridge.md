# Issue 038: GLASS Flows (Remark 21) for MCTS Collapse Bridge

**Opened:** 2026-06-20
**Closed:** 2026-06-20
**Origin:** Research 271 §5 (MIT 6.S184 textbook vocabulary crosswalk)
**Status:** ❌ **CLOSED — NOT APPLICABLE.** Reading the actual GLASS Flows paper (arxiv 2509.25170) revealed the lecture-note Remark over-generalized. GLASS Flows is **narrowly about reward alignment in diffusion models** (SMC, search, guidance) — replacing slow SDE transition sampling with an "inner flow matching model" sampled via ODE. It is **not** a general MCTS-collapse-bridge pattern. `mcts_collapse_bridge.rs` uses a δmg discriminator on MCTS visit statistics, not flow/diffusion sampling, so GLASS doesn't apply. No plan, no Super-GOAT.
**Parent skill rule invoked:** "If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'. Write 'fusion idea — novelty TBD, needs Q1–Q4 check before verdict' and create an issue."

---

## The fusion idea

MIT 6.S184 Remark 21 (GLASS Flows, citing Holderrieth et al. 2025, arxiv 2509.25170): stochastic-looking transition dynamics can be implemented **purely via ODEs** using a sampling trick. This allows search algorithms (MCTS, beam search) over stochastic-looking dynamics while keeping the efficiency and determinism of ODE simulation.

**Proposed fusion:** apply GLASS Flows to `riir-ai/crates/riir-engine/src/cgsp_runtime/mcts_collapse_bridge.rs`. The collapse bridge currently switches between MCTS (stochastic search) and direct flow (deterministic). GLASS Flows could provide a **unified ODE-based formulation** that:
- Looks stochastic to the search algorithm (enables branching / curiosity)
- Is actually deterministic underneath (bit-identical replay, anti-cheat safe)
- Avoids the bridge-switching cost

This connects:
- **MCTS collapse recovery** (existing `mcts_collapse_bridge.rs`)
- **Deterministic replay** (raw sync requirement)
- **Curiosity-driven search** (cgsp_runtime)
- **Freeze/thaw** (no retraining — just a different ODE formulation at runtime)

## Why it might be novel

GLASS Flows (Holderrieth et al. 2025) is recent. A quick codebase grep did not surface "glass_flow", "glass_flow", "ode_stochastic", or "deterministic_stochastic" as shipped primitives. The mcts_collapse_bridge uses an explicit branch-and-switch pattern, not a unified ODE.

## Why it might NOT be novel (Q1–Q4 to run before any verdict)

- **Q1 (no prior art?):** Must grep for: paper vocabulary (`glass_flow`, `transition_sampling`, `alignment_flow`) AND codebase vocabulary (`ode_search`, `deterministic_branch`, `replay_safe_search`, `mcts_ode_unified`). The collapse bridge might already implement the GLASS pattern under a different name.
- **Q2 (new capability class?):** Does unified ODE-based stochastic-looking search enable behavior the current switch-bridge cannot? Or is it just a cleaner implementation of the same capability?
- **Q3 (product selling point?):** "Our NPCs run MCTS-quality strategic search with deterministic-replay safety, no bridge switching." Finish this sentence or downgrade.
- **Q4 (force multiplier ≥2 pillars?):** Connects to MCTS, sync boundary, cgsp, replay verification. Plausible ≥2.

## Caveats

- The GLASS Flows paper (arxiv 2509.25170) is cited only in a Remark in the lecture notes — the actual paper needs to be fetched and read before any verdict. The lecture-note description is too thin to claim novelty from.
- "Search algorithms over stochastic-looking ODE dynamics" is a broad claim. The actual paper may be narrower (e.g., specific to alignment, not general MCTS).

## Q1–Q4 Gate Result (2026-06-20) — ❌ CLOSED NOT APPLICABLE

### Paper read: [arxiv 2509.25170](https://arxiv.org/abs/2509.25170) (GLASS Flows, ICLR 2026)

**The lecture-note Remark was misleading.** Reading the actual paper reveals GLASS Flows is **narrowly scoped** to:

> *"Inference-time reward alignment algorithms ... many algorithms require to sample Markov transitions via SDE sampling, which is significantly less efficient ... GLASS Flows, a method for efficiently sampling flexible Markov transitions via ODEs leveraging pre-trained flow and diffusion models."* (Abstract)

The applications listed in §3 are exclusively:
1. **Sequential Monte Carlo (SMC)** for reward alignment (particles evolved via `pt′|t` proposal)
2. **Search methods** that sample branches from `pt′|t` to build a search tree (specifically: inference-time reward alignment for diffusion models, citing Li et al. 2025b, Zhang et al. 2025)
3. **Guidance methods** that modify vector fields `ut → ut + ct∇rt(x)` for reward alignment

The key constraint (§4): GLASS requires a **pre-trained flow matching / diffusion model** with vector field `ut(x)` and denoiser `Dt(x)`. The whole construction depends on reparameterizing `ut` into `Dt` and back via sufficient statistics. **There is no flow matching model in `mcts_collapse_bridge.rs` to reparameterize.**

### Q1: No prior art? — N/A (not applicable)

GLASS doesn't apply to `mcts_collapse_bridge.rs` because:
- The bridge operates on **MCTS visit statistics** (`weighted_mean_q`, `unweighted_mean_q`, `unique_branches`, `total_visits`) — discrete tree-search diagnostics
- It computes a **scalar δmg divergence** (`‖Q_weighted − Q_unweighted‖`) and cross-validates against CGSP's entropy-based collapse detector
- It returns a **4-valued verdict enum** (`ForceAggressive`, `MctsOnly`, `CgspOnly`, `NoCollapse`)

None of this involves sampling from a transition kernel `pt′|t` of a flow/diffusion model. The bridge is a **post-search diagnostic**, not a sampler.

### Q2: New capability class? — N/A

GLASS Flows would add nothing to the bridge's capability. The bridge doesn't sample transitions; it inspects tree statistics.

### Q3: Product selling point? — N/A

No sentence to finish. The fusion was based on a misreading of the lecture-note Remark.

### Q4: Force multiplier ≥2 pillars? — N/A

The fusion doesn't connect to the existing `mcts_collapse_bridge.rs` mechanism.

### Verdict

**CLOSED NOT APPLICABLE.** No plan, no Super-GOAT, no guide.

### Where GLASS Flows *would* be applicable in our codebase (if anywhere)

For completeness: if we ever build a **reward-aligned diffusion policy** (e.g., NPC behavior diffusion model with SMC steering), GLASS Flows would be the right primitive to replace SDE sampling there. Currently we have no such model — `katgpt-rs/src/speculative/d2f.rs` (D2F discrete diffusion) is a generative decoder, not a reward-aligned policy. So GLASS remains on the shelf until/unless we ship a continuous-flow NPC policy that needs SMC reward steering. Not worth an issue today.

### Lesson (lecture-note Remark over-generalization)

The MIT 6.S184 lecture notes Remark 21 said: *"it is also possible to get the same stochastic transitions purely via ODEs via a simple sampling trick called GLASS Flows. This allows to exploit the stochastic nature of SDEs (e.g., via search algorithms) while keeping the efficiency of ODEs."* This phrasing suggested broad applicability to "search algorithms" in general. The actual paper is narrower: it specifically addresses search/SMC/guidance algorithms **that already use diffusion-model SDE sampling** for reward alignment. Issue 038 was created from the broad lecture-note phrasing without verifying against the paper. Fixing Research 271 §2 to note GLASS's actual scope.

## Reading list (post-mortem, no action needed)

- MIT 6.S184 lecture notes §4.2 Remark 21 (GLASS Flows)
- GLASS Flows paper: [arxiv 2509.25170](https://arxiv.org/abs/2509.25170) — **must read before any verdict**
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` (vocabulary crosswalk)
- `riir-ai/crates/riir-engine/src/cgsp_runtime/mcts_collapse_bridge.rs` (existing collapse bridge)
- `katgpt-rs/.research/215_ECHO_Environment_Prediction_Inference_Time.md` (related inference-time prediction)
