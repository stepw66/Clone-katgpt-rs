# Research 370: Manifold Bandits — Latent Task Tree + Hierarchical Thompson Sampling

> **Source:** McKenzie, Hansen, Wang. *Manifold Bandits: Bayesian Curriculum Learning over the Latent Geometry of Large Language Models*. [arXiv:2606.19750](https://arxiv.org/abs/2606.19750). UCSD. v1, 18 Jun 2026. Code: [github.com/DarrienMcKenzie/manifold-bandits](https://github.com/DarrienMcKenzie/manifold-bandits) (MIT).
> **Date:** 2026-07-03
> **Status:** Active — open GOAT plan spawned.
> **Related Research:** 030/032/025 (multi-armed bandit — flat Thompson, contextual OUT OF SCOPE), 118/155 (LEO AutocurriculumSampler — uniform, flat), 279 (subspace clustering phase gate), 294 (Viable Manifold Graph), 219/296 (DEC Stokes calculus — tree ≡ cell complex), 363 (realtime RL planning budget gate).
> **Related Plans:** 030 (multi-armed bandit PoC — the gap this fills), 155 (LEO All-Goals trait framework), 282 (Dual-Pool Reachable Router — non-stationary handling via dual-pool, not manifold), 312 (riir-ai Dual-Pool CGSP — non-stationary regret, default-on), 251 (DEC operators — tree as cell complex substrate).
> **Cross-ref (riir-train):** the BMC training curriculum (GSPO/GRPO RL on Qwen3-8B) is **training-only** and routes to `riir-train/.research/` (not created in this session — see §3.5 modelless unblock protocol; the value of the *training loop* is in the loop, not transferable here).
> **Classification:** Public (katgpt-rs engine note).

---

## TL;DR

Manifold Bandits ships two components: (1) **Latent Task Trees** — a hierarchical clustering of a problem set by the LLM's own latent representations (PCA → UMAP → Chart Test → HDBSCAN, recursively), and (2) **Bayesian Manifold Curriculum (BMC)** — Hierarchical Thompson Sampling top-down through the tree + per-prompt non-stationary Bayesian Filtering belief updates + bottom-up Empirical Bayes tree updates, used to sample problems during RL-based LLM training.

The **BMC training loop is training-only** (RL curriculum for Qwen3-8B with GSPO/GRPO) and routes to `riir-train`. But the paper's value for **us** is the **modelless bandit formulation**: arms are *related through latent geometry* (not independent), sampling navigates a *hierarchy* (not a flat list), and beliefs are updated by *non-stationary Bayesian Filtering* (not just a Beta conjugate posterior). This is a **contextual, structure-aware, hierarchical Thompson sampler** — and contextual bandits are an **explicitly documented gap** in our bandit stack (Plan 030 "Out of Scope: Contextual bandits (requires feature vectors per arm)"; Plan 032 and riir-ai Plan 025 repeat the same deferral).

**Distilled for katgpt-rs (modelless, inference-time):**

1. **`LatentTaskTree`** — a frozen, offline-built hierarchical clustering of an arm/query/item space by latent similarity. Construction is fully modelless: extract embeddings (already shipped — HLA projection, `ItemEmbedIndex`, `SenseModule::project`), reduce (PCA), embed (UMAP — deterministic with fixed seed), test chart coverage (manifold locality), cluster (HDBSCAN — density-based, no k). Recurse on each cluster until singleton leaves. The tree is **immutable at inference time** (BLAKE3-committable, freeze/thaw-compatible) — it's a routing substrate, not a learned model.
2. **`HierarchicalThompsonSampler`** — top-down Thompson sampling through the tree: sample a child at each node from a Beta(α, β) posterior, descend, repeat to a leaf. Each node carries its own Beta pair; sibling nodes share a parent whose posterior is the **Empirical Bayes** aggregation of children (bottom-up update from observation). This is *strictly more informative* than flat Thompson because the tree encodes prior structure: a reward observed on one arm updates beliefs on its siblings and ancestors.
3. **`BayesianFilterArm`** — per-arm non-stationary belief tracking via a Bayesian Filter (predict-update with a transition model + observation update). Unlike the flat Beta conjugate posterior (stationary assumption), this handles **drifting reward distributions** — the documented gap in Plan 030 ("Non-stationary environments (adversarial bandits) — Out of Scope"). Our Dual-Pool CGSP (Plan 312, default-on) handles non-stationarity via a *dual-pool mechanism* (E-pool exploit / X-pool explore); this primitive handles it via *belief filtering* — a complementary, composable approach.

**Routing:** open primitive in `katgpt-rs/crates/katgpt-core/src/` (generic math, no game/shard semantics) behind a `manifold_bandit` feature flag. The training curriculum → `riir-train` (one-line redirect, no files created this session).

---

## 1. Paper Core Findings

### 1.1 The framing — manifold-structured bandits with endogenous non-stationarity

Existing adaptive curriculum methods treat problem selection as a standard bandit with **independent arms** and **stationary rewards** (reward = pass/fail on a fixed-difficulty problem). The paper argues two structural facts are ignored:

- **Problems are related through the model's latent representation space** — two math problems that look syntactically different may project to nearby points in the LLM's embedding manifold, so learning signal on one *should* inform belief about the other.
- **Sampling decisions steer how learning signals evolve** — the reward distribution is *endogenously non-stationary* because the model itself is changing (in RL training) OR the task distribution is shifting (in deployment). A bandit that assumes stationarity will lock onto a stale optimal arm.

This is operationalized as: arm space = leaf nodes of a **Latent Task Tree**, structured as a manifold; reward = rollout outcome; belief updates = Bayesian Filtering (per-arm) + Empirical Bayes (tree-wide).

### 1.2 Latent Task Trees (modelless construction)

The tree is built offline, once, from a fixed problem set:

1. **Latent extraction** — run each problem through the model, extract a representation (last-layer hidden state, mean-pooled token embeddings, or task-vector). Modelless for us — `SenseModule::project`, `ItemEmbedIndex::query`, HLA dot-projections all produce these.
2. **Dimensionality reduction** — PCA to a manageable dim, then UMAP to 2D for clustering stability. Both deterministic given a fixed seed.
3. **Chart Test** — a manifold-coverage test (from topological data analysis): checks whether the local neighborhood of each point is approximately linear (i.e., the data really does lie on a manifold, not a blob). Points failing the chart test are outliers / boundary cases.
4. **HDBSCAN clustering** — density-based hierarchical clustering (no k, handles variable-density clusters, produces a noise label). This produces the **top-level split**.
5. **Recursion** — apply steps 2–4 to each non-singleton cluster until leaves are singletons or below a min-cluster-size threshold.

The result is a tree where each internal node is a region of latent space, each leaf is a problem (or a small homogeneous group), and the path from root to leaf encodes the manifold geometry. **Construction is fully modelless** — no training, no gradient descent, just deterministic reductions + density clustering.

### 1.3 Bayesian Manifold Curriculum (BMC) — three-part sampling

BMC runs at training time (this part → riir-train), but the **algorithmic structure** is what we distill:

1. **Top-down problem selection (Hierarchical Thompson Sampling).** Starting at the root, sample a child according to a Beta(α_child, β_child) posterior (Thompson sampling). Descend into the sampled child, repeat. Reach a leaf, emit that problem. This is *structure-aware*: siblings compete for selection within their parent's posterior, and the tree topology enforces that you can't sample from two distant manifold regions in the same draw without paying the hierarchical Thompson cost twice.
2. **Non-stationary belief modeling (per-prompt Bayesian Filtering).** After observing a rollout outcome (reward) on a selected prompt, update that prompt's belief via a Bayesian Filter — not just a Beta conjugate update, but a predict-update cycle with a (small) transition model that allows the believed reward to drift over time. This is the non-stationarity handling: the filter tracks *where the reward distribution is now*, not where it was at the start.
3. **Bottom-up tree update (Empirical Bayes).** Aggregate child beliefs into parent beliefs: the parent's Beta posterior is the empirical Bayes estimate from its children's observed rewards. This propagates learning signal *across the manifold structure* — a reward on one arm updates beliefs on its siblings and ancestors, which is the structural advantage over independent-arm bandits.

### 1.4 BMC-T — target-driven utility

BMC-T extends BMC with a **target set**: a separate set of prompts (could be the eval set, could be a different distribution) whose latent positions define a utility bonus. Before training, the overlap between train and target trees is precomputed; the utility of a training prompt is its overlap with target-region leaves. The Thompson sampler then optimizes **productivity × utility** (learning signal × evaluation relevance), not just productivity. Empirically, BMC-T trades diversity for utility — useful when you know what you'll be evaluated on.

### 1.5 Empirical findings (the productivity/diversity/utility tradeoff)

The paper's headline empirical finding is a **three-way tradeoff**:
- **Productivity** (learning signal) — maximized by intermediate-difficulty sampling (classic result).
- **Diversity** (manifold coverage) — maximized by uniform / entropy-spread sampling.
- **Utility** (eval relevance) — maximized by target-driven sampling (BMC-T).

No single strategy dominates all three. **Prioritizing difficulty alone is insufficient** — the paper shows this empirically by comparing BMC against difficulty-only baselines and finding BMC wins on downstream eval precisely because it preserves diversity. This is the structural insight we care about: **flat bandits trade productivity for diversity implicitly via ε; structure-aware bandits get both for free via the tree.**

---

## 2. Distillation

### 2.1 What's modelless (stays in katgpt-rs)

| Component | Modelless? | Why |
|---|---|---|
| Latent Task Tree construction (PCA + UMAP + Chart Test + HDBSCAN) | ✅ | Deterministic reductions + density clustering. No training, no GD. Fixed-seed reproducible. BLAKE3-committable. |
| Hierarchical Thompson Sampling (top-down Beta sampling) | ✅ | Pure posterior sampling on a fixed tree. The Beta posteriors are runtime state (latent), updated by observation — no weight mutation. |
| Per-arm Bayesian Filtering (non-stationary belief) | ✅ | A predict-update filter on a scalar belief per arm. The transition model is a small config (drift rate). No training. |
| Empirical Bayes tree update (bottom-up aggregation) | ✅ | Deterministic aggregation of child beliefs into parent. Pure arithmetic. |
| BMC-T utility precomputation (train/target overlap) | ✅ | One-time offline pass: project target set into the tree, count overlaps. Deterministic. |

### 2.2 What's training-only (→ riir-train)

| Component | Why training-only |
|---|---|
| The BMC loop wired into GSPO/GRPO RL training of Qwen3-8B | The value is the *training curriculum* — sampling problems during gradient descent on base weights. The Bayesian updates consume rollout *training rewards*. This is the loop that mutates weights. |
| Non-stationary belief updates driven by *training* reward drift | The non-stationarity in the paper comes from the model itself improving during training. At inference time (our case), the model is frozen — non-stationarity comes from the *environment* (game balance shifts, player distribution changes), which is a different signal and is already handled by Dual-Pool CGSP. |

**§3.5 modelless unblock check** (mandatory before any riir-train deferral):
- *Path 1 (freeze/thaw snapshot correction)*: the Latent Task Tree is a frozen, BLAKE3-committable artifact — it IS a freeze/thaw-compatible routing substrate. The training-loop belief drift is not correctable by snapshot (it's the training signal itself), but the *tree* ships modellessly. ✅ for the tree.
- *Path 2 (raw/lora reader-writer hot-swap)*: not applicable — no LoRA overlay corrects a sampling distribution. The bandit is already modelless. ✅ (vacuously).
- *Path 3 (latent-space correction)*: the per-arm Bayesian Filter IS a latent-space correction (scalar belief update via dot-product-style predict-update). ✅.

All three modelless paths cover the *inference* use. The training-loop use (BMC inside GSPO) is genuinely training — note "→ riir-train" for that slice only.

### 2.3 The gap this fills (prior art, both layers, vocabulary-translated)

**Paper vocabulary grep** (`Thompson`, `BanditSampler`, `manifold bandit`, `Latent Task Tree`, `curriculum`, `Empirical Bayes`, `Bayesian Filtering`, `non-stationary`):

- **Zero hits** on `Latent.Task.Tree`, `manifold.bandit`, `Empirical.Bayes`, `Bayesian.Filter` — the specific combination is novel to our corpus.
- `Thompson` / `ThompsonSampling`: **shipped flat** in `katgpt-rs/crates/katgpt-core/src/pruners/bandit.rs` (Plan 030) — `BanditStrategy::ThompsonSampling` (Beta conjugate posterior via Jöhnk's algorithm), used by `BanditPruner`, `MetaRouter` (Plan 196), `DeltaBanditPruner`, `BanditFrameSampler` (riir-ai combat). **All flat, independent-arm.**
- `AutocurriculumSampler` (Plan 155, default-on SUPER GOAT): **uniform sampling** over observed goals — `sample_goal`, `observe_goal`, `update_goals_seen`. Flat, not manifold-structured.
- `curriculum`: appears in SDAR (R038, sigmoid-gate token curriculum), Survive-or-Collapse (R075, dataset eligibility), LEO (R118, autocurriculum). None are manifold-structured hierarchical Thompson.
- `non-stationary`: Dual-Pool CGSP (Plan 312, default-on) handles non-stationary *reward* domains via dual-pool mechanism (E-pool exploit / X-pool explore). Plan 030 explicitly lists "Non-stationary environments (adversarial bandits)" as **OUT OF SCOPE**. Plan 025 (riir-ai) and Plan 032 both defer contextual bandits as future work.

**Codebase vocabulary grep** (`BanditPruner`, `BanditStrategy`, `BanditStats`, `contextual_bandit`, `hierarchical`, `cell_complex`, `CochainField`):

- The flat bandit is wired into: `MetaRouter` (VortexFlow policy selection), `GZeroPlayer` / `RmsdPlayer` / `RubricPlayer` (bomber template selection), `BanditFrameSampler` (riir-ai combat frame sampling), `refinement_bandit.rs` (riir-train refinement depth). All independent-arm.
- DEC substrate (`katgpt-rs/crates/katgpt-core/src/dec/`): `CellComplex`, `CochainField`, `exterior_derivative`, `codifferential`, `hodge_decompose` — a tree is *literally* a 1D hierarchical cell complex, and Thompson sampling / Empirical Bayes updates are cochain pushforward / pullback operations. **This is the Super-GOAT fusion hook (see §2.5).**
- Subspace clustering prior art: `phase_gate.rs` (riir-neuron-db, Participation Ratio + Numerical Rank + Jacobian SVD), `diverse_retrieval` (wedge-diverse shard ensemble), `TEMP_Perturbed_Loss_Vector` (Plan 005, Lipschitz-bound diversity selection), R279 (Diffusion ≡ Subspace Clustering). None produce a *hierarchical routing tree + bandit*.

**Verdict on prior art:** the *flat* Thompson bandit is heavily shipped and battle-tested; the *contextual / manifold-structured / hierarchical* variant is an **explicitly documented gap** (Plans 030, 032, 025 all defer it). This paper is the canonical reference for closing that gap.

### 2.4 Latent-space reframing (mandatory before verdict)

How the mechanism looks on each Super-GOAT factory module:

| Module | Reframing |
|---|---|
| **HLA** (`katgpt-core/src/sense/`) | A Latent Task Tree over the HLA affect space (valence/arousal/desperation/calm/fear + 3) clusters NPCs by emotional archetype. Hierarchical Thompson sampling selects which *emotional region* to explore next — a per-NPC curiosity router that respects affect geometry instead of flat ε-exploration. |
| **`latent_functor/`** (riir-engine) | Tree navigation IS a functor application: each level applies a "select-child-and-descend" functor. The Beta posterior at each node is the functor's *state*. This composes with `reestimation.rs` (coherence-driven re-estimation) — a stale subtree triggers a belief reset. |
| **`cgsp_runtime/`** (riir-engine) | Curiosity signal becomes the *reward* for the bandit. The Dual-Pool CGSP (E-pool/X-pool) and the manifold bandit compose: X-pool conjecture samples *which manifold region* to explore (hierarchical Thompson), E-pool exploits *within* the selected region. |
| **LatCal** (riir-chain `encoding/`) | Empirical Bayes tree updates are *deterministic aggregations* — they can be committed via LatCal fixed-point bridges if the tree state crosses the sync boundary (e.g., a shared quest-distribution tree across quorum nodes). The Beta posteriors cross as raw scalars (α, β per node), never the full embedding. |
| **`NeuronShard`** (riir-neuron-db) | A Latent Task Tree over `style_weights[64]` space organizes shards by latent similarity — a *hierarchical ShardIndex*. Retrieval becomes hierarchical Thompson sampling: sample a region, then a shard within it. Composes with `diverse_retrieval` (wedge-diverse selection) — the tree gives structure, the wedge gives diversity. |
| **DEC** (`katgpt-core/src/dec/`) | **The tree is a 1D hierarchical cell complex.** Thompson sampling top-down = cochain pushforward (push belief from root to leaf). Empirical Bayes bottom-up = cochain pullback / codifferential (aggregate child beliefs into parent). The `exterior_derivative` d on the belief cochain measures *belief divergence* across the tree — a formal notion of "how much does this region's belief disagree with its parent?" See §2.5. |

### 2.5 Fusion (per §Workflow step 5)

**Closest cousins across all five repos:**

1. **Plan 030 (Multi-Armed Bandit PoC)** — the flat Thompson sampler this generalizes. Explicitly defers contextual + non-stationary. **The gap.**
2. **Plan 155 (LEO AutocurriculumSampler)** — default-on SUPER GOAT, uniform sampling over observed goals. The manifold bandit is the *structure-aware* upgrade: same trait shape (`sample_goal`, `observe_goal`), but sampling is hierarchical Thompson over a latent tree instead of uniform.
3. **Plan 312 (Dual-Pool CGSP)** — default-on for non-stationary reward domains. Handles non-stationarity via dual-pool; the manifold bandit handles it via Bayesian Filtering. **Composable, not competing** — Dual-Pool picks *which pool*, manifold bandit picks *which arm within the structure*.
4. **Plan 251 (DEC operators)** — the tree-as-cell-complex substrate. Thompson sampling = cochain pushforward; Empirical Bayes = cochain pullback.
5. **R279 (subspace clustering phase gate)** — the N≥d phase transition is the *sufficiency check* for whether a cluster in the Latent Task Tree has enough samples to support a stable Beta posterior. Fusion: a node with N<d children should *not* propagate belief (it's below the phase transition — its posterior is noise).

**Fusion idea (novelty TBD — needs PoC per §3.6 before any Super-GOAT claim):**

> **Manifold Thompson Cochain Router** — a `LatentTaskTree` built as a hierarchical cell complex (DEC substrate), where:
> - Each node carries a `BayesianFilterArm` (non-stationary Beta belief).
> - Top-down sampling = cochain pushforward (Thompson descent).
> - Bottom-up update = codifferential (Empirical Bayes aggregation), gated by the R279 N≥d phase transition (don't aggregate from under-sampled subtrees).
> - Belief divergence across the tree = `exterior_derivative` on the belief cochain — a formal "where is the model's belief most inconsistent with its structure?" signal, usable as a curiosity trigger for `cgsp_runtime`.
>
> This fuses Plan 030 (bandit) × Plan 155 (autocurriculum) × Plan 251 (DEC) × R279 (phase gate) into a single structure-aware routing primitive. **If it beats flat Thompson on a structured-domain benchmark (the productivity × diversity frontier from §1.5), it's a GOAT. If the DEC-cochain reframing also beats a naive-tree implementation (i.e., the cochain operations are not just notation but compute), it's a Super-GOAT candidate — but that requires a PoC at `riir-ai/crates/riir-poc/` per §3.6 before any Super-GOAT verdict.**

This fusion is recorded here as a **fusion idea — novelty TBD, needs Q1–Q4 check + PoC before verdict** (per §1.5, "candidate" language triggers mandatory outputs, so I am NOT calling it a candidate yet).

---

## 3. Verdict

**Tier: GOAT** (open primitive in katgpt-rs; training curriculum → riir-train).

| Criterion | Result |
|---|---|
| Modelless? | ✅ — tree construction is deterministic reductions + density clustering; sampling is posterior sampling on a fixed tree; updates are scalar Bayesian filtering. No GD, no weight mutation. |
| Fills a documented gap? | ✅ — Plan 030/032/025 all explicitly defer contextual + non-stationary bandits. This is the canonical reference for closing that gap. |
| Provable gain? | ✅ — the paper's productivity × diversity × utility tradeoff (§1.5) is directly benchmarkable: hierarchical Thompson vs flat Thompson on a structured-domain bandit (arms clustered in latent space). The structural advantage (belief propagation across siblings) is theoretically grounded and empirically demonstrated in the paper. |
| Force multiplier? | ✅ — composes with AutocurriculumSampler (P155), Dual-Pool CGSP (P312), DEC operators (P251), subspace phase gate (R279), HLA per-NPC state, ShardIndex retrieval. ≥2 pillars. |
| New capability class? | ⚠️ Partial — structure-aware sampling is qualitatively different from flat sampling, but it's a *refinement* of the bandit primitive, not a new class. We have bandits; this is a better bandit. **Not Super-GOAT.** |

**One-line reasoning:** Fills the explicitly-documented contextual/manifold bandit gap (Plans 030/032/025) with a modelless, structure-aware hierarchical Thompson sampler that composes with our existing bandit/autocurriculum/DEC substrate. The gain is provable (productivity × diversity frontier), the routing is correct (open primitive in katgpt-rs, training loop in riir-train), and the fusion angle (DEC cochain router) is recorded as a TBD Super-GOAT candidate pending PoC.

**MOAT gate (§1.6) — katgpt-rs domain:**
- **In scope:** paper-derived fundamental/principle primitive (manifold-structured contextual bandit) passing GOAT via fusion with existing bandit + DEC + autocurriculum substrate. Promote/demote tracked per stack: this lands in the **bandit/pruning stack slot** (alongside `BanditPruner`, `MetaRouter`, `DeltaBanditPruner`). If it beats flat Thompson on structured domains → promote to default for structure-aware callers; flat Thompson stays default for unstructured callers (no forced demote — they're different domains).
- **Not Super-GOAT** → no private guide created this session. The DEC-cochain fusion (§2.5) is tracked as a TBD candidate; if a future PoC proves the cochain reframing beats a naive tree, *then* create the riir-ai guide.

**§3.6 defend-wrong check:** this verdict makes **no quality-parity claim** against the paper (we are not claiming "matches BMC's downstream eval gains" — those are training-loop results that belong in riir-train). The GOAT claim is *architectural + latency* only: (a) the primitive exists modellessly (architectural — proven by construction in §2.1), (b) it's sub-µs per sample (latency — to be benchmarked in the plan). **No PoC required for this verdict** — no quality parity is asserted. If a future plan claims "matches BMC's productivity × diversity frontier at inference time", *then* a PoC at `riir-ai/crates/riir-poc/` becomes mandatory per §3.6.

---

## 4. Follow-ups

- [ ] **Plan 370 (katgpt-rs)** — open primitive: `LatentTaskTree` + `HierarchicalThompsonSampler` + `BayesianFilterArm` behind `manifold_bandit` feature flag. GOAT gate: hierarchical Thompson vs flat Thompson on a structured-domain bandit (arms clustered in latent space), measuring productivity × diversity frontier. Latency budget: sub-µs per sample (plasma tier).
- [ ] **riir-train note** — the BMC training curriculum (GSPO/GRPO RL on Qwen3-8B) is training-only. Note "→ riir-train" with this research as cross-ref. **Not created this session** (out of scope for this workflow).
- [ ] **Issue (katgpt-rs/.issues/)** — track the §2.5 DEC-cochain fusion as a TBD Super-GOAT candidate. If a future PoC at `riir-ai/crates/riir-poc/` proves the cochain reframing (Thompson = pushforward, Empirical Bayes = pullback) beats a naive-tree implementation *on compute*, then create the riir-ai guide and re-gate as Super-GOAT.

---

## TL;DR

Manifold Bandits (arXiv:2606.19750) is a **training paper at its headline** (Bayesian Manifold Curriculum for RL-based LLM training — routes to riir-train, one-line redirect, no files created this session). Its **modelless distillation** — a **Latent Task Tree** (offline-built hierarchical clustering of an arm space by latent similarity, via PCA + UMAP + Chart Test + HDBSCAN) + **Hierarchical Thompson Sampling** (top-down Beta posterior descent) + **per-arm Bayesian Filtering** (non-stationary belief) + **Empirical Bayes** (bottom-up tree update) — fills an **explicitly documented gap** in our bandit stack (Plans 030/032/025 all defer contextual + non-stationary bandits as out-of-scope). Verdict: **GOAT** — open primitive in `katgpt-rs/crates/katgpt-core/src/` behind `manifold_bandit` feature flag, with a recorded **DEC-cochain fusion idea** (tree ≡ cell complex, Thompson ≡ pushforward, Empirical Bayes ≡ pullback, gated by R279 N≥d phase transition) tracked as a TBD Super-GOAT candidate pending a §3.6 PoC. Composes with AutocurriculumSampler (P155, structure-aware upgrade), Dual-Pool CGSP (P312, complementary non-stationarity handling), DEC operators (P251, cochain substrate), and subspace phase gate (R279, sufficiency check for belief propagation). The honest move is GOAT-now, Super-GOAT-maybe-after-PoC — no parity claim is made against the paper's training-loop results.
