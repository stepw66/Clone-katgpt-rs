# Research 394: GNN Survey — Within-Class Effective Rank Fusion + Oversquashing Diagnostics

> **Source:** "Introduction to Graph Neural Networks for Machine Learning Engineers" (Tanis, Giannella, Mariano, Meerzaman; MITRE Corporation + National Cancer Institute) — [arXiv:2412.19419](https://arxiv.org/abs/2412.19419), v2 2026-06-01, 73 pp.
> **Date:** 2026-07-09
> **Status:** Done
> **Related Research:** 113 (NITP representation geometry — ships `effective_rank`), 123 (sigmoid margin), 219 (DEC), 296 (Stokes vocab crosswalk), 300 (closed-unit compaction rubric)
> **Related Plans:** 151 (NITP geometry diagnostics — ships `effective_rank`), 251/252 (DEC operators), 303 (latent_functor quality gate — ships `within_class_adjacency`), 336 (committed personality blend)
> **Classification:** Public

---

## TL;DR

A 73-page MITRE/NCI survey/tutorial on Graph Neural Networks. The bulk of the paper — message-passing encoders (GCN / GraphSAGE / GATv2), the encoder-decoder training framework, hyperparameter/architecture tuning, RevGNN comparison, LLM-based explainability — is **training-only** (→ riir-train, out of scope) or **already pervasive** in our codebase (the `sigmoid(z_i^T z_j)` link-prediction decoder is the CLR verifier, bridge attention, and BCKVSS affinity, all over the repo). **One genuinely novel modelless primitive survives distillation: the Within-Class Effective Rank** (entropy-based effective rank of the *within-class residual* covariance, claimed novel by the authors) — and it is a **fusion of two shipped halves** (`data_probe/geometry.rs::effective_rank` × `latent_functor/quality_gate.rs::within_class_adjacency`) that have never been combined. A second novel metric, **MSBE (Message Squeeze per Bridge Edge)**, is model-aware (Jacobian-based) and has a modelless structural analog already shipped (DEC `exterior_derivative` boundary operator), so it is a Gain for offline zone-topology analysis only.

**Distilled for katgpt-rs (modelless, inference-time):**
- **GOAT** — Within-Class Effective Rank: `effective_rank` applied to the within-class residual covariance matrix. Fuses `data_probe::effective_rank` with the class-conditioning machinery of `latent_functor::quality_gate`. Modelless collapse diagnostic for any class-labeled latent state (NPC personalities, HLA emotion classes, archetype blends, KG-cluster memberships).
- **Gain** — MSBE: Jacobian-influence ÷ bridge-edge-count oversquashing diagnostic. Model-aware; deferred to offline analysis. Structural analog (DEC boundary operator) already ships.
- **Gain** — NCMq0.05: nearest-centroid 5th-percentile margin in whitened space. Offline decision-boundary diagnostic.
- **PASS** — the GNN training machinery (GCN/GraphSAGE/GATv2 message passing, encoder-decoder training, RevGNN, LLM-explainability) → riir-train. The link-prediction decoder `sigmoid(z_i^T z_j)` → already pervasive (CLR verifier, rat_bridge, BCKVSS).

---

## 1. Paper Core Findings

A survey, not a single mechanism. Three transferable items plus a large training-only body.

### 1.1 Within-Class Effective Rank (oversmoothing diagnostic) — NOVEL per authors

Paper §5.3.1 + Supplementary S.1.2. The authors claim (§5.3.1): *"to our knowledge, applying effective rank specifically to the covariance of within-class residuals as an oversmoothing diagnostic is novel."*

**Definition.** Given node features `{x_i}` with class labels `{y_i}`:

1. Compute pooled within-class covariance `Σ_w = (1/Σ(n_c−1)) Σ_c Σ_{i∈S_c} (x_i − μ_c)(x_i − μ_c)^T`, where `μ_c` is the class-`c` centroid.
2. Eigendecompose `Σ_w → {λ_1, …, λ_d}`, normalize to probabilities `p_i = λ_i / Σ λ_j`.
3. Shannon entropy `H(p) = −Σ p_i log p_i`.
4. **Within-Class Effective Rank** `r_WC(Σ_w) = exp(H(p)) ∈ [1, d]`.

Interpretation: the effective number of independent directions of within-class variation. Lower → more oversmoothing / representation collapse. The paper shows it decreases monotonically with GNN depth for all three architectures (GCN collapses fastest, GATv2 retains the most, GraphSAGE in between), and that residual skip-sum preserves it best among skip strategies.

### 1.2 MSBE — Message Squeeze per Bridge Edge (oversquashing diagnostic) — NOVEL per authors

Paper §5.3.2 + Supplementary S.1.4. Authors claim (§5.3.2): *"the first model-aware diagnostic that combines node-to-node influence with an expander-style normalization."*

**Definition.** For target node `n` and radius `r`:
- `B_r(n) = {u : dist(u, n) ≤ r}` (the r-hop ball)
- `∂B_r(n) = {(u,v) ∈ E : u ∈ B_r(n), v ∉ B_r(n)}` (bridge edges crossing the boundary)
- `F_r(n) = {u : dist(u, n) > r}` (the far set)
- `J(u → n) = (1/√d_eff) ||g̃(u→n)||_2` where `g̃(u→n) = (1/σ_k) ∂φ(n)/∂x_u` (standardized node-to-node Jacobian influence)
- **MSBE_r(n) = (Σ_{u∈F_r(n)} J(u→n)) / (|∂B_r(n)| + ε)**

Interpretation: how much far-node signal must squeeze through each bridge edge. Higher → worse oversquashing. The paper also defines ΔMSBE — the change after rewiring (adding shortcut edges to high-influence far nodes) — as a mitigation-effectiveness probe.

### 1.3 NCMq0.05 — Nearest-Centroid 5th-percentile Margin

Supplementary S.1.3. After whitening by `Σ_w,tr`, compute for each test node the signed distance to the nearest competing decision boundary; take the 5th percentile. Lower (more negative) → more nodes near decision boundaries. Tracks oversmoothing from the *margin* side (complementary to effective-rank-from-the-rank side).

### 1.4 Link prediction decoder `sigmoid(z_i^T z_j)` — pervasive in our codebase

Paper §4.2 eq. (37): `Dec(z_i, z_j) = sigmoid(z_i^T z_j)`, trained with binary cross-entropy against adjacency. **Already pervasive** in our codebase (see §2 prior-art table).

### 1.5 Modularity for community detection (paper §4.2.4, eq. 45–51)

`Q = (1/2|E|) Tr(U^T B U)` where `B = A − dd^T/2|E|`. Used as a differentiable loss for end-to-end GNN community detection. **Not shipped** in our codebase (no `modularity` / `community_detect` hits).

### 1.6 Training-only body (out of scope)

- GCN / GraphSAGE / GATv2 message-passing layers with learnable parameters (`ψ`, `ϕ`, attention weights `a`) — require backprop through base weights → **→ riir-train**.
- Encoder-decoder training framework, inductive vs transductive paradigms.
- Pre-/post-processing FC layers, skip-connection variants (residual sum, JK-max).
- Hyperparameter tuning sweep (Table 7), RevGNN comparison (Table 10).
- LLM-based explainability (GraphXAIN, GraphNarrator) — semantic text generation, no modelless analog per the R368 rule (LLM-as-mechanism, not LLM-as-implementation).

---

## 2. Distillation

### 2.1 Prior-art check (three-layer: notes + code + vocab translation)

| Paper concept | Codebase prior art | Verdict |
|---|---|---|
| Within-Class Effective Rank (entropy rank on within-class residual covariance) | `effective_rank()` ships in `crates/katgpt-core/src/data_probe/geometry.rs` (Roy & Vetterli 2007, applied to **raw** hidden states — no class conditioning). Separately, `within_class_adjacency` + `between_class_adjacency` + `score_direction` ship in `riir-ai/crates/riir-engine/src/latent_functor/quality_gate.rs` (Dirichlet energy over class-conditioned adjacency, Plan 303 T5.1). **The two halves have never been fused**: nowhere do we compute `effective_rank` of the *within-class residual* covariance matrix. | **GOAT fusion** — novel combination of two shipped primitives |
| MSBE (Jacobian influence ÷ bridge-edge count) | None. The DEC `exterior_derivative` (in `katgpt-dec`) is a *structural* coboundary operator (counts boundary cells), not a model-aware Jacobian-influence metric. `katgpt-core::roofline` and `ane_roofline` compute compute/memory bottlenecks, a different concept. | **Gain** — novel primitive, but model-aware (needs a trained Jacobian); structural analog ships |
| NCMq0.05 (5th-pct margin in whitened space) | `nearest_centroid_accuracy` in `bench_319_geometric_product_goat.rs` and `SchemaCentroidCache` (Plan 237) compute centroids; the *percentile margin* in whitened space is not shipped. | **Gain** — offline decision-boundary diagnostic |
| `sigmoid(z_i^T z_j)` link decoder | Pervasive: `katgpt-claim/src/clr/verifier.rs::SigmoidProjectionVerifier` (Plan 284 T1.5), `katgpt-attn/src/rat_bridge/{bridge,fuse,vortex}.rs` (bridge gate), `katgpt-band/src/bckvss.rs::perplexity_proxy`. The paper's eq. (37) is literally our CLR verdict function. | **PASS** — already pervasive |
| Modularity community detection | None. | **Gain** — not shipped, but orthogonal to current pillars |
| GCN/GraphSAGE/GATv2 message passing | Out of scope — training machinery → riir-train | **PASS** → riir-train |

### 2.2 Latent-space reframing (mandatory before verdict)

The Within-Class Effective Rank reframes cleanly onto our latent-state kernels:

- **On HLA per-NPC state** (`riir-engine/src/hla/`): classes = NPC emotional archetypes (e.g., "calm", "afraid", "aggressive"). `r_WC` measures whether HLA's 8-dim latent state preserves within-archetype variation as the runtime evolves it across ticks. A collapse (low `r_WC`) means NPCs of the same archetype are becoming indistinguishable — exactly the failure mode the committed-personality runtime (Plan 336) is supposed to prevent. **Direct fit.**
- **On committed personality blends** (`riir-neuron-db/src/archetype_blend_shard.rs` + `riir-engine` `committed_blend/`): classes = archetype IDs. `r_WC` on the style-weights residual covariance detects when a population of NPCs committed to the same blend has collapsed onto a single point — the failure mode the freeze/thaw cadence exists to avoid.
- **On latent_functor direction vectors** (`riir-engine/src/latent_functor/`): classes = condition labels. `r_WC` is the natural complement to the existing Dirichlet-energy separation ratio in `quality_gate.rs`: the existing gate measures *separation* (between > within), the new metric measures *within-class subspace health* (is the within-class variation still high-dimensional, or has it collapsed?).
- **On `ShardIndex` / `NeuronShard` `style_weights[64]`**: classes = zone-of-origin. `r_WC` on the within-zone residual covariance detects shard-population collapse in a zone — relevant to `ConsolidationPipeline::can_freeze` (Plan 002) which reads `intrinsic_dim` from `style_weights`.
- **On DEC cochains**: no clean reframing — DEC operates on raw scalar fields over a fixed cell complex, not on class-labeled latent embeddings. The boundary operator `d` is the structural analog of MSBE's bridge-edge normalization, but not of the within-class rank.

MSBE reframing is weaker: it is model-aware by construction (needs `∂φ(n)/∂x_u`), and at 20Hz tick with thousands of NPCs we cannot afford per-NPC Jacobian computation. The modelless structural analog (DEC boundary-edge count ÷ ball volume) already ships as `exterior_derivative`. MSBE is therefore an **offline zone-topology analysis** tool at best, not a hot-path primitive.

### 2.3 Fusion

**Fusion A (the GOAT): Within-Class Effective Rank = `effective_rank` × `within_class_adjacency`.**

Compose the two shipped halves into a single function:

```rust
/// Within-class effective rank: entropy-based effective rank of the
/// within-class residual covariance matrix. (Research 394, distilled from
/// arXiv:2412.19419 §5.3.1 + S.1.2.)
///
/// Returns r_WC ∈ [1, min(d, n−C)] where C = number of classes.
/// Lower → more within-class collapse (oversmoothing analog).
pub fn within_class_effective_rank(
    states: &[f32],       // [n × d] flat
    dim: usize,
    class_labels: &[usize], // [n]
) -> f32 { … }
```

Implementation is `effective_rank` with step 2 (centering) replaced by class-mean centering. The covariance is `Σ_w = (1/Σ(n_c−1)) Σ_c Σ_{i∈S_c} (x_i − μ_c)(x_i − μ_c)^T`. Reuses the existing Jacobi eigensolver in `data_probe/geometry.rs`. Zero new dependencies. ~40 lines.

**Closest cousins (across all five repos):**
- `katgpt-rs/.research/113` (NITP) — the parent note that motivated `effective_rank`. Does not mention class conditioning.
- `katgpt-rs/.research/286` (Depth-Invariance) — ships `effective_rank_slope` on recursive latent-state chains, detects `Collapsed` (rank → 1). Class-agnostic.
- `katgpt-rs/.research/300` (Closed-Unit Compaction) — rubric-gated compaction; the rubric *could* consume `within_class_effective_rank` as a "is the population still diverse?" gate.
- `riir-ai/.research/158` (Committed Personality Blend) — the runtime that *needs* this diagnostic to verify personality divergence survives the freeze/thaw cadence.

**Fusion B (Gain, offline): MSBE-structural = DEC `exterior_derivative` boundary count ÷ ball cell-volume.**

Already effectively shipped via the DEC substrate; the paper's contribution is the *model-aware* Jacobian-weighted variant, which is out of scope for hot-path modelless inference. Park as an offline zone-topology diagnostic for riir-ai.

---

## 3. Verdict

### Tiers (high → low)

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier ≥2 pillars | Open primitive → katgpt-rs. Private guide → riir-ai/.research OR riir-chain/.research OR riir-neuron-db/.research. Plans → appropriate repo(s). |
| **GOAT** | Provable gain over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only. | One-line note. No files created. |

### Per-component verdict

| Component | Tier | One-line reasoning |
|---|---|---|
| **Within-Class Effective Rank** (Fusion A) | **GOAT** | Novel fusion of two shipped halves; modelless; provably detects within-class collapse that the shipped class-agnostic `effective_rank` and the shipped class-conditioned Dirichlet-energy gate each miss individually. Not a new capability class — it strengthens existing pillars (committed personality, HLA, latent_functor quality gate) rather than creating one. |
| MSBE (Jacobian-weighted oversquashing) | **Gain** | Novel primitive, but model-aware (needs a trained Jacobian); conflicts with modelless-first and 20Hz budget. Structural analog (DEC boundary operator) already ships. Park as offline zone-topology diagnostic for riir-ai. |
| NCMq0.05 (whitened nearest-centroid margin) | **Gain** | Offline decision-boundary diagnostic; percentiles are cheap but the whitening + per-node margin is not a hot-path fit. |
| Modularity community detection | **Gain** | Not shipped, orthogonal to current pillars; could support faction/emergent-social detection later. |
| GCN/GraphSAGE/GATv2 message passing | **Pass → riir-train** | Training machinery with learnable parameters + backprop. |
| `sigmoid(z_i^T z_j)` link decoder | **Pass** | Already pervasive (CLR verifier, rat_bridge, BCKVSS). |
| LLM explainability (GraphXAIN) | **Pass** | LLM-as-mechanism (semantic text generation) — no modelless analog per R368 rule. |

### MOAT gate per domain (§1.6)

- **katgpt-rs** (this repo): the Within-Class Effective Rank is a **paper-derived fundamental diagnostic** — pure modelless math, no game/chain/shard semantics, fits the `data_probe` pillar alongside the existing `effective_rank`. ✅ In scope. Promote/demote tracked per the existing `data_probe` stack.
- **riir-ai**: would consume the katgpt-rs primitive to diagnose HLA / committed-personality / latent_functor collapse — pillar-level use, but the primitive itself belongs in the public engine. No private guide created (not Super-GOAT).
- **riir-chain / riir-neuron-db**: orthogonal.
- **riir-train**: the GNN training body belongs here; out of scope for this workflow.

### Novelty gate (Q1–Q4) for Fusion A — GOAT, not Super-GOAT

- **Q1 (no prior art?)** — partial. The two halves ship separately; the fusion does not. The paper itself claims the within-class-residual variant is novel in the GNN literature.
- **Q2 (new class of behavior?)** — NO. It is a *better diagnostic* for an existing class (representation collapse). Not a new capability.
- **Q3 (product selling point?)** — weak. "Our NPCs detect their own personality collapse before it ships" is a *quality* claim, not a *capability* claim. A competitor without this metric still ships NPCs; they just ship collapsed ones more often.
- **Q4 (force multiplier ≥2 pillars?)** — borderline. Touches HLA + committed personality + latent_functor quality gate, but the connection is "diagnostic consumed by each" rather than "new composition that none could do alone."

Q2 fails → **GOAT, not Super-GOAT.** No private guide required.

---

## 4. Plan 415 (executed 2026-07-09) — Fusion A → katgpt-rs

**Status:** COMPLETE — all gates PASS. Plan file: `katgpt-rs/.plans/415_within_class_effective_rank.md`.

**Correction from the original sketch below:** `data_probe/geometry.rs` is gated `sink_aware_attn` (NOT default-on — the original sketch was wrong). The primitive inherits that gate, ships opt-in alongside its sibling `effective_rank`, and requires no Cargo.toml change. No promotion is attempted (the parent Plan 287 G2/G3 gate that would promote `sink_aware_attn` is still pending).

**Original sketch (preserved for reference):**

```markdown
# Plan NNN: Within-Class Effective Rank — Class-Conditioned Collapse Diagnostic

**Date:** (TBD)
**Research:** katgpt-rs/.research/394
**Source paper:** arXiv:2412.19419 §5.3.1 + S.1.2
**Target:** crates/katgpt-core/src/data_probe/geometry.rs (extend) + feature `data_probe` (already default-on)
**Status:** Active — Phase 1

## Goal
Ship `within_class_effective_rank(states, dim, class_labels) -> f32` — the
entropy-based effective rank of the within-class residual covariance. Fuses
the shipped `effective_rank` (class-agnostic) with the shipped
`within_class_adjacency` machinery (currently used for Dirichlet-energy
scoring). Modelless collapse diagnostic for any class-labeled latent state.

## Phase 1 — Primitive (CORE)
- [ ] T1.1 Add `within_class_effective_rank` to `data_probe/geometry.rs`.
      Reuse the Jacobi eigensolver; replace global-mean centering with
      class-mean centering. ~40 lines.
- [ ] T1.2 Unit tests: (a) identical-class degenerate case returns ~0;
      (b) two well-separated isotropic classes returns ~d; (c) two collapsed
      classes (each rank-1) returns ~1; (d) matches the shipped
      `effective_rank` when all labels are identical (degenerate single-class).
- [ ] T1.3 Add `WithinClassGeometryReport { within_class_erank, n_classes,
      global_erank_for_contrast }` and a `within_class_geometry_report`
      convenience function.

## Phase 2 — GOAT gate
- [ ] T2.1 G1 (correctness): synthetic two-class case, verify r_WC ∈ [1, d−1]
      and monotone in within-class variance.
- [ ] T2.2 G2 (non-redundancy vs shipped `effective_rank`): construct a case
      where global `effective_rank` is high but `within_class_effective_rank`
      is low (between-class variance dominates, within-class collapsed) —
      prove the two metrics disagree.
- [ ] T2.3 G3 (latency): sub-µs per call on dim=64, n=256, C=4 (reuse the
      existing Jacobi hot path).
- [ ] T2.4 G4 (alloc-free hot path): `within_class_effective_rank_into` with
      caller-supplied scratch, mirroring the existing `effective_rank` pattern.
```

**GOAT gate note (per AGENTS.md):** this is a UQ-adjacent diagnostic (it measures representation health, not a probability distribution), so the "Report the Floor" conformal-naive baseline (Plan 340) does not apply. The G1–G4 gates above are sufficient.

---

## 5. What is NOT distilled (and why)

- **GCN / GraphSAGE / GATv2 architectures** — message-passing layers with learnable `ψ`, `ϕ`, attention `a`. Requires backprop through base weights. → riir-train.
- **Encoder-decoder training framework** (§3) — defines the training loop, loss, ground-truth function. Training-only.
- **Hyperparameter tuning sweep** (§5.4.1, Tables 6–9) — empirical architecture search on real GNNs. Training-only.
- **RevGNN comparison** (§5.4.1 Q3, Table 10) — reversible GNN training. Training-only.
- **LLM-based explainability** (GraphXAIN §4.4.5) — natural-language narrative generation. LLM-as-mechanism (the value IS the generated text); no modelless analog per the R368 rule.
- **Spectral GNNs** (§4.1 last paragraph) — eigenvector-of-Laplacian convolutions. Training-only; our DEC substrate already covers the *structural* spectral side modellessly (Stokes/Hodge).
- **Homophily / label scarcity mitigation experiments** — architecture-tuning results on real datasets. Training-only.

---

## TL;DR

A GNN survey. Bulk is training-only (→ riir-train) or already pervasive (`sigmoid(z_i^T z_j)` link decoder = our CLR verifier). **One GOAT survives: Within-Class Effective Rank** — a novel fusion of `data_probe::effective_rank` (class-agnostic, shipped) × `latent_functor::quality_gate::within_class_adjacency` (class-conditioning machinery, shipped) that has never been combined. Modelless, ~40 lines, detects within-class latent collapse that each shipped half misses individually. Strengthens the committed-personality / HLA / latent_functor pillars but does not create a new one — GOAT, not Super-GOAT, no private guide. MSBE and NCMq0.05 are Gains (model-aware / offline); structural analogs already ship via DEC. Modularity community detection is a Gain, not shipped, orthogonal.
