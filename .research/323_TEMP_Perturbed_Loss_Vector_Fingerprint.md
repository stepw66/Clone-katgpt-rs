# Research 323: TEMP — Token-Efficient Model Perturbation → Perturbed-Loss-Vector Fingerprint

> **Source:** [Reasoning Quality Emerges Early: Data Curation for Reasoning Models](https://arxiv.org/abs/2606.26797) — Hongyi Henry Jin, Wenhan Yang, Meysam Ghaffari, Carlos Morato, Baharan Mirzasoleiman (UCLA / Optum AI), ICML 2026, PMLR 306.
> **Date:** 2026-06-29
> **Status:** Done — **Super-GOAT** verdict (modelless reframe only; the paper's SFT pipeline → `riir-train`).
> **Classification:** Public
> **Related Research:** 317 (Attractor Dynamics — flat/sharp minima single-checkpoint Gibbs, the closest cousin), 318 (Sleep-Time Anticipator — primary application target), 322 (Conformal UQ — single-checkpoint predictive interval, the UQ floor companion), 276 (MicroRecurrentBeliefState — the "trained basins needed" caveat), 125 (Weight Norm = Kolmogorov — flat minima downstream of MDL), 169 (Oscillatory SSM — flat minima as convergence heuristic).
> **Related Plans:** 341 (open primitive plan, this session), 334 (Sleep-Time Anticipator), 308 (KARC), 281 (BoMSampler), 311 (Alien Sampler), 276 (AttractorKernel, null result).
> **Cross-ref (riir-neuron-db):** Research 010 — `Perturbed_Loss_Vector_Sleep_Consolidation_Guide.md` (the **private selling-point guide** — Super-GOAT mandatory output). Plan 005 — `temp_consolidation_diversity_selector.md` (private shard integration plan).

---

## TL;DR

The paper's headline contribution is **SFT data curation for reasoning models** — a training recipe. That part → `riir-train` (one-line note, no files in this session). But the paper's *latent insight* is a **modelless primitive with no shipped prior art across the 5-repo corpus**: the perturbed-loss-vector Lipschitz bound (Theorem 3.1) reduces "how different are the gradients two experiences would induce during the next weight-mutation cycle?" to "how different are their loss vectors across a few directionally-extrapolated checkpoints?" — and the directional extrapolation is along `v = θ_f − θ_0`, which in our modelless regime is exactly the vector between two committed snapshots.

**Distilled for katgpt-rs (modelless, inference-time):**
> Given two frozen checkpoints `S_0` (current) and `S_1` (next — a personality variant, a sleep-time consolidation result, a divergence branch), generate K directionally-extrapolated checkpoints `θ_j = S_0 + λ·(1+ξ_j)·(S_1 − S_0)` (deterministic noise schedule, BLAKE3-reproducible). For any candidate experience `z` (a wake event, a query, a replay tick), compute its **perturbed-loss vector** `L_z = [L_z(θ_1), ..., L_z(θ_n)]` over a *short prefix* of `z` (first N tokens / N ticks / N HLA updates). Then:
> **Theorem 3.1 (modelless reframe):** if `|L_{z1} − L_{z2}|_∞ ≤ δ` across the K checkpoints, then the gradient component along `v` differs by at most `(2δ/λ + G)(1/√2 + τ) + C_H·ε` during the next weight-mutation cycle. This bounds gradient diversity **without computing gradients**.

The modelless primitive is the **perturbed-loss-vector diversity fingerprint**: a deterministic signature of "how much unique gradient signal would this experience contribute to the next freeze/thaw cycle", computed from K short-prefix forward passes at extrapolated snapshots. No backprop, no training.

---

## 1. Paper Core Findings

### 1.1 The headline training contribution (→ riir-train)

The paper's *training pipeline* is **TEMP** (Token-Efficient Model Perturbation):
1. **Difficulty filtering (§3.1):** the first 100 tokens of each reasoning trace, evaluated at a *randomly perturbed* checkpoint `θ_rnd = θ_0 + λ·ξ` with `ξ ~ N(0, I)` (scaled so the perturbed loss is 2–3× the pretrained loss), are a reliable difficulty indicator. Easy examples reside in flat minima (low loss spike under noise); difficult examples reside in sharp minima (high loss spike). Cluster per-source losses into 2 groups, keep the high-loss (difficult) group.
2. **Diversity sampling (§3.2):** generate K *directionally-extrapolated* checkpoints `θ_j = θ_{j-1} + λ·(1+ξ_j)⊙v` where `v = θ_f − θ_0`. Compute each remaining example's **perturbed-loss vector** `L_z = [L_z(θ_1), ..., L_z(θ_n)]` over the first 1k reasoning tokens. Cluster the loss vectors; sample K-per-cluster by *brittle difficulty* (`L_z(θ_n) − L_z(θ_1)`).
3. **Result:** outperforms LLM-judge baselines by up to 1.7% on Qwen2.5-7B / Llama3.1-8B (M23K medical, OpenThoughts-Math) while being **91% more token-efficient** (only short-prefix loss evaluation).

**That pipeline is a training recipe** — uses Qwen2.5-7B as the base, applies real gradient descent in the SFT step. **→ riir-train** (not distilled here). The paper's repo: https://bigml-cs-ucla.github.io/TEMP-project-page/.

### 1.2 The latent insight that survives the modelless filter

Two findings of the paper are *not* training-specific:

**Finding A — Difficulty via perturbation (§3.1).** Easy examples reside in flat minima and stay stable under weight noise; difficult examples occupy sharp minima and spike. The loss of the first 100 response tokens ("problem-understanding phase") at a noisy checkpoint is the cleanest signal — the full-trace loss is dominated by token-level stochasticity. **Modelless reframe:** for any latent state kernel (HLA, functor, shard style_weights), a Gaussian-perturbed snapshot reveals which inputs the kernel has *memorized* (low loss variance under noise) vs which inputs the kernel is *genuinely reasoning about* (high loss spike under noise).

**Finding B — The Lipschitz bound (Theorem 3.1, the load-bearing math).** Under local-SFT assumptions (curvature bounded by `C_H`, gradient norm by `G`, parameter updates within `ε` of init):

> If `|L_{z1}(θ_j) − L_{z2}(θ_j)| ≤ δ` for all perturbed checkpoints `j ∈ {1, ..., n}`,
> then `|⟨∇L_{z1}(θ) − ∇L_{z2}(θ), v⟩| ≤ (2δ/λ + G)(1/√2 + τ) + C_H·ε`.

The proof (Appendix A) decomposes `v` into the checkpoint-step direction plus orthogonal slack, applies the quadratic-loss assumption to cancel symmetric terms, and transfers the bound from the midpoint to any SFT-reachable `θ` via the curvature bound. **Crucially:** the bound is on the *gradient component along `v`* — i.e. the part of the gradient that actually moves during the next fine-tuning cycle. Perpendicular components are unconstrained but irrelevant (they don't move during local SFT).

**Modelless reframe:** in our regime, `v = S_1 − S_0` is the divergence direction between two committed snapshots. The next "weight-mutation cycle" is the next freeze/thaw swap or the next consolidation tick. **Theorem 3.1 says: if two candidate experiences have similar perturbed-loss vectors at extrapolated snapshots, then including both in the next swap/consolidation produces redundant gradient signal along `v`.** Diversity selection by perturbed-loss-vector spread = diversity selection by future-gradient diversity — without ever running gradients.

### 1.3 Why "short prefix" works (the token-efficiency win)

Paper §3.2 / Fig. 6: the loss of the first 1k reasoning tokens correlates ~0.9+ with the loss of the full reasoning trace (up to 91k tokens), across perturbation strengths. The reason: fine-tuning is local — it re-weights features, doesn't create new ones — so the *initial* reasoning steps (which set up the problem representation) carry the diversity signal. The full trace adds token-level stochasticity that *dilutes* the signal.

**Modelless reframe:** for our latent state kernels, the *initial* ticks of a replay (the first few HLA updates, the first few functor applications, the first few KARC forecast steps) carry the gradient-diversity signal. The full replay adds noise. **Token-efficient fingerprinting is not just a perf optimization — it's a signal-quality optimization.**

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase — both layers hit)

| Paper term | Codebase equivalent (grep target) | Closest shipped cousin |
|---|---|---|
| Perturbed checkpoint `θ_rnd` | Noisy snapshot, perturbed shard, `S_0 + λ·ξ` | None (FaithfulnessProbe perturbs *memory*, not weights; CCE `PerturbedPlayer` perturbs *payoff*, not weights) |
| Directional extrapolation `θ_j = θ_0 + λ(1+ξ_j)⊙v` | Extrapolated snapshot, `S_0 + λ·(S_1 − S_0)` | None (mmorpg-sync `SnapshotInterpolator::extrapolate` is rendering interpolation, not weight-space) |
| Fine-tuning direction `v = θ_f − θ_0` | Snapshot divergence direction, `S_1 − S_0` | Plan 336 Committed Personality tracks `pi_max − pi` (personality divergence); not used as extrapolation axis |
| Perturbed-loss vector `L_z` | Loss trajectory, loss fingerprint, multi-checkpoint loss | None (PEIRA Plan 046 has a single "loss trajectory" plot, not a per-example multi-checkpoint vector) |
| Gradient diversity | Diversity, alien, coherence×availability | Alien Sampler Plan 311 (single-checkpoint coherence); BoMSampler Plan 281 (single-checkpoint K-trajectory); **none use multi-checkpoint gradient proxies** |
| Wide basin / flat minimum | Flat, stable, robust | Research 317 (single-checkpoint `1/E²` Gibbs); CUCG `spectral_flatness < 0.3` freeze gate (single-checkpoint) |
| Sharp minimum / brittleness | Brittle, sensitive, unstable | None as a multi-checkpoint fingerprint |
| First-100-token loss | Short-prefix loss, initial-tick loss | None — `ac_prefix::conditional_logprob` (Plan 313) computes single-prefix logprobs but not as a diversity fingerprint |
| Lipschitz bound on gradients | (none — the missing primitive) | **This is the gap.** |

**Both-layer grep results:** paper vocabulary (`perturbed loss vector | gradient diversity | token-efficient | first-1k | problem understanding | fine-tuning direction`) → ZERO hits across all 5 repos in both `.research/`+`.plans/` AND `src/`+`crates/`. Codebase vocabulary (`loss_trajectory | loss_fingerprint | wide_basin | flat_minim | sharp_minim | snapshot_vector | delta_snapshot | directional_perturb`) → ZERO hits in code (only single-checkpoint flatness gates and single-checkpoint Gibbs weighting, all via Research 317).

### 2.2 The actual novel primitive (≈40 lines)

```rust
//! Perturbed-loss-vector diversity fingerprint (modelless).
//!
//! Paper: TEMP (arxiv 2606.26797), Theorem 3.1.
//! Modelless reframe: v = S_1 - S_0 is the divergence between two committed
//! snapshots; theta_j = S_0 + lambda * (1 + xi_j) * v are directionally
//! extrapolated snapshots; L_z(theta_j) is the short-prefix loss of candidate
//! experience z at theta_j. Theorem 3.1: similar loss vectors => similar
//! gradient-along-v during the next freeze/thaw cycle. Diversity selection by
//! loss-vector spread = diversity selection by future-gradient diversity.

/// Directionally-extrapolated snapshot schedule (deterministic, BLAKE3-reproducible).
/// Produces K snapshots `theta_j = S_0 + lambda_j * v` where `lambda_j` is a
/// fixed geometric schedule and `xi_j` is a fixed RNG seed (deterministic noise).
/// Paper Eq. 5: `theta_j = theta_{j-1} + lambda * (1 + xi_j) ⊙ v`.
pub fn extrapolated_snapshot_schedule(
    s0: &[f32],         // current snapshot (e.g. shard style_weights)
    s1: &[f32],         // next snapshot (e.g. candidate personality variant)
    k: usize,           // number of extrapolated checkpoints
    lambda_schedule: &[f32],  // len == k, deterministic
    noise_seeds: &[u64],      // len == k, deterministic (BLAKE3-reproducible)
    out: &mut [Vec<f32>],     // len == k, caller-allocated
);

/// Compute the short-prefix loss of candidate `z` at extrapolated snapshot `theta_j`.
/// `z_prefix` is the first N tokens / ticks / HLA-updates of the candidate.
/// `loss_fn` is the kernel's per-step negative-log-probability (e.g.
/// `ac_prefix::conditional_logprob`, or HLA's belief-update surprise).
/// Returns `L_z(theta_j) = sum_{t <= N} -log p(z_prefix[t] | z_prefix[<t], theta_j)`.
pub fn short_prefix_loss<L: LossKernel>(
    theta_j: &[f32],
    z_prefix: &[f32],   // first N steps of candidate
    loss_fn: &L,
) -> f32;

/// Lipschitz bound from Theorem 3.1 (the modelless gradient-diversity proxy).
/// Given `delta = ||L_{z1} - L_{z2}||_inf` across K checkpoints, returns the
/// upper bound on `|<grad L_{z1} - grad L_{z2}, v>|` during the next
/// weight-mutation cycle along `v`.
/// Bound: `(2*delta/lambda + G) * (1/sqrt(2) + tau) + C_H * epsilon`.
pub fn lipschitz_gradient_bound(
    delta: f32, lambda: f32, g: f32, tau: f32, c_h: f32, epsilon: f32,
) -> f32;

/// Diversity selection: from a candidate set, pick the K-subset whose
/// perturbed-loss vectors have maximal spread (greedy max-min or k-medoids on
/// the loss vectors). This is the modelless analog of TEMP §3.2 Algorithm 1.
pub fn select_diverse_subset(
    loss_vectors: &[&[f32]],  // one per candidate
    k: usize,                 // target subset size
    scratch: &mut [usize],    // caller-allocated
) -> Vec<usize>;              // selected indices
```

That is the entire primitive. Everything else is composition of existing infrastructure (committed shards as `S_0`/`S_1`, `ac_prefix` / HLA surprise as `loss_fn`, BLAKE3 as the deterministic-noise source).

### 2.3 Why "modelless" survives the §3.5 unblock protocol

| Sub-component | Path 1 (freeze/thaw)? | Path 2 (raw/lora)? | Path 3 (latent projection)? | Verdict |
|---|---|---|---|---|
| `S_0`, `S_1` snapshot pair | YES — both are committed shards (`NeuronShard`, `ArchetypeBlendShard`, `KarcShard`, `SleepAnticipationShard`) | n/a | n/a | **Modelless** ✅ |
| Extrapolated `theta_j` | YES — deterministic linear combination of two committed shards, BLAKE3-reproducible | YES — `theta_j` is a deterministically-constructed snapshot | YES — linear projection of `(S_0, S_1)` | **Modelless** ✅ |
| Short-prefix loss `L_z(theta_j)` | YES — `ac_prefix::conditional_logprob` already computes single-prefix NLL (Plan 313); HLA surprise is already a per-tick NLL analog | n/a | YES — `L_z` is a dot-product projection of `z_prefix` onto the snapshot's logit directions | **Modelless** ✅ |
| Lipschitz bound | Pure arithmetic inequality — no learning | n/a | n/a | **Modelless** ✅ |
| Diversity selection (k-medoids on loss vectors) | Pure algorithmic | n/a | n/a | **Modelless** ✅ |

**No riir-train deferral needed.** The training-only parts of the paper (real SFT, real gradient descent to validate the bound) → riir-train. The primitive itself is fully modelless.

### 2.4 Fusion (the Super-GOAT move)

| Fusion partner | What it ships | What TEMP adds | Fusion product |
|---|---|---|---|
| **R318 / Plan 334 Sleep-Time Anticipator** | Offline query anticipation; predictability-gated sleep-time budget allocation | Diversity selection for the sleep-time *consolidation queue* — pick the maximally-spread experiences to consolidate | "Sleep-time consolidation that picks the maximally-diverse experiences for the next freeze/thaw cycle using a Lipschitz-bound fingerprint that costs 1/N of full replay" |
| **riir-neuron-db `ConsolidationPipeline`** | Raven/δ-Mem wake→sleep→consolidate cycle; **currently averages ALL wake events equally** | Diversity-preserving subset selection on `wake_events` before `sleep()` — keep the K events with maximal loss-vector spread | "Consolidation that doesn't drown diverse signal in averaging — picks the diverse minority before the mean collapses it" |
| **R317 Attractor Dynamics** | Single-checkpoint Gibbs `1/E²` basin weighting | Multi-checkpoint Lipschitz fingerprint — moves from "rank by current energy" to "rank by future-gradient diversity along the snapshot-divergence direction" | "From single-snapshot energy ranking to multi-snapshot diversity fingerprinting — strict generalization of R317" |
| **Plan 336 Committed Personality** | Tracks personality divergence `pi_max − pi`; commits personality snapshots | The divergence vector `pi_max − pi` IS the `v` for TEMP's extrapolation — extrapolate along personality-divergence to fingerprint which experiences would push personality furthest | "Personality-divergence-aware experience selection — pick experiences that maximally differentiate NPCs at the next snapshot swap" |
| **Plan 308 KARC** | Closed-form delay-basis ridge forecaster of next latent state | Loss-vector fingerprint as a "which future states would this experience make more predictable" signal | "KARC-guided experience selection — experiences ranked by their future-state forecasting diversity" |
| **Plan 311 Alien Sampler** | Coherence × availability frontier ranking (single-checkpoint) | Availability = "produces a new gradient direction" — TEMP gives a modelless, multi-checkpoint estimator for this | "Alien Sampler's availability axis upgraded from heuristic to Lipschitz-bound-grounded" |
| **Plan 281 BoMSampler** | K-trajectory single-pass diverse sampling (single-checkpoint) | Multi-checkpoint fingerprinting of the K hypotheses — which hypotheses would produce diverse gradients during the next update | "BoM hypotheses ranked by multi-snapshot gradient-diversity fingerprint" |
| **Plan 284 CLR Test-Time Scaling** | Claim-Level Reliability + self-adaptive test-time compute allocation | Token-efficient difficulty estimation (TEMP §3.1) — first 100 tokens at noisy checkpoint = difficulty | "CLR difficulty gate at 1/100th the compute — early-exit easy claims after 100 tokens at a perturbed snapshot" |
| **Plan 276 AttractorKernel** (G2.1 null) | Random-init attractors don't have flat basins | TEMP's `S_0 = shard.style_weights` (Fusion B from R317) + `S_1 = shard.style_weights + delta` gives a real `v` to extrapolate along — the missing piece for R317's Fusion B | "R317's shard-loaded attractor + TEMP's directional extrapolation = trained-basin-equivalent diversity fingerprinting, fully modelless" |
| **katgpt-rs/crates/katgpt-core/src/dec/** (DEC / Stokes) | `exterior_derivative d`, `codifferential δ`, `hodge_decompose` | The "fine-tuning direction" `v` is a vector field on the latent manifold; the Lipschitz bound is `|⟨d(loss), v⟩| ≤ bound` — a directional-derivative bound on the loss cochain | "Stokes-theoretic reframing: loss-vector spread bounds the directional exterior derivative of loss along `v`. Curse-of-dimensionality caveat: only meaningful for d ≤ 3 (HLA regions, KG embeddings) — NOT high-dim shards." |

### 2.5 Latent-space reframing (mandatory per workflow §1.3)

Re-cast the perturbed-loss-vector fingerprint as a latent-to-latent op on each Super-GOAT factory module:

| Substrate | Reframing | Status |
|---|---|---|
| **HLA** (`katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/crates/riir-engine/src/hla/`) | Two committed HLA snapshots (pre-event, post-event) form `v = hla_1 − hla_0`. Extrapolate along `v` at K checkpoints. For any candidate replay tick (first N HLA updates of a recent event), compute `L_z(theta_j)` = HLA surprise at extrapolated snapshot. Theorem 3.1: similar loss vectors ⇒ similar HLA-update gradients. | Direct ship — HLA surprise kernel already exists; just multi-snapshot. |
| **`latent_functor/`** (`riir-ai/crates/riir-engine/src/latent_functor/`) | Two functor states = two checkpoints. The functor-update direction (which `reestimation.rs` tracks as "coherence drift") IS `v`. Extrapolate; for any candidate (source, target) relation, compute loss vector. | Direct ship — `reestimation.rs` already tracks drift direction. |
| **`cgsp_runtime/`** (`riir-ai/crates/riir-engine/src/cgsp_runtime/`) | Two curiosity snapshots = two checkpoints. Curiosity direction (which NPC is exploring) IS `v`. Extrapolate; for any candidate exploration thread, compute loss vector. | Direct ship — gives CGSP a principled exploration-diversity signal. |
| **LatCal fixed-point commitment** (`riir-chain/src/encoding/latcal*.rs`) | `theta_j` are deterministic linear ops on committed shards → LatCal-committable as fixed-point blocks. `L_z(theta_j)` is a deterministic scalar per snapshot → crosses sync as raw value. | **The full fingerprint is sync-committable** — any node loading `S_0, S_1` derives the same `L_z` vector → diversity ranking is quorum-reproducible. |
| **`NeuronShard` / `MerkleFrozenEnvelope` / Raven consolidation** (`riir-neuron-db/src/`) | **The primary substrate.** `S_0, S_1` = pretraining + fine-tuned shards. `v = S_1 − S_0` IS the consolidation direction. Raven's wake-event queue IS the candidate training data. **Currently averages ALL wake events — TEMP gives a modelless diversity selector for the queue.** | Direct ship — see riir-neuron-db Research 010 / Plan 005. |
| **DEC Stokes operators** (`katgpt-rs/crates/katgpt-core/src/dec/`) | `v` is a vector field on the latent manifold. Lipschitz bound `|⟨∇ΔL, v⟩| ≤ bound` is a directional-derivative bound on the loss cochain. `|⟨d(loss_cochain), v⟩|` is the directional exterior derivative. | Geometry-grounded interpretation; curse-of-dim caveat for high-d shards. |

---

## 3. Verdict

### Super-GOAT

**One-line reasoning:** The paper's *latent insight* (Theorem 3.1's perturbed-loss-vector Lipschitz bound on gradient differences) reduces "gradient diversity during the next weight-mutation cycle" to "loss-vector spread across K directionally-extrapolated snapshots" — a deterministic, modelless, BLAKE3-reproducible primitive with **zero shipped prior art** across all 5 repos (verified by both-layer grep with vocabulary translation). The extrapolation axis `v = S_1 − S_0` is exactly the divergence direction between two committed shards — a native freeze/thaw concept. The primitive multiplies ≥6 pillars (Sleep-Time Anticipator, Raven consolidation, Alien Sampler, KARC, CLR, Committed Personality, BoMSampler, latent_functor, DEC), with a clear product selling point ("consolidation that picks the maximally-diverse experiences for the next freeze/thaw cycle using a Lipschitz-bound fingerprint that costs 1/N of full replay — modelless, quorum-reproducible").

### Novelty gate (§1.5)

1. **No prior art?** Three-layer check: (notes) paper-vocabulary grep (`perturbed loss vector | gradient diversity | token-efficient | first-1k | problem understanding | fine-tuning direction`) → ZERO hits across all 5 repos. (code) codebase-vocabulary grep (`loss_vec | gradient_diversity | gradient_proxy | extrapolat.*snapshot | directional_extrapolat | perturb.*checkpoint | short_prefix_loss`) → ZERO hits (only `extrapolate` in mmorpg-sync rendering, irrelevant). (vocabulary translation) "perturbed checkpoint" ↔ "noisy snapshot / shard + λ·ξ", "fine-tuning direction" ↔ "snapshot divergence `S_1 − S_0`", "perturbed loss vector" ↔ "multi-checkpoint loss trajectory / loss fingerprint". The closest cousin is **Research 317** (single-checkpoint Gibbs `1/E²` basin weighting) — confirmed by `read_file` of its §2.2: it lists `β ∝ E⁻²` as the paper-specific contribution and treats multi-checkpoint loss vectors as future work ("Latent Space Navigation: gradient descent on energy" — paper §1.7, R317 §2.2 row "Latent gradient descent on energy ❌"). **TEMP's multi-checkpoint perturbed-loss vector + Lipschitz bound is the missing primitive R317 explicitly flagged.** ✅
2. **New class of behavior?** Yes — "rank candidate experiences by gradient-diversity fingerprint WITHOUT running gradients" is a new capability class. We have latent-space diversity (Alien Sampler), score-space diversity (BoMSampler), cost-space diversity (gain/cost halting). We do NOT have *gradient-proxy-space diversity at extrapolated snapshots*. ✅
3. **Product selling point?** Yes — "Consolidation that picks the maximally-diverse experiences for the next freeze/thaw cycle using a Lipschitz-bound fingerprint that costs 1/N of full replay — modelless, quorum-reproducible, no gradient descent." Complete sentence, not an optimization. ✅
4. **Force multiplier?** Yes — multiplies Sleep-Time Anticipator + Raven/δ-Mem + Alien Sampler + KARC + CLR + Committed Personality + BoMSampler + latent_functor + DEC (≥6 pillars). ✅

**All 4 YES → Super-GOAT.**

### Tiers (high → low)

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | **← this paper (modelless reframe only)**. Open primitive → katgpt-rs Plan 341. **Architectural guide → riir-neuron-db/.research/010** (the selling point lives at the shard + freeze/thaw + consolidation layer — two committed snapshots = the fine-tuning direction; Raven's wake-event queue = the candidate training data). Plan → katgpt-rs Plan 341 (open) + riir-neuron-db Plan 005 (private shard integration). |
| → riir-train | The paper's SFT data-curation recipe itself (Qwen2.5-7B fine-tuning, LLM-judge difficulty labeling, the actual gradient-descent validation of the bound) | One-line note: "→ riir-train: TEMP SFT data-curation recipe (arxiv 2606.26797 §3–4). Out of scope for this session." |

---

## 4. Mandatory outputs (created this session)

1. **Open primitive plan** → `katgpt-rs/.plans/341_temp_perturbed_loss_vector_primitive.md` — generic math primitive (extrapolated snapshot schedule, short-prefix loss, Lipschitz bound, diversity selection). No game/chain/shard semantics. The adoption hook.
2. **Private selling-point guide** → `riir-neuron-db/.research/010_Perturbed_Loss_Vector_Sleep_Consolidation_Guide.md` — the consolidation-queue diversity selection selling point. Connection map to Raven/δ-Mem, SleepAnticipationShard, KarcShard, ArchetypeBlendShard. Latent-vs-raw boundary. Validation protocol (G1–G6).
3. **Private shard integration plan** → `riir-neuron-db/.plans/005_temp_consolidation_diversity_selector.md` — wires the open primitive into `ConsolidationPipeline::sleep()` as a diversity-preserving pre-filter on `wake_events`.

---

## 5. Latent vs raw boundary (sync semantics)

| Signal | Domain | Synced? | Notes |
|---|---|---|---|
| Snapshot pair `(S_0, S_1)` | Frozen latent blob | YES (committed) | BLAKE3-hashed, Merkle-wrapped; both already ship via existing freeze/thaw |
| Extrapolated snapshots `theta_j` | Latent (semantic) | NO (local) | Deterministic linear combination; any node with `(S_0, S_1)` re-derives bit-identically |
| Short-prefix loss `L_z(theta_j)` per candidate | Latent (semantic) | NO | Local to the consolidating node; computed during sleep-time |
| **Diversity-selected subset indices** | **Raw scalar** (semantic metadata) | **YES (synced)** | The K selected wake-event indices cross sync as raw values; the events themselves are already in the shard's wake log |
| Lipschitz bound value (audit) | Raw scalar | YES (committed) | Audit-trail scalar, committed alongside the freeze envelope |

**Bridge function (zero-allocation, gateable, no sync dependency):**
```rust
/// Map a diversity-selected wake-event subset to the synced representation.
/// Only the selected indices + their aggregate Lipschitz-bound audit scalar
/// cross sync; the loss vectors themselves stay local.
fn diverse_subset_to_synced_scalars(
    selected_indices: &[usize],   // K indices into wake_events
    aggregate_bound: f32,         // audit: max pairwise bound across selected
) -> SyncedSubsetReport {
    SyncedSubsetReport {
        count: selected_indices.len() as u32,
        bound: aggregate_bound.to_bits(),  // raw scalar, deterministic
    }
}
```

**Two-brain compatibility:** the fingerprint operates on the **think brain** (latent state, NOT synced). It does NOT touch the info brain (raw `MapPos`, synced ground truth). Diversity selection of wake events is purely about which semantic experiences to consolidate — no impact on physical-domain replay determinism.

**Anti-cheat:** the extrapolated `theta_j` are deterministic functions of `(S_0, S_1)` and a fixed seed schedule. Two nodes processing the same wake-event queue with the same snapshot pair produce bit-identical `L_z` vectors and bit-identical selected subsets. No model-in-the-loop divergence at the sync boundary.

---

## 6. Honest risks

### 6.1 Why this might fail

- **The bound is for SFT, not for our consolidation.** Theorem 3.1 assumes local quadratic loss with bounded curvature `C_H` and bounded gradient norm `G`. Our Raven/δ-Mem consolidation is a single-pass average (`CompressedMemory::weight_delta = mean(embeddings)`), not gradient descent — the bound's assumptions may not hold. **G1 must verify the bound empirically on real shard consolidation, not assume it transfers.**
- **`v = S_1 − S_0` may not be a meaningful "fine-tuning direction".** In the paper, `v = θ_f − θ_0` is the actual SFT trajectory direction. In our regime, `S_0` is the current shard and `S_1` is the *next* shard — but the *next* shard doesn't exist yet (we're trying to *decide what to consolidate into it*). **Resolution:** use `S_0` and a *target* snapshot (e.g. an `ArchetypeBlendShard` target, or a personality-direction vector from Plan 336) as the divergence axis. The `v` is "the direction we want consolidation to move in", not "the direction consolidation did move in". This is a *proactive* use of the bound (plan the consolidation), vs the paper's *retroactive* use (analyze past SFT).
- **Short-prefix may not carry the signal for non-LLM kernels.** Paper shows first-100-token loss correlates with full-trace loss for LLM CoT. For HLA updates, functor applications, KARC forecasts — the prefix length that carries the signal is an empirical question. **G2 must sweep prefix lengths per kernel.**
- **K checkpoints may be too expensive.** Each checkpoint is a full forward pass. If `K = 8` and prefix `N = 100`, that's 800 forward steps per candidate — for a 256-event wake queue, 205k forward steps. May exceed the sleep-time budget. **G3 must bench and tune K, N against the plasma/warm budget.**

### 6.2 Why it is still worth committing to

- The primitive is **fully modelless** — no training, no gradients, no riir-train deferral.
- The primitive is **strictly novel** — R317 explicitly flagged this exact gap as future work.
- The primitive is **deterministic and quorum-reproducible** — no model-in-the-loop divergence at sync.
- The primitive **multiplies 6+ pillars** — Sleep-Time, Raven, Alien, KARC, CLR, Committed Personality.
- Even if the bound is loose (the Lipschitz constants are hard to estimate tightly), **the relative ordering** of candidates by loss-vector spread may still be a useful diversity signal — diversity selection only needs ranking, not absolute bounds.

### 6.3 What we will NOT do

- **Will NOT validate the bound against real SFT.** That requires gradient descent → riir-train. We validate the *diversity-selection utility* (does picking spread-out loss vectors produce better consolidation?) not the *bound tightness*.
- **Will NOT compute the Lipschitz constants `C_H, G` from first principles.** Treat them as tuning parameters; the bound's *form* (linear in `δ/λ`) is what matters for ranking, not its absolute value.
- **Will NOT ship this for high-dimensional shards.** The DEC cross-reference (§2.5 row "DEC") flags the curse-of-dimensionality caveat: this works for HLA regions (d ≤ 8), KG embeddings (d ≤ 64), game maps (d = 2) — NOT for full `style_weights[64]` per-channel diversity (which is already covered by `spectral_flatness`).

---

## 7. Connection map

```
                     TEMP Perturbed-Loss-Vector Fingerprint (this note)
                                       │
                  ┌────────────────────┼─────────────────────┐
                  ▼                    ▼                     ▼
        [R317 Attractor]      [R318 Sleep-Time]    [Plan 336 Committed]
        single-ckpt Gibbs     offline query         personality divergence
        ↓ upgrade             anticipation          pi_max − pi = v
        multi-ckpt            ↓ diversity           ↓ extrapolation axis
        Lipschitz             queue ranking         for fingerprint
        ↓                     ↓                     ↓
   [Plan 276 AttractorKernel null → §3.5 unblock via S_0=shard, S_1=shard+δ]
                                       │
                                       ▼
                          [riir-neuron-db ConsolidationPipeline]
                          Raven/δ-Mem wake→sleep→consolidate
                          currently averages ALL wake events
                          ↓ TEMP diversity pre-filter
                          diversity-preserving subset before sleep()
                                       │
                                       ▼
                       [Plan 308 KARC] [Plan 311 Alien] [Plan 281 BoM]
                       future-state    availability    K-hypothesis
                       forecast        = gradient       single-pass
                       diversity       direction        diversity
                                       │
                                       ▼
                            [Plan 284 CLR Test-Time Scaling]
                            TEMP §3.1: first-100-token loss at noisy ckpt
                            = difficulty → early-exit easy claims
                                       │
                                       ▼
                       [DEC Stokes operators]
                       v = vector field on latent manifold
                       |<d(loss_cochain), v>| = directional exterior deriv.
                       bound = Stokes-theoretic Lipschitz on cochains
```

---

## TL;DR

The paper's SFT data-curation recipe is training → `riir-train` (one-line note, no files). The paper's *latent insight* is a **modelless primitive with zero prior art**: Theorem 3.1's perturbed-loss-vector Lipschitz bound reduces "gradient diversity during the next weight-mutation cycle" to "loss-vector spread across K directionally-extrapolated snapshots", where the extrapolation axis `v = S_1 − S_0` is exactly the divergence between two committed shards. **Super-GOAT** (4/4 novelty gate: no prior art ✓, new capability class ✓, selling point ✓, force multiplier ✓ × 6+ pillars). Mandatory outputs created this session: open primitive plan `katgpt-rs/.plans/341_temp_perturbed_loss_vector_primitive.md`, private guide `riir-neuron-db/.research/010_Perturbed_Loss_Vector_Sleep_Consolidation_Guide.md`, private shard plan `riir-neuron-db/.plans/005_temp_consolidation_diversity_selector.md`. **Honest risk:** the bound is for SFT, our consolidation is single-pass averaging — G1 must verify the bound transfers; even if it doesn't, the *relative ranking* by loss-vector spread may still be a useful diversity signal. **Curse-of-dim caveat:** works for HLA (d ≤ 8), KG (d ≤ 64), maps (d = 2) — NOT for full `style_weights[64]` per-channel (already covered by `spectral_flatness`).
