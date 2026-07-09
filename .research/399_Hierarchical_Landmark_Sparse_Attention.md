# Research 399: HiLS-Attention — Hierarchical Landmark Sparse Attention

> **Source:** Hu et al., "Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling", arXiv:2607.02980, Jul 2026 (Tencent Hunyuan). Code: https://github.com/Tencent-Hunyuan/HiLS-Attention
> **Date:** 2026-07-09
> **Status:** Done
> **Related Research:** 071 (DashAttention — closest cousin, cited as ref [20] in the paper), 225 (MSA blockwise sparse), 176 (Vortex programmable sparse), 379 (HGA chunk-group routing — GOAT FAILED), 392 (SSMax attention dilution)
> **Related Plans:** 106 (DashAttention — ships `chunk_summary.rs`), 196 (VortexFlow), 044 (PFlash block-sparse prefill)
> **Classification:** Public

---

## TL;DR

HiLS-Attention is a chunk-wise sparse attention mechanism that **learns chunk
selection end-to-end** under the LM loss via two ideas: (1) a **LogSumExp
linearization** (Prop 3.1) that derives an entropy-calibrated chunk summary key
`k'_c` + bias `b'_c` from a landmark query, and (2) a **hierarchical softmax
factorization** that makes the chunk-mass surrogate participate in the forward
attention weights so gradients flow to the landmark representation. It matches
or beats full attention, extrapolates 512× (8K → 4M), and is 13.5×/15.7× faster
(prefill/decode) at 512K — **after 50B tokens of continued pretraining**.

**The headline value is the training recipe** (landmark token tuning + Q-Cal
low-rank adapter + HoPE positional encoding + 50B-token CPT). That part →
**riir-train**. The modelless-transferable kernel is the **entropy-calibrated
chunk summary** (Prop 3.1): a deterministic formula proving that the optimal
chunk summary score is `q^T k'_c / √d + b'_c` where `b'_c = -Σ p_j log p_j` is
the entropy of the intra-chunk attention distribution. We already ship the
attention-weighted key sum `k'_c = Σ p_j k_j` in `dash_attn/chunk_summary.rs`
(Plan 106); the **entropy bias `b'_c` is the genuine gap**.

**Distilled for katgpt-rs (modelless, inference-time):**
The entropy bias `b'_c` is a pure deterministic computation over the softmax
weights already produced by `summarize_chunk_into`. Adding it to the routing
score makes the chunk summary faithful to LogSumExp chunk mass (the paper proves
this is the first-order Taylor expansion) rather than just mean-logit. The bias
interpolates between the two regimes mean/max-pooling cannot simultaneously
satisfy: `b'_c → log S` when logits are uniform, `b'_c → 0` when one dominates.

---

## 1. Paper Core Findings

### 1.1 The problem with existing chunk-wise sparse attention

NSA, MoBA, InfLLM v2, DashAttention all use chunk summaries (mean-pooled keys
or learned keys) to score chunks for top-K selection. After selection, the
summary is **discarded** — it never participates in the forward pass, so the LM
loss cannot optimize it. Result: inaccurate chunk selection, especially exposed
on NIAH retrieval at the 345M scale (Fig 2: NSA/Dash/InfLLM v2 all fail
single-NIAH in-domain; only Naive-BSA and HiLS sustain 100%).

The mathematical root cause (Eq 5): the true chunk mass is a LogSumExp
`log Σ_j exp(s_{i,j})`, which behaves as `mean(s) + log S` when logits are
uniform but `max(s)` when one logit dominates. Mean-pooled summaries capture
only the first regime; max-pooling captures only the second. Neither is
universally correct.

### 1.2 Proposition 3.1 — LogSumExp linearization (the modelless kernel)

For a learned "landmark query" `q'_c`, define the intra-chunk attention
`p_j = softmax(q'_c^T k_j / √d)`. Then the first-order Taylor expansion of the
LogSumExp chunk mass around `q'_c` yields:

```
log Σ_j exp(q^T k_j / √d) ≈ q^T k'_c / √d + b'_c
```

where:
- `k'_c = Σ_j p_j k_j`  (attention-weighted key sum — **already shipped** in `chunk_summary.rs`)
- `b'_c = -Σ_j p_j log p_j`  (entropy of the intra-chunk distribution — **the gap**)

Both `k'_c` and `b'_c` are computed via one local SDPA pass `Attn(q'_c, K_c, K_c)`.
The cost is O(S) per chunk, O(N) for the full sequence. This is fully
deterministic given `q'_c`.

### 1.3 Hierarchical softmax factorization (the training-enabling mechanism)

The attention weight is factored as `w_{i,j} = (intra-chunk) × (inter-chunk)`:

```
w_{i,j} = [exp(s_{i,j}) / Z_{i,c(j)}] × [Ẑ_{i,c(j)} / Ẑ_i]
```

where `Ẑ_{i,c} = exp(ŝ_{i,c})` is the surrogate mass from Prop 3.1. Because
`Ẑ_{i,c}` participates in the forward weights, gradients from the LM loss flow
back to the landmark representation `q'_c`. This is what makes chunk selection
**end-to-end learnable** — the paper's headline contribution.

This factorization is pure algebra (no training); its *purpose* is to enable
gradient flow. For modelless inference it is dormant — without a learned `q'_c`,
the surrogate is constant and the factorization reduces to standard sparse
attention.

### 1.4 Landmark tokens + Q-Cal + HoPE (the training recipe)

- **Landmark token**: a special token appended to each chunk; its query vector
  is `q'_c`. Produced by the full Transformer stack (attention + MLP), so it
  has full capacity to encode chunk semantics. Essential for extrapolation
  (ablation: shared-`q_c` alternative loses extrapolation, Tab 6).
- **Q-Cal (low-rank query calibration)**: `Δq = W_up W_down h_i`, `q̂ = q + Δq`,
  rank r ≪ d_model. Decouples token-level query from chunk-level scoring. The
  paper admits "the underlying mechanism is not yet fully understood."
- **HoPE**: keep RoPE dimensions whose rotation period ≤ training length, NoPE
  for the rest. Avoids OOD positional rotations during chunk compression.

### 1.5 M-query adjacent packing kernel (inference engineering, modelless)

NSA's kernel requires GQA group G ≥ 16 for Tensor Core efficiency. HiLS packs
M adjacent query tokens (M × G ≥ 16), attending to the union of their selected
chunks. Validated empirically: adjacent queries retrieve ~93% overlapping chunks
(Fig 7). This is pure inference-time kernel design — GPU-specific, not directly
relevant to our CPU/SIMD + ANE stack, but the "adjacent queries retrieve
overlapping chunks" observation is useful for any batched sparse-attention
design.

### 1.6 Key empirical results

- 345M, 8K training: HiLS matches full attention PPL (4.94 vs 4.94 at 8K),
  extrapolates to 4M with 90%+ NIAH accuracy (512×).
- 7B Olmo3 CPT (50B tokens): matches/beats full attention on LongBench, exceeds
  YaRN-extended baseline.
- Speed: 13.5× faster prefill / 15.7× faster decode at 512K vs full attention.
- **"Compression enhances retrieval"**: HiLS beats full attention on variable
  tracking (VT) — attributed to noise cancellation in aggregated keys
  (`k_i = semantic(k_i) + noise(k_i)`; aggregation cancels noise, preserves
  signal).

---

## 2. Distillation

### 2.1 What we already ship (prior art — DashAttention, Research 071 / Plan 106)

`katgpt-rs/crates/katgpt-attn/src/dash_attn/chunk_summary.rs`:

```rust
// summarize_chunk_into: k̄_c = softmax(q̄ · K_chunk / √d) · K_chunk
// At zero-init head_cls: uniform softmax → mean pooling (backward compatible)
// After training: weighted attention to informative keys
```

This is **exactly HiLS's Eq 8** for `k'_c`. The DashAttention research note
(071) and Proposal 002 both flag that `head_cls` is "consumed not trained here
— zero-init degrades to mean pooling (backward-compatible), and weight mutation
is a freeze/thaw concern owned elsewhere." So the architecture for a learned
chunk summary query already exists; only the entropy bias is missing.

### 2.2 The genuine gap: the entropy bias `b'_c`

Our `routing.rs` scores chunks via `dot(query, chunk_summary_k)` without the
entropy term. The HiLS-correct score is:

```
ŝ_{i,c} = q^T k'_c / √d + b'_c     where b'_c = -Σ_j p_j log p_j
```

The entropy `b'_c` is a **byproduct of the softmax already computed in
`summarize_chunk_into`** — the `scores_buf[..chunk_size]` array holds `p_j`
after `softmax_inplace`. Computing `b'_c = -Σ p_j log p_j` is one extra
reduction over data already in L1. Zero allocation, O(S) per chunk.

**Why it matters:** without `b'_c`, the chunk score is `q^T k'_c` — this is
the "mean-logit" regime only. With `b'_c`, the score adapts: high-entropy
chunks (uniform logits, many mildly-relevant tokens) get a `+log S` boost;
low-entropy chunks (one dominant token) get `+0`. This is exactly the
LogSumExp behavior that mean-pooling cannot capture. The paper's Tab 6 ablation
("w/o Prop 3.1") shows this term contributes real PPL/extrapolation gains.

### 2.3 §3.5 modelless unblock check (MANDATORY before riir-train deferral)

**Gate:** "HiLS-quality chunk selection appears to need training (landmark
tokens + Q-Cal)."

→ Does the failure (inaccurate chunk selection) have a SYSTEMATIC,
characterizable cause?
- The failure is that mean-pooled summaries dilute concentrated attention mass.
  This IS systematic and characterizable (Prop 3.1 names it: LogSumExp ≠ mean).
- **YES, systematic.**

→ Can freeze/thaw (path 1) fix it?
- riir-train would train the landmark `head_cls` + Q-Cal, then freeze. Standard
  flow. This IS a riir-train dependency for the *full* benefit, but it does not
  address the modelless kernel.
- **Partial** — handles the learned-query half, not the entropy-bias half.

→ Can a deterministically constructed reader/writer LoRA (path 2) fix it?
- Q-Cal is LoRA-shaped (`Δq = W_up W_down h_i`) but the paper random-inits it
  and the mechanism is "not yet fully understood." No closed-form deterministic
  construction exists.
- **NO.**

→ Can a latent-space projection/gate (path 3) fix it?
- **YES for the entropy bias.** `b'_c = -Σ p_j log p_j` is a deterministic
  latent-space correction to the chunk score. It corrects the mean-pooling bias
  by adding the entropy term. This is modelless and implementable now.
- **BUT**: with zero-init `head_cls`, `p_j` is uniform → `b'_c = log S`
  (constant across all chunks) → no ranking change → **dormant at zero-init**.
  The correction only bites when `head_cls` is non-trivial (trained or
  deterministically seeded).

**§3.5 verdict:** the entropy bias is **modelless-validable** (path 3) and
should be implemented now — it is a deterministic correction that is dormant at
zero-init (no regression, backward-compatible) and activates the moment
riir-train provides learned `head_cls` vectors. The learned-query half (landmark
tokens, Q-Cal) is a genuine riir-train dependency.

### 2.4 Latent-space reframing (mandatory per skill)

How HiLS's mechanism looks on the codebase's latent-state kernels:

| Substrate | HiLS reframing | Fit |
|---|---|---|
| **HLA per-NPC latent state** (8-dim, recurrent) | HLA is already a compression of history; chunking a temporal window of HLA states is redundant. | Weak |
| **`latent_functor/` operations** | `k'_c = Attn(q'_c, K_c, K_c)` IS a latent-to-latent operation; the entropy `b'_c` is a concentration quality measure on the compression. Maps to any "summarize N latents into 1" functor. | Moderate |
| **`cgsp_runtime/` curiosity** | Chunking curiosity signals + entropy to find exploration hotspots. | Weak |
| **LatCal fixed-point** | `b'_c` is a scalar; could be committed. But it's a quality metric, not an economic quantity. | Weak |
| **`NeuronShard` consolidation** (riir-neuron-db) | **Strongest.** Raven/δ-Mem consolidates wake events into a shard — structurally identical to HiLS compressing keys into `k'_c`. The entropy `b'_c` is a "how concentrated is this consolidation?" signal, related to but distinct from the existing `output_flatness` / `intrinsic_dim` freeze-gate inputs. The "compression enhances retrieval" insight (§1.6) directly applies: aggregating wake-event latents cancels noise, preserves shared signal. | **Strong (speculative)** |
| **DEC Stokes operators** | Entropy ≈ information content; relates to DEC smoothing/coarsening. The "aggregation cancels noise" insight is a DEC-coarsening property. | Moderate |

The strongest latent reframing is **consolidation in riir-neuron-db**: the
entropy of the consolidation-weight distribution as a concentration metric.
However, riir-neuron-db already has `output_flatness` + `intrinsic_dim` as
freeze-gate inputs; the entropy is related but not obviously superior. This is
a fusion idea worth noting but not strong enough to route a separate note to
riir-neuron-db without a PoC.

### 2.5 Fusion (the closest 2-3 cousins + what the combination produces)

| Cousin | Repo | Relationship |
|---|---|---|
| **DashAttention** (R071, P106) | katgpt-rs | Ships `k'_c` (the attention-weighted key sum). HiLS adds `b'_c` (entropy) + the hierarchical factorization. |
| **VortexFlow** (R176, P196) | katgpt-rs | Programmable sparse KV routing; HiLS's top-K chunk selection is a specific routing policy. |
| **PFlash block-sparse** (P044) | katgpt-rs | Block-level importance scoring for speculative prefill; the entropy bias could improve PFlash's block scorer. |
| **Raven/δ-Mem consolidation** | riir-neuron-db | Structurally identical "compress N vectors into 1 + metadata" operation. The entropy is a concentration metric candidate. |

**Novel combination:** DashAttention `k'_c` × HiLS entropy bias `b'_c` ×
riir-neuron-db consolidation concentration gate. If the entropy of the
consolidation-weight distribution were added as a freeze-gate input alongside
`output_flatness`, it would detect "one wake event dominates" (low entropy →
high-confidence consolidation → freeze-eligible) vs "many equally-weighted
events" (high entropy → diffuse → not yet settleable). This is a fusion idea,
novelty TBD — needs Q1-Q4 check before any verdict, and a PoC before any claim.

---

## 3. Verdict

**Tier: Gain**

**One-line reasoning:** The paper's headline value (matching/beating full
attention, 512× extrapolation) is a **training recipe** (landmark token tuning
+ Q-Cal + HoPE + 50B-token CPT) → riir-train; the modelless-transferable
kernel is the **entropy bias `b'_c`** (Prop 3.1), which is a deterministic,
zero-alloc, backward-compatible add-on to our existing `chunk_summary.rs` —
dormant at zero-init, activated when riir-train provides learned `head_cls`.
Incremental improvement to an existing opt-in primitive, not a new capability
class.

### Why NOT Super-GOAT (novelty gate Q1-Q4)

- **Q1 (no prior art?):** NO. DashAttention (R071/P106) ships `k'_c` — the
  attention-weighted key sum — already. The entropy bias `b'_c` is incremental.
  Grep confirmed: `chunk_summary.rs` computes `softmax(q̄·K/√d)·K` (Eq 8) but
  no `b'_c`, no surrogate score, no hierarchical factorization in
  `dash_attn/*.rs`.
- **Q2 (new capability class?):** NO. Same capability (chunk-wise sparse
  attention selection), incremental scoring improvement.
- **Q3 (product selling point?):** NO. "Slightly better chunk scoring formula"
  is not a moat.
- **Q4 (force multiplier?):** NO. Single primitive, refines one existing module.
  The consolidation-concentration fusion idea (§2.5) is speculative and needs
  a PoC before it could be claimed.

All NO → NOT Super-GOAT. No private guide created (per §1.5 "no candidate
escape hatch" rule — this is a firm Gain, not a deferred Super-GOAT).

### Why NOT GOAT

No provable **modelless** gain: at zero-init `head_cls`, the entropy `b'_c` is
constant (`log S`) across all chunks → no ranking change → no quality or latency
gain. The benefit requires a non-trivial `head_cls`, which is a riir-train
dependency. The entropy bias is correct and ready, but dormant until trained
queries exist.

### MOAT gate (katgpt-rs domain)

- **In scope:** yes — paper-derived fundamental primitive for the attention
  stack, behind the existing `dash_attn` feature flag.
- **Strengthens moat:** neutrally — it completes the theoretical correctness of
  an existing opt-in primitive but does not create a new selling point.
- **Promote/demote:** stays opt-in (`dash_attn` is not default-on). The entropy
  bias is backward-compatible (dormant at zero-init). No stack-slot change.

### Routing

| Component | Destination | Rationale |
|---|---|---|
| Entropy bias `b'_c` primitive | **katgpt-rs** (this note + `.issues/044`) | Modelless, deterministic, completes `chunk_summary.rs`. Issue, not plan (optimization of existing primitive per AGENTS.md). |
| Landmark token tuning recipe | **→ riir-train** | Training method (continued pretraining, <1% param tuning). |
| Q-Cal low-rank query calibration | **→ riir-train** | LoRA-style adapter, random-init, trained. No deterministic construction (mechanism "not yet fully understood" per paper). |
| HoPE positional encoding | **→ riir-train** | Training-time positional strategy. |
| M-query adjacent packing kernel | (not routed) | GPU Tensor-Core-specific; our stack is CPU/SIMD + ANE. The "adjacent queries overlap" observation is noted for any future batched sparse-attn design. |
| Hierarchical softmax factorization | (noted, not routed) | Pure algebra whose purpose is gradient flow; dormant for modelless inference. |
| Consolidation-concentration fusion idea | (noted in §2.5) | Speculative; needs Q1-Q4 + PoC before any riir-neuron-db note. |

---

## 4. What to implement (modelless, katgpt-rs)

**Issue `.issues/044` — DONE (2026-07-09).** All six tasks (T1-T6) landed:
`summarize_chunk_into_with_entropy` computes `b'_c = -Σ p_t log p_t` as one
reduction over the already-resident softmax weights (zero alloc);
`score_blocks_entmax_with_entropy_into` adds `b'_c` to each chunk logit
before α-entmax; `ChunkSummaryCache` now stores per-chunk-per-head entropy
alongside summary keys; `forward.rs` prefill stores entropy, decode threads
it into routing. **Issue file removed** (per AGENTS.md noise rule); the
behavioral contract is captured by the tests in `chunk_summary.rs`
(`test_entropy_bias_*`) and `routing.rs`
(`test_score_blocks_with_*_entropy_*`).

Backward-compatible: at zero-init the bias is constant (`ln(chunk_size)`)
and the entmax ranking is bit-identical. `goat_106_dash_attn` GOAT-proof
re-passes unchanged.

The entropy computation reuses `scores_buf` after `softmax_inplace`:

```rust
// After softmax_inplace(&mut scores_buf[..chunk_size]):
// scores_buf[t] now holds p_t. Compute b'_c = -Σ p_t log p_t.
let mut entropy = 0.0f32;
for &p in &scores_buf[..chunk_size] {
    if p > 0.0 {
        entropy -= p * p.ln();
    }
}
// entropy is b'_c. Return alongside the summary key.
```

**No GOAT gate yet** — the gain is dormant at zero-init. The gate becomes
meaningful only when riir-train provides learned `head_cls`; at that point,
re-gate with a before/after on chunk-selection accuracy (NIAH-style) at
fixed budget.

---

## TL;DR

HiLS-Attention's value is its **training recipe** (landmark tokens + Q-Cal +
HoPE + 50B-token CPT) → riir-train. The modelless kernel is **Prop 3.1's
entropy bias `b'_c = -Σ p_j log p_j`** — a deterministic, zero-alloc add-on
to our shipped `dash_attn/chunk_summary.rs` (Plan 106) that makes the chunk
score faithful to LogSumExp mass. It is dormant at zero-init (backward-
compatible) and activates when riir-train provides learned landmark queries.
**Verdict: Gain** for katgpt-rs; training recipe → riir-train. Issue `.issues/044`
tracks the entropy-bias implementation.
