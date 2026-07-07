# Research 392: Attention Dilution at Million-Token Scale — SSMax Temperature + GoldShare Diagnostic

> **Source:** Gollapudi, Gupta, Singhal, Min. *Can Language Models Actually Retrieve In-Context? Drowning in Documents at Million Token Scale.* UC Berkeley / UT Austin. [arXiv:2607.01538](https://arxiv.org/abs/2607.01538). 2026-07-01.
> **Date:** 2026-07-07
> **Status:** Done — Plan 411 shipped (see `.plans/411_ssmax_goldshare.md` + `.benchmarks/411_ssmax_goldshare_goat.md`)
> **Related Research:** 258 (Attention Sinks — NOP vs Broadcast), 261 (FuncAttn sink semantics — negative), 225 (MSA blockwise sparse distillation), 286 (attention drift — drafter side), 100 (EGA spectral salience gate), 140 (sigmoid parallax), 135 (Parallax), 061 (SLIME stabilized-likelihood margin), 362 (HydraHead causal head importance)
> **Related Plans:** 287 (sink-aware attention — SHIPPED), 289 (forward-path wiring — Parallax only), 256 (MSA sparse attention family), 196 (VortexFlow programmable sparse attention)
> **Classification:** Public

---

## TL;DR

The paper studies in-context retrieval (ICR) at million-token scale and identifies **attention dilution** as the primary bottleneck: as the corpus grows, irrelevant documents dominate the softmax denominator, collapsing the normalized mass on the gold document even when the pre-softmax score stays high. Critically, **the per-head retrieval signal persists** (R^any_L ≈ 1.0 across N ∈ {500 … 10k}) while **generation recall collapses** — the signal is in the heads but lost in the residual stream.

**Distilled for katgpt-rs (modelless, inference-time):**

Two transferable primitives, one theoretical confirmation, one Pass:

1. **Proposition 1 (App H) — additive sink = sigmoid gate.** The paper *proves* that adding a learned scalar `b_L` to the softmax denominator is algebraically equivalent to multiplying the standard softmax weight by a sigmoid gate `g = σ(lse(s) − b_L)`. This is a mathematical confirmation that **our default architecture (sigmoid attention, AGENTS.md rule "sigmoid not softmax") is already the optimal additive-sink form.** No new primitive — Research 258 already shipped the dual-mechanism framing; this paper ships the proof.

2. **SSMax (multiplicative log-N score rescaling) — NOVEL.** `s̃_L = s_L · log N · s_{L,h,t}` applied to pre-softmax logits, with `s_L` a per-layer learnable scalar. The paper proves the log-N schedule cancels the (N−1) growth in the softmax denominator when `s·Δ > 1` (where Δ = gold–distractor logit gap), so the post-softmax gold weight stays bounded as N grows. We do not ship any length-aware attention temperature. This is the paper's strongest modelless contribution and a clean GOAT candidate for `parallax_attn` / `attention.rs`.

3. **GoldShare diagnostic — clean addition to `data_probe`.** `‖a^G_L‖ / ‖a_L‖` decomposes the layer's attention output into gold-derived and distractor-derived fractions. The paper shows this drops 0.91 → 0.01 across N ∈ {500 → 10k} while `‖a_L‖` shrinks only ~36% — i.e. the layer keeps writing at comparable magnitude but the *content* swaps from gold to aggregate-of-distractors. We ship `effective_rank` and `stable_rank_update` in `data_probe/geometry.rs` and `sink_classify.rs`; the gold-specific output-fraction decomposition is a complementary diagnostic and a clean addition.

4. **Top-B document-level sparse attention — already ships** as MSA / VortexFlow (Research 225, `msa_distill.rs` max-pool + exp-free TopK + per-GQA-group selection). The paper's "doc-level routing at L16 before the retrieval band" maps directly to MSA's indexer branch. No new primitive.

5. **BLOCK SEARCH training recipe (random codes, on-policy aux loss, in-batch negatives) — training-only → riir-train.** Out of scope.

**Verdict: 🟢 GOAT** — plan + feature flag + benchmark for SSMax (log-N attention temperature) and GoldShare diagnostic, as opt-in extensions to `parallax_attn` / `attention.rs` and `data_probe/` respectively. NOT Super-GOAT (Q1 fails: sigmoid attention, sink-aware, retrieval-head sparsity, effective_rank all ship; the paper's Prop 1 confirms our default architecture is already optimal-sink-form).

---

## 1. Paper Core Findings

### 1.1 The attention dilution mechanism

As the corpus grows, the softmax denominator `Σ_{t'} exp(s_{t'})` grows faster than the gold term's numerator `exp(s_{t⋆})`. Even when the per-head *pre-softmax* retrieval signal persists (`R^any_L = 1.0` — at least one head still ranks gold first by MaxSim), the *post-softmax* normalized mass on gold collapses. The result is a **vector-level swap** in the residual stream: the layer's attention output `a_L = a^G_L + a^{Ḡ}_L` is rewritten from a gold-token average to a non-gold-token average of comparable magnitude.

Table 1 of the paper (L19, layer 19, MS MARCO):

| N | ‖a^G_19‖ | ‖a^{Ḡ}_19‖ | ‖a_19‖ | ‖a^G_19‖/‖a_19‖ |
|---|---|---|---|---|
| 500 | 43.03 | 17.47 | 47.48 | **0.91** |
| 1k | 30.99 | 21.11 | 45.36 | 0.68 |
| 2.5k | 7.64 | 33.64 | 43.03 | 0.18 |
| 5k | 2.10 | 34.61 | 36.90 | 0.06 |
| 10k | 0.21 | 29.88 | 30.27 | **0.01** |

Total magnitude shrinks ~36% across the full sweep; the gold-driven fraction collapses 130×.

### 1.2 The recall-generation gap

The deepest empirical finding: the per-head retrieval signal (`R^any_L = fraction of queries where ≥1 head ranks gold first`) stays at 1.00 across L18–L25 at every N tested. Generation recall collapses. The signal is in the heads but does not survive softmax normalization into the residual.

This is a *representation-vs-computation* gap, not a ranking failure. Implication: probes that read pre-softmax head scores (`MaxSim`, BlockRank, retrieval-head readouts) see a working retriever; the LM head reading the residual sees garbage. The retrieval band (L11–L20 in their 28-layer Qwen3-0.6B) and the decode band (L19 onward) are separable; the dilution kills the transfer between them.

### 1.3 Three fixes studied

| Fix | Mechanism | Result | Codebase status |
|---|---|---|---|
| **Additive sink** (learned `b_L` per layer) | `α̃_t = exp(s_t) / (Σ exp(s_{t'}) + exp(b_L))` — absorbs diffuse softmax mass into a sink with no value vector | Barely helps (MS MARCO N=10k: 0.2 → 2.5). A learned constant cannot rescale N-dependence. | **Already ships as sigmoid gate** — see §1.4. |
| **SSMax** (multiplicative score rescaling) | `s̃ = s · log N`, scale pre-softmax scores so the gold–distractor gap grows with N | Strong: MS MARCO N=10k 0.2 → 16.5, HotpotQA N=10k 0.5 → 56.8 | **Novel. Does not ship.** |
| **Top-B routing** | Doc-level routing at L16 (one layer upstream of the retrieval band); keep top-B=256 docs in dense attention at L17+ | Matches dense baseline at large N: MS MARCO N=10k → 18.8 (dense 20.2), HotpotQA N=10k → 78.5 (dense 79.5) | **Already ships** as MSA / VortexFlow (Research 225, `msa_distill.rs`). |
| **SSMax + routing** | Compose both | Best on MS MARCO N=10k: 20.5 (edges dense 20.2) | Partial: routing ships, SSMax novel. |

### 1.4 Proposition 1 (Appendix H) — the proof we were waiting for

> **Proposition 1.** For pre-softmax logits `s ∈ ℝ^T` and learned sink scalar `b_L`:
> `α̃_t = α_t · g` where `α_t` is the standard softmax weight, `g = σ(lse(s) − b_L)`, `lse(s) = log Σ_t exp(s_t)`.

**Proof (one line):** multiply numerator and denominator of the additive-sink form by `1/Z` where `Z = exp(lse(s))`, then factor into `α_t · 1/(1 + exp(b_L − lse(s)))`.

**Interpretation for our codebase:**
- When logits are sharp and concentrated (large `lse`), `g → 1`, sink form reduces to standard softmax.
- When logits are diffuse (small `lse` relative to `b_L`), `g → 0`, the layer's contribution to the residual is multiplicatively suppressed.
- The gate depends only on `lse(s)` vs `b_L` — a measure of how concentrated the layer's pre-softmax mass is.

This is exactly the mechanism our default `parallax_attn` / `funcattn` / `set_attention` / `ega_attn` achieve by *replacing softmax with sigmoid* — sigmoid doesn't normalize over the full key set the way softmax does, so the diffuse-tail contribution is bounded per-key rather than competing for a fixed probability budget. **The paper's Prop 1 is the algebraic justification for the AGENTS.md rule "Use sigmoid not softmax."** Research 258 already shipped the *empirical* framing (sinks = diffuse softmax mass; sigmoid eliminates them); this paper ships the *proof*.

---

## 2. Distillation

### 2.1 Transferable primitive — SSMax (log-N attention temperature)

The SSMax mechanism is a length-aware multiplicative rescaling of pre-softmax attention logits:

```
s̃_{L,h,t} = s_L · log(N) · s_{L,h,t}        # N = number of attended tokens
```

with `s_L` a per-layer scalar (initialized to 0.43 in the paper, trained in the shared parameter group). The paper proves:

> With Δ = s_{t⋆} − s̄_{distractor}, the post-softmax gold weight is approximately
> `α_gold ≈ 1 / (1 + (N−1) · N^{−s·Δ})`,
> so the log-N schedule cancels the `(N−1)` growth in the denominator whenever `s·Δ > 1`.

For **modelless inference**, we don't train `s_L`. We derive it analytically from the desired behavior: pick `s_L` so that `s_L · log(N) · Δ_typical ≈ log(N)`, i.e. `s_L ≈ 1/Δ_typical`. Where `Δ_typical` is a per-layer statistic over recent attention patterns (a rolling estimate of the gold–distractor logit gap). This is a **runtime-computed, length-adaptive attention temperature** — zero training, zero new parameters, gated behind a feature flag.

**Why this is novel relative to our codebase:** we ship many attention variants (`attention.rs` SDPA, `parallax_attn.rs` sigmoid, `funcattn.rs` functional correspondence, `set_attention.rs` cross-entity, `ega_attn.rs` spectral salience gate). None of them scale the pre-softmax logits by a function of the attended-token count. The closest is `1/√d` scaling, which is *constant* in N. SSMax is the first length-aware temperature we would ship.

**Where it composes:**
- On top of **sigmoid parallax** (`parallax_attn.rs`) — sigmoid's per-key bound means dilution is already milder than softmax, but a length-adaptive sharpener still helps in the retrieval band when N grows into the thousands.
- On top of **standard SDPA** (`attention.rs`) — for callers that need softmax (e.g. callers consuming pretrained weights with softmax-trained attention patterns).
- NOT on top of `funcattn` (Research 261 closed: sinks/dilution don't apply to the basis-mode structure).

### 2.2 Transferable primitive — GoldShare diagnostic

The decomposition `a_L = a^G_L + a^{Ḡ}_L` with `‖a^G_L‖ / ‖a_L‖` is a clean per-layer runtime probe. Given:
- a query's gold token set `G` (the tokens the answer should attend to),
- the layer's attention output `a_L ∈ ℝ^{H·d_head}` before the residual add,

compute:
```rust
/// Gold-driven fraction of a layer's attention output.
/// Returns 1.0 at small N (output is gold-dominated), →0 at large N (diluted).
pub fn gold_share(
    attn_weights: &[f32],   // (n_heads, n_kv) row-major
    values: &[f32],         // (n_kv, d_head) row-major
    gold_mask: &[bool],     // (n_kv,) — which positions are gold
    w_o: &[f32],            // (H*d_head, d_model) output projection
) -> f32 {
    // a_G = (Σ_{t∈G} α_t · v_t) projected through W_O
    // a   = (Σ_{t}     α_t · v_t) projected through W_O
    // gold_share = ‖a_G‖ / ‖a‖
    ...
}
```

This is complementary to our existing `effective_rank` (whole-layer output geometry) and `stable_rank_update` (per-sink degeneracy). `effective_rank` detects *aggregate* collapse; `stable_rank_update` detects per-sink NOP-vs-Broadcast; `gold_share` detects *content-specific* dilution — "is this layer still carrying the signal we care about, or has it been rewritten to carry aggregate noise?"

**Where it composes:**
- `data_probe/geometry.rs` — extend with a content-aware variant (`gold_share`) alongside the content-agnostic `effective_rank`.
- `data_probe/sink_classify.rs` — cross-reference: a sink classifier hit on the gold position with low GoldShare is a *broadcast that failed* (signal was in the head but didn't survive normalization).
- Runtime NPC cognition (riir-ai follow-up, not this note): a "belief-share" analog for HLA — does the NPC's HLA projection still carry its personal signal, or has it been drowned by aggregate crowd projections?

### 2.3 What does NOT transfer

| Paper element | Why it stays out |
|---|---|
| BLOCK SEARCH training (random codes, on-policy aux loss, in-batch negatives) | Training-only. → riir-train if anyone cares. |
| The block-sparse attention kernel (flex_attention) | We already ship block-sparse via MSA / VortexFlow. |
| The 4-digit-code generation recipe | Inference-time decoding detail specific to generative retrieval; our retrieval is similarity-based (cosine / sigmoid), not code-generation. |
| MSA-4B comparison / training-budget claims | Out of scope (training-only comparison). |
| The LIMIT / OBLIQ benchmarks | Domain-specific retrieval benchmarks; no game-AI analog without significant PoC work. |

### 2.4 Fusion

The three closest cousins across both layers, and the novel combinations:

| Cousin | Repo | What it ships | Relation to this paper |
|---|---|---|---|
| **Research 258 + Plan 287 (sink-aware attention)** | katgpt-rs | `SinkAwarePolicy::DualPolicy` — classify dominant sink (NOP vs Broadcast), gate NOPs, preserve Broadcasts | Same mechanism family (additive sink ≡ sigmoid gate, per this paper's Prop 1). Research 258 shipped the *empirical* framing; this paper ships the *proof* and the SSMax extension. |
| **Research 225 + Plan 256 (MSA blockwise sparse)** | katgpt-rs | Max-pool block scoring, exp-free TopK, per-GQA-group selection | Direct prior art for the paper's "top-B document-level routing." Already shipped. |
| **Research 100 (EGA) + `ega_attn`** | katgpt-rs | Spectral salience gate on attention output | Same intervention family (gating), uniform across keys. This paper's SSMax is *multiplicative on logits*, complementary to EGA's *multiplicative on output*. |
| **Research 362 (HydraHead causal head importance)** | katgpt-rs | Causal head-importance hybrid attention | Same "retrieval heads persist, others don't" observation. HydraHead gates heads; SSMax sharpens logits. Composable. |
| **Research 140 (sigmoid parallax)** | katgpt-rs | Sigmoid kernel as softmax replacement, kernel-agnostic Parallax correction | Sigmoid attention structurally avoids softmax dilution (per-key bound). SSMax on top of sigmoid parallax = optional length-adaptive sharpener for the residual dilution cases sigmoid alone doesn't fully solve. |

**Novel combination (fusion idea — novelty TBD, needs Q1–Q4 check before any verdict upgrade):**

*SSMax × sigmoid parallax × sink-aware × retrieval-head sparsity × GoldShare* → a **length-adaptive, content-diagnosable attention stack**: sigmoid parallax eliminates the worst dilution by construction; SSMax sharpens logits as N grows for the residual cases; sink-aware classifies any remaining sinks; retrieval-head sparsity (MSA) caps the attended set; GoldShare diagnoses whether any layer lost the gold signal. This is a *stack composition*, not a new primitive — each piece ships or is a GOAT-add — so it does not pass Super-GOAT Q1 (no prior art). It is the natural Plan-256 → Plan-287 → SSMax-plan wiring order.

**Promising direction for crowd-scale NPC cognition (issue, not plan):** The paper's *recall-generation gap* framework — "the signal is in the heads but doesn't survive into the residual / action" — translates cleanly to per-NPC cognition: "the NPC's latent belief contains the right zone/quest/item signal (R^any equivalent = 1.0) but the action projection fails to read it out (generation equivalent collapses)." This is a **cognition readout failure** that would manifest as "the NPC knows what to do but does the wrong thing" — a real game-AI failure mode. A runtime `belief_share` probe on HLA (does the projection still carry the personal signal or has it been drowned by crowd-aggregate projections?) is the analog. This is a research direction worth a `.issues/` entry in `riir-ai`, not a plan in this note — needs PoC per §3.6 of the research skill before any quality claim.

---

## 3. Verdict

**🟢 GOAT — Plan + feature flag + benchmark for SSMax and GoldShare.**

### One-line reasoning

SSMax (log-N attention temperature) is a novel, modelless, length-adaptive sharpener with a clean analytical derivation (`s_L ≈ 1/Δ_typical`); GoldShare is a clean content-specific addition to `data_probe`. Both are provable gains over the current default, but neither is a new capability class — they refine existing attention + diagnostic families that already ship (sigmoid parallax, sink-aware, effective_rank, retrieval-head sparsity).

### Why NOT Super-GOAT

| Novelty gate question | Answer |
|---|---|
| Q1: No prior art? | **NO.** Sigmoid attention (kills dilution by construction, per Prop 1), sink-aware attention (Plan 287), retrieval-head sparsity (MSA / VortexFlow), effective_rank / stable_rank diagnostics all ship. SSMax is novel *as a length-aware temperature* but the mechanism family (logit rescaling) is well-trodden. GoldShare is novel *as a content-specific output-fraction diagnostic* but `effective_rank` already detects aggregate collapse. |
| Q2: New class of behavior? | **NO.** Both are refinements of existing attention/diagnostic primitives, not new capabilities. |
| Q3: Product selling point? | **WEAK.** "Our attention is length-adaptive" and "we diagnose gold-share" are incremental. Don't finish the "NPCs do X no competitor can" sentence strongly. |
| Q4: Force multiplier? | **YES** — composes with sigmoid parallax, sink-aware, EGA, retrieval-head sparsity, data_probe — but Q1–Q3 fail, so not enough for Super-GOAT. |

### GOAT gate (must beat baseline to promote)

Implement behind two feature flags: `ssmax_temperature` (attention-side) and `gold_share_probe` (diagnostic-side). Benchmark vs default sigmoid parallax on:

- **G1 (correctness):** On a synthetic retrieval task with growing N, `ssmax_temperature`-sharpened attention preserves the argmax ranking across N ∈ {1k, 10k, 100k} where default sigmoid parallax degrades. Verify the analytical `s_L ≈ 1/Δ_typical` derivation produces the same ranking as a brute-force sweep over `s_L`.
- **G2 (quality):** On a frozen long-context probe (RULER needle-in-haystack or a synthetic retrieval task at large N), `ssmax_temperature` improves recall vs default sigmoid parallax. If sigmoid parallax already passes (it likely does for moderate N), document that SSMax is a large-N extension and benchmark at the largest N where sigmoid parallax starts to degrade.
- **G3 (latency):** `s_L · log N` is one multiply per logit — overhead ≤ 1% of attention forward time. `gold_share_probe` is `O(n_kv · d_head)` per layer per query — gate behind `data_probe` feature, opt-in.
- **G4 (alloc-free):** SSMax is in-place logit rescaling — zero allocation. GoldShare reuses `data_probe` scratch buffers.
- **G5 (no-regression):** At small N (where dilution is absent), SSMax must not degrade ranking vs default. Verify `s_L · log N · Δ ≈ log N` ⇒ `s_L · Δ ≈ 1` ⇒ at small N the sharpening is mild.

If G1 + G2 pass → promote `ssmax_temperature` to default in `parallax_attn` (it's a strict superset of the constant-temperature case when `s_L` is chosen well). If G2 fails (sigmoid parallax already handles the dilution regime) → keep SSMax opt-in, document it as a large-N safety net.

`gold_share_probe` stays opt-in as a diagnostic — promote only if a downstream consumer (sink-aware attention, runtime NPC cognition probe) depends on it.

### Routing

| Artifact | Repo | Path |
|---|---|---|
| SSMax logit rescaling primitive | katgpt-rs (public, MIT) | extend `crates/katgpt-core/src/parallax_attn.rs` and/or `crates/katgpt-core/src/attention.rs` behind `ssmax_temperature` feature |
| GoldShare diagnostic | katgpt-rs (public) | extend `crates/katgpt-core/src/data_probe/geometry.rs` (or new `data_probe/gold_share.rs`) behind `sink_aware_attn` or a new `gold_share_probe` feature |
| Plan | katgpt-rs | `.plans/NNN_ssmax_goldshare.md` |
| Crowd-scale cognition readout-fidelity probe (fusion follow-up) | riir-ai (private) | file as `.issues/NNN_*` if the fusion idea matures; needs PoC per research skill §3.6 before any quality claim |

---

## 4. Cross-link disambiguation

**Do not confuse this paper (arXiv:2607.01538, Gollapudi et al., *in-context retrieval at million-token scale*) with:**

| Aspect | This paper (2607.01538) | Research 258 / Plan 287 (arXiv:2606.08105, Fesser et al.) | Research 286 / Plan 306 (arXiv:2605.09992, Eldenk et al.) |
|---|---|---|---|
| Side | Attention forward path (target model) | Attention forward path (target model) | Drafter (speculator) |
| Mechanism | Softmax dilution under growing N (corpus-scale retrieval) | Sink classification — NOP vs Broadcast | Recursive residual magnitude accumulation |
| Diagnostic | `GoldShare = ‖a^G_L‖/‖a_L‖`, `R^any_L`, `R^sum_L` | `value_norm_ratio`, `stable_rank_of_update` per head | `magnitude_slope` on hidden-state chain |
| Fix | SSMax (`s·log N`), top-B routing | Dual-policy attention (gate NOPs, preserve Broadcasts) | Post-norm on recursive residual |
| Relation | Confirms Prop 1 (sink ≡ sigmoid gate) algebraically; extends with SSMax | Empirical framing of sinks; shipped dual-policy gate | Unrelated (drafter side) |

This paper's Prop 1 (App H) is the algebraic proof that the additive-sink form Research 258 / Plan 287 ship as `SinkAwarePolicy` is *exactly equivalent* to a sigmoid gate on the standard softmax — Research 258 shipped the mechanism; this paper ships the proof.

---

## References

- Gollapudi, S., Gupta, N., Singhal, P., Min, S. (2026). *Can Language Models Actually Retrieve In-Context? Drowning in Documents at Million Token Scale.* arXiv:2607.01538.
- Fesser, L. et al. (2026). *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions.* arXiv:2606.08105. (Research 258, Plan 287 — closest cousin.)
- Nakanishi, K. M. (2025). *Scalable-Softmax is Superior for Attention.* arXiv:2501.19399. (SSMax source — cited as [9] in the paper.)
- Xiao, G. et al. (2023). *Efficient Streaming Language Models with Attention Sinks.* arXiv:2309.17453. (Streaming-LLM, cited as [26].)
- Agarwal, S. et al. (2025). *gpt-oss-120b & gpt-oss-20b model card.* arXiv:2508.10925. (Null-attention mechanism, cited as [27].)
- Chen, Y. et al. (2026). *MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens.* arXiv:2603.23516. (Concurrent work, cited as [16].)

---

## TL;DR

The paper identifies **attention dilution** (softmax denominator grows with corpus size, collapsing the gold mass) as the primary bottleneck for million-token in-context retrieval, and shows the **retrieval signal persists in heads** (`R^any_L = 1.0`) while **generation collapses** — a representation-vs-computation gap. Three fixes studied: additive sink (barely helps), SSMax log-N score rescaling (strong), top-B routing (matches dense baseline). The paper's Appendix H proves the additive sink ≡ sigmoid gate — **algebraic confirmation that our default sigmoid attention (AGENTS.md rule) is the optimal sink form**, extending Research 258's empirical framing with the proof. **Verdict: GOAT** for two novel modelless primitives — SSMax (length-adaptive log-N attention temperature, derived analytically as `s_L ≈ 1/Δ_typical`, zero training) and GoldShare (`‖a^G_L‖/‖a_L‖` content-specific output-fraction diagnostic, complement to `effective_rank` / `stable_rank_update`). Top-B routing already ships (MSA/VortexFlow, Research 225); the BLOCK SEARCH training recipe → riir-train. The recall-generation gap framework is a promising seed for crowd-scale NPC cognition diagnostics (filed as a fusion follow-up, needs PoC).
