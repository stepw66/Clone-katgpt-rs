# Research 407: Trees from Marginals — GDN Rollback-Free Tree Verification + Factorized-Drafter Acceptance Ceiling

> **Source:** [Trees from Marginals: Autoregressive drafting with factorized priors](https://arxiv.org/abs/2607.06763) — Yuma Oda, Ryan Mathieu, Roman Knyazhitskiy, Artur Chakhvadze (trymirai), arXiv:2607.06763, 7 Jul 2026
> **Code:** [github.com/trymirai/sglang](https://github.com/trymirai/sglang) (SGLang fork, CUDA kernels — not publicly fetchable at time of writing)
> **Date:** 2026-07-10
> **Status:** Active — GOAT (GDN tree-verify) + Gain (acceptance ceiling) + riir-train redirect (Weaver)
> **Related Research:** 070 (GDN2 backbone), 378 (HOLA hippocampal KV for linear attention), 002 (speculative decoding — original), 053 (δ-mem — modelless correction negative result), 026 (Gemma4 MTP), 059 (MoE speculative), 091 (SpecHop), 162 (Trust Region), 256 (GzipLM drafter), 316 (DSpark)
> **Related Plans:** 105 (GDN2 — default-on backbone), 012 (DDTree + KV rollback tree verify), 182 (QwenDeltaNet hybrid), 395 (HOLA KV for linear attn), 424 (this note's plan — GDN tree verify primitive)
> **Classification:** Public

---

## TL;DR

Oda et al. introduce **DFlash-TfM** ("Trees from Marginals"), a speculative decoding method that combines a factorized drafter (DFlash) with a lightweight autoregressive adapter (**Weaver**) that restores conditional dependencies between top-K marginal predictions by constructing proposal trees. The paper makes three contributions: (1) **Weaver** — a trained 56.7M-param adapter; (2) a **rollback-free tree-verification algorithm for Gated Delta Net (GDN) layers** — a pure-math derivation reducing delta-rule tree verification to a masked triangular solve; (3) an **acceptance-ceiling analysis** proving factorized drafters have a structural limit from their independence assumption, and showing argmax-of-marginal empirically beats full-marginal sampling at long draft depths. Combined result: 4.37× over AR decoding, 24.7% over tuned DFlash, on Qwen3.6-27B / B200 / SGLang.

**Distilled for katgpt-rs (modelless, inference-time):**

The **GDN rollback-free tree-verification algorithm** (§3.4) is the modelless gold. It extends the chunked delta-rule recurrence to tree-structured drafts via a partial order (ancestor relation), producing a masked triangular solve `(I + X)U = βV` that scores the entire tree in one pass without rolling back the recurrent state. No training, no learned parameters — pure linear algebra. This fills a **confirmed gap**: katgpt-rs ships GDN2 (Plan 105, default-on) for the main forward path and KV-cache snapshot/rollback tree verification for attention models (Plan 012), but has **no tree verification for GDN/delta-rule recurrent layers**. The paper explicitly frames this as an open problem (STree only handles diagonal Mamba recurrences; GDN's non-commutative `I − βkkᵀ` admits no cumulative-product form).

The **acceptance-ceiling insight** (§3.5) is a modelless Gain: for deep draft positions, argmax-of-marginal outperforms full-marginal sampling because the marginal averages over many possible prefixes (diluting signal) while argmax picks the most-likely token (more likely to align with the verifier's conditional). This is a one-line DDTree tuning change.

The **Weaver adapter** is a trained component (LK loss, Muon optimizer, 300k completions) → **riir-train** redirect after exhausting §3.5 modelless paths (see §3 below).

**Verdict: GOAT** for the GDN tree-verify primitive (provable 7.1× verify speedup at T=128, modelless, fills a real gap). **Gain** for the acceptance-ceiling insight (DDTree tuning). **→ riir-train** for Weaver.

---

## 1. Paper Core Findings

### 1.1 The factorized-drafter acceptance ceiling (§3.5)

A factorized drafter predicts T future positions in a single forward pass as **conditionally independent marginals** given the prefix. The per-position acceptance probability under speculative sampling is `1 − TV(p_draft, p_verifier)`. For an **oracle** marginal drafter using the true future-token marginal:

```
p(x_{L+t} | x_{<L}) = E_{x_{L+1},...,x_{L+t-1}} [p_verifier(x_{L+t} | x_{<L+t})]
```

the conditional acceptance rate at position L+t is:

```
p'_accept(x_{L+t}) = 1 − E_{x_{L+1..L+t-1}|A} [TV(p(·|x_{<L}), p(·|x_{<L+t}))]
```

**Key finding:** standard training objectives target matching per-token marginal probabilities, but this does **not** maximize expected acceptance rate. The marginal at depth t averages over exponentially many possible prefixes, diluting the signal. Empirically, **a Dirac delta at the argmax-of-marginal outperforms the full marginal distribution** at later positions — even for non-greedy (temperature 1.0) decoding (Figure 4, Figure 6). This is because the argmax token is more likely to be on a high-probability path that the verifier will accept.

This ceiling is **structural** (from the independence assumption), not a capacity limitation. Weaver exceeds it by conditioning on realized draft tokens within the top-K support.

### 1.2 Weaver — autoregressive residual over top-K marginals (§3.2)

Weaver is a lightweight autoregressive transformer (56.7M params, 1 layer, dim 2048, 16 heads) that predicts the **residual** between the factorized marginals and the verifier output distribution, operating exclusively over the top-K=512 candidate tokens:

```
u₀ = W_c · RMSNorm(h_verifier)          # conditioning from verifier
uᵢ = W_c · RMSNorm(hᵢ_dflash) + pᵢ      # conditioning from drafter lookaheads

ℓ_draft(u₀..L, t₁..d) = WeaverStep(u₀..L, t₁..d)   # autoregressive over draft path
```

The key efficiency trick: Weaver **never projects to the full vocabulary**. It only reads K=512 rows of the vocabulary projection matrix (selected as the DFlash top-K tokens), adds its residual logits to the DFlash output logits, and normalizes over the candidate set. This avoids the memory-bandwidth bottleneck of a standard autoregressive drafter.

**Training:** LK loss (λ-mixed KL + TV), Muon optimizer for large matrices, AdamW for the rest, WSD schedule, peak LR 2×10⁻⁴, single epoch on 300k completions (Nemotron V2 + LMSYS Chat 1M + OpenHermes 2.5 + CodeAlpaca). Trained against a frozen DFlash checkpoint.

**Result:** +77% mean acceptance length over chain DFlash, +32% over DDTree at the same tree size.

### 1.3 Rollback-free tree verification for Gated Delta Net (§3.4) — THE MODELLESS CONTRIBUTION

A GDN layer maintains a state matrix `Sₜ ∈ ℝ^{d_k × d_v}` updated by:

```
Sₜ = αₜ(I − βₜ kₜ kₜᵀ)Sₜ₋₁ + βₜ kₜ vₜᵀ       (state update)
oₜ = qₜᵀ Sₜ / √d_k                             (output read)
```

Tree verification requires scoring every token in the tree against only its ancestors. For attention this is a simple mask modification; for **recurrent** layers it was an open problem because:
- STree (prior work) exploits that Mamba's **diagonal** transition gives a cumulative-product form — the state at a node is a product of gates along its path.
- GDN's transition `I − βₜ kₜ kₜᵀ` is a **non-commuting matrix** — no cumulative-product form exists.

**The paper's solution:** use the **dual chunk form** of the linear recurrence. Define a partial order `≺` (ancestor relation) on the tree. Sort tokens topologically. Build lower-triangular interaction matrices:

```
Xᵢⱼ = 𝟙[j ≺ i] · (aᵢ/aⱼ) · βᵢ kᵢᵀkⱼ       (key-key interactions, ancestor-masked)
Yᵢⱼ = 𝟙[j ⪯ i] · (aᵢ/aⱼ) · qᵢᵀkⱼ           (query-key interactions, ancestor-masked)
```

where `aₜ = ∏_{i ⪯ t} αᵢ` is the cumulative decay along tree branches. Solve:

```
(I + X)U = βV           (forward substitution — the masked triangular solve)
(I + X)W = βaK          (auxiliary)
```

Then outputs and post-block state:

```
O = (1/√d_k)(aQS₀ + Y(U − WS₀))
```

**Critical design choice:** the state is **never speculatively written** during verification. The verify pass is read-only (uses S₀ from the committed prefix). Once Traversal verification picks the accepted leaf, a **single commit pass** replays the delta-rule recurrence along that one path and writes S₀. This is the only state write in the entire decode step — **zero rollback**.

**Kernel performance (B200, batch 1):**

| Tree size T | Per-branch recurrent | Fused masked-solve | Speedup |
|---|---|---|---|
| 16 | 55.2 µs/layer | 37.0 µs/layer | 1.5× |
| 32 | 103.9 µs/layer | 39.0 µs/layer | 2.7× |
| 64 | 195.0 µs/layer | 42.8 µs/layer | 4.6× |
| 128 | 408.3 µs/layer | **57.3 µs/layer** | **7.1×** |

The fused kernel tiles the forward substitution into Bc=32 sub-blocks; each diagonal block `(I + X_bb)` is inverted in registers by repeated squaring; off-diagonal coupling cascades over sub-blocks.

### 1.4 Batched tree construction (§3.3, Appendix A)

DySpec-style best-first expansion, modified for parallelism: extract top-w nodes from the max-heap (w=2–8) and expand them concurrently in a batched WeaverStep, rather than one-at-a-time. This constructs the complete tree in ⌈B/w⌉ sequential operations instead of B. Trade-off: includes some sub-optimal draft-probability nodes, but eliminates sequential Weaver inefficiency.

---

## 2. Distillation

### 2.1 What transfers modellessly to katgpt-rs

**Primitive 1 (GOAT): `GdnTreeVerifier` — rollback-free masked triangular solve for delta-rule tree verification.**

The entire §3.4 algorithm is pure linear algebra — no training, no learned parameters. The inputs are:
- A draft tree (parent pointers → ancestor bitmask per node)
- GDN layer parameters (keys K, values V, queries Q, decays α, write strengths β)
- The committed prefix state S₀

The output is the per-node output matrix O and, after acceptance, the committed state at the accepted leaf.

This maps cleanly to CPU SIMD (katgpt-rs is CPU-first): the forward substitution tiles into SIMD-width blocks, the ancestor mask is a bitmask, the cumulative decay is computed in log-space. The sub-block cascade (Eq. 13) maps to blocked SIMD matmul.

**Primitive 2 (Gain): Argmax-of-marginal for deep DDTree positions.**

The §3.5 analysis shows that at deep draft positions (t > ~4), using argmax-of-marginal (deterministic Dirac) gives higher acceptance than sampling from the full marginal, because the marginal averages over many prefixes and dilutes. This is a one-line change to `build_dd_tree`: at tree depth > crossover_point, use the argmax token instead of sampling. The crossover occurs around draft length 2–4 (Figure 6).

### 2.2 Fusion

The 2–3 closest existing primitives across the quintet:

1. **Plan 105 (GDN2)** — the backbone. The tree-verify algorithm reads GDN2's state S₀ and produces verified outputs. Direct consumer: enables speculative tree decoding on the `QwenDeltaNet` config (Plan 182).
2. **Plan 012 (DDTree + KV rollback)** — the existing tree verification infrastructure for attention. The GDN tree-verify is the **delta-rule analog** of Plan 012's attention KV-cache snapshot/rollback. Where Plan 012 snapshots/restores the KV cache per branch, the GDN verify avoids rollback entirely via the masked solve.
3. **Research 378 / Plan 395 (HOLA)** — hippocampal exact KV for linear attention. HOLA is about **what to store** (surprise-evicted bounded cache); the GDN tree-verify is about **how to verify trees** without rolling back the recurrent state. They compose: verify tree branches against both the recurrent state AND the hippocampal cache.

**Fusion idea (novel combination):** GDN tree-verify × HOLA hippocampal cache × DDTree. The rollback-free verify algorithm composes with HOLA's surprise-evicted cache: verify speculative tree branches against the GDN recurrent state via the masked solve, AND against the HOLA hippocampal cache via a parallel ancestor-masked softmax path, with no rollback on either. This produces a speculative decoding path for GDN+HOLA models that has zero rollback overhead — a combination none of the three primitives alone achieves. (Not planned in this note; tracked as a future fusion.)

**Fusion idea (latent reframing):** The factorized→coupled transition (independent marginals → conditional dependencies restored) is the **mean-field → beyond-mean-field** pattern. In our latent-space stack, this maps to:
- Per-NPC independent emotion scalars (factorized/mean-field) → crowd attention coupling (conditional dependencies) — already shipped (Plan 371 mean-field regime classifier + Plan 355 crowd set attention).
- The DEC analog: a 0-cochain (independent per-position values) → the exterior derivative d measures cross-position coupling. But this is a stretch — DEC operators are for spatial fields, not token sequences.

The latent reframing confirms: the factorized→coupled insight is well-represented in the latent-space stack. The genuinely new, modelless, transferable piece is the **GDN tree-verify algorithm** itself (a systems/math contribution), not a latent-space operation. This correctly lands as GOAT (not Super-GOAT) for katgpt-rs.

### 2.3 Vocabulary crosswalk (paper → codebase)

| Paper term | Codebase equivalent | Ships? |
|---|---|---|
| "factorized drafter" / "marginal drafter" | DFlash (`speculative/dflash.rs`), DDTree marginals | ✅ Plan 012 |
| "tree verification" / "tree attention" | KV-cache snapshot/rollback (`speculative_step_rollback_paged`) | ✅ Plan 012 (attention only) |
| "Gated Delta Net" / "GDN" / "delta rule" | GDN2 (`gdn2/` module, `Gdn2State`, `Gdn2Gate`) | ✅ Plan 105 (forward only) |
| "rollback-free tree verification for GDN" | — | ❌ **GAP (this note's plan)** |
| "top-K constrained vocabulary" | MTP threshold gating (Plan 055/117) | ✅ partial |
| "Weaver" (autoregressive adapter) | — | ❌ trained → riir-train |
| "speculative sampling" / "Leviathan verification" | `LeviathanVerifier` | ✅ |
| "Traversal verification" | — | ❌ (related to, but distinct from, our DDTree verify) |

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars). | — |
| **GOAT** | Provable gain over existing approach, but not a new class of capability. | **GDN tree-verify → katgpt-rs Plan 424.** Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | **Argmax-of-marginal DDTree tuning → captured here, no separate plan.** |
| **Pass** | Not relevant, OR training-only. | **Weaver → riir-train (one-line redirect).** |

### 3.1 GOAT verdict — GDN rollback-free tree verification

**One-line reasoning:** The masked triangular solve for delta-rule tree verification (§3.4) is a pure-math, modelless primitive with a provable 7.1× verify speedup at T=128 over the per-branch recurrent baseline. It fills a confirmed gap (we have GDN2 for forward decode + KV-rollback tree-verify for attention, but no GDN tree-verify), extends three existing systems (Plan 105 GDN2, Plan 012 DDTree, Plan 182 QwenDeltaNet), and is behind a feature flag with a GOAT gate.

**Novelty gate (§1.5):**
- Q1 (no prior art?): ✅ Grep confirmed zero hits for "gdn tree", "delta rule tree verify", "triangular solve" in speculative context across all repos. The paper itself frames this as an open problem (STree only handles diagonal recurrences).
- Q2 (new class of behavior?): ✗ No — faster verification of an existing capability (speculative tree decoding) on a specific layer type (GDN). Does not enable a new capability class.
- Q3 (product selling point?): ✗ No — "verify speculative trees on linear-attention models without rollback" is a perf claim, not a new capability. Cannot finish "our NPCs do X that no competitor can" with a new verb.
- Q4 (force multiplier?): Partial — connects GDN2 + DDTree + QwenDeltaNet (3 systems), but as a perf multiplier on the verify path, not a capability multiplier.

**Result: NO on Q2/Q3 → GOAT (not Super-GOAT).**

### 3.2 Gain verdict — acceptance ceiling insight

The §3.5 analysis (argmax-of-marginal beats full marginal at long draft depths) is a modelless DDTree tuning insight. It's useful (can improve acceptance length at deep tree levels) but incremental — a one-line change to `build_dd_tree`. No separate plan; captured here as a note for the DDTree module.

### 3.3 → riir-train redirect — Weaver adapter

**§3.5 modelless unblock protocol (MANDATORY before deferral):**

The Weaver adapter appears to need training. Checked all three modelless paths:

1. **Freeze/thaw snapshot correction** (path 1): ✗ The correction must be computed at runtime, conditioned on the **realized draft tokens** (which vary every step). A frozen snapshot is static; it cannot provide a context-dependent coupling correction.
2. **Raw/lora reader-writer hot-swap** (path 2): ✗ The correction is inherently **sequential/autoregressive** — each position's correction depends on all previously drafted tokens in the path. A deterministically-constructed LoRA applies a fixed linear overlay; it cannot model the data-dependent conditional dependency structure. The AC-Prefix lesson (Issue 003) showed systematic biases (e.g., "signal doubled") can be fixed modellessly, but Weaver's correction is not a systematic bias — it is a **context-dependent coupling restoration** that requires modeling the joint distribution over token sequences.
3. **Latent-space correction** (path 3): ✗ The §3.5 analysis proves that even an **oracle** marginal drafter has a structural ceiling. The correction needed is the full conditional dependency structure `p(x_{L+t} | x_{<L+t})` given only the prefix `p(x_{L+t} | x_{<L})` — this requires modeling the conditional, which is exactly what a learned autoregressive model does. A dot-product projection + sigmoid gate onto a direction vector cannot restore discrete-token conditional dependencies without modeling the joint.

**Direct negative evidence:** δ-mem (Plan 053) tried modelless delta-rule correction for DDTree and found **NO GAIN** ("26× latency overhead, corrections too small to flip branch ordering" — `.docs/09_feature_catalog/negative_results.md`). This is the exact failure mode: a modelless associative correction is too weak to restore the conditional coupling that Weaver learns via gradient descent.

**Conclusion: genuine riir-train dependency.** The Weaver adapter requires gradient descent to learn the conditional coupling between factorized marginals. Training recipe (LK loss + Muon + top-K constrained vocabulary projection + 300k completions) → riir-train. The modelless residue (the §3.4 algorithm + §3.5 insight) stays in katgpt-rs.

### MOAT gate (§1.6)

Domain: **katgpt-rs** (public engine). In scope: transformer stack (speculative decode, DDTree). The GDN tree-verify is a paper-derived fundamental primitive for the speculative decode stack. It passes GOAT via the 7.1× verify gain on the GDN2 backbone.

- **Per-stack slot:** speculative decode → verify path (for GDN/delta-rule layers). Currently this slot is occupied by "KV-cache snapshot/rollback" (Plan 012, attention-only). The GDN tree-verify is the delta-rule complement, not a replacement — they serve different layer types.
- **Promote/demote:** ship behind `gdn_tree_verify` feature flag (opt-in). GOAT gate (G1 correctness vs per-branch recurrent, G2 perf, G3 no-regression, G4 alloc-free). Promote to default if it wins on the QwenDeltaNet config.
- **Strengthens moat:** yes — the public engine gains a capability (GDN speculative tree decode) that competitors lack. Neutral for the private repos (no game/chain/shard IP implicated).

---

## 4. Implementation Notes (for Plan 424)

### 4.1 The algorithm on CPU SIMD

The paper targets B200/CUDA, but the algorithm is mathematical and maps to CPU SIMD:

1. **Ancestor bitmask:** for each node i, store a bitmask of its proper ancestors (packed into u64 words). Computed once per tree from parent pointers.
2. **Cumulative decay:** `aₜ = ∏_{i ⪯ t} αᵢ` computed in log-space (log-sum) along branches, then exponentiated. tf32x3 precision in the paper; f64 accumulation on CPU is exact enough.
3. **Interaction matrix X:** lower-triangular, ancestor-masked. Built per key head. Dense within the ancestor set, zero elsewhere.
4. **Forward substitution (Eq. 13):** tiled into Bc=32 sub-blocks. Each diagonal block `(I + X_bb)` inverted by repeated squaring (small matrix, in registers/L1). Off-diagonal cascade over sub-blocks. Maps to blocked SIMD matmul (`simd_matmul_rows`).
5. **Output stage (Eq. 11):** read S₀ once (decayed), apply ancestor-masked attention against the solve result.
6. **Commit:** single pass replaying the delta-rule recurrence (Eq. 7) along the accepted path. Only state write.

### 4.2 Integration points

- **GDN2 state (Plan 105):** the verify reads `Gdn2State` (the S₀ matrix) and layer params (K, V, Q, α, β). No modification to GDN2 itself — the verify is a read-only consumer.
- **DDTree (Plan 012):** the tree structure (parent pointers, token ids) comes from `TreeBuilder`. The verify replaces the KV-rollback path for GDN-layer configs.
- **QwenDeltaNet (Plan 182):** the natural consumer — `Config::qwen_deltanet()` has hybrid DeltaNet/Attention layers. The GDN layers use the new verify; the attention layers use the existing KV-rollback verify.

### 4.3 What stays open (not in this note's plan)

- **Traversal verification** (the acceptance coupling scheme from [10]) — the paper uses it but it's a separate algorithm. Our DDTree has its own verify; integrating Traversal is a follow-up.
- **Weaver on katgpt-rs** — blocked on riir-train producing trained Weaver weights. The architecture (top-K constrained residual adapter) could be wired as a `SpeculativeGenerator` implementation once weights exist, but that's a riir-ai runtime task.
- **GPU kernel** — katgpt-rs is CPU-first. The GPU fused kernel is a riir-gpu task if needed for production throughput.

---

## 5. References

- **Paper:** [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) — Oda, Mathieu, Knyazhitskiy, Chakhvadze, Jul 2026
- **Code:** [github.com/trymirai/sglang](https://github.com/trymirai/sglang) (SGLang fork)
- **Baselines:** DFlash [9] (arXiv:2602.06036), DDTree [11] (arXiv:2604.12989 — already distilled in Plan 012), STree [34] (arXiv:2505.14969 — diagonal recurrence tree verify), Traversal verification [10] (arXiv:2505.12398)
- **GDN backbone:** Gated Delta Networks [35] (arXiv:2412.06464), chunked delta rule [36] (arXiv:2406.06499)
- **Internal:** Plan 105 (GDN2), Plan 012 (DDTree + KV rollback), Plan 182 (QwenDeltaNet), Research 070 (GDN2 distillation), Research 378 (HOLA hippocampal KV), Plan 053 (δ-mem negative result)
