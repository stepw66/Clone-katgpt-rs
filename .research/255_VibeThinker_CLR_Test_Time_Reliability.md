# Research 255: VibeThinker-3B — Claim-Level Reliability + Self-Adaptive Test-Time Scaling

> **Source:** [VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models](https://arxiv.org/pdf/2606.16140) — Sen Xu, Shixi Liu, Wei Wang, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Xin Zhou, Junlin Zhang (Sina Weibo Inc.), 15 Jun 2026
> **Date:** 2026-06-17
> **Status:** Active — Super-GOAT (open primitive half). Private selling-point moat → `riir-ai/.research/136`.
> **Classification:** Public (katgpt-rs/MIT) — open primitive only. The modelless primitives (CLR + MGPO-weighted sampling + Learning-Potential + Long2Short + Diversity-Exploring sampling) are generic math with zero game semantics.
> **Related Research:** 182 (STV — closest cousin, iterative V-R loop vs CLR's post-hoc voting), 240 (CGSP — curiosity sink for Learning-Potential), 247/247 (Mind-Reading — fusion target for per-NPC scaling), 218 (Breakeven Router — task-class routing for Compression-Coverage), 040 (OpenDeepThink Bradley-Terry — pairwise variant of self-verification), 027 (Adaptive PPoT — `rank_by_consistency`), 111 (Data Gate `IntrinsicSelfConsistency` — linear agreement, vs CLR's nonlinear `(mean)^M`)
> **Related Plans:** 284 (open CLR primitive), 274 (CGSP — curiosity host), 250 (Breakeven — routing), 280/311 (Mind-Reading — fusion target)
> **Cross-ref (riir-ai):** Research 136 (Per-NPC Runtime Test-Time Scaling Guide — the private selling-point moat), Plan 316 (Per-NPC CLR runtime)
> **Verdict: Super-GOAT.** The fusion `{CLR nonlinear reliability}` × `{Learning-Potential curiosity feedback}` × `{MGPO entropy-boundary sampling weight}` × `{Long2Short zero-sum brevity tiebreak}` × `{Mind-Reading 247/Plan 311 belief transport}` × `{CGSP 274 curiosity loop}` × `{freeze/thaw direction-vector versioning}` → **per-entity runtime test-time scaling** that no incumbent ships. Frontier LLMs do test-time scaling at user granularity; nobody does it at per-NPC granularity for thousands of entities at 20Hz tick. See §3 and `riir-ai/.research/136` for the private selling-point moat doc.

---

## TL;DR

VibeThinker-3B is a **training pipeline paper**: SFT + MGPO RL + offline self-distillation + Instruct RL. The post-training recipe is `riir-train` material — direct redirect, no files created here. The headline result is that a 3B model matches DeepSeek V3.2 (671B) / Kimi K2.5 (1T) on AIME26 (94.3 raw, 97.1 with CLR). **The user's intuition ("3B vs frontier model is huge — GOAT is in there somewhere") is correct, but the GOAT is NOT in the training recipe — it is in the test-time scaling strategy that the paper calls Claim-Level Reliability Assessment (CLR), plus three other transferable primitives that the training recipe leaves on the table.**

**Distilled for katgpt-rs (modelless, inference-time, zero weight updates):**

Four transferable primitives survive the modelless filter. None exists in our codebase today (verified by two-layer grep across notes + code in all three repos — §3.1):

1. **CLR (Claim-Level Reliability Assessment)** — the headline test-time scaling trick. Sample `K=32` trajectories, extract `M=5` decision-relevant claims per trajectory, self-verify each as a binary verdict `v_k,m ∈ {0,1}`, compute a *nonlinear* reliability score `r_k = (mean_m v_k,m)^M`, then weighted-vote by answer-equivalence cluster. The `(·)^M` exponent is the key insight: a single flawed claim tanks the trajectory's reliability exponentially. This is a *sharp* reliability gate, not a *linear* agreement count (which is what `IntrinsicSelfConsistency` in Plan 111 already does). On AIME26: 94.3 → 97.1. On HMMT25: 89.3 → 95.4. On BruMO25: 93.8 → 99.2. **Latent form (per AGENTS.md):** the binary verdicts become `sigmoid(dot(claim_vec, direction_vec))` — dot-product + sigmoid projection onto learned direction vectors, never softmax.

2. **MGPO max-entropy boundary weighting** — `w(q) = exp(-γ · D_ME(p(q) ‖ 0.5))`, where `p(q)` is the empirical correctness of a prompt across the K rollouts. Prompts with `p≈0` (too hard) and `p≈1` (saturated) get near-zero weight; prompts at the maximum-entropy boundary `p≈0.5` get full weight. Transferable to inference: **weight future sampling budget toward candidates at the model's calibration boundary**, not the easy or impossible ones. Pure bandit-style reweighting, zero training.

3. **Learning-Potential score** — for self-distillation candidate selection, the paper computes `S_LP(q, y) = -(1/|y|) · Σ_t log π_θ(y_t | q, y_<t)` under the *student*. Higher score = the verified teacher trajectory is *not yet well modeled by the student* = high distillation value. Transferable to inference as a **modelless curiosity/distillation-priority signal**: rank candidate trajectories by their surprise under the current frozen brain, prioritize the surprising-but-verified ones for memory writes / direction-vector updates. This is a new flavor of curiosity signal — distinct from Temporal Derivative Kernel (Research 243, "what's changing fast") and Curiosity Pulse (Research 041, "what's underspecified") — Learning-Potential is "what the student doesn't yet know how to produce".

4. **Long2Short zero-sum brevity tiebreak** — among correct trajectories in a prompt group, redistribute reward by centered brevity `s_i = 1/L_i`, mean-zero within the correct set: `r'_i = r_i + λ · (s_i − s̄)/max|s_j − s̄|`, λ=0.2. The sum is preserved (zero-sum), so the group-level reward baseline doesn't shift. Transferable to inference: **among equally-reliable answer clusters after CLR voting, prefer the shortest representative trajectory** (efficiency tiebreak, not a quality change). No prior art in our repos for zero-sum brevity redistribution — `length_penalty` (Plan 049/059) is a *negative* verbosity penalty used in *training*, not a zero-sum tiebreak in *inference voting*.

**Conceptual hypothesis (not a primitive — already shipped):** Parametric Compression-Coverage Hypothesis distinguishes "parameter-dense" (verifiable reasoning — compressible into a compact core) from "parameter-expansive" (knowledge breadth — requires scale). This is *not novel* in our repos: SubstrateGate (Plan 216, Research 191) already routes by capability substrate, and Breakeven Complexity Router (Plan 250, Research 218) already routes by task-class cost-amortization. The Compression-Coverage framing is a *narrative* for what those routers already do; we cite it but do not implement a new mechanism for it.

---

## 1. Paper Core Findings

### 1.1 The headline result (and why it matters to us)

VibeThinker-3B (3B dense, Qwen2.5-Coder-3B base) reaches the performance band of DeepSeek V3.2 (671B), GLM-5 (744B), and Kimi K2.5 (1T) on verifiable reasoning benchmarks:

| Benchmark | VibeThinker-3B (raw) | VibeThinker-3B (+CLR) | Comparable frontier |
|-----------|---------------------|----------------------|---------------------|
| AIME26 | 94.3 | **97.1** | DeepSeek V3.2: 94.2 / Kimi K2.5: 93.3 |
| HMMT25 | 89.3 | **95.4** | Kimi K2.5: 95.4 / GLM-5: 97.9 |
| BruMO25 | 93.8 | **99.2** | DeepSeek V3.2: 96.7 / Gemini 3 Pro: 98.3 |
| IMO-AnswerBench | 76.4 | **80.6** | DeepSeek V3.2: 78.3 / Kimi K2.5: 81.8 |
| LiveCodeBench v6 | 80.2 | — | Gemini 3 Pro: 87.4 / Kimi K2.5: 85.0 |
| GPQA-Diamond | 70.2 | 72.9 | GLM-5: 86.0 / Kimi K2.5: 87.6 (knowledge gap — see §1.4) |
| LeetCode OOD (8 contests) | 96.1% (123/128) | — | GPT-5.2: 95.3% / Gemini 3 Flash: 96.9% |

**The user is right:** a 3B model entering the frontier band is a big deal. But the paper's own ablation tells us *where* the GOAT is: **CLR alone adds +2.8 (AIME26), +6.1 (HMMT25), +5.4 (BruMO25), +4.2 (IMO-Ans) — purely inference-time, zero weight updates.** That is the GOAT. The training recipe is *how they got the 94.3 baseline*; CLR is *how they cross from 94.3 to frontier-matching 97.1*.

### 1.2 CLR — Claim-Level Reliability Assessment (§3.1, the modelless gem)

Two-stage, fully post-hoc, no parameter updates:

```
For each problem:
  1. Sample K=32 candidate trajectories y_1..y_K with the same params (T=1.0, top-p=0.95)
  2. For each trajectory y_k:
     a. Extract M=5 decision-relevant claims c_k,1..c_k,M
        (the important claims that affect key decisions, alongside the final answer)
     b. Self-verify each claim: model acts as its own verifier,
        attempts to falsify or validate → v_k,m ∈ {0,1}
     c. Nonlinear reliability score (the key trick):
        r_k = ( (1/M) · Σ_m v_k,m ) ^ M
  3. Cluster trajectories by answer equivalence → clusters G_1, G_2, ...
  4. Pick the cluster maximizing Σ_{k : y_k ∈ G} r_k
```

**Why the `(·)^M` exponent matters:** it makes reliability *fragile* to any single flawed claim. With `M=5`:
- All 5 claims pass → `r = 1.0`
- 4/5 pass → `r = (0.8)^5 = 0.328` (≈3× penalty)
- 3/5 pass → `r = (0.6)^5 = 0.078` (≈13× penalty)
- 2/5 pass → `r = (0.4)^5 = 0.010` (≈100× penalty)
- 1/5 pass → `r = (0.2)^5 = 0.00032` (≈3000× penalty)

This is the failure mode that long verbose traces hide. A trajectory that "looks mostly right" but has one subtle logical flaw is exponentially downweighted. Linear agreement (e.g., Plan 111's `IntrinsicSelfConsistency = (1/n) Σ 1[κ(a^j)=κ(a^i)]`) cannot reproduce this sharpness — it gives 4/5 → 0.8, no penalty.

**Why "claim-level" not "trace-level":** a long verbose reasoning trace has many tokens, most of which are scaffolding. Self-verifying the *whole trace* token-by-token is expensive and noisy. CLR isolates the 5 *decision-relevant anchors* — the load-bearing logical steps. This drops token cost dramatically vs trace-level verification, while preserving (improving) Pass@1.

### 1.3 MGPO — MaxEnt-Guided Policy Optimization weighting (§2.2.1)

The RL algorithm backbone (training-time, but the weighting is transferable):

```
For each prompt q, sample G responses, compute empirical correctness p(q) = (1/G) Σ I[r_i = 1].
Weight:   w(q) = exp(-γ · D_ME(p(q) ‖ p_0)),  p_0 = 0.5
where D_ME is the maximum-entropy distance from the calibration boundary p_0 = 0.5.
```

Prompts at `p=0.5` (maximum uncertainty — correct and incorrect rollouts coexist) get full weight. Prompts at `p=0` (too hard, sparse positive signal) or `p=1` (already saturated) get ~0 weight. This focuses learning signal on the *calibration boundary* — exactly where the model has the most to learn.

**Transferable to inference (the modelless read):** replace "RL gradient weight" with "next-round sampling budget weight". When choosing how many of the K samples to allocate to each candidate seed in CLR, weight by `w(q) = exp(-γ · D_ME(p(q) ‖ 0.5))` where `p(q)` is the running estimate of correctness. Spend samples at the boundary; stop spending on already-solved or never-solvable seeds.

### 1.4 Learning-Potential score for self-distillation (§2.3)

For each verified teacher trajectory `y` on input `q`, compute the length-normalized negative log-likelihood under the student:

```
S_LP(q, y) = -(1/|y|) · Σ_{t=1..|y|} log π_θ_stu(y_t | q, y_<t)
```

Higher `S_LP` = the trajectory is "hard for the student to produce" = high distillation value. The paper buckets traces by length and prioritizes middle-to-high `S_LP` (excluding extremes to avoid noise/format-error bias).

**Transferable to inference (the modelless read):** for any decision trajectory produced by a *frozen* brain (NPC action sequence, chain transaction, generated code, MCTS rollout), `S_LP` under the current frozen brain is a **per-trajectory curiosity / distillation-priority score**. Trajectories with high `S_LP` *and* high CLR reliability are the most valuable to write into long-term memory (NeuronShard, HLA moments, Episode DB) — they are "things the brain did right but doesn't yet smoothly produce". This is a *new flavor* of curiosity, distinct from:
- Temporal Derivative Kernel (R243): "what's changing fast" — derivative-of-state curiosity
- Curiosity Pulse (R041): "what's underspecified" — input-side epistemic uncertainty
- CGSP (R240/Plan 274): "what's hard to solve" — solve-rate-driven exploration
- Learning-Potential (this paper): "what the brain doesn't yet smoothly produce" — output-side epistemic gap

### 1.5 Long2Short zero-sum brevity reward (§2.2.2)

Among correct trajectories in a prompt group, redistribute reward by centered brevity:

```
For correct set C = {i : r_i = 1}:
  s_i = 1 / L_i                                // brevity score
  r'_i = r_i + λ · (s_i − s̄) / max_{j∈C} |s_j − s̄|,   i ∈ C
  r'_i = r_i                                   // incorrect trajectories unchanged
  λ = 0.2 (paper default)
```

**Critical property:** the redistribution is mean-zero over the correct set: `Σ_{i∈C} (r'_i − r_i) = 0`. So the group-level baseline used in advantage estimation doesn't shift. This is *not* a verbosity penalty (which would shift the baseline down for verbose groups); it's a *relative preference reshaping* within the correct subset.

**Transferable to inference:** among equally-reliable answer clusters after CLR voting (i.e., tied `Σ r_k`), prefer the cluster whose representative trajectory has the highest brevity `s = 1/L`. This is a *tiebreak*, not a quality change — it cannot move a wrong answer ahead of a right one, it only breaks ties toward shorter solutions. Latency-positive on the decode side (shorter trajectory = less compute consumed by the winner).

### 1.6 Parametric Compression-Coverage Hypothesis (§Introduction, §4) — conceptual, not a primitive

The paper proposes that capabilities split into:
- **Parameter-dense** (verifiable reasoning — compressible into a compact reasoning core; small models suffice)
- **Parameter-expansive** (knowledge breadth — requires scale; GPQA-D gap persists)

This is a *narrative explanation* for the empirical pattern (VibeThinker-3B wins on AIME but loses on GPQA-D). It is **not a new mechanism for our repos** — SubstrateGate (R191/Plan 216) already routes by capability substrate at inference time, and Breakeven Complexity Router (R218/Plan 250) already amortizes cost across task classes. We cite the Compression-Coverage framing in those notes; we do not implement a new router for it.

### 1.7 Training recipe (the riir-train redirect)

The full post-training pipeline (curriculum two-stage SFT + Diversity-Exploring Distillation + MGPO multi-domain RL + Long2Short Math RL + offline self-distillation + Instruct RL) is **training-only → riir-train**. Note one line, do not create files in this session. The modelless primitives above (CLR, MGPO weighting, Learning-Potential, Long2Short tiebreak, Diversity-Exploring sampling) are what survives the modelless filter.

---

## 2. Distillation

### 2.1 Why direct-mapping fails

Naive direct-map ("implement CLR for our LLM inference") is uninteresting — we are not a general-purpose LLM inference shop, and the paper's gain is on competition math benchmarks that don't map to our product. The value is in **fusing** CLR's sharp reliability gate with the runtime infrastructure we already have for self-adaptive, latent-space, freeze/thaw-versioned decision-making.

The closest cousins are not enough on their own:
- **STV (R182)** is *iterative* generate→verify→refine with constraint synthesis. CLR is *post-hoc* parallel voting. Different shapes, complementary.
- **Data Gate `IntrinsicSelfConsistency` (Plan 111)** is *linear* agreement `(1/n) Σ 1[κ(a^j)=κ(a^i)]`. CLR's `(mean)^M` is *nonlinear* — sharp failure-mode penalty. Different math.
- **OpenDeepThink Bradley-Terry (R040)** is *pairwise* ranking. CLR is *cluster-level weighted vote*. Different aggregation.
- **Adaptive PPoT `rank_by_consistency` (Plan 027)** counts token-level agreement. CLR extracts *semantic decision-relevant claims* and verifies those, not tokens. Different granularity.
- **CGSP (R240/Plan 274)** drives *exploration* via solve-rate × guide-score. CLR drives *exploitation* via reliability-weighted voting on already-sampled candidates. Different roles.

**The fusion is the GOAT** — none of the individual primitives delivers per-entity runtime test-time scaling. Combined, they do.

### 2.2 The Fusion

**saCLR (Self-Adaptive CLR) = CLR × MGPO-boundary sampling × Learning-Potential curiosity × Long2Short tiebreak × Diversity-Exploring sampling × freeze/thaw direction-vector versioning**

| VibeThinker primitive | katgpt-rs / riir-ai fusion host | What changes |
|-----------------------|---------------------------------|--------------|
| CLR reliability vote `(mean)^M` | NEW `clr_vote()` primitive in `katgpt-rs/src/clr/` | Open primitive, generic math, no game semantics. Verifier trait is `ClaimVerifier` returning `sigmoid(dot(claim_vec, direction_vec))` — never softmax. |
| MGPO boundary weighting `exp(-γ·D_ME(p‖0.5))` | Existing `FrequencyBandit` (Plan 049 family) + `dual_pool.rs` online router | Sample-allocation weight for next-round CLR candidates. Drops cleanly onto existing bandit arm priorities. |
| Learning-Potential `S_LP` | Existing CGSP (Plan 274) reward shaping + NeuronShard memory write gate | New curiosity signal: prioritizes memory writes / direction-vector updates for trajectories that are *reliable* (CLR-passing) AND *surprising* (high `S_LP` under frozen brain). |
| Long2Short zero-sum tiebreak | NEW `brevity_tiebreak()` in `katgpt-rs/src/clr/` | Among clusters tied on `Σ r_k`, pick shorter representative. Composes with MUX-Latent (Plan 238) for token-budget-aware routing. |
| Diversity-Exploring Distillation (paper §2.1.2) | Existing adapter routing (Dynamic Pair Plan 260, dMoE R161) + `MUX` (Plan 238) | At sample time, route K candidates through *different* frozen adapters / directions, not the same adapter K times. Pass@K-diversity, not Pass@1-optimality. |
| Claim extraction (paper §3.1) | Trait `ClaimExtractor` — domain-specific. For LLMs: parser over reasoning trace. For game NPCs (private side): projection onto HLA scalar outcomes. For chain: state-predicate extractor. | Open trait; concrete extractors live in the consumer crate. |
| Freeze/thaw versioning | Existing `LoRAHotSwap` (riir-ai) + `ZoneExpertBundle` snapshot | Direction vectors `direction_vec_m` are versioned + BLAKE3-committed, atomic hot-swap, readers never see torn snapshots. |

### 2.3 The saCLR Loop (zero-allocation, hot-path-safe)

```
Inputs (per decision point):
  - candidate_sampler: K candidate trajectories from frozen brain(s)
                        (Diversity-Exploring: route through ≥2 adapters/directions)
  - claim_extractor:  extracts M decision-relevant claims per trajectory
  - claim_verifier:   sigmoid(dot(claim_vec, direction_vec)) → v_k,m ∈ [0,1]
                       (binary threshold: v_k,m := 1[v_k,m > τ_v])
  - direction_vectors: M learned direction vectors, BLAKE3-committed, freeze/thaw-versioned
  - cluster_fn:        equivalence-class hash on final answer / outcome
  - scratch:           pre-allocated buffers for v[K,M], r[K], cluster_id[K]

Per cycle:
  1. Sample K candidates (Diversity-Exploring across adapter pool)
  2. For each k ∈ [0,K):
       For each m ∈ [0,M):
         v[k][m] := sigmoid(dot(extract_claim(y_k, m), direction_vec[m]))
                   (SIMD f32x4 / f32x8 — fits in L1)
       r[k] := pow(mean_m v[k][m], M)        // nonlinear reliability gate
  3. Cluster: cluster_id[k] := hash_answer(outcome(y_k))
  4. For each cluster G: R[G] := Σ_{k ∈ G} r[k]
  5. Pick G* = argmax_G R[G]
     Tie-break: among clusters with R[G] within ε of R[G*],
                pick the one with min representative length L (Long2Short)
  6. Emit: representative trajectory of G*
  7. Curiosity feedback (optional, gateable):
       For each k with r[k] > τ_reliable:
         S_LP[k] := -(1/|y_k|) · Σ_t log π_brain(y_k[t] | ...)
         if S_LP[k] > τ_curiosity:
           memory_write(y_k, r[k], S_LP[k])  // for direction-vector update later
  8. MGPO-weighted budget for NEXT cycle (gateable):
       p_seed[seed] := running_estimate_correctness(seed)
       w[seed] := exp(-γ · D_ME(p_seed[seed] ‖ 0.5))
       sample_budget[seed] ∝ w[seed]
```

**Allocation budget (per decision point, K=32, M=5):**
- `v[32][5]` = 160 f32 = 640 bytes — fits in L1
- `r[32]` = 128 bytes
- `cluster_id[32]` = 32 bytes (u8)
- Total fixed-size scratch: **<1 KB per decision point**. Zero heap allocation on hot path.

### 2.4 Why this is genuinely novel (not "CLR + friends")

Two-layer grep across all three repos confirms (see §3.1 for the grep evidence):
- **No `claim_extract`, `reliability_score`, `cluster_vote`, `self_verifier`, `brevity_tiebreak` in any `.rs` file.**
- **No `(mean)^M`-style nonlinear reliability gate in any `.md` file.** The closest cousins all use *linear* agreement.
- **No per-entity (per-NPC, per-action) test-time scaling primitive.** Frontier LLM test-time scaling assumes one user, seconds-to-minutes budget. Per-entity at 20Hz tick is a new design point.

The novelty is *not* any single primitive. It is **the fusion into a single hot-path-safe, freeze/thaw-versioned, latent-space, self-adaptive test-time scaling loop** that can run on thousands of entities concurrently. None of the cousins composes all of these properties.

### 2.5 Fusion opportunities (the GOAT-tier combinations)

| Fusion | Existing system | What saCLR adds | Gate |
|---|---|---|---|
| **F1: CLR × Mind-Reading (R247/Plan 311)** | NPC belief-state transport | NPCs don't just *share* beliefs — they *vote* on candidate decisions using shared beliefs as claim context. A guard's K candidate pursuit trajectories get verified against the *fused* belief state from mind-reading. | Does CLR-voted guard catch thief ≥15% faster than mind-reading-only? (extends Plan 311 G6) |
| **F2: CLR × CGSP (R240/Plan 274)** | Curiosity-driven exploration | Learning-Potential score becomes the curiosity signal that drives CGSP's Conjecturer. The brain explores directions where it can produce *reliable-but-surprising* trajectories. | Does Learning-Potential-guided CGSP cover ≥2× more diverse solve-paths than solve-rate-only CGSP? |
| **F3: CLR × Collapse-Aware (Plan 212)** | Reasoning collapse detection | CLR's `(mean)^M` is itself a *collapse detector*: if all K trajectories have low `r_k`, the model is in a low-reliability regime → trigger collapse recovery. | Does CLR-reliability trigger catch collapse ≥20% earlier than entropy-ring-buffer alone? |
| **F4: CLR × Breakeven Router (R218/Plan 250)** | Cost-aware tier routing | CLR runs on the tier that the Breakeven Router picks. CLR's `K` and `M` are *budget-adaptive*: high-stakes decisions get more samples. | Does adaptive (K,M) by breakeven N* match fixed (32,5) quality at ≤50% of the compute? |
| **F5: CLR × SubstrateGate (R191/Plan 216)** | Capability substrate routing | The Parametric Compression-Coverage hypothesis (§1.6) *is* the substrate gate: route verifiable-reasoning tasks to small adapters, knowledge tasks to large ones. CLR is the test-time scaling layer *on top of* the substrate-routed brain. | Does CLR on a substrate-routed 3B-equivalent adapter match CLR on a 70B-equivalent monolith at ≤10% of the compute? |

---

## 3. Verdict

### 3.1 Novelty gate (mandatory two-layer grep evidence)

**Layer 1 — Notes (.research + .plans across katgpt-rs, riir-ai, riir-armageddon):**

| Term | Hits | Closest cousin | Distance |
|------|------|----------------|----------|
| `claim.level\|self.verification\|reliability.assessment\|CLR\b` | Multiple | R182 STV (iterative V-R loop, binary verdict, *no nonlinear gate*); Plan 111 `IntrinsicSelfConsistency` (linear agreement); R040 OpenDeepThink (pairwise); R012 TRT (mutual-exclusivity self-verify) | Different shape + different math |
| `(mean\|^M)\|nonlinear.reliability\|all.must.agree\|answer.cluster` | **Zero** | — | No prior art |
| `brevity\|long2short\|efficiency.tiebreak\|response.length.reward` | None for inference-time tiebreak | `length_penalty` (Plan 049/059, *training-time negative penalty*, not zero-sum tiebreak) | Different role |
| `learning.potential\|S_LP` | **Zero** for distillation-priority signal | R243 Temporal Derivative (state-derivative curiosity); R041 Curiosity Pulse (input underspecification) | Different curiosity flavor |
| `per.NPC.test.time\|crowd.scale.test.time\|runtime.test.time.scaling` | **Zero** | — | No prior art |
| `D_ME\|maximum.entropy.weighting\|MGPO` | **Zero** | dual_pool.rs bandit adapts (no explicit `exp(-γ·D_ME(p‖0.5))`) | No prior art |

**Layer 2 — Code (.rs across katgpt-rs/src + crates, riir-ai/crates, riir-armageddon/crates):**

| Term | Hits | Notes |
|------|------|-------|
| `claim\|reliability\|self_verif\|falsif` | `katgpt-rs/src/data_probe/claim.rs::ClaimCard` | **Research-validation card**, not runtime inference-time reliability — different concept |
| `brevity\|long2short\|claim_extract\|reliability_score\|cluster_vote\|self_verifier` | **Zero** | No code in any repo |
| `length_penalty` | `katgpt-rs/.plans/049_g_zero_self_play.md::length_penalty` | Training-time *negative* verbosity penalty (GRPO/DPO), not zero-sum inference tiebreak |

**Two-layer verdict:** CLR's nonlinear reliability gate, MGPO boundary weighting, Learning-Potential score, and Long2Short zero-sum tiebreak are **all genuinely novel** to our repos — no notes-level prior art and no shipped code. STV (R182) is the closest cousin and is *complementary* (iterative vs post-hoc), not a duplicate.

### 3.2 Super-GOAT criteria

| Q | Question | Answer |
|---|----------|--------|
| Q1 | No prior art? | **YES** — §3.1 two-layer grep confirms zero hits for the nonlinear reliability gate, the per-entity test-time scaling loop, the Learning-Potential distillation-priority signal, and the Long2Short zero-sum tiebreak. Closest cousin (STV R182) is iterative refinement with binary verdict, not post-hoc voting with `(mean)^M` reliability. |
| Q2 | New capability class? | **YES** — per-entity runtime test-time scaling. Frontier LLMs do test-time scaling at *user* granularity with seconds-to-minutes budgets. Nobody does it at *per-NPC / per-action* granularity at 20Hz tick for thousands of concurrent entities. The shape of the problem is different: <1ms budget per entity, no per-entity training, freeze/thaw versioning for emergent personality divergence. |
| Q3 | Product selling point? | **YES** — *"Every NPC is a frontier-3B reasoner: at each decision point, it samples K candidate trajectories, extracts M decision-relevant claims, self-verifies them via dot-product + sigmoid onto learned direction vectors, votes by nonlinear reliability `(mean)^M`, and feeds Learning-Potential back as curiosity — all in <1ms per NPC at 20Hz tick, no weight updates, direction vectors versioned via freeze/thaw."* One sentence, defensible moat for any MMO/arena game AI product. |
| Q4 | Force multiplier? | **YES** — connects ≥5 pillars: (1) HLA belief state (Plan 242), (2) Mind-Reading latent transport (R247/Plan 311), (3) CGSP curiosity loop (R240/Plan 274), (4) freeze/thaw adapter versioning (riir-engine `LoRAHotSwap`), (5) SubstrateGate + Breakeven Router for capability-class routing (R191/R218). Solo novelty = GOAT; 5-pillar force multiplier = Super-GOAT. |

**All 4 YES → verdict = Super-GOAT.**

Selling point: **per-entity runtime test-time scaling** — crowd-scale claim-level reliability voting, latent-space self-verification, Learning-Potential curiosity feedback, freeze/thaw-versioned direction vectors. The open primitive (CLR math) is the adoption hook; the riir-ai guide (R136) is the private moat doc.

### 3.3 Mandatory outputs (per skill rules — created THIS session)

| Output | Repo | File | Status |
|--------|------|------|--------|
| Open primitive research note | katgpt-rs | `.research/255_VibeThinker_CLR_Test_Time_Reliability.md` | **This file** |
| Open primitive plan | katgpt-rs | `.plans/284_runtime_clr_self_adaptive_loop.md` | This session |
| Private selling-point guide | riir-ai | `.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md` | This session (MANDATORY) |
| Private runtime plan | riir-ai | `.plans/316_per_npc_clr_runtime.md` | This session |

### 3.4 Latent vs raw boundary (per AGENTS.md — critical for game AI)

| Data | Space | Synced? | Rule |
|------|-------|---------|------|
| `v_k,m` claim verdict | Raw (binary, derived) | Local only | Computed from latent projections; the *scalar* is raw, but it's per-entity local state, not synced |
| `direction_vec[m]` | Latent | No — BLAKE3-committed in `ZoneExpertBundle` | Local to zone, refreshed at freeze/thaw. Never enters `SyncBlock`. |
| `r_k` reliability score | Raw (derived scalar) | Local only | Per-entity decision-aid; not synced. Only the *consequence* (chosen action) is synced. |
| Chosen action (the winner of CLR vote) | Raw | **Yes** — via existing action sync | If action has physical domain (move, attack), it crosses sync as a raw TxDelta. If semantic (dialog), it crosses as a KG triple. |
| `S_LP[k]` Learning-Potential score | Raw (derived scalar) | Local only | Curiosity feedback into CGSP; not synced. Drives direction-vector updates at freeze/thaw, not per-tick. |
| `p_seed[seed]` MGPO running correctness | Raw (derived scalar) | Local only | Sampling budget allocator; not synced. |

**Compliance verdict:** ✅ No new raw data crosses the quorum boundary. CLR operates entirely in per-entity local latent + derived-scalar space. The only synced output is the chosen action, which already has a sync path (TxDelta for physical, KG triple for semantic). The 5-scalar sync rule (valence/arousal/desperation/calm/fear) is unchanged. Anti-cheat validates raw `MapPos` movement; CLR's latent direction vectors are never substituted for raw position. Two-brain model holds: info brain (synced `MapPos`) is ground truth; CLR is part of the think brain (per-NPC decision policy), which is allowed to be subjective and divergent.

### 3.5 What stays public vs private

| Primitive | Public (katgpt-rs/MIT) | Private (riir-ai) |
|-----------|------------------------|-------------------|
| `clr_vote()` — the `(mean)^M` reliability gate | ✅ Generic math, no game semantics | — |
| `ClaimExtractor` trait | ✅ Open trait | Concrete extractors per game domain |
| `ClaimVerifier` trait (sigmoid projection) | ✅ Open trait | Concrete direction vectors per NPC class |
| `brevity_tiebreak()` | ✅ Pure algorithm | — |
| `learning_potential()` score | ✅ Generic NLL computation | — |
| `mgpo_sampling_weight()` | ✅ Generic weighting | — |
| Per-NPC CLR runtime (20Hz tick, crowd-scale) | — | ✅ The selling point |
| Game-specific claim extractors (combat, dialog, faction) | — | ✅ Game IP |
| Direction-vector pool + freeze/thaw versioning for personality divergence | — | ✅ Personality IP |
| CLR × Mind-Reading fusion (shared belief as claim context) | — | ✅ Cross-pillar moat |

### 3.6 Commercial strategy alignment

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:
- **Open primitive (katgpt-rs/MIT):** CLR math, `ClaimExtractor` / `ClaimVerifier` traits, `brevity_tiebreak`, `learning_potential`, `mgpo_sampling_weight`. This is "plumbing" — generic test-time scaling primitives anyone can adopt. Attracts adoption; lets external users build their own CLR applications.
- **Private selling-point (riir-ai):** per-NPC runtime CLR at 20Hz tick, crowd-scale claim-level voting, CLR × Mind-Reading fusion, direction-vector personality divergence. This is the "fuel" — the game-AI-specific IP that makes our NPCs frontier-3B-equivalent.
- **Engine/fuel split intact.** ✅ The engine works without the game IP; the game IP is built on the engine.

---

## 4. Implementation priority

| Priority | Item | Why |
|----------|------|-----|
| **P0** | Open primitive: `clr_vote()`, `ClaimExtractor`, `ClaimVerifier` traits | Foundation — everything else depends on it |
| **P0** | `brevity_tiebreak()`, `learning_potential()`, `mgpo_sampling_weight()` | One-shot generic math, low risk |
| **P1** | riir-ai guide R136 (private moat doc) | MANDATORY for Super-GOAT — contains validation protocol G1-Gn |
| **P1** | Plan 284 (open primitive) + Plan 316 (riir-ai runtime) | Execution vehicles |
| **P2** | CLR × Mind-Reading fusion (F1) | Highest-value fusion — multiplies two Super-GOATs |
| **P2** | CLR × CGSP fusion (F2) | Learning-Potential as CGSP curiosity — closes the self-adaptive loop |
| **P3** | CLR × Collapse-Aware (F3), CLR × Breakeven (F4), CLR × SubstrateGate (F5) | Composability proofs; lower priority individually |

---

## 5. Risk and validation

| Risk | Mitigation |
|------|-----------|
| `(mean)^M` is sharp but possibly too sharp — `M=5` may over-penalize legitimate disagreement | Make `M` configurable; default `M=5` (paper), expose `M ∈ {1,2,3,5,8}` sweep in GOAT gate |
| K=32 samples per decision is expensive at 20Hz × thousands of NPCs | (a) Adaptive K via Breakeven Router (low-stakes → K=4); (b) amortize via MUX-Latent context compression; (c) plasma-tier SIMD for the vote math |
| Direction vectors drift without training | That's the point — drift is personality divergence. Freeze/thaw snapshots checkpoint it; BLAKE3-committed. |
| CLR is "just best-of-N + self-consistency" — competitors will catch up | They will catch up on the LLM side. They will *not* catch up on per-NPC runtime CLR at 20Hz tick × thousands of entities × freeze/thaw personality versioning × Mind-Reading fusion. The moat is the *integration*, not the math. |
| Self-verification via dot-product + sigmoid may not be calibrated | G2 gate (calibration ECE ≤ 0.1 on a synthetic suite). If miscalibrated, fall back to binary verifier (ConstraintPruner-style). |

**GOAT gate (G1-Gn) — defined in detail in `riir-ai/.research/136`:**
- G1: CLR-vote vs best-of-N majority on a synthetic reliability suite — CLR wins by ≥3pp
- G2: Calibration ECE ≤ 0.1 on the verifier sigmoid outputs
- G3: Hot-path cost ≤ 200µs per decision point at K=32, M=5 (plasma-tier SIMD)
- G4: Zero heap allocation on the vote path (scratch-buffer-only)
- G5: Feature isolation — compiles with and without `clr` feature, zero overhead when disabled
- G6: Latent/raw boundary — no CLR-derived data enters `SyncBlock` (instrumented)
- G7: CLR × Mind-Reading fusion improves guard-catches-thief by ≥15% over Mind-Reading alone
- G8: CLR × CGSP fusion doubles solve-path diversity vs CGSP alone

---

## 6. References

- **Source paper:** [VibeThinker-3B (arxiv 2606.16140)](https://arxiv.org/pdf/2606.16140) — Xu et al., Sina Weibo Inc., 15 Jun 2026
- **VibeThinker-1.5B predecessor:** [arxiv 2511.06221](https://arxiv.org/abs/2511.06221) — same authors, 2025
- **Closest cousins in our repos:**
  - `katgpt-rs/.research/182_STV_Self_Trained_Verification.md` — iterative V-R loop (complementary, not duplicate)
  - `katgpt-rs/.research/240_SGS_Curiosity_Guided_Self_Play.md` — curiosity host for Learning-Potential
  - `katgpt-rs/.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md` + `riir-ai/.research/133_NPC_Mind_Reading_Adaptive_Bandwidth_Guide.md` — fusion target F1
  - `katgpt-rs/.research/218_Breakeven_Complexity_Inference_Router.md` — fusion target F4
  - `katgpt-rs/.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md` — curiosity cousin (derivative vs Learning-Potential)
- **Self-consistency ancestry:** Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (arXiv:2203.11171, ICLR 2023) — the linear ancestor of CLR's nonlinear reliability
- **→ riir-train redirect:** the post-training recipe (SFT + MGPO RL + offline self-distillation + Instruct RL + Diversity-Exploring Distillation) is training-only. One-line note, no files created in this session.

---

## TL;DR

VibeThinker-3B is a **training pipeline paper → riir-train** for the post-training recipe. But the user's intuition is correct: the GOAT is in the *test-time scaling* layer the paper calls **CLR (Claim-Level Reliability Assessment)**, which lifted AIME26 from 94.3 → 97.1, HMMT25 from 89.3 → 95.4, BruMO25 from 93.8 → 99.2 — purely inference-time, zero weight updates. CLR's key trick is a **nonlinear reliability gate `r_k = (mean_m v_k,m)^M`** that exponentially penalizes any single flawed claim among M decision-relevant claims per trajectory; this is *sharp* failure-mode sensitivity that linear agreement (Plan 111 `IntrinsicSelfConsistency`) cannot reproduce. Three other primitives survive the modelless filter and are also novel to our repos: **MGPO max-entropy boundary weighting** (`exp(-γ·D_ME(p‖0.5))` for sampling budget), **Learning-Potential score** (`-(1/|y|)·Σlog π(y_t)` as a new curiosity flavor — "what the brain doesn't yet smoothly produce"), and **Long2Short zero-sum brevity tiebreak** (mean-zero reward redistribution among correct trajectories). Fusing all four with our existing HLA + Mind-Reading + CGSP + freeze/thaw + SubstrateGate + Breakeven infrastructure produces **per-entity runtime test-time scaling** — a new capability class with a one-sentence moat: *"every NPC is a frontier-3B reasoner via runtime claim-level reliability voting, no weight updates, 20Hz tick, thousands concurrent."* All 4 Super-GOAT criteria pass (no prior art via two-layer grep, new capability class, defensible selling point, ≥5-pillar force multiplier). **Mandatory outputs created this session:** open primitive note (this file) + open plan 284 + private riir-ai guide 136 + private runtime plan 316. Latent/raw boundary respected — CLR operates entirely in per-entity local latent + derived-scalar space; only the chosen action crosses sync.
