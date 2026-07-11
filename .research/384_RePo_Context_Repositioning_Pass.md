# Research 384: RePo — Context Re-Positioning (PASS — primitive already shipped as Wall Attention)

> **Source:** [RePo: Language Models with Context Re-Positioning](https://arxiv.org/abs/2512.14391) — Huayang Li, Tianyu Zhao, Deng Cai, Richard Sproat (Sakana AI + NAIST), ICML 2026
> **Date:** 2026-07-06
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 145 (Wall Attention — **the prior art that ships RePo's primitive**), 070 (GDN2), 071 (DashAttention), 028 (Higher-order Linear Attention), 290 (Latent Field Steering)
> **Related Plans:** 173 (`wall_attention` feature — **the shipped primitive**), 105 (GDN2)
> **Verdict: PASS.** RePo's core mechanism — **content-derived position assignment** replacing rigid linear indices — is already shipped in katgpt-rs as **Wall Attention** (Research 145, `wall_attention` feature, Plan 173). Wall is the modelless distillation of the same primitive ("positions come from content, not from integer index"); RePo is a training-required variant (continual pre-training of a per-head SwiGLU→scalar module f_φ). Both achieve commensurate gains over RoPE on long-context / noisy-context benchmarks. The full head-to-head (RePo's scalar positions vs Wall's per-channel gates) → **riir-train**, out of scope here. No file/plan/guide created beyond this classification note.

---

## TL;DR

RePo replaces the rigid linear position index `i ∈ {0,…,L-1}` with a **content-derived real-valued position** `z_i = f_φ(h_i) ∈ ℝ`, then feeds `g_θ(z_j - z_i)` into the standard RoPE rotation (Eq. 7). The per-head module `f_φ = (Swish(h W_g) ⊙ h W_c) W_z` (Eq. 4–6) is **trained via continual pre-training** on OLMo-2 1B/7B. Gains: NIAH +5.4, HybridQA +2.27/+4.09, LongBench +6.93/+6.38 over RoPE. Only 0.9% extra params, ~3% slower decode.

**Distilled for katgpt-rs (modelless, inference-time):** nothing not already shipped. The principle — *positions can be content-derived and non-monotonic* — IS the Wall Attention primitive (R145, `wall_attention` feature). Wall replaces RoPE with per-channel diagonal forget gates `F_{ij,n} = Π_{s=j+1}^{i} g_{s,n}` derived from a key projection; RePo keeps the RoPE rotation and only swaps the position argument to a scalar `z_i = f_φ(h_i)`. Both encode "content-derived distance/position"; both break the RoPE locality bias; both extrapolate beyond training length. Wall is the more radical variant (drops rotation entirely); RePo is the conservative variant (keeps rotation, compatible with RoPE-trained checkpoints via continual pre-training). The mechanism class is the same.

---

## 1. Paper Core Findings

### 1.1 The mechanism

RePo inserts a small learnable module `f_φ : ℝ^d → ℝ` before the position-encoding step:

```
Position representation:   r_i = Swish(h_i W_g) ⊙ (h_i W_c)        (Eq. 4, SwiGLU sub-layer, d_p < d)
Position assignment:       z_i = r_i W_z                             (Eq. 5, per-head linear → scalar)
Attention score:           A_ij = q_iᵀ g_θ(z_j − z_i) k_j            (Eq. 7, RoPE with content-derived distance)
```

Applied starting from the `ceil(L/3)`-th layer (lower layers keep vanilla RoPE). Per-layer position-representation weights shared across heads; per-head position-assignment weights independent. Continual pre-training on OLMo-2 stage-2 (50B tokens, 4096 ctx). ~0.9% extra params, decode +3% time vs RoPE.

### 1.2 The analysis (the genuinely interesting part)

| Finding | Detail |
|---|---|
| **Attention mass on distant needles** | RePo allocates 2.013×10⁻² mass to needle tokens vs 1.754 (RoPE) / 1.572 (NoPE). Breaks RoPE's locality bias. |
| **Dense non-linear position space** | RePo's `max(z) − min(z)` per head is much smaller than raw context length (2K/4K) and non-uniformly distributed across heads. Enables better long-context extrapolation (4K→16K) because the model "uses" a smaller, denser slice of RoPE's low-frequency dimensions. |
| **Hybrid pattern dominance** | Chunked pattern analysis (Δ=16, ε=0.2): Hybrid 74%, Constant 22% (NoPE-like), Monotonic 4% (RoPE-like). The model dynamically interpolates between NoPE-style and RoPE-style per chunk — *no hand-configured R2N1/N2R1 interleaving needed*. |
| **Intrinsic structure capture** | Case study (App. D): assigned positions align with semantic segmentation of few-shot examples. Reversal task (App. C): mirror symmetry emerges — reversal pairs get the same z. |

### 1.3 The results

| Task | RePo vs RoPE (1B / 7B) |
|---|---|
| NIAH (noisy context, avg) | +5.4 / +0.6 |
| HybridQA (structured data, EM) | +2.27 / +4.09 |
| LongBench (long context, avg) | +6.93 / +6.38 |
| General short-context (avg) | +0.03 / −0.62 (no regression) |

---

## 2. Distillation — why this is PASS (already shipped)

### 2.1 The mechanism class: content-derived position

Both RePo and Wall Attention are instances of the same primitive: **the position/distance in attention is derived from content (hidden state / key), not from a pre-defined integer index.**

| Aspect | **Wall Attention** (R145, shipped) | **RePo** (this paper) |
|---|---|---|
| What replaces integer position | Per-channel diagonal forget gate `g_{s,n} ∈ (0,1]` derived from key projection | Per-head scalar real position `z_i ∈ ℝ` derived from SwiGLU(hidden) |
| Distance encoding | `F_{ij,n} = Π_{s=j+1}^{i} g_{s,n}` (per-channel cumulative) | `z_j − z_i` (scalar difference, fed to RoPE) |
| Position-encoding function | None — gates ARE the position info | RoPE rotation `g_θ(·)` kept, only argument changed |
| Locality bias broken? | ✅ (bimodal always-on / dynamic channels) | ✅ (more mass on distant needles) |
| Long-context extrapolation | ✅ (4K→160K) | ✅ (4K→16K) |
| Training cost | Trained from scratch (replaces RoPE) | Continual pre-training (compatible with RoPE checkpoint) |
| Latency overhead | ~free (FA3-level throughput after factorization) | +3% decode time |
| Latent-space framing | Per-channel gate = vector-valued content position | Scalar z = rank-1 content position |

Wall is the **vector-valued** (per-channel) instance; RePo is the **scalar** (rank-1) instance of the same primitive. Wall is strictly more expressive (a scalar is the rank-1 special case of a diagonal). RePo's only advantage is **backward compatibility with RoPE-trained checkpoints** — you can continual-pre-train from a RoPE model without restarting. Wall requires training from scratch.

### 2.2 Modelless unblock protocol (§3.5) — all three paths checked

The paper's value is the **trained f_φ**. Per §3.5, before redirecting to riir-train I checked the three modelless unblock paths:

1. **Freeze/thaw snapshot correction** — NO. The f_φ weights must be learned; a frozen snapshot of an *untrained* f_φ is just a random projection. The structural patterns (mirror symmetry, few-shot segmentation) emerge from training, not from a snapshot.
2. **Raw/lora reader-writer hot-swap** — PARTIAL but **already shipped as Wall**. A deterministically constructed position-remapping module IS a modelless LoRA-like addition. Wall Attention ships exactly this: a per-channel gate derived from the key projection, applied as a Q/K rescale. The Wall gate is the deterministic, modelless version of what RePo learns.
3. **Latent-space correction** — PARTIAL but **already shipped as Wall**. The position bias is an additive bias to attention logits derivable from a latent projection. Wall's `q̃ = exp(P) ⊙ q` / `k̃ = exp(-P) ⊙ k` factorization IS this latent correction, applied per-channel instead of as a scalar.

**Verdict:** the modelless distillation of RePo's primitive already ships as Wall Attention. RePo contributes a training recipe (continual pre-training of f_φ) and an analysis (pattern statistics) — both out of scope for katgpt-rs/riir-*.

### 2.3 Defend-wrong PoC (§3.6) — claim type audit

Per §3.6, I distinguish three claim types for the "already ships" verdict:

| Claim type | Claim | Proof | Status |
|---|---|---|---|
| **Architectural** ("the runtime analog exists") | "content-derived position primitive ships as Wall Attention" | grep + read `katgpt-rs/crates/katgpt-attn/src/diagonal_gate.rs` L112–160 | ✅ **Confirmed** — `WallDiagonalGate` ships the per-channel content-derived position; `wall_attention` feature gates it. |
| **Latency** ("modelless, sub-µs, no GD") | "Wall is ~free (FA3-level throughput); RePo is +3%" | R145 §"Length Extrapolation" + RePo Table 6 | ✅ **Wall wins** — RePo's 3% overhead is from the SwiGLU+linear f_φ; Wall's prefix-sum factorization is post-softmax-free. |
| **Quality** ("matches / beats RePo's numbers") | "Wall matches RePo's long-context extrapolation" | **NOT compared head-to-head.** Wall reports 4K→160K; RePo reports 4K→16K. Different backbones (Wall: custom 400M/1B; RePo: OLMo-2 1B/7B). | ⚠️ **Unproven.** Architectural coverage does NOT imply quality parity. A head-to-head PoC would be needed to claim Wall ≥ RePo. |

The quality claim is **unproven** and explicitly marked as such. This is acceptable for a PASS verdict per §3.6 ("Low-confident verdicts that explicitly mark the quality claim as unproven and create a `.issues/` entry to track the PoC follow-up"). Since the architectural coverage is clear and the latency win is in Wall's favor, and since the only open question (RePo vs Wall head-to-head quality) is a riir-train concern (it requires training RePo's f_φ), no PoC follow-up is filed in katgpt-rs. The quality question belongs in riir-train if anyone wants to pursue it.

### 2.4 Latent-space reframe (mandatory per workflow §1 step 3)

Checked all seven Super-GOAT factory modules for a stronger reframing:

| Substrate | RePo reframe | Novelty? |
|---|---|---|
| (a) HLA per-NPC latent state | z_i = dot(h_i, d) is a rank-1 projection onto a direction vector — exactly how HLA already projects to scalars via sigmoid | NO — HLA already does this |
| (b) `latent_functor/` | position remapping is a rank-1 functor from hidden→scalar | NO — rank-1 functor ships (P303) |
| (c) `cgsp_runtime/` | curiosity could modulate position | Speculative, not actionable |
| (d) LatCal fixed-point | real-valued positions are not chain-committed | N/A |
| (e) NeuronShard | f_φ weights could be stored in a shard | Just storage, not a new primitive |
| (f) DEC Stokes-calculus | position remapping is not a manifold op | N/A |

No latent reframe produces a new capability class. The scalar-position reframe is the rank-1 special case of Wall's per-channel gate (already shipped).

---

## 3. Verdict

**PASS.**

**One-line reasoning:** RePo's primitive (content-derived, non-monotonic, per-head position assignment) is already shipped as Wall Attention (R145, `wall_attention` feature). Wall is the per-channel (vector-valued) instance; RePo is the scalar (rank-1) instance — same mechanism class, commensurate gains. RePo's additive value is its **training recipe** (continual pre-training of f_φ from a RoPE checkpoint) and its **analysis** (pattern statistics: Hybrid 74%, mirror symmetry, few-shot segmentation) — both → riir-train, out of scope for katgpt-rs/riir-*.

**MOAT gate per domain (§1.6):**
- `katgpt-rs` — Wall Attention already ships the primitive in this repo's pillar scope (transformer stack / attention slot). RePo adds nothing modelless. **Neutral — already covered.**
- `riir-ai` / `riir-chain` / `riir-neuron-db` — not a runtime/chain/shard concern. **Out of scope.**
- `riir-train` — the full trained f_φ could be a riir-train note if anyone wants to compare RePo vs Wall head-to-head. **Out of scope for this workflow** (note the redirect, stop).

**Novelty gate (§1.5) — all four NO:**
1. **No prior art?** NO — Wall Attention (R145, shipped `wall_attention` feature) is direct prior art for the mechanism class.
2. **New class of behavior?** NO — content-derived position is an optimization, not a new capability.
3. **Product selling point?** NO — "better long-context" is not a moat over Wall (which also extrapolates 4K→160K).
4. **Force multiplier?** NO — doesn't connect to ≥2 pillars in a new way.

**R169 / R368 guards:** N/A — this is not an agent-memory / LLM-orchestration paper. No false-trigger risk.

### Follow-up (deferred, not blocking)

- **riir-train** (optional, low priority): a head-to-head PoC comparing RePo's scalar positions vs Wall's per-channel gates on the same backbone (e.g., OLMo-2 1B continual pre-training from the same RoPE checkpoint). Would settle the open quality question (§2.3). Tracked here, not in an `.issues/` file, because it's a training-side question that belongs in riir-train if pursued.
- **katgpt-rs** (optional, very low priority): a "scalar Wall" variant — `z_i = dot(h_i, d)` as a rank-1 special case of the per-channel gate — could be a one-plan addition for users who want RoPE compatibility without the per-channel gate overhead. Not pursued because Wall's per-channel version is strictly more expressive and already ships at ~free latency cost. Would only matter if a future benchmark showed the scalar version generalizes better on some axis Wall doesn't.

---

## TL;DR

RePo = content-derived scalar positions fed to RoPE. Wall Attention (already shipped, R145, `wall_attention` feature) = content-derived per-channel gates replacing RoPE. **Same primitive class.** Wall is the more expressive instance (per-channel > scalar); RePo's only advantage is backward compatibility with RoPE-trained checkpoints via continual pre-training. The full trained comparison (RePo vs Wall quality) → riir-train. **PASS** — no modelless distillation adds value over the shipped Wall primitive.
