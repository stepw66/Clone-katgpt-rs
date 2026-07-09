# Research 399: Sparse Delta Memory (SDM) — Sparsifying Gated DeltaNet via PKM

> **Source:** Loïc Cabannes, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Matthijs Douze, Maria Lomeli, Ilze Amanda Auzina, Justin Carpentier, Gabriel Synnaeve, Hervé Jégou (Meta FAIR + Inria/ENS), "Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity", [arXiv:2607.07386](https://arxiv.org/abs/2607.07386), 9 Jul 2026.
> **Code:** https://github.com/facebookresearch/sparse-delta-memory
> **Date:** 2026-07-09
> **Status:** Done — PASS
> **Classification:** Public
> **Related Research:** 387 (FwPKM — **the closest related work SDM itself cites**), 070 (Gated DeltaNet-2), 024 (δ-Mem), 199 (Memory Caching Growing RNN)
> **Related Plans:** 408 (PKM primitive — shipped), 053 (δ-Mem modelless — shipped)

---

## TL;DR

SDM sparsifies Gated DeltaNet's dense key-value outer product using **Product-Key Memory** (PKM) addressing, scaling the recurrent state to N ≈ 10⁶ slots (vs GDN's ~10³–10⁴) while keeping per-token FLOPs constant. The two contributions: (a) replace the dense `M_t ∈ ℝ^{d_qk × d_v}` update with sparse top-W writes / top-R reads selected via PKM's two √N codebooks + Cartesian-product top-k; (b) **learn the initial state M₀** so the large sparse memory doubles as a parametric pretraining-knowledge store (zero extra inference FLOPs). Trained 8B on 1T tokens; beats GDN and even full attention at scale on RULER long-context recall.

**Verdict: PASS.** The modelless architectural mechanism — *sparsify a delta-rule recurrent memory via PKM addressing* — is already distilled and shipped through three prior notes: **Research 387 (FwPKM)** is the *exact* paper SDM cites as its closest related work, and it already distilled (1) the PKM factorization → shipped as Plan 408, (2) the "fast weight update" → δ-rule analog (Plan 053), (3) the F1 fusion "PKM × δ-Mem write gate" (exactly SDM's sparsified delta rule), (4) the F4 fusion "PKM × `MerkleFrozenEnvelope`" (exactly SDM's learned-initial-state M₀ pattern). Research 024 (δ-Mem) and 070 (GDN-2) cover the delta-rule + gated-delta baseline. SDM's only genuinely new contributions over FwPKM are *training-side*: the isoFLOP parameter/FLOP matching, the 8B/1T-token scaling ladder, and the empirical demonstration that learned M₀ helps at scale. Those are training contributions → **riir-train**. No plan, no implementation in this session.

---

## 1. Paper Core Findings

### 1.1 The mechanism (the part that IS modelless)

SDM replaces GDN's dense state update `M_t ← α_t M_{t-1} + β_t k_t (v_t − α_t M_{t-1}ᵀ k_t)ᵀ` with a **sparse** version operating on an explicit memory table `M ∈ ℝ^{N × d_v}`:

1. **Sparse key selection (PKM).** Project input `x_t` to write keys `k'_t` and read queries `q'_t` via `W_k, W_q ∈ ℝ^{d × 2√N}`. Split each into two halves `∈ ℝ^{√N}`. Outer sum `k'_{1,t} ⊕ k'_{2,t} ∈ ℝ^{√N × √N}` yields N scores; `top-W` on writes, `top-R` on reads. Complexity `O(√N·d + W² + R²)` — **independent of N**.
2. **Gated delta write** (Eq 3–4): for each selected write slot `i ∈ I^w_t`:
   - `M̃_t[i] ← α_t · M_{t-1}[i]` (decay)
   - `M_t[i] ← M̃_t[i] + β_t · k^{(i)}_t · (v_t − M̃_t[i]ᵀ k^{(i)}_t)` (delta rule)
   - Unselected slots unchanged.
3. **Sparse read** (Eq 5): `y_t = Σ_{i ∈ I^r_t} q^{(i)}_t · M_t[i]`.
4. Norm, gating, head mixing (standard).

**Connection to GDN:** when N = d_qk, W = R = d_qk, and the sparse keys form a dense vector, Eq 4 recovers GDN exactly. So SDM is a strict generalization: GDN is the dense special case.

### 1.2 The learned initial state M₀ (the part that needs training)

Because SDM's state is large (up to ~8 GB at 8B scale), M₀ can be **learned during pretraining** and reused at inference as parametric knowledge. Ablation (Fig 5, Table 3): learned M₀ improves code NLL (0.845 → 0.822), average accuracy (37.3 → 38.5), and RULER (28.0 → 31.2) at 1.4B. Learning M₀ on a vanilla GDN does *not* help (state too small to be useful as a knowledge store).

**This is the genuinely interesting idea**, and it is exactly the "frozen parametric memory" pattern — but it requires gradient descent during pretraining to *learn* M₀. At inference it is a zero-FLOP frozen read.

### 1.3 Scaling results (training-side)

- 8B SDM trained on 1.141T tokens beats isoFLOP GDN on validation NLL (2.253 vs 2.298) and *beats full attention* (2.285).
- RULER long-context recall: SDM 50.2 vs GDN 34.2 vs FullAttn 61.2 at 8B. SDM matches/beats FullAttn on 3/6 RULER tasks at 8B despite fixed memory.
- State size scales as O(d³) per layer but FLOPs stay constant — the "free" capacity comes from sparsity.
- Ablation: monotonic NLL improvement as N grows (27 MB → 432 MB at 0.8B: 0.947 → 0.914). Even 27 MB SDM beats GDN on RULER.
- Memory-access adaptation: read distributions more uniform than write; writes peak (~85% mass in top-32 of 64); distributions adapt (limited reads → more uniform writes).

### 1.4 Efficient training (training-side, Appendix A)

Chunkwise parallel via WY representation (intra-chunk parallel + chunkwise recurrent). Sparse inner product via two-pointer merge on sorted slot indices (O(W) per token pair vs O(d_qk) for dense). Memory-efficient backward via in-place sparse updates + undo (226 GB → 8 GB at 16k/8B). fp8/int4 snapshot quantization has no detrimental effect.

---

## 2. Distillation — modelless coverage check

### 2.1 Vocabulary translation (paper → codebase)

| SDM term | Codebase equivalent | Where it ships / is distilled |
|---|---|---|
| sparse delta memory `M ∈ ℝ^{N × d_v}` | `ProductKeyMemory` value table `V ∈ ℝ^{N × d_v}` + `NeuronShard` slots | Plan 408 (PKM), `riir-neuron-db/src/shard.rs` |
| PKM factorization (two √N codebooks + Cartesian top-k) | `ProductKeyMemory::query` (O(√N) retrieval) | Plan 408 — shipped, default-on |
| gated delta write (Eq 3–4) | `DeltaMemoryState::write_segment` δ-rule (bit-identical to one GD step at η=1) | Plan 053 (Research 024) — shipped behind `delta_mem` |
| decay gate `α_t = exp(−A·softplus(...))` | GDN-2 channel-wise decay (E2) | Research 070 E2 (medium value, not yet implemented) |
| input gate `β_t = σ(W_b x_t)` | sigmoid curiosity gate, CommittedFieldBlend output gate | Plan 277, Plan 321 |
| **learned initial state M₀** | **frozen `NeuronShard::style_weights[64]`** + `MerkleFrozenEnvelope` snapshot | Research 387 F4 fusion — `riir-neuron-db/src/shard.rs`, `freeze.rs` |
| slot collapse (write concentration) | TEMP `sleep_diverse` diversity selector | Plan 005 — default-on |
| catastrophic forgetting (continual learning) | Raven/δ-Mem consolidation sleep-cycle | `riir-neuron-db/src/consolidation.rs` — default-on |
| adaptive read/write distributions | runtime curiosity-driven write gating | Plan 277 (Temporal Derivative) |
| hybrid SWA:SDM (3:1 short:long) | Hybrid Recurrent + SWA architecture | Research 070 E5 (medium value) |

### 2.2 What is ALREADY covered (modelless)

| SDM component | Prior-art coverage | Status |
|---|---|---|
| **PKM sparse addressing** (the headline mechanism) | Research 387 §1.2 + §2.2; shipped as `ProductKeyMemory` Plan 408 | ✅ shipped, default-on |
| **Gated delta write on sparse slots** (Eq 3–4) | Research 387 §2.4 **F1 fusion** ("PKM × δ-Mem write gate — δ-Mem currently bounded by rank r; PKM unbounds to 10⁶ slots while keeping write cost O(1) per slot"). Research 024 (Plan 053) ships the δ-rule. Research 070 ships the GDN baseline. | ✅ distilled + fusion tracked; δ-rule ships, PKM-scaled δ-rule is the F1 follow-up |
| **Learned initial state M₀** | Research 387 §2.4 **F4 fusion** ("PKM × `MerkleFrozenEnvelope` — the PKM value table V becomes a freeze/thaw-committed snapshot"). `NeuronShard::style_weights[64]` is exactly a frozen parametric memory read at inference. | ✅ pattern distilled; concrete wiring is the F4 follow-up |
| **Catastrophic-forgetting retention** (SDM doesn't claim this but it's the natural pair) | Research 387 §1.4 + F5 fusion — Raven/δ-Mem consolidation solves it | ✅ shipped, default-on |
| **Slot-collapse prevention** | Research 387 §2.3 — TEMP `sleep_diverse` | ✅ shipped, default-on |
| **Hybrid SWA + global** | Research 070 E5 | distilled, optional |

### 2.3 What is NOT modelless (→ riir-train)

| SDM component | Why it needs training | Routing |
|---|---|---|
| **Training SDM from scratch** (the 8B/1T-token scaling ladder) | Requires backprop through base weights + the sparse memory across 1T tokens | → riir-train |
| **Learning M₀ via gradient descent** | M₀ is a learned parameter; the *pattern* (frozen parametric memory) is modelless, but *learning what to freeze* requires GD | → riir-train (the runtime freeze/thaw of a learned M₀ stays modelless in riir-ai/riir-neuron-db) |
| **IsoFLOP parameter/FLOP matching** (the experimental methodology) | Pure training-side experimental design | → riir-train |
| **Empirical scaling laws** (Fig 3, R²=0.999) | Training-curve measurement | → riir-train |
| **Efficient training kernel** (WY representation, two-pointer merge, fp8 snapshot) | Backward-pass + Triton kernel work | → riir-train |

### 2.4 §3.5 modelless-unblock check

The §3.5 protocol is for cases where a *gate or mechanism appears to need training*. SDM is not a gate — it's a full architecture. The check still applies to the one mechanism that looks training-bound:

- **"Learning M₀"** — can freeze/thaw (path 1) provide it? **YES, architecturally.** A frozen `NeuronShard` snapshot *is* a learned-initial-state M₀; the snapshot is produced offline (training) and thawed at inference (modelless). The runtime half is modelless; the offline half is riir-train. Path 1 returns modelless-validable for the runtime; the offline learning is a genuine riir-train dependency (you cannot derive a useful M₀ deterministically — it encodes dataset-wide statistics).

The genuinely training-only part is the empirical claim "SDM beats GDN/FullAttn at 8B scale" — that requires the full pretraining run. **→ riir-train.**

### 2.5 §3.6 PoC requirement

**Not triggered.** This verdict does NOT claim quality parity with SDM's 8B results, nor that our modelless substrate "covers" SDM's scaling behavior. It claims only **architectural coverage** (the PKM + δ-rule + freeze-envelope pattern is distilled and shipped). The SDM empirical results (8B beats full attention) are explicitly a training contribution → riir-train, with no parity claim attached. Pure architectural redirect → no PoC required per §3.6.

---

## 3. Verdict

**Tier: PASS.**

**One-line reasoning:** The modelless architectural mechanism SDM introduces — *sparsify a delta-rule recurrent memory via PKM addressing, with a frozen parametric initial state* — is already distilled across Research 387 (FwPKM, the exact paper SDM cites as closest related work) + Research 024 (δ-Mem) + Research 070 (GDN-2), with the PKM primitive shipped as Plan 408 and the δ-rule shipped as Plan 053. SDM's only genuinely new contributions over FwPKM are training-side (isoFLOP methodology, 8B/1T-token scaling, empirical learned-M₀ benefit) → **riir-train**. No new modelless primitive, no new fusion not already tracked in Research 387 §2.4 (F1 + F4).

### Novelty gate (Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | **NO.** Research 387 (FwPKM) is the *exact* paper SDM cites as closest related work and already distills PKM + δ-rule + freeze-envelope. Research 024 + 070 cover the delta-rule / GDN baseline. The F1 fusion ("PKM × δ-Mem write gate") and F4 fusion ("PKM × `MerkleFrozenEnvelope`") in Research 387 §2.4 are literally SDM's two architectural contributions. | this note §2.2 |
| **Q2: New class of behavior?** | **NO.** Sparse delta-rule memory with PKM addressing is the F1 fusion already tracked in Research 387. Frozen parametric initial state is the F4 fusion. Neither produces a capability none of the parts has alone. | Research 387 §2.4 |
| **Q3: Product selling point?** | **NO (modelless side).** The modelless substrate (PKM + δ-rule + freeze envelope) is already the selling point tracked under Research 387. SDM adds nothing modelless. | — |
| **Q4: Force multiplier?** | **NO (new).** SDM does not multiply any pillar beyond what Research 387 already identifies. | — |

**Q1 fails → not Super-GOAT. Q2/Q3/Q4 also fail → not GOAT, not Gain.** PASS.

### MOAT gate per domain (§1.6)

| Domain | In scope? | MOAT contribution |
|---|---|---|
| `katgpt-rs` (public engine) | NO new primitive. PKM already ships (Plan 408). | None — already covered. |
| `riir-ai` (private runtime) | NO new runtime. F1/F4 fusions already tracked in Research 387. | None — already tracked. |
| `riir-chain` (private chain) | NO. | None. |
| `riir-neuron-db` (private shards) | NO new shard primitive. `NeuronShard` + `MerkleFrozenEnvelope` already cover the M₀ pattern. | None — already covered. |
| `riir-train` (private training) | **YES — training SDM from scratch, learning M₀, scaling laws.** | Genuine training dependency. Out of scope for this workflow — note "→ riir-train" and stop. |

### UQ-bearing primitive check ("Report the Floor" rule)

**Not applicable.** SDM does not ship a UQ-bearing primitive (no probability distribution, predictive interval, quantile, coverage guarantee, or calibrated uncertainty claim in the modelless sense). The gates `α_t, β_t` are mixing coefficients, not calibrated probabilities.

---

## 4. What does NOT ship (and why)

| SDM contribution | Status | Reason |
|---|---|---|
| PKM sparse addressing | **Already shipped** (Plan 408) | Research 387 distilled; Plan 408 implemented. |
| Gated delta write on sparse slots | **Already distilled + fusion tracked** | Research 387 F1 fusion; δ-rule ships (Plan 053); PKM-scaled δ-rule is the F1 follow-up. |
| Learned initial state M₀ (runtime half) | **Already distilled + fusion tracked** | Research 387 F4 fusion; `NeuronShard::style_weights[64]` + `MerkleFrozenEnvelope` cover the pattern. |
| Learning M₀ (offline half) | **→ riir-train** | Requires gradient descent during pretraining. |
| Training SDM from scratch (8B/1T tokens) | **→ riir-train** | Full architecture pretraining. |
| IsoFLOP scaling laws | **→ riir-train** | Training-curve measurement. |
| Efficient training kernel (WY, two-pointer, fp8) | **→ riir-train** | Backward-pass + Triton kernel. |
| 8B beats full attention at scale | **NOT claimed modellessly** | Requires the full pretraining run. Our substrate makes no quality-parity claim per §3.6. |

---

## 5. Cross-references

- **Closest prior art (the paper SDM cites as closest related work):** [Research 387 — FwPKM](387_Fast_Weight_Product_Key_Memory_PKM.md). SDM's two architectural contributions are exactly Research 387's F1 (PKM × δ-Mem write gate) and F4 (PKM × `MerkleFrozenEnvelope`) fusions.
- **Delta-rule baseline:** [Research 024 — δ-Mem](024_Delta_Mem_Online_Associative_Memory.md) (Plan 053 shipped); [Research 070 — Gated DeltaNet-2](070_Gated_DeltaNet_2_Decoupled_Erase_Write_Linear_Attention.md).
- **Shipped primitives:** Plan 408 (`ProductKeyMemory`), Plan 053 (`DeltaMemoryState`), Plan 005 (TEMP `sleep_diverse`), Plan 277 (Temporal Derivative), Plan 321 (CommittedFieldBlend).
- **Private substrate:** `riir-neuron-db/src/shard.rs` (`NeuronShard::style_weights[64]` = the M₀ analog), `riir-neuron-db/src/freeze.rs` (`MerkleFrozenEnvelope`), `riir-neuron-db/src/consolidation.rs` (Raven/δ-Mem — solves SDM's natural catastrophic-forgetting pair).
- **Training redirect:** `riir-train/.research/` — SDM pretraining, M₀ learning, scaling laws, efficient training kernel.

---

## TL;DR

SDM = Gated DeltaNet (Research 070) + PKM sparse addressing (Research 387 / Plan 408) + learned initial state M₀ (Research 387 F4 fusion: PKM × `MerkleFrozenEnvelope`). All three architectural pieces are already distilled; two are shipped (PKM, δ-rule), one is a tracked fusion (F4). SDM's only genuinely new contributions over FwPKM are training-side (isoFLOP methodology, 8B/1T-token scaling, empirical learned-M₀ benefit) → **riir-train**. **Verdict: PASS** — no new modelless primitive, no new fusion not already tracked. No plan, no implementation in this session.
