# Research: Higher-order Linear Attention — HLA (28)

> Source: [Higher-order Linear Attention](https://arxiv.org/abs/2510.27258v3) — Yifan Zhang, Zhen Qin, Mengdi Wang, Quanquan Gu (Princeton/UCLA), 2026
> Date: 2026-05, distilled 2026-06
> **Verdict: HIGH VALUE — Second-order HLA replaces O(N) KV-cache with O(d²) constant-size state per head, enabling truly infinite-context inference. The asymmetric variant AHLA achieves O(d·dv) state (no d² matrices) with a different inductive bias. Both are drop-in attention replacements with proven associative scans for parallel training. Sweet spot: seq_len >> head_dim². Implementation priority: AHLA first (smaller state), then symmetric HLA (higher expressivity).**

## TL;DR

HLA generalizes linear attention by maintaining **prefix sufficient statistics** — compact running summaries that capture higher-order interactions between queries, keys, and values. The core insight: the second-order attention matrix (QKᵀ)(QKᵀ)ᵀ = Q(KᵀK)Qᵀ depends only on KᵀK (a d×d matrix), not the full n×n attention matrix.

Three equivalent forms:
1. **Recurrent** — constant O(d² + d·dv) state per head, O(1) per token update
2. **Parallel** — full n×n masked second-order tensor attention (for theory, not practice)
3. **Chunk-parallel** — associative scan with semidirect product operator (for GPU training)

Key results:
- Second-order HLA: 5-tuple state (SK, CQV, mQ, G, h), strictly causal, streaming
- AHLA (asymmetric): 4-tuple state (PKV, mK, E, n), O(d·dv) state, O(d·dv) per token
- Third-order HLA: 4 corrected cross-summaries, 3 segment maps for chunk-parallel scan
- With exponential decay γ: all associativity preserved, just scale carry-in by γ

---

## Core Mechanisms (What We Need)

### 1. Second-Order HLA — Symmetric (AAᵀV)

The output at time t is computed from prefix sufficient statistics:

```
State tuple per head: S_t = (SK_t, CQV_t, mQ_t, G_t, h_t)

SK_t  = Σ_{i≤t} k_i k_iᵀ          ∈ R^{d×d}   (key second moment)
CQV_t = Σ_{i≤t} q_i v_iᵀ          ∈ R^{d×dv}  (query-value cross moment)
mQ_t  = Σ_{i≤t} q_i                ∈ R^d        (query mass)
G_t   = Σ_{i≤t} k_i k_iᵀ CQV_{i-1} ∈ R^{d×dv}  (causal correction numerator)
h_t   = Σ_{i≤t} k_i k_iᵀ mQ_{i-1}  ∈ R^d        (causal correction denominator)
```

**Streaming updates (O(d² + d·dv) per token):**

```
SK_t  = SK_{t-1} + k_t k_tᵀ
CQV_t = CQV_{t-1} + q_t v_tᵀ
mQ_t  = mQ_{t-1} + q_t
G_t   = G_{t-1} + k_t (k_tᵀ CQV_{t-1})     ← uses OLD CQV
h_t   = h_{t-1} + k_t (k_tᵀ mQ_{t-1})       ← uses OLD mQ
```

**Output (masked, unnormalized):**

```
o_t = q_tᵀ (SK_t · CQV_t − G_t)
```

**Optional normalization:**

```
o_t = q_tᵀ (SK_t · CQV_t − G_t) / (q_tᵀ (SK_t · mQ_t − h_t) + ε)
```

**Connection to standard attention:** Setting SK=I recovers linear attention with kernel K(q,q')=qᵀq'. The key moment SK_t acts as a learned, data-dependent metric on query space.

### 2. AHLA — Asymmetric (AAV)

A complementary variant with **O(d·dv) state** (no d×d matrices):

```
State tuple per head: S_t = (PKV_t, mK_t, E_t, n_t)

PKV_t = Σ_{j≤t} k_j v_jᵀ          ∈ R^{d×dv}  (key-value prefix)
mK_t  = Σ_{j≤t} k_j                ∈ R^d        (key mass)
E_t   = Σ_{i≤t} k_i (q_iᵀ PKV_i)  ∈ R^{d×dv}  (routed accumulation)
n_t   = Σ_{i≤t} k_i (q_iᵀ mK_i)   ∈ R^d        (denominator accumulator)
```

**Streaming updates (O(d·dv) per token — no d² ops!):**

```
PKV_t = PKV_{t-1} + k_t v_tᵀ
mK_t  = mK_{t-1} + k_t
r_t   = q_tᵀ PKV_t                   ← 1×dv vector
E_t   = E_{t-1} + k_t r_t
n_t   = n_{t-1} + k_t (q_tᵀ mK_t)
```

**Output:**

```
o_t = q_tᵀ E_t
```

**Key difference:** AHLA routes value information through an intermediate key index i: o_t = Σ_{j≤t} Σ_{i=j}^{t} (q_tᵀ k_i)(q_iᵀ k_j) v_jᵀ. This is a left-cascaded product A·A·V instead of A·Aᵀ·V.

**State comparison:**

| Variant | State size per head | Per-token cost |
|---------|---------------------|----------------|
| Standard attention | O(N·d) (growing KV cache) | O(N·d) |
| Symmetric HLA (2nd) | O(d² + d·dv) | O(d² + d·dv) |
| AHLA (asymmetric) | O(d·dv + d) | O(d·dv) |
| Linear attention (1st) | O(d·dv + d) | O(d·dv) |

AHLA matches linear attention in state size but captures second-order interactions.

### 3. Exponential Decay

Both variants support decay γ ∈ (0,1):

```
Symmetric:
  SK_t  = γ·SK_{t-1} + k_t k_tᵀ
  CQV_t = γ·CQV_{t-1} + q_t v_tᵀ
  G_t   = γ·G_{t-1} + k_t (k_tᵀ CQV_{t-1})

AHLA:
  PKV_t = γ·PKV_{t-1} + k_t v_tᵀ
  E_t   = γ·E_{t-1} + k_t (q_tᵀ PKV_t)
```

Decay controls spectral growth and adds recency bias. All associativity is preserved.

### 4. Chunk-Parallel Training (Associative Scans)

For GPU training, both variants define associative operators for Blelloch scans:

**Symmetric semidirect product:**

```
(A) ⊕ (B) = (
  SK_A + SK_B,
  CQV_A + CQV_B,
  mQ_A + mQ_B,
  G_A + G_B + SK_B · CQV_A,     ← cross-term
  h_A + h_B + SK_B · mQ_A        ← cross-term
)
```

**AHLA concatenation:**

```
(A) ⊕_AHLA (B) = (
  RKQ_A + RKQ_B,
  PKV_A + PKV_B,
  mK_A + mK_B,
  E_A + E_B + RKQ_B · PKV_A,     ← cross-term
  n_A + n_B + RKQ_B · mK_A        ← cross-term
)
```

Both are proven associative. An exclusive Blelloch scan produces the same activations as serial recurrence.

### 5. Third-Order HLA

Third-order uses AAᵀA with 3 corrected cross-summaries G^(1), G^(2), G^(3) and 3 segment maps. State is larger and includes O(d³) segment maps for exact chunk composition. The paper provides complete pseudocode but notes this is mainly theoretical — second-order is the practical variant.

---

## Key Experimental Findings (from paper)

The paper focuses on algorithmic structure. Key claims:

1. **Exact equivalence** — Scan outputs match serial recurrence exactly (proven in Theorems 4.1, 6.1, 7.2)
2. **No approximation** — Unlike kernel-based linear attention, HLA doesn't approximate softmax. It computes a different (but well-motivated) operator
3. **O(d²) vs O(N·d)** — HLA wins when d² < N·d, i.e., d < N. For head_dim=64, this means seq_len > 64
4. **Symmetric SK is a data-dependent metric** — richer than linear attention's identity metric
5. **AHLA is strictly O(d·dv)** — no d² matrices, matches linear attention cost but captures higher interactions

---

## Mapping to Our Stack

### Direct Replacements in `transformer.rs`

| Current Code | HLA Replacement |
|---|---|
| `KVCache { key: [block_size × kv_dim], value: [block_size × kv_dim] }` | `HlaLayerState { sk, cqv, mq, g, h }` per head (constant size) |
| `attention_head()` — O(t_n · hd) loop over past positions | `hla_attention_head()` — O(d² + d·dv) constant-time matmul |
| `MultiLayerKVCache` — allocates block_size × kv_dim per layer | `MultiLayerHlaCache` — allocates d² + d·dv per head per layer |
| `forward_base()` — stores K,V then loops | `forward_hla()` — updates state then reads out |
| Config `block_size` caps context window | No context window limit (streaming is O(1) per token) |

### Memory Comparison by Config

| Config | head_dim | KV Cache per head | HLA State per head | AHLA State per head | Break-even seq_len |
|--------|----------|-------------------|--------------------|--------------------|--------------------|
| micro() | 4 | 16 × 4 = 64 | 5 × 16 = 80 | 4 × 4 = 16 | >4 (immediate) |
| game() | 8 | 170 × 8 = 1360 | 5 × 64 = 320 | 4 × 8 = 32 | >8 (immediate) |
| bpe() | 8 | 256 × 8 = 2048 | 5 × 64 = 320 | 4 × 8 = 32 | >8 (immediate) |
| small_target() | 16 | 256 × 16 = 4096 | 5 × 256 = 1280 | 4 × 16 = 64 | >16 (immediate) |
| "Real" LLM (d=128) | 128 | N × 128 | 5 × 16384 = 81920 | 4 × 128 = 512 | >128 |

**Key insight:** For our tiny configs, HLA and AHLA are competitive immediately. AHLA is always smaller than KV cache. Symmetric HLA is smaller for all configs where d < block_size.

### GQA Interaction

With GQA (`n_kv_head < n_head`), multiple Q heads share K/V. For HLA:
- `SK_t` is shared per KV-group (key moment is key-only)
- `CQV_t` is per Q-head (query-value cross moment)
- `mQ_t` is per Q-head
- `G_t`, `h_t` are per Q-head (they mix Q and K)

For AHLA:
- `PKV_t`, `mK_t` are shared per KV-group
- `E_t`, `n_t` are per Q-head

This matches our existing GQA structure where kv_group = h × n_kv / n_head.

---

## Modelless Distillations

### D1: HLA Cache State — Symmetric Second-Order

```rust
/// Per-head state for symmetric second-order HLA.
/// Constant size: O(d² + d·dv) independent of sequence length.
#[derive(Clone)]
pub struct HlaHeadState {
    pub sk: Vec<f32>,   // [head_dim × head_dim] key second moment
    pub cqv: Vec<f32>,  // [head_dim × head_dim] query-value cross (dv=head_dim for us)
    pub mq: Vec<f32>,   // [head_dim] query mass
    pub g: Vec<f32>,    // [head_dim × head_dim] causal correction numerator
    pub h: Vec<f32>,    // [head_dim] causal correction denominator
}
```

### D2: AHLA Cache State — Asymmetric Second-Order

```rust
/// Per-head state for asymmetric second-order HLA (AHLA).
/// Constant size: O(d·dv + d) — no d×d matrices!
#[derive(Clone)]
pub struct AhlaHeadState {
    pub pkv: Vec<f32>,  // [head_dim × head_dim] key-value prefix (dv=head_dim)
    pub mk: Vec<f32>,   // [head_dim] key mass
    pub e: Vec<f32>,    // [head_dim × head_dim] routed accumulation
    pub n: Vec<f32>,    // [head_dim] denominator accumulator
}
```

### D3: HLA Attention Head — Replaces `attention_head()`

```rust
/// Second-order HLA readout: o_t = q_tᵀ (SK·CQV - G)
/// Constant time, no loop over past positions.
fn hla_attention_head(
    q: &[f32],         // [head_dim] query for this head
    state: &HlaHeadState,
    hd: usize,
) -> Vec<f32> {
    // u = q_tᵀ SK_t (1×d matvec)
    let mut u = vec![0.0f32; hd];
    for j in 0..hd {
        for i in 0..hd {
            u[j] += q[i] * state.sk[i * hd + j];
        }
    }
    // num = u · CQV_t - q_tᵀ · G_t (1×dv)
    let mut out = vec![0.0f32; hd];
    for j in 0..hd {
        let mut val = 0.0f32;
        for i in 0..hd {
            val += u[i] * state.cqv[i * hd + j];
            val -= q[i] * state.g[i * hd + j];
        }
        out[j] = val;
    }
    out
}
```

### D4: HLA State Update — Streaming Recurrence

```rust
/// Update HLA state with new (q_t, k_t, v_t).
/// MUST update cross-terms G,h BEFORE updating main accumulators SK,CQV,mQ.
fn hla_state_update(
    state: &mut HlaHeadState,
    q: &[f32],  // [hd]
    k: &[f32],  // [hd]
    v: &[f32],  // [hd]
    hd: usize,
) {
    // 1. Cross-terms using OLD state
    // kᵀ CQV_{t-1} (1×dv)
    let mut k_cqv = vec![0.0f32; hd];
    for j in 0..hd {
        for i in 0..hd {
            k_cqv[j] += k[i] * state.cqv[i * hd + j];
        }
    }
    // kᵀ mQ_{t-1} (scalar)
    let mut k_mq = 0.0f32;
    for i in 0..hd {
        k_mq += k[i] * state.mq[i];
    }
    // G_t += k_t · (kᵀ CQV_{t-1})
    for i in 0..hd {
        for j in 0..hd {
            state.g[i * hd + j] += k[i] * k_cqv[j];
        }
    }
    // h_t += k_t · (kᵀ mQ_{t-1})
    for i in 0..hd {
        state.h[i] += k[i] * k_mq;
    }

    // 2. Main accumulators
    // SK_t += k_t k_tᵀ
    for i in 0..hd {
        for j in 0..hd {
            state.sk[i * hd + j] += k[i] * k[j];
        }
    }
    // CQV_t += q_t v_tᵀ
    for i in 0..hd {
        for j in 0..hd {
            state.cqv[i * hd + j] += q[i] * v[j];
        }
    }
    // mQ_t += q_t
    for i in 0..hd {
        state.mq[i] += q[i];
    }
}
```

### D5: AHLA Streaming (Lower State Cost)

```rust
/// AHLA streaming update and readout.
/// O(d·dv) state, O(d·dv) per token. No d×d matrices.
fn ahla_step(
    state: &mut AhlaHeadState,
    q: &[f32],  // [hd]
    k: &[f32],  // [hd]
    v: &[f32],  // [hd]
    hd: usize,
) -> Vec<f32> {
    // PKV_t = PKV_{t-1} + k_t v_tᵀ
    for i in 0..hd {
        for j in 0..hd {
            state.pkv[i * hd + j] += k[i] * v[j];
        }
    }
    // r = q_tᵀ PKV_t (1×dv)
    let mut r = vec![0.0f32; hd];
    for j in 0..hd {
        for i in 0..hd {
            r[j] += q[i] * state.pkv[i * hd + j];
        }
    }
    // mK_t += k_t
    for i in 0..hd {
        state.mk[i] += k[i];
    }
    // E_t += k_t · r
    for i in 0..hd {
        for j in 0..hd {
            state.e[i * hd + j] += k[i] * r[j];
        }
    }
    // n_t += k_t · (q_tᵀ mK_t)
    let q_mk: f32 = (0..hd).map(|i| q[i] * state.mk[i]).sum();
    for i in 0..hd {
        state.n[i] += k[i] * q_mk;
    }

    // Output: o_t = q_tᵀ E_t
    let mut out = vec![0.0f32; hd];
    for j in 0..hd {
        for i in 0..hd {
            out[j] += q[i] * state.e[i * hd + j];
        }
    }
    out
}
```

---

## Relationship to Existing Work

| Our Existing | HLA Relationship |
|---|---|
| `MultiLayerKVCache` (flat) | **Replaced by** HLA/AHLA cache — constant vs growing |
| `RavenKVCache` (O(1) slots) | **Complementary** — Raven is heuristic routing, HLA is exact algebra |
| `TurboQuantKVCache` (compressed) | **Complementary** — quantization compresses KV entries, HLA eliminates them |
| `PagedKVCache` (virtual memory) | **Replaced** — paged alloc manages growing cache, HLA doesn't grow |
| `Percepta` (convex hull) | **Different axis** — Percepta is O(log N) spatial search, HLA is O(1) linear recurrence |
| Linear attention (1st order) | **Generalization** — SK=I recovers linear attention |
| Mamba/SSM | **Different** — SSMs use fixed dynamics, HLA uses data-dependent queries/keys |

---

## What Won't Transfer

1. **Drop-in on pretrained weights** — HLA computes a different function than SDPA. Models must be trained/fine-tuned with HLA from scratch. Swapping SDPA→HLA on pretrained weights will degrade quality.
2. **Third-order for tiny models** — With head_dim=4-8, third-order adds minimal expressivity but significant complexity. Second-order is the ceiling for our configs.
3. **Chunk-parallel scan for inference** — Scans are for training (GPU parallelism). Inference uses the streaming/recurrent form, which is already O(1).
4. **Symmetric HLA for large head_dim** — At head_dim=128, SK is 128×128 = 16K floats per head. AHLA's O(d·dv) state is strictly better for large heads.

---

## Key Insight for Implementation

**AHLA is the better first target for microgpt-rs:**

| Criterion | Symmetric HLA | AHLA |
|---|---|---|
| State per head (hd=4) | 80 floats | 16 floats |
| State per head (hd=8) | 320 floats | 32 floats |
| State per head (hd=16) | 1280 floats | 64 floats |
| Per-token cost | O(d² + d·dv) | O(d·dv) |
| d×d matrix ops | Yes (SK) | No |
| Expressivity | Higher (data-dependent metric) | Moderate (left-cascaded) |
| Implementation complexity | Medium (5-tuple state) | Low (4-tuple state) |
| Default choice? | For training quality | **For inference perf** |

For microgpt-rs's tiny head_dims (4-16), AHLA's lower state overhead and simpler implementation make it the practical choice. Symmetric HLA can be added later for quality-sensitive scenarios.

**Critical implementation note:** The update order matters! Cross-terms G,h must be computed using OLD CQV,mQ BEFORE updating them. This is the #1 correctness trap.

---

## References

- Zhang, Y., Qin, Z., Wang, M., & Gu, Q. (2026). Higher-order Linear Attention. arXiv:2510.27258v3.
- [HLA Project Page](https://github.com/yifanzhang-pro/HLA)
- Related: Linear Transformers (Katharopoulos et al., 2020) — first-order baseline
- Related: RetNet (Sun et al., 2023) — decay-aware linear attention
- Related: GLA (Yang et al., 2023) — gated linear attention with chunk parallelism
- Related: Delta Networks (Schlag et al., 2021) — fast weight programmer equivalence
- Related: Mamba2 (Dao & Gu, 2024) — SSM with data-dependent dynamics