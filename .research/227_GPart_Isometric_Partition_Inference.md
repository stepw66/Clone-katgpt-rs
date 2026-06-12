# Research 227: GPart — Isometric Partition for Inference-Time Adaptation

**Paper:** GPart: End-to-End Isometric Fine-Tuning via Global Parameter Partitioning (arxiv 2605.14841)
**Date:** 2026-06
**Status:** VERDICT — **CONDITIONAL GAIN**

---

## TL;DR

GPart replaces LoRA's bilinear BA factorization with a single isometric partition matrix P (seed-generated, deterministic) that maps a d-dimensional trainable vector θ_d into full weight space: `W = W₀ + Pθ_d` where `P^T P = I_d`. Storage: d+1 values (θ_d + seed) vs LoRA's r(m+n). In katgpt-rs's modelless context we can't train θ_d — but we **can** use GPart's partition as a drop-in LoRA replacement for adapter loading (4–8× storage compression), as structured pruning groups for BanditPruner, as isometric weighting for MUX-Latent superposition, and as deterministic routing keys for consensus. The mathematical family is identical to our existing `JlProjectionMatrix` (Gram-Schmidt orthogonal projection with BLAKE3 commitment). Four concrete fusion ideas, one high-value (adapter loading), one medium (partition pruning), two speculative (MUX weighting, seed-route consensus). Gate behind `gpart_adapter`, GOAT-prove against LoRA baseline.

---

## Paper Core Ideas

### 1. Isometric Partition Matrix P

GPart defines `W = W₀ + Pθ_d` where:

- `P ∈ R^{N×d}` is a **partition matrix**: each row has exactly one nonzero entry (value `1/√n_g` where `n_g` is group size)
- `P^T P = I_d` — isometric (orthonormal columns), preserves geometry
- Assignment function `g: {1,...,N} → {1,...,d}` via seed-determined pseudorandom permutation
- Storage: **d + 1 values** (θ_d vector + one seed integer)

This eliminates LoRA's bilinear map `BA` entirely. No rank bottleneck. No geometric distortion from two low-rank matrices.

### 2. Key Properties vs LoRA

| Property | LoRA (BA) | GPart (Pθ_d) |
|----------|-----------|---------------|
| Expressive capacity | Bounded by rank r × (m+n) | Bounded by d (partition dimension) |
| Geometric distortion | Yes (bilinear rank bottleneck) | No (isometric — angles/distances preserved) |
| Storage | r(m+n) per layer | d+1 per layer |
| Hyperparameters | r, α | d |
| Weight reconstruction | W₀ + α/r · BA | W₀ + Pθ_d |
| Mathematical family | Low-rank approximation | Random orthogonal projection |

### 3. Group Assignment Mechanics

The seed `s` determines a pseudorandom permutation π_s over {1,...,N}. Parameter i is assigned to group `g(i) = π_s(i) mod d`. Each group has ~N/d members. The 1/√n_g normalization ensures `P^T P = I_d` exactly.

This is **the same mathematical operation** as our `JlProjectionMatrix` — Gram-Schmidt orthogonal projection from high-dim to low-dim — except GPart uses partition (sparse, exactly 1 nonzero per row) while JL uses dense projection.

---

## Novel Fusion Ideas

### Idea 1: Isometric Adapter Loading — Replace LoRA BA with GPart P for Inference

**Mapping:** `LoraAdapter` → `GpartAdapter` (new struct)

Current `LoraAdapter` stores `(rank, in_dim, out_dim, a: Vec<f32>, b: Vec<f32>, alpha: f32)`. For our micro-transformer at rank=8, dim=16: storage = 8×16 + 8×16 = 256 values per target × 6 targets = 1536 values.

GPart alternative: `GpartAdapter { seed: u64, theta: Vec<f32> }`. With d=128: storage = 129 values per target × 6 targets = 774 values. At d=64: 390 values. **~2–4× compression**. For riir-ai's larger models the ratio improves further.

**Implementation sketch:**

```rust
pub struct GpartAdapter {
    pub seed: u64,
    pub theta: Vec<f32>,    // d values
    pub d: usize,           // partition dimension
}

impl GpartAdapter {
    /// Regenerate partition matrix from seed at load time.
    /// Assignment: g(i) = fastrand::with_seed(seed + i) % d
    /// Value: 1/√n_g where n_g = count of params assigned to group g
    pub fn apply(&self, base_weights: &mut [f32]) {
        let n = base_weights.len();
        let mut group_counts = vec![0usize; self.d];
        // First pass: count group sizes
        let mut assignments = vec![0usize; n];
        let mut rng = fastrand::Rng::with_seed(self.seed);
        for i in 0..n {
            let g = rng.usize(..self.d);
            assignments[i] = g;
            group_counts[g] += 1;
        }
        // Second pass: apply Pθ_d
        for i in 0..n {
            let g = assignments[i];
            let scale = 1.0 / (group_counts[g] as f32).sqrt();
            base_weights[i] += scale * self.theta[g];
        }
    }
}
```

**Benefits:**
- Single-pass O(N) broadcast instead of two matmuls
- BLAKE3 commitment: `hash(seed.to_le_bytes() || theta.as_bytes())`
- Compatible with `ContiguousWeights` — apply in-place during loading
- NeuronShard-compatible: seed(8 bytes) + θ_d(d×4 bytes) fits in 368-byte Pod for d≤90

**Risk:** θ_d must be trained (training-side, riir-ai). katgpt-rs only loads and applies. This is the same relationship as current LoRA — we load, not train.

### Idea 2: Partition Pruning — GPart Groups as BanditPruner Arms

**Mapping:** `BanditPruner` arms ↔ GPart partition groups

GPart's assignment `g: {1,...,N} → {1,...,d}` creates d groups of ~N/d parameters each. Each group is a natural arm for our multi-armed bandit:

```rust
pub struct GpartBanditPruner {
    adapter: GpartAdapter,
    bandit: BanditPruner,       // d arms, one per partition group
    active_groups: Vec<bool>,   // which groups to apply
}

impl GpartBanditPruner {
    /// Apply only groups where bandit says active.
    /// Groups that don't improve output get zeroed — automatic structured pruning.
    pub fn apply_selective(&self, base_weights: &mut [f32]) {
        for i in 0..base_weights.len() {
            let g = self.adapter.assignment(i);
            if self.active_groups[g] {
                base_weights[i] += self.adapter.delta_for_group(g);
            }
        }
    }
}
```

**Connection to existing infrastructure:**
- `BanditPruner` already does multi-armed bandit arm selection at inference time
- Each GPart group = one arm; bandit learns which groups improve output quality
- Natural structured pruning: zeroed groups remove ~N/d parameters each
- Composable with PlasmaPath: prune → ternary quantize remaining weights

**Value: MEDIUM.** Depends on having trained GPart adapters first. The bandit infrastructure exists; the missing piece is training-side θ_d production.

### Idea 3: Isometric MUX-Latent — GPart Partition for Context Compression

**Mapping:** MUX superposition weights ↔ GPart group scalars

Our MUX-Latent (Research 158, Plan 238) compresses context via `Σ decay^j × onehot(t_j)`. GPart's isometric grouping can improve the compression quality:

- **Current MUX:** `Σ decay^j × onehot(t_j)` — uniform geometric decay
- **GPart-MUX:** `Σ θ_{g(j)}/√n_{g(j)} × onehot(t_j)` — isometric group weighting

The partition groups related token dimensions together, preventing cross-contamination between unrelated semantic directions. The isometry property (`P^T P = I_d`) guarantees that the superposition preserves distances — the same guarantee our `JlProjectionMatrix` already provides for shard embeddings.

**Value: SPECULATIVE.** Requires Research 158 (MUX) to be implemented first. The mathematical connection is sound but unvalidated. The superposition-as-compression primitive may not benefit from partition-based weighting over simple geometric decay.

### Idea 4: Seed-Route Consensus — GPart Seed as Deterministic Routing Key

**Mapping:** Chain/consensus layer ↔ GPart seed

In our consensus layer, the seed `s` determines the partition. Two nodes with the same seed produce **identical** adapter behavior without transmitting full adapter matrices:

```
Node A: loads seed=42, θ_d → regenerates P_42 → applies P_42·θ_d
Node B: loads seed=42, θ_d → regenerates P_42 → applies P_42·θ_d
Result: bit-identical weight deltas
```

Commitment: `BLAKE3(s || θ_d)` — tamper-proof, 32 bytes regardless of d.

**NeuronShard zero-copy compatibility:**
- NeuronShard is a 368-byte Pod with BLAKE3 commitment
- seed(8 bytes) + θ_d(up to 360 bytes = d≤90) fits in the Pod
- No need to serialize/store the full P matrix — regenerated from seed on load

**Value: SPECULATIVE.** Our consensus layer is designed for deterministic replay of raw values (per `.research/` latent vs raw space rules). GPart's seed-regeneration is deterministic, but we'd need to verify that floating-point permutation generation is bit-identical across platforms. The `fastrand::Rng` we use is deterministic and cross-platform, which mitigates this risk.

---

## GOAT Gate Design

### Feature Gate: `gpart_adapter`

```toml
[features]
gpart_adapter = []   # GPart isometric partition adapter loading
```

### GOAT Gates (per benchmark methodology)

| Gate | Metric | Threshold | Measurement |
|------|--------|-----------|-------------|
| G1: Storage | Adapter bytes | `< 50%` of equivalent LoRA | `size_of(GpartAdapter) / size_of(LoraAdapter)` |
| G2: Apply speed | Time to apply adapter | `≤ 110%` of LoRA apply | Micro-bench: `GpartAdapter::apply` vs `lora_apply` |
| G3: Quality | Output quality with adapter | `≥ 95%` of LoRA quality | Requires trained θ_d from riir-ai |
| G4: Determinism | Cross-platform bit-identical | `100%` | Same seed + θ_d → identical weights on x86 + ARM + WASM |
| G5: Commitment | BLAKE3 verification | `100%` pass | Tamper detection on seed or θ_d |

### Benchmark File

`tests/bench_227_gpart_adapter_goat.rs` — mirrors `bench_230_shard_embedding_goat.rs` structure.

### Promotion/Demotion Rules

- If G1–G5 all pass → promote `gpart_adapter` to default feature
- If G3 fails (quality regression) → keep gated, investigate θ_d training
- If G2 fails (>10% slower) → demote, LoRA remains default adapter format
- `LoraAdapter` is NEVER removed — GPart is an alternative loading path, not a replacement

---

## Honest Assessment

### What GPart Gives Us

1. **Mathematical family match.** GPart's partition matrix is a sparse variant of the same JL orthogonal projection we already use in `JlProjectionMatrix`. The infrastructure (seed-based generation, BLAKE3 commitment, SIMD projection) is already built.

2. **Storage compression for adapters.** d+1 values vs r(m+n). For our micro-transformer: ~2–4× compression. For riir-ai's larger models: potentially 10–100× compression at equivalent quality.

3. **No bilinear distortion.** Isometry means the adapter doesn't warp the weight space geometry. This matters for downstream operations (quantization, pruning) that assume Euclidean structure.

### What It Doesn't Give Us

1. **We still can't train.** katgpt-rs is modelless — inference-time only. θ_d must be produced by riir-ai's training pipeline. This is the same constraint as LoRA.

2. **No quality improvement without training.** A random θ_d is useless. The value is entirely in the storage/apply-side improvements, not in producing better adapters.

3. **Partition ≠ learned structure.** GPart's group assignment is pseudorandom (seed-determined), not learned. It doesn't discover meaningful parameter groupings. Structured pruning via BanditPruner can discover useful groups post-hoc, but the initial partition is random.

4. **Scale mismatch for some ideas.** Ideas 3 (MUX-MUX) and 4 (seed-route consensus) are architecturally interesting but require infrastructure that doesn't exist yet (MUX implementation, multi-node consensus). These are speculative.

---

## Verdict

**CONDITIONAL GAIN — Implement Idea 1 (Isometric Adapter Loading) behind `gpart_adapter` feature gate.**

Reasoning:
- GPart's partition matrix is the same math as our existing `JlProjectionMatrix` — minimal new infrastructure
- Storage compression (2–4× for micro, 10–100× for larger models) is a concrete, measurable win
- Single-pass O(N) apply is simpler and potentially faster than LoRA's two matmuls
- The modelless constraint is the same as LoRA — we load, not train — no new dependency
- BLAKE3 commitment + seed regeneration fits naturally into our existing security model
- Ideas 2–4 are gated behind further infrastructure (trained adapters, MUX, consensus)

**NO GAIN for Ideas 3–4 without dependencies:** MUX-Latent isn't implemented yet; consensus layer needs multi-node testing. Revisit when those land.

**MAYBE GAIN for Idea 2 (Partition Pruning):** BanditPruner infrastructure exists, but needs trained GPart adapters to be useful. Create issue, don't plan yet.

---

## What to Implement

- [ ] `GpartAdapter` struct in `katgpt-core/src/types.rs` (seed, theta, d)
- [ ] `GpartAdapter::generate_partition()` — seed-based pseudorandom group assignment
- [ ] `GpartAdapter::apply()` — single-pass O(N) weight delta application
- [ ] `GpartAdapter::commitment()` — BLAKE3(seed || theta)
- [ ] `GpartAdapter::verify()` — commitment check
- [ ] Binary format: `[GPART(5) | version(4) | d(4) | seed(8) | blake3(32) | theta(d×4)]`
- [ ] Feature gate `gpart_adapter` in `katgpt-core/Cargo.toml`
- [ ] `GpartPair` (mirroring `LoraPair`) for prefill/decode split
- [ ] GOAT benchmark `tests/bench_227_gpart_adapter_goat.rs`
- [ ] Interop: `GpartAdapter` ↔ `LoraAdapter` conversion (lossy: train-side computes θ_d = P⁺ΔW)

### Not Implementing (Deferred)

- Idea 2 (Partition Pruning) — create issue at `katgpt-rs/.issues/`
- Idea 3 (MUX-MUX) — blocked on Research 158 / Plan 238
- Idea 4 (Seed-Route Consensus) — blocked on multi-node consensus layer

---

## Related Research

| Research | Connection |
|----------|------------|
| `004_LoRA_Architecture_Verdict.md` | Current adapter architecture GPart would augment |
| `132_LoRAPrune_Structured_Pruning_LoRA.md` | Structured pruning via LoRA gradients; GPart offers partition-based alternative |
| `141_C_LoRA_Continuous_Multi_LoRA_Training.md` | Multi-LoRA fused dispatch; GPart single-pass replaces two-matmul bottleneck |
| `158_MUX_Multiplexed_Latent_Reasoning.md` | MUX superposition Idea 3 builds on this |
| `110_Ciot_Ternary_Inference_CPU_Distillation.md` | PlasmaPath ternary quantization composes with GPart pruning |
| `037_REAP_Model-Based_Modelless_Duality.md` | Model-based/modelless spectrum — GPart is modelless inference loading |
| `062_SHINE_Scalable_In_Context_Hypernetwork.md` | Alternative inference-time adaptation approach |
| `226_Browser_Inference_WebGPU_WASM_SIMD_Verdict.md` | WASM SIMD dispatch — GPart apply is SIMD-friendly (broadcast) |
| `140_sigmoid_parallax.md` | Sigmoid not softmax — GPart uses sigmoid for activation gating in Idea 2 |

### Code References

- `katgpt-core/src/shard_embedding.rs` — `JlProjectionMatrix` (same math family)
- `katgpt-core/src/types.rs` — `LoraAdapter`, `lora_apply()`, `LoraPair`
- `katgpt-core/src/simd.rs` — SIMD dispatch tiers (GPart apply uses `simd_dot_f32`)
- `katgpt-core/tests/bench_230_shard_embedding_goat.rs` — GOAT benchmark template

---

TL;DR: **GPart's isometric partition is the same math as our JL projection. Implement as `GpartAdapter` behind feature gate. 2–4× storage compression at modelless inference time. Ideas 2–4 deferred. GOAT-prove before promoting.**