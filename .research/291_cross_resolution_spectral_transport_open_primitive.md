# Research 291: Cross-Resolution Spectral Transport — Continuous-Field FUNCATTN for Asymmetric Bases

> **Source:** Synthesized from FUNCATTN (arxiv 2605.31559, Research 257), Topological Neural Operators (arxiv 2606.09806, Research 219), and the Gemini "continuous field over a manifold" reframing (2026-06-23)
> **Date:** 2026-06-23
> **Status:** Active
> **Related Research:** 257 (FUNCATTN — symmetric basis transport), 219 (TNO/DEC — topological operators), 280 (Resolution-Tiered Deterministic Commitment)
> **Related Plans:** 286 (FUNCATTN — symmetric), 310 (this primitive)
> **Cross-ref (riir-neuron-db):** Research 004 (Cross-Resolution Shard Transport Guide)
> **Classification:** Public

---

## TL;DR

Cross-Resolution Spectral Transport extends FUNCATTN's `k×k` operator to
**asymmetric bases** (`Φ_src ∈ R^{d_src × k}`, `Ψ_dst ∈ R^{d_dst × k}`), enabling
**train-on-small-deploy-on-large**: train a personality on a plasma-tier 16-dim
shard, deploy it on a cold-tier 256-dim shard without retraining. The math is
the FUNCATTN closed-form Tikhonov solve generalized to `d_src ≠ d_dst`; the
capability (cross-resolution transfer preserving behavior rankings) is genuinely
new for us.

**Distilled for katgpt-rs (modelless, inference-time):**
A `CrossResolutionTransport` primitive that takes a source latent slice +
source basis + destination basis, and produces a destination latent slice via
the existing FUNCATTN `solve_convex_combo_dual` + asymmetric projections. No
training; bases are frozen BLAKE3-committed artifacts.

---

## 1. Mechanism

### 1.1 The reframe

Research 257 §1.3 Theorem A.3 already notes FUNCATTN *"is a Monte-Carlo
discretization of a regularized integral operator"* — i.e., the continuous-field
interpretation is in the paper. Research 257 §1.3 also notes **resolution
invariance** (linear-in-n; train at n=2048, test at n=8192).

But the shipped FUNCATTN primitive (`katgpt-rs/crates/katgpt-core/src/funcattn.rs`)
is **symmetric** — `d_src = d_dst = d`. The G2 benchmark
(`tests/funcattn_g2_funcattn_vs_parallax_vs_sdpa.rs`) only tests same-dim.
**Cross-resolution transfer (different d on input vs output) is unshipped.**

The Gemini "continuous field over a manifold" framing points directly at this
gap: a continuous field can be projected onto any basis, at any resolution. The
mathematical operator C is resolution-invariant; the bases Φ, Ψ are what
specialize it to a particular d.

### 1.2 The math

FUNCATTN's transport (Research 257 §1.2):

```
slice_token = Ψ^T · X        (d_dst × k, projecting to spectral)
C            = Q̃ · (K̃^T K̃ + αI)^{-1} · K̃^T   (k × k, the operator)
out          = Φ · C · slice_token   (d_src × 1, reconstructing)
```

For **symmetric** FUNCATTN, `Φ = Ψ` and `d_src = d_dst`. For **cross-resolution**:

```
Φ_src ∈ R^{d_src × k}   (e.g., d_src = 16, plasma tier)
Ψ_dst ∈ R^{d_dst × k}   (e.g., d_dst = 256, cold tier)
```

The operator C stays `k × k` (resolution-invariant). The source slice projects
via `Ψ_dst^T` (NOT `Ψ_src^T`) to get the spectral coefficients, then
reconstructs via `Φ_src`. Wait — this needs care.

**Correct asymmetric formulation:**

Given a **source latent state** `s ∈ R^{d_src}` and we want to produce a
**destination latent state** `t ∈ R^{d_dst}` that is the "same field, different
resolution":

1. Project source to spectral: `a = Φ_src^T · s` (k-dim spectral coefficients).
   Requires `Φ_src ∈ R^{d_src × k}` with `Φ_src^T Φ_src = I_k` (orthonormal
   columns).
2. Reconstruct at destination resolution: `t = Ψ_dst · a` (d_dst-dim).
   Requires `Ψ_dst ∈ R^{d_dst × k}` with `Ψ_dst^T Ψ_dst = I_k`.

If the field is genuinely band-limited to k components, this is exact (Parseval).
If not, this is the least-squares projection — exactly the FUNCATTN ridge
regularization interpretation.

**Crucially:** the FUNCATTN operator C (which transports between two manifolds
*of the same resolution*) is **separate from** the cross-resolution projection
above. The two compose:

- Cross-resolution projection (`Φ_src^T`, `Ψ_dst`): change resolution, preserve
  band-limited structure.
- FUNCATTN C operator: transport between semantic domains (e.g., emotion →
  behavior) *within a resolution*.

The full cross-resolution cross-domain transport is:
`t = Ψ_dst · C · Φ_src^T · s` — a clean 4-matrix product, all small (`k ≪ d`).

### 1.3 Why this works modellessly

Both bases (`Φ_src`, `Ψ_dst`) are **frozen learned artifacts** — trained offline
(via standard orthogonalized linear projection on a corpus of paired
small-dim/large-dim shards), committed via BLAKE3, loaded at init. The runtime
transport is four matmuls + the existing closed-form C solve. No training at
inference time. No gradients.

This is exactly the freeze/thaw pattern: the bases are the frozen part, the
transport is the inference-time operation.

### 1.4 Resolution invariance, formally

If `Φ_src` and `Ψ_dst` are learned from the **same underlying functional
subspace** (just discretized at different resolutions), then for any band-limited
field `f`:

```
||Ψ_dst · Φ_src^T · f_src − f_dst|| ≤ λ · ||f_high_freq||
```

where `f_high_freq` is the component of `f` above the k-th basis function. This
is the Lipschitz bound from Research 257 §1.3 Prop 4.5, restated for asymmetric
bases. **The transport is lossless for band-limited fields; lossy only for
high-frequency content that exceeds the basis rank k.**

---

## 2. Distillation

### 2.1 Transferable primitive

1. `CrossResolutionBases { phi_src: Matrix(d_src, k), psi_dst: Matrix(d_dst, k) }`
   — frozen, BLAKE3-committed.
2. `transport_cross_resolution(src_state: &[f32], bases: &CrossResolutionBases)
   -> Vec<f32>` — projects src to k-dim spectral, reconstructs at dst dim.
3. `transport_cross_resolution_into(src_state, bases, dst_scratch: &mut [f32])`
   — zero-alloc variant, writes into pre-allocated scratch.
4. `transport_cross_domain_cross_resolution(src_state, bases, c_op: &Matrix(k,k),
   dst_scratch)` — full 4-matrix product for cross-resolution + cross-domain.

### 2.2 Where the pieces already live

| Piece | Existing location | Reuse |
|---|---|---|
| Tikhonov / convex-combo solve | `funcattn.rs::solve_convex_combo_dual` | ✅ same solver |
| Sigmoid-normalized basis | `funcattn.rs::compute_basis_into` | ✅ extend to asymmetric |
| Partition-of-unity check | `funcattn.rs` tests `basis_rows_partition_of_unity` | ✅ same check, both bases |
| Cholesky solve | `funcattn.rs::cholesky_solve_into` | ✅ same |
| BLAKE3 commitment | `MerkleFrozenEnvelope` (riir-neuron-db/src/freeze.rs) | ✅ same envelope for bases |
| Frozen artifact loading | `LoRAHotSwap`, `EmotionDirections` loader | ✅ same pattern |
| Pre-allocated scratch | `FuncAttnScratch` pattern | ✅ extend with `d_dst` slots |

**The math is 95% shipped.** What's new: asymmetric d, separate Φ_src/Ψ_dst
commitment, and the operationalization of cross-resolution transfer as a
first-class capability (vs the current implicit same-d assumption).

### 2.3 Closest cousins (3)

1. **FUNCATTN (R257, P286)** — same operator, same solver, symmetric d only.
   Cross-resolution is the strict generalization.
2. **Deep Manifold (R051, P085)** — fixed-point boundary conditions on
   manifolds. Touches the manifold-geometry angle but doesn't do
   cross-resolution.
3. **Resolution-Tiered Deterministic Commitment (R280)** — resolution tiering
   for chain commitment. Different domain (chain, not latent) but same
   "tier-as-resolution" thesis.

### 2.4 Fusion

**F1 (PRIMARY — riir-neuron-db): Cross-Resolution × NeuronShard tier transfer**
The headline. Train a personality shard on plasma-tier hardware (16-dim
`style_weights`), deploy on cold-tier (256-dim `style_weights`) via
cross-resolution transport. **No retraining.** Today, shards are fixed-size
(`STYLE_DIM = 64`); cross-resolution makes the shard size match the hardware
tier. Unblocks: tier-aware deployment, personality migration across hardware
generations.

**F2 (SECONDARY — katgpt-rs): Cross-Resolution × FUNCATTN attention**
FUNCATTN at attention time can route through a smaller basis for hot-path
compute, then reconstruct at full d for downstream layers. Plasma/hot tier uses
k=4 basis; warm/cold tier uses k=16. Same personality, different fidelity.

**F3 (TERTIARY — speculative): Cross-Resolution × Apollonian geometry**
If Apollonian sphere packings (Issue 001) provide the natural multi-resolution
basis, cross-resolution transport becomes a change of Apollonian scale rather
than a change of d. Speculative — depends on Issue 001 resolving with a
concrete use case.

---

## 3. Verdict

### Tier: **Super-GOAT (candidate — pending G1–G4 validation)**

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **Partial-but-novel.** FUNCATTN (R257) ships the symmetric `k×k` operator; cross-resolution (`d_src ≠ d_dst`) is unshipped — G2 benchmark only tests same-d. The continuous-field reframing + asymmetric basis extension is novel *as an operationalized capability*. Research 219 (TNO/DEC) touches topology but not cross-resolution transfer. | Vocabulary translation: "continuous field" → "asymmetric basis projection", "resolution invariant" → "cross-d transport", "functional map" → "spectral coefficient transfer". |
| Q2 New class of behavior? | **YES.** Today, `NeuronShard::style_weights` is fixed-size (STYLE_DIM=64). Cross-resolution transfer enables **train-once-deploy-across-tiers**: 16-dim plasma, 64-dim hot, 256-dim cold, all from one trained personality. No incumbent can do this — they'd retrain per tier. | |
| Q3 Product selling point? | **YES.** "Train NPC personality once on cheap hardware; deploy across plasma/hot/warm/cold tiers without retraining. Same personality, different fidelity, automatic." Concrete, demoable, hard to replicate without our full stack. | |
| Q4 Force multiplier? | **YES.** Connects FUNCATTN (R257) + NeuronShard + freeze/thaw + plasma/hot/warm/cold tiering + R280 (Resolution-Tiered Commitment) + ShardIndex + plan 255 (ANE latent brain compute). ≥6 pillars. | |

**Selling point:** Train-once-deploy-across-tiers — one personality shard,
multiple hardware fidelities, no retraining.

**Not Super-GOAT if:** G1 (reconstruction cos) <0.85 — if cross-resolution
transport loses too much information, the transfer is useless and the primitive
demotes to Gain (math curiosity, no deployed use).

### Routing

- **katgpt-rs/.plans/310_cross_resolution_spectral_transport_primitive.md** —
  open primitive. Asymmetric basis transport + zero-alloc scratch. Feature flag
  `cross_resolution_transport`. GOAT gate G1–G4.
- **riir-neuron-db/.research/004_cross_resolution_shard_transport_guide.md** —
  private guide (this Super-GOAT's selling-point doc, shard domain).
- **riir-neuron-db/.plans/** — deferred until katgpt-rs primitive passes G1–G2.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Bases are frozen artifacts; transport is matmuls + closed-form solve. |
| Latent-to-latent preferred | ✅ Operates entirely in latent space; never crosses to tokens. |
| Use sigmoid not softmax | ✅ Bases use sigmoid-normalized rows (same as FUNCATTN §2.4 mandate). |
| Freeze/thaw over fine-tuning | ✅ Bases are BLAKE3-committed; per-tier bases are atomic Arc-swap. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; shard integration → riir-neuron-db. |
| Raw scalars at sync boundary | ✅ Transport stays latent; sync crosses only via existing scalar projections. **The bases themselves are committed artifacts (LatCal/BLAKE3) but the transported vectors are local latent state.** |
| Zero-alloc hot path | ✅ All matmuls into pre-allocated scratch (`CrossResScratch` mirroring `FuncAttnScratch`). |

---

## 5. Open questions / risks

1. **Does cross-resolution preserve behavior rankings?** The headline risk.
   Transporting 16→256 must preserve which actions the NPC would select.
   **Mitigation:** G2 measures cosine similarity of action rankings
   pre/post-transport; gate requires ≥0.85 (looser than latent steering's 0.95
   because some information loss is expected — the question is whether the
   *ranking* survives).
2. **How to learn the bases?** Bases are trained offline. Need a corpus of
   paired (small-d, large-d) shards where both are derived from the same
   underlying personality. **Mitigation:** generate synthetically by taking
   existing 64-d shards, projecting to 16-d via random projection, then learn
   the inverse map. riir-train territory but the bases themselves are
   inference-time artifacts.
3. **What k?** Rank k trades off fidelity vs compute. k=4 may suffice for
   personality (which is low-rank per Research 257 §5.5); k=16 may be needed
   for fine-grained style. **Mitigation:** G3 sweeps k ∈ {4, 8, 16, 32}.
4. **Numerical conditioning of asymmetric Φ^T Φ.** If `d_src < k`, `Φ_src^T
   Φ_src` is rank-deficient. **Mitigation:** require `k ≤ min(d_src, d_dst)`;
   enforce in constructor; document.

---

## TL;DR

Cross-Resolution Spectral Transport generalizes FUNCATTN to asymmetric bases
(`d_src ≠ d_dst`), enabling train-once-deploy-across-tiers: train a personality
on 16-d plasma, deploy on 256-d cold, no retraining. Math is 95% shipped
(symmetric FUNCATTN + Tikhonov solve + sigmoid basis); the asymmetric extension
+ operationalization is the novel capability. Super-GOAT candidate pending G1–G4:
reconstruction cos, behavior rank preservation, k sweep, zero-alloc. Kills itself
if G1 <0.85 (information loss) or G2 <0.85 (ranking corruption). Plan 310; guide
at `riir-neuron-db/.research/004`.
