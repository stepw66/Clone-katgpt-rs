# Issue 042: `FunctionSpaceEncoderDecoder` Trait — CLOSED (False-DRY Confirmed by T1)

> **Spawned from:** Research 395 (NNs → NOs Function-Space Operator Learning Recipe — Pass)
> **Confidence:** RESOLVED — the trait is net-negative. T1 (encode-body diff) shows the inner-product encode is mathematically identical across `funcattn` × `cross_resolution` but **not extractable as shared code** (different batching, memory layouts, SIMD primitives, fusion). Settles alongside Research 311's earlier drop of `AnalyticLatticeEncoder`.
> **Date:** 2026-07-09
> **Status:** CLOSED (2026-07-09) — false-DRY confirmed; no impl, no plan.

---

## TL;DR

Research 395 distilled arXiv:2506.10973 (Berner/Liu-Schiaffini/Kossaffi/Anandkumar, NVIDIA+Caltech — the progenitor recipe paper for the FNO/GNO/SFNO/DeepONet family). Verdict: **Pass** (0/4 novelty gate). All modelless content is subsumed by shipped primitives.

The paper's one notable DRY observation (§3.6, Appendix A.8): the **encoder-decoder operator pattern** —

```
encode:  v[j] = Σ_i basis[j](x_i)* · f(x_i) · Δ_i   // inner-product project → R^k
map:     w = K · v                                    // k×k latent linear map
decode:  g(y) = Σ_j basis[j](y) · w[j]                // reconstruct
```

— explicitly unifies four shipped instances in our codebase as special cases. The question: should we extract a single `FunctionSpaceEncoderDecoder` trait?

**Verdict: CLOSED — do not impl.** Three converging reasons:

1. **Research 311 already rejected the encoder half.** Its TL;DR (lines 49–54) states verbatim: *"the `AnalyticLatticeEncoder` trait originally proposed here is redundant with `FourierEncoder::encode_*_into` which already ships closed-form `entity → [f32; N]` encoding. We do NOT re-ship a parallel encoder API."* A full-pipeline trait re-opens a decision that was deliberately closed.

2. **The four instances have genuinely-different shapes** (see §3). A single trait generic enough to cover all four loses the specialization that makes each one fast/correct (Tikhonov regularization, frozen BLAKE3 basis, topological boundary matrix). High risk of "false DRY" — a leaky abstraction over fundamentally different implementations.

3. **T1 (this session, 2026-07-09) confirmed the false-DRY empirically.** The encode inner-product body `v = Φᵀ · f` is mathematically identical in `funcattn` and `cross_resolution`, but the implementations differ on every axis that matters for extraction: batching `(n,d)→(k,d)` vs `(d,)→(k,)`, memory layout (contiguous vs strided columns → forces different SIMD primitives), and fusion (funcattn fuses encode+normalize; cross_resolution is bare). There is **no shared code to extract** — a trait would degrade to generic matmul. See §6 (Resolution).

This issue exists to (a) record the DRY observation from Research 395, (b) capture the Research 311 precedent so it isn't re-litigated blindly, (c) define the PoC bar (§4), and (d) **record the T1 findings that close it** (§6).

---

## 1. The DRY observation (from Research 395 §2.2)

Paper §3.6 / Appendix A.8 shows that spectral convolution, GNO (integral transform), DeepONet, and the encoder-decoder operator are all instances of the same 3-step pattern. In our vocabulary, four shipped primitives are instances:

| Instance | Encode | Map `K` | Decode | File |
|---|---|---|---|---|
| `funcattn_forward` | `Φ^T Q` (sigmoid basis, **learned**) | Tikhonov-regularized solve `C = Q̃·reg⁻¹·K̃ᵀ` | `Ψ w` (symmetric `Φ=Ψ`) | `katgpt-rs/crates/katgpt-core/src/funcattn.rs` |
| `transport_cross_resolution_into` | project onto basis A (**frozen, BLAKE3**) | pre-computed `K` | reconstruct via basis B (**asymmetric `Φ≠Ψ`**) | `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs` |
| DEC `exterior_derivative` | boundary matrix `Bₖ₊₁ᵀ` (**topological**, not learned) | identity (the `d` operator IS the map) | none (map output is the result) | `katgpt-rs/crates/katgpt-dec/src/operators.rs` |
| (DeepONet-style) | branch net `b(f)` | `K` | trunk net `t(y)` | **not shipped as a literal primitive** — covered by funcattn + cross_resolution per Research 395 §2.2 |

The first three are real code with real, divergent implementations.

---

## 2. The Research 311 precedent (the reason this is contested)

Research 311 (`katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md`) — the Super-GOAT that produced Plan 330 (`analytic_lattice`) — **already considered** a unifying encoder trait and **explicitly dropped it**:

> **Redundant (intentionally dropped):** the `AnalyticLatticeEncoder` trait originally proposed here is redundant with `FourierEncoder::encode_*_into` which already ships closed-form `entity → [f32; N]` encoding. We do NOT re-ship a parallel encoder API. The decoder primitive `direction_vector_decode` remains novel as a *generalization* of `riir-games::scalar_projection` out of HLA-specific 5-scalar semantics into a generic single-direction primitive.

Research 311's resolution:
- **Encode half** → use existing `FourierEncoder::encode_position_into` / `encode_offset_into` (`riir-ai/crates/riir-engine/src/fourier/encoder.rs`). No new trait.
- **Decode half** → ship `direction_vector_decode` as a *generic single-direction primitive* (generalizing HLA's `scalar_projection`), NOT as a full encoder-decoder trait.
- **Map half** → ship `TransportOperator` (k×k matrix) + `compose_chain` / `batch_compose_chain`. The `RederiveOp` trait shapes the *async production* of a `TransportOperator`, not the encode→map→decode pipeline.

So `analytic_lattice` already operates at the **k×k-matrix + decode** layer — one layer below the full pipeline. `RederiveOp` is `rederive(&self, ctx) -> Fut` where `Fut: GpuFuture<Output = TransportOperator>`; it is **not** an encoder-decoder abstraction.

**Implication:** the only genuinely-unexplored part of the encoder-decoder unification was whether `funcattn_forward`, `transport_cross_resolution_into`, and DEC `exterior_derivative` should share a *full-pipeline* trait (encode+map+decode). The encoder half alone was settled (use `FourierEncoder`); T1 (this session) settles the full-pipeline question.

---

## 3. Why the four instances resist a single trait (the "false DRY" risk)

| Axis | `funcattn` | `cross_resolution` | DEC `exterior_derivative` |
|---|---|---|---|
| Basis | learned sigmoid, **symmetric** (`Φ=Ψ`) | **frozen** BLAKE3, **asymmetric** (`Φ≠Ψ`) | topological **boundary matrix** (not a basis at all — it's an incidence operator) |
| Map `K` | **Tikhonov-regularized solve** (regularization is load-bearing for PD guarantee) | pre-computed, applied directly | identity — the `d` operator is itself the map |
| Decode | `Ψ w` reconstruction | basis-B reconstruction | **none** — output is the map result, no decode-back |
| Allocation discipline | zero-alloc via `FuncAttnScratch` (G4 gate) | zero-alloc via `CrossResScratch` (G4 gate) | zero-alloc via `CochainField::zeros` + `_into` variants |
| Feature gate | `funcattn` | `cross_resolution_transport` (**DEFAULT-ON**) | `dec_operators` |

A trait covering all three must abstract over: symmetric-vs-asymmetric-vs-incidence basis, regularized-solve-vs-direct-vs-identity map, and reconstruct-vs-none decode. At that generality the trait is `fn encode(&self, f) -> Vec<f32>; fn map(&self, v) -> Vec<f32>; fn decode(&self, w) -> CochainField;` — i.e. three `&self` methods returning owned `Vec`s, which **violates the zero-alloc G4 gates** both `funcattn` and `cross_resolution` enforce. Making it zero-alloc requires `&mut scratch` parameters, at which point the trait signature is no simpler than the current free functions.

This is the canonical false-DRY shape: a shared interface over implementations whose specialization (Tikhonov regularization, frozen basis, topological incidence) is exactly what makes each one correct/fast.

---

## 4. The PoC bar (defined before T1 ran)

Per AGENTS.md GOAT-gate discipline and the "do not create plan for refactor" rule, before promoting this from "issue" to "plan", a PoC must have shown **all** of:

1. **Real shared logic** — ≥2 of the 4 instances share extractable non-trivial code (not just a signature). Candidate: the inner-product encode `v[j] = Σ_i basis[j]* · f_i · Δ_i` appears in both `funcattn` and `cross_resolution`; verify whether the bodies are genuinely duplicated or already share a helper.
2. **No G4 regression** — the trait abstraction does not force allocation on the `funcattn` / `cross_resolution` hot paths (both currently G4-zero-alloc).
3. **No specialization loss** — Tikhonov regularization (funcattn PD guarantee), BLAKE3 basis commitment (cross_resolution freeze/thaw invariant), and boundary-matrix topology (DEC `d∘d=0`) all remain expressible through the trait without per-impl escape hatches that defeat the abstraction.
4. **Net LoC reduction** — the trait + 4 impls is shorter than the 4 free functions they replace (otherwise it's surface-area bloat, not DRY).

**Gate 1 failed (T1, this session) → issue closes.** See §6.

---

## 5. Where the PoC would have lived (not written — gate 1 failed)

`riir-ai/crates/riir-poc/benches/encoder_decoder_trait_re_examination.rs` — the defend-wrong R&D crate. Compare:
- **A (status quo):** the four free functions as they ship today.
- **B (candidate):** the proposed `FunctionSpaceEncoderDecoder` trait + 4 impls.
- **Floor:** `FourierEncoder::encode_*_into` alone (the encoder-half baseline Research 311 already established as sufficient).

Metric: lines of code (DRY), and a micro-bench confirming zero-alloc / no-regression on the funcattn + cross_resolution paths.

**Not written** — T1 closed the issue before a PoC was warranted.

---

## 6. Resolution — T1 findings (2026-07-09)

**T1 executed:** diffed the encode inner-product body across `funcattn.rs` and `cross_resolution.rs`.

### funcattn encode (Stage 2+3, `funcattn.rs` L878–898)

```rust
// slice_token[g,:] = Σ_n Φ[n,g]·x_value[n,:], fused with col_sum accumulation
for i in 0..n {
    let phi_row = &scratch.phi[i * k..(i + 1) * k];
    let x_row   = &x_value[i * d..(i + 1) * d];
    simd::simd_outer_product_acc(&mut scratch.slice_token, phi_row, x_row, k, d);
}
// then normalize by col_sum[g] (eps-guarded)
```

### cross_resolution encode (`project_to_spectral_into`, L242–265)

```rust
// spectral[j] = Σ_r phi_src[r*k + j] * src_state[r]  (strided-column dot)
for j in 0..k {
    let mut acc = 0.0f32;
    for r in 0..d_src {
        acc += bases.phi_src[r * k + j] * src_state[r];
    }
    spectral[j] = acc;
}
```

### Verdict: mathematically identical, not extractable

| Axis | `funcattn` encode | `cross_resolution` encode |
|---|---|---|
| Math | `slice_token = Φᵀ · x_value` | `spectral = Φ_srcᵀ · src_state` |
| Input shape | `(n, d)` → encode **n tokens at once** | `(d_src,)` → **single vector** |
| SIMD primitive | `simd_outer_product_acc` (fused outer-product accumulate) | manual `for j { for r { acc += } }` strided-dot |
| Memory layout | Φ is `(n,k)` | `phi_src` is `(d_src,k)` row-major → columns **strided** |
| Fusion | fused with col_sum normalization (Stage 2+3) | bare encode, no fusion |
| Pipeline role | 1 of 7 stages (then Tikhonov solve, Q̃, C·Ṽ, inverse-project) | 1 of 2 stages (encode → decode) |

The encode math `v = Φᵀ · f` is the same in both. **But there is no shared code to extract:**

- **Batching differs** — funcattn encodes an `(n,d)` token batch down to `(k,d)`; cross_resolution encodes a single `(d,)` vector down to `(k,)`. A unified signature must abstract over both shapes.
- **Layout differs** — funcattn's `Φ` is `(n,k)` (contiguous-friendly); cross_resolution's `phi_src` is `(d_src,k)` row-major, making every column **strided**. This forces different SIMD primitives (`simd_outer_product_acc` vs a manual gather-dot). LLVM auto-unrolls the short strided inner loop in cross_resolution; it cannot do so for the batched funcattn path.
- **Fusion differs** — funcattn fuses encode with `col_sum` accumulation (Stage 2+3, the partition-of-unity normalization is load-bearing for its PD guarantee); cross_resolution is a bare projection with no fusion target.

A shared `encode()` trait method abstracting over all three axes degrades to `fn encode(&self, input: &Matrix, basis: &Matrix) -> Matrix` — i.e. **generic matmul**, not a meaningful encoder abstraction. At that point the "trait" is a re-statement of `TransportOperator` (the k×k map layer Plan 330 already ships), which is precisely where Research 311 landed.

**Gate 1 of §4 fails. Per the issue's own rule, the issue closes as "settled by Research 311 — encoder half redundant, full-pipeline trait is false DRY."**

DEC `exterior_derivative` was not separately diffed — its encode is a boundary-matrix application (topological incidence), not an inner-product projection, so it shares even less with the other two (no basis, no decode). It cannot rescue the trait's generality.

---

## Tasks

- [x] **T1** (DONE 2026-07-09) Diffed `funcattn.rs` Stage 2+3 (L878–898) vs `cross_resolution.rs::project_to_spectral_into` (L242–265). **Result: no extractable shared code** — encode math identical but batching / layout / SIMD primitive / fusion all differ. Gate 1 of §4 FAILS. See §6.
- [x] **T2** (DONE — moot) Gate 1 failure obviates the trait-signature sketch; the conclusion (signature degrades to generic matmul) is documented in §6 by direct inspection rather than by sketching.
- [-] **T3** (CLOSED — gate 1 failed) PoC not written; no candidate trait to benchmark.
- [-] **T4** (CLOSED — gate 1 failed) No `.plans/` promotion; the `FunctionSpaceEncoderDecoder` trait will not ship.
- [x] **T5** (DONE 2026-07-09) Research 395 §2.2 DEC path typo fixed (`katgpt-core/src/dec/` → `katgpt-dec/src/`); this closed issue is the forward-pointer so the DRY observation isn't re-raised on re-read.

---

## Cross-references

- **Research 395** (`katgpt-rs/.research/395_NNs_to_NOs_Function_Space_Operator_Learning_Recipe.md`) — spawned this issue (Pass verdict, DRY observation §2.2). DEC path typo fixed in the same session.
- **Research 311** (`katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md`) — **the precedent**. TL;DR L49–54 explicitly dropped `AnalyticLatticeEncoder` as redundant with `FourierEncoder`. T1 independently re-confirms for the full-pipeline case.
- **Plan 330** (`katgpt-rs/.plans/330_analytic_lattice_encoder_decoder_primitive.md`) — shipped `analytic_lattice` (`compose_chain`, `TransportOperator`, `direction_vector_decode`, `RederiveOp`). The closest existing abstraction; operates at k×k-matrix layer, not full pipeline.
- **Plan 286** — `funcattn_forward` (`katgpt-rs/crates/katgpt-core/src/funcattn.rs`).
- **Plan 310** — `transport_cross_resolution_into` (`katgpt-rs/crates/katgpt-core/src/cross_resolution.rs`), DEFAULT-ON.
- **Plan 251** — DEC operators (`katgpt-rs/crates/katgpt-dec/src/operators.rs`).
- **Issue 041** (`katgpt-rs/.issues/041_smooth_min_similarity_no_consumer_poc_gate.md`) — sibling "do not impl until PoC clears" issue; same discipline.

---

## TL;DR

**CLOSED — false-DRY confirmed by T1.** Research 395 surfaced the encoder-decoder DRY pattern from arXiv:2506.10973, but Research 311 **already evaluated and explicitly dropped** the encoder-half trait (`AnalyticLatticeEncoder`) as redundant with `FourierEncoder`, and T1 (this session) diffed the actual encode bodies in `funcattn.rs` × `cross_resolution.rs` to find the inner-product encode is mathematically identical but **not extractable as shared code** — different batching `(n,d)→(k,d)` vs `(d,)→(k,)`, different memory layouts (contiguous vs strided columns → different SIMD primitives), different fusion (encode+normalize vs bare). A unified trait degrades to generic matmul (= the `TransportOperator` layer Plan 330 already ships). DEC `exterior_derivative` shares even less (topological incidence, no basis, no decode). No impl, no plan; this issue is the record so the DRY observation isn't re-raised on the next re-read of the paper.
