# Issue 042: `FunctionSpaceEncoderDecoder` Trait — Re-Examination (Research 311 Already Dropped the Encoder Half)

> **Spawned from:** Research 395 (NNs → NOs Function-Space Operator Learning Recipe — Pass)
> **Confidence:** LOW-MEDIUM that the trait is net-positive. Research 311 already evaluated and **explicitly dropped** a closely-related `AnalyticLatticeEncoder` trait as redundant. This issue is a re-examination, not a green-lit refactor.
> **Date:** 2026-07-09
> **Status:** OPEN (investigation only — do NOT impl)

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

**Recommendation: do not impl now.** Two reasons:

1. **Research 311 already rejected the encoder half.** Its TL;DR (lines 49–54) states verbatim: *"the `AnalyticLatticeEncoder` trait originally proposed here is redundant with `FourierEncoder::encode_*_into` which already ships closed-form `entity → [f32; N]` encoding. We do NOT re-ship a parallel encoder API."* A full-pipeline trait re-opens a decision that was deliberately closed.

2. **The four instances have genuinely-different shapes** (see §3). A single trait generic enough to cover all four loses the specialization that makes each one fast/correct (Tikhonov regularization, frozen BLAKE3 basis, topological boundary matrix). High risk of "false DRY" — a leaky abstraction over fundamentally different implementations.

This issue exists to (a) record the DRY observation from Research 395, (b) capture the Research 311 precedent so it isn't re-litigated blindly, and (c) define a concrete PoC bar that must clear before any impl.

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

**Implication:** the only genuinely-unexplored part of the encoder-decoder unification is whether `funcattn_forward`, `transport_cross_resolution_into`, and DEC `exterior_derivative` should share a *full-pipeline* trait (encode+map+decode). The encoder half alone is settled (use `FourierEncoder`).

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

## 4. The PoC bar (mandatory before any impl)

Per AGENTS.md GOAT-gate discipline and the "do not create plan for refactor" rule, before promoting this from "issue" to "plan", a PoC must show **all** of:

1. **Real shared logic** — ≥2 of the 4 instances share extractable non-trivial code (not just a signature). Candidate: the inner-product encode `v[j] = Σ_i basis[j]* · f_i · Δ_i` appears in both `funcattn` and `cross_resolution`; verify whether the bodies are genuinely duplicated or already share a helper.
2. **No G4 regression** — the trait abstraction does not force allocation on the `funcattn` / `cross_resolution` hot paths (both currently G4-zero-alloc).
3. **No specialization loss** — Tikhonov regularization (funcattn PD guarantee), BLAKE3 basis commitment (cross_resolution freeze/thaw invariant), and boundary-matrix topology (DEC `d∘d=0`) all remain expressible through the trait without per-impl escape hatches that defeat the abstraction.
4. **Net LoC reduction** — the trait + 4 impls is shorter than the 4 free functions they replace (otherwise it's surface-area bloat, not DRY).

If the PoC fails any gate, the issue closes as "settled by Research 311 — encoder half redundant, full-pipeline trait is false DRY."

---

## 5. Where the PoC would live

`riir-ai/crates/riir-poc/benches/encoder_decoder_trait_re_examination.rs` — the defend-wrong R&D crate. Compare:
- **A (status quo):** the four free functions as they ship today.
- **B (candidate):** the proposed `FunctionSpaceEncoderDecoder` trait + 4 impls.
- **Floor:** `FourierEncoder::encode_*_into` alone (the encoder-half baseline Research 311 already established as sufficient).

Metric: lines of code (DRY), and a micro-bench confirming zero-alloc / no-regression on the funcattn + cross_resolution paths.

---

## Tasks (tracking only — no impl)

- [ ] **T1** Grep `funcattn.rs` and `cross_resolution.rs` for the encode inner-product body; determine whether the code is genuinely duplicated or already shares a helper. (Determines whether gate 1 of §4 is even reachable.)
- [ ] **T2** Sketch the trait signature required to cover all 3 shapes (§3). If the signature degrades to `fn encode/decode(&self, ...) -> Vec<f32>`, document the G4 violation and close the issue.
- [ ] **T3** (only if T1+T2 pass) Write the PoC at §5; run G4 zero-alloc check on funcattn + cross_resolution paths.
- [-] **T4** (won't-do unless T1–T3 pass) Promote to a `.plans/` file with the `FunctionSpaceEncoderDecoder` trait impl. Blocked on T1/T2/T3.
- [ ] **T5** If the issue closes as "false DRY", add a one-line note to Research 395 §2.2 pointing forward to this issue's resolution, so the DRY observation isn't re-raised on the next re-read of the paper.

---

## Cross-references

- **Research 395** (`katgpt-rs/.research/395_NNs_to_NOs_Function_Space_Operator_Learning_Recipe.md`) — spawned this issue (Pass verdict, DRY observation §2.2).
- **Research 311** (`katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md`) — **the precedent**. TL;DR L49–54 explicitly dropped `AnalyticLatticeEncoder` as redundant with `FourierEncoder`. Any resolution of this issue must reconcile with R311.
- **Plan 330** (`katgpt-rs/.plans/330_analytic_lattice_encoder_decoder_primitive.md`) — shipped `analytic_lattice` (`compose_chain`, `TransportOperator`, `direction_vector_decode`, `RederiveOp`). The closest existing abstraction; operates at k×k-matrix layer, not full pipeline.
- **Plan 286** — `funcattn_forward` (`katgpt-rs/crates/katgpt-core/src/funcattn.rs`).
- **Plan 310** — `transport_cross_resolution_into` (`katgpt-rs/crates/katgpt-core/src/cross_resolution.rs`), DEFAULT-ON.
- **Plan 251** — DEC operators (`katgpt-rs/crates/katgpt-dec/src/operators.rs`).
- **Issue 041** (`katgpt-rs/.issues/041_smooth_min_similarity_no_consumer_poc_gate.md`) — sibling "do not impl until PoC clears" issue; same discipline.

---

## TL;DR

**Do not impl.** Research 395 surfaced the encoder-decoder DRY pattern from arXiv:2506.10973, but Research 311 **already evaluated and explicitly dropped** the encoder-half trait (`AnalyticLatticeEncoder`) as redundant with `FourierEncoder`, and the four shipped instances (`funcattn`, `cross_resolution`, DEC `exterior_derivative`, DeepONet-via-funcattn) have genuinely-different shapes (symmetric-learned / asymmetric-frozen / topological-incidence / no-decode) that resist a single zero-alloc trait. Re-open only if T1–T3 PoC gates clear; otherwise close as "settled by R311 + false-DRY."
