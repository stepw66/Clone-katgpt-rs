# Plan 286: Functional Attention — Spectral Transport Operator (Open Primitive)

**Date:** 2026-06-17
**Research:** [257_Functional_Attention_Spectral_Transport_Operator](../.research/257_Functional_Attention_Spectral_Transport_Operator.md)
**Source paper:** [arxiv 2605.31559](https://arxiv.org/pdf/2605.31559) — Functional Attention: From Pairwise Affinities to Functional Correspondences (Xiao et al., ICML 2026)
**Target:** `crates/katgpt-core/src/funcattn.rs` (new module) + Cargo feature `funcattn`
**Status:** Active — Phase 0 (spec)
**Tier:** Gain (open primitive; await GOAT proof before opt-in promotion; **do not promote to default** until LLM-domain evidence exists)

---

## Goal

Ship Functional Attention (FUNCATTN) as a new attention operator in katgpt-rs. **The paper's math, not the paper's softmax basis** — per AGENTS.md we use sigmoid-normalized basis (partition-of-unity property holds for any row-normalized non-negative kernel, see Research 257 §4).

This is **Gain-tier** because:
- The paper itself has not verified FUNCATTN on NLP / token prediction (§6: "investigating functional attention in domains with less direct function-space interpretations, such as natural language processing, remains a promising future task").
- All math pieces (ridge solve, eigenbasis, sigmoid partition-of-unity) are already in our stack.
- Parallax (Plan 135) is the closest shipped cousin; its audit (2026-05-30) found **NO GAIN** without Muon-trained weights. FUNCATTN may share the same fate.

**Ship the primitive, run an honest GOAT gate, demote if it loses.**

**GOAT gate (must pass before opt-in promotion to default-features list):**
- G1: FUNCATTN with random-init weights produces finite, bounded output for any input ‖X‖≤B (mechanics — Prop 4.5 Lipschitz check)
- G2: FUNCATTN approximates SDPA on a synthetic regression task better than Parallax at fixed parameter budget (Research 257 §2.4 F2 hypothesis)
- G3: sigmoid-basis FUNCATTN ≈ softmax-basis FUNCATTN on PDE-style proxy (no accuracy loss from sigmoid swap)
- G4: linear-in-n scaling verified at n ∈ {512, 2048, 8192} (per paper Fig 5)
- G5: zero-alloc hot path — single forward pass reuses pre-allocated scratch, no per-call Vec allocation

**Out of scope (this plan):**
- LLM-domain token-prediction GOAT gate (await Research 257 §5 Q2 — needs real LM weights, deferred until evidence exists)
- riir-ai integration (that's Plan 318 — rank-k latent functor upgrade, primary value path)

---

## Phase 1 — Skeleton (CORE)

Minimal module, behind feature flag, not in default features.

### Tasks

- [ ] **T1.1** Add `funcattn` feature to `katgpt-rs/Cargo.toml` and `katgpt-rs/crates/katgpt-core/Cargo.toml`. **Not in default features.** Add to `full` feature aggregation.
- [ ] **T1.2** Create `crates/katgpt-core/src/funcattn.rs` with the core types:
  ```rust
  pub enum FuncAttnBasis {
      /// Paper Eq. 9: Φ = Softmax(Linear(X)) along k-dim.
      Softmax,
      /// AGENTS.md compliance: Φ = Sigmoid(Linear(X)) then row-normalize.
      /// Partition-of-unity still holds (any row-normalized non-negative kernel).
      Sigmoid,
  }

  pub struct FuncAttnConfig {
      pub k: usize,                  // basis dimension, paper default 64
      pub lambda: f32,               // Tikhonov regularization, default 1e-4
      pub basis: FuncAttnBasis,      // default Sigmoid
      pub transpose_proj: bool,      // paper Rem 4.1: use Φᵀ not Φᵀ⁺. Default true.
  }

  pub struct FuncAttnScratch {
      // Pre-allocated scratch buffers for zero-alloc hot path:
      // phi (n×k), psi (n×k), q_tilde (k×d), k_tilde (k×d), v_tilde (k×d),
      // kktt (k×k), kktt_reg (k×k), c_op (k×k), pv (n×d), scores (n×k)
  }
  ```
- [ ] **T1.3** Implement `compute_basis_into(x, w, bias, n, d, k, kind, out)` — writes row-normalized basis to `out: &mut [f32]` of length `n*k`. Zero-alloc.
- [ ] **T1.4** Implement `funcattn_forward(q, k, v, w_phi, w_psi, cfg, scratch, out)`:
  - Compute Φ, Ψ via `compute_basis_into`
  - Project: `Q̃ = Φᵀ·Q`, `K̃ = Ψᵀ·K`, `Ṽ = Ψᵀ·V` (all via existing `simd_matmul_rows`)
  - Form `KKᵀ + λI_k` (k×k), Cholesky-decompose, solve for `C = Q̃·K̃ᵀ·(KKᵀ+λI)⁻¹`
  - `out = Φ · C · Ṽ` (two matmuls)
  - All in `scratch`, output to caller-owned `out: &mut [f32]`
- [ ] **T1.5** Reuse `crates/katgpt-core/src/simd.rs` for matmuls. Reuse Cholesky from `riir-gpu/schur.rs` if license-compatible (Apache-2.0 attribution required per file header); otherwise vendor a minimal k×k Cholesky (≤30 lines).

---

## Phase 2 — Mechanics Gate (no accuracy claim yet)

### Tasks

- [ ] **T2.1 (G1)** `g1_lipschitz_bounded`:
  - Generate 1000 random inputs with ‖X‖≤B for B ∈ {1, 10, 100}.
  - For each λ ∈ {1e-2, 1e-4, 1e-8}, verify `‖A(X+Δ) − A(X)‖_F / ‖Δ‖_F ≤ C₁/λ + C₂/λ²` empirically (Prop 4.5).
  - Assert no NaN/Inf across all configs.
- [ ] **T2.2 (G4)** `g4_linear_in_n_scaling`:
  - n ∈ {512, 2048, 8192, 32768}, d=128, k=64.
  - Measure forward time. Assert linear scaling (R² > 0.95 on log-log fit, slope ≈ 1.0).
  - Compare against `tiled_attention` baseline — at n=32768, FUNCATTN should be >10× faster.
- [ ] **T2.3 (G5)** `g5_zero_alloc`:
  - Run `cargo test --features funcattn` with allocator counting (or `cargo bench` with `--bench allocator_count` if available).
  - Assert 0 allocations per forward call after warmup.

---

## Phase 3 — Accuracy Gate (the actual GOAT decision)

### Tasks

- [ ] **T3.1 (G3 — sigmoid vs softmax)** `g3_sigmoid_matches_softmax`:
  - Synthetic PDE proxy: Burgers-equation-style dataset (paper §5.6 setup).
  - Train two FUNCATTN models (softmax basis vs sigmoid basis) for 1000 steps with identical seeds.
  - Assert sigmoid model's relative L2 error ≤ softmax model's + 5%.
  - **If sigmoid is >10% worse**: we have a problem (AGENTS.md says sigmoid, but if it doesn't work here, escalate as issue).
- [ ] **T3.2 (G2 — vs Parallax)** `g2_beats_parallax_on_regression`:
  - Sinusoidal few-shot regression (paper §5.1 setup, Fig 2).
  - Compare FUNCATTN vs Parallax (sigmoid) vs SDPA at matched parameter count.
  - Assert FUNCATTN MSE ≤ Parallax MSE × 0.5 AND FUNCATTN MSE ≤ SDPA MSE × 0.1.
  - This is the **paper's headline result** — we should reproduce it.

---

## Phase 4 — Verdict

### Tasks

- [ ] **T4.1** Write `katgpt-rs/.benchmarks/046_funcattn_goat.md` with G1–G5 results.
- [ ] **T4.2** If G1, G3, G4, G5 pass AND G2 shows FUNCATTN beats Parallax → **promote `funcattn` to opt-in (in `full` aggregation, NOT in default features)**. Document in `.docs/01_overview.md` Feature Flags table.
- [ ] **T4.3** If G2 fails (FUNCATTN does not beat Parallax on regression) → keep feature flag, document null result, **do not promote**. Note that the paper's gain is PDE-specific and may not transfer to our domains.
- [ ] **T4.4** **Do NOT promote to default until LLM-domain token-prediction evidence exists.** This is a separate gate (deferred per Research 257 §5 Q2).

---

## Phase 5 — Composition (post-GOAT only)

If Phase 4 promotes, wire composability. Each opt-in.

### Tasks

- [ ] **T5.1** Compose with SpectralQuant: pre-rotate basis weights via `calibrate_eigenbasis`. Hypothesis: eigenbasis-aligned FUNCATTN basis is more expressive per parameter.
- [ ] **T5.2** Compose with CHIAR (Plan 269): route between FUNCATTN and Parallax by per-token spectral entropy. FUNCATTN for low-entropy (structured) tokens, Parallax for high-entropy (chaotic) tokens.
- [ ] **T5.3** Compose with freeze/thaw: version basis snapshots `W_Φ, W_Ψ` as atomic Arc-swapped, BLAKE3-committed. Per-domain basis hot-swap. (This is the bridge to riir-ai Plan 318.)

---

## Files

- `crates/katgpt-core/Cargo.toml` — `funcattn` feature
- `crates/katgpt-core/src/funcattn.rs` — new module
- `crates/katgpt-core/src/lib.rs` — `#[cfg(feature = "funcattn")] pub mod funcattn;`
- `Cargo.toml` — top-level `funcattn = ["katgpt-core/funcattn"]`
- `.docs/01_overview.md` — Feature Flags table entry (Phase 4 if promoted)

## Open Questions

1. **Cholesky source.** Vendor minimal k×k Cholesky (clean, MIT-compatible) or reuse `riir-gpu/schur.rs` (Apache-2.0, requires attribution header)? Vendor is simpler for the public engine. ~30 lines.
2. **PDE proxy data.** Do we have a Burgers-equation dataset, or do we generate one synthetically? Paper uses Kovachki et al. 2023 benchmark — we'd need to either download or generate. For G2/G3, synthetic sinusoidal regression (paper §5.1) is sufficient and self-contained.
3. **Training loop for G2/G3.** The basis matrices `W_Φ, W_Ψ` need to be trained. This is technically "training" but it's standard transformer training (AdamW on a small model), not a new training method. Acceptable per skill constraint §1 ("no LLM training" refers to fine-tuning base LLMs, not training small diagnostic models for GOAT gates).

## Constraints Check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ C solve is closed-form given trained W_Φ,W_Ψ |
| Latent-to-latent preferred | ✅ All in spectral space; only final `Φ·C·Ṽ` returns to raw |
| Sigmoid not softmax | ✅ `FuncAttnBasis::Sigmoid` is the default |
| Freeze/thaw over fine-tuning | ✅ W_Φ,W_Ψ are swappable snapshots (Phase 5.3) |
| 3-repo discipline | ✅ Open primitive in katgpt-rs; no game IP; no training know-how |
| Zero-alloc hot path | ✅ `FuncAttnScratch` pre-allocated; all `_into` APIs |
| CPU/SIMD first | ✅ All matmuls via `simd_matmul_rows`; Cholesky is k×k (L1-resident for k=64) |
