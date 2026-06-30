# Plan 354: Cross-Datapoint Set Attention — Open Sigmoid-Gated Permutation-Equivariant Primitive

**Date:** 2026-07-01
**Research:** [katgpt-rs/.research/354_Cross_Datapoint_Set_Attention_NPT.md](../.research/354_Cross_Datapoint_Set_Attention_NPT.md)
**Companion private guide:** [riir-ai/.research/167_crowd_joint_inference_cross_npc_set_attention_guide.md](../../riir-ai/.research/167_crowd_joint_inference_cross_npc_set_attention_guide.md)
**Companion runtime plan:** [riir-ai/.plans/355_crowd_joint_inference_runtime.md](../../riir-ai/.plans/355_crowd_joint_inference_runtime.md)
**Source paper:** [arXiv:2106.02584](https://arxiv.org/pdf/2106.02584) — Kossen et al., NeurIPS 2021 (Non-Parametric Transformers)
**Target:** `katgpt-rs/crates/katgpt-core/src/set_attention/` (new module) + Cargo feature `set_attention`
**Status:** Active — Phase 1 scoping

---

## Goal

Ship the open primitive half of the Super-GOAT from Research 354 / riir-ai 167:
a generic, sigmoid-gated (NEVER softmax), permutation-equivariant cross-entity
set-attention kernel. Given `N` query vectors, `N` key vectors, and `N` value
vectors (or pre-projected `Q`, `K`, `V` matrices), produce `N` refined output
vectors where each output is a residual sigmoid-gated weighted sum of the values,
weighted by per-pair sigmoid gates.

This is the inference-time half of NPT's Attention Between Datapoints (ABD) —
the training half (end-to-end backprop on Q/K/V via BERT-style masking) stays in
riir-train. The primitive is *substrate-agnostic*: no game semantics, no HLA
awareness, no NPC awareness. It's just the math.

**GOAT gate (open primitive):** G1 (permutation equivariance, bit-exact) + G2
(identity-floor meaningfulness on synthetic 2-cluster) + G3 (latency < 5 µs on
N=64, d=8) + G4 (zero-alloc steady state) + G5 (sigmoid-not-softmax correctness:
lonely-query case is bit-exact identity). All five must pass before
default-on promotion. **The Super-GOAT fusion gate (G6: CS-ranking adds value
over identity) lives in the riir-ai runtime plan (P355).**

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/set_attention/mod.rs` with module doc referencing Research 354 §2.
- [ ] **T1.2** Define `SetAttentionConfig` struct in `set_attention/types.rs`:
  ```rust
  pub struct SetAttentionConfig {
      /// Query projection dim. k ≤ d. If k == d and W_Q is identity, this is the
      /// modelless floor.
      pub k: usize,
      /// Per-pair sigmoid temperature β. Sigmoid argument is `(q·k)/√k · β`.
      /// Default 1.0. Higher β → sharper attention.
      pub beta: f32,
      /// Residual step size γ. Output is `h_i + γ · Σ_j α_ij (v_j - h_i)`.
      /// Default 0.1. Bounded to preserve magnitude hygiene.
      pub gamma: f32,
      /// Optional top-k cap on attended peers per query. None = dense (all N).
      /// Some(k_max) = sparse; only the top-k_max highest-α peers contribute.
      pub top_k: Option<usize>,
  }
  ```
- [ ] **T1.3** Implement the core kernel `set_sigmoid_attention_into` in `set_attention/kernel.rs`:
  ```rust
  /// Permutation-equivariant, sigmoid-gated cross-entity set attention.
  ///
  /// Given N entity state vectors in `states` (shape [N*d]), pre-projection
  /// matrices W_Q, W_K (shape [d*k]) and W_V (shape [d*d] or identity), compute
  /// the refined states in `output` (shape [N*d], must be pre-allocated).
  ///
  /// Scratch buffers `q_buf`, `k_buf`, `alpha_row` must be pre-allocated by the
  /// caller. Zero allocations in steady state.
  ///
  /// # Permutation equivariance
  /// Permuting the rows of `states` permutes the rows of `output` identically.
  /// This is proven (paper Appendix A, Lemma 4); the test in `kernel.rs`
  /// verifies it bit-exactly.
  pub fn set_sigmoid_attention_into(
      states: &[f32],          // [N*d]
      w_q: &[f32],             // [d*k]
      w_k: &[f32],             // [d*k]
      w_v: Option<&[f32]>,     // [d*d] or None for identity
      output: &mut [f32],      // [N*d]
      cfg: &SetAttentionConfig,
      n: usize,
      d: usize,
      scratch_q: &mut [f32],   // [N*k]
      scratch_k: &mut [f32],   // [N*k]
      scratch_alpha: &mut [f32], // [N]
  )
  ```
  Inner loop:
  1. Project all N states: `scratch_q[i*k..(i+1)*k] = W_Q · states[i*d..(i+1)*d]`
  2. Project all N keys: `scratch_k[j*k..(j+1)*k] = W_K · states[j*d..(j+1)*d]`
  3. For each query i: compute `α_ij = σ(scratch_q[i]·scratch_k[j]/√k · β)` for each j, accumulate `output[i] += γ · α_ij · (v_j − states[i])`.
  4. If `cfg.top_k` is Some, restrict the inner sum to the top-k highest `α_ij`.
- [ ] **T1.4** Add the `set_attention` feature gate in `katgpt-rs/crates/katgpt-core/Cargo.toml` (`[features] set_attention = []`).
- [ ] **T1.5** Register `pub mod set_attention;` (cfg-gated) in `katgpt-rs/crates/katgpt-core/src/lib.rs`.
- [ ] **T1.6** Re-export the public API at the crate root:
  ```rust
  #[cfg(feature = "set_attention")]
  pub use set_attention::{SetAttentionConfig, set_sigmoid_attention_into};
  ```

### Acceptance

- `cargo check -p katgpt-core --features set_attention` compiles.
- `cargo test -p katgpt-core --features set_attention --lib set_attention` passes a smoke test (N=4, d=8, identity W_Q/W_K/W_V, output is finite).

---

## Phase 2 — GOAT Gate (G1–G5)

### Tasks

- [ ] **T2.1 (G1 — permutation equivariance)** Test: for a fixed seed, generate N=16 random states, run `set_sigmoid_attention_into`. Then permute the input rows by a random permutation σ, run again, verify the output is permuted identically (bit-exact for f32). Repeat for 10 random permutations. **Target: bit-exact pass on all 10.**
- [ ] **T2.2 (G2 — identity-floor meaningfulness)** Test: synthetic 2-cluster set — N=32, d=8, two clusters of 16 each, clusters separated by 4σ in feature space. With identity W_Q/W_K/W_V, run `set_sigmoid_attention_into` with γ=0.5. Verify the per-cluster mean of the output differs from the per-cluster mean of the input by less than 0.1σ (consensus pulls toward cluster center) AND the cross-cluster mean separation is preserved (clusters don't merge). **Target: separation preserved, intra-cluster variance reduced.**
- [ ] **T2.3 (G3 — latency)** Benchmark in `katgpt-rs/crates/katgpt-core/benches/set_attention_bench.rs`:
  - N=64 entities × d=8 HLA dims × k=4 (CS-ranked) projection.
  - Criterion bench, sample_size=500.
  - **Target: < 5 µs per call** (well within 20Hz tick budget = 50ms; this leaves headroom for 100+ calls per tick).
- [ ] **T2.4 (G4 — zero-alloc)** Test using a custom allocator hook (or `#[track_caller]` + `Vec::with_capacity` audit): verify `set_sigmoid_attention_into` performs 0 heap allocations when called with pre-allocated scratch. **Target: 0 allocs/call.**
- [ ] **T2.5 (G5 — sigmoid-not-softmax correctness)** Test: lonely-query case — N=2, d=8, query 0 has W_Q·h_0 orthogonal to W_K·h_1 (so α_01 < 0.5 by a wide margin). Verify `output[0]` equals `states[0]` bit-exactly (no contribution from peer 1). **Target: bit-exact.** Also: all-α-low case (β very small) — all outputs equal inputs.
- [ ] **T2.6** If all G1–G5 pass, file results in `katgpt-rs/.benchmarks/354_set_attention_goat.md`. Promote `set_attention` from opt-in to default-on per AGENTS.md feature-flag discipline (only if the Super-GOAT G6 also passes in riir-ai P355 — keep opt-in until both clear).

### Acceptance

- All 5 gates pass. Bench doc filed. Decision recorded (promote or keep opt-in pending riir-ai G6).

---

## Phase 3 — Sparse Top-K (for N > 100 zones)

Deferred until riir-ai P355 G9 (crowd-scale latency) demands it. Ship dense by default; sparse is opt-in.

### Tasks

- [ ] **T3.1** Implement `top_k` config branch in `set_sigmoid_attention_into`: for each query i, compute all N α_ij, select the top-k_max by partial sort, accumulate the sum only over those.
- [ ] **T3.2** Benchmark N=1000 × d=8 × k=4 × top_k=16. **Target: < 100 µs** (still well within budget; this is for big crowded zones).
- [ ] **T3.3** Alternative: LSH-based approximate top-k (locality-sensitive hashing on the projected queries/keys). Defer unless T3.2 misses.

---

## Phase 4 — Fusion Examples (optional, demonstrates non-game use)

### Tasks

- [ ] **T4.1** Example in `katgpt-rs/examples/set_attention_demo.rs`: simple "consensus averaging" use case — 10 sensors each produce a noisy 4-dim reading, set attention refines each reading by sigmoid-weighted averaging over similar sensors. Shows the open primitive is useful beyond NPC AI.

---

## References

- Research 354 (open primitive): `katgpt-rs/.research/354_Cross_Datapoint_Set_Attention_NPT.md`
- riir-ai Guide 167 (private selling point): `riir-ai/.research/167_crowd_joint_inference_cross_npc_set_attention_guide.md`
- riir-ai Plan 355 (private runtime): `riir-ai/.plans/355_crowd_joint_inference_runtime.md`
- Source paper: [arXiv:2106.02584](https://arxiv.org/pdf/2106.02584) — Kossen et al., NeurIPS 2021
- Related: Research 247 (CS-KV-Importance Probe — offline Q construction source), Research 242 (HLA), Research 144 (Functional Emotions — direction vectors)

---

## TL;DR

Open primitive half of the Super-GOAT from R354 / riir-ai-167. Ship a generic,
sigmoid-gated (NEVER softmax), permutation-equivariant cross-entity set-attention
kernel behind the `set_attention` feature flag. Five-gate GOAT: G1 (permutation
equivariance bit-exact) + G2 (identity-floor meaningfulness on 2-cluster) + G3
(latency < 5 µs at N=64) + G4 (zero-alloc) + G5 (sigmoid-not-softmax lonely-query
correctness). Promote to default-on only when both this gate AND the riir-ai
runtime G6 (CS-ranking fusion adds value) pass. The training half (BERT-style
masking, end-to-end Q/K/V backprop) stays in riir-train.
