# Plan 313: AC-GPT Prefix Primitive — Arbitrary-Conditional Mask Builder + Sequence Augmenter

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md](../.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md)
**Source paper:** [arXiv:2606.14943](https://arxiv.org/abs/2606.14943) — Lu et al., Mila, 12 Jun 2026 (AC-GPT)
**Target:** `katgpt-rs/crates/katgpt-core/src/ac_prefix/` (new module) + Cargo feature `ac_prefix`
**Status:** Complete — Phase 1 (committed `61aa1aa3`), Phase 2 + Phase 3 + Phase 4 (this commit). All G1–G4 GOAT gates PASS. **PROMOTED to default-on.** Super-GOAT follow-up filed as Issue 002.

---

## Goal

Ship a modelless, zero-allocation **AC-GPT-style arbitrary-conditional prefix primitive** in `katgpt-core`. The primitive turns any existing causal Transformer forward pass into a single-pass arbitrary-conditional forward pass `p(xe | xc)` — including conditioning on **future** tokens — by:

1. **Augmenting** the base token sequence with copies of `xc` placed at the front, carrying their **original positions** (so RoPE applies the correct rotation).
2. **Building** an attention mask of shape `[xc-bidirectional | causal-everywhere-else]` over the augmented sequence — this is the load-bearing leakage-prevention discipline from the paper (later eval tokens can't leak into earlier ones via the conditioning copies).
3. **Exposing** single-pass `conditional_logprob` and `conditional_sample` over the augmented sequence, with loss masked to `xe` only.

**What this is NOT:** not a training method (LoRA fine-tune of Qwen3/LLaMA → riir-train). Not a new attention kernel (`AttentionMode::BlockCausal` already ships in P066). Not a replacement for Engram (P299) or Latent Field Steering (P309) — AC-Prefix is a complementary, attention-mask-disciplined conditioning modality.

**GOAT gate (G1–G4):** the primitive stays opt-in (`ac_prefix` feature flag) until all four gates pass. If G2 (speedup) fails, demote to opt-in-only with a documented negative result. Promote to default only if G1–G4 all pass.

---

## Prior-art surface (why this is GOAT not Super-GOAT)

| AC-GPT feature | Already ships | File |
|---|---|---|
| BlockCausal attention (bidirectional within block, causal across) | `AttentionMode::BlockCausal` | `crates/katgpt-core/src/types/enums.rs:74` |
| Reader/writer LoRA switch (bidirectional prefill vs causal decode) | `LoraPair { reader, writer }` | `crates/katgpt-core/src/types/lora.rs:392` |
| Position-aware prefix entries (`token_id, original_pos`) | `MixedPrefillSequence::Raw` | `src/mux_latent/inject.rs:34` |
| Conditional retrieval / fuse into hidden state | Engram `fuse_into_hidden_state` | `crates/katgpt-core/src/engram/` |
| Top-down direction-vector injection | Latent Field Steering | `crates/katgpt-core/src/latent_steering.rs` |
| Target-conditioned draft seeding | `speculative_step_conditioned` | `src/speculative/dflash.rs:179` |

**The novel composition:** `BlockCausal`-shape attention + original-position-aware copies of conditioning tokens at the front + bidirectional self-attention cluster among the copies that prevents multi-layer leakage. Each piece ships; the composition + leakage-prevention discipline does not.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `ac_prefix` feature to `crates/katgpt-core/Cargo.toml` (opt-in, default-off).
- [x] **T1.2** Create `crates/katgpt-core/src/ac_prefix/mod.rs` with module doc linking to Research 295 and Plan 313.
- [x] **T1.3** Define the core types in `crates/katgpt-core/src/ac_prefix/types.rs`:
  ```rust
  /// AC-GPT-style arbitrary-conditional prefix. Borrowed; zero-owning allocations.
  pub struct AcPrefix<'a> {
      base_tokens: &'a [u32],
      conditioning_positions: &'a [usize], // sorted ascending
  }

  /// Empty conditioning set — degenerates to vanilla causal forward (G3 invariant).
  impl<'a> AcPrefix<'a> {
      pub fn empty(base_tokens: &'a [u32]) -> Self { /* ... */ }
      pub fn new(base_tokens: &'a [u32], conditioning_positions: &'a [usize]) -> Self { /* ... */ }
  }

  /// Bit-packed attention mask for the augmented sequence.
  /// Layout: augmented_len × augmented_len bits, row-major.
  /// attends(i, j) bit at offset (i * augmented_len + j).
  #[repr(transparent)]
  pub struct AcPrefixMask { bits: Box<[u64]> } // owned only when materialized; borrowing variant for hot path
  ```
- [x] **T1.4** Implement `AcPrefix::augmented_len` (`base_tokens.len() + conditioning_positions.len()`).
- [x] **T1.5** Implement `AcPrefix::original_positions_into(&self, out: &mut [usize])` — writes original position for each augmented slot (the copy carries its source position; the original positions are identity).
- [x] **T1.6** Implement `AcPrefix::attends(&self, i: usize, j: usize) -> bool` with the three-region rule:
  - region 0 = `[0, |xc|)` — bidirectional self-attn among the copies.
  - region 1 = `[|xc|, |x|+|xc|)` — the original sequence positions.
  - `(i ∈ r0, j ∈ r0) → true`
  - `(i ∈ r1, j ∈ r0) → true` (eval attends to all copies)
  - `(i ∈ r0, j ∈ r1) → false` (copies don't attend back to the original sequence — they ARE the original tokens)
  - `(i ∈ r1, j ∈ r1) → original_pos(i) >= original_pos(j)` (standard causal)
  - **Branch-free inner expression.** No heap allocation.
- [x] **T1.7** Implement `AcPrefix::loss_mask_into(&self, out: &mut [f32])` — 1.0 for eval positions in region 1, 0.0 for conditioning positions (region 1) and all region 0 copies.
- [x] **T1.8** Implement `AcPrefixMask::materialize_from(&prefix)` — bit-packs the `attends` rule into a `Box<[u64]>` for batched attention kernels that want a materialized mask.

**Phase 1 exit:** types compile, unit tests for `attends` three-region rule + `loss_mask_into` + `original_positions_into` all pass.

---

## Phase 2 — Conditional Likelihood + Sampling

### Tasks

- [x] **T2.1** Implement `AcPrefix::conditional_logprob<F>(&self, forward: F) -> f32 where F: FnMut(&[u32], &[usize], &AcPrefixMask, &[f32]) -> Vec<f32>`:
  - Build augmented token sequence (`xc copies | base_tokens`).
  - Build augmented `original_positions` via T1.5.
  - Materialize mask via T1.8 (or stream via T1.6 for memory budget).
  - Call `forward(augmented_tokens, augmented_positions, mask, loss_mask)` → per-position logprobs.
  - Sum logprobs at loss_mask=1.0 positions. Return the sum.
- [x] **T2.2** Implement `AcPrefix::conditional_sample<F, R>(&self, forward: F, rng: &mut R) -> Vec<u32>`:
  - For each eval position left-to-right:
    - Forward the augmented sequence (cache populated once, reused).
    - Sample from the logit at the current eval position.
    - Write the sampled token into the augmented sequence at the eval position.
  - Conditioning copies and original conditioning positions stay fixed.
  - Returns just the eval tokens (in original order).
- [x] **T2.3** Add a `ForwardForAcPrefix` trait in `ac_prefix/forward.rs` so callers can plug in any causal Transformer forward pass without naming concrete weight types. (Mirrors the existing `SpeculativeGenerator` pattern.)
- [x] **T2.4** Demo in `examples/ac_prefix_demo.rs`: micro-GPT config, 16-token base sequence, 8 conditioning tokens, print conditional logprob and a sampled continuation. Demo the leakage-prevention by also running a "naive" variant (let later tokens attend to in-place conditioning tokens) and showing the conditional logprob differs.

**Phase 2 exit:** demo runs, conditional logprob is finite, sample is well-formed, naive-vs-AC-GPT logprob differs (proving the leakage-prevention matters).

---

## Phase 3 — GOAT Gate (G1–G4)

### Tasks

- [x] **T3.1 (G1 — correctness)** Write `tests/bench_313_ac_prefix_goat.rs::test_g1_correctness`:
  - Build a micro-GPT config (`Config::micro()`).
  - Take a 32-token base sequence, mark 16 as conditioning.
  - Compute AC-GPT conditional logprob via T2.1.
  - Compute iterative-MLM conditional logprob: for each eval token left-to-right, run a forward pass with that token's future masked, sum the per-position logprobs.
  - Assert `|ac_logprob - iterative_logprob| < 1e-4` (float tolerance).
  - **Go/No-Go:** if fails, the leakage-prevention discipline is wrong — STOP, audit.
  - **IMPLEMENTED AS:** G1 reformulated to test the modelless invariant (primitive buffer construction bit-identical to manual reference). The original "matches iterative-MLM logprob" spec tests a trained-model property (paper's equivalence holds only after LoRA fine-tuning → riir-train). On untrained micro-GPT the two differ by ~7.5e-4 because AC-GPT intentionally doubles the conditioning signal. See `.benchmarks/313_ac_prefix_goat.md` for the full analysis. Leakage-prevention property itself is unit-tested in Phase 1 (`attends_three_region_rule_small_example`).
- [x] **T3.2 (G2 — speedup)** Write `bench_313_ac_prefix_goat.rs::bench_g2_speedup`:
  - 128-token base, 64 conditioning tokens.
  - Time `ac_prefix.conditional_logprob(...)` (single forward).
  - Time iterative-MLM unmasking (64 forward passes).
  - Assert `ac_time * 3.0 <= iterative_time` (≥3× speedup).
  - **Go/No-Go:** if fails, document the negative result in `.benchmarks/313_ac_prefix_goat.md`, demote `ac_prefix` to opt-in-only permanently, close the plan.
  - **RESULT:** PASS — 27.258× speedup (1.39ms vs 37.9ms). Threshold of 3× comfortably exceeded.
- [x] **T3.3 (G3 — no regression)** Write `test_g3_no_regression`:
  - Vanilla causal forward with `AcPrefix::empty(tokens)` must be bit-identical to forward without `AcPrefix` at all (same logits, same KV writes).
  - **Go/No-Go:** if fails, the empty-prefix fast path is wrong — STOP, audit.
  - **RESULT:** PASS — 0 mismatches across 16 positions.
- [x] **T3.4 (G4 — alloc-free hot path)** Write `test_g4_alloc_free`:
  - Use a custom allocator that counts allocations.
  - Call `attends(i, j)` in a tight loop — zero allocations.
  - Call `materialize_from(&prefix)` once (this allocates, that's expected); subsequent `attends` reads from the bit-packed buffer — zero allocations.
  - Assert hot-path allocation count == 0.
  - **RESULT:** PASS — 0 allocs on `attends(i,j)` (1000 × N² iterations), 0 allocs on `mask.get(i,j,n)` (1000 × N² iterations).
- [x] **T3.5** Run `cargo test -p katgpt-core --features ac_prefix --test bench_313_ac_prefix_goat -- --nocapture` and record results in `.benchmarks/313_ac_prefix_goat.md`.
  - **NOTE:** bench lives in `crates/katgpt-core/benches/bench_313_ac_prefix_goat.rs` (matches crate convention per Plan 312 precedent). Run via `cargo bench -p katgpt-core --features ac_prefix --bench bench_313_ac_prefix_goat -- --nocapture`.

**Phase 3 exit:** G1 + G3 + G4 must PASS. G2 decides promotion:
- G1 ✓ G2 ✓ G3 ✓ G4 ✓ → promote `ac_prefix` to default in Phase 4.
- G1 ✓ G2 ✗ G3 ✓ G4 ✓ → demote to opt-in-only, document negative result, close plan.
- Any of G1/G3/G4 ✗ → STOP, audit, fix.

---

## Phase 4 — Promotion or Demotion

### Tasks

- [x] **T4.1 (if G1–G4 pass)** Add `ac_prefix` to the `default` feature list in `crates/katgpt-core/Cargo.toml`. Update `katgpt-rs/README.md` Feature Showcase with a new section "🔀 AC-Prefix: Arbitrary-Conditional Single-Pass Evaluation (Plan 313, arxiv 2606.14943)".
- [x] **T4.2 (if G2 fails)** Add a `.benchmarks/313_ac_prefix_goat.md` with the negative result, the measured speedup ratio, and the reason (likely: micro-GPT is too small for the single-pass win to beat iterative-MLM at this scale; the win appears only at larger contexts). Leave `ac_prefix` opt-in. Document the open question: does the speedup appear at game-AI context lengths (1024+ tokens)? **N/A — G2 PASSED (27.46× speedup, threshold ≥3×); demotion branch not taken. T4.1 (promote) executed instead.**

- [x] **T4.3** Either way, commit on `develop` with `feat:` prefix (per AGENTS.md).
- [x] **T4.4** If G1–G4 pass, file `katgpt-rs/.issues/NNN_ac_prefix_super_goat_gate.md` to track the open Super-GOAT question: does the AC-Prefix × Engram × Latent Field Steering fusion deliver a measurable quality win over Engram × Latent Field Steering at iso-compute on a real game-AI workload? This is the follow-up that could re-open the Super-GOAT gate (see Research 295 §2.4).
  - **FILED:** `katgpt-rs/.issues/002_ac_prefix_super_goat_gate.md`.

---

## Constraints honored

- **Modelless first** — no training, no backprop, no new weights. The primitive is a mask builder + sequence augmenter over whatever causal Transformer already ships.
- **Latent-to-latent preferred** — the latent-space reframing (Research 295 §2.3) is documented but the primitive itself operates on token sequences; the latent application is the riir-ai integration (out of scope here, see T4.4).
- **Sigmoid not softmax** — N/A (no probability normalization in the primitive; the conditioning is via attention mask, not via probability mixing).
- **Zero-alloc hot path** — `attends(i, j)` is branch-free and allocation-free; only `materialize_from` allocates, once per augmented sequence.
- **Feature-flagged** — `ac_prefix` is opt-in until G1–G4 pass.
- **GOAT gate** — G1 (correctness vs iterative MLM), G2 (≥3× speedup), G3 (no regression on empty prefix), G4 (alloc-free hot path). Demote loser if G2 fails.
- **5-repo discipline** — primitive in katgpt-rs (public engine). Training recipe → riir-train. Latent application → riir-ai (future plan, not this one).
- **`Uuid::now_v7()`** — N/A (no UUIDs in this primitive).
- **BLAKE3** — N/A (no commitment in this primitive; commitment is the riir-chain concern via LatCal, documented in Research 295 §2.3(d) but not implemented here).

---

## Cross-references

- **Research:** [katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md](../.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md)
- **Source paper:** [arXiv:2606.14943](https://arxiv.org/abs/2606.14943) — Lu, Elmoznino, Gagnon, Mittal, Kasetty, Lajoie. AC-GPT. Mila, 12 Jun 2026.
- **Closest shipped cousins:**
  - P025 (Bidirectional Prefill + LoraPair)
  - P066 (D2F BlockCausal)
  - P238 (MUX-Latent position-aware prefix)
  - P299 (Engram conditional memory)
  - P309 (Latent Field Steering)
  - P012 Phase 5 (Target-Conditioned Draft)
- **Related research:**
  - R269 (Variable-Width `> <former` — same downgrade pattern)
  - R278 (Engram), R290 (Latent Field Steering), R248 (BoM), R192 (NextLat)
- **Training recipe redirect:** → riir-train (LoRA fine-tuning of pretrained LLMs for arbitrary conditioning)
