# Plan 395: HOLA — Hippocampal Exact KV Cache for Linear Attention

**Date:** 2026-07-05
**Research:** [katgpt-rs/.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md](../.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md)
**Source paper:** [arxiv 2607.02303](https://arxiv.org/abs/2607.02303) — Cui 2026, HOLA
**Target:** `crates/katgpt-core/src/hippocampal_cache.rs` (new module) + Cargo feature `hippocampal_cache` + `gdn2` integration point
**Status:** Active — Phase 1 (skeleton) not started

---

## Goal

Ship a **surprise-evicted bounded exact KV cache** primitive that recovers the long-range exact recall the GDN2 fixed-size recurrent state (Plan 105, default-on backbone) loses. The cache stores the top-`w` tokens by intrinsic delta-rule write magnitude `β·‖e‖` (free from the existing GDN2 update), and reads them via a **decoupled RMSNorm-γ** sharpened softmax that turns the exact copies into near-argmax retrieval instead of a soft average. The paper reports −16% Wikitext perplexity at 340M and robust RULER S-NIAH-1 recall at 16× training length (0.58 vs GDN 0.14 at 32k).

The GOAT gate (G1–G4) is **modelless** (synthetic correctness + latency + no-regression + retrieval on a controlled toy). The perplexity gate (G5) requires a trained GDN2 model → **deferred to riir-train** (tracked in `.issues/`, does not block modelless promotion).

Lands in the **KV-compression slot** alongside AM (Plan 271, opt-in), Sink-Aware (Plan 287, opt-in), StillKV (Plan 245). HOLA is opt-in until G1–G4 PASS; promotion to default-on is a deliberate parent decision pending G5 (per the per-stack promote/demote ledger, Research 378 §3 MOAT gate).

**Modelless unblock check (§3.5) for the γ parameter:** the decoupled RMSNorm-γ needs a value. The paper trains it end-to-end. Before deferring γ to riir-train, exhaust:
1. **Freeze/thaw** — N/A (γ is a fresh parameter, no snapshot to thaw).
2. **Raw/lora reader-writer** — N/A (no adapter construction needed; γ is a vector).
3. **Latent-space correction** — ✅ applicable. If γ=1 (identity) gives "logits too flat" (the canonical HOLA diagnostic), a **deterministically-constructed γ** closes the gap: `γ_i = √d / max(‖k_i‖, ε)` (per-key norm rescale, recovering the paper's `‖k̃‖ ≈ √d` target without training). This is the closed-form sharpening factor the paper's RMSNorm-γ approximates by gradient descent. **Plan 395 ships γ=identity as the modelless default + γ=√d-rescale as the deterministic-correction variant.** If G4 retrieval still fails on both, *then* defer γ-tuning to riir-train with the §3.5 documentation (which path failed, why).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/hippocampal_cache.rs` skeleton behind `#[cfg(feature = "hippocampal_cache")]`. Add feature `hippocampal_cache = []` to `crates/katgpt-core/Cargo.toml`. Keep it OFF by default.
- [ ] **T1.2** Define `HippocampalCache<D, W>` generic struct (no model semantics):
  ```rust
  pub struct HippocampalCache<const D: usize, const W: usize> {
      // Top-w min-heap by score. score_bits(u32) || slot_idx(u32) packed into u64 for cache-line density.
      heap: [u64; W],
      heap_len: usize,
      // Slot ring storing the actual (k, v) tensors. Free list tracks recycled slots.
      keys: [[f32; D]; W],
      vals: [[f32; D]; W],
      scores: [f32; W],
      free_head: Option<u32>,
      // γ for the decoupled cache-read RMSNorm. Model parameter (not runtime-learned).
      gamma: [f32; D],
  }
  ```
  - Const generics `D` (head_dim) and `W` (cache capacity). Defaults: `D=64`, `W=16` for micro configs.
- [ ] **T1.3** Implement `observe(&mut self, k: &[f32; D], v: &[f32; D], beta: f32, residual_norm: f32)`:
  - Compute `score = beta * residual_norm` (single f32 mul).
  - If `heap_len < W`: insert into heap, claim a free slot, copy k/v.
  - Else if `score > heap_min_score`: replace heap-min, free the evicted slot, claim it, copy k/v.
  - Heap discipline: min-heap on `score_bits` (use `f32::to_bits` for total order — deterministic, NaN-safe per AGENTS.md).
  - O(log W) per observe.
- [ ] **T1.4** Implement `read_cache_into(&self, q: &[f32; D], gamma: &[f32; D], block_kv: &[(Vec<f32>, Vec<f32>)], out: &mut [f32; D])`:
  - Build `V_t` = cache slots ∪ block_kv ∪ null sink (1 zero vector).
  - Compute `q̃ = RMSNorm_γ(q)` using `katgpt_types::rmsnorm_with_gamma` (already ships).
  - For each `(k_j, v_j)` in `V_t`: `k̃_j = RMSNorm_γ(k_j)`, logit `= q̃·k̃_j / √D`, accumulate softmax-weighted `v_j` into `out`.
  - Use a stack-local `[f32; W + C + 1]` logits buffer (no heap allocation on the read path — hot path rule).
  - O((W + |block| + 1) · D) per read.
- [ ] **T1.5** Implement `reset(&mut self)` — clear heap, free all slots. Used at sequence boundaries.
- [ ] **T1.6** Add `hippocampal_cache` to the `katgpt-core` `lib.rs` feature-gated module list. Run `cargo check -p katgpt-core --features hippocampal_cache` — must compile clean.

### Validation (Phase 1)

- [ ] **T1.V1** Unit test: insert 100 tokens with random scores into a `HippocampalCache<8, 4>`. Verify the surviving 4 slots are the top-4 by score, in any order. Verify scores match the inserted values.
- [ ] **T1.V2** Unit test: order-independence. Insert the same 100 (k, v, score) triples in 5 different random orders. Verify the final cache set is **identical** (same 4 slots, same k/v/scores). This is the HOLA contract: "the same top-w set can be maintained online or blockwise without order dependence" (paper §3.3).
- [ ] **T1.V3** Unit test: read determinism. Given a fixed cache state and fixed q + γ, `read_cache_into` produces byte-identical output across two calls (no hidden state, no allocation-dependent ordering).
- [ ] **T1.V4** `cargo check --all-features` — must pass (the `merkle_root` / `can_freeze` lesson: combo-only regressions).

---

## Phase 2 — GOAT Gate G1 (Eviction Correctness) + G2 (Latency)

### Tasks

- [ ] **T2.1** **G1 — eviction correctness.** Test suite in `crates/katgpt-core/src/hippocampal_cache.rs` `#[cfg(test)]`:
  - Synthetic 8-key multi-needle stream (4k tokens, 8 keys placed at controlled depths).
  - Each token has `(beta, residual_norm)` assigned; the 8 needles get the top-8 scores.
  - After feeding the full stream into `HippocampalCache<64, 8>`, assert all 8 needles are in the cache. **PASS bar: 8/8.**
  - Negative: 100 distractor tokens with low scores, verify they are evicted.
- [ ] **T2.2** **G2 — latency benchmark** in `crates/katgpt-core/benches/hippocampal_cache_goat.rs`:
  - `bench_observe` — `HippocampalCache<256, 64>`, 10k observes, criterion micro-bench. Target: **≤ 100 ns/observe** (heap maintain at W=64).
  - `bench_read` — same cache, single `read_cache_into` call. Target: **≤ 1 µs/read** at W=64, D=256.
  - `bench_observe_micro` — `HippocampalCache<8, 16>` (game micro config). Target: **≤ 30 ns/observe**.
  - Run with `CARGO_TARGET_DIR=/tmp/hippocampal_cache_goat` per AGENTS.md rule; clean up after.
- [ ] **T2.3** Heap-vs-sorted-vec micro-bench (Risk #3). At W=64, also benchmark a sorted-vec variant `SortedSlotCache<D, W>` that does linear-scan eviction. Pick the winner as the production backing store. Document the choice in the module doc-comment.
- [ ] **T2.4** Mark G1, G2 in the GOAT gate table at the bottom of this plan. Commit.

### Validation (Phase 2)

- G1 PASS: 8/8 needles retained, all distractors evicted, order-independent.
- G2 PASS: observe ≤ 100 ns (W=64) / ≤ 30 ns (micro); read ≤ 1 µs (W=64).

---

## Phase 3 — GOAT Gate G3 (No-Regression on Bare GDN2)

### Tasks

- [ ] **T3.1** Wire `HippocampalCache` into `gdn2::State` behind the `hippocampal_cache` feature:
  - Add `#[cfg(feature = "hippocampal_cache")] pub cache: Option<HippocampalCache<HEAD_DIM, W>>` to the layer state.
  - Default `None` — zero overhead when feature is on but cache is not requested.
  - When `Some`, `gdn2_recurrent_step` (per Plan 105 / Research 070 §E4) calls `cache.observe(k, v, beta_t, residual_norm)` after each delta-rule write. `beta_t` and `residual_norm` are already computed on the hot path — verify by reading the existing `gdn2_recurrent_step` implementation; pipe through.
- [ ] **T3.2** **G3 — no-regression.** Test: run `gdn2_recurrent_step` on a fixed input stream with `cache = None` and `cache = Some(..)` (cache observed but read discarded). Assert the **GDN2 state S_t is byte-identical** between the two runs. The cache is a pure observer of the delta-rule update — it must not perturb the state.
- [ ] **T3.3** Test: with `W = 0` (cache disabled via config), the decode output is byte-identical to bare GDN2 (no `hippocampal_cache` feature). This proves the feature gate is clean (the `merkle_root` lesson applied).
- [ ] **T3.4** `cargo test -p katgpt-core --features hippocampal_cache,gdn2_attention --lib` — all GDN2 + cache tests green.

### Validation (Phase 3)

- G3 PASS: bare GDN2 output == GDN2+cache-observed-but-not-read output, byte-identical.
- Feature-gate isolation PASS: `--no-default-features --features gdn2_attention` still compiles and passes GDN2 tests.

---

## Phase 4 — GOAT Gate G4 (Synthetic Retrieval Gain, Modelless)

This is the load-bearing modelless gate. The paper trains GDN2 + HOLA end-to-end and measures perplexity + RULER. We cannot do that modellessly. G4 is a **controlled toy** that isolates the cache mechanism from training.

### Tasks

- [ ] **T4.1** Build a synthetic multi-key associative recall harness in `crates/katgpt-core/tests/hippocampal_cache_retrieval.rs`:
  - Generate a 4k-token stream with 8 inserted needles. Each needle is `(key_vec, value_vec)` where `key_vec` is a random unit-norm vector and `value_vec` is a distinct random unit-norm vector.
  - Distractor tokens: random `(k, v, beta=0.3, residual_norm=random_in_[0.05, 0.2])`.
  - Needle tokens: random `(k, v, beta=0.9, residual_norm=random_in_[0.5, 1.0])` — i.e., the delta-rule genuinely found these surprising (large residual, strong write). This is the HOLA ablation setup.
  - Query: pick a needle's `key_vec`, run `cache.read_cache_into(q, gamma=ones, block=[], out)`.
  - **PASS bar**: cosine-sim(out, true_value_vec) ≥ 0.8 for ≥ 6/8 needles (paper achieves near-1.0 at this scale with trained γ).
- [ ] **T4.2** Apply §3.5 modelless unblock for γ (Risk #2). If γ=ones fails G4:
  - Construct `γ_i = sqrt(D) / max(‖k_i‖, eps)` per-key (the deterministic sharpening analog of trained γ). Re-run G4 with this `γ`.
  - If this variant passes G4, ship **both** `gamma=ones` (default) and `gamma=per_key_norm_rescale` (deterministic-correction variant, opt-in via config field).
  - If both fail, document per §3.5: "freeze/thaw N/A (fresh param); raw/lora N/A (vector not adapter); latent-correction attempted (γ=rescale) — insufficient because [specific reason]. γ-tuning deferred to riir-train." Then create `.issues/NNN_hippocampal_cache_gamma.md` and ship the primitive opt-in anyway (the mechanism + eviction + decoupled read are still GOAT; γ is one tunable).
- [ ] **T4.3** Run G4 with three competitors on the same stream (defend-wrong pattern from §3.6, even though no parity claim — it strengthens the gate):
  - **Baseline A**: no cache (pure GDN2 state read `q^T S`).
  - **Baseline B**: HOLA+recency cache (matched control from paper §4.5 — keep most-recent w, same read path).
  - **HOLA**: top-w by β·‖e‖ + decoupled RMSNorm-γ read.
  - Print verdict table. HOLA should beat both baselines on the 8-needle recovery rate. If HOLA loses to recency on this synthetic, that's a real finding — record it honestly per §3.6.
- [ ] **T4.4** Commit. Update GOAT gate table.

### Validation (Phase 4)

- G4 PASS: HOLA recovers ≥ 6/8 needles (cosine ≥ 0.8); both baselines recover ≤ 4/8 at the same w budget.

---

## Phase 5 — Promotion Decision + Issue Tracking

### Tasks

- [ ] **T5.1** GOAT gate summary table at the bottom of this plan filled in. If G1–G4 all PASS → primitive is **modelless-GOAT**, opt-in default stays OFF (not promoted to default until G5).
- [ ] **T5.2** Create `.issues/NNN_hippocampal_cache_g5_riir_train.md`:
  - Title: "HOLA cache perplexity + RULER gate (G5) — needs trained GDN2 weights".
  - Body: train a matched GDN2 model at 46M (smallest paper scale) on FineWeb-Edu 0.5B tokens (paper's 46M config, App. A) with and without HOLA cache. Report Wikitext PPL and a 4k-context needle probe. This is a riir-train job; not blocking the katgpt-rs modelless promotion.
  - Cross-link Research 378 §3 MOAT gate (G5 deferred).
- [ ] **T5.3** README.md "Feature Showcase" entry for `hippocampal_cache` (one paragraph + paper link + GOAT gate status). Update `katgpt-rs/.docs/01_overview.md` Feature Flags table.
- [ ] **T5.4** Cross-reference: add a one-line entry to `.research/070_Gated_DeltaNet_2_*.md` §Relationship noting HOLA cache as a pluggable complement to GDN2 (the "Phase 4 alternative to SWA" identified in Research 070 §Verdict).
- [ ] **T5.5** Cross-reference: add a one-line entry to `.research/243_Temporal_Derivative_Kernel_*.md` §Fusion noting HOLA's β·‖e‖ as a *second* (instantaneous, per-token) surprise channel alongside temporal_deriv's dual-fast/slow EMA — the F2 fusion-potential.
- [ ] **T5.6** Commit on `develop` per global rule. `docs:` prefix (this is a research+plan+primitive commit, but the headline user-facing artifact is the docs + open primitive; per AGENTS.md "use git naming convention", use `feat:` for the primitive + `docs:` for the research+plan in separate commits or a single `feat:` — pick one).

---

## GOAT Gate Table

| Gate | Target | Status |
|---|---|---|
| **G1** — Eviction correctness | 8/8 needles retained, distractors evicted, order-independent | ⏳ Pending Phase 2 |
| **G2** — Latency | observe ≤ 100 ns (W=64), ≤ 30 ns (micro); read ≤ 1 µs (W=64) | ⏳ Pending Phase 2 |
| **G3** — No-regression | GDN2 state byte-identical with/without cache observer; W=0 == bare GDN2 | ⏳ Pending Phase 3 |
| **G4** — Retrieval (modelless) | HOLA ≥ 6/8 needles (cosine ≥ 0.8); baselines ≤ 4/8 | ⏳ Pending Phase 4 |
| **G5** — Perplexity + RULER (riir-train) | ≥ −10% Wikitext PPL vs bare GDN2 at 46M; S-NIAH-1 @ 4k ≥ 0.7 | ⏳ Deferred (Issue, Phase 5) |

**Promotion rule (per Research 378 MOAT gate):** G1–G4 modelless PASS → opt-in `hippocampal_cache` ships. G5 PASS (riir-train) → promotion decision to default-on is a deliberate parent call, weighing against AM (Plan 271) and Sink-Aware (Plan 287) for the same KV-compression slot. Demote the loser when the slot is contested.

---

## Out of Scope (tracked elsewhere)

- **F2 fusion (HOLA × temporal_deriv)** — two-signal cache eviction. Validate after Plan 395 ships; track in a follow-up plan if F2 gate passes.
- **F3 fusion (HOLA × DualPoolBandit)** — consolidation rule for cache → state absorption. Research 249 territory; follow-up plan.
- **F4 fusion (HOLA × AM)** — online + offline compaction composition. Plan 271 territory; follow-up plan.
- **Per-NPC HLA cache variant** — using HLA `evolve_hla` (in `katgpt-sense/src/reconstruction.rs`) as the "compressive recurrent state" and a per-NPC episodic KV cache as the hippocampus. This is a riir-ai follow-up; the katgpt-rs primitive is substrate-agnostic and the riir-ai guide is not triggered (verdict is GOAT not Super-GOAT).
- **riir-neuron-db fusion** — persisting HOLA cache contents across sessions via `MerkleFrozenEnvelope`. Out of scope; the cache is inference-local state.

---

## TL;DR

Plan 395 ships the HOLA hippocampal exact KV cache as a `katgpt-rs` open primitive (`crates/katgpt-core/src/hippocampal_cache.rs`, feature `hippocampal_cache`, opt-in). The cache stores the top-w tokens by intrinsic delta-rule write magnitude `β·‖e‖` (free from the shipped GDN2 backbone, Plan 105) and reads them via a decoupled RMSNorm-γ sharpened softmax. GOAT gate G1–G4 is modelless (eviction correctness, latency, no-regression, synthetic retrieval); G5 (perplexity + RULER) is deferred to riir-train (tracked issue). Modelless unblock for the γ parameter (§3.5): ship `gamma=ones` as default + `gamma=per_key_norm_rescale` as the deterministic-correction variant; only defer γ-tuning to riir-train if both fail G4 with documented reason. The primitive competes for the KV-compression slot alongside AM (Plan 271) and Sink-Aware (Plan 287); demote the loser at G5 time.
