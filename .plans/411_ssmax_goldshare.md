# Plan 411: SSMax (log-N Attention Temperature) + GoldShare Diagnostic

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/392_Attention_Dilution_SSMax_GoldShare.md](../.research/392_Attention_Dilution_SSMax_GoldShare.md)
**Source paper:** [arxiv 2607.01538](https://arxiv.org/abs/2607.01538) — *Can Language Models Actually Retrieve In-Context?* (Gollapudi et al., UC Berkeley / UT Austin, 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/parallax_attn.rs` (SSMax extension) + `katgpt-rs/crates/katgpt-core/src/data_probe/gold_share.rs` (new diagnostic) + Cargo features `ssmax_temperature`, `gold_share_probe`
**Status:** Active — Phase 0 (planning)

---

## Goal

Ship two novel modelless primitives distilled from Research 392:

1. **SSMax** — a length-aware multiplicative attention-temperature primitive: `s̃ = s_L · log(N) · s_{L,h,t}` applied to pre-softmax / pre-sigmoid logits. Cancels the `(N−1)` growth in the attention denominator (paper's bound: `α_gold ≈ 1/(1 + (N−1) · N^{−s·Δ})`, bounded when `s·Δ > 1`). Default `s_L = 1.0` is **truly modelless** (zero training, zero new parameters); an optional rolling-`Δ` estimator gives a runtime-adaptive `s_L`. Composes with sigmoid parallax (length-adaptive sharpener for the residual dilution cases sigmoid alone doesn't fully solve) and with standard SDPA (length-extrapolation fix for callers consuming softmax-trained weights).

2. **GoldShare** — `‖a^G_L‖ / ‖a_L‖` content-specific output-fraction diagnostic. Decomposes a layer's attention output into gold-derived and distractor-derived fractions; detects when the layer's output has been *rewritten* from gold-content to aggregate-noise at comparable magnitude (the paper's Table 1 shows `‖a_L‖` shrinks ~36% while gold-share collapses 130× across N ∈ {500→10k}). Complements `effective_rank` (content-agnostic aggregate) and `stable_rank_update` (per-sink degeneracy) in `data_probe/`.

Both behind opt-in feature flags. GOAT gate (G1–G5) decides promote-to-default vs demote-opt-in per AGENTS.md §Feature Flag Discipline.

**Why modelless (§3.5 check):** SSMax is a deterministic logit rescale — zero training, zero backprop. GoldShare is a read-only diagnostic. Neither touches weights. The paper trains `s_L`; we derive it analytically (`s_L > 1/Δ_typical`) and ship `s_L = 1.0` as the truly-modelless default. No riir-train deferral.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed)

Goal: two compiling, feature-gated, minimally-tested modules with the public API surface frozen. No GOAT gate yet.

### Tasks

- [ ] **T1.1** Add two feature flags to `katgpt-rs/Cargo.toml` features section (alphabetical, near `sink_aware_attn`):
  ```toml
  ssmax_temperature = ["katgpt-core/ssmax_temperature"]   # Length-aware log-N attention temperature (Plan 411, Research 392, arxiv 2607.01538). Modelless; default s_L = 1.0. Composes with parallax_attn (sigmoid) and attention.rs (SDPA). Opt-in pending G1/G2 GOAT gate.
  gold_share_probe = ["data_probe"]                       # GoldShare content-specific output-fraction diagnostic ‖a^G_L‖/‖a_L‖ (Plan 411, Research 392). Complements effective_rank / stable_rank_update. Opt-in diagnostic.
  ```
- [ ] **T1.2** Forward the features to katgpt-core: add `ssmax_temperature = []` and `gold_share_probe = ["dirichlet_energy"]` (or the right `data_probe` umbrella dep) to `crates/katgpt-core/Cargo.toml`.
- [ ] **T1.3** Add `#[cfg(feature = "ssmax_temperature")] pub mod ssmax;` to `crates/katgpt-core/src/lib.rs` (alphabetical).
- [ ] **T1.4** Implement `crates/katgpt-core/src/ssmax.rs` types:
  - [ ] `SsmaxConfig { scale: f32, log_n: f32 }` — `scale` is `s_L`, `log_n` is precomputed `ln(N)` (avoids recomputing in the hot loop; caller passes it because they already know N).
  - [ ] `SsmaxMode` enum: `Fixed { s_l: f32 }` (truly modelless, default `s_l = 1.0`), `Adaptive { rolling_delta: f32 }` (caller-managed rolling estimate; `s_l = 1.0 / rolling_delta` clamped to `[0.1, 10.0]`).
  - [ ] `SsmaxScratch { scaled_logits: Vec<f32> }` — caller-owned scratch, reused across calls (no hot-loop alloc per AGENTS.md).
- [ ] **T1.5** Implement `crates/katgpt-core/src/ssmax.rs` core function:
  ```rust
  /// Rescale pre-attention logits in place by `s_L · log(N)`.
  /// `logits` is `(n_heads, n_kv)` row-major f32, modified in place.
  /// `n` is the number of attended keys (drives `log_n`).
  #[inline]
  pub fn apply_ssmax_inplace(logits: &mut [f32], mode: &SsmaxMode, log_n: f32) {
      let s_l = match mode {
          SsmaxMode::Fixed { s_l } => *s_l,
          SsmaxMode::Adaptive { rolling_delta } => (1.0 / rolling_delta.max(1e-3)).clamp(0.1, 10.0),
      };
      let mult = s_l * log_n;
      // Chunked 8-wide loop for SIMD auto-vectorization (AGENTS.md hot-loop rule)
      for chunk in logits.chunks_exact_mut(8) {
          for x in chunk { *x *= mult; }
      }
      for x in logits.chunks_exact_mut(8).remainder() { *x *= mult; }
  }
  ```
- [ ] **T1.6** Implement `crates/katgpt-core/src/data_probe/gold_share.rs`:
  - [ ] `GoldShareReport { gold_norm: f32, total_norm: f32, gold_share: f32, gold_pre_softmax_max: f32, noise_gap: f32 }`
  - [ ] `gold_share(attn_weights, values, gold_mask, w_o, scratch) -> GoldShareReport` — signature sketched in Research 392 §2.2. Reuse `data_probe/geometry.rs` scratch conventions.
  - [ ] `gold_share_flat(...)` — flat-`&[f32]` variant matching `apply_dual_policy_gate_flat`'s convention in `sink_classify.rs` (consistency with the existing data_probe API).
- [ ] **T1.7** Wire `pub mod gold_share;` into `crates/katgpt-core/src/data_probe/mod.rs` behind `#[cfg(feature = "gold_share_probe")]`, with re-exports mirroring the `sink_classify` block.
- [ ] **T1.8** Write unit tests in `crates/katgpt-core/src/ssmax.rs`:
  - [ ] `apply_ssmax_inplace` scales every logit by `s_l · log_n` exactly (bit-exact on small input).
  - [ ] `SsmaxMode::Adaptive` clamps `s_l` to `[0.1, 10.0]` for both tiny and huge `rolling_delta`.
  - [ ] In-place modification is confirmed (input == output on identity scale).
  - [ ] SIMD chunk path and remainder path produce identical results to a naive scalar loop.
- [ ] **T1.9** Write unit tests in `crates/katgpt-core/src/data_probe/gold_share.rs`:
  - [ ] When `gold_mask` is all-true, `gold_share == 1.0`.
  - [ ] When `gold_mask` is all-false, `gold_share == 0.0`.
  - [ ] On the paper's Table 1 toy (4-head, 8-key, half gold), `gold_share` matches the hand-computed `‖a^G‖/‖a‖` to 4 decimal places.
  - [ ] `gold_share` and `gold_share_flat` agree bit-exactly on the same input.
- [ ] **T1.10** Add a doc example to `ssmax.rs` showing `SsmaxMode::Fixed { s_l: 1.0 }` applied to a synthetic logit vector with `n = 10_000`.
- [ ] **T1.11** Run `cargo check -p katgpt-core --features ssmax_temperature,gold_share_probe --lib` — must pass with no new warnings.

**STATUS: ☐** — Phase 1 not started.

---

## Phase 2 — SSMax Composition Wiring

Goal: SSMax is callable from both attention paths (sigmoid parallax + standard SDPA), behind the feature flag, without changing default behavior.

### Tasks

- [ ] **T2.1** Extend `ParallaxConfig` in `crates/katgpt-core/src/parallax_attn.rs` with an optional `ssmax: Option<SsmaxMode>` field (default `None` — no change to existing behavior). Field is `#[cfg(feature = "ssmax_temperature")]`.
- [ ] **T2.2** In the parallax forward path, if `config.ssmax.is_some()`, call `apply_ssmax_inplace` on the score-matrix scratch *before* the sigmoid/softmax kernel is applied. The score matrix is already computed into a scratch buffer (see `compute_score_matrix` in `attn_match/score_matrix.rs` for the pattern); SSMax is one extra in-place pass.
- [ ] **T2.3** Add a sink-aware + SSMax composition entry point `tiled_attention_parallax_forward_sink_aware_ssmax` behind `#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn", feature = "ssmax_temperature"))]` — mirrors the existing `tiled_attention_parallax_forward_sink_aware` (Plan 289). SSMax applied first (logit level), then sink-aware gate (output level) — they compose cleanly because they operate at different stages.
- [ ] **T2.4** In `crates/katgpt-core/src/attention.rs` (standard SDPA), add an optional `ssmax: Option<SsmaxMode>` to the config struct (if one exists) OR expose a standalone `scaled_dot_product_ssmax` entry point that wraps the base SDPA with a logit-rescale pre-pass. Prefer the wrapper to avoid touching the hot default path.
- [ ] **T2.5** Document in `ssmax.rs` module doc: SSMax does NOT apply to `funcattn` (Research 261 closed: basis-mode structure has no `(n,n)` attention matrix, so dilution is structurally absent). Don't wire it.
- [ ] **T2.6** Run `cargo check -p katgpt-core --features parallax_attn,ssmax_temperature --lib` — must pass.
- [ ] **T2.7** Run `cargo check -p katgpt-core --features parallax_attn,sink_aware_attn,ssmax_temperature --lib` — must pass (the 3-way composition).

**STATUS: ☐** — Phase 2 not started.

---

## Phase 3 — GoldShare Composition Wiring

Goal: GoldShare is callable as a diagnostic from any layer that already exposes its attention weights and values (the existing sink-aware and data_probe callers).

### Tasks

- [ ] **T3.1** Extend `data_probe/mod.rs` re-export block to include `GoldShareReport`, `gold_share`, `gold_share_flat`.
- [ ] **T3.2** Add a cross-reference doc in `sink_classify.rs`: when a sink classifier hits the gold position with low `gold_share`, that is a *broadcast that failed* (signal was in the head per the classifier, but didn't survive normalization per GoldShare). Document the joint interpretation.
- [ ] **T3.3** Add an optional `gold_share: Option<GoldShareReport>` field to `SinkDiagnostic` (behind `gold_share_probe` feature) so a single classifier pass can populate both diagnostics when both features are on.
- [ ] **T3.4** Write an integration test: run `classify_all_sinks` + `gold_share` on the paper's Table 1 toy (4-head, 8-key, half gold), verify the joint report is self-consistent (gold position is classified as Broadcast, but `gold_share` is low — the "broadcast that failed" signature).
- [ ] **T3.5** Run `cargo check -p katgpt-core --features sink_aware_attn,gold_share_probe --lib` — must pass.

**STATUS: ☐** — Phase 3 not started.

---

## Phase 4 — GOAT Gate (G1–G5)

Goal: prove the gain over the default (sigmoid parallax without SSMax; no GoldShare). Per AGENTS.md §Feature Flag Discipline + Research 392 §3 GOAT gate. Use `CARGO_TARGET_DIR=/tmp/ssmax_goldshare_gate` to avoid locking the main target dir; clean up when done.

### Bench

- [ ] **T4.1** Create `benches/ssmax_goat.rs` — a synthetic retrieval task with growing N ∈ {1k, 10k, 100k}: generate `n_heads × n_kv` logit matrices with a planted gold position (top-1 pre-softmax) and a growing pool of distractors; measure argmax preservation across N with and without SSMax. Default sigmoid parallax as baseline. Report a verdict table.
- [ ] **T4.2** Create `benches/gold_share_goat.rs` — replay the paper's Table 1 sweep synthetically: construct attention outputs where `‖a_L‖` is held ~constant but `‖a^G_L‖/‖a_L‖` drops from 0.91 → 0.01; verify `effective_rank` (existing) does NOT detect the swap while `gold_share` (new) does. This is the diagnostic's differentiating test.
- [ ] **T4.3** Implement and run **G1 (correctness)**: SSMax preserves argmax ranking across N ∈ {1k, 10k, 100k} where default sigmoid parallax degrades. Verify the analytical `s_L ≈ 1/Δ_typical` derivation (`SsmaxMode::Adaptive`) produces the same ranking as a brute-force sweep over `s_L ∈ [0.1, 10.0]`. PASS = ranking preserved at all three N for both modes.
- [ ] **T4.4** Implement and run **G2 (quality)**: on a frozen long-context probe — if a RULER-style needle-in-haystack test harness exists in the repo, use it; otherwise construct a synthetic "find the gold key among N distractors" probe at the largest N where default sigmoid parallax starts to degrade. SSMax must improve recall. If sigmoid parallax already passes at all tested N (likely for moderate N), document SSMax as a large-N safety net and benchmark at the largest N achievable. PASS = SSMax recall ≥ default recall at every N, strictly > at the largest N.
- [ ] **T4.5** Implement and run **G3 (latency)**: criterion bench (or `std::time::Instant` matching repo convention) of `apply_ssmax_inplace` overhead — one multiply per logit, must be ≤ 1% of attention forward time at n_kv ≥ 1024. GoldShare overhead ≤ 5% of one attention forward (it's a diagnostic, opt-in). PASS = both under budget.
- [ ] **T4.6** Implement and run **G4 (alloc-free)**: confirm via inspection + a debug-allocations test (if the repo has one; else a `#[test]` that runs the function in a loop and asserts no growth in a pre-allocated scratch). SSMax is in-place logit rescale — zero allocation. GoldShare reuses `data_probe` scratch buffers. PASS = zero allocations in either hot path.
- [ ] **T4.7** Implement and run **G5 (no-regression)**: at small N (where dilution is absent, e.g. N=64), SSMax must not degrade argmax ranking vs default. Verify `s_L · log N · Δ ≈ log N` ⇒ `s_L · Δ ≈ 1` ⇒ at small N the sharpening is mild. PASS = identical ranking at N=64 with and without SSMax.
- [ ] **T4.8** Capture the gate results in `.benchmarks/411_ssmax_goldshare_goat.md` (create the `.benchmarks/` dir if missing). Record raw numbers; honest verdict (PASS/FAIL per gate). Per research skill §3.6: if a PoC refutes a quality claim, do NOT silently revise — record the raw numbers and explicitly state which axis was confirmed vs refuted.

**STATUS: ☐** — Phase 4 not started.

---

## Phase 5 — Promotion / Wiring Decision

Goal: based on Phase 4 results, decide promote-to-default vs demote-opt-in, and update the README Feature Showcase.

### Tasks

- [ ] **T5.1** **Promotion decision (per G1+G2 outcome):**
  - If G1 AND G2 PASS → promote `ssmax_temperature` to default in `parallax_attn` (it's a strict superset of the constant-temperature case when `s_L` is chosen well; `s_L = 1.0` default preserves small-N behavior per G5). Add to the "Always-On Hot Path" features list in `README.md` §E2E Inference Flow.
  - If G2 FAILS (sigmoid parallax already handles the dilution regime at all tested N) → keep `ssmax_temperature` opt-in, document it as a large-N safety net in the README Feature Showcase opt-in section.
- [ ] **T5.2** GoldShare stays opt-in as a diagnostic regardless — promote only if a downstream consumer (sink-aware attention wiring, future runtime NPC cognition probe) depends on it.
- [ ] **T5.3** If SSMax promoted: demote any loser per AGENTS.md ("demote the loser when a newer primitive wins the same slot"). The slot is "attention temperature / logit scaling" — check if any existing constant-temperature primitive (e.g. the `1/√d` in base SDPA) is now strictly dominated. If so, document the demotion in the README and the gate bench.
- [ ] **T5.4** Add a Feature Showcase entry to `README.md` for SSMax (and GoldShare if interesting enough) — model on the existing Plan 287 sink-aware entry: TL;DR, paper cite, what it does, GOAT gate summary, default/opt-in status.
- [ ] **T5.5** Run full CI guard: `cargo check --workspace` (default features) AND `cargo check --workspace --all-features` (the merkle_root lesson — combo regressions). Both must pass.
- [ ] **T5.6** Update Research 392 status from "Done" to "Done — Plan 411 shipped" with a one-line link to this plan and the gate bench.

**STATUS: ☐** — Phase 5 not started.

---

## Stretch (optional, defer with `- [-]`)

- [-] **S1** A `belief_share` analog of GoldShare for HLA per-NPC latent state in `riir-ai/crates/riir-engine/src/hla/` — does the NPC's projection still carry its personal signal, or has it been drowned by aggregate crowd projections? Deferred to a separate riir-ai issue per Research 392 §2.4 — needs PoC per research skill §3.6 before any quality claim. File as `riir-ai/.issues/NNN_*` if pursued.
- [-] **S2** Runtime-adaptive `s_L` via a lock-free rolling-Δ estimator (papaya hashmap per layer). Deferred — the `SsmaxMode::Adaptive` API in T1.4 ships the *contract* (caller-managed `rolling_delta`); a built-in estimator is a Phase 6+ refinement once G2 confirms the adaptive mode is worth the complexity.
- [-] **S3** Lean 4 theorem that `s_L = 1.0, N ≥ 2 ⇒ α_gold(SSMax) ≥ α_gold(base)` under a `Δ ≥ 1` assumption (the paper's bound). Deferred to the cross-repo FV coordinator (`katgpt-rs/.issues/012`) — would extend `KatgptProof` with a length-aware-temperature theorem. Only worth it if SSMax promotes to default.

---

## Cross-references

- **Research 392** — the distillation this plan implements.
- **Research 258 / Plan 287** (arxiv 2606.08105) — sink-aware attention. SSMax + sink-aware compose at different stages (logit level vs output level); the joint entry point in T2.3 wires them.
- **Research 225 / Plan 256** (MSA) — block-sparse attention. Top-B routing from the same paper already ships here; SSMax is the complementary logit-level fix.
- **Research 140** (sigmoid parallax) — the default attention path SSMax extends.
- **Research 261** — closed negative: SSMax does NOT apply to `funcattn` (no `(n,n)` matrix). Documented in T2.5.
- **Research 100 / `ega_attn`** — spectral salience gate on output. Complementary to SSMax (logit level); both can be on simultaneously.

---

## TL;DR

Plan 411 implements two modelless primitives distilled from Research 392 / arXiv:2607.01538: **SSMax** (length-aware `s_L · log N` logit rescaling, default `s_L = 1.0` truly modelless, composes with sigmoid parallax + standard SDPA + sink-aware) and **GoldShare** (`‖a^G_L‖/‖a_L‖` content-specific output-fraction diagnostic, complement to `effective_rank` / `stable_rank_update`). Five phases: skeleton → SSMax wiring → GoldShare wiring → GOAT gate (G1 correctness, G2 quality, G3 latency ≤1%, G4 alloc-free, G5 no-regression at small N) → promotion decision. The paper's Prop 1 (App H) already confirms our default sigmoid attention is the optimal additive-sink form — SSMax is the length-adaptive extension for the residual dilution cases sigmoid alone doesn't fully solve. Promote `ssmax_temperature` to default if G1+G2 pass; GoldShare stays opt-in diagnostic.
