# Plan 275: SwiR Switch-Thinking — Explicit↔Latent Mode Controller (Modelless)

**Date:** 2026-06-15
**Research:** [katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md](../.research/241_SwiReasoning_Explicit_Latent_Switch.md)
**Source paper:** [arxiv 2510.05069](https://arxiv.org/abs/2510.05069) — SwiReasoning (ICLR 2026, Shi et al., Georgia Tech / Microsoft)
**Target:** `katgpt-rs/src/swir/` (new module) + Cargo feature `swir_switch_thinking`
**Status:** Active — Phase 1 ✅ complete, Phase 2 pending
**Depends On:** `thinking_cot` (Plan 194, for `ThinkingStrategy` integration point), `rim_slots` (Plan 172, for latent workspace reuse — optional, can use standalone buffer), `selectivity_router` (Plan 204, for explicit-only fallback on rigid-constraint tasks)
**GOAT Criteria:** G1 (+1.5pp accuracy vs `thinking_cot`), G2 (1.3× token efficiency at fixed accuracy), G3 (<200ns per `step()` call, zero alloc), G4 (soft-embedding in vocab convex hull), G5 (no regression when disabled), G6 (auto-fallback on rigid-constraint tasks)

---

## Goal

Distill SwiReasoning (ICLR 2026) into a generic, modelless, MIT-licensed Rust module under `katgpt-rs/src/swir/`. Provides training-free explicit↔latent reasoning mode switching driven by block-relative entropy trends, with asymmetric dwell windows and a switch count controller for overthinking suppression. Integrates into `thinking_cot` (Plan 194) as a new `ThinkingStrategy`. Default-off behind `swir_switch_thinking` until GOAT gate G1–G6 passes.

**Selling point:** First katgpt-rs primitive that switches between token-space and latent-space reasoning at runtime, filling the exact gaps Research 187 identified in the thinking-cot stack ("no signal to stop thinking mid-reasoning", "no per-instance early exit", "resamples from same distribution — no mode switch"). Paper reports +1.8–3.1pp accuracy and 1.36–6.8× efficiency, plug-and-play at inference.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed)

Goal: compiling, tested, feature-gated module implementing the core SwiR state machine on synthetic entropy streams. Public API surface frozen. No model integration yet.

### Tasks

- [x] **T1.1** Create `src/swir/` directory with empty `mod.rs`
- [x] **T1.2** Add feature flag `swir_switch_thinking = []` to `katgpt-rs/Cargo.toml` features section (after `thinking_cot`)
- [x] **T1.3** Add `#[cfg(feature = "swir_switch_thinking")] pub mod swir;` to `src/lib.rs` (alphabetical, after `spectralquant` or similar)
- [x] **T1.4** Implement `src/swir/types.rs`:
  - [x] `ThinkMode` enum (`Explicit`, `Latent`) with `#[repr(u8)]`
  - [x] `SwiRConfig` struct (w_e_to_l: u32 default 512, w_l_to_e: u32 default 0, c_max: u32 default 20, c_convergence_fraction: f32 default 0.5, answer_budget_b: u32 default 256, alpha_0: f32 default 0.6, beta_0: f32 default 0.7, max_steps: u32)
  - [x] `SwiRConfig::default()` returning paper's best-practice values
  - [x] `StepAction` enum: `EmitToken(u32)`, `EmitSoftEmbedding`, `InjectControlToken(ControlToken)`, `Terminate`
  - [x] `ControlToken` enum: `CloseThink` (`</think>`), `ForceAnswerPrefix` (`</think>\n\nThe final answer is`)
  - [x] `SwiRStats` struct (switches_total, latent_steps, explicit_steps, mode_at_termination) for debugging/benchmarks
- [x] **T1.5** Implement `src/swir/controller.rs` — `SwiRController` state machine:
  - [x] Struct fields: mode, reference_entropy, dwell_steps, switch_count, injection_queue (small VecDeque or fixed `[u32; 8]` ring), answer_budget_remaining, config, stats
  - [x] `SwiRController::new(config)` initializes mode=Latent, reference_entropy=NaN (set on first step), switch_count=0, queue empty
  - [x] `fn step(&mut self, entropy: f32, step_index: u32) -> StepAction` — Algorithm 1 of the paper:
    1. If queue non-empty: pop and return `InjectControlToken`. If termination budget hits 0, return `Terminate`.
    2. First-step init: if reference_entropy is NaN, set `reference_entropy = entropy`, `dwell_steps = 0`.
    3. Mode-switch logic (paper §3.3):
       - If `mode == Latent ∧ entropy < reference_entropy`: switch to Explicit, increment switch_count, reset reference_entropy + dwell_steps
       - Else if `mode == Explicit ∧ entropy > reference_entropy ∧ dwell_steps ≥ w_e_to_l`: switch to Latent, reset
       - Else: keep mode, increment dwell_steps
    4. Switch count triggers (paper §3.4):
       - If `mode == Explicit ∧ ½c_max ≤ switch_count ≤ c_max`: enqueue `CloseThink` (convergence)
       - Else if `mode == Explicit ∧ switch_count > c_max`: enqueue `ForceAnswerPrefix`, set answer_budget_remaining = answer_budget_b (termination)
    5. Return `EmitToken(0)` (caller fills token id) if mode==Explicit, `EmitSoftEmbedding` if mode==Latent
  - [x] `fn should_mix_signal(&self) -> Option<(SignalMixKind, f32)>` — returns `Some((LatentEntry, α_t))` or `Some((ExplicitExit, β_t))` only on the first step after a switch, None otherwise. Schedule: `α_t = α_0 + (1 - α_0) * step_index / max_steps`, same for β.
  - [x] `fn stats(&self) -> SwiRStats`
- [x] **T1.6** Implement `src/swir/soft_embedding.rs` — latent-mode soft embedding:
  - [x] `fn soft_embedding(probs: &[f32], embedding_matrix: &[f32], embedding_dim: usize, out: &mut [f32])` — `ẽ_t = Σ_v p_t[v] * e(v)`, writes to `out` (length=embedding_dim, caller-allocated)
  - [x] Zero-overhead: no allocation. Caller responsible for `out.zero_fill()` before call (or document that this is "accumulate" semantics — TBD which is cleaner; lean toward zero-internal-alloc by requiring caller to pre-zero).
  - [x] SIMD chunked loop (8-wide) over `embedding_dim` for the inner reduction.
  - [x] Numerical guard: if `probs` does not sum to ≈1, normalize on the fly with a single pre-pass (documented cost).
- [x] **T1.7** Implement `src/swir/signal_mix.rs`:
  - [x] `fn mix_thinking_signal(soft_embed: &mut [f32], control_token_embed: &[f32], ratio: f32)` — `out ← ratio * out + (1 - ratio) * control_token_embed`. In-place, no alloc.
  - [x] Assert `ratio ∈ [0, 1]` in debug builds.
- [x] **T1.8** Implement `src/swir/convex_hull_check.rs` (G4 invariant):
  - [x] `fn in_vocab_convex_hull(soft_embed: &[f32], embedding_matrix: &[f32], embedding_dim: usize) -> bool` — for each dim d, check `min_v e(v)[d] ≤ soft_embed[d] ≤ max_v e(v)[d]`. O(vocab * embedding_dim) but only runs in test/debug, not hot path.
  - [x] Used in unit tests to verify Lyapunov-style invariant.
- [x] **T1.9** Unit tests in `src/swir/controller.rs` (#[cfg(test)]):
  - [x] `test_first_step_initializes_reference_entropy` — NaN → real value
  - [x] `test_latent_to_explicit_on_confidence_rise` — H_t < H̄ triggers switch
  - [x] `test_explicit_to_latent_requires_dwell_window` — H_t > H̄ but dwell < W_E→L stays explicit
  - [x] `test_explicit_to_latent_fires_after_dwell` — dwell ≥ W_E→L + H_t > H̄ triggers switch
  - [x] `test_switch_count_incremented_only_on_latent_to_explicit`
  - [x] `test_convergence_trigger_at_half_cmax` — switch_count=½c_max enqueues CloseThink
  - [x] `test_termination_trigger_above_cmax` — switch_count>c_max enqueues ForceAnswerPrefix + sets budget
  - [x] `test_terminate_after_answer_budget_exhausted`
  - [x] `test_signal_mix_schedule_at_switch_instants` — ratio increases with step_index per α_t/β_t schedule
  - [x] `test_no_signal_mix_on_non_switch_steps`
- [x] **T1.10** Unit tests in `src/swir/soft_embedding.rs`:
  - [x] `test_uniform_probs_returns_centroid` — uniform p over k one-hot vectors returns mean embedding
  - [x] `test_one_hot_prob_returns_token_embedding` — p concentrated on token v returns e(v)
  - [x] `test_result_lies_in_vocab_convex_hull` — random probs, G4 invariant holds (covered by convex_hull_check::tests::random_soft_embeddings_all_in_hull)
  - [x] `test_simd_matches_naive` — chunked SIMD matches naive O(vocab·dim) loop
- [x] **T1.11** Doc tests in `src/swir/mod.rs` showing a minimal end-to-end trace on a synthetic entropy stream (no real model) — exercises the controller through Explicit→Latent→Explicit cycle and verifies stats.
- [x] **T1.12** Feature gate audit: `cargo build --no-default-features --features "swir_switch_thinking"` compiles; `cargo build` (with defaults) does NOT include swir code.

**Exit criteria for Phase 1:** ✅ MET. All 12 task groups complete. `cargo test --features swir_switch_thinking swir::` passes 26/26 (10 controller + 4 entropy + 4 soft_embedding + 4 signal_mix + 4 convex_hull_check). Public API (`SwiRController`, `SwiRConfig`, `StepAction`, `soft_embedding`, `mix_thinking_signal`) frozen. Bonus: `SwiRConfig::default_for_model(embedding_dim)` constructor and `ControlTokenIds` wiring type added per T2.4 anticipation.

---

## Phase 2 — Integration with thinking_cot (Plan 194)

Goal: wire SwiR into the existing `thinking_cot` framework so it can actually drive a real decode loop. No new model required — uses Gemma/Qwen-style models already supported.

### Tasks

- [ ] **T2.1** Audit `src/lib.rs` exports and `thinking_cot` module (Plan 194) for the existing `ThinkingStrategy` trait (or equivalent trait/type that switches between direct/CoT/early-exit modes). If no such trait exists yet, define a minimal one in `src/thinking_cot/strategy.rs`:
  ```rust
  pub trait ThinkingStrategy {
      fn on_step(&mut self, ctx: &mut StepContext) -> StepDirective;
  }
  pub struct StepContext<'a> {
      pub logits: &'a [f32],
      pub step_index: u32,
      pub max_steps: u32,
      pub embedding_matrix: &'a [f32],
      pub embedding_dim: usize,
      pub control_token_ids: ControlTokenIds,  // <think>, </think>, etc.
  }
  pub enum StepDirective {
      EmitToken(u32),
      EmitSoftEmbedding(/* written into ctx scratch */),
      InjectTokens(Vec<u32>),
      Terminate,
  }
  ```
- [ ] **T2.2** Implement `src/swir/strategy_adapter.rs` — `impl ThinkingStrategy for SwiRController`:
  - [ ] Compute entropy from `ctx.logits` ( Shannon: `H = -Σ p log p`, with a SIMD-friendly chunked loop; clamp `log(0)` to 0 via masked select).
  - [ ] Call `self.step(entropy, ctx.step_index)`.
  - [ ] Translate `StepAction` to `StepDirective`. For `EmitSoftEmbedding`, call `soft_embedding()` writing into the strategy's pre-allocated scratch buffer, then apply signal mixing if `should_mix_signal()` returns Some.
  - [ ] Hold scratch buffer as a field on the adapter (Vec<f32>::with_capacity(embedding_dim) once, reused).
- [ ] **T2.3** Wire entropy computation: if `katgpt-rs` already has a SIMD entropy kernel (check `src/simd.rs`, `src/llmexec_guard.rs`, `src/breakeven/`), reuse. If not, add a minimal `pub fn shannon_entropy(probs: &[f32]) -> f32` to `src/swir/entropy.rs` with a chunked SIMD loop and a `fastmax` trick for `p log p` stability.
- [ ] **T2.4** Add `SwiRController::default_for_model(embedding_dim)` constructor returning the paper's best-practice config with `alpha_0=0.6, beta_0=0.7, w_e_to_l=512, w_l_to_e=0, c_max=20, c_convergence_fraction=0.5, answer_budget_b=256`. Document in rustdoc that these are paper defaults (Qwen3-8B Tab. 6) and may need tuning per model.
- [ ] **T2.5** Integration test: drive SwiR through a mock decode loop with synthetic logits (e.g., a Gaussian-mixture entropy schedule that triggers Latent→Explicit→Latent→Explicit). Verify:
  - [ ] Soft-embedding outputs satisfy G4 convex-hull invariant at every latent step.
  - [ ] Switch count matches expected schedule from the synthetic entropy.
  - [ ] Convergence trigger fires when switch_count = ½c_max.
  - [ ] Termination trigger fires when switch_count > c_max.
- [ ] **T2.6** Feature gate composition: add `swir_switch_thinking = ["thinking_cot"]` dependency in Cargo.toml. Document that this enables latent mode via soft embedding (requires embedding matrix access on every decode step — verify `thinking_cot` exposes this).

**Exit criteria for Phase 2:** `cargo test --features swir_switch_thinking` passes including integration tests with synthetic logits. SwiR can drive a decode loop end-to-end against a mock.

---

## Phase 3 — Real Model Integration & GOAT Gate

Goal: prove the GOAT gate on a real model (Gemma 2 or Qwen3 family already supported). Benchmarks against `thinking_cot` baseline.

### Tasks

- [ ] **T3.1** Pick the smallest real model that supports a `<think>` token: Qwen3-1.7B if available locally; otherwise Gemma-scope model with a synthetic `<think>` token added via prompt engineering. Document the choice in `src/swir/README.md`.
- [ ] **T3.2** Benchmark harness in `src/swir/bench.rs`:
  - [ ] Load MATH500 subset (50 problems for speed; full 500 for final GOAT proof).
  - [ ] Run two configurations: (a) `thinking_cot` baseline, (b) `thinking_cot` + `swir_switch_thinking`.
  - [ ] Measure: accuracy (Pass@1), total tokens generated, wall-clock latency, TFLOPs (estimate from layer counts).
  - [ ] Report: average accuracy delta, token efficiency ratio, latency ratio, Pareto curve at multiple C_max values (4, 8, 16, 20, 32, ∞).
- [ ] **T3.3** GOAT gate G1 (accuracy): avg accuracy delta ≥ +1.5pp on MATH500 subset. If fails on subset but full-set passes, escalate to full 500.
- [ ] **T3.4** GOAT gate G2 (efficiency): at 90% of baseline accuracy, swir uses ≥ 1.3× fewer tokens. Plot the Pareto curve.
- [ ] **T3.5** GOAT gate G3 (perf): benchmark `SwiRController::step()` in isolation — verify < 200ns per call on the target hardware. Use `criterion` or `divan`. If over budget, profile: the main suspect is the entropy compute (O(vocab_size) per step). Consider: (a) entropy from top-k logits only (paper uses full softmax entropy, but top-k is a reasonable approximation), (b) cache entropy EMA across steps and only recompute every k steps.
- [ ] **T3.6** GOAT gate G4 (convex hull): run the convex-hull check on 1000 random soft-embedding outputs from the real model — all must pass. If any fail, investigate numerical drift in the SIMD kernel.
- [ ] **T3.7** GOAT gate G5 (no regression): run the existing `thinking_cot` and `collapse_aware_thinking` test suites with `swir_switch_thinking` disabled — 100% pass.
- [ ] **T3.8** GOAT gate G6 (auto-fallback): construct a synthetic "rigid-constraint" task (paper's 3D-surface-shortest-path style) and verify that `selectivity_router`'s kurtosis signal forces explicit-only mode, bypassing SwiR's latent mode. If selectivity_router doesn't fire, add a manual escape hatch: `SwiRConfig::disable_latent_mode_on_high_kurtosis: bool` (default true) that consults an externally-supplied kurtosis scalar each step.
- [ ] **T3.9** Ablation studies on the internal benchmark:
  - [ ] W_E→L ∈ {64, 128, 256, 512, 1024} — expect 512 to win (paper Tab. 3).
  - [ ] α_0 ∈ {0.3, 0.6, 0.9, 1.0} — expect broad plateau (paper Tab. 2).
  - [ ] C_max ∈ {4, 8, 16, 20, 32, ∞} — expect 20 to be sweet spot (paper Tab. 10).
  - [ ] Signal mixing on/off — expect +0.6pp from mixing (paper Tab. 9).
- [ ] **T3.10** Write `src/swir/BENCHMARKS.md` with all results. If G1–G6 pass → proceed to T4.1. If G1 fails → keep `swir_switch_thinking` opt-in, document the partial win (G2 efficiency gain alone is still useful).
- [ ] **T3.11** Update `katgpt-rs/.benchmarks/` with a `NNN_swir_switch_thinking.md` (next free NNN — check folder first).

**Exit criteria for Phase 3:** G1–G6 verdict recorded in `BENCHMARKS.md`. Decision: promote to default (all pass) / keep opt-in (partial pass) / demote (G3 fail or G1 catastrophic fail).

---

## Phase 4 — Default Promotion & Demotion (conditional)

Only execute if Phase 3 T3.10 verdict is "promote to default".

### Tasks

- [ ] **T4.1** Add `swir_switch_thinking` to the `default = [...]` feature list in `Cargo.toml`.
- [ ] **T4.2** Add `swir_switch_thinking` to the `full = [...]` feature list.
- [ ] **T4.3** Update `katgpt-rs/README.md` to mention SwiR in the reasoning module list.
- [ ] **T4.4** If SwiR wins decisively (G1 ≥ +2pp AND G2 ≥ 1.5×), evaluate demoting the existing `collapse_aware_thinking` default — does SwiR subsume it? Run ablation: SwiR alone vs `collapse_aware_thinking` alone vs both. If SwiR alone matches or beats the combination, demote `collapse_aware_thinking` to opt-in. If complementary, keep both default-on with documented composition semantics.
- [ ] **T4.5** Commit with `feat(swir): promote swir_switch_thinking to default — GOAT proved G1-G6` (or similar).

---

## Phase 5 — Fusion Explorations (Stretch, post-GOAT)

Only execute after Phase 3 ships. Each fusion from Research 241 §2.3 warrants its own plan.

### Tasks

- [ ] **T5.1** **Fusion A** (sub-token continuous-mode router): create `katgpt-rs/.research/242_swir_dmax_continuous_router.md` exploring the sigmoid-weighted blend `ẽ_t = σ(λ·(H̄−H_t)) · ẽ_latent + (1 − σ(...)) · e_argmax_token` using DMax SPD's hybrid embedding pattern. Validate Pareto vs binary SwiR on MATH500. If wins → `katgpt-rs/.plans/276_swir_continuous_router.md`. **Super-GOAT candidate per Research 241.**
- [ ] **T5.2** **Fusion B** (MUX × SwiR bandit arm): create `katgpt-rs/.research/243_swir_mux_bandit_arm.md` exploring adding a Latent arm to Plan 211's Three-Mode Router. Validate bandit convergence on a mixed-difficulty benchmark suite. If wins → extend Plan 211 (no new plan). **Super-GOAT candidate per Research 241.**
- [ ] **T5.3** **Fusion C** (NPC two-brain): create `riir-ai/.research/NNN_swir_npc_think_info_bridge.md` (private) exploring SwiR's entropy-trend switch as the missing think→info brain commit trigger per AGENTS.md latent-vs-raw rules. Latent mode = think brain exploration; Explicit mode = info brain commit. Switch count = bounded deliberation budget. **Routing: riir-ai guide only if Fusion A validates the core primitive.** This is the riir-ai selling-point doc, not katgpt-rs.

---

## Dependencies

```
swir_switch_thinking ──┬── thinking_cot (Plan 194, for ThinkingStrategy trait)
                       ├── rim_slots (Plan 172, OPTIONAL — latent workspace reuse, can use standalone)
                       └── selectivity_router (Plan 204, OPTIONAL — auto-fallback on rigid tasks)

Phase 5 fusions:
  Fusion A: swir_switch_thinking + dmax_spd (Plan 109)
  Fusion B: swir_switch_thinking + three_mode_router (Plan 211) + mux_pruner (Research 158)
  Fusion C: swir_switch_thinking (open primitive) + riir-ai game IP (private)
```

---

## Notes

- **Paper's scope:** SwiR is plug-and-play at the model.generate() level in HuggingFace. We're porting the *algorithm* (the controller), not the integration layer — our integration layer is `thinking_cot`.
- **What we ship publicly (MIT, katgpt-rs):** `SwiRController`, `soft_embedding`, `mix_thinking_signal`, `SwiRConfig`, `StepAction`. No game semantics, no chain semantics.
- **What stays private (riir-ai):** Fusion C's mapping to NPC think-brain/info-brain, the game-AI selling-point narrative. Only if Phase 5 T5.3 is pursued.
- **Hyperparameter defaults:** Qwen3-8B Tab. 6 best-practices. Will likely need per-model tuning; `SwiRConfig::default_for_model(model_name)` helper is a P2 nicety, not blocking.
- **Numerical stability:** entropy of full softmax can be expensive O(vocab_size). If G3 (200ns) is hard to hit on a 256k vocab, consider top-k entropy approximation (paper uses full softmax but top-k is a documented approximation in the entropy-estimation literature).
- **Convex-hull invariant (G4):** the soft embedding `ẽ_t = Σ_v p_t[v] · e(v)` is a convex combination of vocabulary embeddings, so it MUST lie in the per-dim [min, max] range of the vocab embeddings. This is a free correctness check — any violation indicates a SIMD bug or numerical drift.
- **Feature gate naming:** `swir_switch_thinking` matches the existing naming pattern (`collapse_aware_thinking`, `three_mode_router`, `regime_transition`). The `swir_` prefix avoids collision with potential future `switch_thinking` generic.

---

## TL;DR

Implement SwiReasoning (ICLR 2026, arXiv:2510.05069) as a modelless, MIT-licensed `src/swir/` module in katgpt-rs. Three primitives: `SwiRController` (state machine for entropy-trend-driven Explicit↔Latent mode switch with asymmetric dwell windows and switch count controller), `soft_embedding` (probability-weighted vocabulary mixture for latent mode, SIMD), `mix_thinking_signal` (control-token embedding blending at switch instants). Integrates into `thinking_cot` (Plan 194) as a new `ThinkingStrategy`. Five phases: (1) skeleton with synthetic tests, (2) thinking_cot integration with mock logits, (3) real-model GOAT gate G1–G6 (≥+1.5pp accuracy, ≥1.3× token efficiency, <200ns/call, convex-hull invariant, no regression, auto-fallback), (4) promote to default if all gates pass (demote `collapse_aware_thinking` if subsumed), (5) stretch — three Super-GOAT fusion explorations (continuous-mode router, MUX bandit arm, riir-ai NPC two-brain). Feature flag `swir_switch_thinking`, default-off until Phase 3 GOAT proof. Research note: `katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md`.
