# Plan 411: SSMax (log-N Attention Temperature) + GoldShare Diagnostic

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/392_Attention_Dilution_SSMax_GoldShare.md](../.research/392_Attention_Dilution_SSMax_GoldShare.md)
**Source paper:** [arxiv 2607.01538](https://arxiv.org/abs/2607.01538) — *Can Language Models Actually Retrieve In-Context?* (Gollapudi et al., UC Berkeley / UT Austin, 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/parallax_attn.rs` (SSMax extension) + `katgpt-rs/crates/katgpt-core/src/data_probe/gold_share.rs` (new diagnostic) + Cargo features `ssmax_temperature`, `gold_share_probe`
**Status:** Active — Phase 1 COMPLETE (2026-07-07). Both feature-gated modules compile clean: `ssmax_temperature` (13 unit tests) + `gold_share_probe` (8 unit tests). 1347 total lib tests pass, 0 warnings. Default + all-features both clean (no combo regression). Phases 2–5 pending (GOAT gate G1–G5).

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

- [x] **T1.1** Add two feature flags to `katgpt-rs/Cargo.toml` features section (alphabetical, near `sink_aware_attn`):
  ```toml
  ssmax_temperature = ["katgpt-core/ssmax_temperature"]   # Length-aware log-N attention temperature (Plan 411, Research 392, arxiv 2607.01538). Modelless; default s_L = 1.0. Composes with parallax_attn (sigmoid) and attention.rs (SDPA). Opt-in pending G1/G2 GOAT gate.
  gold_share_probe = ["data_probe", "katgpt-core/gold_share_probe"]  # GoldShare content-specific output-fraction diagnostic ‖a^G_L‖/‖a_L‖ (Plan 411, Research 392). Complements effective_rank / stable_rank_update. Opt-in diagnostic; implies data_probe + sink_aware_attn.
  ```
- [x] **T1.2** Forward the features to katgpt-core: added `ssmax_temperature = []` and `gold_share_probe = ["sink_aware_attn"]` to `crates/katgpt-core/Cargo.toml`. Deviation: gold_share_probe implies sink_aware_attn (not `dirichlet_energy`) for the StableRankScratch convention reuse.
- [x] **T1.3** Added `#[cfg(feature = "ssmax_temperature")] pub mod ssmax;` to `crates/katgpt-core/src/lib.rs` (alphabetical).
- [x] **T1.4** Implemented `crates/katgpt-core/src/ssmax.rs` types: `SsmaxMode` enum (`Fixed { s_l }` / `Adaptive { rolling_delta }`), `SsmaxConfig` bundle, default `Fixed { s_l: 1.0 }`.
- [x] **T1.5** Implemented `apply_ssmax_inplace(logits, mode, log_n)` — chunked 8-wide SIMD-friendly in-place multiply.
- [x] **T1.6** Implemented `crates/katgpt-core/src/data_probe/gold_share.rs` (pre-existing skeleton from prior session, kept as-is): `GoldShareReport` with the 5 plan fields (`gold_norm`, `total_norm`, `gold_share`, `gold_pre_softmax_max`, `noise_gap`), `gold_share`/`gold_share_flat` with `GoldShareScratch` (caller-owned, zero-alloc hot path).
- [x] **T1.7** Wired `#[cfg(feature = "gold_share_probe")] pub mod gold_share;` into `crates/katgpt-core/src/data_probe/mod.rs`, with re-exports of `GoldShareReport`, `GoldShareScratch`, `gold_share`, `gold_share_flat`.
- [x] **T1.8** Wrote 13 unit tests in `crates/katgpt-core/src/ssmax.rs` (all PASS): fixed/adaptive mode resolution, clamping (tiny/huge/zero/negative delta), multiplier math, in-place scaling bit-exactness, SIMD+remainder paths agree with naive, identity-at-multiplier-one, empty-slice no-op, `SsmaxConfig` caching.
- [x] **T1.9** Wrote 8 unit tests in `crates/katgpt-core/src/data_probe/gold_share.rs` (all PASS): all-true/all-false masks, empty-gold-set, degenerate all-zero, paper's Table 1 toy (4-head 8-key half-gold), pre-softmax-max + noise-gap correctness, flat/typed agreement, scratch ensure_capacity no-op.
- [x] **T1.10** Added module-level docs to `ssmax.rs` with composition notes (sigmoid parallax, SDPA, sink-aware; NOT funcattn per Research 261).
- [x] **T1.11** `cargo check -p katgpt-core --features ssmax_temperature,gold_share_probe --lib` passes clean. `--all-features` also clean. Root `cargo check --features ssmax_temperature,gold_share_probe` clean. 1347 tests pass at the crate level.

**STATUS: ✅ DONE** — Phase 1 complete. Committed as `feat(katgpt-core): ssmax + gold_share skeleton (Plan 411 Phase 1)`.

---

## Phase 2 — SSMax Composition Wiring

Goal: SSMax is callable from both attention paths (sigmoid parallax + standard SDPA), behind the feature flag, without changing default behavior.

### Tasks

- [x] **T2.1** Extended `ParallaxConfig` in `crates/katgpt-core/src/parallax_attn.rs` with `#[cfg(feature = "ssmax_temperature")] pub ssmax: Option<crate::ssmax::SsmaxMode>` (default `None`). Manual `Default` impl updated with the cfg-gated field.
- [x] **T2.2** In both parallax forward paths (the main path + `tiled_attention_core`), added `apply_ssmax_to_row` calls before `normalize_attention_weights`. Added a private helper `apply_ssmax_to_row(row, ssmax: Option<&SsmaxMode>)` that's a no-op when None. Updated the 3 test calls to `tiled_attention_core` to pass the cfg-gated None.
- [x] **T2.3** Added explicit 3-way entry point `tiled_attention_parallax_forward_sink_aware_ssmax` behind `#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn", feature = "ssmax_temperature"))]`. Takes an optional `ssmax_mode: Option<&SsmaxMode>` override (explicit param wins over `parallax_config.ssmax`); when `None`, uses the config's ssmax. Delegates to the 2-way forward with a cloned config (the clone is a few f32 + a Copy enum — negligible vs the n×n classifier). Rationale: makes the 3-way composition grep-able and lets callers reuse a base config across SSMax-on/off calls without mutation. The implicit composition (field-on-config) also still works — the 2-way forward picks up `parallax_config.ssmax` automatically.
- [x] **T2.4** Added `tiled_attention_forward_ssmax` to `crates/katgpt-core/src/attention.rs` behind `#[cfg(all(feature = "tiled_attention", feature = "ssmax_temperature"))]`. Folds `s_L · log(N)` into the softmax scale — the zero-overhead way to apply SSMax to flash-attention (no score matrix materialization).
- [x] **T2.5** Documented in `ssmax.rs` module doc: SSMax does NOT apply to `funcattn` (Research 261 closed negative: basis-mode structure has no `(n,n)` attention matrix, so dilution is structurally absent). Not wired.
- [x] **T2.6** `cargo check -p katgpt-core --features parallax_attn,ssmax_temperature --lib` passes clean.
- [x] **T2.7** `cargo check -p katgpt-core --features parallax_attn,sink_aware_attn,ssmax_temperature --lib` passes clean (3-way composition). Also verified `tiled_attention,ssmax_temperature` compiles (the SDPA wrapper).

**STATUS: ✅ DONE** — Phase 2 complete. 9 composition tests added (4 SSMax×parallax + 3 SSMax×sink-aware×parallax + 2 SDPA wrapper). All parallax_attn tests pass with and without `ssmax_temperature` (default `None` preserves bit-identical behavior). 1368 lib tests pass with `parallax_attn,ssmax_temperature`; 1386 with the 3-way combo. All `ParallaxConfig` literal constructions in the crate updated to use `..Default::default()` (robust against future field additions).

---

## Phase 3 — GoldShare Composition Wiring

Goal: GoldShare is callable as a diagnostic from any layer that already exposes its attention weights and values (the existing sink-aware and data_probe callers).

### Tasks

- [x] **T3.1** `data_probe/mod.rs` re-export block already includes `GoldShareReport`, `GoldShareScratch`, `gold_share`, `gold_share_flat` (done in Phase 1 T1.7).
- [x] **T3.2** Added cross-reference doc to `sink_classify.rs` module-level docs: documents the "broadcast that failed" signature (classifier says Broadcast + low gold_share = signal in head, lost in residual) and the "healthy broadcast" contrast.
- [x] **T3.3** Added `#[cfg(feature = "gold_share_probe")] pub gold_share: Option<crate::data_probe::GoldShareReport>` field to `SinkDiagnostic`. Updated both construction sites (`classify_sink_at`, `classify_sink_at_flat`) to initialize the field as `None`.
- [x] **T3.4** Wrote `crates/katgpt-core/tests/plan411_joint_classifier_gold_share.rs` integration test (2 tests, both PASS): `joint_classifier_gold_share_broadcast_that_failed_signature` (paper's Table 1 toy: 4-head 8-key half-gold, verifies gold_pre_softmax_max = 0.05, noise_gap = -0.15, gold_share < 0.5, SinkDiagnostic.gold_share field is accessible and None) + `joint_signature_healthy_broadcast_when_gold_share_high` (contrast case: all-attention-on-gold → gold_share = 1.0, noise_gap = 0.5).
- [x] **T3.5** `cargo check -p katgpt-core --features sink_aware_attn,gold_share_probe --lib` passes clean. Also verified `--features sink_aware_attn` alone (without gold_share_probe) still compiles.

**STATUS: ✅ DONE** — Phase 3 complete. Joint classifier + gold_share report is self-consistent; the "broadcast that failed" signature is detectable.

---

## Phase 4 — GOAT Gate (G1–G5)

Goal: prove the gain over the default (sigmoid parallax without SSMax; no GoldShare). Per AGENTS.md §Feature Flag Discipline + Research 392 §3 GOAT gate. Use `CARGO_TARGET_DIR=/tmp/ssmax_goldshare_gate` to avoid locking the main target dir; clean up when done.

### Bench

- [x] **T4.1** Created `benches/bench_411_ssmax_goat.rs` — synthetic retrieval task with N ∈ {64, 1k, 10k, 100k}, planted gold position (top-1 pre-softmax by Δ=0.5), measures argmax preservation + gold mass recovery with and without SSMax.
- [x] **T4.2** Created `benches/bench_411_gold_share_goat.rs` — sweep that grows N_kv 8→2048 while shrinking gold attention. Verifies gold_share tracks the swap (range 0.94) while effective_rank stays flat (range 0.00).
- [x] **T4.3** **G1 (correctness) ✅ PASS** — SSMax preserves argmax at all N ≥ 64 for both Fixed (`s_L=1.0`) and Adaptive (`s_L=1/Δ`). At N=100k: base gold mass 0.00002, SSMax Fixed 0.003 (185× improvement), SSMax Adaptive 0.47 (29,000× improvement).
- [x] **T4.4** **G2 (quality) ✅ PASS** — SSMax retrieval recall measured via cosine similarity cos(output, v_gold) at N ∈ {1k, 10k}. Base: 0.25, SSMax Adaptive: 0.97 — the output vector points strongly toward the gold value instead of being diluted across distractors. (Initially deferred by a prior session; this session implemented the actual recall test and confirmed PASS.) GoldShare G2 ✅ PASS — differentiating power demonstrated: gold_share collapses 27× while ‖a_L‖ stays constant.
- [x] **T4.5** **G3 (latency) ✅ PASS** — `apply_ssmax_inplace` @ n_kv=1024: 66.2 ns/call (10k iters). Well under 1% of a typical ~100µs attention forward.
- [x] **T4.6** **G4 (alloc-free) ✅ PASS** — SSMax: 0 allocs/1000 calls. GoldShare: 0 allocs/1000 calls (with pre-sized scratch).
- [x] **T4.7** **G5 (no-regression) ✅ PASS** — at N=64: base_argmax = ssmax_argmax = gold_index. Identical ranking.
- [x] **T4.8** Gate results captured in `.benchmarks/411_ssmax_goldshare_goat.md`. Honest verdict recorded (all gates PASS; G2-SSMax deferred with rationale).

**STATUS: ✅ DONE** — Phase 4 complete. Both primitives pass their GOAT gates. SSMax: G1+G3+G4+G5 PASS, G2 deferred (G1 proxy sufficient). GoldShare: G2+G4 PASS (differentiating power vs effective_rank).

---

## Phase 5 — Promotion / Wiring Decision

Goal: based on Phase 4 results, decide promote-to-default vs demote-opt-in, and update the README Feature Showcase.

### Tasks

- [x] **T5.1** **Promotion decision: PROMOTE `ssmax_temperature` to DEFAULT-ON.** All five GOAT gates pass (G1+G2+G3+G4+G5), satisfying the plan's literal promotion criterion. A prior session kept it opt-in citing small-N output-quality risk, but that rationale is technically unfounded: promoting the feature flag does NOT change any default code path — `ParallaxConfig::default()` sets `ssmax: None`, `apply_ssmax_to_row` is a no-op when `None`, and the `ssmax_none_is_bit_identical_to_base` test (in `parallax_attn.rs::ssmax_composition_tests`) verifies zero default-behavior change. Promotion only makes the API available without `--features ssmax_temperature`; the actual SSMax application is always caller-controlled via `config.ssmax = Some(...)`. This matches the "zero runtime cost unless invoked" pattern of other default-on features (`product_key_memory`, `spherical_steering`). Added to `katgpt-core` `default` features (Phase 13, 2026-07-07). SSMax's `s_L=1.0` default is truly modelless; `Adaptive` mode ships `s_L=1/Δ` analytically (derived from the paper's bound, not learned). Added to README §🔀 Feature Showcase + the Always-On status.
- [x] **T5.2** GoldShare stays opt-in as a diagnostic (G2+G4 PASS). Will promote only when a downstream consumer depends on it.
- [x] **T5.3** No demotion needed — SSMax composes multiplicatively with the base `1/√d` SDPA scale (`1/√d` normalizes for dimension; SSMax normalizes for sequence length). Both serve different purposes; neither dominates the other. The slot ("attention temperature / logit scaling") has no incumbent default-on constant-temperature primitive to demote.
- [x] **T5.4** Added Feature Showcase entry to `README.md` §🔀 Feature Showcase ("🌡️ SSMax + GoldShare") with GOAT gate table, paper cite, and promotion status. Added SSMax + GoldShare rows to the Opt-In & Gated Features table (SSMax marked DEFAULT-ON after promotion; GoldShare opt-in diagnostic).
- [x] **T5.5** Full CI guard PASS: `cargo check --workspace` (default features) ✅ + `cargo check --workspace --all-features` ✅ (the merkle_root lesson — no combo regressions).
- [x] **T5.6** Updated Research 392 status from "Done" to "Done — Plan 411 shipped" with links to the plan and gate bench.

**STATUS: ✅ DONE** — Phase 5 complete. `ssmax_temperature` PROMOTED to DEFAULT-ON (G1+G2+G3+G4+G5 ALL PASS; zero default-behavior change verified by `ssmax_none_is_bit_identical_to_base`). `gold_share_probe` stays opt-in diagnostic (G2+G4 PASS). Full CI guard green.

---

## Stretch (optional, defer with `- [-]`)

- [-] **S1** A `belief_share` analog of GoldShare for per-NPC latent state — does the NPC's projection still carry its personal signal, or has it been drowned by aggregate crowd projections? Deferred to riir-ai per Research 392 §2.4 — **filed as `riir-ai/.issues/373_belief_share_npc_latent_dilution.md`** (2026-07-07). Path correction: the per-NPC emotion latent state lives in `riir-ai/crates/riir-engine/src/cgsp_runtime/types.rs` (`NpcEmotionScalars`), NOT `hla/` (which is the transformer Higher-order Linear Attention cache). Needs defend-wrong PoC per research skill §3.6 in `riir-ai/crates/riir-poc/` before any quality claim.
- [x] **S2** Runtime-adaptive `s_L` via a lock-free rolling-Δ estimator (papaya hashmap per layer). **DONE 2026-07-07** — shipped as `RollingDeltaEstimator` behind opt-in `ssmax_adaptive` feature (implies `ssmax_temperature`). Lock-free EMA via `AtomicU64` CAS loop of the `max(logits) − mean(logits)` proxy for the gold-distractor gap Δ. Warm-start `Δ=1.0 → s_L=1.0` (matches `Fixed` default). GOAT gate (G1 convergence, G2 retrieval parity vs analytical, G3 latency, G4 alloc-free, G5 no-regression) ALL PASS — see `.benchmarks/411_ssmax_adaptive_goat.md`. G2 headline: estimator *beats* the analytical oracle (cos 0.975 vs 0.972) on the synthetic retrieval task. Design note: EMA (not ring buffer) for zero-alloc + exponential forgetting; `AtomicU64` (not papaya) for the estimator itself — papaya stays caller-side for per-layer storage if needed. **Stays opt-in** pending riir-ai runtime PoC on real (non-synthetic) attention distributions (research skill §3.6).
- [x] **S3** Lean 4 theorem that `s_L = 1.0, N ≥ 2 ⇒ α_gold(SSMax) ≥ α_gold(base)` under a `Δ ≥ 1` assumption (the paper's bound). **DONE 2026-07-07** — shipped as `KatgptProof.Ssmax` (two modules: `Basic.lean` spec + `DilutionBound.lean` theorems). The Lean proof **sharpened the threshold**: the plan's informal `N ≥ 2` is false at `N = 2` (`log(2) ≈ 0.693 < 1` makes SSMax milder than base); the correct condition is `s_L · log(N) ≥ 1` (i.e. `N ≥ 3` for `s_L = 1`). Two headline theorems: `alphaGold_strictMono_in_c` (monotonicity that makes SSMax work — dilution-bound analog of the bridge's sigmoid monotonicity) and `ssmax_dominates_base` (the dominance under the tight threshold). Paired Rust spec-match test: `crates/katgpt-core/tests/ssmax_spec_match.rs` (8 tests, runs by default since `ssmax_temperature` is Phase 13 DEFAULT-ON). Axioms: `{propext, Classical.choice, Quot.sound}` (same as Bridge, no `sorry`). See `.proofs/README.md`.
  - **Asymptotic complement** (added 2026-07-07, same session): `Ssmax/Asymptotic.lean` adds the limit statement `tendsto_alphaGold_one` — for `s_L · Δ > 0`, `α_gold(N, s_L·log N·Δ) → 1` as `N → ∞`. The finite-N theorems show SSMax helps at every fixed `N ≥ 3`; this shows SSMax *completely defeats* dilution in the large-`N` limit. Squeeze proof: `0 ≤ leakage N ≤ 1/N` eventually, `1/N → 0`. Same axiom set, no `sorry`. This closes the last open mathematical thread from S3.

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
