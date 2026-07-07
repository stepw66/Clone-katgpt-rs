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
- [x] **T2.3** The 3-way composition (parallax + sink-aware + SSMax) is **implicit** — since SSMax is a field on `ParallaxConfig`, the existing `tiled_attention_parallax_forward_sink_aware` entry point picks it up automatically (it passes `parallax_config` through to `tiled_attention_parallax_forward_retaining`). No separate entry point needed — the composition compiles and works. Deviation from plan T2.3 documented: instead of a new entry point, the field-on-config approach (which the plan T2.1 specified) makes the composition automatic.
- [x] **T2.4** Added `tiled_attention_forward_ssmax` to `crates/katgpt-core/src/attention.rs` behind `#[cfg(all(feature = "tiled_attention", feature = "ssmax_temperature"))]`. Folds `s_L · log(N)` into the softmax scale — the zero-overhead way to apply SSMax to flash-attention (no score matrix materialization). `#[cfg(all(feature = "tiled_attention", feature = "ssmax_temperature"))]`.
- [x] **T2.5** Documented in `ssmax.rs` module doc: SSMax does NOT apply to `funcattn` (Research 261 closed negative: basis-mode structure has no `(n,n)` attention matrix, so dilution is structurally absent). Not wired.
- [x] **T2.6** `cargo check -p katgpt-core --features parallax_attn,ssmax_temperature --lib` passes clean.
- [x] **T2.7** `cargo check -p katgpt-core --features parallax_attn,sink_aware_attn,ssmax_temperature --lib` passes clean (3-way composition). Also verified `tiled_attention,ssmax_temperature` compiles (the SDPA wrapper).

**STATUS: ✅ DONE** — Phase 2 complete. All 12 parallax_attn tests pass with and without ssmax_temperature (default None preserves bit-identical behavior).

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
