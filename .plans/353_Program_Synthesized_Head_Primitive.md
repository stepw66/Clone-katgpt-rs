# Plan 353 (REVISED): HeadSubstitutionGate — IoU+FaiithfulnessProbe Wrapper for FuncAttn/Percepta

**Date:** 2026-06-30 (revised same day — see "Revision" below)
**Research:** [katgpt-rs/.research/353_Program_Synthesized_Attention_Head_Surrogates.md](../.research/353_Program_Synthesized_Attention_Head_Surrogates.md)
**Source paper:** [arXiv:2606.19317](https://arxiv.org/abs/2606.19317) — Hayes, Li, Andreas. *Explaining Attention with Program Synthesis*. MIT CSAIL / NJIT, 30 Jun 2026.
**Target:** `katgpt-rs/crates/katgpt-core/src/functional_substitution/` (new module) + Cargo feature `functional_substitution_gate`
**Status:** Phases 1–4 complete (2026-07-01). T3.4 (real-head G2) deferred to riir-ai. Gain-tier — stays opt-in.

---

## Revision (2026-06-30, same day as initial)

**Initial plan** proposed shipping a new `ProgramSynthesizedHead` primitive + `Box<dyn SynthesizedAttentionFn>` trait. **User-prompted re-review** ("sound like percepta? and a bit of functional attention?") identified that:

1. **FuncAttn** (`katgpt-core/src/funcattn.rs`, R257 / Plan 286) already ships the `tokens → attention via external operator` primitive shape. The proposed `SynthesizedAttentionFn` trait is structurally `dyn FuncAttnKernel` — redundant.
2. **Percepta** (`katgpt-percepta` crate, R031 / R032 / Plan 064) already ships the programs-as-attention paradigm.

The initial plan was revised: **drop `ProgramSynthesizedHead`**, ship only `HeadSubstitutionGate` as a small wrapper around FuncAttn's existing trait surface. Verdict dropped GOAT → Gain. See Research 353 §3.3 for the full revision log.

---

## Goal

Ship a **small gate wrapper** that decides when to substitute a real attention head with a FuncAttn-style surrogate (any callable conforming to FuncAttn's existing trait surface), validated by the IoU cheap-proxy → FaithfulnessProbe expensive-validation cadence pattern (per Plan 287 SinkAware).

This is **not** a new primitive — FuncAttn is the primitive. This is the **control loop** that gates when the primitive fires, using the paper's empirical finding that IoU `r > 0.9` correlates with substitution cost (paper §3 Fig 5b).

**Modelless discipline:** zero training, zero backprop. Surrogate arrives as a FuncAttn-compatible callable. The gate is a pure decision function over cached measurements.

**GOAT gate (Gain-tier):** feature ships opt-in (`functional_substitution_gate`, default-off) and must demonstrate (G1) IoU computation correctness, (G2) IoU→substitution-cost correlation on a synthetic harness reproducing paper's `r > 0.9`, (G3) per-call latency overhead ≤ 5% on the non-substituted path, (G4) zero allocations on hot path. **No promotion to default-on is anticipated** — Gain-tier primitives stay opt-in unless a fusion upgrades them.

---

## Prior-art surface (what already ships — must not duplicate)

| Mechanism | Where | What's missing |
|---|---|---|
| **FuncAttn** (R257, Plan 286) — surrogate representation | `katgpt-core/src/funcattn.rs` | No substitution gate — FuncAttn computes attention, doesn't decide when to use itself vs a real head |
| **Percepta** (R031/032, Plan 064) — programs-as-attention | `katgpt-percepta` crate | Compile-time only; no runtime substitution decision |
| `FaithfulnessProbe` causal intervention (R244, Plan 278) | `katgpt-core/src/faithfulness/probe.rs` | Detects unfaithfulness; doesn't prescribe a surrogate or gate substitution |
| `SmearClassifier` hallucinated-feature detector (R277, Plan 298) | `katgpt-core/src/faithfulness/smear.rs` | Detects smear; doesn't gate substitution |

**The novel piece this plan ships:** `HeadSubstitutionGate` — a small wrapper that decides when FuncAttn should fire as a substitute for a real head, using the paper's IoU `r > 0.9` finding as the cheap-proxy gate and FaithfulnessProbe as the expensive-validation cadence.

---

## Phase 1 — Skeleton (CORE, gate-only)

### Tasks

- [x] **T1.1** Create module directory `katgpt-rs/crates/katgpt-core/src/functional_substitution/` with `mod.rs`, `gate.rs`, `iou.rs`.
- [x] **T1.2** Add feature flag `functional_substitution_gate` to `katgpt-rs/crates/katgpt-core/Cargo.toml` (default-off). The feature depends on `funcattn` (for the surrogate trait) and `faithfulness_probe` (for the validation primitive). Wire into `katgpt-core/src/lib.rs` behind `#[cfg(feature = "functional_substitution_gate")]`.
- [x] **T1.3** Define `iou` function in `iou.rs`: `iou(a: &[f32], b: &[f32]) -> f32`. Formula per paper eq. 3: `Σ min(a,b) / Σ max(a,b)`. SIMD-friendly chunked loop, no allocations. Unit-tested against hand-computed cases (identity → 1.0, disjoint → 0.0, half-overlap → 0.5).
- [x] **T1.4** Define `HeadSubstitutionGate` struct in `gate.rs`:
  ```rust
  /// Gate that decides whether to substitute a real head with a FuncAttn-style
  /// surrogate during a forward pass. Combines the paper's IoU gate (cheap
  /// proxy) with FaithfulnessProbe (expensive validation, cached at audit
  /// cadence per Plan 287 SinkAware pattern).
  pub struct HeadSubstitutionGate {
      /// IoU threshold for attempting substitution (paper default: ~0.4).
      tau_iou: f32,
      /// Behavioral-tolerance threshold for accepting substitution
      /// (FaithfulnessProbe-measured; paper's perplexity delta ≤ ~16%).
      tau_behavior: f32,
      /// Cached FaithfulnessProbe results per head — re-measured at audit
      /// cadence, not per-token.
      cached_faithfulness: Vec<FaithfulnessProfile>,
  }
  
  impl HeadSubstitutionGate {
      /// Hot-path decision: should head `h` be replaced by its surrogate on
      /// this forward pass? Pure decision over cached measurements — no I/O,
      /// no allocation. The actual surrogate callable is supplied separately
      /// by the caller (typically a `FuncAttn` instance).
      pub fn should_substitute(&self, h: usize, head_iou: f32) -> bool {
          if head_iou < self.tau_iou {
              return false;
          }
          let f = &self.cached_faithfulness[h];
          f.behavior_delta_when_replaced <= self.tau_behavior
      }
  }
  ```
  Note: the gate does **not** hold the surrogate itself — the caller owns the FuncAttn instance. This keeps the gate pure and avoids the redundant primitive that was deleted in revision.
- [x] **T1.5** Re-export public API from `mod.rs` and gate behind the feature flag.
- [x] **T1.6** Verify with `cargo check -p katgpt-core --features functional_substitution_gate` (use `CARGO_TARGET_DIR=/tmp/katgpt_353` per AGENTS.md rule).

**Phase 1 exit criterion:** the module compiles standalone, `iou` is correct on hand-computed cases, `HeadSubstitutionGate::should_substitute` is instantiable in a unit test.

---

## Phase 2 — GOAT-style Gate (G1 + G3 + G4)

### Tasks

- [x] **T2.1 (G1 — correctness)** Write unit tests in `tests/functional_substitution_g1.rs`:
  - Identity surrogate (IoU = 1.0, faithfulness delta = 0) → gate accepts.
  - Disjoint surrogate (IoU = 0.0) → gate rejects (regardless of faithfulness).
  - Partial-overlap surrogate at known IoU (e.g., 0.5) → gate accepts iff `tau_iou ≤ 0.5 AND faithfulness ≤ tau_behavior`.
  - High IoU but high behavior delta → gate rejects (faithfulness veto).
- [x] **T2.2 (G3 — hot-path latency)** Benchmark `HeadSubstitutionGate::should_substitute` against a baseline that always returns `false`. Target: ≤ 5% overhead. Use `criterion` bench at `benches/functional_substitution_g3.rs`. Head counts: 4, 16, 144.
- [x] **T2.3 (G4 — zero-alloc)** Add `#[inline]` to `should_substitute`. Verify the gate itself allocates nothing on the hot path (no `Vec` growth, no `Box`).
- [x] **T2.4** Run full crate test suite to confirm no regressions: `cargo test -p katgpt-core --features functional_substitution_gate --lib`.

**Phase 2 exit criterion:** G1 + G3 + G4 green. Feature remains opt-in.

---

## Phase 3 — G2 (IoU → substitution-cost correlation, synthetic)

The paper's strongest empirical claim is that IoU is a valid *cheap proxy* for *expensive* causal substitution cost. G2 reproduces this on a synthetic attention head harness.

### Tasks

- [x] **T3.1** Build a synthetic harness in `tests/functional_substitution_g2.rs`:
  - Generate a synthetic "real" attention matrix with a known structure (e.g., first-token + lower-diagonal per paper Fig 4b GPT-2 categories).
  - Generate a family of surrogates with controlled IoU (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) by blending the real matrix with noise.
  - For each surrogate: measure (a) IoU against real, (b) behavioral delta — KL divergence between softmax(real_tokens) and softmax(surrogate_tokens) on a downstream "task" (a fixed linear projection to a scalar "perplexity proxy").
- [x] **T3.2** Compute Spearman correlation between IoU and behavioral delta across the surrogate family. Target: `ρ ≤ -0.9` (negative because high IoU → low delta). This reproduces the paper's `r > 0.9` finding on the synthetic harness.
- [x] **T3.3** Document the synthetic harness limitations in the test file header: this is *not* a real attention head, the correlation is on controlled-noise surrogates, and the real-head validation requires a forward-pass integration that is out of scope for katgpt-rs (it belongs in riir-ai or in a downstream consumer).

**Phase 3 exit criterion:** G2 green on synthetic harness. Real-head G2 is **deferred** — it requires a real transformer forward pass, which lives in riir-ai not katgpt-rs.

- [-] **T3.4 (DEFERRED — real-head G2)** Run the G2 harness on a real attention head from a small transformer (GPT-2 small). Requires riir-ai integration. Out of scope for this plan.

---

## Phase 4 — Documentation

### Tasks

- [x] **T4.1** Add module-level rustdoc to `mod.rs` explaining: source paper, why this is a gate (not a primitive — see revision note), the IoU gate rationale (paper §3 Fig 5b `r > 0.9`), the cadence pattern (cached faithfulness, per Plan 287 lesson). Cross-link to `funcattn.rs` as the primitive being gated.
- [x] **T4.2** Add an entry to `katgpt-rs/.docs/01_overview.md` Feature Flags table for `functional_substitution_gate` with status "opt-in: G1+G3+G4 green, G2 synthetic green, G2 real-head deferred. Gate wrapper around FuncAttn; not a new primitive."
- [x] **T4.3** Do NOT add a `katgpt-rs/README.md` Feature Showcase entry — Gain-tier wrappers don't get showcase entries. Cross-link from the existing FuncAttn showcase entry instead (when one is added — currently R257 has no README showcase entry).
- [x] **T4.4** Do NOT create a `riir-ai/.research/` guide. Per the revised verdict (Gain, not GOAT/Super-GOAT), no private guide is created.

**Phase 4 exit criterion:** docs updated, no showcase entry, no Super-GOAT guide created (correct per revised verdict).

---

## Out of Scope

- **The redundant `ProgramSynthesizedHead` primitive.** Dropped in revision — use `FuncAttn` directly.
- **Real transformer forward-pass integration.** Belongs in riir-ai. This plan ships the gate; the integration is a follow-up.
- **LM-driven program synthesis.** The paper uses Claude Sonnet 4 offline. That step is out of scope for the runtime crate — FuncAttn-compatible callables arrive pre-constructed.
- **NeuronShard per-dimension audit (Fusion B from research note §2.4).** Latent reframing candidate; needs its own research note.
- **Promotion to default-on.** Gain-tier primitives stay opt-in unless a fusion upgrades them.

## Risks

- **The gate may be too thin to justify a feature flag.** If `HeadSubstitutionGate::should_substitute` ends up as 4 lines of code, consider folding it directly into a FuncAttn consumer instead of a separate module. Revisit after Phase 1.
- **G2 synthetic-only is a weak gate.** Honest disclosure in T3.3 is mandatory; do not claim real-head validation when it only passed synthetic.
- **Naming collision risk.** "functional_substitution" might collide with future naming. Alternative: `funcattn_substitution_gate`. Decide in T1.1.

## Cross-references

- **Research note:** `katgpt-rs/.research/353_Program_Synthesized_Attention_Head_Surrogates.md` (esp. §3.3 revision log)
- **Primitive being gated:** `katgpt-core/src/funcattn.rs` (R257, Plan 286)
- **Validation primitive:** `katgpt-core/src/faithfulness/probe.rs` (R244, Plan 278)
- **Cadence pattern source:** Plan 287 (SinkAware cached-cadence)
- **Cousin research:** 257 (FuncAttn), 031/032 (Percepta), 244 (FaithfulnessProbe), 229 (ProgramAsWeights), 277 (SmearClassifier), 178 (Rosetta Neurons), 302 (FAME — latent reframing target)

## TL;DR

Ship `HeadSubstitutionGate` (revised — was `ProgramSynthesizedHead`, dropped after user-prompted re-review identified FuncAttn as the existing primitive surface). Small gate wrapper that decides when a FuncAttn-style surrogate should substitute for a real attention head, using IoU cheap-proxy (paper §3 `r > 0.9`) → FaithfulnessProbe expensive-validation, Plan 287 cadence pattern. Three phases: skeleton → G1/G3/G4 (correctness, latency, zero-alloc) → G2 synthetic IoU→delta correlation. Gain-tier; stays opt-in. The paper's empirical findings (25-40% heads programmable, library-search beats per-head synthesis) are recorded as facts that update how aggressively we apply FuncAttn/Percepta, not as motivation for a new primitive.
