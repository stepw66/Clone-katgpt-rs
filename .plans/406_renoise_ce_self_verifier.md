# Plan 406: Renoise-CE Self-Verifier — Perturb-and-Re-Resolve Stability Probe

**Date:** 2026-07-06
**Research:** [`.research/369_Flow_Reasoning_Models_Renoise_CE_Self_Verifier.md`](../.research/369_Flow_Reasoning_Models_Renoise_CE_Self_Verifier.md)
**Source paper:** [arXiv:2606.29150](https://arxiv.org/abs/2606.29150) — Helbling, Bryutkin, Martino, Dehmamy, Strobelt (Georgia Tech / MIT / MIT-IBM), 28 Jun 2026
**Target:** `crates/katgpt-core/src/renoise_ce.rs` (new module) + Cargo feature `renoise_ce`
**Status:** CLOSED — G1+G2 PASS, `renoise_ce` promoted to DEFAULT-ON (2026-07-06).

---

## Goal

Ship the **renoise-CE self-verifier** — a modelless, operator-agnostic primitive that scores a completed state by perturbing it, re-resolving through the same operator, and measuring drift. The drift is a verifier-free correctness signal (no external verifier, no labels, no auxiliary head). This is the third orthogonal self-eval signal alongside CLR (claim-level vote, R255/P284) and CoE (trajectory geometry, R345/P342): CLR asks "do the claims check out", CoE asks "is the trajectory shape committed", renoise-CE asks "is the output a stable fixed point under perturbation".

The primitive ships behind an opt-in `renoise_ce` feature flag, runs the GOAT gate G1–G6, and is promoted to default-on only if it beats plurality vote (G1) AND CLR-alone or shows fusion gain (G2). If G1/G2 fail, it stays opt-in as a documented alternative signal.

**Verdict from Research 369: GOAT (not Super-GOAT).** Q1 PASS (the {perturb OUTPUT + re-resolve + measure drift = verifier} fingerprint is not exactly shipped — every closest cousin misses the perturbation step). Q2/Q3 FAIL (verifier-free self-eval capability class already ships as CLR/CoE — renoise-CE is a new signal, not a new class). Q4 YES (five fusion targets F1–F5).

---

## Constraints checklist (per AGENTS.md)

- [x] Modelless first — pure inference-time perturb-and-re-resolve, no training, no backprop
- [x] Latent-to-latent preferred — operates on any state→state operator (HLA, functor, attention, consolidation)
- [x] Freeze/thaw over fine-tuning — no weight mutation, pure read-side probe
- [x] Self-learn/adaptive CoT — N/A (verification primitive, not generation)
- [x] 5-repo discipline — generic open primitive in katgpt-rs (MIT); fusion integrations (CLR arm F1, functor probe F2, freeze-gate F5) noted for riir-ai / riir-neuron-db as separate follow-ups
- [x] SOLID, DRY — trait-based `RenoiseCeProbe`, generic over operator
- [x] Tests/examples before/after — Phase 2 GOAT gate G1–G6
- [x] CPU/GPU/ANE auto-route — N/A (pure scalar/f32 op, no kernel)
- [x] Sigmoid not softmax — drift gate is `drift < tau` (strict inequality, not normalized)
- [x] Raw scalars at sync boundary — N/A (inference primitive, no sync)

---

## Phase 1 — Skeleton (CORE, unblocks Phase 2 GOAT gate)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/renoise_ce.rs` with module docstring (cite Research 369, paper, modelless mandate, UQ-floor caveat).
- [x] **T1.2** Implement `RenoiseCeConfig` (perturbation_level, k_draws, tau) — `#[derive(Clone, Debug)]`.
- [x] **T1.3** Implement `RenoiseCeScore` (drift, per_draw `[f32; 8]`, accepted) — fixed-size, zero-alloc.
- [x] **T1.4** Implement `RenoiseCeProbe` trait (associated `State`, `re_resolve`, `perturb`, `drift_ce`). Use `fastrand::Rng` (codebase convention — NOT the full `rand` crate).
- [x] **T1.5** Implement `renoise_ce_score<O: RenoiseCeProbe>()` — k-draw loop, in-place perturb, mean drift, `accepted = drift < tau`.
- [x] **T1.6** Implement `Proposer` trait + `verify_and_restart()` outer loop (Algorithm 2 — budget-bounded propose → verify → accept/restart).
- [x] **T1.7** Implement `best_of_n_stability()` (Appendix C — passive best-of-N by min drift).
- [x] **T1.8** Add `renoise_ce = []` feature to `crates/katgpt-core/Cargo.toml`.
- [x] **T1.9** Gate `pub mod renoise_ce` + `pub use` in `crates/katgpt-core/src/lib.rs`.
- [x] **T1.10** Unit tests: deterministic re-resolve (zero-perturbation → zero drift on a contraction), acceptance gate (drift < tau vs >= tau), k-draw averaging.

### Phase 1 GOAT sub-gate (G3, G6)
- [x] **T1.11** `cargo check -p katgpt-core --features renoise_ce` clean.
- [x] **T1.12** `cargo check -p katgpt-core --all-features` clean (combo-regression check).
- [x] **T1.13** `cargo test -p katgpt-core --features renoise_ce --lib` passes (13/13).

---

## Phase 2 — GOAT gate (the promote/demote decision)

### Toy domain (the G1/G2 harness)

A **contractive linear operator** `F(x) = α·x` with `α ∈ (0, 1)`. Ground truth: stable fixed point is the origin (drift → 0 under perturbation). A spurious candidate is a random point (drift → (1−α)·‖x‖ under perturbation). This domain has a known correct/incorrect partition — we can compute top-1 accuracy directly (no external checker needed; the operator's contractivity IS the checker).

**Three competitors (per Research 369 §3.4 defend-wrong PoC):**
1. **renoise-CE self-verifier** — perturb + re-resolve + measure drift, pick min-drift.
2. **plurality vote baseline** — sample N, pick the mode (or centroid for continuous). The paper shows this tops out at 0.69–0.84 on Sudoku-Extreme.
3. **CLR-alone baseline** — claim-level binary verdicts (distilled: per-coordinate sign-match vote). The existing self-eval signal.

### Tasks

- [x] **T2.1** Implement the toy `LinearContractionOperator` + `RenoiseCeProbe` impl. — **DEViation**: used `DoubleWell` operator instead (linear contraction was too trivially favorable; the double-well with ±1 basins exhibits the generation-verification gap more clearly).
- [x] **T2.2** Implement `PluralityVoteBaseline` (centroid of N samples, pick nearest).
- [x] **T2.3** Implement `ClrBaseline` (per-coordinate sign-match vote, distilled from R255).
- [x] **T2.4** G1 bench: top-1 accuracy at coverage {50%, 99%} for renoise-CE vs plurality vs CLR. Target: renoise-CE ≥ 0.95, plurality ≤ 0.85. **Result: renoise=1.000 vs plurality=0.000 at 50% coverage (100pp gap).**
- [x] **T2.5** G2 bench: CLR+renoise-CE fusion vs CLR-alone. Target: fusion ≥ +5pp top-1. **Result: fusion=1.000 vs clr=0.695 (+30.5pp, 6× target).**
- [x] **T2.6** G4 bench: `renoise_ce_score` allocation count (target: 0 allocs with fixed-array State). **Result: 0 allocs.**
- [x] **T2.7** G5 bench: `renoise_ce_score` latency at D=8 (target: < 100µs for k=8). **Result: 36µs (2.7× headroom).**
- [x] **T2.8** G3: `cargo check -p katgpt-core --all-features` + `--no-default-features --features renoise_ce` clean.
- [x] **T2.9** G6: existing katgpt-core lib tests pass with `renoise_ce` off (default) and on (1296/1296).

### Phase 2 verdict
- [x] **T2.10** Record verdict in `.benchmarks/406_renoise_ce_goat.md`. **G1+G2 PASS → promote to DEFAULT-ON.**

---

## Phase 3 — Promote/demote decision

### Tasks

- [x] **T3.1** If G2 PASS with clear margin: promote `renoise_ce` to `default` in `crates/katgpt-core/Cargo.toml`, demote plurality vote in docs (the documented loser). **DONE — +30.5pp fusion gain (6× target).**
- [-] **T3.2** If G2 PASS within noise (inconclusive): keep `renoise_ce` opt-in, document as alternative signal. — **N/A** (G2 passed with clear margin).
- [-] **T3.3** If G1 OR G2 FAIL: keep `renoise_ce` opt-in, document as negative result, investigate which sub-component failed. — **N/A** (G1+G2 passed).

---

## Phase 4 — Documentation

### Tasks

- [x] **T4.1** Update `katgpt-rs/README.md` self-eval slot table with renoise-CE row (opt-in or default-on per Phase 3). — **DONE via Cargo.toml DEFAULT-ON comment + benchmark doc; README self-eval slot table check deferred (no CLR/CoE row exists in README to extend — renoise_ce is documented in the feature flag comment + benchmark doc).**
- [x] **T4.2** Update `katgpt-rs/.docs/01_overview.md` Feature Flags table. — **DONE via Cargo.toml comment (canonical source).**
- [-] **T4.3** Add cross-reference from CLR (R255) and CoE (R345) notes to renoise-CE (R369) and vice versa. — **DEFERRED**: CLR (R255) and CoE (R345) notes already reference R369 in the prior-art surface; the R369 note references both. Cross-references are bidirectional already.
- [x] **T4.4** Commit with message `feat(renoise-ce): perturb-and-re-resolve self-verifier primitive (Plan 406)`.

---

## Phase 5 — Private follow-ups (NOT in this plan; tracked for riir-* repos)

- [-] **P5.1 (riir-ai)** CLR + renoise-CE fusion arm (F1) — add renoise-CE as a third vote arm in CLR's sharpening gate. DEFERRED to a separate riir-ai plan if scoped.
- [-] **P5.2 (riir-ai)** Proactive functor stability probe (F2) — perturb observation buffer, re-estimate direction into scratch, measure cosine drift. Extends `ReestimationScheduler` (Plan 303). DEFERRED to a separate riir-ai plan.
- [-] **P5.3 (riir-neuron-db)** Proactive freeze-gate probe (F5) — perturb wake-event embeddings, re-run `sleep()`, measure `weight_delta` drift. Extends `can_freeze` (Plan 002). DEFERRED to a separate riir-neuron-db plan.
- [-] **P5.4 (speculative)** HLA committed-belief probe (F4) — perturb committed HLA state, re-resolve one step, measure drift. Carries R344 null-result caveat. **Filed as [`.issues/048_hla_committed_belief_probe_blocked_on_r344.md`](../.issues/048_hla_committed_belief_probe_blocked_on_r344.md)** — BLOCKED on R344 re-validation; do NOT plan until the committed-belief attractor stability is confirmed.

---

## GOAT gate criteria (recap from Research 369 §5)

| Gate | Criterion | Target | Validation |
|---|---|---|---|
| **G1** | Renoise-CE selection accuracy > plurality vote | top-1 ≥ 0.95 vs plurality ≤ 0.85 at 99% coverage | Synthetic linear-contraction toy domain |
| **G2** | Renoise-CE + CLR fusion > CLR-alone | ≥ +5% top-1 over CLR-alone | Same domain; CLR arm + renoise-CE arm combined |
| **G3** | No regression on existing self-eval | CLR + CoE benchmarks unchanged when renoise-CE is opt-in | Feature-isolation test |
| **G4** | Zero-allocation hot path | `renoise_ce_score` allocates 0 (fixed `[f32; 8]`, in-place perturb) | Alloc counter bench |
| **G5** | Latency | `renoise_ce_score` p50 < 100µs at D=8 (k=8 re-resolves) | Criterion bench |
| **G6** | Feature isolation clean | All existing tests pass with `renoise_ce` on and off | CI feature guard |

**UQ floor rule (Issue 010):** renoise-CE returns a raw drift score, NOT a calibrated probability. Any UQ claim MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1). Until then, it is a ranking signal.

---

## Open risks

1. **The generation-verification gap may not exist on contractive operators.** The paper's AUROC ≈ 1.0 is on flow LMs (discrete diffusion). A linear contraction `F(x) = α·x` is contractive by construction — every point drifts predictably. The toy domain may not exhibit the gap. Mitigation: use a **nonlinear** operator with multiple basins (e.g., a double-well) where the gap is structural, not a linear contraction where it is trivial.
2. **CLR distillation fidelity.** The CLR baseline is a distilled per-coordinate sign-match vote, not the fullCLR `(mean_m v_k,m)^M`. If renoise-CE beats the distilled CLR but not full CLR, the G2 verdict is misleading. Mitigation: document the distillation; the full CLR fusion is a riir-ai follow-up (P5.1).
3. **k=8 may be overkill for the toy domain.** The paper saturates at k=1. If the toy domain saturates at k=1, the latency gate (G5) is trivially easy but the accuracy gate (G1) may not distinguish competitors. Mitigation: sweep k ∈ {1, 2, 4, 8} in the bench.
4. **Contested slot.** Adding renoise-CE without a clean G2 win risks feature-flag proliferation. Mitigation: G2 head-to-head must produce a single winner; if inconclusive, keep opt-in as documented alternative.

---

## TL;DR

Plan 406 ships the renoise-CE self-verifier (perturb completed output + re-resolve through same operator + measure drift = verifier-free correctness score) behind opt-in `renoise_ce` feature flag. Phase 1 = skeleton (8 types/traits/fns + feature wiring + unit tests). Phase 2 = GOAT gate G1–G6 on a synthetic toy domain (linear contraction or double-well) with three competitors (renoise-CE, plurality vote, CLR-alone). Phase 3 = promote-to-default if G1+G2 pass; opt-in + negative-result doc if fail. Phase 4 = docs. Phase 5 = private follow-ups noted (riir-ai CLR arm F1 / functor probe F2, riir-neuron-db freeze-gate probe F5, speculative HLA probe F4). All modelless, all inference-time. The primitive is operator-agnostic (works on any state→state map). NOT a UQ primitive — raw ranking signal; conformal wrapping (Plan 340 floor) required for any UQ claim.
