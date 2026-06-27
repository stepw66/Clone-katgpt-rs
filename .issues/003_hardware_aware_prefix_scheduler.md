# Issue 003: Hardware-Aware Prefix Scheduler — Multi-Request Verification Budget Allocator

**Date:** 2026-06-27
**Research:** [katgpt-rs/.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md](../.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md)
**Source paper:** [DSpark (DeepSeek-AI, 2026)](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) §3.2.2, Algorithm 1, Appendix A
**Target:** `katgpt-rs/src/speculative/prefix_scheduler.rs` (new module) + Cargo feature `hardware_aware_scheduler`
**Status:** Open — optimization task (Gain-tier, behind feature flag, GOAT-gated before promotion)

---

## Problem

katgpt-rs has per-request verification budget selectors (`caddtree_budget.rs`, `budget.rs`) but no **multi-request global** verification budget allocator. When multiple spec-decode requests share a target model forward pass (batch serving, or crowd-scale NPC cognition in riir-ai), static per-request block lengths waste target compute on low-survival suffix tokens while starving high-survival tokens in other requests. DSpark §3.2.2 formulates this as a global throughput maximization `Θ = τ · SPS(B)` and solves it greedily with a non-anticipating early-stop that preserves the lossless distribution guarantee (Appendix A correctness proof).

## Goal

Ship a generic, modelless, zero-allocation `HardwareAwarePrefixScheduler` behind `hardware_aware_scheduler` (default-off). Given:
- R active requests, each with per-position survival probabilities `a_{r,j} = Π_{i≤j} c_{r,i}` (monotone non-increasing in j)
- A profiled engine cost curve `SPS(B)` (steps-per-second vs total verification batch size B)

Produce per-request prefix lengths `ℓ*_1..ℓ*_R` that maximize `Θ = τ · SPS(B)` via:
1. Globally sort all `(r, j)` candidates descending by `a_{r,j}`
2. Greedily admit candidates; update `B += 1`, `τ += a_{r,j}`; O(1) lookup `SPS(B)`
3. **Early-stop when `Θ ≤ Θ_best`** — this is the non-anticipating property required for lossless speculative decoding (DSpark Appendix A). Without it, retrospective global search leaks future token info into the current-token admission decision, introducing selection bias that breaks distribution preservation.

## Scope (what this issue IS and IS NOT)

**IS:**
- The generic scheduler primitive (sort + greedy + cost-curve lookup + non-anticipating early-stop).
- A profiled `SPS(B)` curve abstraction (load once at init, store as `Box<[f32]>` or interpolation LUT).
- A single-request-isolated correctness test proving the scheduler's output matches `LeviathanVerifier` exactly when R=1 (the early-stop must reduce to "verify the full block" or "skip" depending on SPS shape, never bias the accepted distribution).

**IS NOT:**
- A multi-request batch execution engine (katgpt-rs is single-request by default; multi-request execution is the caller's concern — the scheduler just outputs prefix lengths).
- The confidence head that produces `c_k` (reuse `AcceptanceForecast`, Bebop Plan 243, as the producer).
- The semi-autoregressive drafter (training → riir-train).
- Sequential Temperature Scaling (small calibration refinement; separate issue if pursued).

## Tasks

- [ ] **T1** `katgpt-rs/src/speculative/prefix_scheduler.rs` — `HardwareAwarePrefixScheduler` struct + `schedule(&self, survival_probs: &[&[f32]]) -> Box<[usize]>`. Reuse `cumprodsum_scalar` for `a_{r,j} = Π c_i` if the caller passes raw `c_k` instead of pre-computed `a_{r,j}`.
- [ ] **T2** `SpsCurve` abstraction — `from_profile(samples: &[(usize, f32)]) -> Self`, `steps_per_second(batch_size: usize) -> f32` (linear interpolation between samples; clamp at ends).
- [ ] **T3** Non-anticipating early-stop — break the greedy loop when `Θ ≤ Θ_best`. Document the Appendix A counterexample in a doc comment.
- [ ] **T4** Feature flag `hardware_aware_scheduler` (default-off), wired into `speculative/mod.rs`.
- [ ] **T5** Correctness test: R=1 with synthetic SPS curve must produce a distribution-identical result to `LeviathanVerifier` (no selection bias). Port the Appendix A counterexample: without early-stop, vocab {A,B}, p_t=(0.7,0.3), p_d=(0.5,0.5) → output must be (0.7,0.3), not (0.85,0.15).
- [ ] **T6** Multi-request throughput test: R=4, synthetic SPS curve (monotone-decreasing with a cliff), verify the scheduler allocates longer prefixes to high-survival requests and shorter to low-survival, and that `Θ` is at least as high as uniform-length allocation.
- [ ] **T7** GOAT gate benchmark (`benches/prefix_scheduler_goat.rs`): multi-request workload, measure `accepted_tokens/sec` and `μs/step` with vs without the scheduler. Gate: ≥5% throughput gain, zero quality regression on the R=1 correctness test.
- [ ] **T8** If T7 passes → promote `hardware_aware_scheduler` to default-on and demote the per-request `caddtree_budget.rs` path if it strictly dominates.

## Risks

- **Single-request default.** katgpt-rs is single-request by default. The scheduler only helps when the caller batches multiple spec-decode requests into one target forward pass. The benchmark (T7) must construct a multi-request workload or the gate is vacuous. If the engine never batches, this primitive has no leverage and stays opt-in indefinitely.
- **SPS(B) curve shape.** DSpark §5.2 notes real hardware has jagged, step-wise SPS curves, not smooth/unimodal. The early-stop assumes unimodality for global optimality. The paper works around this in production via asynchronous 2-step-prior prediction + removing the early-stop (causality maintained by the temporal offset). Our CPU/SIMD/wgpu stack may have different SPS characteristics — re-profile on our hardware before promoting.
- **Non-anticipating early-stop is a correctness theorem, not a heuristic.** Removing it for throughput (as DSpark does in production via async) requires a separate causality argument. Do NOT remove it in katgpt-rs without porting the async-ZOS causality proof.

## Cross-references

- `katgpt-rs/.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md` — distillation
- `katgpt-rs/src/speculative/acceptance_forecast.rs` — `AcceptanceForecast` (producer of `c_k`)
- `katgpt-rs/src/speculative/caddtree_budget.rs` — per-request analog (`expected_accepted_length_at_budget`)
- `katgpt-rs/src/speculative/budget.rs` — per-request adaptive budget (predecessor)
- `katgpt-rs/src/cumprodsum.rs` — `cumprodsum_*` (SIMD `Π c_i`)
- DSpark paper §3.2.2 (Algorithm 1), §5.2 (production async variant), Appendix A (non-anticipating correctness proof)
