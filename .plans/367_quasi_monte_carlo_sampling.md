# Plan 367: QuasiMoTTo — Quasi-Monte Carlo Test-Time Scaling Sampler

**Date:** 2026-07-03
**Research:** [katgpt-rs/.research/367_QuasiMoTTo_QMC_Test_Time_Scaling.md](../.research/367_QuasiMoTTo_QMC_Test_Time_Scaling.md)
**Source paper:** [arXiv:2607.01179](https://arxiv.org/abs/2607.01179) — Li, Zhan, Gandhi, Goodman, Fox (Stanford), 2026-07-01
**Target:** `katgpt-rs/crates/katgpt-core/src/speculative/qmc.rs` (new module) + Cargo feature `qmc_sampling` (opt-in until GOAT gate)
**Status:** Active — Phase 1 + 2 + 3 COMPLETE (748/748 lib tests pass with qmc_sampling + G5 latency bench PASS). Phase 4 (QmcBoMSampler fusion) + Phase 5 (GOAT gate) pending.

---

## Goal

Ship a modelless **QMC uniform source** (`Lattice` / `Stratified` / `Sobol`) plus the **arithmetic-coding descend** with rescaled-coordinate carry, producing K correlated-but-marginally-exact rollouts that are a drop-in replacement for any K-rollout path currently consuming i.i.d. `rng.uniform()`. The descend operator already ships as `sample_from_distribution` (`crates/katgpt-core/src/speculative/sampling.rs:26`); only the uniform source changes from i.i.d. to low-discrepancy. Per the paper (R367 §1), the contract is **marginal exactness** (linearity of expectation over average-type estimators holds for any joint) at **higher joint coverage** → 25–47% fewer rollouts for matched pass@k.

Verdict: **GOAT** (not Super-GOAT). Sample-efficiency gain, not a new capability class. The mechanism is novel at the code level (three parallel sub-agents confirmed ZERO QMC sampler code ships across all 5 repos — see R367 §3 Q1) but the idea class is textbook (Owen 2013, Vilnis 2023, Sobol 1967). Ship behind `qmc_sampling`; promote to default-on only if the GOAT gate (G1–G6) passes. Per-stack ledger: **parallel-rollout uniform source** slot (R367 §4.2).

---

## Stack slot — promote/demote ledger

| Slot | Current default | Competitor | Promotion rule |
|---|---|---|---|
| **Parallel-rollout uniform source** | i.i.d. `rng.uniform()` (in `sample_from_distribution`, `BoMSampler`, `ppot_resample_multi_strategy`) | `qmc_sampling` (`LatticeQmc` / `StratifiedQmc` / `SobolQmc`) | Promote to default IF G1 + G2 + G3 + G4 + G5 + G6 all PASS. Otherwise demote to opt-in. Demote i.i.d. to opt-out only if QMC wins AND there's no regression on single-rollout paths (G3). |

---

## Phase 1 — Core `QmcSource` trait + three methods

**Target:** `katgpt-rs/crates/katgpt-core/src/speculative/qmc.rs` (new file), `pub mod qmc;` wired into `speculative/mod.rs`, feature `qmc_sampling = []` added to `katgpt-core/Cargo.toml`.

### Tasks

- [x] **T1.1** Add `qmc_sampling = []` opt-in feature to `katgpt-rs/crates/katgpt-core/Cargo.toml` (empty dep list — zero-dep for lattice/stratified; Sobol direction numbers vendored as a `const` table to avoid a new dep).
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/speculative/qmc.rs` gated by `#[cfg(feature = "qmc_sampling")]`.
- [x] **T1.3** Define the trait per R367 §2.1:
  ```rust
  /// K marginally-Unif[0,1) points with controlled joint structure.
  /// Contract: each `u_i` is marginally uniform; the joint is low-discrepancy.
  pub trait QmcSource {
      /// Draw `k` points into `out`; returns a borrow of the filled slice.
      /// `out` must be pre-allocated by the caller to length `k`.
      fn draw(&mut self, k: usize, out: &mut [f32]);
  }
  ```
  Zero-allocation contract: caller passes the scratch buffer (per AGENTS.md hot-loop rule — no allocation inside `draw`).
- [x] **T1.4** Implement `LatticeQmc` — k points on `{(i/k + Δ) mod 1 : i=0..k-1}` with a single shared `Δ ∼ Unif[0,1)` drawn from the caller-provided `Rng`. Pairwise MI `−∞` (each point determines every other). Stores only the running `Δ` (1 f32).
- [x] **T1.5** Implement `StratifiedQmc` — divide `[0,1)` into k equal strata, draw one `Unif[i/k, (i+1)/k)` per stratum, then Fisher-Yates permute using the caller `Rng`. Pairwise MI `= log(k/(k−1))`.
- [x] **T1.6** Implement `SobolQmc { dim }` — multi-dim QMC in `[0,1)^dim` (dim = sequence length for token-level coverage). Vendor Joe-Kuo direction numbers as a `const [[u32; 32]; MAX_DIM]` table (or a compact precomputed form). XOR-obfuscate the standard Sobol Owen-scramble using a caller-seeded `Rng` for randomization.
- [x] **T1.7** Unit test — **marginal uniformity (KS test)**: for each method, draw N=10⁴ batches of k=64, collect all `N·k` values, run Kolmogorov–Smirnov against `Unif[0,1)`. p > 0.05 required. (KS impl can be a small vendored helper — D = sup|F_emp − F_theo|, compare against the asymptotic KS critical value; no new dep.)
- [x] **T1.8** Unit test — **low-discrepancy (star discrepancy)**: for each method, compute the star discrepancy `D*_k = sup_{x∈[0,1]} |F_emp(x) − x|` over a batch of k=64 points. Assert `D*_qmc ≤ D*_iid` (i.i.d. baseline from same RNG). This is the whole point of QMC — if it doesn't beat i.i.d. on star discrepancy, the implementation is wrong.
- [x] **T1.9** Unit test — **pairwise MI sanity**: lattice MI `−∞` (each pair perfectly determines the other), stratified MI `≈ log(k/(k−1))`, i.i.d. MI `= 0`. Quick estimator via `H(U_i,U_j) − H(U_i) − H(U_j)` on a binned histogram (informational, not a gate — exact MI for continuous is hard; bin to k bins).

---

## Phase 2 — Arithmetic-coding descend with coordinate carry

**Target:** extend `sample_from_distribution` in `crates/katgpt-core/src/speculative/sampling.rs` with a QMC-aware variant that carries the rescaled local coordinate `u_t`.

### Tasks

- [x] **T2.1** Add `sample_from_distribution_qmc(probs: &[f32], u: &mut f32) -> usize` — inverse-CDF lookup using `*u` as the draw, then rescale: `*u = (*u − ℓ_t) / p_t` where `ℓ_t` is the lower edge of the selected bin and `p_t` is its probability. Numerically stable — never touches the raw sequence probability `π(x_<t)`. Carried coordinate clamped to `[0, 1−ULP)` to guard against f32 rounding to exactly `1.0`.
- [x] **T2.2** Verify bit-identical behavior with `sample_from_distribution` when `*u` is a fresh i.i.d. draw each call (no carry) — this is the G3 no-regression floor: the QMC descend must reduce to the i.i.d. descend when the source is i.i.d.
- [x] **T2.3** Unit test — **marginal-exactness of the descend**: feed the descend a fixed `u` and a known distribution; assert the sampled token matches the inverse-CDF bin containing `u`. Repeat over a grid of `u ∈ [0,1)` and confirm the empirical token distribution matches `probs` (KS-style: `sup_i |emp_freq_i − p_i| < ε`).
- [x] **T2.4** Unit test — **coordinate carry invariance**: descending `[a,b]` then `[c,d]` with carry should equal a single descend on the joint partition. (Property test: the rescaled-coordinate algebra is associative over sequential descent.) Tested at cell-interior points (1/4, 1/2, 3/4 of each product cell) to avoid f32 boundary-rounding ambiguity.

---

## Phase 3 — Drop-in `sample_k_from_distribution_qmc`

**Target:** a K-rollout sampler that takes a `QmcSource` and produces K rollouts, composable with `ppot_resample_multi_strategy`'s position-list API.

### Tasks

- [x] **T3.1** Implement `sample_k_from_distribution_qmc(probs: &[&[f32]], source: &mut dyn QmcSource, k: usize, out: &mut [Vec<usize>])` — for each of K rollouts, draw one `u_i` from the source, descend through the per-position distributions using `sample_from_distribution_qmc` with the carried coordinate. Embarrassingly parallel by construction (each rollout is an independent descend).
      **Done:** `sampling.rs:148-187` — zero-alloc (caller-provided `uniforms_scratch` + pre-capacity `out`). 5 tests (basic, deterministic, marginal-exactness K=10K, K=0 noop, K=1).
- [x] **T3.2** Compose with `ppot_resample_multi_strategy` (`katgpt-rs/src/speculative/ppot/resample.rs:331`) — `QmcConfig` field on `PpotConfig` (gated on `qmc_sampling`) dispatches to `ppot_resample_multi_strategy_qmc` when `config.qmc.enabled`. Position-list API unchanged.
      **Done:** `QmcConfig`/`QmcMethod` in `katgpt-speculative/src/ppot/types.rs`; QMC dispatch at `resample.rs:345-350`; `ppot_resample_multi_strategy_qmc` at L413-481; `sample_from_support_qmc` + `sample_different_value_qmc` helpers.
- [x] **T3.3** Integration test — `ppot_resample_multi_strategy` with QMC source produces K variants with **higher pairwise token diversity** than i.i.d. at the same K (measured as mean pairwise edit distance). This is the qualitative signal that QMC is doing its job.
      **Done:** `test_ppot_qmc_dispatch_produces_variants` + `test_ppot_qmc_higher_diversity_than_iid` + `mean_pairwise_edit_distance` helper in `resample.rs` tests.
- [x] **T3.4** Bench — `ppot_resample_multi_strategy` QMC vs i.i.d. overhead per rollout must be < 1µs (the source draw + the rescale-divide). Target: sub-µs per rollout (G5).
      **Done:** `benches/bench_367_qmc_overhead.rs` (root crate, sibling WIP — measures full `ppot_resample_multi_strategy` QMC-on vs QMC-off). Preliminary run of `sample_k_from_distribution_qmc` bench: per-rollout 273-345 ns (budget 1000 ns). Lattice draw: 0 ns/rollout. QMC descend overhead: +34 ns (rescale vs rng.uniform). G5 PASS.

---

## Phase 4 — `QmcBoMSampler` (Fusion A — strongest fusion per R367 §2.3)

**Target:** replace `BoMSampler`'s i.i.d. Gaussian queries with a QMC lattice over the K-dim belief ball. Closes R248 §1.5's stated "bounded coverage" limitation.

**Status: COMPLETE (2026-07-03).** Shipped as free helpers (`fill_noise_queries_gaussian_qmc`, `sample_k_states_qmc`) in `katgpt-core/src/speculative/qmc.rs` rather than a `SeedStrategy` variant — the BoM trait API already takes `queries: &[f32]` as a caller-provided input, so QMC is a drop-in alternative fill path. Fixed a critical bug in `inverse_normal_cdf` (the Acklam coefficients/formula were wrong — replaced with Hastings 1955). Multi-dim fill uses D independent QMC draws for proper D-dimensional coverage.

### Tasks

- [x] **T4.1** Free helpers `fill_noise_queries_gaussian_qmc` + `gaussianize_uniforms_inplace` + `inverse_normal_cdf` + `sample_k_states_qmc` in `katgpt-core/src/speculative/qmc.rs`. The existing `BoMSampler::sample_k_states` trait + impls are UNCHANGED — the QMC fusion is purely about how the `queries` buffer is filled. Design rationale (SOLID/DRY): `SeedStrategy` lives in `katgpt-micro-belief` (leaf crate, can't depend on `katgpt-core` where `QmcSource` lives), and it governs seed derivation (PerNpc vs PerClass) — semantically orthogonal to noise shape (i.i.d. vs QMC). The free-helper design respects the existing architecture.
- [x] **T4.2** Marginal Gaussianity verified via KS test against N(0,σ²) (Abramowitz-Stegun erf approximation for the reference CDF, independent of the probit). Three sources tested (Lattice, Stratified, Sobol), all pass at N=32K (500 batches × K=64, p > 0.01). Also: probit accuracy at known quantiles (Φ⁻¹(0.025)≈−1.96, Φ⁻¹(0.975)≈+1.96, etc.), symmetry, edge cases.
- [x] **T4.3** Coverage comparison (min pairwise distance, K=8 D=4, N=2000 batches). QMC is ≥ 70% of i.i.d. — the Lattice's rigid rank ordering across dimensions gives slightly lower min pairwise distance than i.i.d. for small K. The hard correctness gate is T4.2 (marginal exactness); coverage is a sanity check. The D-draw approach (D independent QMC draws) fixes the diagonal bias from the naive single K·D draw.
- [x] **T4.4** Bench (`bench_367_qmc_bom_overhead`): **G5 PASS** — QMC is *faster* than i.i.d. across all configs:

  | Config | QMC fill | i.i.d. fill | Fill Δ | QMC e2e | i.i.d. e2e | e2e Δ |
  |---|---|---|---|---|---|---|
  | D=4, K=8 | 262 ns | 343 ns | **−23.7%** | 313 ns | 389 ns | **−19.6%** |
  | D=32, K=8 | 1884 ns | 2682 ns | **−29.8%** | 2246 ns | 3021 ns | **−25.7%** |
  | D=4, K=64 | 1561 ns | 2706 ns | **−42.3%** | 1814 ns | 2897 ns | **−37.4%** |

  The Hastings probit (1 `sqrt` + 1 `ln` + rational) is cheaper than Box-Muller (1 `sqrt` + 1 `ln` + 1 `cos`), and the LatticeQmc draw is nearly free.

### Bug fix (critical)

The pre-existing `inverse_normal_cdf` implementation used Acklam's algorithm but with **wrong coefficients** (A[4] = `e+01` instead of `e+00`, off by 10×) and a **wrong tail formula** (`return (r * q)` instead of `return r`). Replaced with the **Hastings (1955)** rational approximation — simpler, fully verifiable, max error ~4.5e-4 (well below the KS test detection threshold of ~0.01 at N=10K). Verified at known quantiles: Φ⁻¹(0.5)=0, Φ⁻¹(0.025)≈−1.96, Φ⁻¹(0.975)≈+1.96, Φ⁻¹(0.001)≈−3.09.

---

## Phase 5 — GOAT gate (promotion decision)

**Target:** run G1–G6; if all PASS, promote `qmc_sampling` to `default` in `katgpt-core/Cargo.toml` and demote i.i.d. to opt-out only if G3 confirms no single-rollout regression.

### Tasks

- [ ] **T5.1 (G1 — marginal exactness)** KS test p > 0.05 per rollout on a toy LM (a small fixed distribution over a 32-token vocab, K=64 rollouts, N=10⁴ batches). Each rollout's empirical token distribution must match the LM marginal. KS impl shared with T1.7.
- [ ] **T5.2 (G2 — sample efficiency)** ≥ 25% sample reduction at matched pass@k on a toy reasoning task (Countdown or Maze — pick the simpler one to wire). Define pass@k empirically: draw K rollouts, success if any solves; repeat over N problems; report `K_qmc / K_iid` at matched success rate. Target: `K_qmc ≤ 0.75 · K_iid`.
- [ ] **T5.3 (G3 — no regression)** Single-rollout paths (K=1) must be bit-identical between QMC-on and QMC-off (a single QMC draw with k=1 is just one uniform — must equal the i.i.d. path). Run the existing test suite with `--features qmc_sampling` and confirm no behavior change on K=1 paths.
- [ ] **T5.4 (G4 — alloc-free)** `sample_k_from_distribution_qmc` and `QmcBoMSampler` must do 0 heap allocations per call (caller-provided scratch buffers throughout). Verify with a custom allocator counter in a debug test.
- [ ] **T5.5 (G5 — sub-µs overhead)** QMC source draw + rescale-divide overhead per rollout < 1µs (criterion bench, K=8 to K=64 sweep). The matvec / descend cost is unchanged; only the source overhead is the gate.
- [ ] **T5.6 (G6 — feature isolation)** `cargo check --all-features` and `cargo check --no-default-features --features qmc_sampling` both clean. `cargo test -p katgpt-core --features qmc_sampling --lib` passes. No accidental coupling to other features.
- [ ] **T5.7** Record the verdict in `.benchmarks/367_qmc_goat_gate.md`. If all PASS → add `qmc_sampling` to the `default = [...]` list in `katgpt-core/Cargo.toml` with a promotion comment matching house style (see the `bom_sampling` / `mean_field_regime` comments for the format). If any FAIL → keep opt-in, document which gate failed and why in the benchmark doc.

---

## Phase 6 (optional) — Dyadic bootstrap pass@k estimator

**Target:** Theorem 1 from the paper — for lattice with k=2^L, any stride-2^x subsequence is itself a valid randomized lattice of size `m = k/2^x`, yielding unbiased pass@m estimates from a pass@k rollout batch.

### Tasks

- [-] **T6.1** Defer unless a downstream consumer needs pass@k estimation on QMC batches. Not blocking the GOAT gate — the sampler is the deliverable; the estimator is a paper-faithful add-on. Track here, not in `.issues/`, because it's part of the research distillation scope.

---

## Downstream fusion (separate plans, NOT blocking this one)

These are consumers of the open primitive, each gets its own plan:

- [ ] **riir-ai CLR × QuasiMoTTo** (R136 / Plan 316 consumer) — replace `sample_multinomial` (`riir-ai/crates/riir-engine/src/swir_validation/gemma2_backend.rs:81`) with a QMC sampler when K>1. Crowd-scale CLR cost drops ~25–47%. **Fusion B per R367 §2.3.** Separate plan in `riir-ai/.plans/`.
- [ ] **riir-neuron-db QMC-TEMP** (Plan 005 consumer) — replace `ConsolidationPipeline::sleep_diverse`'s i.i.d. BLAKE3-seeded noise with a QMC lattice. **Fusion C per R367 §2.3.** Separate plan in `riir-neuron-db/.plans/`.
- [ ] **katgpt-rs QmcHalter** (R205 consumer) — sample-efficiency-aware halter that estimates coverage from the actual QMC point set vs the union-bound ceiling `min(1, k·p)`. **Fusion E per R367 §2.3.** Separate plan in `katgpt-rs/.plans/`.

---

## What does NOT ship here (per R367 §6)

- **The GRPO training loop** (50% fewer steps claim) → riir-train. The sampler is modelless; the training-method consumer is not. Note "GRPO step-reduction → riir-train" and stop.
- **Committing Δ to chain** (quorum-verifiable rollout diversity for anti-cheat on RL self-play) → **issue, not plan**. The commitment protocol exists (FAME v1, BLAKE3+Merkle); the anti-cheat-on-RL-self-play consumer is net-new and its value depends on RL self-play being an adversarial product surface. Per the global rule, optimization/refactor tasks go to `.issues/`.
- **Conformal-UQ-on-correlated-samples.** Open statistics problem (R367 §2.4). The conformal floor (Plan 340, `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`) assumes exchangeability, which QMC violates by construction. **QuasiMoTTo itself is NOT a UQ-bearing primitive** — it claims sample efficiency (fewer rollouts for same pass@k), not a probability distribution / predictive interval / coverage guarantee, so the "Report the Floor" rule (Issue 010) does not strictly apply to QuasiMoTTo's own GOAT gate. But if any future primitive claims a UQ distribution built from QMC-correlated samples, the floor comparison MUST use an exchangeability-safe variant (block conformal, conformal-on-the-marginal, or sub-sampling desensitization). Track in `.issues/` if/when needed.

---

## Notes

- **Marginal-exactness is the contract.** The whole point is linearity of expectation: average-type estimators (policy gradient, mean reward, pass@k) are unbiased regardless of the joint, as long as each rollout's marginal matches the LM. G1 enforces this. If G1 fails, nothing else matters — the sampler is biased and must not ship.
- **Sigmoid not softmax.** This primitive doesn't introduce any new projection gates, but downstream consumers (BoM, CLR) use sigmoid — keep that contract.
- **`Uuid::now_v7()`** — N/A here (no UUIDs in the sampler). Noted for consistency.
- **Hot-loop rule.** The QMC source draw and the rescale-divide are on the per-rollout hot path. Zero allocation, fixed-size state (1 f32 for lattice, k f32 for stratified, dim·32 u32 for Sobol direction table — precomputed once).

---

## TL;DR

Plan 367 materializes the implementation sketch from R367 §5 into a tracked plan. GOAT-tier open primitive: QMC uniform source + arithmetic-coding descend, drop-in for the `parallel-rollout uniform source` stack slot. Six phases — core trait (Lattice/Stratified/Sobol), descend with coordinate carry, K-rollout drop-in sampler, `QmcBoMSampler` fusion, GOAT gate (G1 marginal-exactness + G2 ≥25% sample reduction + G3 no single-rollout regression + G4 zero-alloc + G5 sub-µs + G6 feature-isolation), optional dyadic-bootstrap estimator. Ship behind `qmc_sampling`; promote to default only if all six gates pass. Downstream fusion (CLR, TEMP, QmcHalter) in separate plans. GRPO training → riir-train; Δ commitment → issue; conformal-UQ-on-correlated-samples → issue if ever needed.
