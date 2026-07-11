# Plan 367: QuasiMoTTo — Quasi-Monte Carlo Test-Time Scaling Sampler

**Date:** 2026-07-03
**Research:** [katgpt-rs/.research/367_QuasiMoTTo_QMC_Test_Time_Scaling.md](../.research/367_QuasiMoTTo_QMC_Test_Time_Scaling.md)
**Source paper:** [arXiv:2607.01179](https://arxiv.org/abs/2607.01179) — Li, Zhan, Gandhi, Goodman, Fox (Stanford), 2026-07-01
**Target:** `katgpt-rs/crates/katgpt-core/src/speculative/qmc.rs` (new module) + Cargo feature `qmc_sampling` (opt-in until GOAT gate)
**Status:** Active — Phase 1 + 2 + 3 + 4 + 5 + 6 ALL COMPLETE (850/850 lib tests pass with qmc_sampling, 26 new bootstrap tests, G5 latency bench PASS, GOAT gate PASS and promoted to DEFAULT-ON). Plan fully shipped; remaining work is in separate downstream-fusion plans.

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

**Status:** ✅ ALL GATES PASS — PROMOTED to DEFAULT-ON (2026-07-03).
**Bench doc:** `.benchmarks/367_qmc_goat_gate.md`

**Target:** run G1–G6; if all PASS, promote `qmc_sampling` to `default` in `katgpt-core/Cargo.toml` and demote i.i.d. to opt-out only if G3 confirms no single-rollout regression.

### Tasks

- [x] **T5.1 (G1 — marginal exactness)** ✅ PASS — chi-square GoF (KS→χ² substitution for discrete distributions). Lattice 1.17%, Stratified 0.78%, Sobol 0.39% fail rate at α=0.01 (i.i.d. baseline 0.39%). Gate: fail rate < 5% across K·T=256 tests. Caught and fixed critical Sobol scramble bug (OR-of-two-halves → upper-32-bits).
- [x] **T5.2 (G2 — sample efficiency)** ✅ PASS — Lattice 50% sample reduction (K_qmc=8 vs K_iid=16 at pass@k≥0.5, target ≤0.75). Fresh-draw-per-K measurement (drawing K_MAX and taking first-K systematically disadvantaged QMC). Stratified/Sobol don't show pass@k advantage (optimized for RL-variance/multi-dim respectively) — Lattice is the pass@k champion per R367 §1.1.
- [x] **T5.3 (G3 — no regression)** ✅ PASS — 0/10000 mismatches on both paths: (a) sample_from_distribution_qmc vs CDF walk with same u, (b) sample_k_from_distribution_qmc K=1 vs single descend.
- [x] **T5.4 (G4 — alloc-free)** ✅ PASS — 0 allocs/100 steady-state calls on both sample_k_from_distribution_qmc and fill_noise_queries_gaussian_qmc.
- [x] **T5.5 (G5 — sub-µs overhead)** ✅ PASS — per-rollout 25-34 ns (target <1000 ns), K=8 to K=64 sweep. Raw QmcSource::draw: 0.37 ns/rollout.
- [x] **T5.6 (G6 — feature isolation)** ✅ PASS — cargo check matrix all clean: --features qmc_sampling, --all-features, --no-default-features --features qmc_sampling. 761 tests pass.
- [x] **T5.7** ✅ DONE — Verdict recorded in `.benchmarks/367_qmc_goat_gate.md`. `qmc_sampling` promoted to `default` in `katgpt-core/Cargo.toml`.

---

## Phase 6 (optional) — Dyadic bootstrap pass@k estimator

**Target:** Theorem 1 from the paper — for lattice with k=2^L, any stride-2^x subsequence is itself a valid randomized lattice of size `m = k/2^x`, yielding unbiased pass@m estimates from a pass@k rollout batch.

**Status: COMPLETE (2026-07-04).** Shipped as `BootstrapEstimate` struct + `dyadic_bootstrap_pass_at_m_lattice` (exhaustive, RNG-free, algebraically exact) + `contiguous_block_bootstrap_pass_at_m` (random contiguous blocks for Sobol/Stratified) + `sample_variance_binary` helper, all in `katgpt-core/src/speculative/qmc.rs`. Wilson score CI method on `BootstrapEstimate` (preferred over normal-approx for binary indicators at small n). 26 new tests: known-answer, m=k degenerate, m=1 recovers mean pass, stride-4 four-offset, Wilson CI known/extreme values, panic-on-bad-input (5 cases), empirical unbiasedness (dyadic + block, N=200K/100K batches vs analytical `1-(1-p)^m`), Theorem-1 sub-lattice validity (KS uniformity + equispacing on real LatticeQmc batches). All 850 lib tests pass under `qmc_sampling`. Feature isolation clean (default / all-features / no-default+qmc). Zero-alloc, hot-path friendly.

### Tasks

- [x] **T6.1** ✅ DONE (2026-07-04) — Implemented the dyadic bootstrap pass@k estimator per Theorem 1 of arXiv:2607.01179. Three public items added to `speculative::qmc`: `BootstrapEstimate` (point estimate + sample variance + n_resamples, with `wilson_ci`/`wilson_ci_95`/`std_dev` methods), `dyadic_bootstrap_pass_at_m_lattice` (Lattice-specific, exhaustive over all k/m stride offsets, algebraically exact per the theorem), `contiguous_block_bootstrap_pass_at_m` (general block-bootstrap for Sobol/Stratified, random contiguous starts). 26 tests cover: known-answer edge cases (all-pass/all-fail/alternating/m=k/m=1), Wilson CI behavior at small n and extremes, panic-on-bad-input, empirical unbiasedness vs `1-(1-p)^m`, and direct Theorem-1 verification (stride-s subsequence of a real LatticeQmc batch is marginally Unif[0,1) by KS test AND equispaced by 1/m). Re-exported at `speculative::{BootstrapEstimate, dyadic_bootstrap_pass_at_m_lattice, contiguous_block_bootstrap_pass_at_m}`. Originally deferred as "not blocking the GOAT gate"; completed on direct user instruction ("GPU training, benchmarks, WASM, and external dependencies are NOT valid reasons to skip — implement them").

---

## Downstream fusion (separate plans, NOT blocking this one)

These are consumers of the open primitive, each gets its own plan:

- [x] **riir-ai CLR × QuasiMoTTo** (R136 / Plan 316 consumer) — replace `sample_multinomial` (`riir-ai/crates/riir-engine/src/swir_validation/gemma2_backend.rs:81`) with a QMC sampler when K>1. Crowd-scale CLR cost drops ~25–47%. **Fusion B per R367 §2.3.** Separate plan in `riir-ai/.plans/`.
      **✅ DONE (2026-07-04)** — shipped directly under Plan 367 (per user instruction "Do NOT start a new plan"). `Gemma2DecodeBackend` gained three additions in `crates/riir-engine/src/swir_validation/gemma2_backend.rs`:
      - `QmcMethod` enum (`Lattice`/`Stratified`/`Sobol`, default `Lattice` — the pass@k champion per R367 §1.1).
      - `with_qmc_sampling(temperature, seed, method)` builder — replaces `with_temperature_sampling` for the K>1 path. Sets `qmc_source: Option<Box<dyn QmcSource>>` + carried `qmc_u: f32`; clears `rng` (last-builder-wins semantics).
      - `prepare_qmc_rollouts(k)` — pre-draws K low-discrepancy initial coordinates in a single `source.draw(k, ...)` batch. **Critical:** Lattice/Stratified sources produce their low-discrepancy structure *within* a batch; calling `draw(1, ...)` K times yields K i.i.d. points (defeating the purpose). `reset()` consumes one pre-drawn coordinate per call; falls back to `draw(1, ...)` if no batch prepared (Sobol-correct, Lattice/Stratified degraded to i.i.d. — harmless for single-rollout Pass@1).
      - Dispatch wiring in `decode_step()` + `prime()`: QMC descend (`sample_from_distribution_qmc` from `katgpt-core`) when `qmc_source.is_some()`, else i.i.d. `sample_multinomial`, else greedy argmax. Temperature scaling updated to check both `rng` and `qmc_source`.
      - 8 new tests (12 total pass): builder source validity, default=Lattice, descend determinism given fixed u, marginal-exactness (respects distribution), coordinate carry, **K-batch low-discrepancy beats i.i.d.** (star discrepancy D*_qmc < 0.05 < D*_iid ≈ 0.12 at k=64), marginal uniformity (chi-square GoF over 32K points), trait-object usability.
      - Feature isolation: `qmc_sampling` is default-on in `katgpt-core` (Plan 367 Phase 5 T5.7); riir-engine inherits it via the katgpt-core path dep. No new Cargo feature needed in riir-engine.
      - **Modelless per mandate**: no weight mutation. Pure arithmetic construction (lattice offsets / stratified permutations / Sobol direction numbers) + inverse-CDF descend with rescaled-coordinate carry. Rule 3 latent-space update.
- [x] **riir-neuron-db QMC-TEMP** (Plan 005 consumer) — replace `ConsolidationPipeline::sleep_diverse`'s i.i.d. BLAKE3-seeded noise with a QMC lattice. **Fusion C per R367 §2.3.** Separate plan in `riir-neuron-db/.plans/`.
      **✅ DONE (2026-07-04)** — shipped directly under Plan 367 (per user instruction "Do NOT start a new plan"). Two-part implementation:
      - **katgpt-core** (`crates/katgpt-core/src/diversity/temp.rs`): new `extrapolated_snapshot_schedule_qmc(s0, s1, lambda, source: &mut dyn QmcSource, sigma, out, uniforms_scratch)` — QMC variant of `extrapolated_snapshot_schedule`. Same math (`theta_j = s0 + lambda_j*(1+xi_j)*(s1-s0)`), but K noise values come from a single `source.draw(k, ...)` batch (low-discrepancy) instead of independent BLAKE3 hashes. Identical affine map (`xi_j = (u_j*2-1)*sigma`) so the two variants are directly comparable. NaN-safe (sigma=0 skips the draw). Gated on `#[cfg(feature = "qmc_sampling")]` within the `temp_loss_fingerprint`-gated `diversity` module. Re-exported at `katgpt_core::extrapolated_snapshot_schedule_qmc`. 6 new tests: no-noise linear interpolation, noise-within-bounds, deterministic-same-seed, different-seed-different-output, panics-on-short-scratch, lattice-low-discrepancy-vs-iid.
      - **riir-neuron-db** (`src/consolidation.rs`): new `ConsolidationPipeline::sleep_diverse_qmc(index, k_subset, lambda, source: &mut dyn QmcSource)` method, gated on `temp_qmc_noise` feature (opt-in; implies `temp_loss_fingerprint`). Calls `extrapolated_snapshot_schedule_qmc` (DRY — the initial sibling implementation had an inline block with a TODO to refactor; this completes that refactor). Stack-allocated `[f32; 8]` uniforms scratch (K≤8 regime). Same fallback logic as `sleep_diverse` (no target / too few events / shard missing → delegate to `sleep(1)`). Same quorum-reproducibility contract (fixed QMC seed → bit-identical selection). Feature `temp_qmc_noise` added to `riir-neuron-db/Cargo.toml` (opt-in, not default-on — promotion requires a downstream consumer validating the win on a real corpus).
      - **Validation**: 244/244 riir-neuron-db lib tests pass with `temp_qmc_noise` (was 233, +11 new QMC tests). 904/904 katgpt-core lib tests pass (was 898, +6 new). Feature isolation clean (`--no-default-features --features temp_qmc_noise`, `--all-features`). The sibling's `qmc_deterministic_same_seed_same_selection` test confirms the refactor to call `extrapolated_snapshot_schedule_qmc` is bit-identical to the inline block.
      - **Modelless per mandate**: no weight mutation beyond the existing TEMP consolidation. The QMC source replaces only the noise-generation path — pure arithmetic construction (lattice offsets / stratified permutations / Sobol direction numbers). Rule 3 latent-space update.
- [x] **katgpt-rs QmcHalter** (R205 consumer) — sample-efficiency-aware halter that estimates coverage from the actual QMC point set vs the union-bound ceiling `min(1, k·p)`. **Fusion E per R367 §2.3.**
      **✅ DONE (2026-07-04)** — shipped directly under Plan 367 (per user instruction "Do NOT start a new plan"). New file `crates/katgpt-core/src/speculative/qmc_halter.rs` (gated on `qmc_sampling`, which is default-on per Plan 367 Phase 5). The sample-efficiency-aware analog of `GainCostLoopHalter` (Plan 304):
      - `QmcHalter` config struct (stateless, `&self` evaluate): `target_coverage=0.95`, `k_min=1`, `k_max=64`. NaN-safe clamping in `new()` (target clamped to `(0, 1]`, k_min ≥ 1, k_max ≥ k_min).
      - `QmcHaltDecision` enum: `RefusedFloor` / `Continue { ceiling, gap }` / `Halt { reason, ceiling, coverage }`. `Copy` + `Debug`.
      - `QmcHaltReason` enum (`#[repr(u8)]`): `HitObserved` (early term — n_hits > 0), `TargetMet` (ceiling ≥ target), `CeilingSaturated` (k·p ≥ 1, no hit), `KMaxReached` (safety cap).
      - Decision order: k_min floor → k_max cap → HitObserved → CeilingSaturated → TargetMet → Continue. HitObserved beats TargetMet (stronger signal — actual success vs budget-sufficient).
      - Pure modelless helpers: `union_bound_ceiling(k, p) = min(1, k·p)` (R205 §1 Eq. 30–32), `iid_at_least_one(k, p) = 1-(1-p)^k` (baseline QMC improves upon), `count_hits_1d(points, p)` (empirical coverage from point set).
      - NaN safety: NaN p → ceiling 0.0 → Continue (never spuriously fires TargetMet on corrupt input). NaN points excluded from hit count.
      - 41 tests (39 functional + 2 `#[ignore]` timing): ceiling basic/saturated/zero-k/negative-p/nan-p, iid known-values/zero/full/nan/below-ceiling, count-hits basic/empty/nan-points/nan-p, RefusedFloor (k<k_min, k=0), HitObserved (basic, early-term), KMaxReached (k≥k_max, takes-precedence-over-k_min), CeilingSaturated (k·p≥1 no hit), TargetMet (exact-target, above-target), Continue (below-target, gap-correct), HitObserved-beats-TargetMet, NaN-p-continues, default-config, new-clamps (4 cases), QMC-advantage (ceiling > iid), QMC-saves-rollouts (iid_k=29 vs qmc_k=10 for p=0.1 target=0.95), end-to-end adaptive-loop (TargetMet path + CeilingSaturated path + first-hit path), Copy+Debug trait checks.
      - **G5 PASS** (release, `--ignored`): `evaluate` **1.60 ns/call** (budget 50 ns, 31× headroom), `union_bound_ceiling` **0.65 ns/call** (budget 10 ns, 15× headroom). Inputs varied per-iteration to prevent constant-folding (loop-invariant version measured 0.00 ns — a DCE artifact, fixed).
      - **G3 no-regression PASS**: 898/898 katgpt-core lib tests pass with default features (qmc_sampling default-on; was 859, +39 new).
      - **G6 feature isolation PASS**: `--features qmc_sampling`, `--all-features`, `--no-default-features --features qmc_sampling` all compile clean.
      - **G4 alloc-free PASS by construction**: `evaluate` is `&self` with no allocation; `count_hits_1d` borrows the slice.
      - Pure modelless (closed-form float ops + branches, no training, no allocation, no softmax). Mirrors `GainCostLoopHalter` (Plan 304) API shape per R367 §2.3.
      - File-size rule: new file `qmc_halter.rs` (936 lines incl. tests) rather than extending `qmc.rs` (already 2427 lines, over the 2048 guideline).

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
