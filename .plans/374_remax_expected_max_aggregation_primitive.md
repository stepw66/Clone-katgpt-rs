# Plan 374: ReMax Expected-Max-Over-m Aggregation Primitive

**Date:** 2026-07-03
**Research:** [katgpt-rs/.research/373_ReMax_Expected_Max_Retry_Aggregation.md](../.research/373_ReMax_Expected_Max_Retry_Aggregation.md)
**Source paper:** [arxiv:2606.00151](https://arxiv.org/pdf/2606.00151) — Nishimori et al. ICML 2026, "Emergence of Exploration in Policy Gradient RL via Retrying"
**Target:** `katgpt-rs/src/pruners/remax.rs` (new module) + Cargo feature `remax_aggregation`
**Status:** Phase 1 COMPLETE (2026-07-03). 14/14 unit tests passing, clean compile with/without feature, doctests passing. Ready for Phase 2 (Monte-Carlo G1 gate).

---

## Goal

Ship the closed-form ReMax aggregation operator (`expected_max_over_m`) and Expected Improvement (`expected_improvement`) as a modelless inference-time primitive, behind the `remax_aggregation` feature flag. Benchmark against UCB1 / softmax / best_of_k on a controlled bandit regret domain. Promote to default-on only if it matches or beats UCB1's sublinear regret with a single m parameter (no count tracking, no bonus coefficient). If it fails → document as negative result alongside SDAR/RMSD/FFO.

**Why:** The ReMax operator is novel (no exact prior art in the codebase). It offers bonus-free exploration as an emergent property of objective curvature, controlled by a single continuous parameter m > 0. The codebase's negative prior (SDAR/RMSD/FFO — reward modulation doesn't improve action selection) makes a GOAT gate mandatory before any promotion.

**Stack slot:** action-selection / bandit layer. Coexists with UCB1 (BanditPruner), softmax, and best_of_k_rollouts — does NOT replace them unless the GOAT gate proves a clear win.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `remax_aggregation` feature flag to `katgpt-core/Cargo.toml` + root forwarding
  - Added `remax_aggregation = []` to `katgpt-core/Cargo.toml` (opt-in, NOT in `default`)
  - Added `remax_aggregation = ["katgpt-core/remax_aggregation"]` forwarding to root `Cargo.toml`
  - Added entry to README.md "Opt-In & Gated Features" table
  - **Location note:** placed in `katgpt-core/src/pruners/remax.rs` (not root `src/pruners/`),
    because katgpt-core is the only crate that ships to crates.io and riir-ai depends on
    it non-optionally. Matches `active_state.rs` pattern (zero-dep pruner in katgpt-core
    for downstream consumption).

- [x] **T1.2** Create `katgpt-core/src/pruners/remax.rs` with the core functions
  - `pub fn expected_max_over_m(pi: &[f32], q: &[f32], m: f32) -> f32` — Eq 4, scalar
  - `pub fn expected_improvement(r: f32, pi: &[f32], q: &[f32], m: f32) -> f32` — Eq 10, scalar
  - `pub fn expected_improvement_per_action(pi: &[f32], q: &[f32], m: f32) -> Vec<f32>` — Q_plus for RePPO baseline
  - `pub fn expected_improvement_per_action_inplace(pi, q, m, out: &mut [f32])` — zero-output-alloc variant
  - **Plan deviation:** T1.2 specified `expected_improvement(R, ...) -> Vec<f32>`. The paper's
    Eq 10 is a **scalar** formula (verified against Appendix D JAX code). The per-action
    `Q_plus` is a separate computation. Implemented as scalar `expected_improvement` +
    separate `expected_improvement_per_action`. See module docs §"Plan deviation note".

- [x] **T1.3** Re-export from `katgpt-core/src/pruners/mod.rs` (feature-gated)
  - `#[cfg(feature = "remax_aggregation")] pub mod remax;`

- [x] **T1.4** Unit tests (14 tests, all passing)
  - `test_m1_equals_mean` ✓
  - `test_m0_converges_to_min` ✓ (corrected from plan's erroneous "converges_to_max" —
    (1-C)^0=1 telescopes to q_K = min(q), not max(q))
  - `test_m_inf_converges_to_max` ✓
  - `test_deterministic_bandit_closed_form` ✓ (1-(1-p)^m for multiple m values)
  - `test_ei_zero_when_r_below_all_q` ✓
  - `test_ei_positive_when_r_above_all_q` ✓
  - `test_q_replacement_reduces_ei` ✓ (adapted: Q-replacement reduces scalar EI, not
    zeroes a per-action element — the plan's per-action interpretation was inconsistent)
  - `test_k1_max_returns_q_directly` ✓
  - `test_k1_ei_returns_clamped_diff` ✓
  - `test_numerical_stability_m_below_1` ✓
  - `test_monotone_in_m` ✓
  - `test_per_action_matches_scalar` ✓ (bonus: validates per-action against scalar)
  - `test_baseline_non_negative` ✓ (bonus: validates RePPO baseline property)
  - `test_inplace_matches_allocating` ✓ (bonus: validates zero-alloc variant)

---

## Phase 2 — Correctness Gate (G1)

### Tasks

- [ ] **T2.1** Monte-Carlo validation test
  - For K ∈ {2, 5, 10, 50, 128}, random pi (Dirichlet), random q (uniform [-1, 1])
  - Brute-force: sample M ∈ {1, 2, 3, 5, 10} draws from pi, take max, average over 10⁶ trials
  - Compare to closed-form `expected_max_over_m(pi, q, M as f32)`
  - Assert max abs error < 1e-3 for all (K, M) combos
  - Test with m ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} (continuous-m generalization)

- [ ] **T2.2** Run `cargo test -p katgpt-rs --features remax_aggregation --lib`
  - All Phase 1 + Phase 2 tests must pass

---

## Phase 3 — Bandit Regret Benchmark (G2 — the load-bearing gate)

### Tasks

- [ ] **T3.1** Create `benches/remax_bandit_regret.rs` (or inline benchmark in `src/benchmark.rs`)
  - K=10 Beta-Bernoulli bandit: means drawn from Beta(1,1), rewards Bernoulli(μ_a)
  - T=1000 rounds, 256 seeds
  - Methods to compare:
    - **UCB1** (c=1.0): select arm with highest `mean_a + c·√(ln(t)/n_a)` (after initial pull-all)
    - **Thompson sampling**: sample μ_a ~ Beta(α_a, β_a), select argmax
    - **Softmax** (τ=0.1): select arm ~ softmax(mean_a / τ)
    - **Greedy**: argmax(mean_a)
    - **ReMax(m)** for m ∈ {1.2, 1.4}: maintain posterior Beta(α,β), at each round optimize ReMax objective via gradient ascent on π (Alg 2/3 from App C.3), draw action from π
  - Report: mean ± std-error cumulative regret at T=1000, over 256 seeds
  - **Pass threshold:** ReMax(m∈[1.2,1.4]) cumulative regret within 1 std-error of UCB1 OR better

- [ ] **T3.2** Gaussian-Gaussian bandit variant (same structure, N(μ,1) rewards, N(0,1) prior)
  - Same methods, same metrics
  - This is the harder bandit (the paper shows Softmax fails worse here)

- [ ] **T3.3** If G2 PASSES → proceed to Phase 4. If G2 FAILS (ReMax worse than Softmax or far worse than UCB1) → stop, document as negative result in `.benchmarks/374_remax_goat.md`, keep `remax_aggregation` opt-in, update the research note verdict to "NO GOAT — negative result, same class as SDAR/RMSD."

---

## Phase 4 — Latency + No-Regression Gate (G3, G4, G5)

### Tasks

- [ ] **T4.1** G4 latency benchmark (criterion)
  - `benches/remax_latency.rs`: measure `expected_max_over_m` and `expected_improvement` for K ∈ {8, 16, 32, 64, 128}
  - Baseline: UCB1 score computation (mean + sqrt term per arm)
  - Budget: < 500 ns per call for K ≤ 128 (plasma tier, sub-µs)
  - Use `CARGO_TARGET_DIR=/tmp/remax_bench` per AGENTS.md rule

- [ ] **T4.2** G3 no-regression: bomber arena (or equivalent toy game)
  - Run 1000 games with action selection via ReMax(m=1.2) vs greedy vs UCB1
  - Win/loss/draw rate must be within 5% of best baseline
  - This is the "does it actually help game AI?" gate

- [ ] **T4.3** G5 feature-isolation check
  - `cargo check` (no features) — clean
  - `cargo check --features remax_aggregation` — clean
  - `cargo check --all-features` — clean (no combo regression)
  - 0 warnings

---

## Phase 5 — GOAT Gate Verdict + Promotion Decision

### Tasks

- [ ] **T5.1** Write `.benchmarks/374_remax_goat.md` with all gate results
  - G1 (correctness): PASS/FAIL with numbers
  - G2 (bandit regret): PASS/FAIL with regret curves + table
  - G3 (no-regression): PASS/FAIL with arena results
  - G4 (latency): PASS/FAIL with criterion numbers
  - G5 (feature-isolation): PASS/FAIL

- [ ] **T5.2** Promotion decision
  - **If ALL gates PASS** → add `remax_aggregation` to `default` feature list, update README "Always-On Hot Path" section, update research note status to "Done — GOAT promoted"
  - **If G2 FAILS** → keep opt-in, document as negative result (NO GOAT), update research note verdict, reference in `.docs/20_negative_results.md` alongside SDAR/RMSD/FFO
  - **If G2 PASSES but G3/G4 FAIL** → keep opt-in, note the gate that failed, create issue for optimization

- [ ] **T5.3** Update per-stack ledger in research note §3
  - Record: stack slot = action-selection/bandit; outcome = promoted-to-default / opt-in-negative / opt-in-needs-optimization

---

## Phase 6 — riir-train Cross-Ref (out of scope, noted only)

- [ ] **T6.1** Note the RePPO training algorithm redirect in `riir-train/.research/` (one-line: "ReMax/RePPO training loop — see katgpt-rs Research 373 for the modelless distillation; the training algorithm (PPO variant + EI advantage + Q-critic) belongs here.")
  - Do NOT implement RePPO in this plan. Training is out of scope for katgpt-rs.

---

## Deferred / Fusion (not in this plan)

- [ ] **[-]** ReMax × BoMSampler fusion (R248) — replace K-sample BoM with closed-form expected-max. Needs BoM's value distribution API. Track as issue after GOAT gate.
- [ ] **[-]** ReMax × AdvantageMarginGate fusion (R250) — use EI as recursion-loop gating signal. Needs the primitive to ship first.
- [ ] **[-]** ReMax × Manifold Bandit (R370) — expected-max-over-m at Thompson tree nodes. Needs the manifold bandit to stabilize first.
- [ ] **[-]** riir-ai per-NPC action selection guide (HLA → action with curiosity-driven m). Create `riir-ai/.research/NNN_*.md` only if the GOAT gate passes and the game-AI use case is validated in G3.
