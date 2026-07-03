# Research 367: QuasiMoTTo — Quasi-Monte Carlo Test-Time Scaling

> **Source:** [QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling](https://arxiv.org/abs/2607.01179) — Li, Zhan, Gandhi, Goodman, Fox (Stanford), 2026-07-01
> **Date:** 2026-07-03
> **Status:** Done
> **Related Research:** 205 (union-bound branch confidence — same ceiling math), 248 (BoM diverse sampling — the i.i.d. baseline this replaces), 318 (sleep-time anticipation — UQ-on-correlated-samples cousin), 322 (conformal floor — open problem)
> **Related Plans:** 281 (BoMSampler — drop-in target), 316 (per-NPC CLR runtime — crowd-scale fusion consumer)
> **Cross-ref (riir-ai / riir-neuron-db):** riir-ai R136 (Per-NPC CLR), R142 (Distributional Branching Point), R158 (Committed Personality Blend — vocabulary collision, see §3 disambiguation); riir-neuron-db Plan 005 (TEMP diversity selector — QMC-TEMP fusion)
> **Classification:** Public

---

## TL;DR

QuasiMoTTo replaces i.i.d. parallel sampling with **correlated but marginally exact** samples drawn via randomized Quasi-Monte Carlo (QMC) and mapped to token sequences via arithmetic coding (inverse-CDF descend). Each rollout is marginally distributed EXACTLY per the LM — no bias — but the batch covers the output space more evenly than i.i.d. This buys 25–47% fewer samples for matched pass@k and 50% fewer GRPO training steps. The RL claim is training-time; the **sampler is modelless** and drop-in for any K-rollout pool.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is a **low-discrepancy uniform source** (`LatticeQmc`, `StratifiedQmc`, `SobolQmc`) plus the **arithmetic-coding descend operator** with rescaled-coordinate carry (`u_{t+1} = (u_t - ℓ_t)/p_t`). The descend operator already ships as `sample_from_distribution` (`crates/katgpt-core/src/speculative/sampling.rs`); only the uniform source changes from i.i.d. to QMC. The marginal-exactness theorem (linearity of expectation: average-type estimators are unbiased regardless of joint) is the contract — a QMC batch is a valid drop-in wherever the consumer only needs the per-rollout marginal.

---

## 1. Paper Core Findings

### 1.1 The mechanism

Two-stage construction:

**Stage A — QMC uniform coupling (the joint design):**
Three methods, all producing k points in [0,1) that are marginally uniform but more evenly spread than i.i.d.:
- **Lattice** — k points on the unit circle `{(i/k + Δ) mod 1 : i=0..k-1}` with a single shared offset `Δ ∼ Unif[0,1]`. One degree of freedom (Δ); each grid point is uniform because Δ is. Pairwise MI `I(U_i;U_j) = -∞` (each point determines every other).
- **Stratified** — divide [0,1) into k equal strata, draw one point per stratum `U_i ∼ Unif[i/k, (i+1)/k)`, then permute. Pairwise MI `= log(k/(k-1))`.
- **Token-level Sobol** — multi-dim QMC in `[0,1)^n` (n = sequence length); coordinate j drives token position j. Local coverage at each token rather than global.

**Freedom vs. coverage tradeoff:** i.i.d. (max freedom, 0 MI, weak coverage) → stratified → lattice (min freedom, max coverage). Lattice dominates pass@k; stratified empirically wins RL (lower RLOO bias under dependence).

**Stage B — arithmetic coding / inverse-CDF descend (the marginal-preserving map):**
Sequence sampling as `Φ: [0,1) → V*` via recursive interval partition: prefix `x_<t` has interval `I(x_<t)` of length `π(x_<t)`; the next-token conditional partitions that interval into bins of width `π(x_t | x_<t)`. The descend carries only the **local coordinate** `u_t` (rescaled to current interval), avoiding the numerically unstable raw `π(x_<t)`:

```
x_t = F_t^{-1}(u_t | x_<t)           # inverse-CDF lookup
u_{t+1} = (u_t - ℓ_t) / p_t           # rescale to local coordinate of selected bin
```

Each rollout is an independent descend of the trie steered by its own `u` — **generation is embarrassingly parallel**, no inter-rollout communication. The only added per-sample state is the running coordinate `u_t`.

### 1.2 The theorem that makes it drop-in

**Linearity of expectation over average-type estimators:** for any per-rollout quantity h,
```
E_μ[ (1/k) Σ h(τ_i) ] = (1/k) Σ E_{τ_i∼μ_i}[h(τ_i)] = E_{τ∼π_θ}[h(τ)]
```
The first equality holds for ANY joint μ; the second uses only the marginal condition `μ_i = π_θ`. Vanilla policy gradient (`∇log π(τ)·R(τ)`) is one such average. So a marginally-correct QMC batch is a **drop-in replacement for i.i.d.** in every average-type estimator.

### 1.3 The ceiling

Union bound: `pass@k ≤ min(1, k·p)` where p = pass@1. No marginal-preserving sampler can exceed it. QuasiMoTTo **nearly saturates** this ceiling on Maze / Sudoku / Countdown / 1D-ARC — leaving little room for any training-free sampler to do better.

### 1.4 Estimator fixes for the joint

Two standard estimators break under dependence and get fixes:
- **pass@k dyadic bootstrap** (Theorem 1): for lattice with k=2^L, any stride-2^x subsequence is itself a valid randomized lattice of size m=k/2^x. So a pass@8 rollout yields 2 unbiased pass@4 estimates (the 2 starting offsets). Sobol/stratified get analogous block-bootstrap.
- **RLOO bias** (§2.4.2): the leave-one-out baseline is no longer independent of `τ_i` under QMC. Empirically negligible; theoretically correctable via a product-of-differences importance-sampling rewrite, but the lattice's single degree of freedom violates the support condition for the importance ratio (observed ratio ≪ 1).

### 1.5 RL training dynamics

QuasiMoTTo reduces **zero-variance groups** (GRPO groups where all G rollouts succeed or all fail → no gradient signal). Higher coverage → larger effective sample size → 50% fewer steps to target pass@1 on Maze/Sudoku.

---

## 2. Distillation

### 2.1 The transferable primitive

```
trait QmcSource { fn draw(&mut self, k: usize) -> &[f32]; }  // k marginally-Unif[0,1) points

enum QmcMethod { Lattice, Stratified, Sobol { dim: usize } }

// Drop-in replacement for K calls to rng.uniform() in K-rollout paths.
// Each u_i feeds an independent arithmetic-coding descend.
```

The descend operator (`sample_from_distribution`) already ships in katgpt-core. The token-coordinate carry (`u_{t+1} = (u_t - ℓ_t)/p_t`) is the only addition — and it's numerically stable (no raw sequence probability).

### 2.2 Latent-space reframing (mandatory before verdict)

Re-cast QMC as a low-discrepancy operator on each of the seven Super-GOAT factory substrates:

| Substrate | Reframing | Strength |
|---|---|---|
| **HLA 8-dim latent state** | Sobol/lattice over the HLA action simplex gives low-discrepancy coverage of the personality blend's output distribution. QMC basis vectors live in the same space as the BLAKE3-committed direction vectors. | **Medium** — clean latent op, but HLA already projects via sigmoid gates; the marginal-exactness contract is on the *projection target*, not the basis. |
| **latent_functor** | QMC over the functor's K-selector — correlated-but-exact K applications covering the operator space. | Weak — functor applications are deterministic ops, not stochastic rollouts. |
| **cgsp_runtime curiosity** | Curiosity-driven exploration with QMC-correlated rollouts — coverage of the curiosity simplex. | Medium — but cgsp already uses i.i.d. rollouts and the gain is sample efficiency, not new behavior. |
| **NeuronShard / freeze envelope** | Commit Δ (the lattice shift, 1 f32) raw via BLAKE3+Merkle per FAME v1 protocol → quorum-verifiable rollout diversity for anti-cheat on RL self-play. | **Weak-narrow** — commitment protocol already exists (FAME); the anti-cheat consumer is net-new and the value depends on RL self-play being an adversarial surface. |
| **LatCal fixed-point** | Commit QMC direction numbers (Sobol) as LatCal fixed-point blocks → algebraically-verifiable low-discrepancy structure. | Weak — overkill; BLAKE3 is sufficient for ∆ commitment, and LatCal algebraic verification adds no value for a 1-scalar payload. |
| **DEC Stokes operators** | Low-discrepancy sampling over a cochain — QMC quadrature on the Hodge decomposition. | Orthogonal — DEC operators are deterministic; QMC quadrature is a separate numerical-methods concern. |
| **BoMSampler (katgpt-rs)** | Replace K i.i.d. Gaussian-noise belief queries with a QMC lattice over the K-dim belief ball. Same batched matvec, correlated-but-exact K hypotheses. | **Strong** — directly addresses R248 §1.5's stated BoM limitation ("No mechanism encourages diverse query-space utilization"). |

**Reframing verdict:** the strongest latent reframing is the **BoM × QMC fusion** (belief-space low-discrepancy coverage) — but that is a *belief-space* application of QMC, not a new latent op. The HLA-subspace reframing is medium-strength. No reframing produces a new capability class; all are sample-efficiency gains on existing substrates. This is the signal that the verdict is **GOAT, not Super-GOAT**.

### 2.3 Fusion

**Fusion A — QuasiMoTTo × BoMSampler (katgpt-rs, strongest):**
`QmcBoMSampler` — replace `SeedStrategy::PerNpc`'s i.i.d. Gaussian queries with a rank-1 lattice / Sobol point set over the K queries. The batched-matvec structure is unchanged (K elementwise perturbations on one pre-computed base activation); cost stays at 1 matvec + K·(D adds + D sigmoids). The K hypotheses become marginally `N(0,σ²I)` exact but jointly low-discrepancy. Closes R248 §1.5's "bounded coverage" limitation. **Pure katgpt-rs open primitive.**

**Fusion B — QuasiMoTTo × Per-NPC CLR (riir-ai R136 / Plan 316, crowd-scale):**
When CLR's underlying brain is an LM (dialog NPCs, quest-generation NPCs, SwiR validation backend), replace `sample_multinomial` (the only LM-token sampler in riir-ai, `gemma2_backend.rs:81`) with a QMC sampler. The K candidates cover claim-space more evenly; same `(mean v)^M` CLR vote, lower K for same reliability verdict → **crowd-scale CLR cost drops ~25–47%**. Note: CLR's guard/Patrol/Alarm/Loot enumeration path is NOT LM sampling — fusion applies only to the LM-brain CLR path.

**Fusion C — QuasiMoTTo × TEMP diversity selector (riir-neuron-db Plan 005):**
`ConsolidationPipeline::sleep_diverse` currently uses i.i.d. BLAKE3-seeded noise for extrapolated snapshot perturbation. Replace with a QMC lattice → tighter coverage of the perturbed-loss-vector space with the same K checkpoints. Plausible GOAT (fewer checkpoints for same diversity), modelless, shard-side stores only the seeds.

**Fusion D — QuasiMoTTo × ICT Distributional Branching (riir-ai R142):**
ICT picks *when* to spend K (branching points only); QuasiMoTTo makes that K more productive per sample. The two signals reinforce: QMC's i.i.d.→correlated swap costs ~nothing when K is small, but the union-bound coverage gain is largest exactly where ICT says budget matters.

**Fusion E — QuasiMoTTo × Union-Bound Branch Confidence (katgpt-rs R205):**
R205 ships the additive-combination direction of the union bound (`P(deviation) ≤ Σ P_i`); QuasiMoTTo ships the saturate-the-bound-via-correlation direction. Together: a `QmcHalter` that estimates current coverage from the actual QMC point set, computes the gap to `min(1,k·p)`, and decides whether to draw more or stop. The sample-efficiency-aware analog of `GainCostLoopHalter` (Plan 304).

### 2.4 Conformal UQ — open problem (honesty flag)

QuasiMoTTo's dyadic bootstrap is an unbiased estimator on **dependent** samples. But the conformal floor (Plan 340, `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`) assumes **exchangeability**, which QMC-correlated samples **violate by construction** (the whole point of QMC is non-exchangeable coverage). Conformal calibration would need modification (block conformal, conformal-on-the-marginal, or desensitize via sub-sampling) to handle QMC dependence. This is a statistics-research question for katgpt-rs, not a fusion. **If any future primitive claims a UQ distribution built from QMC-correlated samples, the "Report the Floor" rule (Issue 010) applies and the floor comparison must use an exchangeability-safe variant.**

QuasiMoTTo itself is **NOT a UQ-bearing primitive** — it claims sample efficiency (fewer rollouts for same pass@k), not a probability distribution / predictive interval / coverage guarantee. So the conformal floor rule does not strictly apply to QuasiMoTTo's own GOAT gate.

---

## 3. Novelty Gate (§1.5)

| Question | Answer | Evidence |
|---|---|---|
| **Q1. No prior art?** | **YES (mechanism)**, NO (commitment shape) | Three sub-agents grepped BOTH paper vocab (`QMC\|Quasi-Monte\|Sobol\|lattice\|stratified\|antithetic\|low-discrepancy\|arithmetic.cod\|inverse-CDF\|marginal.*preserv`) AND codebase vocab (`BoM\|MaxProof\|set_attention\|DiversitySampler\|sampling.rs\|committed_blend\|FAME\|sleep_diverse`) across all 5 repos, BOTH layers (.research/.plans/.docs AND src/crates). **ZERO QMC sampler code ships anywhere.** The descend operator exists (`sample_from_distribution`) but consumes i.i.d. uniforms. The "marginal-preserving, joint-designed" *commitment shape* is prior art (FAME R158/R302 — but that's personality-freeze-under-dropout, a different concept; see disambiguation below). |
| **Q2. New class of behavior?** | **NO** | It's a sample-efficiency gain (fewer rollouts, same quality). Not a new capability — every consumer of K rollouts still works with i.i.d.; QuasiMoTTo just makes each rollout more productive. |
| **Q3. Product selling point?** | **NO** | Cannot finish "our NPCs do X no competitor can" — competitors can drop in any QMC library (PyTorch ships `SobolEngine`; the arithmetic-coding descend is textbook MacKay). The gain is engineering-grade, not moat-grade. |
| **Q4. Force multiplier ≥2 pillars?** | **PARTIAL** | Connects to CLR (pillar-adjacent, riir-ai R136), BoM (katgpt-rs substrate), Sleep-Time (riir-ai R163). But all connections are "make existing K-rollout paths cheaper" — multiplicative cost reduction, not new capability multiplication. |

**Verdict: NOT Super-GOAT.** Q2 and Q3 fail. The mechanism is novel at the code level (no QMC ships) but the *idea class* (correlated-but-exact sampling) is textbook (Owen 2013, Vilnis 2023, Sobol 1967). The gain is real and well-scoped, but it is an efficiency gain, not a moat.

### 3.1 Vocabulary-collision disambiguation (false-novelty-prevention)

| Term | FAME / R158 meaning | QuasiMoTTo meaning |
|---|---|---|
| "sampling invariance" | π frozen → observation dropout doesn't perturb dynamics | each rollout's marginal distribution EXACTLY matches the LM |
| "marginal" | per-NPC scalar (emotion projection onto a direction) | per-rollout marginal distribution |
| "commitment" | BLAKE3-frozen artifact | n/a |
| "joint-designed" | Lipschitz-composed archetype blend | QMC correlation structure on K rollouts |

These are **different concepts**. R158's "sampling invariance" is a determinism-under-dropout property of frozen personalities; QuasiMoTTo's "marginal correctness" is a distributional-exactness property of correlated samplers. Neither subsumes the other. A naive grep for "sampling invariance" produces a false-positive prior-art hit; the disambiguation is mandatory.

---

## 4. Verdict

**Tier: GOAT** — provable gain (sample efficiency: 25–47% fewer rollouts for matched pass@k; 50% fewer GRPO steps per the paper) over the existing i.i.d. baseline, but not a new class of capability. Promotes to default-on if the GOAT gate passes.

**One-line reasoning:** The QMC-correlated sampler is a genuinely unshipped modelless primitive that wins the "parallel-rollout uniform source" stack slot, but the gain is sample-efficiency (better numbers) not new-behavior (moat). Competitors can drop in any QMC library; the value is in fusing it with the existing K-rollout consumers (BoM, CLR, sleep_diverse) where the per-stack tuning lives.

### 4.1 MOAT gate per domain (§1.6)

| Repo | In-scope? | MOAT contribution | Verdict |
|---|---|---|---|
| **katgpt-rs** | YES | Paper-derived fundamental primitive (QMC uniform source + arithmetic-coding descend), passes GOAT via fusion (BoM × QMC). Promote/demote tracked per stack: **parallel-rollout sampler** slot. | **Ship behind `qmc_sampling` feature flag; GOAT gate decides promote-to-default vs demote-loser.** |
| **riir-ai** | consumer | Fusion target (CLR × QuasiMoTTo for crowd-scale cost reduction). Not a pillar-level contribution. | Consume the open primitive; no private guide needed (not Super-GOAT). |
| **riir-chain** | NO | Commitment shape already exists (FAME v1). ∆ commitment is a new instance, not a new primitive. | Out of scope — LatCal/BLAKE3 commitment of ∆ is mechanical reuse if ever needed. |
| **riir-neuron-db** | consumer (weak) | QMC-TEMP fusion is a sampler-quality GOAT, not a shard-storage moat. | Consume the open primitive; shard stores only the seeds. |
| **riir-train** | partial | The RL claim (50% fewer GRPO steps) is training-time → riir-train note only. The sampler itself stays here. | Note "GRPO step-reduction → riir-train" and stop; do NOT distill the sampler there. |

### 4.2 Per-stack promote/demote ledger (katgpt-rs MOAT contract)

| Stack slot | Current default | QuasiMoTTo competitor | Promotion rule |
|---|---|---|---|
| **Parallel-rollout uniform source** | i.i.d. `rng.uniform()` (in `sample_from_distribution`, `BoMSampler`, `ppot_resample_multi_strategy`) | `qmc_sampling` feature (`LatticeQmc` / `StratifiedQmc` / `SobolQmc`) | Promote to default IF G1 (marginal-exactness empirical test, KS p>0.05 per rollout) + G2 (≥25% sample reduction at matched pass@k on a toy reasoning task) + G3 (no regression on single-rollout paths) + G4 (zero-alloc, O(k) extra state) all PASS. Otherwise demote to opt-in. |

---

## 5. Implementation sketch (plan-worthy)

**Target:** `katgpt-rs/crates/katgpt-core/src/speculative/qmc.rs` (new module) + Cargo feature `qmc_sampling` (opt-in).

**Phases:**
1. **Phase 1 — QmcSource trait + three methods** (`Lattice`, `Stratified`, `Sobol`). Zero-dep (no `rand` for the lattice/stratified; Sobol direction numbers either vendored or behind a `sobol-tables` optional dep). Unit tests: marginal uniformity (KS test), low-discrepancy (star-discrepancy ≤ i.i.d. baseline).
2. **Phase 2 — Arithmetic-coding descend with coordinate carry.** Extend `sample_from_distribution` to accept a `&mut f32` coordinate (the running `u_t`). Token-level Sobol composes naturally: each rollout's `u_t` is the t-th coordinate of its Sobol point.
3. **Phase 3 — Drop-in `sample_k_from_distribution_qmc`** that takes a `QmcSource` and produces K rollouts. Compose with `ppot_resample_multi_strategy`'s position-list API.
4. **Phase 4 — `QmcBoMSampler` (Fusion A).** Replace BoM's i.i.d. Gaussian queries with QMC lattice over the K-dim belief ball.
5. **Phase 5 — GOAT gate.** G1 marginal-exactness (KS test per rollout on a toy LM), G2 sample-efficiency (≥25% reduction at matched pass@k on Countdown/Maze toy), G3 no-regression (single-rollout paths unchanged), G4 alloc-free, G5 sub-µs overhead per rollout, G6 feature-isolation clean. If all PASS → promote `qmc_sampling` to default; demote the i.i.d. uniform source to opt-out.
6. **Phase 6 — Dyadic bootstrap pass@k estimator** (Theorem 1). Optional — only needed if a consumer wants pass@k estimation on QMC batches.

**Downstream fusion (separate plans, not blocking):**
- riir-ai: CLR × QuasiMoTTo wiring (replace `gemma2_backend.rs:sample_multinomial` when K>1).
- riir-neuron-db: QMC-TEMP (replace `sleep_diverse`'s i.i.d. noise with QMC lattice).

---

## 6. What does NOT ship here

- **The GRPO training loop** (50% fewer steps claim) → riir-train. The sampler is modelless; the training-method consumer is not.
- **Committing ∆ to chain** (quorum-verifiable rollout diversity). The commitment protocol exists (FAME v1, BLAKE3+Merkle); the anti-cheat-on-RL-self-play consumer is net-new and the value depends on RL self-play being an adversarial product surface. **Defer to an issue, not a plan** — this is an optimization/refactor task per the global rule.
- **Conformal-UQ-on-correlated-samples.** Open statistics problem (exchangeability violation). Track in `.issues/` if a future primitive needs it.

---

## TL;DR

QuasiMoTTo is a **GOAT-tier open primitive for katgpt-rs**: a QMC uniform source (`Lattice` / `Stratified` / `Sobol`) + arithmetic-coding descend that produces K correlated-but-marginally-exact rollouts, drop-in for any K-rollout path. The descend operator already ships (`sample_from_distribution`); only the uniform source changes. **Not Super-GOAT** — the mechanism is novel at the code level (zero QMC ships in any of the 5 repos, confirmed by three parallel sub-agent fusion searches across both paper and codebase vocabulary, both notes and code layers) but the gain is sample-efficiency (25–47% fewer rollouts, 50% fewer GRPO steps per the paper), not a new capability class or product moat. Strongest fusion: **QuasiMoTTo × BoMSampler** (`QmcBoMSampler` — closes R248 §1.5's stated "bounded coverage" limitation); crowd-scale fusion in riir-ai (CLR × QuasiMoTTo → ~25–47% lower K for same reliability verdict at 20Hz tick). Ship behind `qmc_sampling` feature flag in `katgpt-core::speculative::qmc`; GOAT gate decides promote-to-default vs demote-loser on the **parallel-rollout uniform source** stack slot. Open problem: conformal floor (Plan 340) assumes exchangeability, which QMC violates — track in `.issues/` if a future UQ-bearing primitive consumes QMC samples.
