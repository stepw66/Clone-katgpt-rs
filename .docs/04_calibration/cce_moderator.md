# CCE Moderator — API Reference & Worked Examples

**Plan:** [295](../.plans/295_lp_cce_moderator_primitive.md)
**Research:** [274](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
**Paper:** [arxiv 2606.20062](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzouanas 2026
**Feature gate:** `cce_moderator` (default-off; promote pending G1+G2 GOAT gate)
**Crate:** `katgpt-rs/src/cce/`

---

## TL;DR

Generic, game-agnostic Coarse Correlated Equilibrium (CCE) primitives for
finite state-action games. Three public types:

- **`ExternalRegret`** — closed-form external-regret functional `ER(ρ) = max_κ (γ(ρ) − γ_dev(ρ, κ))`, plus uniqueness check (Assumption 6.2) and linear derivative (Lemma 6.5).
- **`CceLp`** — LP solver over occupation measures `ρ ∈ P(S × A)`. Finds the optimal CCE `ρ⋆ = argmin_{ρ ∈ CCE} γ₀(ρ)` via basic-feasible-solution enumeration.
- **`CcePrimalDual`** — Bregman primal-dual iterator with `O(N⁻¹ᐟ²)` averaged-iterate convergence (Euclidean potential = projected gradient descent).

All three are **modelless** (no backprop, no training), **generic** over
`<const N, const A>`, and contain **no game semantics** — the latent-space
reframing (state = HLA bucket, action = CGSP arm) lives in riir-ai Plan 325.

---

## Quick Start

```rust
use katgpt_rs::cce::{
    CceLp, CcePrimalDual, Deviation, DeviationClass, ExternalRegret,
    OccupationMeasure, PayoffTensor,
};

// 1. Define your game: impl PayoffTensor<N, A>.
struct MyGame;
impl PayoffTensor<4, 2> for MyGame {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        // cost(s, a) — MINIMIZE convention.
        COST_MATRIX[state][action]
    }
    fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
        self.gamma(rho) // default: moderator objective = player cost.
    }
}

// 2. Define the deviation class.
struct MyDevs { v: Vec<Deviation<4, 2>> }
impl DeviationClass<4, 2> for MyDevs {
    fn deviations(&self) -> &[Deviation<4, 2>] { &self.v }
}
let devs = MyDevs {
    v: vec![
        Deviation::<4, 2>::constant(0, 0),
        Deviation::<4, 2>::constant(1, 1),
    ],
};

// 3a. Solve for the optimal CCE via LP.
let rho_star = CceLp::new().solve(&devs, &MyGame).expect("LP feasible");
assert!(CceLp::new().is_cce(&rho_star, &devs, &MyGame, 1e-4));

// 3b. Or learn it online via primal-dual.
let report = CcePrimalDual::new::<4, 2>()
    .with_eta(0.05)
    .run(&devs, &MyGame, 10_000);
assert!((report.gamma0_avg - MyGame.gamma0(&rho_star)).abs() < 0.05);
```

---

## API Reference

### Core Types (`types.rs`)

#### `OccupationMeasure<const N, const A>`

A probability distribution over `S × A` (length `N·A`, row-major, sums to 1).

| Method | Description |
|---|---|
| `new(entries: Vec<f32>) -> Result<Self, OccupationMeasureError>` | Validate + construct. |
| `uniform() -> Self` | Uniform distribution `1/(N·A)` per entry. |
| `dirac(state, action) -> Self` | Point mass on one `(s, a)`. |
| `at(state, action) -> f32` | `ρ(s, a)`. |
| `marginal_state(state) -> f32` | `μ(s) = Σ_a ρ(s, a)`. |
| `flat_index(state, action) -> usize` | `(s, a) → s·A + a`. |

#### `Deviation<const N, const A>`

A fixed alternative policy `κ : S → P(A)`. Stored as `kernel: [[f32; A]; N]`.

| Constructor | Description |
|---|---|
| `constant(id, action)` | Always play `action` regardless of state. |
| `identity(id)` | Play the recommended action (requires `N == A`). |
| `from_kernel(id, kernel)` | Custom kernel (caller validates). |

#### `trait DeviationClass<N, A>`

A finite set of deviations `D = {κ₁, …, κ_K}`.

| Method | Description |
|---|---|
| `deviations(&self) -> &[Deviation<N, A>]` | Slice of all deviations. |
| `apply(κ, ρ) -> OccupationMeasure` (default) | Deviated measure `ρ'(s, a') = μ(s)·κ(s)[a']`. |

#### `trait PayoffTensor<N, A>`

The cost tensor. **Cost convention**: minimize.

| Method | Description |
|---|---|
| `reward_follow(s, a) -> f32` | Per-index cost `cost(s, a)`. **Required.** |
| `reward_deviate(s, κ) -> f32` (default) | `Σ_{a'} κ(s)[a']·cost(s, a')`. |
| `gamma(ρ) -> f32` (default) | `Γ(ρ) = Σ ρ·cost`. Cost of following. |
| `gamma_dev(ρ, κ) -> f32` (default) | `Γ_dev(ρ, κ) = Σ_s μ(s)·reward_deviate(s, κ)`. |
| `gamma0(ρ) -> f32` | Moderator objective `Γ₀`. **Required.** |
| `gamma0_coeff(s, a) -> f32` (default) | Per-index coefficient of `Γ₀` (default: `= reward_follow`). |

### `ExternalRegret` (`external_regret.rs`)

Stateless regret evaluator. All methods take `&D` and `&P` per call.

```text
ER(ρ) = max_{κ ∈ D} (γ(ρ) − γ_dev(ρ, κ))
```

| Method | Returns | Notes |
|---|---|---|
| `er(ρ, d, p)` | `f32` | External regret. CCE condition: `ER ≤ 0`. |
| `best_deviation(ρ, d, p)` | `Option<&Deviation>` | Argmax κ. `None` if `D` empty. |
| `is_unique_maximizer(ρ, d, p, ε)` | `bool` | Assumption 6.2: top-2 gap > ε. |
| `linear_derivative(ρ, m_flat, d, p)` | `f32` | `∂ER/∂ρ[m]` per Lemma 6.5. |

**Convention**: `ER = 0` at Nash. `ER < 0` at strict CCE. `ER > 0` is NOT a CCE.

### `CceLp` (`lp.rs`)

LP-CCE solver via BFS enumeration.

| Method | Returns | Notes |
|---|---|---|
| `solve(d, p)` | `Result<OccupationMeasure, CceLpError>` | Optimal `ρ⋆ = argmin γ₀`. |
| `is_cce(ρ, d, p, ε)` | `bool` | Verify `ER(ρ) ≤ ε`. |

**Complexity**: `O(C(N·A + |D|, 1 + |D|) · m³)` where `m = 1 + |D|`. Exact for
`N·A + |D| ≤ ~25` (emission-abatement N=4,A=4: `C(20, 5) = 15504` candidates, <1ms).

### `CcePrimalDual` (`primal_dual.rs`)

Bregman primal-dual iterator (Algorithm 1).

```text
ρ⁰ = uniform, λ⁰ = 0
for n = 1, 2, …:
    grad[m] = gamma0_coeff(m) + λⁿ⁻¹ · linear_derivative(m)
    ρⁿ = project_simplex(ρⁿ⁻¹ − η · grad)
    λⁿ = max(0, λⁿ⁻¹ + (1/√n) · ER(ρⁿ))
    ρ̄ⁿ = ((n−1)/n)·ρ̄ⁿ⁻¹ + (1/n)·ρⁿ
```

| Method | Returns |
|---|---|
| `new::<N, A>()` | Self (uniform init, λ=0, η=0.1). |
| `with_eta(η)` | Builder: override step size. |
| `with_initial_rho(ρ)` | Builder: override ρ⁰. |
| `step(d, p)` | `StepReport` (one iteration). |
| `run(d, p, n_steps)` | `ConvergenceReportRaw<N, A>` (averaged iterate + history). |

**Convergence**: averaged iterate `ρ̄ᴺ` satisfies `|γ₀(ρ̄ᴺ) − γ₀(ρ⋆)| = O(N⁻¹ᐟ²)`
and `ER(ρ̄ᴺ) ≤ O(N⁻¹ᐟ²)` (Theorem 6.1).

---

## Worked Example: Chicken Game

**Setup**: 2-player chicken, modeled as player-1-only CCE. State = (s₁, s₂)
joint recommendation (N=4). Action = a₁ ∈ {S, T} (A=2).

Reward matrix `R[a₁][s₂]`:

```text
        s₂=S  s₂=T
a₁=S     3     1
a₁=T     4     0
```

Cost = -reward (minimize convention).

### LP Solution

With `γ₀ = γ` (player 1 cost):

```text
ρ⋆ = δ_{(state (T,S), action T)}    (player 1 plays T against opponent S)
γ₀(ρ⋆) = -4    (reward 4)
```

With `γ₀ = -welfare` (welfare maximization):

```text
ρ⋆ = 0.5·δ_{(state (S,S), action S)} + 0.5·δ_{(state (S,T), action S)}
γ₀(ρ⋆) = -5.5    (welfare 5.5)
```

### Primal-Dual Convergence (Emission-Abatement, N=4, A=4)

| n | γ₀(ρ̄ⁿ) | gap to LP |
|---|---|---|
| 100 | 1.0799 | 0.0799 |
| 1000 | 1.0080 | 0.0080 |
| 10000 | 1.0008 | 0.0008 |

Empirical rate: `O(N⁻¹)` (slope -1.0), steeper than the paper's `O(N⁻¹ᐟ²)` worst-case bound.

---

## GOAT Gate Status

| Gate | Target | Status |
|---|---|---|
| G1 — CCE ≥ Nash | Welfare gain ≥ 5% on chicken + BoS | **PASS** (chicken +37.5%, BoS +108%) |
| G2 — Primal-dual convergence | gap < 0.05, ER ≤ 0.05, slope ≤ -0.3 | **PASS** (gap=0.0008, ER=0.00003, slope=-1.0) |
| G3 — Designer steering | Two Γ₀ → two different CCEs | **PASS** (selfish welfare 5.0 vs welfare-max 5.5) |
| G4 — Crowd-scale latency | < 50µs per NPC update | Pending (riir-ai Plan 325) |
| G5 — LatCal commitment | Bit-identical | Pending (riir-ai Plan 325) |

See `.benchmarks/029_cce_convergence.md` for G2 details, `tests/cce_vs_nash.rs` for G1, `examples/cce_demo.rs` for G3.

---

## Limitations

1. **Player-1-only CCE.** The deviation class `D` models only one player's deviations. Multi-player CCE (both players' constraints) requires extending `D` — deferred to riir-ai Plan 325.
2. **No dynamics.** The LP treats the state distribution as free. MFG dynamics (occupation-measure flow constraints) are a Plan 325 follow-up.
3. **BFS enumeration LP.** Exact for `N·A + |D| ≤ ~25`. Larger problems need a real simplex — swap `CceLp::solve` internals when needed.
4. **Euclidean Bregman only.** KL potential (entropic mirror descent) is implemented in `bregman.rs` but not wired into `CcePrimalDual`.

---

## Cross-References

- **Plan**: [`katgpt-rs/.plans/295_lp_cce_moderator_primitive.md`](../.plans/295_lp_cce_moderator_primitive.md)
- **Research**: [`katgpt-rs/.research/274_Optimal_CCE_Moderator_LP_No_Regret.md`](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
- **Private selling-point guide**: `riir-ai/.research/143_Latent_CCE_Moderator_Crowd_Emergent_Coordination.md`
- **Private runtime plan**: `riir-ai/.plans/325_latent_cce_moderator_runtime.md`
- **Benchmarks**: [`.benchmarks/029_cce_convergence.md`](../.benchmarks/029_cce_convergence.md) (G2)
- **Tests**: `tests/cce_convergence.rs` (G2), `tests/cce_vs_nash.rs` (G1)
- **Example**: `examples/cce_demo.rs` (G3 designer steering)
