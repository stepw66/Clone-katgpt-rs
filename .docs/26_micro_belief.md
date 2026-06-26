# MicroRecurrentBeliefState (Plan 276)

Per-entity implicit state-tracking kernel — small frozen recurrent kernels implementing
`s_t = f(s_{t-1}, x_t)` over a fixed-size latent belief vector, applied once per
(entity, tick). The belief vector is **latent and local** (never synced); a bridge projects
it to **bounded raw scalars** that may cross the sync boundary.

**Research:** [`katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md`](../.research/242_Topological_State_Tracking_Recurrent_Belief.md)
**Plan:** [`katgpt-rs/.plans/276_micro_recurrent_belief_state.md`](../.plans/276_micro_recurrent_belief_state.md)
**Feature flag:** `micro_belief` (opt-in — see GOAT verdict below)

---

## TL;DR

The trait unification + `LeakyIntegrator` (Family C) are the **promotable outputs** —
the latter is byte-identical to the shipped `ReconstructionState::evolve_hla` and now shares
one math body via `katgpt_core::leaky_core::leaky_step`. The `AttractorKernel` (Family A)
and `LatentThoughtKernel` (Family B) remain behind the flag as **opt-in experiments** —
they lost both GOAT gates (G1.4 latency ~273ns/step vs <100ns; G2.1 coherence 569× more
flip-flops than leaky on the 1000-step benchmark).

---

## GOAT verdict

| Gate | Result | Notes |
|---|---|---|
| G1.1 Determinism | ✅ PASS | Bit-identical `s_T` for fixed input sequence. |
| G1.2 Boundedness | ✅ PASS | `‖s_t‖` stays bounded over 10k ticks. |
| G1.3 Bridge ranking | ✅ PASS | Sigmoid is monotone → dot ranking preserved. |
| G1.4 Latency | ❌ FAIL | ~273 ns/step vs <100ns target (Issue 024). |
| G1.5 Snapshot atomicity | ✅ PASS | Readers never see a torn kernel swap. |
| **G2.1 Coherence** | ❌ **FAIL** | Attractor flips **569×** more than leaky (1 flip) on the 1000-step ambiguous-window benchmark. |

**Decision (Plan 276 T5.2):** attractor family **demoted to Gain**. The hysteresis
hypothesis needs *trained* recurrent weights (fixed-point basins); at random Xavier init
the attractor is a generic nonlinear dynamical system whose argmax is noise-sensitive. The
leaky integrator wins because its per-tick `max_delta` clamp makes it robust to small
ambiguous-window perturbations. Training the weights is out of scope (katgpt-rs is
training-free / freeze-thaw only → that would be riir-train).

See [`katgpt-rs/.benchmarks/276_micro_belief_goat.md`](../.benchmarks/276_micro_belief_goat.md).

---

## API

### Trait

```rust
pub trait MicroRecurrentBeliefState: Send + Sync {
    fn dim(&self) -> usize;
    fn step(&self, state: &mut [f32], input: &[f32]);                 // zero-alloc
    fn project_to_scalars(&self, state: &[f32], directions: &[f32],   // bridge
                          dim: usize, out: &mut [f32]);
    fn family(&self) -> RecurrenceFamily;                              // routing
}
```

### Families

| Family | Struct | Update rule | Status |
|---|---|---|---|
| A — Attractor | `AttractorKernel` | `s_t = 2·σ(W_s·s + W_x·x + b) − 1` | Opt-in experiment (G1.4 + G2.1 FAIL) |
| B — LatentThought | `LatentThoughtKernel` | K iters of Family A per tick | Opt-in experiment (K=1 bit-identical to A: G1.6) |
| C — DeltaRule | `LeakyIntegrator` | monotone additive, `±max_delta` clamp | **Promotable** — byte-identical to `evolve_hla` |

### Snapshot (freeze/thaw)

`MicroRecurrentKernelSnapshot { family, dim, weights_blob, blake3, version }` —
BLAKE3-committed over `(family, dim, weights_blob)`. Reuses the `SenseModule::commit()`
pattern. The future `KernelHotSwap` will reuse the `SenseHotSwap` `AtomicPtr` primitive.

### Bridge

`project_to_scalars(state, directions, dim, out)` → `out[k] = fast_sigmoid(dot(state, direction_k))`.
Latent → raw, one-way, zero-allocation. Reuses `crate::simd::{simd_dot_f32, fast_sigmoid}`.

### Shared core (Phase 2 refactor)

`katgpt_core::leaky_core::leaky_step(state, input, total, lr, max_delta)` — the single source
of truth for the leaky-integrator update body. `ReconstructionState::evolve_hla` (sum-over-6
total) and `LeakyIntegrator::step` (sum-over-dim total) both delegate to it. The `total`
parameter is caller-controlled because the two callers aggregate differently (evolve_hla
sums 6 source activations; the generic kernel sums all `dim`).

### Reconstruction early-stop criteria (4)

The HLA reconstruction loop (`ReconstructionState::reconstruct*`) halts via four
complementary signals, checked in order each step:

| # | Criterion | Signal | Semantics | Source |
|---|-----------|--------|-----------|--------|
| 1 | `max_steps` (default 3) | step count | "MRAgent shows diminishing returns after 3-4" | `ReconstructionConfig.max_steps` |
| 2 | `entropy_threshold` (default 0.05) | activation entropy | "evidence is sharp enough — distribution converged" | `TripleEvidence::activation_entropy()` via `sufficient()` |
| 3 | `adaptive_budget` (Phase 6) | measured cycle latency | "spending too much time — reduce `max_steps`" | `LATENCY_BUDGET_NS = 500` |
| 4 | **advantage-margin gate** (Plan 283 T5.1) | `A(candidate) − E[A(a)]` | **"did this step help?" — improvement signal** | `ReconstructionConfig.advantage_margin_threshold` (default `NaN` = disabled; feature `self_advantage_gate`) |

Criterion 4 is the **only** one that asks "did this step help?" (improvement) rather than
"is this step done?" (sufficiency) or "is this step slow?" (budget). It catches dead compute
that the other three miss — e.g., argmax-drift-with-sharp-entropy where the activation
distribution stays sharp but the top module changes between steps.

The math is an inline minimal of the canonical `AdvantageMarginGate` (root crate,
`src/pruners/self_advantage.rs`, Plan 283). Kept inline because katgpt-core cannot depend on
the root crate. Module activations are sigmoid-bounded `[0, 1]` (treated as logits over 6
module candidates); the advantage math is scale-invariant but the threshold needs separate
tuning from the LLM-logit benchmark.

---

## Latent vs raw boundary (AGENTS.md)

| Quantity | Space | Synced? | Versioned? |
|---|---|---|---|
| `belief_vector s_t` (live state) | Latent | NO | NO (ephemeral) |
| Kernel weights (`W_s, W_x, b`)   | Latent | NO | YES (snapshot, BLAKE3) |
| Bridge direction vectors         | Latent | NO | YES (in snapshot) |
| Projected scalars (valence, …)   | Raw    | YES | NO (event stream) |
| `kernel_swap_event`              | Raw    | YES | YES (audit trail) |

Never sync the full belief vector — sync the K projected scalars instead (5 equations, 32
unknowns: syncing scalars does not let an attacker reconstruct `s_t`).

---

## Usage

See [`katgpt-rs/examples/micro_belief_demo.rs`](../examples/micro_belief_demo.rs) for a
minimal end-to-end lifecycle (construct → 1000 steps → project to 3 scalars → snapshot).

```bash
cargo run --release --example micro_belief_demo --features micro_belief
```
