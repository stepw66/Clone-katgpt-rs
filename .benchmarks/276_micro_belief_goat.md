# Plan 276 — MicroRecurrentBeliefState GOAT Gate Results

**Date:** 2026-06-16
**Plan:** [276_micro_recurrent_belief_state.md](../.plans/276_micro_recurrent_belief_state.md)
**Issue:** originally tracked in `024_micro_belief_g1_4_attractor_latency.md` (closed + removed; this benchmark is the canonical record)
**Features:** `micro_belief` (opt-in)

---

## TL;DR (read this first)

| Gate | Result | Decision |
|---|---|---|
| G1.1–G1.3, G1.5 | ✅ PASS | Trait + snapshot mechanics ship. |
| G1.4 latency | ❌ FAIL (~273 ns/step vs <100 ns target) | Attractor cannot be default-on (Issue 024). |
| **G1.6** (K=1 ≡ Family A) | ✅ PASS | `LatentThoughtKernel` is correct. |
| **G2.1 coherence** | ❌ **FAIL — attractor flips MORE than leaky** | **Attractor family DEMOTED to Gain (T5.2).** Only the trait unification + `LeakyIntegrator` ship as promotable output. |

**Shippable output of Plan 276:**
1. `MicroRecurrentBeliefState` trait + `RecurrenceFamily` enum (mechanics).
2. `LeakyIntegrator` (Family C — fast, byte-identical to `evolve_hla`).
3. `AttractorKernel` (Family A) — stays behind `micro_belief` as an **opt-in experiment**, NOT promoted (failed both G1.4 latency AND G2.1 quality).
4. `LatentThoughtKernel` (Family B) — correct but inherits the attractor's G2.1 loss; opt-in experiment.
5. `MicroRecurrentKernelSnapshot` (BLAKE3 freeze/thaw).
6. The bridge (`project_to_scalars`).

**NOT promoted:** the attractor family is not a GOAT. The plan's hypothesis (attractor hysteresis reduces long-horizon flip-flopping) does not hold at **random init** — the attractor's recurrent sigmoid nonlinearity with random Xavier weights is a *noisy* dynamical system that flip-flops more than a simple monotone leaky accumulator. The hysteresis property the plan anticipated would require **trained** weights that create actual fixed-point basins. This is an honest null result.

---

## G1.* Mechanics Gate Summary

The G1.* gates test the *mechanics* (determinism, boundedness, bridge correctness, latency, snapshot atomicity). All pass except G1.4 (latency).

| Gate | Test | Result | Notes |
|---|---|---|---|
| **G1.1** Determinism | `g1_1_determinism` + `g1_1_determinism_across_instances` | ✅ PASS | Bit-identical `s_T` for fixed `(s_0, x_1..x_T)` across runs and across kernel instances with the same seed. |
| **G1.2** Boundedness | `g1_2_boundedness_attractor` + `g1_2_boundedness_extreme_input` | ✅ PASS | `‖s_t‖` stays in `(-1, 1)` over 10 000 random inputs AND 100 extreme `±1e30` inputs. State is finite. |
| **G1.3** Bridge ranking | `g1_3_bridge_ranking_preservation` | ✅ PASS | `dot_a > dot_b ⟺ σ(dot_a) > σ(dot_b)` over 1000 random triples. |
| **G1.4** Latency | `g1_4_attractor_step_32_under_100ns` | ❌ **FAIL** | **273.2 ns/step** in release (Apple Silicon arm64), target <100 ns. Originally tracked in Issue 024 (`024_micro_belief_g1_4_attractor_latency`, closed + removed; this benchmark is the canonical record). Root cause: 32 scalar `fast_sigmoid` calls + 64 small-dim `simd_dot_f32` calls. Does NOT block Phase 1 exit. |
| **G1.5** Snapshot atomicity | `g1_5_snapshot_atomicity` | ✅ PASS | 4 reader threads × 50 000 steps, swapper hot-swaps the kernel every 100 µs. No reader ever sees NaN / Inf / torn state. |
| **G1.6** K=1 ≡ Family A | `latent_thought::tests::k_equals_one_is_bit_identical_to_attractor` | ✅ PASS | `LatentThoughtKernel(seed=42, dim=16, k=1)` produces byte-identical state to `AttractorKernel(seed=42, dim=16)` over a 100-step sequence. |

**Test command (passing):**
```
cargo test -p katgpt-core --no-default-features --features sparse_mlp,micro_belief,temporal_deriv --lib micro_belief
→ 45 passed; 0 failed
```

**Build without `micro_belief` (passing):**
```
cargo build -p katgpt-core --no-default-features --features sparse_mlp,temporal_deriv
→ Finished
```

---

## G2.1 Coherence Benchmark — The Actual GOAT Gate (T5.0)

**Setup** (see `micro_belief/coherence_bench.rs`):
- `dim = 16` (smaller than the G1.* tests' `dim = 32` to keep the 1000-step × 3-kernel run fast; documented in the module).
- Synthetic 1000-step input sequence, three phases:
  1. **Steps 0..400** — strong signal on dimension 0 (`input[0] = 0.8`, others `±0.05` noise). A good kernel settles into "dim 0 is dominant".
  2. **Steps 400..600** — ambiguous / noisy phase (all dims `±0.05` uniform noise, no dominant signal). This is the "bank" polysemy analog: a kernel with hysteresis should HOLD its belief; a flip-floppy kernel oscillates.
  3. **Steps 600..1000** — strong signal on dimension 1 (`input[1] = 0.8`, others `±0.05` noise). A good kernel transitions cleanly (ideally one flip).
- Noise is deterministic (`fastrand::Rng::with_seed(0xC0FFEE)`) so the benchmark is reproducible.
- Direction matrix = identity, so `out[k] = sigmoid(state[k])` and `argmax(out) == argmax(state)`.
- **Flip-flop count** = number of ticks where `argmax(projected_scalars)` changes from the previous tick. **Lower = more coherent.**
- **Ambiguous-window argmax variance** = population variance of the argmax stream over steps 400..600. **Lower = more stable under ambiguous evidence.**

### Results

| Kernel | Flip-flops (lower=better) | Ambig-window argmax var | Diverged? |
|---|---|---|---|
| **`LeakyIntegrator` (Family C, HLA default)** | **1** | **0.0000** | no |
| `AttractorKernel` (Family A, seed=42) | 569 | 20.3618 | no |
| `LatentThoughtKernel` (Family B, K=3, seed=42) | 560 | 20.4439 | no |

Numbers are identical in debug and release (math is deterministic, no RNG inside the kernels).

### Verdict: G2.1 FAIL — demote attractor to Gain (T5.2)

The attractor family **flips 569× / 560× more than the leaky integrator**. This is a decisive loss, not a tie or a marginal win. Per Plan 276 T5.2: **the attractor family is DEMOTED to a Gain experiment**. Only the trait unification + `LeakyIntegrator` ship as promotable output.

### Why the attractor loses (honest root-cause analysis)

The plan's hypothesis was that the attractor's fixed-point basins would create hysteresis — beliefs resist noise until evidence accumulates, then flip cleanly. **This hypothesis assumed trained weights.** At **random Xavier init** (which is what `AttractorKernel::from_seed` produces):

1. The recurrent weight matrix `W_s` is a random `dim × dim` matrix scaled by `1/√dim`. Its eigenstructure does NOT create useful attractor basins — it creates a generic nonlinear dynamical system whose trajectory is sensitive to small input perturbations.
2. The sigmoid nonlinearity `2σ(W_s·s + W_x·x + b) − 1` amplifies small input differences into discrete argmax changes. In the ambiguous window (steps 400..600) where inputs are pure noise, the attractor's state gets kicked around the `(−1, 1)^dim` cube and the argmax flip-flops on almost every tick.
3. The leaky integrator, by contrast, is a **monotone additive accumulator** with a per-tick delta clamp (`max_delta = 0.2`). Once it has accumulated strong evidence on dim 0 (after phase 1), the small ambiguous-window noise (`±0.05`) is far below the clamp threshold and barely moves the state. It holds its belief almost perfectly (1 flip — the clean transition to dim 1 at step 600).

**The attractor's hysteresis property is real but it is a property of TRAINED attractor networks (Hopfield-style content-addressable memory), not of randomly-initialised ones.** To make the attractor competitive on G2.1, the recurrent weights would need to be trained (or hand-set) so that the target beliefs (dim 0 dominant, dim 1 dominant) correspond to actual stable fixed points of the dynamics. That training is out of scope for Plan 276 (which is training-free / freeze-thaw only) and belongs to a future plan if the commercial case for attractor-based NPC belief tracking is made.

### Why `LatentThoughtKernel` (K=3) ≈ `AttractorKernel` (K=1)

K=3 does 3× more attractor iterations per tick, so the state moves further along the attractor's vector field per tick. This makes each individual tick *more* decisive (the state commits harder to whatever basin it's currently in), but it does NOT change the underlying problem: the basins themselves are randomly placed. So K=3 flip-flops slightly less than K=1 (560 vs 569) but is still in the same regime — ~560× worse than leaky. Increasing K further would eventually saturate the state to fixed points and reduce flip-flops, but at the cost of making the kernel unable to track genuine evidence changes (it would get stuck in the first basin it hits). This is the classic stability-plasticity tradeoff, and at random init the attractor family is on the wrong end of it.

---

## Decision Matrix (T5.1 vs T5.2)

Per Plan 276 Phase 5:

| Outcome | Action | Applies? |
|---|---|---|
| **T5.1** Attractor wins (strictly less flip-flopping) | Promote `micro_belief_attractor` as opt-in variant (NOT default — HLA leaky is battle-tested). | ❌ NO — attractor lost. |
| **T5.2** Attractor ties or loses | Demote to Gain. Keep trait unification + `LeakyIntegrator` as the only shippable output. Attractor stays behind `micro_belief` sub-flag for experimentation. | ✅ **YES — this is the path taken.** |

### Compounding factor: G1.4 latency

Even if G2.1 had passed, the attractor family could NOT be promoted to default-on because **G1.4 latency fails** (~273 ns/step vs <100 ns target, see Issue 024). The leaky integrator is an order of magnitude faster because it is an elementwise update (no matvec). So the attractor's promotion ceiling was always "opt-in variant at best" — and G2.1 removes even that. The attractor family stays as an opt-in experiment behind `micro_belief` for future research (e.g. trained-weight attractors, ANE batch dispatch where the matvec cost is amortised).

---

## What Ships

| Artifact | Default-on? | Status |
|---|---|---|
| `MicroRecurrentBeliefState` trait + `RecurrenceFamily` enum | (trait, no flag) | ✅ Ships (behind `micro_belief` feature). |
| `LeakyIntegrator` (Family C) | candidate for default-on after Phase 2 refactor | ✅ Ships. Fast, correct, byte-identical to `evolve_hla`. |
| `AttractorKernel` (Family A) | ❌ NO — opt-in experiment only | ⚠️ Stays behind `micro_belief`. Failed G1.4 (latency) AND G2.1 (quality). |
| `LatentThoughtKernel` (Family B) | ❌ NO — opt-in experiment only | ⚠️ Stays behind `micro_belief`. Inherits attractor's losses. |
| `MicroRecurrentKernelSnapshot` (BLAKE3) | (mechanics) | ✅ Ships. |
| Bridge `project_to_scalars` | (mechanics) | ✅ Ships. |

### Outstanding Plan 276 items (not blocked by G2.1)

- **T2.1–T2.3** — refactor `ReconstructionState::evolve_hla` to delegate to `LeakyIntegrator::step` (zero-behavior-change). ✅ **DONE** in commit `3eae61d3` (2026-06-16).
- **T1.14** — canonical criterion bench for G1.4. ✅ **DONE** in commit (this one) — see §G1.4 Criterion Bench below. Informational only; G1.4 still FAILs at ~270 ns.
- **Phase 4** — docs + examples. ✅ **DONE** in commit `3eae61d3`.

---

## G1.4 Criterion Bench (T1.14 — canonical numbers)

**Harness:** `crates/katgpt-core/benches/micro_belief_bench.rs`, criterion 0.5, sample_size=500 (100 for batch).
**Run:**
```bash
cargo bench -p katgpt-core --bench micro_belief_bench --features micro_belief
```

| Bench | Median | 95% CI | Target | Verdict |
|---|---|---|---|---|
| `g1_4_step/attractor_dim32` | **270.47 ns** | [270.26, 270.67] | <100 ns | ❌ FAIL (confirms Issue 024) |
| `g1_4_step/leaky_dim32` | **35.73 ns** | [35.68, 35.78] | <30 ns (HLA ref) | ⚠️ ~5 ns over HLA ref; well under 100 ns target. promotable. |
| `g1_4_step/latent_thought_k1_dim32` | **270.86 ns** | [270.52, 271.21] | attractor ±5% | ✅ PASS — within 0.15% of attractor (G1.6 latency analogue). |
| `g1_4_step/latent_thought_k3_dim32` | **811.46 ns** | [810.20, 812.72] | ~3× attractor | ✅ PASS — exactly 3.00× (270.47×3=811.4). |
| `project_to_scalars/k5_dim32` | **22.34 ns** | [22.32, 22.36] | <50 ns | ✅ PASS. |
| `batch_1000_npcs/leaky_serial_iter_dim8` | **11.34 µs** | [11.16, 11.57] | <15 µs | ✅ PASS — near the 10 µs aspirational target. |
| `batch_1000_npcs/leaky_rayon_par_iter_dim8` | **139.27 µs** | [96.25, 215.31] | — | ❌ rayon LOSES — see note below. |

### Why rayon loses to serial at this work size

At ~10 ns per leaky step (extrapolated from the 35.73 ns dim=32 measurement, dim=8 is ~3.6× less work), 1000 NPCs is ~10 µs of useful work — **500× below rayon's ~5 µs thread-pool spin-up breakeven** (AGENTS.md: "only parallelize when per-task work exceeds thread-pool overhead"). The Mutex lock acquisition per criterion sample adds further constant overhead. The rayon variant is kept in the bench **intentionally** to document this finding: at the per-NPC work size the leaky kernel operates at, serial iteration is the correct tool. Parallelism would only win at much larger per-NPC work (e.g. attractor family at dim=32, or batch sizes >100k NPCs).

### Leaky vs HLA baseline

The leaky integrator at dim=32 measures **35.73 ns/step**, vs the HLA baseline (`evolve_hla_simd` at dim=8) reference of **~30 ns/step** (Issue 024). The ~5 ns gap is the trait-dispatch + precomputed-`total` indirection added by the Phase 2 refactor (`leaky_core::leaky_step`). This is acceptable — the leaky path is still well under the 100 ns G1.4 target and the indirection is the price of DRY (one math body, two callers).

---

## Reproducibility

```bash
# G1.* + G1.6 + G2.1 (45 tests, all pass):
cargo test -p katgpt-core --no-default-features \
  --features sparse_mlp,micro_belief,temporal_deriv --lib micro_belief -- --nocapture

# Build without micro_belief (verifies no ungated code changed):
cargo build -p katgpt-core --no-default-features --features sparse_mlp,temporal_deriv

# Release-mode G2.1 numbers (identical to debug — math is deterministic):
cargo test -p katgpt-core --no-default-features \
  --features sparse_mlp,micro_belief,temporal_deriv --release --lib micro_belief -- --nocapture
```

**Machine:** Apple Silicon arm64, macOS.
**G1.4 release (criterion, T1.14):** attractor 270.47 ns/step; leaky 35.73 ns/step.
**G1.4 release (wall-clock test, superseded by criterion):** 273.2 ns/step.
**G2.1 flip-flops (debug == release):** leaky=1, attractor=569, latent_thought=560.

---

## Cross-references

- **Plan:** [276_micro_recurrent_belief_state.md](../.plans/276_micro_recurrent_belief_state.md) — Phases 1, 3, 5.
- **Issue:** `024_micro_belief_g1_4_attractor_latency` — G1.4 latency failure (~270 ns/step) (closed + removed; this benchmark is the canonical record).
- **Research:** [242_Topological_State_Tracking_Recurrent_Belief.md](../.research/242_Topological_State_Tracking_Recurrent_Belief.md)
- **Source paper:** [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer et al., DeepMind, Jun 2026.
- **Code:**
  - `katgpt-rs/crates/katgpt-core/src/micro_belief/coherence_bench.rs` — G2.1 harness.
  - `katgpt-rs/crates/katgpt-core/src/micro_belief/latent_thought.rs` — Family B + G1.6.
  - `katgpt-rs/crates/katgpt-core/src/micro_belief/tests.rs` — G1.1–G1.5.

---

## TL;DR

**G2.1 FAIL.** Attractor family flips **569× (K=1) / 560× (K=3)** more than the leaky integrator (**1 flip**) on the 1000-step coherence benchmark. The plan's hysteresis hypothesis does not hold at random init — it would require trained weights. Per T5.2: **attractor DEMOTED to Gain**; trait unification + `LeakyIntegrator` are the only promotable outputs. Combined with the pre-existing G1.4 latency failure (~273 ns/step), the attractor family stays behind `micro_belief` as an opt-in experiment. All 45 micro_belief tests pass (G1.1–G1.3, G1.5, G1.6 + the G2.1 informational test).
