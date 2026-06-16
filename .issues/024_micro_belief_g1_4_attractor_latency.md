# Issue 024: MicroRecurrentBeliefState G1.4 Attractor Latency (>100ns/step)

**Date:** 2026-06-16
**Plan:** [katgpt-rs/.plans/276_micro_recurrent_belief_state.md](../.plans/276_micro_recurrent_belief_state.md) — Phase 1, T1.11 / T1.18
**Status:** Open — does NOT block Phase 1 exit (G1.1, G1.2, G1.3, G1.5 all pass). Blocks promotion of `micro_belief` to default-on.

---

## Symptom

`AttractorKernel::step()` at `dim=32` measures **~270 ns/step** in release on Apple Silicon arm64, exceeding the Plan 276 G1.4 target of **<100 ns/step**.

The HLA baseline (`ReconstructionState::evolve_hla_simd`) is an order of magnitude faster because it is a leaky integrator (elementwise updates, no matvec). The attractor family does a full `dim × dim` matvec plus 32 sigmoid calls, which is fundamentally more work.

## Root Cause (profiled)

1. **32× `fast_sigmoid` calls per step** — each calls `exp()` (~5ns each) = ~160ns just for sigmoids.
2. **64× `simd_dot_f32` calls of length 32** — at `dim=32`, the function-call overhead into the NEON/AVX2 dispatch dominates over the actual FMA work. The 4-row chunking helps ILP but doesn't eliminate per-call overhead.
3. **Stack buffer copy** — `[f32; 1024]` zero-init + final copy back to `state` adds ~20ns.

## Mitigations to Explore (in priority order)

### M1: Vectorize the sigmoid (highest impact)

Replace the 32 scalar `fast_sigmoid` calls with a single SIMD-vectorized sigmoid pass over the 32-element activation buffer. The `wide` crate or `std::simd` can compute 4–8 sigmoids in parallel via a Padé approximation:

```
σ(x) ≈ (0.5 + 0.25·x) for |x| < 2      // piecewise linear
σ(x) ≈ clamp(x·0.125 + 0.5, 0, 1)       // very rough
σ(x) = 1/(1+e^{-x})                      // exact, vectorize via exp_ps
```

The `exp_ps` intrinsic (NEON: `vexpq_f32`, AVX2: `_mm256_exp_ps` if available via `sleef` or manual Taylor) would give 4 sigmoids per FMA pipeline cycle. Expected: 32 sigmoids in ~20ns instead of ~160ns. Saves ~140ns.

### M2: Inline the dot products at small dim

At `dim=32`, the `simd_dot_f32` function-call overhead is significant. An inlined `[f32; 32]`-specialized dot (unrolled 8-wide) would eliminate the dispatch cost. Expected: ~50ns saved.

### M3: Fuse matvec + sigmoid + clamp into a single pass

Currently: compute all activations, then sigmoid all, then clamp all. Fusing into a single row-loop with inlined sigmoid (M1) avoids the intermediate buffer and the second pass.

### M4: Reduce dim

The plan defaults to `dim=32` to match Plan 255's L1 budget. If M1–M3 don't close the gap, consider `dim=16` as the attractor default (the leaky integrator stays at `dim=8` to match HLA). Halving dim roughly halves the matvec cost.

### M5: Accept the latency, demote attractor to sub-flag

If M1–M4 don't reach <100ns, keep `micro_belief` opt-in (do NOT promote to default). The trait unification + LeakyIntegrator wrapper (which IS fast) still ship as the only default-on output. The attractor family stays behind `micro_belief_attractor` sub-flag for experimentation. This is the plan's T1.17 fallback path.

## Benchmark Numbers (2026-06-16, Apple Silicon arm64, release)

| Variant | ns/step | Target | Verdict |
|---|---|---|---|
| `AttractorKernel::step()` dim=32 (current) | **270.47** (criterion median, T1.14) | <100 | FAIL |
| `LeakyIntegrator::step()` dim=32 (Family C, baseline) | **35.73** (criterion median, T1.14) | <100 | PASS |
| `LatentThoughtKernel::step()` dim=32 K=1 | **270.86** (criterion median, T1.14) | attractor ±5% | PASS (matches attractor) |
| `LatentThoughtKernel::step()` dim=32 K=3 | **811.46** (criterion median, T1.14) | ~3× attractor | PASS (exactly 3.00×) |
| `project_to_scalars` K=5 dim=32 | **22.34** (criterion median, T1.14) | <50 | PASS |
| `ReconstructionState::evolve_hla_simd()` dim=8 (HLA baseline) | ~30 | — | reference |

## Cross-References

- **Plan:** [276_micro_recurrent_belief_state.md](../.plans/276_micro_recurrent_belief_state.md) §Phase 1 T1.11, T1.18, R2
- **Research:** [242_Topological_State_Tracking_Recurrent_Belief.md](../.research/242_Topological_State_Tracking_Recurrent_Belief.md)
- **HLA baseline:** `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` `evolve_hla_simd` (L657–690)
- **SIMD primitives:** `katgpt-rs/crates/katgpt-core/src/simd.rs` `simd_dot_f32` (L100), `fast_sigmoid` (L1684)

## TL;DR

G1.4 latency gate fails for `AttractorKernel` at dim=32 (~270ns vs <100ns target). Root cause: 32 scalar sigmoid calls + 64 small-dim dot products with function-call overhead. Does NOT block Phase 1 exit — G1.1/G1.2/G1.3/G1.5 pass; the trait unification + LeakyIntegrator (which is fast) ship regardless. Attractor stays opt-in behind `micro_belief` until M1 (vectorized sigmoid) or M4 (dim=16) closes the gap. Filed per Plan 276 R2 mitigation.
