# Plan 168: RecFM Recursive Cross-Scale Consistency (Research 150)

**Source Research:** Research 150 — RecFM Recursive Cross-Scale Consistency for Modelless Inference
**Status:** Proposed
**Feature Gate:** `recfm` (opt-in initially, default-on after GOAT proof)
**Depends on:** Plan 066 (D2F), Plan 136 (LT2), Plan 131 (SpecHop)

---

## Goal

Implement recursive cross-scale consistency from RecFM as modelless inference improvements. Three components: DDTree branch consistency, LT2 acceleration bounding, SpecHop cross-hop scoring. D2F recursive denoising deferred pending benchmark.

---

## Tasks

### Task 1: Recursive DDTree Branch Consistency

- [ ] T1.1: Add `CrossScaleConfig` struct to `src/speculative/dd_tree.rs`
  - Fields: `enable: bool`, `scale_alpha: f32` (default 0.5), `consistency_threshold: f32` (default 0.1)
  - `#[repr(C)]` for predictable layout
- [ ] T1.2: Implement `branch_velocity_at(depth, marginal_slice) -> f32` — discrete velocity at depth
  - Computes: `marginal[depth] - marginal[depth-1]` (change in top-1 probability)
  - Zero-alloc: operate on existing marginal slices
- [ ] T1.3: Implement `cross_scale_consistent(v1, v2, alpha) -> bool` — check consistency
  - Returns `|v2 - alpha * v1| < threshold`
  - Inline, branch-free: `f32::abs(v2 - alpha * v1) <= threshold`
- [ ] T1.4: Integrate into `build_dd_tree_screened()` — filter branches that violate consistency
  - After computing marginals for each depth, check cross-scale consistency with parent
  - Only applies when `CrossScaleConfig::enable == true`
- [ ] T1.5: GOAT proof: `tests/test_recfm_ddtree.rs`
  - Proof 1: Cross-scale consistency reduces invalid branches (count check)
  - Proof 2: Best path unchanged when consistency threshold is loose (safety)
  - Proof 3: Zero allocation on hot path (benchmark with `--nocapture`)

### Task 2: Recursive LT2 Acceleration-Bounded Sub-Stepping

- [ ] T2.1: Add `AccelBoundConfig` to `src/tf_loop.rs`
  - Fields: `enable: bool`, `accel_threshold: f32` (default 0.5), `extra_damp_factor: f32` (default 0.8)
  - `#[repr(u8)]` for compact storage
- [ ] T2.2: Implement `simd_accel_norm(v_curr: &[f32], v_prev: &[f32]) -> f32`
  - Computes: `||v_curr - v_prev||₂ / dim` (normalized acceleration)
  - SIMD via existing `simd_fused_decay_write` infrastructure
- [ ] T2.3: Modify `sub_step_damped_euler()` to accept `AccelBoundConfig`
  - After computing sub-step, check acceleration norm
  - If exceeds threshold, apply additional damping: `x *= extra_damp_factor`
  - Only when `enable == true`
- [ ] T2.4: GOAT proof: `tests/test_recfm_lt2.rs`
  - Proof 1: Acceleration-bounded sub-steps produce smaller residuals
  - Proof 2: Output diverges less over K iterations vs vanilla damped Euler
  - Proof 3: SIMD accel_norm benchmark (< 100ns per call)

### Task 3: Recursive SpecHop Cross-Hop Consistency

- [ ] T3.1: Add `CrossHopConfig` to `src/spechop/speculator.rs`
  - Fields: `enable: bool`, `velocity_threshold: f32`, `min_hops_for_consistency: usize` (default 2)
- [ ] T3.2: Implement `observation_velocity(obs_k: &str, obs_k1: &str) -> f32`
  - Returns normalized Levenshtein distance between consecutive observations
  - Or use simple token overlap ratio for O(1) hot-path
- [ ] T3.3: Modify `BanditSpeculator::speculate()` to check cross-hop velocity
  - Track last N speculated observations
  - Check: velocity between hop k and k+1 should be ≤ velocity between k-1 and k (convergence)
  - Penalize confidence for non-converging hops
- [ ] T3.4: GOAT proof: `tests/test_recfm_spechop.rs`
  - Proof 1: Converging observations get higher confidence
  - Proof 2: Diverging observations get penalized
  - Proof 3: No regression in cache hit rate

### Task 4: Feature Gate + Integration

- [ ] T4.1: Add `recfm` feature flag to `Cargo.toml` (optional, depends on nothing new)
- [ ] T4.2: Add `recfm` to feature list in README under "Gated Features"
- [ ] T4.3: Ensure all three components are behind the same feature gate
- [ ] T4.4: Run full test suite with `--features recfm` — zero regressions

### Task 5: GOAT Proof + Default-On Decision

- [ ] T5.1: Run benchmark suite comparing with/without `recfm`
  - DDTree: branch validity rate, best-path quality
  - LT2: residual convergence, output quality
  - SpecHop: confidence calibration, acceptance rate
- [ ] T5.2: If gain + no perf hurt → move to default-on (remove feature gate)
- [ ] T5.3: If mixed → keep gated with documented tradeoffs
- [ ] T5.4: If no gain → mark as negative result in research doc

---

## Deferred

- **Recursive D2F**: Secondary denoising trajectory with cross-scale velocity blend. Requires 2× forward passes per step. Benchmark first to validate the cost-benefit tradeoff.
- **Model-based RecFM**: Training-time velocity consistency for LoRA. Belongs in riir-ai (Plan TBD).

---

## Implementation Notes

### Alignment with optimization.md

- **No allocation in hot path**: `CrossScaleConfig`, `AccelBoundConfig`, `CrossHopConfig` are all stack-allocated structs
- **SIMD reuse**: `simd_accel_norm` builds on existing `simd_fused_decay_write` infrastructure
- **Fixed-size arrays**: velocity buffers pre-allocated, reused across iterations
- **Benchmark before/after**: T5.1 compares vs baseline for all three components

### Feature Gate Structure

```toml
[features]
recfm = []  # recursive cross-scale consistency (DDTree + LT2 + SpecHop)
```

Single gate for all three components. If one proves negative, gate the individual component internally with its config `enable` flag.
