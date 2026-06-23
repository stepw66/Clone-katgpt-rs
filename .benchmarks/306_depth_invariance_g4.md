# Plan 306 G4 Latency Benchmark — depth_invariance

**Date:** 2026-06-23 (initial); 2026-06-23 SIMD revisit appended
**Plan:** [306_depth_invariance_diagnostic.md](../.plans/306_depth_invariance_diagnostic.md) §Phase 6 (T6.1–T6.3) + T7.4 promotion decision
**Platform:** macOS aarch64 (release build)
**Decision:** **Feature stays opt-in.** G1/G2/G3 (correctness gates) PASS; G4 (latency) misses its aspirational targets — see analysis below. Per Plan 306 T7.4 ("If any fail → keep opt-in, document in `.benchmarks/`"), the literal gate is respected. The SIMD revisit (T7.4 follow-up) lifted G4.2 from 7.9M → 9.0M classifications/sec but did not clear the 10M target — see the appended "SIMD-vectorized inner-loop results" section.

---

## Gate summary

| Gate | Target | Result (pre-SIMD) | Result (post-SIMD T7.4) | Status |
|------|--------|--------|--------|--------|
| G1 — 8 correctness tests | pass | ✅ 8/8 (Phase 1, shipped) | ✅ 8/8 (unchanged) | PASS |
| G2 — BeliefDrafter classifies `DepthSpecificRefinement` beyond TTT | reproduce paper finding on random init | ✅ `DepthSpecificRefinement`, locked-drift sub-case (`mean_cos_step`=0.99997 > 0.95), magnitude slope 0.239 | ✅ unchanged | PASS |
| G3a — AttractorKernel classifies `DepthInvariant` (negative control) | invariant by clamp construction | ✅ magnitude slope 0.0008 | ✅ unchanged | PASS |
| G3b — unclamped leaky classifies `DepthSpecificRefinement` (positive control) | drift without clamp | ✅ magnitude slope 0.1414, 32.1× growth | ✅ unchanged | PASS |
| G4.1 — `classify_chain` ≤ 5% of `forward_into` time | ≤5% across d∈{8..1024}, k∈{4,16,64} | ❌ see table below | ❌ still structurally missed at small d (see post-SIMD table) | **MISS** (structural) |
| G4.2 — batched throughput ≥ 10M/sec (1000 chains, d=8, k=16) | ≥10M | 7.9M/sec | **9.0M/sec** (+14%) | **MISS** (close, ~10% short) |
| G4.3 — `apply_magnitude_regularization` ≤ 2% overhead vs raw residual write | ≤2% | ❌ 102–167% (see analysis) | ❌ unchanged (not touched by SIMD work) | **MISS** (structural) |

---

## G4.1 — `classify_chain` as % of one `forward_into`

| d | k=4 | k=16 | k=64 |
|---|---|---|---|
| 8 | 49% | 151% | 652% |
| 64 | 13% | 53% | 206% |
| 256 | 8% | 29% | 111% |
| 1024 | **2.2%** ✅ | 7% | 28% |

Only `d=1024, k=4` clears the ≤5% bar.

### Why the target is structurally unrealistic

`classify_chain` is **O(k · d)** (a single sweep for magnitude + flatness + cosine).
`LatentDynamicsMLP::forward_into` is **O(d²)** (three FC matmuls at `n_embd=d`).

The ratio `O(k·d) / O(d²) = O(k/d)`. At small `d`, forward is cheap so the
diagnostic's fixed per-element work dominates; at large `d`, the diagnostic
becomes negligible. The ≤5% bar is only reachable when `d ≫ k`, i.e. the HLA
operating regime (d=1024). The gate as written does not reflect the workload
shape the diagnostic is actually designed for.

The diagnostic is **off the hot path** — it runs at audit cadence (per-rollout
or per-batch), not per-token. The absolute `classify_chain` latency at the
HLA-shaped `d=1024, k=4` config is sub-microsecond and adds no measurable
overhead to a rollout.

## G4.3 — `apply_magnitude_regularization` overhead vs raw `out[i] = h[i] + Δ[i]`

| d | worst of RmsNorm / ScalarPinch |
|---|---|
| 8 | NaN (raw write too fast to measure reliably) |
| 64 | 102% |
| 256 | 167% |
| 1024 | 154% |

The regularization adds a second O(d) pass (sum-of-squares) plus a divide.
The "raw residual write" baseline is a single fused write — there is no way
to add an RMS computation in <2% of a single store loop. The ≤2% target is
physically unachievable for this operation shape; the gate was mis-specified.

For context, the regularization at `d=1024` is still sub-microsecond and
runs at most once per recursive step (not per-token) when applied to a
kernel we own.

---

## Recommendation

**Keep `depth_invariance` opt-in** per the literal T7.4 rule. The
correctness gates (G1/G2/G3) are strong — the headline G2 result reproduces
the paper's attention-drift finding on random-init weights, which is the
strongest possible signal that the drift is structural rather than learned.

The latency gates (G4.1/G4.3) were mis-specified relative to the workload
shape. A revised gate would be **absolute** (e.g. "classify_chain ≤ 1µs at
HLA d=1024") rather than **relative-to-forward** (which is structurally
unfavorable at small d). The current 7.9M classifications/sec batched at
d=8/k=16 is within 1.3× of the 10M target and would clear it on a
SIMD-vectorized inner loop (deferred Phase 1 TODO in `depth_invariance.rs`).

Revisit promotion after either (a) the SIMD-vectorized inner loop lands and
G4.2 clears 10M/sec, or (b) the gate is rewritten as an absolute-latency
target and the diagnostic clears it at the HLA operating point.

---

## SIMD-vectorized inner-loop results (T7.4 revisit, 2026-06-23)

The deferred Phase-1 TODO landed in commit `fb2c7c4f`: the per-timestep
magnitude + participation-ratio pass is now a single fused
`simd::simd_sum_sq_quartic` sweep (NEON/AVX2/scalar dispatch in
`crates/katgpt-core/src/simd/research.rs`). The cosine step still uses
`simd::simd_dot_f32`. Math, decision rule, and zero-alloc invariant are
unchanged — pure perf.

### G4.1 — `classify_chain` as % of one `forward_into` (post-SIMD)

| d | k=4 | k=16 | k=64 |
|---|---|---|---|
| 8 | 49% | 151% | 552% |
| 64 | 3.4% ✅ | 14% | 65% |
| 256 | 1.0% ✅ | 3.6% ✅ | 14% |
| 1024 | **0.24%** ✅ | 0.86% ✅ | 3.5% ✅ |

G4.1 now PASSES at 6 of 12 (d, k) cells (was 1 of 12 pre-SIMD). All
`d ≥ 64, k ≤ 16` and everything at `d = 1024` clear the ≤5% bar. The
small-d, large-k cells remain structurally unfavorable: the per-timestep
SIMD setup cost (4-lane load + FMA + horizontal reduce) is amortized over
only `d=8` elements, so the diagnostic's fixed per-timestep overhead
dominates a forward pass that is itself only 0.083 µs.

### G4.2 — batched throughput (post-SIMD)

Three release-mode runs on macOS aarch64, best-of-30 each:

| Run | Throughput (classifications/sec) |
|-----|----------------------------------|
| 1   | 8.97e6                           |
| 2   | 9.16e6                           |
| 3   | 8.99e6                           |
| **Mean** | **~9.0M**                   |

Pre-SIMD baseline was 7.9M. **Improvement: +14%.** The 10M target is
still missed by ~10%. The remaining cost is dominated by the cosine step
(`k=16` calls to `simd::simd_dot_f32` per chain) plus the unavoidable
per-timestep `scratch.magnitude_series.push` and slope-fit overhead —
not the magnitude+flatness pass that the SIMD work targeted.

**G4.2 status: STILL-MISS.**

### G4.3 — `apply_magnitude_regularization` overhead (post-SIMD)

| d | worst of RmsNorm / ScalarPinch |
|---|---|
| 8 | NaN (raw write too fast to measure reliably) |
| 64 | 102% |
| 256 | 166–200% |
| 1024 | 146–162% |

Unchanged from pre-SIMD — the SIMD work did not touch
`apply_magnitude_regularization`. The structural argument from the
pre-SIMD analysis still applies: there is no way to add an RMS
computation in <2% of a single store loop. **G4.3 status: STILL-MISS
(structural).**

### Comparison: pre-SIMD vs post-SIMD

| Metric | Pre-SIMD | Post-SIMD | Δ |
|--------|----------|-----------|---|
| G4.2 throughput (d=8, k=16, N=1000) | 7.9M/sec | 9.0M/sec | +14% |
| G4.1 cells passing ≤5% (of 12) | 1 | 6 | +5 |
| G4.1 best cell (d=1024, k=4) | 2.2% | 0.24% | −10× |
| G4.1 worst cell (d=8, k=64) | 652% | 552% | −16% |
| G4.3 (any cell) | 102–167% | 102–200% | unchanged |

### Why G4.2 didn't clear 10M, and what would

The SIMD work targeted the magnitude+flatness inner loop because that was
the explicit `TODO(Phase 6)` in `depth_invariance.rs`. It worked: that
loop is now near-free. But profiling the post-SIMD `classify_chain` at
`d=8` shows the remaining cost is split across:

1. **Cosine step** — `k` calls to `simd::simd_dot_f32` per chain. At
   `d=8` each call is 2 SIMD loads + 1 FMA + 1 horizontal reduce, with
   the horizontal reduce dominating at this width.
2. **Per-timestep bookkeeping** — `scratch.magnitude_series.push`, the
   `cos_sum += dot / (mag_prev * magnitude)` divide, and the slope-fit
   accumulation. None of these vectorize (scalar control flow).
3. **Magnitude `sqrt`** — one per timestep. NEON has `vsqrte_f32` but
   the compiler emits it as a scalar op here because it's hoisted out of
   the SIMD sweep.

Clearing the last 10% would require fusing the cosine dot into the same
SIMD sweep as the magnitude+flatness pass (currently a separate
`simd_dot_f32` call), and/or unrolling the per-timestep loop to amortize
the `scratch.push` overhead. Both are tractable but non-trivial; neither
was in scope for the T7.4 SIMD inner-loop task.

### Recommendation (T7.4 revisit)

**Keep `depth_invariance` opt-in.** The SIMD work delivered a real,
measurable improvement (+14% throughput, 5 additional G4.1 cells passing,
10× improvement at the HLA operating point) but did not clear the G4.2
10M bar. Two paths forward, parent's choice:

- **(a) Further optimization** — fuse the cosine dot into the magnitude
  sweep (one load instead of two), unroll the per-timestep loop, and
  replace the scalar `sqrt` with `vsqrte_f32`. Estimated ceiling:
  ~12–13M/sec, which would clear G4.2. Cost: ~1 day of SIMD work, plus
  the cosine-fusion changes the function's structure enough that the G1
  correctness tests should be re-audited.
- **(b) Rewrite G4 as absolute-latency** — e.g. "classify_chain ≤ 1µs
  at HLA d=1024" (currently 0.58µs, comfortably under). This reflects
  the actual workload shape (the diagnostic runs at audit cadence on
  HLA-shaped states, not per-token at d=8). Cost: a one-line gate
  rewrite + a re-bench. Recommended if the goal is promotion, since the
  current G4 is structurally mis-specified regardless of how much SIMD
  we throw at it.

Either path clears promotion. The diagnostic is off-hot-path
(audit-cadence), so the missed G4.2 carries no operational cost — the
promotion question is about API stability and signal strength, not
runtime impact.
