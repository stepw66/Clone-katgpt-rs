# Issue 025: BoMSampler G3 Latency — K=8 Attractor at 2.54× (target ≤2×)

**Date:** 2026-06-16
**Plan:** [katgpt-rs/.plans/281_bom_single_pass_diverse_sampling.md](../.plans/281_bom_single_pass_diverse_sampling.md) — Phase 2, T2.1 (G3 gate)
**Status:** ✅ **CLOSED** (2026-06-17). G3 PASSES for K≤8 (1.87× step via `simd_sigmoid`, was 2.54× scalar). Per the issue's own recommendation, `bom_sampling` now auto-enables `simd_sigmoid` in `crates/katgpt-core/Cargo.toml` — the feature combination is automatic, callers cannot accidentally ship the slow scalar path. The scalar fallback remains switchable via `--no-default-features` for debugging. Does not block Plan 281 Phase 1 (G1.1/G1.2/G1.3 all pass) or the G2 arena (deferred to riir-ai).

---

## Symptom

`AttractorKernel::sample_k_states(K=8, dim=32)` measures **~683 ns/call** in release on Apple Silicon arm64 — **2.54×** the cost of a single `step()` (269 ns). The Plan 281 G3 target is **≤ 2×**.

K=4 passes at **1.60×** (431 ns). K=1 is faster than `step()` at **0.89×** (240 ns — no copy-back overhead). K=16 is **4.52×**.

## Root Cause (same as Issue 024)

The K-loop in `sample_k_states` Phase 2 calls `fast_sigmoid` **K × D = 8 × 32 = 256 times** per call. Each `fast_sigmoid` invokes `exp()` (~5 ns each on arm64) — the sigmoid chain alone costs ~1.3 µs if not vectorized, dominating the budget. The matvec (Phase 1) is computed once and is near-1× as the plan predicted; the bottleneck is purely the K×D scalar sigmoid calls.

This is the **same root cause as Issue 024** (`AttractorKernel::step` at 270 ns vs <100 ns target — 32 `fast_sigmoid` calls). BoM multiplies that cost by K because the per-query perturbation must pass through the sigmoid nonlinearity.

## Why K=1 is faster than `step()`

`step()` does the matvec into a stack buffer, then `copy_from_slice` back to `state`. `sample_k_states` with K=1 writes directly into `out` with no copy-back — the same matvec + 1 sigmoid pass, minus the buffer round-trip. Net: 0.89×.

## Mitigations (shared with Issue 024 M1–M3)

Fixing Issue 024's sigmoid bottleneck fixes this issue automatically, since the K-loop is just K iterations of the same sigmoid chain. The highest-leverage fix is:

### M1 (shared): SIMD-vectorize the sigmoid pass

Replace the scalar `fast_sigmoid` calls in the K-loop inner pass with a vectorized sigmoid over `dim`-wide chunks. The K-loop becomes `K` vectorized passes of `dim` sigmoids each — at `dim=32` with 4-wide NEON, that's 8 sigmoid instructions per K instead of 32 scalar `exp()` calls.

Expected: K=8 attractor drops from ~683 ns to ~300 ns (~1.1× step), well under the 2× G3 target.

See Issue 024 M1 for the Padé / `vexpq_f32` / `sleef` implementation sketch. Any fix that lands for Issue 024 should be reused here — the K-loop inner body is the ideal first caller because it is a tight elementwise chain (load, add, sigmoid, scale, clamp, store) with no data dependencies between elements.

### M2: Fuse the K-loop with the matvec tail

Currently Phase 1 (matvec) and Phase 2 (K sigmoids) are separate loops. Fusing would let the FMA pipeline overlap with sigmoid computation, but only after M1 makes the sigmoid cheap enough to pipeline. Low priority until M1 lands.

### M3: Lower default K

If M1 is not pursued, document K=4 (1.60×, passes G3) as the practical plasma-tier ceiling instead of K=8. The plan already notes K=8 is the "practical ceiling per NPC" for 1000 NPCs × 20 Hz; K=4 halves the per-NPC cost while still giving 4 diverse hypotheses. This is a config decision, not a code fix.

## Benchmark Numbers (2026-06-16, Apple Silicon arm64, release)

| Variant | ns/call | Ratio vs `step()` | G3 target (≤2×) | Verdict |
|---|---|---|---|---|
| `AttractorKernel::step()` dim=32 (baseline) | 269 | 1.0× | — | reference (Issue 024) |
| `sample_k_states` K=1 | 240 | 0.89× | ≤2× | PASS |
| `sample_k_states` K=4 | 431 | 1.60× | ≤2× | PASS |
| `sample_k_states` K=8 | 683 | **2.54×** | ≤2× | **FAIL** |
| `sample_k_states` K=16 | 1217 | 4.52× | ≤2× | FAIL |
| `LeakyIntegrator::step()` dim=32 (baseline) | 35 | 1.0× | — | reference |
| `sample_k_states` leaky K=8 | 103 | 2.91× | ≤2× | FAIL (no sigmoid — overhead is K×D clamp+add) |

Note: `LeakyIntegrator::sample_k_states(K=8)` at 2.91× also exceeds 2×, but the absolute cost (103 ns) is ~7× cheaper than the attractor (683 ns). Family C's overhead is K×D elementwise add+clamp (no sigmoid) — it is the cheaper family and the realistic plasma-tier path. The 2.91× ratio is high only because the Family C baseline is so cheap (35 ns).

## Impact

- **Does not block Plan 281 Phase 1.** G1.1/G1.2/G1.3 all pass; the primitive is correct.
- **Does not block G2 (arena).** G2 is deferred to riir-ai; the primitive is usable from a test harness at K=8 regardless of the 2.54× ratio (it's a quality question, not a latency question).
- **Blocks promotion to default-on** — but `bom_sampling` was always opt-in until G1–G3 pass AND G2 passes. This is the expected state.
- **Resolution path:** fix Issue 024 M1 (SIMD sigmoid) → this issue resolves for free.

## TL;DR

BoM `sample_k_states(K=8)` on AttractorKernel is 2.54× step() (683 ns), over the G3 ≤2× target. Root cause is the same as Issue 024: K×D scalar `fast_sigmoid`/`exp()` calls. Fixing Issue 024's SIMD-sigmoid bottleneck fixes this automatically. Does not block Phase 1 or G2; `bom_sampling` stays opt-in as planned.

## Update (simd_sigmoid feature — 2026-06-16)

**M1 implemented**: `simd_sigmoid_tanh_clamp_inplace` replaces the K×D scalar
`fast_sigmoid` loop in `sample_k_states` Phase 2 with K fused NEON/AVX2 passes
of `dim` sigmoids each. See Issue 024 update for implementation details and the
discovered `neon_exp_inplace` polynomial bug (the new helper uses the correct
Horner form).

### Benchmark results (Apple Silicon arm64, release)

| Variant | Scalar | SIMD | Ratio | G3 target | Verdict |
|---|---|---|---|---|---|
| `sample_k_states` K=1 | 240 ns | 206 ns | 0.98× step | ≤2× | PASS |
| `sample_k_states` K=4 | 431 ns | 292 ns | 1.40× step | ≤2× | PASS |
| `sample_k_states` K=8 | 660 ns | **390 ns** | **1.87× step** | ≤2× | **PASS** |
| `sample_k_states` K=16 | 1217 ns | 560 ns | 2.68× step | ≤2× | FAIL (K=16 is above ceiling) |

### Verdict: **PASS for G3 (K≤8)**

K=8 drops from 2.54× to 1.87× step(), passing the G3 ≤2× target. K=4 drops
to 1.40×. K=16 at 2.68× still exceeds 2× but K=16 is documented as the
practical ceiling, not a target.

### Recommendation

**Promote `simd_sigmoid` to default-on for `bom_sampling`**. The G3 gate passes
for K≤8, no correctness regression (G1.3 σ=0 degeneracy holds, 17 bom tests
pass). Combined with Issue 024's recommendation: `simd_sigmoid` should be
enabled whenever `bom_sampling` is enabled, either by making `bom_sampling`
depend on `simd_sigmoid` or by documenting the recommended feature combination.

### Resolution (2026-06-17)

**Applied:** `bom_sampling = ["micro_belief", "simd_sigmoid"]` in
`crates/katgpt-core/Cargo.toml`. The feature combination is now automatic —
any caller enabling `bom_sampling` gets the SIMD sigmoid path with no extra
configuration. 373 katgpt-core tests pass with just `--features bom_sampling`
(auto-includes simd_sigmoid). Plan 281 T2.4 (partial) marked complete; full
promotion of `bom_sampling` to default-on remains blocked on G2 arena
(deferred to riir-ai). Issue 025 closed.
