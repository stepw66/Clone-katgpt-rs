# SwiR Switch-Thinking — Phase 3 GOAT Gate Results

**Plan:** 275 (Phase 3, tasks T3.2–T3.11)
**Test file:** `tests/bench_275_swir_goat.rs` (16 tests)
**Benchmark report:** `.benchmarks/275_swir_switch_thinking_goat.md`
**Profile:** release (G3 enforced) + debug (G7 allocation audit)
**Hardware:** Apple Silicon arm64 (NEON SIMD), Rust 1.93.0
**Model dependency:** None — all gates run on synthetic entropy streams + synthetic embedding matrices. Real-model gates (G1 accuracy, G2 efficiency, T3.9 accuracy ablations) are deferred to riir-ai Plan 313 (SwiR Real-Model Validation), which has the model loader + MATH500 harness. The **benchmark harness** (`src/swir/bench.rs`) ships in katgpt-rs with `ProblemSource` + `DecodeBackend` traits — riir-ai implements them over Gemma 2 2B IT + MATH500.

> **Real-model update (riir-ai Plan 313, 2026-06-19):** G2 = **1.37× (GATE PASS)** at `w_e_to_l=32, c_max=64` (n=5); G1 = 0% (blocked by Gemma 2 2B capability — T4.2e ruled out prompt/checker bugs). See [`riir-ai/.benchmarks/313_swir_real_model_goat.md`](../../../riir-ai/.benchmarks/313_swir_real_model_goat.md).

## Reproduce

```bash
# All synthetic gates (debug — fast, includes allocation audit)
cargo test --features swir_switch_thinking --test bench_275_swir_goat -- --nocapture

# G3 perf gate (release — the actual 200ns budget is release-mode)
cargo test --release --features swir_switch_thinking --test bench_275_swir_goat g3_step_perf -- --nocapture

# G5 isolation: swir code must NOT compile-link when feature is off
cargo check --no-default-features --features thinking_cot
```

## Gate verdicts

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1c** controller correctness | Latent→Explicit switches, convergence at ½c_max, termination above c_max | 6 switches, 3 CloseThink, 1 ForceAnswerPrefix, terminated at step 21 | ✅ PASS |
| **G2p** efficiency proxy | SwiR terminates < ½ fixed budget on switching schedule | 33 steps vs 1024 = **31.03× fewer steps** (97% reduction) | ✅ PASS |
| **G3** step() perf | < 200 ns/call (release) | **3.1 ns/call** (release), 28.0 ns (debug) | ✅ PASS (64× margin) |
| **G4** convex hull | 1000 random probs all in vocab hull | 1000/1000 in hull (100.00%) | ✅ PASS |
| **G5** feature isolation | swir code absent without feature | `cargo check --no-default-features --features thinking_cot` clean | ✅ PASS |
| **G6** kurtosis auto-fallback | High kurtosis forces Explicit mode | kurtosis=5.0 > threshold=3.0 → forced Explicit | ✅ PASS |
| **G7** zero-alloc step() | 0 allocations in `step()` (debug) | 0 allocs, 0 bytes over 1023 steps | ✅ PASS |

> **G7 parallel-safe (resolved 2026-06-16):** `src/alloc.rs` now uses thread-local `Cell<AllocStats>` counters instead of process-global atomics, so each test thread's allocation measurements are isolated from sibling tests. `g7_step_zero_allocation_debug` passes reliably under default parallel execution (verified 5 consecutive runs). The previous `--test-threads=1` pin is no longer required.
| **G8** signal-mix schedule | α_t/β_t monotonic non-decreasing in step_index | [0.70, 0.72, 0.74, 0.78, 0.85, 0.93, 1.0] — monotonic ✓ | ✅ PASS |
| **G9** hyperparameter ablation | W_E→L/C_max/α_0/mixing behavioral response matches paper expectations | W_E→L: 256→1 switches; C_max: term 27→117; α_0: identical 13 switches across 0.3–1.0; mix: fires only on switch steps | ✅ PASS |
| **G1h** accuracy gate harness | `run_benchmark` produces correct metrics; `ComparisonResult::accuracy_delta_pp` computes | harness structure validated, 10 problems × 2 modes | ✅ PASS |
| **G2h** efficiency gate harness | SwiR terminates < baseline; efficiency ratio > 1.0 | SwiR terminates at step 48 vs baseline 64 = 1.33× | ✅ PASS |

**All 11 synthetic-data gates PASS.**

### Deferred to riir-ai Plan 313 (needs real model) — ✅ G2 PASS, G1 blocked by model capability

| Gate | Target | Why deferred |
|------|--------|--------------|
| **G1** accuracy | +1.5pp on MATH500 vs `thinking_cot` baseline | Needs a real model (Gemma 2 / Qwen3) + MATH500 dataset + inference loop. katgpt-rs is the public MIT engine — no model loader. riir-ai has the runtime. |
| **G2** efficiency | 1.3× token efficiency at fixed accuracy | Same — needs real decoding to measure actual token counts at matched accuracy. |
| **T3.9** ablations | W_E→L, α_0, C_max, signal mix sweeps | Same — accuracy ablations need a real task. |

## Algorithmic bugs fixed during Phase 3

### Bug 1: Switch-count guard livelock (fixed earlier)

The original code enqueued `CloseThink`/`ForceAnswerPrefix` on *every step* while in Explicit mode with switch_count in the convergence window. This caused a livelock: the enqueued token was drained at the start of the next step (skipping mode-switch logic), which prevented switch_count from advancing, which re-enqueued the same token forever.

**Fix:** only enqueue when `switched_to == Some(ThinkMode::Explicit)` — i.e., on the step where the Latent→Explicit switch *just happened*. This matches the paper's intent (§3.4 describes switch-count thresholds, not continuous conditions) and fires each guard exactly once per switch event.

Before fix: G2p ran 1024 steps without terminating (0% reduction).
After fix: G2p terminates at step 33 (97% reduction, 31.03× speedup) with the REAL `c_convergence_fraction=0.5` (no workaround).

### Bug 2: Answer-budget countdown allows mode switches (fixed 2026-06-17)

After `ForceAnswerPrefix` fires (switch_count > c_max), the answer-budget countdown began but the mode-switch logic still ran. On alternating entropy schedules, this caused spurious Latent→Explicit switches during the answer window, inflating switch_count far past c_max. `g9b_c_max_sweep_termination_step_scales_monotonically` caught this (C_max=2 produced 7 switches, expected ≤ 4).

**Fix:** when `answer_budget_remaining` is `Some` (post-ForceAnswerPrefix), skip mode-switch logic entirely and emit `EmitToken(0)` directly. The paper's ForceAnswerPrefix means "stop reasoning, start answering" — allowing further mode switches would defeat the overthinking guard (paper §3.4).

Before fix: `g9b` FAILED (7 switches for C_max=2).
After fix: `g9b` PASSES (switch count correctly bounded by c_max + small slop for injection steps).

## G7 adapter allocation note

The `SwiRStrategyAdapter::on_step` path allocates ~2× per step in debug builds:
- 1 clone of the soft-embedding result (embedding_dim × 4 bytes = 128 bytes for dim=32)
- 1 allocation tracked internally by the allocator harness

This is documented in Plan 275 T2.2 ("soft-embedding clone + InjectTokens Vec"). The `step()` path itself (G7 step variant) is zero-allocation. The adapter allocations are bounded and acceptable for the thinking-cot integration; a future optimization could pass a scratch buffer if the adapter becomes a measured hot path.

## Decision

**Keep `swir_switch_thinking` OPT-IN** (default-off) until riir-ai Plan 313 confirms the G2 gate at n=20+ (currently 1.37× at n=5, gate target 1.3× — see the real-model update banner above). The algorithmic invariants (G3–G8, G1c, G2p) all pass on synthetic data — the controller is correct by construction. G2 has now passed on a real model (Gemma 2 2B); G1 remains blocked by model capability (needs Qwen3-4B/8B).
