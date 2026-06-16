# Plan 277: Temporal Derivative Kernel — Phase 1 GOAT Gate Results

**Date:** 2026-06-16
**Plan:** [katgpt-rs/.plans/277_temporal_derivative_kernel.md](../.plans/277_temporal_derivative_kernel.md)
**Research:** [katgpt-rs/.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md)
**Source paper:** [arXiv:2606.08720](https://arxiv.org/abs/2606.08720) — O'Reilly 2026, "This is how the Neocortex Learns"
**Hardware:** Apple Silicon arm64 (M-series), release build (`cargo bench --release`).

---

## Phase 1 — Primitive Skeleton GOAT Gate

### G1: Unit Tests (T1.9)

All 11 unit tests pass:

| Test | Gate | Result |
|---|---|---|
| `zero_signal_yields_zero_derivative` | sanity | ✅ |
| `constant_signal_converges_to_zero_derivative` | paper 25→25, 50→50 | ✅ |
| `step_up_signal_produces_positive_spike` | paper step response | ✅ |
| `step_down_signal_produces_negative_spike` | reverse step | ✅ |
| `swapped_alphas_panics_in_debug` | invariant enforcement | ✅ |
| `reset_zeroes_state` | lifecycle | ✅ |
| `surprise_norm_matches_manual_l2` | numerical correctness | ✅ |
| `derivative_slice_matches_observe_output` | API consistency | ✅ |
| `observe_simd_matches_observe` | SIMD/scalar equivalence | ✅ |
| `sigmoid_surprise_gate_is_bounded_and_monotone` | bridge correctness | ✅ |
| `default_is_paper_ten_to_one_ratio` | config sanity | ✅ |
| `with_initial_preserves_state` | warm-start | ✅ |

### G2: Microbenchmark (T1.10)

`cargo bench -p katgpt-core --features temporal_deriv --bench temporal_deriv_bench`:

| Benchmark | Target | Measured | Verdict |
|---|---|---|---|
| `observe` N=1 | <10ns | 5.8ns | ✅ PASS |
| **`observe` N=8** | **<10ns** | **7.9ns** | **✅ PASS** |
| `observe` N=16 | <10ns | 5.9ns | ✅ PASS |
| `surprise_norm` N=8 | — | 933ps | informational (sub-ns) |
| `sigmoid_surprise_gate` N=8 β=4 | — | 15.7ns | informational |
| **1000-NPC batch serial N=8** | **<10µs** | **7.5µs** | **✅ PASS** |
| 1000-NPC batch rayon N=8 | — | 254µs | **FAIL** vs serial — rayon overhead dominates for tiny per-task work (7.9ns/kernel << 5µs rayon dispatch threshold) |

**Rayon failure analysis:** at 7.9ns/kernel × 1000 = 7.9µs of actual work, the rayon thread-pool dispatch overhead (~5µs/task × N tasks) exceeds the per-task work. The serial loop wins by 34×. This matches the AGENTS.md guidance: "Only parallelize when per-task work exceeds thread-pool overhead (~5μs for rayon)". The 1000-NPC batch should stay serial at N=8; rayon becomes worthwhile only for larger N (e.g., N=64+ where per-kernel work exceeds 50ns) or for cross-NPC independent operations with heavier per-task work.

---

## Phase 1 Exit: ✅ MET

- `cargo test -p katgpt-core --features temporal_deriv` → 11/11 unit tests pass.
- `observe` N=8 = 7.9ns < 10ns target ✅
- 1000-NPC batch serial = 7.5µs < 10µs target ✅
- Rayon path documented as a non-win at this scale (not a regression — serial is the recommended path).

`temporal_deriv` feature stays opt-in (Phase 2–5 fusion gates G2–G5 not yet run). Promote to default-on if ≥2 of {G2 HLA companion, G3 δ-Mem gate, G4 collapse detector, G5 intrinsic curiosity} pass.

---

## Cross-References

- **Plan:** [277_temporal_derivative_kernel.md](../.plans/277_temporal_derivative_kernel.md)
- **Research:** [243_Temporal_Derivative_Kernel_Neocortical_Learning.md](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md)
- **Source:** [arXiv:2606.08720](https://arxiv.org/abs/2606.08720)
- **Implementation:** `katgpt-rs/crates/katgpt-core/src/temporal_deriv.rs`
- **Bench:** `katgpt-rs/crates/katgpt-core/benches/temporal_deriv_bench.rs`

## TL;DR

Phase 1 GOAT gate passes: `observe` N=8 at 7.9ns (target <10ns), 1000-NPC serial batch at 7.5µs (target <10µs). Rayon path is a documented non-win at this kernel size (serial wins 34×). 11/11 unit tests pass. Feature stays opt-in pending Phase 2–5 fusion gates.
