# Plan 381 — Step-Attribution Δ-Qualifier GOAT Gate Report

**Date:** 2026-07-06
**Primitive:** `StepAttributionQualifier` + `StepLocalizer` + `DotProductLocalizer` + `SumAggregator` / `MeanAggregator`
**Module:** `crates/katgpt-pruners/src/step_attribution_qualifier.rs`
**Feature:** `step_attribution_qualifier` (opt-in)
**Source paper:** [arXiv:2606.01311](https://arxiv.org/abs/2606.01311) — SkillAdaptor, Yu et al. 2026
**Bench:** `benches/step_attribution_qualifier_bench.rs` (`cargo run --release --bench step_attribution_qualifier_bench --features step_attribution_qualifier`)

## TL;DR

**G1 / G3 / G4 / G5 / G6 PASS. G2 BLOCKED on riir-ai Plan 313 G6 quality-parity PoC** (mandatory before any default-on promotion per §3.6 defend-wrong rule). The primitive ships **opt-in**; quality-parity is UNPROVEN until the riir-ai PoC runs.

Gate overhead is **13 ns at W=64** (76.9× under the 1µs target). End-to-end `qualify()` is **119 ns at W=64** — far under any realistic consolidation-cycle budget (the 5 ms per-NPC per-cycle budget in riir-ai Plan 313 T3.4 leaves ~42,000× headroom).

## Gate Results

| Gate | Description | Status | Evidence |
|------|-------------|--------|----------|
| **G1** | Correctness — Δ≥0 logic; localize_and_link returns correct fault | ✅ PASS | 14/14 unit tests green (T2.1 commit/rollback/tie; T2.2 threshold; T2.3 localize+link; T2.4 canonical usage; aggregator sanity; TickFaultSite OOB panic) |
| **G2** | Quality-parity — reproduces SkillAdaptor ±8.1→±5.2 variance reduction | ⬜ BLOCKED | riir-ai Plan 313 Phase 5 G6 PoC. Mandatory before default-on. Per §3.6, no quality-parity claim is made until the PoC runs. |
| **G3** | No-regression — feature-off = byte-identical to develop | ✅ PASS | Module is `#[cfg(feature = "step_attribution_qualifier")]`-gated; zero impact when off. CI: `cargo check` (default features) clean. |
| **G4** | Perf — gate overhead < 1µs at W=64, excluding executor | ✅ PASS | 13 ns aggregate-only @ W=64 (76.9× margin). End-to-end `qualify()` 119 ns @ W=64. See §"Latency Numbers" below. |
| **G5** | Modelless — no riir-train/riir_gpu/backprop dep | ✅ PASS | Zero new deps added to `crates/katgpt-pruners/Cargo.toml`. Pure aggregate + compare + sigmoid. |
| **G6** | Feature-isolation — single-feature + all-features clean | ✅ PASS | `cargo check -p katgpt-pruners --features step_attribution_qualifier` ✅; `cargo check --features step_attribution_qualifier` ✅; `cargo check --all-features` ✅ (37.25 s, no errors). |

## Latency Numbers (G4 evidence)

Bench config: median of 11 outer × 1000 inner calls, warmup 2000. NoOpExecutor (returns `*k` per input), SumAggregator, AddConst(0.5) mutation. macOS, release profile.

```
┌──────┬──────────────────┬──────────────────┬──────────────────┐
│  W   │  end-to-end (ns) │  aggregate (ns)  │  alloc+misc (ns) │
├──────┼──────────────────┼──────────────────┼──────────────────┤
│   16 │             52.0 │              2.0 │             50.0 │
│   32 │             67.0 │              4.0 │             63.0 │
│   64 │            119.0 │             13.0 │            106.0 │
│  128 │            231.0 │             37.0 │            194.0 │
└──────┴──────────────────┴──────────────────┴──────────────────┘
```

**Reading:**
- **aggregate-only** is the gate-overhead proxy — pure `SumAggregator::aggregate` on a pre-built `Vec<f32>`. **13 ns at W=64** vs 1000 ns target → 76.9× margin.
- **end-to-end** is the full `qualify()` call — includes 2× `Vec<f32>` alloc + 2× aggregate + compare + branch. **119 ns at W=64.**
- **alloc+misc** = end-to-end − aggregate-only. Dominated by 2× `Vec<f32>` allocation (the executor's `replay` contract). NOT gate overhead — it's the price of the `ReplayExecutor::replay -> Vec<S>` API.

**Scaling:** end-to-end is roughly linear in W (52 → 231 ns across 8× window growth = 4.4×), as expected for a sum + alloc-bound kernel. Aggregate-only scales sub-linearly (2 → 37 ns = 18× across 8× window) — LLVM auto-vectorizes the sum at larger W.

**riir-ai budget headroom:** the Plan 313 T3.4 latency budget is < 5 ms per-NPC per-consolidation-cycle at W=64. End-to-end `qualify()` at 119 ns leaves **~42,000× headroom** — even if the real CLR `ReplayExecutor` is 100× slower than NoOpExecutor (which it will be, since CLR involves HLA dot-products + sigmoid per tick), the budget holds with room to spare.

## Why Modelless (G5)

The primitive carries no game/runtime semantics. The `StepAttributionQualifier` is generic over state `K`, replay input `I`, score `S`, executor `E`, aggregator `A`. The consumer (riir-ai Plan 313) supplies the concrete `ReplayExecutor<BranchBank, CognitiveBranchTickRecord, f32>` (CLR `r_k` reward) and `CandidateMutation<BranchBank>` (branch update diff).

No new dependencies were added to `crates/katgpt-pruners/Cargo.toml`. The module uses only `core::marker::PhantomData` and `std::Vec` / `std::String` (already in the crate's dep tree). No `riir-train`, no `riir-gpu`, no backprop.

## Promotion Status

**Stays opt-in.** Per Plan 381 Phase 5 + §3.6 defend-wrong rule, promotion to default-on requires:
1. riir-ai Plan 313 Phase 5 G6 quality-parity PoC PASSES, OR
2. riir-ai Plan 313 Phase 5 G6 REFUTES parity → record raw numbers as §"PoC Addendum" here; primitive stays opt-in; architectural + latency claims stand; quality claim is a tracked follow-up.

Until one of those happens, `step_attribution_qualifier` is opt-in in both `crates/katgpt-pruners/Cargo.toml` and the root `Cargo.toml`.

## Cross-references

- **Plan:** `.plans/381_step_attribution_delta_qualification_primitive.md`
- **Research:** `.research/381_SkillAdaptor_Step_Level_Fault_Attribution_Delta_Qualification.md`
- **Private guide (riir-ai):** `riir-ai/.research/313_Step_Level_Fault_Attribution_Commit_Gate_Guide.md`
- **Runtime wiring (riir-ai):** `riir-ai/.plans/313_step_attribution_branch_wiring.md`
- **Sibling primitive:** `TrajectoryDoctor` (Plan 223) — `crates/katgpt-pruners/src/trajectory_doctor.rs`
- **Source paper:** [arXiv:2606.01311](https://arxiv.org/abs/2606.01311) — SkillAdaptor, Yu et al. 2026
