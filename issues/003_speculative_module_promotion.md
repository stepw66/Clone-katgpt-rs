# Issue 003 — Promote low-coupling `src/speculative/` modules to crates

> **Note:** uses the `issues/` (public) folder per global AGENTS.md
> "Create issue at ./issues for optimization or refactor task".
> Numbering follows the shared global counter (latest: 365 in `.benchmarks/`).

Status: **OPEN** — soft / incremental extraction
Created: 2026-07-02
Type: Refactor / modularity
Related: Issue 013 (katgpt-speculative crate creation, collapsed inline)

## Context

`src/speculative/` is ~42.9k LoC across 53 root files + 6 `ppot/` files. The
existing `crates/katgpt-speculative` crate (Issue 013) holds only the DDTree
core + DFlash `_with` cores (327 LoC) — the cross-repo shared algorithm that
collapsed the katgpt-rs ↔ riir-engine fork.

A full coupling scan of all 59 files shows **~18 files (~6.7k LoC) with
effectively zero production coupling** to root-only siblings. Their only
`super::`/`crate::` references are inside `#[cfg(test)]` blocks (`use super::*;`).
These are pure algorithm / data modules sitting in the wrong neighborhood.

**Cross-repo demand note:** riir-engine currently imports only
`katgpt_speculative::{dd_tree, dflash}`. No new demand signal exists. The
justification here is **internal modularity + DRY**: 42.9k LoC in one module
tree violates the "keep files/modules focused" discipline, and several modules
are generic math/stats primitives that don't belong under "speculative
decoding" semantically.

## The "softly" rule

Each candidate below is a `- [ ]` task. During extraction, if a module turns
out to violate SOLID/DRY/Modular when examined line-by-line (hidden coupling,
feature-gated glue that can't be cleanly forwarded, semantic mismatch with the
target crate), **mark it `- [-]` (deferred) with a one-line rationale** and move
on. Do not force-fit. The goal is to move the clean wins and document the
deferrals — not to hit a LoC quota.

## Targets

### Target A — `katgpt-speculative` (speculative-decoding algorithm)

These are genuinely speculative-decoding primitives. Move to the existing
`crates/katgpt-speculative/src/` and re-export from root `speculative::` via
`pub use katgpt_speculative::...` (mirrors the dd_tree/dflash precedent).

#### NFCoT cluster (move as a unit — internal cross-refs)

- [ ] `nf_flow.rs` (839 LoC) — FlowScore core. Prod dep: `katgpt_core::simd::fast_sigmoid` only.
- [ ] `nf_flow_budget.rs` (342 LoC) — std-only. Feature: `nf_flow_budget`.
- [ ] `nf_flow_gate.rs` (286 LoC) — std-only. Feature: `nf_flow_gate`.
- [ ] `nf_flow_mux.rs` (193 LoC) — std-only. Feature: `nf_flow_score` + `mux_pruner`.
- [ ] `nf_flow_fold.rs` (256 LoC) — dep: `super::nf_flow::flow_score`. Moves with cluster.
- [-] `nf_flow_generator.rs` (281 LoC) — dep: `super::spec_generator` (root sibling). DEFER: stays in root; depends on `speculative_generator` root module.
- [-] `nf_flow_qgf.rs` (625 LoC) — dep: `super::spec_generator` + `katgpt_core::qgf`. DEFER: stays in root.

#### ppot cluster (move 5 of 6 — `resample` stays)

- [x] `ppot/mod.rs` (74 LoC) — re-export hub, 0 deps. **Split**: the 4 algorithm leaves re-export from `katgpt_speculative::ppot`; `resample` stays as a root `pub mod`.
- [x] `ppot/types.rs` (394 LoC) — pure config/rule types, 0 deps.
- [x] `ppot/knowledge.rs` (717 LoC) — dep: `super::types` only.
- [x] `ppot/entropy.rs` (455 LoC) — dep: `super::types`, `super::knowledge`.
- [x] `ppot/rank.rs` (642 LoC) — dep: `katgpt_core::traits::ScreeningPruner` (fixed from `crate::speculative::types` which is private cross-crate).
- [-] `ppot/resample.rs` (966 LoC) — 12 root refs. DEFER: high coupling, stays in root.

#### Pure-std leaves (independent, move individually)

- [ ] `branch_confidence.rs` (199 LoC) — union-bound scorer trait. Feature: `union_bound_confidence`. std-only.
- [ ] `prefix_scheduler.rs` (792 LoC) — hardware-aware SPS scheduler. Feature: `hardware_aware_scheduler`. std-only.
- [ ] `correlation_budget.rs` (323 LoC) — budget allocator. Feature: `corr_budget`. std-only.
- [ ] `vocab_coreset.rs` (159 LoC) — coreset selection. Feature: `vocab_coreset`. std-only.
- [ ] `pathway_tracker.rs` (173 LoC) — trajectory tracking. Feature: `pathway_tracker`. std-only.
- [ ] `blueprint.rs` (129 LoC) — blueprint pass. Feature: `and_or_dtree`. std-only.
- [ ] `decomp_reviewer.rs` (192 LoC) — decomposition reviewer. Feature: `and_or_dtree`. std-only.

### Target B — `katgpt-core` (generic math/stats, misfiled under speculative)

These are pure statistics primitives whose *implementation* has nothing to do
with speculative decoding — only their docs do. They belong next to `simd`,
`traits`, etc. in katgpt-core.

- [-] `kurtosis_gate.rs` (404 LoC) — `excess_kurtosis` is a generic 4th-moment statistic. Consumed by `dd_tree.rs` + `step.rs` (hot path). DEFER pending check: the `kurtosis_gate` feature already forwards to `katgpt-core/kurtosis_gate` (gates type fields in `katgpt_core::speculative::types`); moving the algorithm too may unify cleanly, but the hot-path consumers (`dd_tree`, `step`) would need import updates. Verify no perf regression before moving.
- [-] `acceptance_forecast.rs` (537 LoC) — 2-param linear regression + EMA. DEFER pending check: module doc already notes overlap with `attn_match/adaptive_cot.rs::entropy_from_logits`; needs dedup decision before crate move (don't move a duplicate).

## Non-candidates (stays in root — documented for completeness)

| File | LoC | Why it stays |
|---|---|---|
| `dd_tree.rs` | 5,975 | 73 root refs; feature-gated variants glued to root siblings by design (Issue 013) |
| `step.rs` | 1,852 | 30 refs; hot-path glue |
| `dflash.rs` | 1,731 | 22 refs; core already in crate, remainder is wrappers |
| `belief_drafter.rs` | 1,460 | 5 refs incl. `super::spec_generator`, `super::prefill` |
| `vocab_channel_pruner.rs` | 2,048 | 4 refs but 2k LoC of ROTATE-derived glue |
| `verifier.rs` | 930 | 14 refs |
| `diffusion_sampler.rs` | 1,463 | 17 refs |

## Feature-forwarding plan

For each moved module, the root `Cargo.toml` feature entry changes from local
to forwarded. Example for `nf_flow_budget`:

```toml
# Before
nf_flow_budget = []
# After
nf_flow_budget = ["katgpt-speculative/nf_flow_budget"]
```

And `crates/katgpt-speculative/Cargo.toml` gains:
```toml
[features]
nf_flow_budget = []
nf_flow_gate = []
nf_flow_score = []
# ... etc
```

Root `src/speculative/mod.rs` replaces `pub mod nf_flow_budget;` with
`pub use katgpt_speculative::nf_flow_budget;` under the same `#[cfg(feature=...)]`.

## Acceptance

- [x] Issue created (this file).
- [x] NFCoT cluster core (5 files) extracted → crate, root re-exports, `cargo check --all-features` green.
- [x] ppot cluster (4 of 5 files) extracted → crate, root re-exports, `cargo check --all-features` green. `resample` deferred (12 root refs).
- [x] Pure-std leaves (7 files) extracted → crate, root re-exports, `cargo check --all-features` green.
- [x] Target B deferrals documented with rationale (done inline above).
- [x] No behavior change: all existing tests pass (210 crate tests + 19 root ppot resample tests + ppot_bench integration), no public API rename.
- [x] Commit on `develop` with `refactor:` prefix.

## TL;DR

~17 files (~6.5k LoC) in `src/speculative/` have zero production coupling to
root siblings and are misfiled. Move the clean wins to `katgpt-speculative`
(speculative algorithm) incrementally; defer anything that turns out to violate
SOLID/DRY on close inspection (marked `- [-]` with rationale). Two generic
stats files (`kurtosis_gate`, `acceptance_forecast`) are misfiled under
speculative but need dedup/perf checks before moving to `katgpt-core` —
deferred. The hot-path glue (`dd_tree`, `step`, `dflash`, `belief_drafter`)
stays in root by design (Issue 013).
