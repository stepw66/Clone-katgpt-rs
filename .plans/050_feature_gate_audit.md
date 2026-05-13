# Plan 050: Feature Gate Audit

> **Status:** Complete
> **Depends on:** None (standalone cleanup)

## Tasks

- [x] T1: Audit all features in `Cargo.toml` against actual `#[cfg]` usage in code
- [x] T2: Remove `leviathan` feature gate — `LeviathanVerifier` is always compiled, gate does nothing
- [x] T3: Add `feedback = []` feature gate — `feedback.rs` was unconditional orphan
- [x] T4: Update `full = [...]` — remove `leviathan`, add `feedback`
- [x] T5: Gate `feedback.rs` behind `#[cfg(feature = "feedback")]` in `lib.rs`
- [x] T6: Update `README.md` Feature Flags table — honest status, plan refs, ungated note
- [x] T7: Fix `.docs/02_architecture.md` — rename "Feature Flag" → "Availability" column
- [x] T8: Fix `.docs/03_speculative_decoding.md` — note REST client lives in riir-ai
- [x] T9: Add `g_zero = ["bandit"]` to Plan 049 feature gate section

---

## Audit Results

### Always Gated (correct)

| Feature | Gate Used | Reason |
|---------|:---------:|--------|
| `bandit` | ✅ | Heavy RL infra, optional strategy |
| `sparse_mlp` | ✅ | Experimental matmul, touches `ForwardContext` |
| `domain_latent` | ✅ | Touches `forward_base()` hot path |
| `ppot` | ✅ | Optional resampling, separate module |
| `bomber` | ✅ | Game-specific (bevy_ecs) |
| `bomber-wasm` | ✅ | Game-specific (wasmtime + papaya) |
| `monopoly` | ✅ | Game-specific (bevy_ecs) |
| `sudoku` | ✅ | Game-specific |
| `validator` | ✅ | Heavy deps (syn, proc-macro2) |
| `feedback` | ✅ (added) | Orphan module, sends to void, needs riir-gpu consumer |

### Always Compiled (no gate needed)

| Component | Why No Gate |
|-----------|-------------|
| `LeviathanVerifier` | Part of `verifier.rs` + `benchmark.rs`. Removed feature flag — was declared but did nothing. |
| `SimulatedVerifier` | Core verification, always available |
| `forward_raven` | Forward variant, zero-cost until `RavenKVCache` instantiated |
| `forward_turboquant` | Forward variant, zero-cost until `TurboQuantKVCache` instantiated |
| `forward_paged` | Forward variant, zero-cost until `PagedKVCache` instantiated |
| `forward_prefill` | Prompt processing, caller picks |
| `percepta.rs` | Sudoku-specific but zero-cost when unused |

### Placeholder / Not Yet Started

| Feature | Status | Note |
|---------|--------|------|
| `rest` | Bridge only | Full client in `riir-ai/riir-rest`. This repo has 1 test + `merge_retrieved_branches()` stub (Plan 009). |
| `embedding_router` | Not started | No Plan 024 file exists. Commented-out test stubs in `step.rs`. |
| `gpu` | Placeholder | GPU training lives in `riir-ai/riir-gpu`. Flag is for future integration. |
| `language_domain` | Future | Intentionally empty per Plan 040 Task 8. |
| `g_zero` | Planned | Plan 049. Gate mandatory — self-play training ≠ inference. |

### Alias

| Feature | Maps To | Independent Code |
|---------|---------|:----------------:|
| `game_domain` | `domain_latent` | None — `Config::game()` always compiled |

---

## Gate Principle

**Gate when:**
- Different concern (training vs inference, game vs core)
- Brings heavy deps (syn, bevy_ecs, wasmtime)
- Touches hot path optionally
- May not be used by all consumers

**Don't gate when:**
- Zero-cost until instantiated (forward variants, cache types)
- Core functionality always needed (SimulatedVerifier, basic forward)
- Code is small and doesn't pull deps

## Files Changed

- `Cargo.toml` — removed `leviathan`, added `feedback`, updated `full`, honest comments
- `src/lib.rs` — `#[cfg(feature = "feedback")]` on `pub mod feedback`
- `README.md` — Feature Flags table + ungated components note
- `.docs/02_architecture.md` — SpeculativeVerifier table column rename
- `.docs/03_speculative_decoding.md` — REST Bridge honest status
- `.plans/049_g_zero_self_play.md` — Feature Gate Strategy section added