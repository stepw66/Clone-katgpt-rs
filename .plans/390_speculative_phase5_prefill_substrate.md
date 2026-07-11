# Plan 390 — Speculative Phase 5: prefill.rs substrate extraction

Status: **closed**
Branch: `develop` (per global rule — no feature branches)
Audit basis: continuation of Plan 389 (CLOSED `bd7e0293`), Proposal 003 Phase 19
Target: `src/speculative/prefill.rs` (1099 LOC) — pure substrate → katgpt-speculative

## TL;DR

Apply the **trait-impl split** technique (refined in Plan 389) to `prefill.rs`.
The file has a clean bimodal split:

- **Pure substrate** (trait + helpers, ~580 LOC): no `forward`, no `SpeculativeContext`.
  Moves to `katgpt-speculative/src/prefill.rs`.
- **Forward-coupled impls** (~520 LOC): `AttentionScorer` / `BlockAttentionScorer` call
  `crate::transformer::forward` and need `SpeculativeContext`. Stay in root, implement
  the trait from katgpt-speculative (Plan 389 pattern).

The `crate::dash_attn::{entmax_1p5, entmax_support}` dep is illusory — they live in
`katgpt-attn` (root's `dash_attn` is a re-export shim). However, `katgpt-attn`'s
`dash_attn` feature pulls a heavy dep chain (katgpt-forward, katgpt-pruners/bandit,
katgpt-kv, katgpt-transformer, serde). Pulling that for one function is overkill, so
`block_select_entmax` stays in root.

## Decision matrix

| Item | Treatment | Reason |
|---|---|---|
| `PrefillScorer` trait | Move to leaf | Signature needs `TransformerWeights` (already mandatory in katgpt-speculative per Plan 389). No `forward`. |
| `RandomScorer`, `UniformScorer` | Move to leaf | Pure; use `katgpt_types::Rng`. |
| `compress_prompt`, `block_select`, `block_compression_ratio`, `block_select_grid`, `compress_prompt_blocks`, `should_compress` | Move to leaf | Pure functions over `FlashPrefillConfig` (already in katgpt-core). |
| `speculative_prefill`, `speculative_prefill_block`, `speculative_prefill_adaptive` | Move to leaf | Orchestrators over the trait + pure helpers. |
| `block_score_maxsim` (gated `maxsim`) | Move to leaf | Add `maxsim = ["katgpt-core/maxsim"]` tracking flag. |
| `AttentionScorer`, `BlockAttentionScorer` (structs + impls) | **Stay in root** | Need `crate::transformer::forward` + `SpeculativeContext`. |
| `block_select_entmax` (gated `dash_attn`) | **Stay in root** | `crate::dash_attn::{entmax_1p5, entmax_support}` re-export from katgpt-attn — leaf dep would pull katgpt-attn's heavy `dash_attn` chain. |
| Tests for substrate fns | Move to leaf | Use only `katgpt_types::Rng` / `katgpt_transformer::TransformerWeights`. |
| Tests for AttentionScorer / BlockAttentionScorer / block_select_entmax | Stay in root | Need `forward` / `dash_attn`. |

## Tasks

- [x] Create `katgpt-speculative/src/prefill.rs` with substrate code.
- [x] Add `prefill = []` (not needed — always-on) and `maxsim = ["katgpt-core/maxsim"]` tracking flags to `katgpt-speculative/Cargo.toml`.
- [x] Add `pub mod prefill;` to `katgpt-speculative/src/lib.rs` (no `pub use prefill::*;` — root re-exports the symbols it needs).
- [x] Rewrite root `src/speculative/prefill.rs` to keep only forward-coupled impls + `block_select_entmax`, re-export substrate from katgpt-speculative.
- [x] Update root `Cargo.toml` feature: `maxsim` forwards to `katgpt-speculative/maxsim`.
- [x] GOAT G3: `cargo check` (default / all-features / no-default) + tests.

## GOAT Gate G3 — PASS

| Check | Result |
|---|---|
| `cargo check --workspace` (default) | Clean ✅ |
| `cargo check --workspace --all-features` | Clean ✅ |
| `cargo check --workspace --no-default-features` | Clean ✅ |
| katgpt-speculative lib tests (all-features) | **1054 passed** ✅ (up +15 from Plan 389's 1039) |
| prefill module tests (`prefill::tests::*`) | **15/15 PASS** ✅ |
| Root prefill tests (default) | **4/4 PASS** ✅ (AttentionScorer + 3 block_select_entmax) |
| Root lib tests (default) | **431 passed** ✅ (down 15 from Plan 389's 446 = tests moved to leaf; expected) |
| Root lib tests (all-features) | **859 passed** ✅ (down 15 from Plan 389's 874; expected) |
| katgpt-pruners lib tests (default) | **126 passed** ✅ (matches Plan 389) |
| `test_133_parallel_probe_ablation` (GOAT integration) | **1/1 PASS** ✅ |
| `speculative_generator_goat` | **3/3 PASS** ✅ |

### Pre-existing test failures (NOT caused by Plan 390)

- `dllm.rs - dllm::denoise_loop_rcd_3sr` doctest — fails identically on
  baseline `bd7e0293` (root doctest, unrelated to prefill).
- 5 katgpt-speculative doctests (`answer_extract::*` × 3 from Plan 386 oversight,
  `progressive_mcgs::*` × 2) — fail identically on baseline.

## Final notes

- The file is **split, not deleted** — root keeps the forward-coupled half.
- Root-only speculative file count: **13** (unchanged from Phase 19 — the file
  still exists in root, just thinner).
