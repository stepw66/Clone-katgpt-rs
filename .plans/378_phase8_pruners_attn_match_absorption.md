# Plan 378 — Proposal 003 Phase 8: katgpt-pruners + katgpt-attn-match absorption

## TL;DR

Move three remaining root-local modules to their destination crates per
Proposal 003 Phase 8 destination map:

| Item | LOC | Destination | Feature gate |
|---|---|---|---|
| `src/closure_wire.rs` | 451 | `katgpt-pruners/src/closure_wire.rs` | `closure_instrument` (new in katgpt-pruners) |
| `src/screening/` (6 files) | 1756 | `katgpt-pruners/src/screening/` | `complexity_prior_sampler` + `mcts_k_prior`/`bandit_k_prior`/`spec_k_prior` (new in katgpt-pruners) |
| `src/rerank.rs` | 526 | `katgpt-attn-match/src/rerank.rs` | `maxsim` + `bt_rank` (new in katgpt-attn-match) |

## Pre-move audit findings

### `closure_wire.rs` — clean
- `katgpt_core::closure::{...}` ✅ katgpt-pruners already depends on katgpt-core
- `crate::speculative::types::ScreeningPruner` → resolves to `katgpt_core::traits::ScreeningPruner` via root shim
- `crate::pruners::AbsorbCompress` → resolves to `katgpt_pruners::absorb_compress::AbsorbCompress` via root shim
- `crate::pruners::ReviewMetrics` → resolves to `katgpt_core::pruners::review_metrics::ReviewMetrics`
- Test imports: `crate::pruners::{AbsorbCompressLayer, CompressConfig}`, `crate::speculative::types::{NoScreeningPruner, ScreeningPruner}` — all re-export shims

### `screening/` — clean
- `fastrand::Rng`, `core::hint::black_box` only external deps
- Internal `crate::screening::complexity_prior::*` refs (intra-module)
- Operates on `&[u8]` / `&[f32]` only — no HLA/functor/shard types (per mod.rs doc)

### `rerank.rs` — clean
- Only dep: `katgpt_core::simd::{maxsim_score, simd_add_inplace, simd_dot_f32, simd_scale_inplace}`
- Contains `bt_rank`-gated `SymmetricBoundaryPair` (orthogonal to `maxsim` gate)

## Tasks

- [x] T1. Copy `src/closure_wire.rs` → `crates/katgpt-pruners/src/closure_wire.rs`, fix imports
- [x] T2. Copy `src/screening/*` → `crates/katgpt-pruners/src/screening/`, fix intra-module refs (none needed)
- [x] T3. Copy `src/rerank.rs` → `crates/katgpt-attn-match/src/rerank.rs`, fix imports (none needed)
- [x] T4. Update `crates/katgpt-pruners/Cargo.toml` (new features) + `src/lib.rs` (new modules)
- [x] T5. Update `crates/katgpt-attn-match/Cargo.toml` (add katgpt-core dep + new features) + `src/lib.rs`
- [x] T6. Update root `Cargo.toml` (forward features to crates) + `src/lib.rs` (re-export shims)
- [x] T7. Delete originals from `src/`
- [x] T8. GOAT gate G3: workspace check (all-features, default, no-default), test suites
- [x] T9. Update `.proposals/003_src_consolidation_master.md` — Phase 8 DONE
- [x] T10. Commit on `develop` with `refactor:` prefix

## Validation plan

- `cargo check --workspace --all-features`
- `cargo check --workspace` (default)
- `cargo check --workspace --no-default-features`
- `cargo test -p katgpt-pruners --lib`
- `cargo test -p katgpt-attn-match --lib`
- `cargo test --lib` (root)
- `cargo test --test bench_290_closure_wire_integration`
- `cargo test --test bench_maxsim_rerank`
- `cargo test --test bench_290_closure_instrument_goat -- --test-threads=1`
- examples: `algorithmic_probability_sampler_demo`, `algorithmic_probability_sampler_bench`
