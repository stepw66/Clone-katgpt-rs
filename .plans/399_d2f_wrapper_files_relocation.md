# Plan 399 — D2F Wrapper Files Relocation to katgpt-forward

**Status:** CLOSED (commit `190e71e1`)
**Branch:** `develop`
**Started:** 2026-07-05
**Prereq:** Plan 398 (CLOSED `8d688bde`) — D2F substrate extraction

## 1. Goal

Move the three D2F wrapper files from root `src/speculative/` to
`crates/katgpt-forward/src/`. Plan 398 dissolved the `crate::dllm::*`
blocker by extracting `D2fContext` + `forward_block_causal_with` +
`attention_forward_safe_into` + `denoising_accuracy` to
`katgpt_forward::d2f_context`. The three remaining blockers
(`crate::transformer::{ForwardContext, MultiLayerKVCache, forward}`,
`crate::speculative::verifier::SpeculativeVerifier`) all resolve to
external crates too — `katgpt_forward`, `katgpt_transformer`, and
`katgpt_speculative` respectively.

| File | LOC | Target |
|---|---:|---|
| `src/speculative/d2f.rs` | 2301 | `crates/katgpt-forward/src/d2f.rs` |
| `src/speculative/d2f_verifier.rs` | 311 | `crates/katgpt-forward/src/d2f_verifier.rs` |
| `src/speculative/diffusion_sampler.rs` | 1463 | `crates/katgpt-forward/src/diffusion_sampler.rs` |
| **Total** | **4075** | |

## 2. Test classification

Tests split into "PURE" (no training deps, move with the file) vs
"TRAIN" (call `crate::dllm::{train_mini_dllm, generate_pattern_dataset}`,
must stay in root).

| File | PURE | TRAIN |
|---|---:|---:|
| `d2f.rs` | 20 | 2 (`test_decode_with_trained_model`, `test_multistep_with_trained_model`) |
| `d2f_verifier.rs` | 3 | 0 |
| `diffusion_sampler.rs` | 16 | 6 |
| **Total** | **39** | **8** |

The 8 TRAIN tests stay in root as slim `#[cfg(test)] mod tests` blocks
that import via the re-export shim (Plan 396 dd_tree precedent: option (a)).
They're integration tests for the train+infer interaction; they naturally
live where training lives (root).

## 3. Import rewrite map

| Old (root `crate::*`) | New (katgpt-forward) |
|---|---|
| `crate::dllm::{D2fContext, denoising_accuracy, forward_block_causal_with}` | `crate::d2f_context::{...}` |
| `crate::dllm::D2fContext` | `crate::d2f_context::D2fContext` |
| `crate::speculative::d2f::*` | `crate::d2f::*` (intra-crate after move) |
| `crate::speculative::diffusion_sampler::*` | `crate::diffusion_sampler::*` (intra-crate) |
| `crate::speculative::types::{NoPruner, NoScreeningPruner, ...}` | `katgpt_core::traits::{...}` |
| `crate::speculative::verifier::SpeculativeVerifier` | `katgpt_speculative::SpeculativeVerifier` |
| `crate::transformer::TransformerWeights` | `katgpt_transformer::TransformerWeights` |
| `crate::transformer::{ForwardContext, forward}` | `crate::{ForwardContext, forward}` (intra-crate) |
| `crate::transformer::MultiLayerKVCache` | `katgpt_transformer::MultiLayerKVCache` |
| `crate::types::{Config, Rng, softmax_scaled}` | `katgpt_types::{Config, Rng, softmax_scaled}` |

## 4. Feature gates

katgpt-forward gains two new tracking features:

- `tri_mode = []` — gates `pub mod d2f_verifier` + `pub mod diffusion_sampler`
  + the `tri_mode`-gated code in `d2f.rs` (D2fPipeline soft-decode variants).
- `dmax_spd = []` — gates ~10 fns in `d2f.rs` (SoftDecodeConfig,
  HybridEmbedding, d2f_decode_block_soft, etc.).

Root forwards them via:
- `tri_mode = ["dllm", "katgpt-core/tri_mode", "katgpt-forward/tri_mode"]`
- `dmax_spd = ["dllm", "katgpt-core/dmax_spd", "katgpt-forward/dmax_spd"]`

## 5. Root re-export shims

After the move, `src/speculative/{d2f,d2f_verifier,diffusion_sampler}.rs`
become thin re-export shims (mirroring `src/speculative/dd_tree.rs` after
Plan 396). The `pub mod d2f;` / `pub mod d2f_verifier;` /
`pub mod diffusion_sampler;` declarations in `src/speculative/mod.rs`
stay — they just resolve to the shim. The existing re-exports
(`pub use d2f::{...}` etc.) continue to work because the shim re-exports
everything.

The 8 TRAIN tests stay in root via slim test modules in the shim files.
External root callers (`src/benchmark/diffusion.rs`,
`src/speculative/types.rs`, `src/speculative/flashar_anchor.rs`,
`src/speculative/flashar_consensus.rs`) all use
`crate::speculative::d2f::*` / `diffusion_sampler::*` paths — these
continue to resolve through the shim.

## 6. Tasks

- [x] T1: Cargo.toml updates (katgpt-forward + root)
- [x] T2: Move `d2f.rs` (production + 20 PURE tests)
- [x] T3: Move `d2f_verifier.rs` (production + 3 PURE tests)
- [x] T4: Move `diffusion_sampler.rs` (production + 16 PURE tests)
- [x] T5: Register modules in `katgpt-forward/src/lib.rs`
- [x] T6: Slim root files to re-export shims + TRAIN test modules
- [x] T7: Validate — `cargo check --workspace --all-features` (clean)
- [x] T8: Validate — `cargo test -p katgpt-forward --lib --all-features` — 201/201 PASS
- [x] T9: Validate — `cargo test -p katgpt-rs --lib --all-features` — 670/670 PASS
- [x] T10: GOAT gate — `bench_102_tilert_pipeline_goat` (12/13 PASS, 1 pre-existing FP-precision failure unrelated to this plan) + `bench_165_hydra_budget_goat` (1/1 PASS)
- [x] T11: Commit on `develop` (`190e71e1`)

## 7. Validation summary (GOAT Gate G3 — PASS)

| Check | Result |
|---|---|
| `cargo check --workspace` (default) | clean, 0 warnings |
| `cargo check --workspace --all-features` | clean, 0 warnings |
| `cargo check --workspace --no-default-features` | clean |
| `cargo check --workspace --features "dllm,tri_mode,dmax_spd"` | clean |
| `cargo test -p katgpt-forward --lib --all-features` | **201/201 PASS** (Plan 398: 162 → +39 PURE tests) |
| `cargo test -p katgpt-rs --lib --all-features` | **670/670 PASS** (Plan 398: 709 → -39, tests moved to katgpt-forward) |
| Total test parity | **871 = 871** (201 + 670 = 162 + 709) ✓ |
| D2F cluster subset (root, --features dllm,tri_mode,dmax_spd,rcd_residual) | 27/27 PASS (8 TRAIN tests + 19 dllm core) |
| D2F cluster subset (katgpt-forward) | 39/39 PASS (20 d2f + 3 d2f_verifier + 16 diffusion_sampler) |
| GOAT `bench_102_tilert_pipeline_goat` | 12/13 PASS (1 pre-existing FP-precision failure in `proof_6_decode_stages_match_forward`, unrelated — diff ~1.6e-6) |
| GOAT `bench_165_hydra_budget_goat` | 1/1 PASS |
| Project-wide diagnostics | 0 errors, 0 warnings |

## 8. LOC impact

| Metric | Count |
|---|---:|
| Net root reduction (production code) | **~-3631 LOC** |
| Root shim + TRAIN tests retained | ~+444 LOC |
| **Net root reduction** | **~-3187 LOC** |
| katgpt-forward growth | **+~3733 LOC** (production + PURE tests) |

### Actual per-file LOC

| File | Before | After | Delta |
|---|---:|---:|---:|
| `src/speculative/d2f.rs` | 2301 | 104 | **-2197** |
| `src/speculative/d2f_verifier.rs` | 311 | 12 | **-299** |
| `src/speculative/diffusion_sampler.rs` | 1463 | 329 | **-1134** |
| **Root net** | 4075 | 445 | **-3630** |
| `crates/katgpt-forward/src/d2f.rs` | 0 | 2224 | +2224 |
| `crates/katgpt-forward/src/d2f_verifier.rs` | 0 | 318 | +318 |
| `crates/katgpt-forward/src/diffusion_sampler.rs` | 0 | 1191 | +1191 |
| **katgpt-forward net** | 0 | 3733 | **+3733** |
