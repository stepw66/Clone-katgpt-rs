# Plan 400 — FlashAR Anchor + Consensus Relocation to katgpt-forward

## Goal

Continue Proposal 003 (master `src/` consolidation). The two `flashar_*` wrapper
files in `src/speculative/` are now unblocked (Plan 398 dissolved the d2f
substrate blocker, Plan 399 dissolved the d2f wrapper cluster blocker). Move
both to `crates/katgpt-forward/src/` using the same thin re-export shim
pattern Plan 399 established.

## Files to move

| File | Before | Pure-inference tests | Train-coupled tests |
|---|---:|---:|---:|
| `src/speculative/flashar_anchor.rs` | 728 LOC | 6 | 2 (`test_anchor_then_fill_produces_valid_output`, `test_anchor_then_fill_reduces_steps`) |
| `src/speculative/flashar_consensus.rs` | 853 LOC | 10 | 0 |
| **Total** | **1581 LOC** | **16 PURE** | **2 TRAIN** |

The 2 TRAIN tests in `flashar_anchor.rs` use `make_trained_weights()` which
calls `crate::dllm::{generate_pattern_dataset, train_mini_dllm}` — root-only
training code. Same pattern as Plan 399's `d2f.rs` shim.

## Strategy

### Move file body to katgpt-forward
Pure-inference production code + PURE tests move to
`crates/katgpt-forward/src/flashar_{anchor,consensus}.rs`. TRAIN tests stay in
root's slimmed-down shim files.

### Import rewrites

| Old (root `crate::*`) | New (katgpt-forward) |
|---|---|
| `crate::dllm::D2fContext` | `crate::d2f_context::D2fContext` |
| `crate::dllm::forward_block_causal_with` | `crate::d2f_context::forward_block_causal_with` |
| `crate::speculative::d2f::*` | `crate::d2f::*` (intra-crate) |
| `crate::speculative::types::{NoPruner, NoScreeningPruner, ConstraintPruner, ScreeningPruner}` | `katgpt_core::traits::*` |
| `crate::speculative::verifier::SpeculativeVerifier` | `katgpt_speculative::SpeculativeVerifier` |
| `crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward}` | `crate::{ForwardContext, forward}` + `katgpt_transformer::{MultiLayerKVCache, TransformerWeights}` |
| `crate::types::{Config, Rng, softmax_scaled}` | `katgpt_types::*` |
| `katgpt_core::speculative::sampling::sample_from_distribution` | unchanged (already external) |
| `katgpt_core::{TernaryWeights, simd_ternary_matvec}` (plasma_path gate) | unchanged |

### Cargo.toml changes

- **katgpt-forward**: 3 new tracking features
  - `flashar_anchor = []`
  - `flashar_consensus = []`
  - `plasma_path = ["katgpt-core/plasma_path"]` (needed because
    `flashar_consensus.rs` has a `#[cfg(feature = "plasma_path")]` block that
    references `katgpt_core::TernaryWeights` / `katgpt_core::simd_ternary_matvec`).
- **root**:
  - `flashar_anchor` extended with `"katgpt-forward/flashar_anchor"`
  - `flashar_consensus` extended with `"katgpt-forward/flashar_consensus"`
  - `plasma_path` extended with `"katgpt-forward/plasma_path"`

### Module registration in `crates/katgpt-forward/src/lib.rs`

```rust
#[cfg(all(feature = "dllm", feature = "flashar_anchor"))]
pub mod flashar_anchor;
#[cfg(all(feature = "dllm", feature = "flashar_anchor"))]
pub use flashar_anchor::{AnchorConfig, AnchorFillResult, anchor_then_fill};

#[cfg(all(feature = "dllm", feature = "flashar_consensus"))]
pub mod flashar_consensus;
#[cfg(all(feature = "dllm", feature = "flashar_consensus"))]
pub use flashar_consensus::{
    ConsensusConfig, ConsensusResult, DualPathResult, FlashARConsensusVerifier,
    MAX_DRAFT_WIDTH, ThermalPath, compute_ternary_consensus, dual_path_draft,
    route_thermal_paths,
};
```

(Module-level `dllm` gate mirrors the Plan 399 pattern — both files consume
`crate::d2f::*` which is itself `dllm`-gated.)

### Public API preserved

Root shim files re-export via `pub use katgpt_forward::flashar_anchor::*;` etc.,
so every historical `crate::speculative::flashar_{anchor,consensus}::*` import
path continues to resolve. Verified external callers:
- `tests/bench_166_flashar_consensus_goat.rs` (uses
  `katgpt_rs::speculative::flashar_consensus::*`).
- `src/speculative/mod.rs` (re-exports).

## Tasks

- [x] 1. Move `flashar_anchor.rs` to katgpt-forward (production + 6 PURE tests)
- [x] 2. Move `flashar_consensus.rs` to katgpt-forward (production + 10 PURE tests)
- [x] 3. Slim root shims (re-export + 2 TRAIN tests for anchor)
- [x] 4. Cargo.toml updates (root + katgpt-forward)
- [x] 5. Module registration in katgpt-forward/lib.rs
- [x] 6. GOAT gate G3 validation
- [x] 7. Commit on develop

## GOAT Gate G3 — Validation Results

```bash
cargo check --workspace                              # default features — PASS
cargo check --workspace --all-features               # all combos — PASS
cargo check --workspace --no-default-features        # zero-dep baseline — PASS
cargo test -p katgpt-forward --lib --all-features    # 218/218 PASS (+17 vs Plan 399's 201 — 16 flashar + 1 sibling-agent addition)
cargo test -p katgpt-rs --lib --all-features         # 653/653 PASS (-17 vs Plan 399's 670 — 16 flashar moved + 1 sibling-agent change)
                                                       Total parity: 871 = 871 ✓
cargo test -p katgpt-forward --lib --features dllm,flashar_anchor,flashar_consensus,plasma_path,tri_mode,dmax_spd,rcd_residual  # 164 PASS
cargo test -p katgpt-rs --lib --all-features flashar  # 2/2 TRAIN tests PASS
cargo test -p katgpt-rs --test bench_166_flashar_consensus_goat --features flashar_consensus  # 9/9 PASS
cargo test -p katgpt-rs --test test_diffusion_sampler_goat --features flashar_consensus  # 5/5 PASS
diagnostics: 0 errors, 0 warnings
```

## LOC Impact

| File | Before | Root shim | katgpt-forward | Root delta |
|---|---:|---:|---:|---:|
| `flashar_anchor.rs` | 728 | 140 | 632 | **-588** |
| `flashar_consensus.rs` | 853 | 19 | 856 | **-834** |
| **Total** | **1581** | **159** | **1488** | **-1422** |
