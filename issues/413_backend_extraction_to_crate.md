# Issue 413 — Extract device backends to `katgpt-backend` crate

> **Note:** uses the `issues/` (public) folder per global AGENTS.md
> "Create issue at ./issues for optimization or refactor task".
> Numbering follows the shared global counter (latest: 412 in `.plans/`).

Status: **OPEN**
Created: 2026-07-08
Type: Refactor / modularity
Related: Issue 033 (original root-residency decision — **stale**, see §1),
        Plan 385 (forward → katgpt-forward, broke the old circular-dep argument),
        Plan 176 (ANE/GPU backend + inference_router inception)

## TL;DR (top)

The three device-backend files (`inference_backend.rs`, `gpu_backend.rs`,
`ane_backend.rs`, ~3.0k LoC total) are root-resident per an Issue 033 §C
"circular dependency" argument that **no longer holds**: `forward` and
`ForwardContext` moved to the `katgpt-forward` leaf in Plan 385 / Issue 007
Phase F. Every type the backends import now lives in a leaf crate
(`katgpt-forward`, `katgpt-transformer`, `katgpt-types`). A `katgpt-backend`
crate can depend on those leaves directly with **zero circular deps**.
`inference_router.rs` stays root (composition layer with 6+ root-only imports).

The CPU/SIMD backend is **not** a separate file — it's the `kog_cpu_fusion`
kernel inside `katgpt-transformer` + `katgpt-forward` (DEFAULT-ON), surfaced
through `CpuBackend` which is just a 3-line trait adapter. No new backend to
move; only the trait + 3 device impls.

## 1. Why the Issue 033 justification is stale

`src/inference_backend.rs` line 8 still reads:

> _Root-resident by design (Issue 033 §C, Option C). ... The trait cannot move
> without its providers; the providers cannot move without root's forward.
> A redundant `ForwardPass` trait was rejected as non-production-grade._

That argument died on 2026-07-05 (Plan 385). Verified in `src/transformer.rs`:

```rust
pub use katgpt_forward::ForwardContext;                       // moved (Issue 007 Phase F)
pub use katgpt_forward::{... forward, forward_base, ...};     // moved (Plan 385)
pub use katgpt_transformer::{... TransformerWeights, MultiLayerKVCache, ...};
```

Backend imports today (all going through root re-export shims):

```rust
// inference_backend.rs, gpu_backend.rs, ane_backend.rs all import:
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::{Config, kv_dim};   // gpu_backend also pulls kv_dim
use crate::inference_backend::InferenceBackend;  // gpu_backend + ane_backend
```

Every one of those resolves to a leaf crate. Issue 033 itself has been
resolved-and-removed (no `.issues/033*` file exists). The line-8 doc comment
is actively misleading and must be fixed as part of this extraction.

## 2. Coupling scan

| File | LoC | `crate::` imports (prod) | Leaf deps | Move? |
|---|---|---|---|---|
| `inference_backend.rs` | 352 | `transformer::forward` (via `katgpt-forward`) | forward, transformer, types | ✅ |
| `gpu_backend.rs` | 1302 | `inference_backend`, `transformer::*`, `types::{Config,kv_dim}` | forward, transformer, types | ✅ |
| `ane_backend.rs` | 1353 | `inference_backend`, `transformer::*`, `types::Config` | forward, transformer, types | ✅ |
| `inference_router.rs` | (large) | **6+ root-only**: `trigger_gate`, `pruners::acceptance_variance`, `dllm_solver`, `chiaroscuro`, `pipeline_pruner`, `katgpt_core::SpeculativeGenerator` | — | ❌ stays root |

Grep confirms leaf crates don't consume the trait:
`katgpt-rs/crates/**` for `InferenceBackend|BackendKind|auto_backend` = **0 hits**.
Sibling repos have their own `npc_ane_backend` (riir-engine, per Issue 007
Phase C) — this general-purpose backend has no external demand signal beyond
root + 3 test files. Justification is internal modularity (matches the active
extraction pattern: every other module is moving to leaves).

## 3. CPU/SIMD backend — not a file

There is no `CpuSimdBackend`. The SIMD work is a compile-time code path
*inside* `forward()`, not a polymorphic backend. This is the correct design —
you don't pick "SIMD vs non-SIMD" at runtime; the fused kernel is always-on.

| Feature | What | Where |
|---|---|---|
| `kog_cpu_fusion` (Plan 160, **DEFAULT-ON**, GOAT 3/3) | RMSNorm gamma folding + QKV interleaving monokernel | `katgpt-transformer` + `katgpt-forward` |
| `tiled_attention` (Plan 115) | Tiled online-softmax flash attention for CPU SIMD | `katgpt-core` + `katgpt-forward` |
| `channel_simd_align` (Plan 227) | Cache-line-padded weight storage | `katgpt-core` |
| `plasma_path` (Plan 148) | Bit-plane ternary SIMD matvec (mul-free CPU inference) | `katgpt-core` |

`CpuBackend` stays a 3-line trait adapter — it just delegates to
`katgpt_forward::forward`, which uses the SIMD kernels when features are on.

## 4. The "softly" rule

Each target is a `- [ ]` task. If extraction violates SOLID/DRY on close
inspection (hidden feature-gate glue that can't forward cleanly, unexpected
root coupling), mark `- [-]` (deferred) with a one-line rationale and move on.
Don't force-fit.

## 5. Targets

### Target A — `katgpt-backend` crate (new)

- [ ] **A0:** Create `crates/katgpt-backend/` skeleton (`Cargo.toml` +
      `src/lib.rs`). Deps: `katgpt-forward`, `katgpt-transformer`, `katgpt-types`.
      Optional macOS deps (behind features): `metal` (gpu_inference),
      `coreml-native` + `coreml-proto` + `prost` (ane). `publish = false`
      (matches every leaf except katgpt-core).
- [ ] **A1:** Move `inference_backend.rs` → `crates/katgpt-backend/src/lib.rs`:
      - `InferenceBackend` trait
      - `CompileError`
      - `CpuBackend` (delegates to `katgpt_forward::forward`)
      - `BackendKind` enum
      - `auto_backend()` selector
      - Rewrite imports: `crate::transformer::*` → `katgpt_forward::*` /
        `katgpt_transformer::*`; `crate::types::*` → `katgpt_types::*`.
      - **Fix the stale line-8 doc comment** — replace the Issue 033 §C note
        with the actual reason this is now a leaf (forward/ForwardContext
        moved to katgpt-forward in Plan 385).
- [ ] **A2:** Move `gpu_backend.rs` → `crates/katgpt-backend/src/gpu.rs`
      (gated `#[cfg(all(target_os = "macos", feature = "gpu_inference"))]`).
      Move the `metal = "0.33"` dep from root `[target.'cfg(target_os = "macos")'.dependencies]`
      into the leaf.
- [ ] **A3:** Move `ane_backend.rs` → `crates/katgpt-backend/src/ane.rs`
      (gated `#[cfg(all(target_os = "macos", feature = "ane"))]`).
      Move `coreml-native`, `coreml-proto`, `prost` deps into the leaf.
- [ ] **A4:** Root `src/lib.rs` re-export for back-compat (mirrors Issue 014/015
      re-export contract used for every prior extraction):
      ```rust
      pub use katgpt_backend as inference_backend;
      ```
      Delete the three root `src/*_backend.rs` files +
      `src/inference_backend.rs`.

### Target B — root stays (documented for completeness)

- [-] `inference_router.rs` — **DEFER, stays root.** Imports
      `crate::trigger_gate`, `crate::pruners::acceptance_variance`,
      `crate::dllm_solver`, `crate::chiaroscuro`, `crate::pipeline_pruner`,
      `katgpt_core::SpeculativeGenerator`. It's the composition layer tying
      backends to tier-selection; moving it would drag 5+ modules into the leaf
      or create a cycle. After A1-A4 it imports `katgpt_backend::InferenceBackend`
      instead of `crate::inference_backend::InferenceBackend` (one-line change).

## 6. Feature-forwarding plan

Root `Cargo.toml` features change from local-impl to forwarded (mirrors the
NFCoT/ppot extraction in Issue 003):

```toml
# Before (local)
ane = ["dep:coreml-native", "dep:coreml-proto", "dep:prost", "kog_cpu_fusion"]
gpu_inference = ["dep:metal", "kog_cpu_fusion"]
inference_router = ["gpu_inference", "ane"]

# After (forwarded to leaf; deps move to katgpt-backend/Cargo.toml)
ane = ["dep:katgpt-backend", "katgpt-backend/ane", "kog_cpu_fusion"]
gpu_inference = ["dep:katgpt-backend", "katgpt-backend/gpu_inference", "kog_cpu_fusion"]
inference_router = ["gpu_inference", "ane"]
```

`crates/katgpt-backend/Cargo.toml`:

```toml
[features]
default = []
ane = ["dep:coreml-native", "dep:coreml-proto", "dep:prost"]
gpu_inference = ["dep:metal"]
```

**Preserve the `kog_cpu_fusion` implication contract**: `ane` and `gpu_inference`
each *imply* `kog_cpu_fusion` because the backends read `attn_norm_gamma` /
`mlp_norm_gamma` / `attn_qkv_fused` unconditionally (documented in root
Cargo.toml). The forwarded form keeps `kog_cpu_fusion` on the right-hand side
of the root feature — the leaf does NOT re-imply it (root owns the kog
forwarding to transformer+forward).

## 7. Test impact

Three test files import `katgpt_rs::inference_backend::*`:

- `tests/bench_176_ane_inference_backend.rs`
- `tests/goat_176_ane_inference_backend.rs`
- `tests/goat_176_trigger_gate.rs`

With the A4 re-export (`pub use katgpt_backend as inference_backend;`), these
resolve unchanged. **Do not rewrite the test imports** — the re-export shim is
the established back-compat pattern. Only touch them if a test goes red.

## 8. Acceptance

- [ ] Issue created (this file).
- [ ] A0: `katgpt-backend` skeleton compiles standalone
      (`cargo check -p katgpt-backend --no-default-features`).
- [ ] A1-A3: trait + 3 device backends moved, imports rewritten to leaf paths.
- [ ] A4: root re-exports `katgpt_backend as inference_backend`; three root
      files deleted.
- [ ] Feature forwards threaded (§6); `kog_cpu_fusion` implication preserved.
- [ ] `cargo check --workspace` green.
- [ ] `cargo check --workspace --all-features` green (combo-regression guard,
      the `merkle_root` lesson class).
- [ ] `cargo test --workspace --lib` green — no behavior change, no public API
      rename (paths resolve via re-export).
- [ ] The 3 `tests/*_176_*.rs` files pass unchanged (re-export holds).
- [ ] Stale Issue 033 doc comment rewritten in the moved trait file.
- [ ] `inference_router.rs` one-line import update (`crate::inference_backend` →
      `katgpt_backend`); stays root, confirmed compiles.
- [ ] Commit on `develop` with `refactor:` prefix.

## 9. Non-goals

- Do NOT move `inference_router.rs` (Target B deferral).
- Do NOT extract the CPU SIMD kernels — they're correctly placed inside
  `katgpt-transformer` / `katgpt-forward` as forward-path code, not a backend.
- Do NOT rename `InferenceBackend` / `BackendKind` / `auto_backend` / `CpuBackend`
  / `GpuBackend` / `AneBackend` — the re-export preserves all historical paths.
- Do NOT touch `src/transformer.rs` (5610 lines, mostly re-exports + composition
  forward variants now). That's a separate refactor concern.

## References

- Issue 003: `issues/003_speculative_module_promotion.md` (template for this
  extraction — same module→crate pattern, same feature-forwarding shape)
- Plan 176: `.plans/176_*` (ANE/GPU backend + inference_router inception)
- Plan 385: forward → katgpt-forward (broke the Issue 033 circular-dep argument)
- Issue 007 Phase F: ForwardContext → katgpt-forward
- Issue 014/015: re-export back-compat contract (applies to A4)
