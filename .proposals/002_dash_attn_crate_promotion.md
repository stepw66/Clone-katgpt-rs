# Proposal 002 — Promote the DashAttention primitive family to `katgpt-dash-attn`

Status: **SUPERSEDED** by Proposal 003 (`003_src_consolidation_master.md`), Phase 2.
The narrow `katgpt-dash-attn` is widened to the full attention stack
(`katgpt-attn`) absorbing base primitives from core + ega + diagonal_gate +
gdn2 + hla + dash_attn + rat_bridge + static_cal + chiaroscuro + funcattn_compose.
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Predecessor: `.proposals/001_quant_crate_promotion.md`

## TL;DR

`src/dash_attn/` is the largest uncoupled research surface in the tree (16
files, Plan 106/196/256 lineage). Unlike the quant family it is NOT a clean
leaf — the forward-integration layer is hard-coupled to the transformer
runtime. The promotion therefore splits the module in two:

- **`katgpt-dash-attn` crate** — the 13-file primitive/routing layer
  (α-entmax, chunk summaries, VortexFlow routers, MSA distill, KV-outer
  prefill). Clean deps on three existing crates: `katgpt-core`,
  `katgpt-types`, `katgpt-pruners`. No root-crate coupling.
- **Stays in root** — `forward.rs` + `tests.rs` (the 3-file
  transformer-integration layer). Needs `crate::transformer::ForwardContext`
  which lives in the root, not in `katgpt-transformer` (that crate is
  explicitly types-only per its Cargo.toml).

This mirrors the `katgpt-attn-match` split (Plan 271, Issue 359): primitives
get promoted, transformer-bound integration stays home.

## The 16-file module, classified

| File | Layer | Promotion? |
|---|---|---|
| `entmax.rs` | primitive (α=1.5 closed-form) | ✅ crate |
| `chunk_summary.rs` | primitive (learned head_cls summarization) | ✅ crate |
| `routing.rs` | primitive (entmax block routing) | ✅ crate |
| `vortex_flow.rs` | primitive (VortexFlow trait + router enum) | ✅ crate |
| `block_topk.rs` | router (Plan 196 Phase 1) | ✅ crate |
| `channel_aware.rs` | router (Plan 196 Phase 2) | ✅ crate |
| `entmax_router.rs` | router (Plan 196 Phase 1) | ✅ crate |
| `value_energy.rs` | router (Plan 196 Phase 1) | ✅ crate |
| `meta_router.rs` | router (Plan 196 Phase 3, bandit) | ✅ crate |
| `adaptive_k.rs` | router (`msa_adaptive_k`) | ✅ crate |
| `msa_distill.rs` | router (`msa_sparse`, Plan 256) | ✅ crate |
| `kv_outer_prefill.rs` | router (`msa_kv_outer`, Plan 256 P2) | ✅ crate |
| `sat_analysis.rs` | analysis (`dash_attn` + `cache_prune`) | ✅ crate |
| `forward.rs` | **transformer integration** | ❌ stays root |
| `tests.rs` | **integration tests** (needs transformer) | ❌ stays root |
| `mod.rs` | facade | split (crate lib.rs + root shim) |

**Crate closure: 13 files. Root retention: 2 files + facade.**

## Why promote despite the split

1. **Research surface needs its own CI guard.** The crate layer carries
   the heaviest feature matrix in `src/`: `vortex_flow`, `msa_sparse`,
   `msa_per_group`, `msa_kv_outer`, `msa_adaptive_k`, `cache_prune`. This
   is exactly the combo-regression class that bit `riir-chain`
   (`merkle_root` / `can_freeze` lessons). A dedicated crate runs
   `ci_feature_guard.sh` over a bounded matrix instead of the whole root.

2. **`katgpt-pruners` and `katgpt-types` already exist.** The two
   non-core deps the primitive layer needs are already promoted crates:
   - `DashAttnConfig` lives in `katgpt-types/src/enums.rs` L231 (not in
     root `types.rs` — that file just re-exports `katgpt_core::types::*`).
   - `BanditPruner` / `BanditStats` / `BanditStrategy` live in
     `katgpt-pruners/src/bandit.rs` (root `pruners/mod.rs` just does
     `pub use katgpt_pruners::*`).
   So the crate is a leaf over three existing siblings — no new
   foundation work, no circular deps.

3. **The split is precedented.** `katgpt-attn-match` (Plan 271, Issue 359)
   made the identical call: modelless primitives ship in the crate,
   transformer-bound forward integration stays in root. `katgpt-dash-attn`
   follows the same seam.

4. **Reverse consumer exists and is clean.** `src/speculative/prefill.rs`
   L324 consumes `crate::dash_attn::{entmax_1p5, entmax_support}`. After
   promotion it imports from `katgpt_dash_attn` — one call-site update,
   no semantic change.

5. **Modelless mandate — mostly clean.** The entmax routing, chunk
   summaries, VortexFlow routers, MSA distillation, and KV-outer prefill
   are all inference-time deterministic (no gradient descent). The one
   caveat: `chunk_summary.rs` carries *learnable* `head_cls` vectors, but
   they're consumed not trained here — zero-init degrades to mean
   pooling (backward-compatible), and weight mutation is a freeze/thaw
   concern owned elsewhere. Fits the modelless bar.

## Dependency DAG (verified by grep)

```
                         katgpt-core (simd kernels)
                              │
                         katgpt-types (DashAttnConfig)
                              │
           ┌─────────── katgpt-pruners (bandit: BanditPruner, BanditStats, BanditStrategy)
           │                    │
           │   ┌────────────────┼────────────────┐
           ▼   ▼                ▼                ▼
      meta_router.rs      channel_aware    entmax_router   (router family)
                              │
                         vortex_flow.rs ◄──── block_topk, value_energy, msa_distill,
                              │                adaptive_k, kv_outer_prefill  (consumers)
                              │
                          routing.rs ◄──── entmax.rs
                              │
                        chunk_summary.rs
                              │
                         sat_analysis.rs ──► crate::cache_prune::SummedAreaTable  ⚠️ ROOT COUPLE
                                                                              (gated, see Risks)

   ──── ROOT-CRUPLE LAYER (stays out of crate) ────
   forward.rs ──► crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights}  ⚠️ ROOT
   tests.rs   ──► same + crate::types::{Config, Rng}                                            ⚠️ ROOT
```

**External crate deps (clean):** `katgpt-core`, `katgpt-types`, `katgpt-pruners`.
**Root-crate couples (must be severed or gated):**
- `sat_analysis.rs` → `crate::cache_prune::SummedAreaTable` — gated
  `dash_attn` + `cache_prune`. Promotion either gates this file behind a
  re-export shim, or moves `SummedAreaTable` into `katgpt-pruners` (it's
  a pruner-adjacent primitive). Phase 0 decides.
- `forward.rs` / `tests.rs` → `crate::transformer::*`. Stays in root.

## Naming decision: `katgpt-dash-attn`

Distinct from:
- `katgpt-core::attention` / `parallax_attn` / `set_attention` / `funcattn`
  (base attention primitives — the substrate layer).
- `katgpt-attn-match` (Attention Matching KV compaction, Plan 271 — a
  different paper, a different concern: NNLS-based cache compaction, not
  α-entmax sparse routing).

`katgpt-dash-attn` is the "adaptive sparse hierarchical attention via
α-entmax routing" family. No collision.

## Proposed crate shape

```
crates/katgpt-dash-attn/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs              # facade, per-feature re-exports
    ├── entmax.rs
    ├── chunk_summary.rs
    ├── routing.rs
    ├── vortex_flow.rs
    ├── block_topk.rs
    ├── channel_aware.rs
    ├── entmax_router.rs
    ├── value_energy.rs
    ├── meta_router.rs
    ├── adaptive_k.rs
    ├── msa_distill.rs
    ├── kv_outer_prefill.rs
    └── sat_analysis.rs     # gated on cache_prune bridge (Phase 0 decision)
```

### `Cargo.toml` skeleton

```toml
[package]
name = "katgpt-dash-attn"
version = "0.1.0"
edition = "2024"
license = "MIT"
description = "Adaptive sparse hierarchical attention via α-entmax routing — DashAttention primitives (Plan 106/196/256): entmax-1.5, learned chunk summaries, VortexFlow router family (BlockTopK, ChannelAware, EntmaxRouter, ValueEnergy, MetaRouter bandit-selected), MSA distill, KV-outer prefill. All modelless. Spun out of katgpt-rs/src/dash_attn/. Transformer-bound forward integration stays in root."
repository = "https://github.com/katopz/katgpt-rs"
keywords = ["attention", "sparse-attention", "entmax", "vortexflow", "modelless"]
categories = ["algorithms", "science"]
publish = false  # mirror katgpt-attn-match — only katgpt-core ships to crates.io

[dependencies]
katgpt-core    = { path = "../katgpt-core" }    # SIMD kernels
katgpt-types   = { path = "../katgpt-types" }   # DashAttnConfig
katgpt-pruners = { path = "../katgpt-pruners" } # BanditPruner family (meta_router)

[features]
default = []

# Mirror the historical root feature surface. These are tracking flags —
# the root crate forwards them via `feature = ["katgpt-dash-attn/feature"]`
# so existing #[cfg(feature = "...")] branches compile unchanged.
dash_attn      = []
vortex_flow    = []
msa_sparse     = []
msa_per_group  = []
msa_kv_outer   = []
msa_adaptive_k = []
cache_prune    = []   # gates sat_analysis.rs SummedAreaTable bridge
```

### Root `katgpt-rs` wiring (back-compat re-export)

Mirror Issue 015 Phase 5 / Issue 359:

1. `Cargo.toml` gains `katgpt-dash-attn = { path = "crates/katgpt-dash-attn" }`.
2. `src/lib.rs` replaces `pub mod dash_attn;` with:
   ```rust
   pub use katgpt_dash_attn as dash_attn;   // primitive layer
   mod dash_attn_forward;                    // root-retained forward.rs + tests.rs
   pub use dash_attn_forward::{
       forward_dash_attn_prefill, forward_dash_attn_decode,
       forward_dash_attn_decode_vortex,
   };
   ```
3. `src/speculative/prefill.rs` L324 `use crate::dash_attn::{...}` —
   unchanged path (hits the re-export), zero call-site churn.

## GOAT gate (must pass before any default-on promotion)

Per `AGENTS.md` feature-flag discipline. Note: `dash_attn` is **already
default-on at root** today, so the gate here is a *no-regression* gate on
the move, not a fresh-promotion gate.

- [ ] **G1 correctness** — all existing `dash_attn/tests.rs` integration
      tests pass unchanged against the re-exported crate (prefill → decode
      round-trip, chunk-summary caching, vortex decode path).
- [ ] **G2 perf** — entmax-1.5 routing wall time and VortexFlow router
      selection latency unchanged (±2%) vs in-tree baseline. The crate
      boundary must not add an indirection cost on the hot path.
- [ ] **G3 no-regression** — `cargo check --workspace --all-features`
      clean. The full feature matrix (`vortex_flow` × `msa_*` ×
      `cache_prune`) compiles. This is the combo-regression guard.
- [ ] **G4 alloc-free hot path** — entmax threshold search, routing
      scratch reuse (`RoutingScratch`, `VortexScratch`), and chunk-summary
      `summarize_chunk_into` stay zero-alloc. No new `Vec` inside the
      routing kernels (scratch buffers passed as `&mut [T]`).
- [ ] **G5 feature-matrix CI** — new `ci_feature_guard.sh` entry for the
      crate runs `--all-features` to catch the `merkle_root`-class bug
      (a field forgotten in one feature combo).

No "promote winner" step here — `dash_attn` is a *family*, not a
competition. The gate is purely no-regression on the move.

## Phased rollout

- [ ] **Phase 0 — decide the two boundary calls.**
      1. `sat_analysis.rs` ↔ `crate::cache_prune::SummedAreaTable`: gate
         behind a root shim OR move `SummedAreaTable` into
         `katgpt-pruners`. Prefer the shim (smaller blast radius).
      2. `forward.rs` / `tests.rs`: confirm they stay in root under a
         `dash_attn_forward` module. Document the seam.
- [ ] **Phase 1 — scaffold crate.** Create `crates/katgpt-dash-attn/`,
      copy the 13 primitive files verbatim, rewrite `use crate::dash_attn::`
      → `crate::` (intra-crate), `use crate::types::DashAttnConfig` →
      `use katgpt_types::DashAttnConfig`, `use crate::pruners::bandit::`
      → `use katgpt_pruners::bandit::`. `cargo check -p katgpt-dash-attn
      --all-features` clean.
- [ ] **Phase 2 — root re-export + forward retention.** Add crate to
      root `Cargo.toml`; replace `pub mod dash_attn;` with the re-export
      + retained-forward pattern above. Move `forward.rs` + `tests.rs`
      into `src/dash_attn_forward.rs` (or keep the dir, gut the primitives).
      `cargo check --workspace` clean.
- [ ] **Phase 3 — update reverse consumer.** Verify
      `src/speculative/prefill.rs` L324 still resolves via the re-export.
      Run `cargo test -p katgpt-rs --features spec_pruner` (or whichever
      feature gates that path) — green.
- [ ] **Phase 4 — delete in-tree primitive copies.** Remove the 13
      primitive files from `src/dash_attn/`. Only `forward.rs` /
      `tests.rs` (now under `dash_attn_forward`) remain in-tree.
      `cargo check --workspace --all-features` clean.
- [ ] **Phase 5 — GOAT no-regression gate.** Run G1–G5. Existing
      `dash_attn/tests.rs` integration tests pass unchanged.
- [ ] **Phase 6 — CI guard.** Add the crate to
      `scripts/ci_feature_guard.sh` with `--all-features`.
- [ ] **Phase 7 — commit + record.** Commit on `develop` with `feat:`
      prefix. Update this proposal status to **done**. Cross-link from
      `katgpt-attn-match` README (sibling primitive-promotion precedent).

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `sat_analysis.rs` → `cache_prune::SummedAreaTable` root couple breaks the leaf claim | medium | Phase 0 shim: gate `sat_analysis` behind a root-provided trait, OR move `SummedAreaTable` to `katgpt-pruners` (it's a SAT primitive, pruner-adjacent). Prefer shim — smaller blast radius. |
| `meta_router.rs` bandit dep pulls heavy `katgpt-pruners` feature surface | low | `katgpt-pruners` is already a sibling crate; gate `meta_router.rs` behind `vortex_flow` (already the case) so consumers who don't want bandits don't pay. |
| `forward.rs` retention creates a confusing split (primitives in crate, forward in root) | low | This is the `katgpt-attn-match` precedent — document the seam in the crate README. The split is honest: forward integration is transformer-bound, primitives aren't. |
| `chunk_summary.rs` `head_cls` is "learnable" — modelless mandate tension | low | Vectors are *consumed* not *trained* here. Zero-init → mean pooling (backward-compatible). Weight mutation is a freeze/thaw concern owned by the runtime, not this crate. Note in README. |
| Hidden consumer of `crate::dash_attn::*` beyond `speculative/prefill.rs` | medium | Phase 0 grep across whole tree; update all call sites to the re-export path. |
| Feature combo regression (the `merkle_root` class) | medium | G5 CI guard with `--all-features`; the `msa_*` × `vortex_flow` matrix is exactly the failure mode this catches. |

## Out of scope

- Moving `ForwardContext` / `MultiLayerKVCache` into `katgpt-transformer`.
  That crate is explicitly types-only today ("No transformer-forward
  composition logic lives here"). Growing it is a separate decision.
- Promoting the `forward.rs` integration layer. It's transformer-bound;
  it stays in root until/unless `katgpt-transformer` grows a forward module.
- `funcattn_compose/` promotion — separate concern, not part of the
  DashAttention family.

## References

- `katgpt-attn-match` precedent: `crates/katgpt-attn-match/Cargo.toml`
  (Plan 271, Issue 359) — identical primitive-vs-integration split.
- `katgpt-pruners` bandit surface: `crates/katgpt-pruners/src/bandit.rs`
  (already a crate, consumed via root re-export today).
- `DashAttnConfig` home: `crates/katgpt-types/src/enums.rs` L231.
- `katgpt-transformer` scope: its `Cargo.toml` description —
  "Transformer substrate types ... No transformer-forward composition
  logic lives here (that stays in katgpt-rs root)."
- Modelless mandate: `katgpt-rs/AGENTS.md` §"Modelless-first mandate".
- GOAT gate: `katgpt-rs/AGENTS.md` §"Feature Flag Discipline".
- `merkle_root` / `can_freeze` combo-regression lesson:
  `riir-chain/AGENTS.md` §"The `merkle_root` Lesson",
  `riir-neuron-db/AGENTS.md` §"The `can_freeze` Audit Lesson".

## TL;DR

Promote the 13-file DashAttention **primitive/routing layer** into
`katgpt-dash-attn` (clean leaf over `katgpt-core` + `katgpt-types` +
`katgpt-pruners`). The 2-file **transformer-integration layer**
(`forward.rs`, `tests.rs`) stays in root because `ForwardContext` lives
in root and `katgpt-transformer` is types-only. This mirrors the
`katgpt-attn-match` split (Plan 271 / Issue 359). Gate is no-regression
on the move, not fresh-promotion — `dash_attn` is already default-on.
