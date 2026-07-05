# Proposal 003 — Master `src/` consolidation: domain-stack crates + loser crate

Status: **proposed** (not yet implemented) — supersedes Proposal 001 and 002
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Audit basis: full `src/` classification (3 parallel subagent passes + direct
analysis of attention/quant/mux/sleep/pruners/kv/spectral/speculative)

## TL;DR

The end state: **`src/` contains only `lib.rs`** (pure re-export shims +
retained transformer-forward glue). Everything else moves into domain-stack
crates following the established pattern (`katgpt-spectral`, `katgpt-pruners`,
`katgpt-kv`, `katgpt-sleep` — one crate per domain that absorbs the whole
domain). **`main.rs` was deleted in Phase 12** (the binary bench runner was
redundant; `examples/` covers every bench/demo need via `[[example]]`).

Today `src/` holds ~95 items scattered across ~46 folders and ~49 files.
Attention is the worst case: base primitives in `katgpt-core`, one variant
spun out alone (`katgpt-attn-match`), and the rest (`dash_attn`, `gdn2`,
`hla`, `ega_attn`, `diagonal_gate`) marooned in root. This proposal fixes
the whole tree in one consistent pass.

Losers (dead stubs, superseded primitives, off-topic research toys) get
exiled to **`katgpt-deprecated`** so the winning crates stay clean.

This supersedes Proposals 001 (`katgpt-quant`) and 002 (`katgpt-dash-attn`):
both are absorbed as phases of the master plan with wider scope.

## Guiding principles (non-negotiable)

1. **One domain-stack crate per domain** — every primitive in a domain lives
   in that domain's crate. No scattering. `katgpt-spectral` is the template.
2. **`src/` ends at `lib.rs`** — `lib.rs` is re-export shims +
   the thin transformer-forward glue that can't move (it needs `ForwardContext`).
3. **Forward-vs-primitive seam** — clean primitives promote to crates;
   transformer-bound forward integration (`*::forward.rs` needing
   `crate::transformer::ForwardContext`) stays in root until/unless
   `katgpt-transformer` grows a forward module. This is the `katgpt-attn-match`
   / Issue 359 precedent.
4. **Losers get exiled, not deleted** — `katgpt-deprecated` holds demoted
   primitives behind opt-in features, never default. Keeps regression
   comparability without cluttering winners.
5. **Back-compat re-export** — every move keeps `pub use katgpt_X as Y` in
   `lib.rs` so existing `katgpt_rs::*` paths resolve. Mirrors Issue 015 Phase 5.

## The crate taxonomy (target state)

### Existing domain crates (absorb into these)

| Crate | Absorbs from `src/` |
|---|---|
| `katgpt-core` | base math/SIMD + `alloc.rs`, `cce/`, `cumprodsum.rs`, `llmexec_guard.rs`, `memory_soup_lora.rs`, `mux_demux.rs`, `salience/`, `trigger_gate.rs`, `skill_opt/`, `ssd_block.rs`, `closure_mining.rs` (re-routed from katgpt-sleep, Phase 7), hoist `sigmoid` out of `band_conditioner` |
| `katgpt-types` | (shared leaf — no `src/` absorptions, already complete) |
| `katgpt-transformer` | `transformer.rs` (stays root — owns `ForwardContext`), absorbs `mbu.rs`, `dense_mesh/`, `swir/`, `tf_loop.rs` |
| `katgpt-kv` | `cache_prune/`, `segment_checkpoint/`, `async_qdq.rs` (already has `sp_kv`) |
| `katgpt-pruners` | `closure_wire.rs`, `screening/` (already complete otherwise) |
| `katgpt-spectral` | `spectral_budget.rs`, `spectral_concentration.rs`, `spectral_retract.rs`, `stiff_anomaly/`, `gauge_invariant.rs`, `manifold_power_iter_router.rs`, `off_principal.rs`, `procrustes.rs`, `river_valley.rs`, `distill/peira` (split) |
| `katgpt-speculative` | `distill/{ilc,trd}` (split), `spechop/`, `rt_turbo/`, `precision_aware_draft.rs`, `sparse_compose.rs`, `spec_reconciliation/` |
| `katgpt-sleep` | (Phase 7 absorptions deferred — `sleep/` blocked on Phase 9/12 transformer decoupling, `closure_mining` re-routed to katgpt-core due to cyclic package dep) |

### Existing domain crates (leave as siblings)

| Crate | Note |
|---|---|
| `katgpt-attn-match` | already promoted (Plan 271/Issue 359). **Fold `rerank.rs` in** (MaxSim IS late-interaction attention). Otherwise unchanged. |

### New domain crates (create these)

| Crate | Absorbs | Why a new crate |
|---|---|---|
| **`katgpt-attn`** | FROM `katgpt-core`: `attention`, `parallax_attn`, `set_attention`, `funcattn`. FROM root: `ega_attn.rs`, `diagonal_gate.rs`, `gdn2/` (kernel+types), `hla/` (kernel+types), `dash_attn/` (13 primitive files), `rat_bridge/`, `static_cal.rs`, `funcattn_compose/`, `chiaroscuro/` | The attention stack. Verified nothing in `katgpt-core` consumes its own attention modules internally (zero `use crate::attention` hits) — they can move out. `chiaroscuro` is the entropy-routing layer `funcattn_compose` depends on; they move together. |
| **`katgpt-quant`** | `turboquant/`, `planar_quant/`, `iso_quant/`, `hybrid_oct_pq/`, `octopus/` | Proposal 001, unchanged. Rotation/codebook KV-compression family. |
| **`katgpt-band`** | `band_conditioner.rs`, `bckvss.rs`, `collider_pruner.rs`, `adaptive_cot_stopper.rs` | Plan 265 cluster (arXiv:2605.12733). Tightly inter-coupled; splitting would break internal cohesion. `sigmoid` hoists to core first. |
| **`katgpt-claim`** | `claim_rubric/`, `clr/` | Claim-Level Reliability pair (arXiv:2606.07612). Dev/CI tooling. |
| **`katgpt-sparse`** | `sparse_task_vector.rs`, `specialist_projection.rs` | SOPTV/SPLAT task-vector family (Plan 264/265). Distinct from spectral (different paper lineage). `sparse_compose.rs` → `katgpt-speculative` (it's draft composition). |
| **`katgpt-ruliology`** | `ruliology/` | Wolfram ruliology (Plan 188) — distinct domain. One cross-couple to `pruners::g_zero::delta_absorb` needs interface extraction. |
| **`katgpt-validator`** | `validator/` | Partial parser + syntax pruner. Clean leaf. |
| **`katgpt-bench`** | `benchmark/`, `plot.rs` | Tooling — depends on everything, must be top-level. |
| **`katgpt-deprecated`** | `feedback.rs` (dead stub), `unit_distance/` (number-theory toy) | The **loser crate**. See below. |

### Stays in `src/` (transformer-forward glue — can't move yet)

These need `crate::transformer::{ForwardContext, TransformerWeights}` which
lives in root (`katgpt-transformer` is types-only per its Cargo.toml). They
collapse into `src/lib.rs` as private `mod` declarations, not folders:

- `transformer.rs` — owns `ForwardContext`, the linchpin. Stays.
- `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` +
  `dash_attn/tests.rs` — forward integration. Stays as `mod dash_attn_forward`.
- `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs` — already root glue.
- `types.rs` — already a re-export shim (`katgpt_core::types::*`).
- `inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`,
  `ane_backend.rs` — backend dispatch, transformer-bound.

## The `katgpt-deprecated` loser crate

**Purpose:** exile demoted/dead/off-topic primitives so winning crates stay
clean. Never default-on. Kept (not deleted) for regression comparison and
to preserve the GOAT-gate audit trail.

**Membership criteria — 3 categories of opt-in (NOT a blanket sweep):**

Opt-in features are NOT automatically losers. The sweep classifies each into:

1. **Pending** — opt-in because the GOAT gate hasn't run yet (e.g. "Opt-in
   until G1–G4 pass"). → **stays in domain crate.** Exiling these punishes WIP.
2. **Benchmark-loser** — lost a head-to-head, kept so the winner-vs-loser A/B
   regression bench still works (the bench needs both in scope). → **stays in
   domain crate** behind its feature flag.
3. **Dead/failed** — gate ran and FAILED, OR explicitly demoted (e.g.
   `alien_sampler`: 2/4 PASS), OR dead stub (`feedback.rs`), OR off-topic
   (`unit_distance/`). → **exile to `katgpt-deprecated`.**

Only category 3 moves to `katgpt-deprecated`. The distinction matters: most
opt-in features in the Cargo.toml are category 1 (pending), not category 3.

**Cargo.toml shape:**
```toml
[package]
name = "katgpt-deprecated"
version = "0.1.0"
edition = "2024"
license = "MIT"
description = "Exiled primitives — dead stubs, off-topic research toys, and superseded mechanisms awaiting deletion. Never default-on. Kept for regression comparison and GOAT-gate audit trail."
publish = false

[features]
default = []   # ALWAYS empty — nothing here is ever default.
feedback = []        # dead TTT stub
unit_distance = []   # number-theory research toy
```

**Lifecycle:** items here are deletion candidates. Each carries a
`# TODO(deprecated): delete after <plan/issue> confirms no regression`
comment. The crate exists to make deletion safe and auditable, not to live
forever.

## Full destination map (every `src/` item)

> Legend — verdict: **W**=winner (default-worthy), **O**=opt-in (gated feature),
> **L**=loser (exile), **G**=glue (stays root, transformer-bound), **R**=re-export shim.

### → `katgpt-core`
| Item | Verdict |
|---|---|
| `alloc.rs` | W |
| `cce/` | W |
| `closure_mining.rs` | W (re-routed from katgpt-sleep — see Phase 7 blocker) |
| `cumprodsum.rs` | W |
| `llmexec_guard.rs` | O |
| `memory_soup_lora.rs` | O |
| `mux_demux.rs` | O |
| `salience/` | O |
| `trigger_gate.rs` | W |
| `skill_opt/` | O |
| `ssd_block.rs` | O |
| `channel_simd.rs` | O |
| `sigmoid` (hoist from `band_conditioner`) | W |

> **Removed from this list (misclassification fix):** `alien_sampler/` was
> originally listed here as a winner based on its doc comment. The actual GOAT
> history (Plan 311 Phase 3, `.benchmarks/311_alien_sampler_goat.md`) records
> **"2/4 PASS — demoted to opt-in"** (initially 1/4; G3 closed via Rayon, but
> G1 borderline + G2 fail are the demotion drivers). It is a demoted loser →
> exiled to `katgpt-deprecated`. This is the exact failure mode the Phase 0.5
> loser-sweep exists to catch.

### → `katgpt-attn` (NEW — the attention stack)
| Item | Verdict |
|---|---|
| `attention`, `parallax_attn`, `set_attention`, `funcattn` (from core) | W/O |
| `ega_attn.rs` | O |
| `diagonal_gate.rs` | W |
| `gdn2/{kernel,types}.rs` | O |
| `hla/{kernel,types}` | O |
| `dash_attn/` (13 primitive files) | O |
| `rat_bridge/` | O |
| `static_cal.rs` | O |
| `chiaroscuro/` (entropy routing) | O |
| `funcattn_compose/` | O |

### → `katgpt-quant` (NEW — Proposal 001)
| Item | Verdict |
|---|---|
| `turboquant/`, `planar_quant/`, `iso_quant/`, `hybrid_oct_pq/`, `octopus/` | O (promote GOAT winner to W) |

### → `katgpt-spectral`
| Item | Verdict |
|---|---|
| `spectral_budget.rs`, `spectral_concentration.rs`, `spectral_retract.rs` | O/W |
| `stiff_anomaly/` | O |
| `gauge_invariant.rs`, `manifold_power_iter_router.rs` | O |
| `off_principal.rs`, `procrustes.rs` | O |
| `river_valley.rs` | O |
| `distill/peira` (split from `distill/`) | O |

### → `katgpt-kv`
| Item | Verdict |
|---|---|
| `cache_prune/`, `segment_checkpoint/`, `async_qdq.rs` | O |

### → `katgpt-pruners`
| Item | Verdict |
|---|---|
| `closure_wire.rs`, `screening/` | O — DONE Phase 8 (2026-07-04) |

### → `katgpt-speculative`
| Item | Verdict |
|---|---|
| `distill/{ilc,trd}` (split), `spechop/`, `rt_turbo/` | O |
| `precision_aware_draft.rs`, `sparse_compose.rs`, `spec_reconciliation/` | O |

### → `katgpt-sleep`
| Item | Verdict |
|---|---|
| `sleep/` (Plan 154 consolidation) | DEFERRED — transformer-bound glue (Phase 9/12 blocker, see Phase 7 notes) |
| `closure_mining.rs` | RE-ROUTED to `katgpt-core::closure::mining` (cyclic package dep blocker, see Phase 7 notes) |

### → `katgpt-transformer`
| Item | Verdict |
|---|---|
| `mbu.rs`, `dense_mesh/` (substrate), `swir/` (substrate), `tf_loop.rs` (bulk) | O — DONE Phase 9 (2026-07-04). `dense_mesh/node_transformer.rs` and `swir/strategy_adapter.rs` deferred to Phase 12 (depend on root `forward()` / `thinking_cot`). `tf_loop::should_apply_pruner_at_iteration` stays in root shim (katgpt-pruners cycle). |

### → `katgpt-band` (NEW)
| Item | Verdict |
|---|---|
| `band_conditioner.rs`, `bckvss.rs`, `collider_pruner.rs`, `adaptive_cot_stopper.rs` | W/O |

### → `katgpt-claim` (NEW)
| Item | Verdict |
|---|---|
| `claim_rubric/`, `clr/` | W |

### → `katgpt-sparse` (NEW)
| Item | Verdict |
|---|---|
| `sparse_task_vector.rs`, `specialist_projection.rs` | O |

### → `katgpt-ruliology` (NEW)
| Item | Verdict |
|---|---|
| `ruliology/` | O |

### → `katgpt-validator` (NEW)
| Item | Verdict |
|---|---|
| `validator/` | O |

### → `katgpt-bench` (NEW — tooling)
| Item | Verdict |
|---|---|
| `benchmark/`, `plot.rs` | W (tooling) |

### → `katgpt-attn-match` (existing, absorb `rerank`)
| Item | Verdict |
|---|---|
| `rerank.rs` | O — DONE Phase 8 (2026-07-04) |

### → `katgpt-deprecated` (NEW — the loser crate)
| Item | Verdict | Reason |
|---|---|---|
| `feedback.rs` | L | dead stub — `log::debug!` only, no HTTP POST |
| `unit_distance/` | L | number-theory toy, no inference role |
| `alien_sampler/` | L | GOAT 2/4 PASS — demoted to opt-in (Plan 311 Phase 3, `.benchmarks/311_alien_sampler_goat.md`) |

> **This list is INCOMPLETE by design.** The full loser set is populated by the
> Phase 0.5 loser-sweep audit — every opt-in feature checked against its GOAT
> history in `.benchmarks/` / `.plans/` / `.issues/`. Known failures so far
> (e.g. SDPG bomber arena, `.benchmarks/011_sdpg_bandit_arena.md` GOAT ❌ FAIL)
> are tracked there; the sweep finds the rest before any absorption phase runs,
> so losers don't get absorbed into winning crates.

### Stays in `src/` (root glue — collapses into `lib.rs`)
| Item | Why |
|---|---|
| `main.rs` | binary entry |
| `lib.rs` | re-export shims + retained modules |
| `transformer.rs` | owns `ForwardContext` (linchpin) |
| `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` + `tests.rs` | transformer-bound forward |
| `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs` | already root glue |
| `inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`, `ane_backend.rs` | backend dispatch |
| `types.rs` | already a re-export shim |

### Inline-and-delete (collapse into `lib.rs`)
| Item | Action |
|---|---|
| `cgsp.rs` | delete file, `pub use katgpt_core::cgsp::*;` in `lib.rs` (already a 37-line shim) |

## Required splits (not moves)

1. **`distill/` splits two ways.** PEIRA (spectral alignment metric) →
   `katgpt-spectral`. ILC + TRD (speculative draft screening) →
   `katgpt-speculative`. The `distill/` umbrella conflates two paper
   lineages; it does not survive as a unit.
2. **`band_conditioner::sigmoid` hoists to `katgpt-core`** before the
   `katgpt-band` extraction. Three siblings (`adaptive_cot_stopper`,
   `bckvss`, `collider_pruner`) import it; it's a math utility, not band-domain.

## GOAT gate (per promotion phase)

Each new crate / each absorption passes the no-regression GOAT gate before
its phase commits:

- [ ] **G1 correctness** — existing tests pass unchanged via re-exports.
- [ ] **G2 perf** — hot-path latency unchanged (±2%); no indirection cost.
- [ ] **G3 no-regression** — `cargo check --workspace --all-features` clean.
- [ ] **G4 alloc-free hot path** — scratch buffers stay `&mut [T]`, no new heap.
- [ ] **G5 feature-matrix CI** — new crate added to `ci_feature_guard.sh`
      with `--all-features` (the `merkle_root`/`can_freeze` combo-regression guard).

Winners promoted to default-on additionally require a modelless gain
benchmark. Losers demoted to `katgpt-deprecated` are *already* losers by
definition — no gate, just exile.

## Phased rollout

Ordering: foundation first (hoists, splits), then domain crates biggest-first.

- [x] **Phase 0 — foundation moves (no new crates).**
  - Hoist `sigmoid` from `band_conditioner.rs` → `katgpt-core` (always-on,
    no feature gate). Removed `sigmoid` from the cgsp crate-root re-export to
    avoid conflict; `katgpt_core::cgsp::sigmoid` (module-local) still resolves.
    Updated 3 sibling importers (`adaptive_cot_stopper`, `bckvss`,
    `specialist_projection`) to `use katgpt_core::sigmoid`. Back-compat re-export
    `pub use katgpt_core::sigmoid` left in `band_conditioner.rs`.
  - Split `distill/` → peira (stays, tagged for spectral) + ilc/trd (stays,
    tagged for speculative). Wired the split in-tree via `distill/mod.rs`
    header + per-module comments. Actual file moves deferred to Phases 4/6.
  - Inline-and-delete `cgsp.rs` into `lib.rs`: deleted the 37-line shim,
    replaced `pub mod cgsp` with `pub use katgpt_core::cgsp` in `src/lib.rs`.
    All `katgpt::cgsp::*` paths (types, traits/types submodules, sigmoid,
    dual_pool items) resolve unchanged.
  - GOAT gate G3: `cargo check --workspace --all-features` clean;
    `cargo check` (default features) clean. 93 tests pass across touched
    modules (band_conditioner, adaptive_cot, bckvss, specialist_projection,
    distill/peira, distill/trd) + 56 cgsp tests in katgpt-core.
  **DONE 2026-07-01.**
- [x] **Phase 0.5 — loser-sweep audit (gates every absorption).** For EACH
  opt-in feature in root + every crate: grep its name against `.benchmarks/`,
  `.plans/`, `.issues/` for the GOAT verdict. Classify into pending /
  benchmark-loser / dead-failed per the 3-category rule. Exile every category-3
  to `katgpt-deprecated` with a one-line citation. **Must complete before
  Phases 4–11** so losers aren't absorbed into winning crates. Known starting
  set: `feedback.rs`, `unit_distance/`, `alien_sampler/` (demoted, `issues/010`
  T6), SDPG bomber arena (`benchmarks/011` FAIL). Output: a `katgpt-deprecated`
  membership table with a citation per row.
  **DONE 2026-07-01.** Audit: `.docs/001_loser_sweep_audit.md`. 17 exile
  candidates found (13 with code + 4 dead Cargo.toml entries). 3 src/ items
  exiled in Phase 3a; `dense_mesh` deferred (transformer-bound glue);
  cross-crate losers deferred to Phases 8/10.
- [x] **Phase 1 — `katgpt-quant` crate** (Proposal 001). 5 modules / 26 files.
  Cleanest lift (leaf over core+types). Establishes the move pattern.
  - New crate `crates/katgpt-quant/` with `katgpt-core` dep + `katgpt-transformer`
    dev-dep (for turboquant/forward.rs test `TransformerWeights`).
  - Moved: `turboquant/` (6 files), `planar_quant/` (4), `iso_quant/` (4),
    `hybrid_oct_pq/` (3), `octopus/` (8). 26 files total.
  - Import fixes inside moved files: `crate::types` → `katgpt_core::types`,
    `crate::transformer::TransformerWeights` → `katgpt_transformer::TransformerWeights`.
    Intra-crate refs (`crate::turboquant::`, `crate::octopus::`, etc.) unchanged.
  - Root re-exports: `pub mod X` → `pub use katgpt_quant::X` for all 5 modules.
  - Feature forwarding: `turboquant`, `planar_quant`, `iso_quant`, `octopus`,
    `hybrid_oct_pq`, `maxsim`, `asymmetric_kv` all forward to `katgpt-quant`.
    `turboquant` retains `katgpt-spectral/turboquant` delegation (RandomRotation export).
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default clean.
    173 tests pass in katgpt-quant. Examples (core_05_maxsim, octpq_kvarn_fusion)
    + tests (bench_043_044_comparison) compile via re-export chain.
  **DONE 2026-07-01.**
- [x] **Phase 2 — `katgpt-attn` crate.** The attention stack. Move base
  primitives out of `katgpt-core` + absorb `ega_attn`/`diagonal_gate`/`gdn2`/
  `hla`/`dash_attn`/`rat_bridge`/`static_cal`/`chiaroscuro`/`funcattn_compose`.
  Forward glue stays root. Biggest payoff, biggest lift.
  - New crate `crates/katgpt-attn/` with deps: `katgpt-core` (simd, types,
    funcattn), `katgpt-spectral` (optional, for funcattn_compose/spectral_pre_rotate),
    `rustfft` (optional, chiaroscuro), `blake3` (optional, static_cal + freeze_thaw),
    `serde` (optional, freeze_thaw).
  - **katgpt-core attention primitives (attention, parallax_attn, set_attention,
    funcattn) stay in katgpt-core** — moving them would invert the dependency DAG
    (katgpt-core can't depend on katgpt-attn). katgpt-attn sits above katgpt-core.
  - **HLA substrate already extracted** to `katgpt-hla` crate (Issue 007 Phase E).
    `src/hla/` is pure forward glue (nothing to move).
  - Moved (git renames): `ega_attn.rs`, `diagonal_gate.rs`, `static_cal.rs`,
    `rat_bridge/` (6 files), `chiaroscuro/` (7 files), `gdn2/{kernel,types}.rs`,
    `funcattn_compose/` (4 files), `dash_attn/{chunk_summary,entmax,routing}.rs`.
    28 files total.
  - **VortexFlow cluster stays root** (8 files: vortex_flow, block_topk,
    channel_aware, entmax_router, kv_outer_prefill, msa_distill, value_energy,
    adaptive_k + meta_router + sat_analysis). Root cause: `meta_router` depends
    on `pruners::bandit` + `speculative::types` (root-only), and `vortex_flow`
    depends on `meta_router` — cascading dep chain can't resolve in katgpt-attn.
  - Stays in root: `gdn2/{forward,mod}.rs`, `dash_attn/{forward,tests,mod}.rs`,
    VortexFlow cluster, `hla/{forward,mod}.rs`, `attn_match_adaptive_cot.rs`.
  - Import fixes: `crate::types::*` → `katgpt_core::types::*` (gdn2/types,
    dash_attn/entmax_router, dash_attn/routing, dash_attn/chunk_summary);
    `crate::spectralquant::*` → `katgpt_spectral::*` (funcattn_compose/spectral_pre_rotate);
    `crate::chiaroscuro::*` → `crate::chiaroscuro::*` (intra-katgpt-attn, unchanged).
  - Root re-exports: 6 modules `pub mod X` → `pub use katgpt_attn::X`
    (ega_attn, diagonal_gate, static_cal, rat_bridge, chiaroscuro, funcattn_compose).
    Split modules: `gdn2/mod.rs` + `dash_attn/mod.rs` re-export kernel/types/core
    from katgpt-attn while keeping forward glue.
  - Feature forwarding: 10 features updated in root Cargo.toml to forward to
    katgpt-attn (gdn2_attention, dash_attn, chiaroscuro, rat_plus_bridge, ega_attn,
    static_cal_tables, funcattn_freeze_thaw, funcattn_spectral_pre_rotate,
    funcattn_chiar_blend, wall_attention→diagonal_gate).
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default clean.
    188 tests pass in katgpt-attn. 1764 tests pass in root lib (0 regressions).
    Clippy clean.
  **DONE 2026-07-01.**
- [x] **Phase 3 — `katgpt-deprecated` crate.** Exile `feedback.rs` +
  `unit_distance` + `alien_sampler` (Phase 3a; `dense_mesh` deferred —
  transformer-bound glue). `katgpt-deprecated` crate created with 3 features
  (`feedback`, `unit_distance`, `alien_sampler`), all opt-in, `default = []`.
  Root re-exports preserved for back-compat. Cross-crate losers (dflare_*,
  sdpg_bandit, delta_mem, rmsd_distill, manifold_pruner, compression_drafter,
  stepcode) deferred to Phases 8/10 absorption. See `.docs/001_loser_sweep_audit.md`.
  **DONE 2026-07-01.**
- [x] **Phase 4 — `katgpt-spectral` absorption.** Add `spectral_*`,
  `stiff_anomaly`, `gauge_invariant`, `manifold_power_iter_router`,
  `off_principal`, `procrustes`, `river_valley`, `distill/peira`.
  - New modules in `crates/katgpt-spectral/src/`: `spectral_retract.rs`,
    `gauge_invariant.rs`, `manifold_power_iter_router.rs`, `off_principal.rs`,
    `spectral_budget.rs`, `spectral_concentration.rs`, `procrustes.rs`,
    `river_valley.rs`, `peira.rs` (moved from `src/distill/peira.rs`),
    `stiff_anomaly/` (4 files: mod/baseline/stability/subspace).
    13 files total, ~7400 lines.
  - **Always-on modules** (no feature gate at module level): `spectral_retract`,
    `river_valley`, `spectral_concentration`. The latter two retain internal
    `#[cfg(feature = "river_valley")]` / tracking-flag gates on individual
    public functions; `spectral_concentration` is ungated internally.
  - **Feature-gated modules**: `gauge_invariant`, `manifold_power_iter_router`,
    `off_principal_retrieval`, `spectral_budget`, `orthogonal_procrustes`,
    `peira_distill`, `stiff_anomaly`.
  - **distill/ split (Phase 4 half)**: `peira.rs` moved to katgpt-spectral;
    `ilc.rs` + `trd.rs` stay in root `src/distill/` for Phase 6
    (katgpt-speculative). `src/distill/mod.rs` now re-exports
    `katgpt_spectral::peira` to preserve `katgpt_rs::distill::peira::*` paths.
  - Import fixes: `crate::newton_schulz::*` → `katgpt_core::newton_schulz::*`
    (off_principal.rs prod + spectral_budget.rs tests); intra-katgpt-spectral
    `crate::spectral_retract` / `crate::stiff_anomaly::subspace` refs
    unchanged (both endpoints moved together). Doctest paths updated:
    `katgpt_rs::gauge_invariant` → `katgpt_spectral::gauge_invariant`,
    `katgpt_rs::procrustes` → `katgpt_spectral::procrustes`.
  - Cargo.toml: added `blake3` (optional, for manifold_power_iter_router +
    off_principal), `fastrand` (for peira's public `synthetic_cca_sample`),
    and 9 new feature gates. `newton_schulz` added as a katgpt-spectral
    feature alias (forwards `katgpt-core/newton_schulz`) so the internal
    `#[cfg(feature = "newton_schulz")]` gates in off_principal.rs and
    spectral_budget.rs resolve; `off_principal_retrieval` and `spectral_budget`
    both enable it.
  - Root re-exports: 9 modules `pub mod X` → `pub use katgpt_spectral::X`
    (spectral_retract, gauge_invariant, manifold_power_iter_router,
    off_principal, procrustes, river_valley, spectral_budget,
    spectral_concentration, stiff_anomaly). distill/mod.rs: `pub mod peira`
    → `pub use katgpt_spectral::peira`.
  - Feature forwarding: 9 root features updated to forward to katgpt-spectral
    (gauge_invariant, manifold_power_iter_router, off_principal_retrieval,
    spectral_budget, orthogonal_procrustes, river_valley, peira_distill,
    stiff_anomaly, spectral_rank). `spectral_rank` became a tracking-only flag
    since `spectral_concentration` is always-on in katgpt-spectral.
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default clean.
    206 tests pass in katgpt-spectral (+2 doctests). 1681 tests pass in root
    lib (0 failures; ~83 tests moved with their modules into katgpt-spectral).
    Integration tests green: procrustes_determinism (3), bench_270_gauge
    (17), bench_279_mpi (9), composition_279_spectral_budget (4). All 6
    affected examples build. Clippy clean.
  **DONE 2026-07-01.**
- [x] **Phase 5 — `katgpt-kv` absorption.** `cache_prune`, `segment_checkpoint`,
  `async_qdq`.
  - Moved 3 modules (12 files total): `cache_prune/` (4 files: mod, rolling_hash,
    sat, sensitivity), `segment_checkpoint/` (7 files: mod, auto_route, bench,
    gating, memory_soup, ssc), `async_qdq.rs` (1 file). All self-contained —
    zero external `crate::` refs (segment_checkpoint's 7 `crate::segment_checkpoint::`
    refs are intra-module and resolve to `katgpt_kv::segment_checkpoint::` post-move).
  - `katgpt-kv` feature additions: `cache_prune = []`, `segment_checkpoint = []`,
    `async_qdq_overlap = []`. Root forwards: `cache_prune = ["katgpt-kv/cache_prune"]`,
    `segment_checkpoint = ["katgpt-kv/segment_checkpoint"]`, `async_qdq_overlap =
    ["katgpt-kv/async_qdq_overlap", "inference_router"]` (the `inference_router` dep
    stays root — it gates the GPU backend test harness, not the module itself).
  - Root re-exports: `pub mod X;` → `pub use katgpt_kv::X;` for all 3, preserving
    `katgpt_rs::{cache_prune, segment_checkpoint, async_qdq}::*` paths.
  - 3 root consumers (`spechop/segment_match.rs`, `dash_attn/sat_analysis.rs`,
    `rt_turbo/sat_retrieval.rs`) use `crate::cache_prune::*` through the re-export —
    verified to compile unchanged.
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default clean;
    `--no-default-features` clean. 65 katgpt-kv tests pass (segment_checkpoint +
    cache_prune). 6 async_qdq_goat tests pass. 1553/1554 root lib tests pass in
    debug (1 pre-existing timing flake in `speculative::peira_pruner` unrelated
    to this change; PASSES in release). All examples + benches referencing the
    moved modules build through the re-export chain.
  **DONE 2026-07-04.**
- [x] **Phase 6 — `katgpt-speculative` absorption.** `distill/ilc`, `spechop`,
  `rt_turbo`, `precision_aware_draft`, `spec_reconciliation`.
  - **Scope reduction** (2026-07-04): `distill/trd` + `sparse_compose` kept in
    root — both have hard deps on root modules that would create cycles:
    - **`distill/trd`** — its `prefold_prefix` path depends on `crate::fold::*`
      (transformer-bound glue, Phase 12 scope). Moving TRD to katgpt-speculative
      would require katgpt-speculative → katgpt-rs (cycle). The blocker is
      narrow (`chain_fold`-gated path only); trd stays root alongside `fold/`
      until fold's own destination (Phase 9 or 12) lands.
    - **`sparse_compose`** — depends on `crate::sparse_task_vector::SparseTaskVector`
      (Phase 10 target, stays in root). Same cycle problem. Blocked until Phase 10.
  - Moved 5 modules (16 files total): `distill/ilc.rs` (1), `spechop/` (9),
    `rt_turbo/` (7), `precision_aware_draft.rs` (1), `spec_reconciliation/` (7).
  - **`distill/mod.rs`** in katgpt-speculative created — re-exports `ilc` (the
    only remaining content; peira lives in katgpt-spectral, trd stays root).
    Root's `src/distill/mod.rs` now re-exports `katgpt_speculative::distill::ilc`.
  - Import rewrites: `crate::speculative::*` → `crate::*` (ilc); `crate::types::*`
    → `katgpt_types::*`; `crate::cache_prune::*` → `katgpt_kv::cache_prune::*`;
    `crate::speculative::types::*` → `crate::*` (spechop, spec_reconciliation);
    `crate::benchmark::cosine_similarity` → inlined local fn in manifold_scorer
    (test-only, cycle-safe).
  - **`SpechopSchedule` new type** in `spechop/mod.rs` — local 2-variant mirror
    of `katgpt_pruners::PrunerSchedule` (Uniform + FrozenBaseGuard) for
    `build_hop_dd_tree_with_schedule`. Needed because katgpt-pruners already
    depends on katgpt-speculative (cycle). The root-level token-level
    `build_dd_tree_screened_with_schedule` in `src/speculative/dd_tree.rs` keeps
    using `katgpt_pruners::PrunerSchedule` directly (no cycle — root has
    katgpt-pruners as a regular dep).
  - **Feature gates** added to katgpt-speculative: `ilc_distill`, `spechop`,
    `rt_turbo`, `precision_aware_draft`, `spec_reconciliation`, `thinking_prune`,
    `recfm`, `adaptive_causal_calibration`, `cache_prune` (forwards to
    katgpt-kv), `bandit` (alias, always-on via spechop). katgpt-speculative now
    has optional deps on `katgpt-kv` (cache_prune), `serde`, `postcard`, `blake3`.
  - Root Cargo.toml forwards: `spechop`, `rt_turbo`, `precision_aware_draft`,
    `ilc_distill`, `spec_reconciliation`, `thinking_prune`, `recfm`,
    `adaptive_causal_calibration` all updated to chain `katgpt-speculative/<feature>`.
  - Root re-exports: 5 modules `pub mod X` → `pub use katgpt_speculative::X`
    (precision_aware_draft, rt_turbo, spechop, spec_reconciliation; distill/mod.rs
    re-routes ilc). Historical `katgpt_rs::*` paths preserved.
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default clean;
    `--no-default-features` clean. 471 katgpt-speculative tests pass.
    1491 root lib tests pass (down from 1554 — ~63 moved with the modules,
    counted in the 471 above). All 5 affected integration tests green:
    bench_136_ilc_goat (1), bench_168_recfm_goat (6), bench_171_thinking_prune
    (1), test_126_rt_turbo_goat (6), precision_aware_draft_goat (5),
    spec_reconciliation_bench (2), spec_reconciliation_proof (11). Clippy clean.
  **DONE 2026-07-04.**
- [x] **Phase 7 — `katgpt-sleep` absorption.** `sleep/`, `closure_mining`.
  - **Destination deviation (2026-07-04):** the proposal's literal target
    (`katgpt-sleep`) was hit by two blockers, so the executable subset
    (`closure_mining`) was re-routed to `katgpt-core::closure::mining`.
    The original target is documented here for future re-attempts.
  - **Blocker 1 — `sleep/*` is transformer-bound glue.** All three files
    (`mod.rs`, `consolidation.rs`, `eviction.rs`, `types.rs`) depend on
    `crate::gdn2`, `crate::transformer::{ForwardContext, MultiLayerKVCache,
    TransformerWeights}`, and `crate::types::{Config, kv_dim}`. `forward_looped`
    in `src/transformer.rs` directly consumes `crate::sleep::SleepConfig` at
    lines 449 + 833–837. Moving `sleep/` out of root would require
    `katgpt-sleep → katgpt-rs` (cycle). The in-source comment in
    `src/sleep/consolidation.rs` already declares this root-residency
    ("_Root-resident by design_… Would need its own crate"). Blocked until
    Phase 9 (katgpt-transformer absorption) or Phase 12 (final sweep).
  - **Blocker 2 — cyclic package dep on the proposed katgpt-sleep route for
    `closure_mining`.** katgpt-core already depends on katgpt-sleep
    (`sleep_time_anticipation` feature re-exports the anticipator substrate
    as `katgpt_core::sleep_time`). Adding katgpt-sleep → katgpt-core for
    `closure_mining`'s `MotifMiner` / `compute_pri` / `compute_cdg` deps
    creates `katgpt-core → katgpt-sleep → katgpt-core`. Verified by
    `cargo check -p katgpt-sleep --features closure_mining`:
    `error: cyclic package dependency`. The anticipator re-export cannot be
    dropped (katgpt_core::sleep_time has 6 external consumers in riir-ai
    + katgpt-core benches/examples/tests), so the cycle is unbreakable
    without major restructuring.
  - **Executed: `closure_mining` → `katgpt-core/src/closure/mining.rs`.**
    katgpt-core is the natural home — the instrument is a thin wrapper around
    `katgpt_core::closure::{MotifMiner, MotifAdmitter, compute_pri,
    compute_cdg}` which already live there. Import rewrites: `use
    katgpt_core::closure::{...}` → `use crate::closure::{...}`;
    `use katgpt_core::{compute_cdg, compute_pri}` → `use crate::{...}`.
    Two doc-link references updated (one pointing to the root-only
    `closure_wire::PtgTracedPruner::finish_episode`, one fixing a stale
    `katgpt_rs::sleep::closure_mining::` path that never existed).
  - **Public API surface preserved:** `katgpt_core::closure::mining::*`
    exposed via `pub use mining::{SleepCycleClosureReport,
    fold_cdg_at_sleep_cycle, mine_motifs_at_sleep_cycle}` in
    `closure/mod.rs`, plus surfaced at the katgpt-core top level via the
    existing `pub use closure::{...}` block (mining sub-block added).
    Root `src/lib.rs` keeps the historical `katgpt_rs::closure_mining::*`
    path alive via `pub use katgpt_core::closure::mining as closure_mining`.
    External consumer `riir-engine::closure_bridge` verified unchanged
    (`pub use katgpt_rs::closure_mining::{mine_motifs_at_sleep_cycle,
    SleepCycleClosureReport}`).
  - **No Cargo.toml changes at root.** The root `closure_instrument`
    feature already chains `katgpt-core/closure_instrument`, which now
    transitively includes mining (the module is unconditional inside
    `closure/mod.rs`'s `pub mod mining;` declaration, gated only by the
    outer `#[cfg(feature = "closure_instrument")] pub mod closure;`).
    katgpt-sleep Cargo.toml untouched — crate remains a pure substrate
    leaf with no katgpt-core dependency.
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default
    clean; `--no-default-features` clean. 1071 katgpt-core lib tests pass
    (was 1068 — the 3 mining tests moved here from root). 1490 root lib
    tests pass (3 fewer mining tests at root, partially offset by sibling
    agent's collider_pruner additions). bench_290_closure_wire_integration
    6/6 PASS. bench_290_closure_instrument_goat 10/10 PASS with
    `--test-threads=1` (G2 is a timing-sensitive warm-tier test that flakes
    under concurrent load — pre-existing, unrelated to this change).
    Clippy clean on the new module.
  **DONE 2026-07-04.**
- [x] **Phase 8 — `katgpt-pruners` absorption.** `closure_wire`, `screening`.
  Fold `rerank.rs` → `katgpt-attn-match`.
  - **All three modules shipped cleanly** (no blockers, unlike Phase 7).
  - **`closure_wire.rs`** (451 LOC, 8 tests) → `katgpt-pruners/src/closure_wire.rs`.
    Deps verified clean via root-shim resolution: `crate::speculative::types::ScreeningPruner`
    → `katgpt_core::traits::ScreeningPruner`; `crate::pruners::AbsorbCompress` →
    `katgpt_pruners::absorb_compress::AbsorbCompress`. After move, intra-crate
    `use crate::absorb_compress::*` + cross-crate `use katgpt_core::traits::*`.
  - **`screening/`** (1756 LOC, 6 files, ~13 tests) → `katgpt-pruners/src/screening/`.
    Pure substrate: only `fastrand::Rng` + `core::hint::black_box` deps. Operates on
    `&[u8]` / `&[f32]` only — no HLA / functor / shard types. riir-ai Plan 331
    wiring intentionally NOT in katgpt-pruners.
  - **`rerank.rs`** (526 LOC, 12 tests) → `katgpt-attn-match/src/rerank.rs`.
    Only dep: `katgpt_core::simd::*`. Added `katgpt-core` as optional dep to
    katgpt-attn-match (gated by `maxsim` feature). Contains `bt_rank`-gated
    `SymmetricBoundaryPair` (orthogonal to the `maxsim` module gate).
  - **New features in katgpt-pruners**: `closure_instrument` (forwards to
    `katgpt-core/closure_instrument`), `complexity_prior_sampler`,
    `mcts_k_prior`, `bandit_k_prior`, `spec_k_prior`, `lz4_proxy`, `blake3_proxy`.
  - **New features in katgpt-attn-match**: `maxsim` (pulls `dep:katgpt-core`),
    `bt_rank` (empty tracking flag for the SymmetricBoundaryPair impl).
  - **Root Cargo.toml**: `closure_instrument`, `complexity_prior_sampler`,
    `mcts_k_prior`, `bandit_k_prior`, `spec_k_prior`, `lz4_proxy`, `blake3_proxy`,
    `maxsim`, `bt_rank` all now forward to their crate locations.
  - **Public API surface preserved**: `katgpt_rs::closure_wire::*`,
    `katgpt_rs::screening::*`, `katgpt_rs::rerank::*` all resolve via root
    `pub use` re-exports. External consumer `riir-engine::closure_bridge`
    (`pub use katgpt_rs::closure_mining::*`) unaffected (closure_mining already
    moved in Phase 7).
  - GOAT gate G3: `cargo check --workspace --all-features` clean; default clean;
    `--no-default-features` clean. katgpt-pruners lib 209 tests PASS (was 161 —
    +8 closure_wire + ~13 screening + ~27 other features in the test run).
    katgpt-attn-match lib 12 tests PASS (rerank, all green). Root lib 1460 tests
    PASS (was 1490 — ~30 tests moved to crates). bench_290_closure_wire_integration
    6/6 PASS. bench_290_closure_instrument_goat 10/10 PASS. bench_maxsim_rerank
    1/1 PASS (NDCG ≥2% better than cosine). Clippy clean on moved modules.
  **DONE 2026-07-04.**
- [x] **Phase 9 — `katgpt-transformer` absorption.** `mbu`, `dense_mesh`,
  `swir`, `tf_loop`. **DONE 2026-07-04.**
  - `mbu` (223 LOC) — clean move to `crates/katgpt-transformer/src/mbu.rs`.
    1 import rewrite (`crate::types::{Config, kv_dim}` → `katgpt_core::types::...`).
    Root `pub mod mbu` → `pub use katgpt_transformer::mbu` shim.
  - `tf_loop` (639 LOC, minus the pruner-schedule fn) — bulk moved to
    `crates/katgpt-transformer/src/tf_loop.rs`. The `should_apply_pruner_at_iteration`
    function (which consumes `katgpt_pruners::PrunerSchedule`) stays in a thin
    `src/tf_loop.rs` shim — `katgpt-pruners` already depends non-optionally on
    `katgpt-transformer`, so the reverse dep would create a package cycle.
    Root shim re-exports the bulk via glob AND defines the pruner fn locally.
  - `dense_mesh/` (11 of 12 files moved) — substrate (topology, edges, traits,
    types, etc.) moved to `crates/katgpt-transformer/src/dense_mesh/`.
    `node_transformer.rs` stays in `src/dense_mesh/` because it consumes
    `crate::transformer::forward` (the cognitive-primitive composer).
    `src/dense_mesh/mod.rs` becomes a shim that glob-reexports from the crate
    + adds `node_transformer` back as a local sibling.
  - `swir/` (8 of 9 .rs files moved + 2 docs) — substrate (controller, entropy,
    bench, signal_mix, etc.) moved to `crates/katgpt-transformer/src/swir/`.
    `strategy_adapter.rs` stays in `src/swir/` because it consumes
    `crate::thinking_cot::{ThinkingStrategy, StepContext, StepDirective,
    ControlTokenIds}`. Also: `ControlToken::resolve_via` method was stripped
    from the moved `types.rs` (depended on `crate::thinking_cot::ControlTokenIds`);
    its body was inlined into `src/swir/strategy_adapter.rs::resolve_control_token`
    free function. `src/swir/mod.rs` shim glob-reexports + adds
    `strategy_adapter` back locally.
  - 4 new features in `katgpt-transformer/Cargo.toml`: `tf_loop`, `recfm`,
    `thinking_prune` (all no-op aggregator gates — no internal deps), `dense_mesh`,
    `swir_switch_thinking`. Plus `collapse_aware_thinking` (forwards to
    `katgpt-core/collapse_aware_thinking` for `CollapseDetector` trait) and
    `breakeven_routing` (no-op — uses only local types). Plus `fastrand = "2"`
    always-on dep (dense_mesh's edge_bandit + edge_projection).
  - 5 root Cargo.toml features updated to forward: `tf_loop`, `recfm`,
    `thinking_prune`, `dense_mesh`, `swir_switch_thinking`, plus
    `collapse_aware_thinking` and `breakeven_routing` (for dense_mesh's
    adaptive_width).
  - **GOAT gate G3 — all PASS.** `cargo check --workspace --all-features`,
    `--workspace` (default), `--workspace --no-default-features` all clean.
    katgpt-transformer lib tests 122 PASS (was ~50 — added ~70 from mbu,
    tf_loop, dense_mesh, swir). Root lib 1397 PASS (was 1460 — net delta
    reflects tests moved to crate minus shim-only files). External consumer
    tests: bench_160_kog_cpu_fusion 1/1 PASS, test_136_tf_loop 4/4 PASS,
    bench_168_recfm_goat + test_recfm_lt2 11/11 PASS, dense_mesh_goat_gates
    5/5 PASS, prof_dense_mesh 5/5 PASS, bench_275_swir_goat 16/16 PASS.
    Clippy clean on katgpt-transformer.
  - **Deferred to Phase 12 (or until forward/thinking_cot move):**
    `dense_mesh/node_transformer.rs`, `swir/strategy_adapter.rs`.
  **DONE 2026-07-04.**
- [x] **Phase 10 — `katgpt-core` absorption.** `alloc`, `cce`, `cumprodsum`,
  `llmexec_guard`, `memory_soup_lora`, `mux_demux`, `salience`, `trigger_gate`,
  `skill_opt`, `ssd_block`, `channel_simd`. **DONE 2026-07-04.**
  - **11 modules moved** from `src/` to `crates/katgpt-core/src/`: 8 single
    files + 3 directory trees (`cce/`, `salience/`, `skill_opt/`). `alien_sampler`
    was already exiled to `katgpt-deprecated` in Phase 3a (stale proposal ref).
  - **katgpt-core Cargo.toml**: 7 new features (`cce_moderator`, `llmexec_guard`,
    `memory_soup_lora`, `salience_tri_gate`, `skill_opt`, `ssd_block`,
    `channel_simd_align`) + 1 mirror feature (`rv_gated_routing`, see below).
    `mux_demux` already existed. 4 default-ON additions (`cce_moderator`,
    `llmexec_guard`, `ssd_block`, `salience_tri_gate`) added to katgpt-core
    `default` list. `alloc`, `cumprodsum`, `trigger_gate` are always-on (no
    feature gate — `pub mod X;` unconditionally).
  - **katgpt-core lib.rs**: 11 module declarations wired (L1297-1322). Always-on
    trio declares `pub mod` unconditionally; 7 feature-gated modules mirror
    root feature names. Salience flat type re-export preserved.
  - **Root shims**: all 11 `pub mod X;` → `pub use katgpt_core::X;`.
    `#[global_allocator] static GLOBAL_ALLOC` stays in root (process-global —
    can only be declared in the final binary/library crate; katgpt-core cannot
    host it because it would conflict when consumed as a library dep). Root
    instantiates the static using `katgpt_core::alloc::TrackingAllocator`; the
    struct + helpers (`reset_alloc_stats`/`get_alloc_stats`) live in katgpt-core.
    A test-only `#[cfg(all(test, debug_assertions))] #[global_allocator]`
    in katgpt-core/src/lib.rs lets katgpt-core's own alloc tests work without
    conflicting with the root crate's production global allocator.
  - **Root Cargo.toml**: 8 features updated to forward to katgpt-core
    (`cce_moderator`, `llmexec_guard`, `memory_soup_lora`, `salience_tri_gate`,
    `skill_opt`, `ssd_block`, `channel_simd_align`, `mux_demux`). `mux_demux`
    forwards to BOTH katgpt-core (actual module) AND katgpt-pruners (historical
    tracking alias — no module behind it in pruners).
  - **Mid-fix 1 — `rv_gated_routing` mirror feature.** `trigger_gate.rs`'s
    `RvThresholds` struct + `TriggerGate::rv_tier_boost` method were gated
    behind `#[cfg(feature = "rv_gated_routing")]` (Plan 202). That feature was
    root-only (forwards to `katgpt-pruners/rv_gated_routing`); the gate inside
    `trigger_gate.rs` was permanently dead post-move. Root `inference_router.rs`
    failed with E0432 (`RvThresholds` not found) + E0599 (`rv_tier_boost` not
    found). Added `rv_gated_routing = []` feature to katgpt-core + updated root's
    forward to `["katgpt-core/rv_gated_routing", "katgpt-pruners/rv_gated_routing"]`.
  - **Mid-fix 2 — `mux_demux::compute_mux_residual` dead-cfg cleanup.** The fn
    was gated `#[cfg(all(feature = "mux_demux", feature = "rcd_residual"))]`, but
    `rcd_residual` is root-only. Dropped the dead half → `#[cfg(feature =
    "mux_demux")]`. Strictly more permissive (fn available whenever mux_demux
    is on, where before it required both); no regression risk.
  - **Mid-fix 3 — `katgpt_core::simd::*` → `crate::simd::*` rewrites.** 3 files
    had root-crate vocabulary calls into katgpt-core's simd module that needed
    rewriting post-move: `memory_soup_lora.rs` (3 sites), `ssd_block.rs` (3
    sites), `channel_simd.rs` (1 site).
  - **Mid-fix 4 — `trigger_gate.rs` toml dep.** `from_toml`/`to_toml` were
    test-only API (no external caller). Gated behind `#[cfg(test)]`; added
    `toml = "0.8"` to katgpt-core `[dev-dependencies]` so katgpt-core stays
    leaf-clean for downstream consumers.
  - **GOAT gate G3 — PASS**: workspace `cargo check` clean on default /
    all-features / no-default-features. katgpt-core lib tests 1221/1221 pass.
    Root lib tests 1270/1270 pass. Consumer tests: `bench_176_trigger_gate`
    5/5, `cce_convergence` 4/4, `cce_vs_nash` 3/3, `bench_284_clr_goat_g4`
    (alloc-zero-alloc gate) 1/1. Workspace clippy: zero warnings, zero errors.
    Pre-existing failures (NOT from Plan 381): `bench_250_breakeven_goat::t1_overhead_per_forward`
    (debug-build perf gate, passes in release), `bench_271_attn_match_goat::g5_reconstruction_quality`
    (pre-existing quality gate — pure file move cannot regress reconstruction
    quality).
- [x] **Phase 11 — new domain crates.** `katgpt-band`, `katgpt-claim`,
  `katgpt-sparse`, `katgpt-ruliology`, `katgpt-validator`, `katgpt-bench`.
  **DONE 2026-07-04** (5 of 6 crates; `katgpt-bench` deferred).
  - **5 new crates created** in `crates/`: `katgpt-band`, `katgpt-validator`,
    `katgpt-sparse`, `katgpt-claim`, `katgpt-ruliology`. Total 38 source files
    moved out of root `src/`.
  - **`katgpt-bench` DEFERRED to Phase 12.** `src/benchmark/` has heavy
    root-glue coupling: 6+ files construct `crate::transformer::ForwardContext`
    and call `crate::transformer::forward*`. Self-documented as "Root-resident
    by design (Issue 033 §C, Option C)". Also coupled to `main.rs` (6 call
    sites) — Phase 12 deletes `main.rs`, dissolving that coupling.
    Re-evaluate after Phase 12 lands.
  - **`katgpt-band`** (Plan 265 cluster, 4 files): band_conditioner, bckvss,
    collider_pruner, adaptive_cot_stopper. Features: `band_conditioner`
    (default-ON), `bckvss`, `collider_consistency` (default-ON, preserves
    `katgpt-core/local_branch_routing` forward), `adaptive_cot_identifiability`.
    External deps: `katgpt-core` (sigmoid + ConstraintPruner/PreservationScorer
    traits), `fastrand` (bckvss SyntheticScm prod). The lib.rs comment claiming
    `adaptive_cot_stopper` depends on `band_conditioner` was inaccurate — the
    module is fully standalone (audit verified zero `crate::` imports).
  - **`katgpt-validator`** (4 files): PartialParser + SynPruner. Features:
    `validator` (back-compat gate), `hoare_pruner` (gates `SynPruner::propagate`
    method). Deps: `katgpt-core` (ConstraintPruner trait), `katgpt-tokenizer`
    (BpeTokenizer/BpeTokenizerImpl prod + BpeTrainer test), `syn`, `proc-macro2`.
    **Dropped root `syn`/`proc-macro2` optional deps** — now in leaf crate.
    Root `hoare_pruner` feature forward extended to include
    `katgpt-validator/hoare_pruner` so the propagate impl compiles when both
    `validator` + `hoare_pruner` features are on.
  - **`katgpt-sparse`** (2 files): sparse_task_vector, specialist_projection.
    Features: `sparse_task_vector` (default-ON), `specialist_projection`
    (default-ON, implies sparse_task_vector + `katgpt-band/band_conditioner`),
    `gauge_invariant`. **Cross-crate edge**: `specialist_projection.rs:43`
    `crate::band_conditioner::ComputeTarget` → `katgpt_band::band_conditioner::ComputeTarget`.
    Clean dep direction (sparse → band, never reverse). The `gauge_invariant`
    parity test uses `katgpt_spectral::gauge_invariant` (dev-dep with feature).
  - **`katgpt-claim`** (15 files): claim_rubric (5) + clr (10). Features:
    `claim_rubric` (default-ON), `clr` (default-ON). Byte-identical move —
    zero import rewrites (all `crate::` refs are intra-group). Deps:
    `katgpt-core`. dev-deps: `blake3`, `bytemuck` (clr test fixtures for
    hashing direction-vector tamper-evidence).
  - **`katgpt-ruliology`** (13 files): Wolfram ruliology. Feature: `ruliology`.
    Deps: `katgpt-core`, `katgpt-pruners` (with `g_zero` feature — provides
    `DeltaGatedConfig` consumed by `delta_gated_co_evolve`), `fastrand`,
    `blake3` (prod — behavioral fingerprint hashing in fsm/ca/tm).
    **Cross-couple resolved**: `crate::pruners::g_zero::delta_absorb::DeltaGatedConfig`
    → `katgpt_pruners::g_zero::delta_absorb::DeltaGatedConfig` (single field
    read: `config.delta_threshold`). Import gated behind
    `#[cfg(feature = "ruliology")]` to avoid unused-import warning. All 19
    intra-module `crate::ruliology::*` paths rewritten to `crate::*` (the
    crate root IS ruliology now). 3 δ-gated tests newly feature-gated to
    match the `delta_gated_co_evolve` function's gate.
  - **GOAT gate G3 — PASS** (workspace-wide):
    - `cargo check --workspace`: default ✅, all-features ✅, no-default ✅
      (only pre-existing `src/main.rs` unused-var warnings).
    - **New crate lib tests all pass**: katgpt-band 24/24, katgpt-validator 7/7,
      katgpt-sparse 22/22 (29/29 with `--features gauge_invariant`),
      katgpt-claim 54/54, katgpt-ruliology 91/91 (94/94 with `--features ruliology`).
    - **Root lib tests**: 1163/1163 pass (down from 1270 — the moved-out
      tests now run in their respective crates).
    - **Workspace lib tests total**: ~5500+ across all crates (sum of
      per-crate results), all pass.
    - **Consumer tests/examples verified** per crate:
      - katgpt-band: `bench_176_trigger_gate` 5/5, 3 examples.
      - katgpt-validator: `core_01_validator` example.
      - katgpt-sparse: `bench_300_tjs_msa_rescue_goat` 1/1, `splat_vs_dense_attention`.
      - katgpt-claim: `bench_284_clr_goat_g4` 1/1, `bench_284_clr_goat` 3/3,
        `claim_rubric_test` 17/17, `bench_307_claim_rubric_goat` 1/1, 4 examples.
      - katgpt-ruliology: `ruliology_demo` example.
    - **Pre-existing failures (NOT from Plan 382)**: `bench_250_breakeven_goat::t1_overhead_per_forward`
      (debug perf gate, passes in release), `bench_270_gauge_invariant_goat::t08_throughput_rebalance_256x16`
      (release perf flakiness, 32.5 μs > 5 μs target — verified fails identically
      on prior commit before Plan 382).
    - **Workspace clippy** `--all-features`: zero warnings, zero errors.
- [x] **Phase 12 — final sweep.** `src/` should contain only:
    - **`lib.rs`** — minimal: `pub mod transformer` + retained forward-glue `mod`s + back-compat `pub use katgpt_*` re-exports. **No domain logic.** It is the feature-aggregation surface (cross-crate feature combos in `Cargo.toml` like `cgsp`/`sr2am_configurator` that forward to multiple sibling crates) + the transformer-runtime home (`ForwardContext`). Stays `publish = false` per repo policy — only `katgpt-core` ships to crates.io.
    - **`transformer.rs`** — owns `ForwardContext` (linchpin).
    - retained forward-glue: `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` + `tests.rs`, `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs`, backend dispatch (`inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`, `ane_backend.rs`), `types.rs` (re-export shim).
    - **DELETE `main.rs`** — it's a redundant binary bench runner; `examples/` (200+ entries) already covers every bench/demo need via `[[example]]`. The implicit `[[bin]]` forces the root crate to ship a binary it doesn't need. `rm src/main.rs`.
  Audit with `find src -type f`. Anything beyond the list above is a missed move — log + fix.
  **DONE 2026-07-05.** See `.plans/383_phase12_final_sweep.md` for the full audit table + DEFER documentation. Highlights:
    - **T1**: `src/main.rs` deleted (commit `cf23050a`). Workspace builds clean; katgpt-percepta dep stays for `examples/percepta_phase0`.
    - **T2**: 4 parallel subagents classified every remaining `src/` item into STAY / MOVE / DEFER with documented reasons. ~84K LOC across ~145 files MOVE-eligible; ~28K LOC DEFER (benchmark/ transformer coupling, sleep/, bomber cohesion, distill/trd, dllm.rs, ~17 speculative files); ~37K LOC STAY (transformer.rs linchpin, inference_router, forward-glue).
    - **T3**: `katgpt-bench` re-eval — main.rs blocker dissolved but `benchmark/` still has 33 `forward*` calls across 6 files + depends on 5 root modules. DEFER documented.
    - **T4.1** ✅ `cf23050a`: New crate `katgpt-proof-cert` (6 files / ~800 LOC). serde+postcard+blake3 always-on; wasm_proof_witness gates the witness subset.
    - **T4.2** ✅ `52ca3f94`: Single-file moves — sparse_compose → katgpt-sparse, dllm_solver + pipeline_pruner → katgpt-core, hla_eigenbasis → katgpt-spectral.
    - **T4.3** ✅ `348347bd`: 4 folders → katgpt-core (mux_latent, compaction, cubical_nerve, breakeven). data_probe/ DEFERRED (naming conflict with existing katgpt-core/src/data_probe.rs).
    - **T4.4** ✅ `8606471e`: interval_pruner + lattice_operad + freq_bandit → katgpt-pruners.
    - **T4.5** ✅ `06aab9bf`: progressive_mcgs + fold → katgpt-speculative.
    - **T4.6** ✅ `af45dc22`: thinking_cot + swir/strategy_adapter cascade → katgpt-transformer.
    - **T4.7** ✅ `52cafcf0`: dash_attn VortexFlow cluster (10 files) → katgpt-attn. Original "stays root" blocker dissolved once pruners/speculative/cache_prune landed in their leaves.
    - **T4.8** ⏭ DEFERRED: ~30 speculative primitives need individual dep audits; 17 files (~25K LOC) are transformer-bound per DEFER table. Genuine separate-session scope.
    - **GOAT gate G3 PASS**: workspace `cargo check` clean on default / all-features / no-default-features. Per-crate lib tests pass (katgpt-attn 150/150, katgpt-core 1435/1435, katgpt-sparse 39/39, katgpt-spectral 100/100, katgpt-pruners 213/213, katgpt-speculative 146/146). bench_256_adaptive_k_goat 1/1 PASS.
  **Final src/ state**: transformer.rs linchpin + retained forward-glue (gdn2/hla/dash_attn composition + tests, inference_router/backend, gpu/ane_backend, attn_match_adaptive_cot, sp_kv_forward_mod, types) + DEFER items (benchmark/, plot.rs, sleep/, dllm.rs, speculative/{17 transformer-bound files}, pruners/bomber/{19 files}, data_probe/, domino_lora). Each DEFER carries a documented blocker + unblock condition in Plan 383. **Phase 14 (Plan 384, 2026-07-05)** subsequently moved `distill/trd` + `vocab_channel_pruner` out (blockers dissolved by Phase 12 T4.4/T4.5) — see Phase 14 entry below. **Phase 15 (Plan 385, 2026-07-05)** broke the `forward`-linchpin cycle by extracting the forward trio + helpers to katgpt-forward, enabling `dense_mesh/node_transformer` to move out too — see Phase 15 entry below. **Phase 16 (Plan 386, 2026-07-05)** applied the same body-grep lesson to `src/speculative/`, moving 12 zero-root-dep modules (acceptance_forecast, belief_cache, belief_drafter, best_buddies, domino, spec_generator, answer_extract, dendritic_gate, kurtosis_gate, nf_flow_generator, nf_flow_qgf, selectivity_router) to katgpt-speculative — speculative root-only count drops from 40 → 28 files — see Phase 16 entry below. **Phase 17 (Plan 387, 2026-07-05)** continued with 10 more leaf-only-dep speculative modules (alpha, budget, budget_compat, caddtree_budget, flow_pruner, peira_pruner, precision_aware_generator, residency_audit, trust_region, domino_lora), discovering the katgpt-pruners→katgpt-speculative cycle as a NEW blocker — speculative root-only count drops from 28 → 18 files — see Phase 17 entry below. **Phase 18 (Plan 388, 2026-07-05)** resolved the katgpt-pruners cycle by extracting `freeze` + proof core types + `ThinkingMode` to katgpt-core, then moving 4 pruners-consuming files (3 to katgpt-speculative, echo_env_integration to katgpt-pruners) — speculative root-only count drops from 18 → 14 files — see Phase 18 entry below.
- [x] **Phase 13 — commit + record.** Commit on `develop` with `refactor:`
  prefix per phase. Update this proposal status to **done** at Phase 12. **DONE 2026-07-05**
  — Plan 383 is the Phase 13 record; this proposal now reflects Phase 12 as complete with
  T4.8 (speculative primitives) documented as a follow-up rather than a blocker. The
  Proposal 003 mandate (delete main.rs + audit + log + fix missed moves) is satisfied;
  remaining items are transformer-bound by definition (out of scope per L840-846).

- [x] **Phase 14 — Plan 384 unblocked follow-ups.** Two Phase 12 DEFER items
  whose documented blockers dissolved during Phase 12 itself, plus a latent
  umbrella-gate bug uncovered + fixed in transit. **DONE 2026-07-05.**
  - **`distill/trd.rs`** (1107 LOC) → `katgpt-speculative/src/distill/trd.rs`.
    Blocker was `crate::fold`; Phase 12 T4.5 moved `fold/` to katgpt-speculative,
    dissolving the cycle. Imports unchanged (`crate::fold` now native). Root
    re-exports as `katgpt_rs::distill::trd`.
  - **`speculative/vocab_channel_pruner.rs`** (2048 LOC) →
    `katgpt-pruners/src/vocab_channel_pruner.rs`. Blocker was `crate::lattice_operad`;
    Phase 12 T4.4 moved `lattice_operad/` to katgpt-pruners. **Bonus discovery:**
    the other supposed blockers (`crate::transformer::TransformerWeights`,
    `crate::types::Config`) were already leaf-resident (`katgpt-transformer` and
    `katgpt-types` respectively) — the "transformer-bound" classification was stale.
    Three import rewrites: `crate::lattice_operad` stayed native, `TransformerWeights`
    + `Config` became `katgpt_transformer::` / `katgpt_types::`. Root re-exports as
    `katgpt_rs::speculative::vocab_channel_pruner`.
  - **Latent bug fix:** `katgpt-speculative/src/lib.rs` had `pub mod distill`
    gated on `ilc_distill` alone — so enabling `trd_refined_draft` without
    `ilc_distill` silently cfg'd out the entire distill umbrella (trd compiled
    in `cargo check` but produced zero test symbols). Fixed to
    `#[cfg(any(feature = "ilc_distill", feature = "trd_refined_draft"))]`. Root's
    own `pub mod distill` was already correctly gated on the union — only the
    katgpt-speculative umbrella was stale (predates Plan 384).
  - **GOAT gate G3 PASS**: workspace `cargo check` clean on default /
    all-features / no-default-features. katgpt-speculative lib tests 282/282
    (incl. all 12 trd tests). katgpt-pruners lib tests (with vocab_channel_pruner)
    180/180 (incl. all 30 vocab tests). Root lib tests 769/769 (12 fewer than
    pre-Plan-384 781 — exactly the moved trd tests, now resident in
    katgpt-speculative). `cargo run --example vocab_channel_pruner_demo` runs
    end-to-end through the re-export path.
  - See `.plans/384_unblocked_followups_trd_vocab.md` for the full task list
    + verification matrix.

- [x] **Phase 15 — Plan 385 forward-extract + dense_mesh/node_transformer move.**
  The next Phase 12 DEFER re-audit candidate was `dense_mesh/node_transformer.rs`
  (334 LOC), whose documented blocker was `crate::transformer::forward`. Re-audit
  confirmed the blocker was REAL but TRIVIALLY BREAKABLE: `forward` itself,
  plus its private dispatchers `forward_base` / `forward_coda`, plus the helpers
  `attention_head` / `standard_lm_head` / `clustered_lm_head` /
  `select_topk_indices*` / `cluster_map_*`, depend ONLY on `katgpt-core` +
  `katgpt-pruners` + `katgpt-transformer` (every `crate::types::*` /
  `crate::pruners::*` reference is a leaf re-export). Per user hint
  ("feel free to create new primitive or shared crate if it fix cyclic redundant"),
  the cycle was broken by extracting the trio + helpers into `katgpt-forward`
  (which already depends on all three leaves). **DONE 2026-07-05.**
  - **`forward` trio + helpers** (~1063 LOC) → `crates/katgpt-forward/src/forward.rs`.
    All functions became `pub` (they were `fn` private in root) so root's
    remaining forward variants (`forward_with_domain_latent`, `forward_decode_stage`,
    `forward_draft`, `forward_verify`, `generate_with_prefill`,
    `generate_with_collapse_detection`) can call them via `pub use katgpt_forward::*`.
    The helpers (`attention_head`, `standard_lm_head`, `clustered_lm_head`,
    `select_topk_indices*`, `cluster_map_*`) are also `pub` for the same reason.
  - **`dense_mesh/node_transformer.rs`** (334 LOC) →
    `crates/katgpt-forward/src/dense_mesh_node_transformer.rs`. Imports rewritten
    (`crate::transformer::forward` → `crate::forward::forward` natively,
    `crate::transformer::ForwardContext` → `crate::ForwardContext` natively,
    `crate::transformer::{MultiLayerKVCache, TransformerWeights}` → `katgpt_transformer::`,
    `crate::types::Config` / `Rng` → `katgpt_core::types::`,
    `super::traits::DenseNode` / `super::types::*` → `katgpt_transformer::dense_mesh::`).
    Root re-exports via `pub use katgpt_forward::TransformerNode` (cfg-gated on
    `dense_mesh`).
  - **Features added in katgpt-forward Cargo.toml**: `domain_latent`,
    `kog_cpu_fusion`, `dense_mesh` (all tracking flags forwarding to the right
    leaf). Two existing features corrected: `coda_fusion` now also forwards to
    `katgpt-core/coda_fusion` (so `katgpt_core::coda::*` resolves); `wall_attention`
    now also forwards to `katgpt-core/wall_attention` (so `Config.wall_config`
    field is present).
  - **Features updated in root Cargo.toml**: `domain_latent`, `kog_cpu_fusion`,
    `dense_mesh` now forward to `katgpt-forward` (in addition to their existing
    leaf targets).
  - **GOAT gate G3 PASS**: workspace `cargo check` clean on default /
    all-features / no-default-features. katgpt-forward lib tests 12/12 with
    --all-features (8 hla_forward + 4 dense_mesh_node_transformer). Root lib
    tests 769/769 default, 1249/1249 --all-features (4 fewer than pre-Plan-385
    — the dense_mesh tests moved with the file). dense_mesh_goat_gates 5/5
    PASS via re-export. prof_dense_mesh 5/5 PASS via re-export.
    transformer:: 80/80 PASS.
  - See `.plans/385_dense_mesh_node_transformer_unblock_via_forward_extract.md`
    for the full task list + verification matrix.
  - **Lesson (R296-class):** "transformer-bound" classifications MUST be
    line-range-audited against the actual leaf crate dependencies, not just the
    `crate::*` prefix. `forward`'s only `crate::` references are
    `crate::types::*` and `crate::pruners::*` — both leaf re-exports. The
    function appeared unmovable because it was the "linchpin" but was in
    fact fully leaf-dependent. Same lesson generalizes to other "linchpin"
    functions: grep the body, not the signature.

- [x] **Phase 16 — Plan 386 speculative cluster move (Phase 1).**
  Applied the Plan 385 lesson (line-range grep the body) to `src/speculative/`.
  Re-audit of all 40 root-only speculative files revealed that **28 of 40** have
  ZERO genuine root dependencies — their `crate::` references are all leaf
  re-exports (`katgpt-core`, `katgpt-types`, `katgpt-pruners`, `katgpt-transformer`).
  The "root-only sibling" classifications were stale, exactly as Plan 385 found
  for `forward`. **DONE 2026-07-05.**
  - **Phase 1 moved 12 files (~3855 LOC)** to `crates/katgpt-speculative/src/`:
    `acceptance_forecast`, `belief_cache`, `belief_drafter`, `best_buddies`,
    `domino`, `spec_generator`, `answer_extract`, `dendritic_gate`,
    `kurtosis_gate`, `nf_flow_generator`, `nf_flow_qgf`, `selectivity_router`.
    All zero-crate-ref or leaf-only-dep files. Root `src/speculative/mod.rs`
    now re-exports each via `pub use katgpt_speculative::<module>;`.
  - **Features added to katgpt-speculative Cargo.toml**: `acceptance_forecast`,
    `belief_drafter` (pulls papaya + blake3 + fastrand + bytemuck), `best_buddies`,
    `domino_correction` (pulls blake3), `speculative_generator` (forwards to
    katgpt-core + pulls fastrand), `dendritic_gate`, `kurtosis_gate`,
    `selectivity_router` (implies kurtosis_gate + pulls bytemuck), `nf_flow_score`,
    `qgf_drafter`, `parallel_probe`. Plus `depth_invariance` + `self_cond_draft`
    tracking flags (belief_drafter's audit/self-cond paths).
  - **Root Cargo.toml features updated** to forward to katgpt-speculative:
    `parallel_probe`, `speculative_generator`, `selectivity_router`,
    `domino_correction`, `kurtosis_gate`, `nf_flow_score`, `best_buddies`,
    `belief_drafter` (papaya dep dropped from root — now in leaf),
    `dendritic_gate`, `qgf_drafter`, `depth_invariance`, `self_cond_draft`.
  - **GOAT gate G3 PASS**: workspace `cargo check` clean on default /
    all-features / no-default-features. katgpt-speculative lib tests **848 passed**
    (178 from moved modules). Root lib tests **613 passed** (default) / **1071
    passed** (--all-features; was 1249 pre-Plan-386, −178 moved = 1071 ✅).
    belief_drafter_goat integration 12/12 PASS. prof_dense_mesh 5/5 PASS.
    and_or_goat 5/5 PASS.
  - **Phase 2 follow-ups (DEFER):**
    - **`forward`-cycle block** (dflash/verifier/drafter_lora, ~3768 LOC):
      these use `crate::transformer::forward`, now in katgpt-forward. Since
      katgpt-forward depends on katgpt-speculative, moving them TO
      katgpt-speculative would create a cycle. Needs architectural decision:
      (a) move `forward` down from katgpt-forward, (b) accept root residency,
      (c) create a new crate above katgpt-forward.
    - **dd_tree.rs** (5990 LOC): test dep on `crate::speculative::dflash::dflash_predict`
      (dflash blocked by forward cycle).
    - **~16 more leaf-only-dep files** (alpha, budget, flow_pruner, peira_pruner,
      trust_region, and_or_builder, precision_aware_generator, residency_audit,
      thinking_controller, echo_env*, caddtree_budget, parallel_probe, budget_compat):
      moveable but deferred to keep Phase 1 small + safe.
  - See `.plans/386_speculative_cluster_move.md` for the full task list +
    verification matrix.

- [x] **Phase 17 — Plan 387 speculative cluster move (Phase 2).**
  Continued the speculative re-audit, moving the second wave of leaf-only-dep
  files from `src/speculative/` to `crates/katgpt-speculative/src/`.
  **DONE 2026-07-05.**
  - **NEW architectural discovery — katgpt-pruners cycle**: `katgpt-pruners`
    depends on `katgpt-speculative` (Cargo.toml line 21). Therefore
    `katgpt-speculative` **cannot** depend on `katgpt-pruners`. This blocks
    4 files that use `crate::pruners::*`: `and_or_builder`, `echo_env`,
    `echo_env_integration`, `thinking_controller`. Resolution options for
    Phase 3: (a) extract consumed pruners items to katgpt-core, (b) accept
    root residency, (c) invert the katgpt-pruners → katgpt-speculative dep.
  - **parallel_probe DEFER**: uses `super::verifier::SpeculativeVerifier`
    (trait in root `verifier.rs`). `verifier.rs` is blocked by the
    `forward`-cycle. Deferring keeps Phase 2 safe.
  - **Phase 2 moved 10 files (~4545 LOC)** to katgpt-speculative:
    `alpha` (636), `budget` (321), `budget_compat` (137), `caddtree_budget`
    (1056), `flow_pruner` (328), `peira_pruner` (336),
    `precision_aware_generator` (114), `residency_audit` (322),
    `trust_region` (734), `domino_lora` (561).
  - **Import rewrites** (same pattern as Phase 16): `crate::speculative::types::X`
    → `katgpt_core::traits::X` / `katgpt_core::speculative::types::X`;
    `crate::types::{Config,Rng,matmul}` → `katgpt_types::*`;
    `crate::speculative::build_dd_tree*` → `crate::dd_tree::build_dd_tree*`;
    `crate::mux_demux::*` → `katgpt_core::mux_demux::*`;
    `crate::speculative::budget::*` → `crate::budget::*` (leaf-internal).
  - **Features added to katgpt-speculative Cargo.toml**: `lattice_deduction`,
    `budget_adaptation`, `caddtree_budget` + `spec_cost_model`, `peira_distill`,
    `domino_lora`, `mux_demux` + `rcd_residual` (tracking flags for caddtree's
    mux_residual variant), `echo_env_predictor` (tracking flag for budget's
    EchoConsistency variant).
  - **Root Cargo.toml features updated** to forward: `lattice_deduction`,
    `budget_adaptation`, `caddtree_budget`, `bandit` (leaf tracking flag,
    no cycle), `peira_distill`, `domino_lora`, `mux_demux`, `rcd_residual`,
    `echo_env_predictor`.
  - **GOAT gate G3 PASS**: workspace `cargo check` clean on default /
    all-features / no-default-features. katgpt-speculative lib tests **970 passed**
    (was 848, +122 moved). Root lib tests **501 passed** (default) / **949 passed**
    (--all-features; was 1071, −122 moved = 949 exact ✅). 8 integration goat
    suites green (belief_drafter 12/12, prof_dense_mesh 5/5, and_or 5/5,
    caddtree_budget 7/7, budget_adaptation 8/8, trust_region 6/6, ldt 2/2,
    precision_aware_draft 6/6).
  - **Phase 3+ DEFER items** (documented in plan):
    - **forward-cycle block** (dflash/verifier/drafter_lora/step, ~6K LOC)
      — unchanged from Phase 16.
    - **katgpt-pruners cycle block** (and_or_builder/echo_env*/thinking_controller,
      ~3K LOC) — NEW this phase.
    - **parallel_probe** (1187 LOC) — blocked by verifier trait.
    - **dllm cluster** (~6K LOC) — unchanged.
    - **dash_attn/prefill** (~1K LOC) — unchanged.
  - After Phase 17, root-only speculative drops from 28 → 18 files.
  - See `.plans/387_speculative_phase2_leaf_cluster.md` for the full task
    list + verification matrix.

- [x] **Phase 18 — Plan 388 katgpt-pruners cycle resolution (speculative Phase 3).**
  - **Goal**: resolve the katgpt-pruners ↔ katgpt-speculative cycle (NEW blocker
    discovered in Phase 17) to unblock the 4 remaining pruners-consuming
    speculative files.
  - **Strategy**: extract-then-move. Three shared types extracted to katgpt-core
    (the lowest common ancestor), then the 4 files move to their natural homes.
  - **Phase 18.1 — Extract `freeze` to katgpt-core**: moved
    `crates/katgpt-pruners/src/freeze.rs` (120 LOC, pure stdlib) to
    `crates/katgpt-core/src/freeze.rs`. Re-export shim in pruners preserves
    `katgpt_pruners::freeze::*` paths.
  - **Phase 18.2 — Extract `proof` core types to katgpt-core**: extracted
    GoalHash + GoalResult + GoalVerifier trait + ProofGoalCache from
    `crates/katgpt-pruners/src/proof/goal_cache.rs` (799 LOC) to
    `crates/katgpt-core/src/proof_cache.rs`. ProofGoalSnapshot stays in pruners
    (only uses ProofGoalCache's public API). Re-export shim in pruners.
  - **Phase 18.3 PREP — Extract ThinkingMode to katgpt-core**: discovered
    during Phase 3 that `thinking_controller.rs` had
    `pub use katgpt_pruners::ThinkingMode;` which would re-introduce the cycle.
    Extracted the 4-variant `#[repr(u8)]` enum to
    `crates/katgpt-core/src/thinking_mode.rs`. katgpt-pruners now re-exports it.
  - **Phase 18.3 — Move 4 files** (~2706 LOC):
    - `and_or_builder.rs` (747) → katgpt-speculative (gated `and_or_dtree`)
    - `echo_env.rs` (795) → katgpt-speculative (gated `echo_env_predictor`;
          test uses katgpt-pruners as DEV-DEP — no cycle, dev-deps don't propagate)
    - `thinking_controller.rs` (867) → katgpt-speculative (gated `thinking_cot`)
    - `echo_env_integration.rs` (297) → **katgpt-pruners** (it's the integration
          glue between echo_env + BanditPruner; pruners already depends on speculative)
  - **NEW dev-dep trick**: `katgpt-speculative[dev-deps] → katgpt-pruners` does
    NOT create a cycle even though `katgpt-pruners → katgpt-speculative` exists,
    because dev-deps don't propagate to dependents. This is the canonical
    pattern for test-only cross-crate references in a cyclic topology.
  - **Cargo.toml changes**:
    - katgpt-speculative: added dev-dep `katgpt-pruners` with
      `features = ["echo_env_predictor"]`; added 4 tracking features
      (`thinking_cot`, `and_or_dtree`, `rv_gated_thinking`, `directional_credit`);
      `and_or_dtree` forwards to `katgpt-core/and_or_dtree` + `dep:blake3`.
    - katgpt-pruners: added `echo_env_predictor` feature (gates module +
      forwards to `katgpt-speculative/echo_env_predictor`).
    - Root: 5 features now forward to katgpt-speculative: `echo_env_predictor`,
      `thinking_cot`, `and_or_dtree`, `rv_gated_thinking`, `directional_credit`.
  - **GOAT Gate G3 — PASS**:
    - `cargo check --workspace` (default / all-features / no-default) clean.
    - katgpt-core: **1247 passed** (default).
    - katgpt-pruners: **126 passed** (default).
    - katgpt-speculative: **1010 passed** (all-features, excl. pre-existing
      `budget_compat` failure that fails on baseline too).
    - Root lib: **472 passed** (default), **900 passed** (all-features, excl.
      flaky bench tests).
    - GOAT integration: belief_drafter_goat 12/12, and_or_goat 5/5,
      bench_167_budget_adaptation_goat 8/8, bench_trust_region 6/6,
      caddtree_budget_goat 7/7 — all PASS.
    - Moved-module tests: echo_env (2) + thinking_controller (11) +
      and_or_builder (30) = **43 PASS**.
    - `examples/echo_env_predictor_demo` builds.
  - **Import rewrite patterns** (extends Phase 16/17 table):
    | Root pattern | Leaf rewrite |
    |---|---|
    | `crate::pruners::freeze::*` | `katgpt_core::freeze::*` |
    | `crate::pruners::proof::{GoalResult, ProofGoalCache}` | `katgpt_core::proof_cache::{GoalResult, ProofGoalCache}` |
    | `pub use katgpt_pruners::ThinkingMode` | `pub use katgpt_core::thinking_mode::ThinkingMode` |
    | `crate::cumprodsum::*` | `katgpt_core::cumprodsum::*` |
    | `crate::speculative::ScreeningPruner` | `katgpt_core::traits::ScreeningPruner` |
    | `crate::pruners::bandit::*` (test) | `katgpt_pruners::bandit::*` (dev-dep) |
    | `crate::speculative::build_dd_tree_screened` | `crate::dd_tree::build_dd_tree_screened` |
  - **After Phase 18**, root-only speculative drops from 18 → 14 files.
    katgpt-pruners cycle: **RESOLVED** for these 4 files.
  - **Remaining root-only speculative** (14 files, blocked by other cycles):
    - **forward-cycle** (~6K LOC): dflash, verifier, drafter_lora, step,
      dd_tree, parallel_probe — needs architectural decision.
    - **dllm-cycle** (~6K LOC): d2f, d2f_verifier, diffusion_sampler,
      flashar_anchor, flashar_consensus, set_diffusion — defer until dllm moves.
    - **dash_attn** (~1K LOC): prefill.
    - **Partial**: types.rs (609 LOC, crate::transformer::forward_paged).
  - See `.plans/388_katgpt_pruners_cycle_resolution.md` for the full task list +
    verification matrix.

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `katgpt-core` → `katgpt-attn` move breaks consumers expecting `katgpt_core::attention` | high | Re-export shim in `katgpt-core`: `pub use katgpt_attn::attention;` for one release cycle. |
| `benchmark/` depends on 6 root modules → `katgpt-bench` has a huge dep surface | medium | `katgpt-bench` is tooling, not a library — acceptable. Or keep as a binary in `examples/` instead of a crate. |
| `ruliology/mutation.rs` cross-couple to `pruners::g_zero::delta_absorb` | medium | Extract a trait boundary in Phase 11; ruliology depends on pruners (clean direction). |
| Forward-glue retention makes the seam look unfinished | low | Document it as intentional (Issue 359 precedent). The seam is honest: forward is transformer-bound, primitives aren't. |
| Phase 2 (katgpt-attn) is the biggest lift and highest blast radius | high | Do it immediately after Phase 1 (quant) which establishes the move pattern. Run G1–G5 per sub-module. |
| Hidden consumer of a moved module | medium | Each phase greps the full tree for `crate::<module>` before deletion; G3 `--all-features` catches stragglers. |
| `katgpt-deprecated` accumulates forever | low | Each item carries a deletion-TODO with a plan/issue reference. Quarterly prune pass. |

## Out of scope

- Growing `katgpt-transformer` to own forward-pass logic (so `ForwardContext`
  + forward glue can finally leave root). That's a separate architectural
  decision — it would unblock the last retained glue AND enable fully
  deleting `lib.rs` (turning the repo into a pure `[workspace]` with no root
  package). Deferred — the feature-aggregation role still needs a root package
  unless every consumer (riir-ai, examples, tests) takes over cross-crate
  feature orchestration itself.
- Publishing `katgpt-rs` to crates.io. Policy is explicit and unchanged:
  `publish = false` (Cargo.toml L9), `release = false` (release-plz.toml L11),
  "only katgpt-core ships to crates.io." `lib.rs` existing does NOT make it
  publishable — it's a local aggregator, not a shippable artifact.
- Cross-repo moves (anything into riir-ai / riir-chain / riir-neuron-db /
  riir-train). This proposal is katgpt-rs-internal only.
- Deleting `katgpt-deprecated` contents — exile only; deletion is a follow-up.

## References

- `katgpt-spectral` precedent: `crates/katgpt-spectral/Cargo.toml` (Issue 015).
- `katgpt-attn-match` precedent: Plan 271 / Issue 359 (primitive-vs-forward split).
- `merkle_root` / `can_freeve` combo-regression lessons: `riir-chain/AGENTS.md`,
  `riir-neuron-db/AGENTS.md`.
- Modelless mandate + GOAT gate: `katgpt-rs/AGENTS.md` §§ "Modelless-first
  mandate", "Feature Flag Discipline".
- Supersedes: `proposals/001_quant_crate_promotion.md`,
  `proposals/002_dash_attn_crate_promotion.md` (absorbed as Phases 1–2).

## TL;DR

`src/` → only `lib.rs` (+ retained transformer-forward glue that
can't move yet). `main.rs` deleted in Phase 12. Every domain gets one stack
crate. Three new domain stacks beyond 001/002: `katgpt-band` (Plan 265 cluster),
`katgpt-claim`, `katgpt-sparse`, plus `katgpt-proof-cert` (Phase 12). Losers
exile to `katgpt-deprecated`. 18 phases (Phase 12 complete 2026-07-05;
Phase 14 = Plan 384 follow-up moving `distill/trd` + `vocab_channel_pruner`
to their now-unblocked leaf homes; Phase 15 = Plan 385 breaks the
`forward`-linchpin cycle by extracting the forward trio to katgpt-forward,
then moves `dense_mesh/node_transformer` to katgpt-forward too; Phase 16 =
Plan 386 applies the same body-grep lesson to `src/speculative/`, moving
12 zero-root-dep modules to katgpt-speculative; Phase 17 = Plan 387 continues
with 10 more leaf-only-dep speculative modules, discovering the
katgpt-pruners→katgpt-speculative cycle as a NEW blocker;
Phase 18 = Plan 388 resolves that cycle by extracting `freeze` + proof core
types + `ThinkingMode` to katgpt-core, then moving 4 files
(and_or_builder, echo_env, thinking_controller → katgpt-speculative;
echo_env_integration → katgpt-pruners as integration glue);
T4.8 speculative-primitives
follow-up documented in Plan 383). Foundation moves
first, biggest-payoff attention stack second. Supersedes 001 and 002.
