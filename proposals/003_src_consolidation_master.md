# Proposal 003 — Master `src/` consolidation: domain-stack crates + loser crate

Status: **proposed** (not yet implemented) — supersedes Proposal 001 and 002
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Audit basis: full `src/` classification (3 parallel subagent passes + direct
analysis of attention/quant/mux/sleep/pruners/kv/spectral/speculative)

## TL;DR

The end state: **`src/` contains only `main.rs` + `lib.rs`** (pure re-export
shims + retained transformer-forward glue). Everything else moves into
domain-stack crates following the established pattern (`katgpt-spectral`,
`katgpt-pruners`, `katgpt-kv`, `katgpt-sleep` — one crate per domain that
absorbs the whole domain).

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
2. **`src/` ends at `main.rs` + `lib.rs`** — `lib.rs` is re-export shims +
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
| `closure_wire.rs`, `screening/` | O |

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
| `mbu.rs`, `dense_mesh/`, `swir/`, `tf_loop.rs` | O |

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
| `rerank.rs` | O |

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
- [ ] **Phase 8 — `katgpt-pruners` absorption.** `closure_wire`, `screening`.
  Fold `rerank.rs` → `katgpt-attn-match`.
- [ ] **Phase 9 — `katgpt-transformer` absorption.** `mbu`, `dense_mesh`,
  `swir`, `tf_loop`.
- [ ] **Phase 10 — `katgpt-core` absorption.** `alloc`, `cce`, `cumprodsum`,
  `llmexec_guard`, `memory_soup_lora`, `mux_demux`, `salience`, `trigger_gate`,
  `skill_opt`, `ssd_block`, `channel_simd`, `alien_sampler`.
- [ ] **Phase 11 — new domain crates.** `katgpt-band`, `katgpt-claim`,
  `katgpt-sparse`, `katgpt-ruliology`, `katgpt-validator`, `katgpt-bench`.
- [ ] **Phase 12 — final sweep.** `src/` should contain only:
    - **`lib.rs`** — minimal: `pub mod transformer` + retained forward-glue `mod`s + back-compat `pub use katgpt_*` re-exports. **No domain logic.** It is the feature-aggregation surface (cross-crate feature combos in `Cargo.toml` like `cgsp`/`sr2am_configurator` that forward to multiple sibling crates) + the transformer-runtime home (`ForwardContext`). Stays `publish = false` per repo policy — only `katgpt-core` ships to crates.io.
    - **`transformer.rs`** — owns `ForwardContext` (linchpin).
    - retained forward-glue: `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` + `tests.rs`, `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs`, backend dispatch (`inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`, `ane_backend.rs`), `types.rs` (re-export shim).
    - **DELETE `main.rs`** — it's a redundant binary bench runner; `examples/` (200+ entries) already covers every bench/demo need via `[[example]]`. The implicit `[[bin]]` forces the root crate to ship a binary it doesn't need. `rm src/main.rs`.
  Audit with `find src -type f`. Anything beyond the list above is a missed move — log + fix.
- [ ] **Phase 13 — commit + record.** Commit on `develop` with `refactor:`
  prefix per phase. Update this proposal status to **done** at Phase 12.

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

`src/` → only `main.rs` + `lib.rs` (+ retained transformer-forward glue that
can't move yet). Every domain gets one stack crate. Three new domain stacks
beyond 001/002: `katgpt-band` (Plan 265 cluster), `katgpt-claim`,
`katgpt-sparse`. Losers exile to `katgpt-deprecated`. 13 phases, foundation
moves first, biggest-payoff attention stack second. Supersedes 001 and 002.
