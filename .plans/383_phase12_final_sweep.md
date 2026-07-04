# Plan 383 — Proposal 003 Phase 12: Final `src/` sweep

## Context

Proposal 003 Phases 0–11 are DONE (commits through `90c52bea` on `develop`).
Phase 12 is the **final sweep**: delete `main.rs`, audit `src/` for missed
moves, re-evaluate `katgpt-bench`, and either move or document every
remaining item.

The proposal's destination map was an **aspirational sample**, not an
exhaustive inventory. Phases 0–11 only moved what was explicitly listed.
The remaining `src/` content (~120K LOC across ~190 files) includes both
(a) items the proposal explicitly marked "stays" (transformer-forward
glue) and (b) items the proposal never classified (the true "missed
moves" Phase 12 must catch).

The proposal's Phase 12 instruction is: *"Audit with `find src -type f`.
Anything beyond the [stay] list is a missed move — log + fix."*

## Goals

1. **DELETE `main.rs`** — explicitly mandated. The binary bench runner is
   redundant; `examples/` (200+ entries) covers every bench/demo need via
   `[[example]]`. Deleting it dissolves the `katgpt-bench` coupling.
2. **Audit + classify every remaining `src/` item** into:
   - **STAY** — transformer-bound glue (`ForwardContext`, forward
     integration, backend dispatch). Documented with reason.
   - **MOVE** — no transformer-bound deps; relocate to the proper crate.
   - **DEFER** — has a real blocker (cycle, cross-crate dep). Documented
     with the blocker + the unblock condition.
3. **Re-evaluate `katgpt-bench`** after `main.rs` is gone.
4. **Move the cleanly-movable items** (the easy wins).
5. **GOAT gate G3** — `cargo check --workspace` clean on default +
   all-features + no-default-features; clippy clean.
6. **Update Proposal 003** Phase 12 → DONE with the audit table.

## Non-goals

- Moving every missed module in one pass. The proposal's "log + fix"
  language is `log` first; items with hard blockers stay root with a
  documented reason. The audit table itself is the Phase 12 deliverable
  for those items.
- Growing `katgpt-transformer` to own `ForwardContext` (Out of scope per
  proposal L840-846).
- Cross-repo moves (Out of scope per proposal L851-852).

## Tasks

- [x] **T0** — Plan file + initial `find src -type f` audit captured.
      Initial inventory: ~190 files / ~120K LOC. Audited via 4 parallel
      subagents (one per cluster). Results: ~84K LOC across ~145 files
      are MOVE-eligible; ~36K LOC DEFER (benchmark/ transformer coupling,
      sleep/, bomber cohesion, distill/trd, sparse couple); ~30K LOC STAY
      (transformer.rs linchpin, inference_router, forward-glue).
- [x] **T1** — DELETE `src/main.rs`. Verified workspace builds clean
      (cargo check --workspace, isolated CARGO_TARGET_DIR). katgpt-percepta
      dep stays — examples/percepta_phase0 still uses it.
- [x] **T2** — Per-module classification audit (parallel subagents):
      Audit captured (4 subagents, one per cluster). Summary table in
      `## Audit results` below. **MOVE-eligible: ~84K LOC across ~145 files.**
- [x] **T3** — Re-evaluate `katgpt-bench`: main.rs blocker is DISSOLVED,
      but `benchmark/` still has 33 `forward*` calls across 6 files +
      depends on 5 root modules (speculative/, hla/, pruners/, dllm*,
      breakeven/, inference_backend). DEFER with updated blocker
      (see DEFER table).
- [ ] **T4** — Move cleanly-movable items per the priority list above.
      One commit per cluster (`refactor:` prefix). Each move: file rename
      + import rewrite + root shim re-export + feature forward + GOAT gate.
      Sub-tasks (one per cluster):
      - [x] T4.1 — Create `katgpt-proof-cert` crate + move proof_cert/ (6 files) ✅ cf23050a
      - [x] T4.2 — Single-file moves: sparse_compose, dllm_solver, pipeline_pruner, hla_eigenbasis ✅
        (dllm_solver + pipeline_pruner → katgpt-core; sparse_compose → katgpt-sparse;
        hla_eigenbasis → katgpt-spectral. Root re-exports as `pub use`. Features
        forwarded. 5 dllm_solver sub-features added to katgpt-core
        [q_sample_solver, self_cond_draft, mbr_tree_select, d2f_3sr_warm_start,
        rcd_residual]. katgpt_core:: refs in dllm_solver.rs rewritten to crate::.)
      - [x] T4.3 — katgpt-core folders: mux_latent, compaction, cubical_nerve, breakeven ✅
        (4 folders moved to katgpt-core. data_probe/ DEFERRED — naming conflict
        with existing katgpt-core/src/data_probe.rs (sink_aware_attn module,
        Plan 287); root src/data_probe/ is the original Plan 141 diagnostics
        module with 8 files. Rust can't have both data_probe.rs and data_probe/
        in the same crate. Needs a rename or merge before moving.)
      - [x] T4.4 — katgpt-pruners: interval_pruner, lattice_operad, freq_bandit ✅
        (interval_pruner/ 3 files, lattice_operad/ 5 files, freq_bandit.rs 1 file →
        katgpt-pruners. freq_bandit.rs imports rewritten: crate::trigger_gate →
        katgpt_core::trigger_gate, crate::types::Rng → katgpt_core::types::Rng.
        rv_bandit_pruning sub-feature forwarded. Stale check-cfg allowlist for
        freq_bandit removed — it's now a real feature.)
      - [x] T4.5 — katgpt-speculative: progressive_mcgs, fold ✅
        (progressive_mcgs/ 9 files — self-contained, zero deps. fold/ 8 files —
        import rewrites: crate::still_kv → katgpt_kv::still_kv, crate::types::Rng
        → katgpt_core::types::Rng, crate::speculative::types::ScreeningPruner →
        katgpt_core::traits::ScreeningPruner. chain_fold feature pulls katgpt-kv/
        still_kv + half + fastrand optional deps. still_kv + bake_precision
        sub-features added.)
      - [x] T4.6 — katgpt-transformer cascade: thinking_cot + swir/strategy_adapter ✅
        (thinking_cot/ 2 files + swir/strategy_adapter.rs moved to katgpt-transformer.
        Cascade unblocked: thinking_cot was the blocker that kept strategy_adapter
        in root. Now both live in katgpt-transformer; crate::thinking_cot resolves
        within the crate. Root swir/ shim deleted; root re-exports directly from
        katgpt_transformer::swir + katgpt_transformer::thinking_cot.)
      - [ ] T4.7 — katgpt-attn: 8 dash_attn primitives (+ dep wiring for meta_router, sat_analysis)
      - [ ] T4.8 — katgpt-speculative/katgpt-pruners: ~30 speculative primitives
- [ ] **T5** — Update `src/lib.rs` comments to reflect the final
      classification. Each `pub mod X` for a STAY item gets a one-line
      "stays root because <reason>" comment.
- [ ] **T6** — GOAT gate G3 (workspace):
      - `cargo check --workspace` (default)
      - `cargo check --workspace --all-features`
      - `cargo check --workspace --no-default-features`
      - `cargo test --workspace --all-features` (or per-crate lib tests)
      - `cargo clippy --workspace --all-features`
- [ ] **T7** — Update Proposal 003:
      - Mark Phase 12 `[x]` with audit table + DONE date.
      - Mark Phase 13 `[x]` (this plan IS the Phase 13 record).
      - Update proposal status to **done**.
- [ ] **T8** — Commit on `develop` (per global rule). Use `refactor:`
      prefix for code moves, `docs:` for plan/proposal updates.

## Audit results

Run via 4 parallel subagents (read-only). Verdict summary:

| Cluster | STAY (LOC) | MOVE (LOC) | DEFER (LOC) | Notes |
|---|---:|---:|---:|---|
| A: speculative + dllm + proof_cert | ~26,816 | ~13,918 | ~3,168 | proof_cert/ is 797-LOC clean crate candidate; many speculative primitives are zero-dep |
| B: pruners/mcgs/fold/breakeven/interval/lattice/freq | ~1,436 | ~10,239 | ~17,140 | bomber DEFERs are transformer-free but intra-module-coupled; clean wins elsewhere |
| C: dash_attn/mux_latent/compaction/cubical_nerve/data_probe/hla_eigenbasis/pipeline_pruner | ~488 | ~18,653 | 0 | Zero DEFERs; all primitives movable; only 2 dash_attn glue files stay |
| D: transformer/sleep/thinking_cot/swir/dense_mesh/inference_router/tf_loop/benchmark/plot | ~8,857 | ~683 | ~7,776 | thinking_cot+swir/strategy_adapter is a clean cascade; benchmark/ still blocked |
| **TOTAL** | **~37,597** | **~43,493** | **~28,084** | |

### Phase 12 execution priority (MOVE clusters)

Ordered by (a) clean dep surface, (b) per-cluster LOC, (c) blast radius.

1. **New crate: `katgpt-proof-cert`** — 6 files / 797 LOC / zero non-std deps. Cleanest crate creation.
2. **Single-file moves** (no dep wiring):
   - `sparse_compose.rs` (625) → katgpt-sparse
   - `dllm_solver.rs` (1906) → katgpt-core
   - `pipeline_pruner.rs` (192) → katgpt-core
   - `hla_eigenbasis.rs` (984) → katgpt-spectral
3. **Clean folders → katgpt-core** (zero non-std, non-intra deps):
   - `mux_latent/` (12 files / ~3K LOC)
   - `compaction/` (12 files / ~4.3K LOC)
   - `cubical_nerve/` (5 files / ~2K LOC)
   - `data_probe/` (8 files / ~2.5K LOC)
   - `breakeven/` (2 files / ~1.2K LOC)
4. **Clean folders → katgpt-pruners** (deps already in katgpt-core/katgpt-pruners):
   - `interval_pruner/` (3 files / ~1.3K LOC)
   - `lattice_operad/` (5 files / ~882 LOC)
   - `freq_bandit.rs` (907 LOC) — also resolves a Cargo.toml-noted cyclic concern
5. **Clean folders → katgpt-speculative**:
   - `progressive_mcgs/` (9 files / ~3K LOC) — completely self-contained
   - `fold/` (8 files / ~2K LOC) — needs katgpt-speculative → katgpt-kv dep add
6. **Cascade → katgpt-transformer**: `thinking_cot/` (153 LOC, zero deps) +
   `swir/strategy_adapter.rs` (530 LOC). Substrate side explicitly waits
   for thinking_cot to move.
7. **dash_attn primitives → katgpt-attn**: 8 files. 6 are zero-dep; 2 need
   dep wiring (meta_router needs katgpt-pruners dep; sat_analysis needs
   katgpt-kv dep).
8. **Speculative primitives → katgpt-speculative / katgpt-pruners**: ~30
   files, ~13K LOC. Zero-dep primitives + leaf-trait primitives.

### DEFER items (documented blockers)

| Item | LOC | Blocker | Unblock condition |
|---|---:|---|---|
| `benchmark/` + `plot.rs` | ~6.9K | 33 `forward*` calls across 6 files + 5 root modules (speculative/, hla/, pruners/, dllm*, breakeven/, inference_backend) | Move the root modules first, then either move transformer.rs forward family (out of scope) or rewrite benchmark to call leaf-resident forward variants |
| `sleep/` | 707 | Phase 7 blocker: ForwardContext + crate::gdn2 root deps | forward family + gdn2 relocate first |
| `dense_mesh/node_transformer.rs` | 334 | Calls root `crate::transformer::forward` composer | transformer.rs linchpin move (out of scope) |
| `distill/trd.rs` + `distill/mod.rs` | 1142 | chain_fold-gated prefold_prefix depends on `crate::fold` | fold moves first (this plan T4.5 — TRD unblocks after) |
| `sparse_compose.rs` (revisit) | — | was DEFER; now MOVE-eligible (sparse_task_vector in katgpt-sparse since Phase 11) | unblocked |
| `vocab_channel_pruner.rs` | 2048 | Gated `lattice_operad` feature path needs `crate::lattice_operad` | lattice_operad moves first (this plan T4.4 — unblocks after) |
| `domino_lora.rs` | 561 | Imports `crate::types::matmul` | types::matmul relocation (separate plan) |
| `pruners/bomber/*` (19 files) | ~17K | Tightly coupled to 2 transformer-bound players + BomberPlayer trait | Extract bomber substrate + trait + non-TF players to new leaf; TF players stay root (separate plan) |
| `dllm.rs` | 4782 | Heavy transformer coupling (TransformerWeights everywhere) | transformer.rs move (out of scope) |
| `speculative/{mod, types, step, verifier, prefill, drafter_lora, d2f, d2f_verifier, dflash, diffusion_sampler, flashar_anchor, flashar_consensus, parallel_probe, set_diffusion, caddtree_budget, residency_audit, dd_tree}` | ~25K | Forward* composer family + ForwardContext + TransformerWeights everywhere | transformer.rs move (out of scope) |

## GOAT gate

Per Proposal 003 §"GOAT gate" (L299-315):
- **G1 correctness** — existing tests pass unchanged via re-exports.
- **G3 no-regression** — `cargo check --workspace --all-features` clean.
- (G2/G4/G5 apply only to promoted-default winners; Phase 12 is mostly
  relocations, not feature promotions.)

## References

- Proposal 003: `katgpt-rs/proposals/003_src_consolidation_master.md`
- Plan 382 (Phase 11 close-out): `katgpt-rs/.plans/382_phase11_domain_crates.md`
- Phase 12 spec: proposal L817-822
