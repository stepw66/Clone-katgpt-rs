# Cross-Repo Consolidation Audit — riir-ai / riir-chain / riir-neuron-db

**Date:** 2026-07-06
**Trigger:** Proposal 003 endgame (katgpt-rs) — apply the same endgame-audit pattern to the 3 sibling repos.
**Method:** 3 parallel read-only subagent audits (one per repo), each classifying every significant module into 5 buckets (pure shim / coupled glue / root-resident / tooling / extractable). Synthesized here.

## TL;DR

| Repo | Verdict | Action taken |
|---|---|---|
| **riir-ai** (628K LOC, 15 crates) | Architecturally at endgame; file-hygiene + cohesion debt moderate | Documented; deferred (see riir-ai issues) |
| **riir-chain** (93K LOC, 2 members) | At natural endgame; ONE clear DRY violation | **DONE** — `cold_store` DRY extraction (issue 001, commit `93a91de`) |
| **riir-neuron-db** (27K LOC, leaf) | At natural endgame; excellent cohesion | **DONE** — `consolidation.rs` test split (commit `4bea4f0`) |

The katgpt-rs analogy (18K LOC of pure substrate trapped at root) **does not hold** for these repos. They are already better-architected than katgpt-rs was pre-Proposal-003: root leaves are already isolated, integration code already lives in separate crates, and the neuron-db spin-off already captured the only clean cross-crate extraction. The remaining debt is **intra-crate** (file splits, DRY dedup), not cross-crate extraction.

---

## riir-ai — 628K LOC, 15 crates (+3 auxiliary)

### Executive summary

riir-ai carries **substantial file-level consolidation debt but minimal cross-crate extraction debt**. Three "monster" crates — `riir-engine` (196K), `riir-games` (169K), `riir-gpu` (125K) — hold **78% of the workspace LOC**, and **23 `.rs` files break the 2048-line rule**.

However, unlike katgpt-rs pre-Plan-404, the oversized code is **not pure substrate trapped at root** — it is mostly cohesive GPU kernel code, cohesive per-domain runtime modules, and legitimate game-integration glue. **~85% of the oversized code is permanently resident where it is.**

### Findings

**Cross-crate extraction debt: LOW.** Only ONE genuine extraction candidate exists:
- `riir-games/src/meta_router.rs` (2,449 LOC) — Vortex meta-routing bandit, depends only on `fastrand`. Pristine deps. Could move to `riir-router` (fits its "inference routing" mandate) or its own `riir-meta-router` crate.

**File-hygiene debt: MODERATE.** 23 files break the 2048-line rule. Notable offenders:
- `riir-gpu/src/forward.rs` — **7,894 LOC** (workspace's largest file; 20+ WGSL param structs)
- `riir-engine/src/latent_functor/arithmetic.rs` — 3,667 LOC (6 functor primitives)
- `riir-games/src/civ/skill.rs` — 3,556 LOC (skill YAML schema + tiers)

Most are mechanical internal-splits (carve WGSL param structs into `forward_params.rs`, split 6 functors into 6 files), not cross-crate extraction.

**Cohesion debt: MODERATE, concentrated in `riir-games`.** The `civ/` submodule alone is 83 files / 43K LOC — bigger than 16 of the 18 crates. 27 top-level modules span civ, quest, plasma, worms, ruliology, etc. The real opportunity is **splitting `riir-games` along domain lines** (e.g. `riir-games-civ`, `riir-games-quest`).

**DRY debt: LOW.** One proven intentional fork: `FrameSampler` duplicated between `riir-engine/src/frame/sampler.rs` and `riir-games/src/combat/sampler.rs` (games version is a perf-improved fork with `sim_tick_into` buffer reuse). Other apparent duplicates (`SpatialBelief`, `KgTriple`) are coincidental name collisions with different fields/semantics — NOT true duplication.

### Blockers

1. **GPU substrate coupling.** Every oversized `riir-gpu` file depends on `crate::buffer` + `crate::kernels` + `crate::lora` + `crate::speft`. Four intertwined substrates — the riir-ai equivalent of katgpt-rs's `ForwardContext` linchpin.
2. **`riir-engine` is already the root leaf.** Zero outbound riir deps. Its 65 top-level modules are each cohesive subsystems. No "trapped substrate at root" — the substrate IS the root.
3. **Mod-cohesion, not file-bloat.** `latent_functor/arithmetic.rs` (3,667 LOC) isn't one god-file — it's six functor primitives alongside 11 siblings in a 12K-LOC cohesive module. The fix is per-primitive file-splitting, not crate extraction.

### Recommended next steps (deferred)

Tracked in riir-ai issues (to be created):
- Extract `meta_router.rs` to `riir-router` or own crate (~2.4K LOC)
- Split `riir-games` along domain lines (civ/quest/core) — biggest structural win
- Internal file-splits for top-5 rule-violators in `riir-gpu`/`riir-engine` (mechanical)
- Upstream `FrameSampler` perf fork from games to engine

---

## riir-chain — 93K LOC, 2-member workspace

### Executive summary

**At natural endgame for crate-level consolidation.** The neuron-db spin-off (Plan 001) is exemplary: `src/neuron_db/mod.rs` is an 85-LOC shim, `catchup/merkle.rs` correctly delegates generics to the leaf crate and retains chain-specifics. Zero files over the 2048-LOC limit.

**One clear DRY violation found and FIXED:** ~895 LOC of acknowledged `cold_store` boilerplate duplicated across 4 `*_commit.rs` files.

### Findings

**Spin-off cleanliness: EXEMPLARY.** Verified:
- `src/neuron_db/mod.rs` — 85-LOC shim, single `pub use riir_neuron_db::*;` + the `LatCalWalletExt` trait (legitimately stays — references chain-side `LatCalMatrix`)
- `src/catchup/merkle.rs` — re-exports generic `MerkleTree`/`MerkleProof` from leaf, retains `DataTier`/`build_*_root` (chain block-commitment concepts)
- Zero duplication between riir-chain and riir-neuron-db

**DRY violation (FIXED):** Four `*_commit.rs` files each carried ~210-230 LOC of near-identical `mod cold_store` boilerplate. Code comments said: *"Verbatim from ShardStore/KarcBatchStore/... — kept local rather than shared"*. Extracted into generic `ColdBatchStore<B: ColdBatch>` (see `issues/001_cold_batch_store_dry.md`).

**Largest files (all under 2048 limit):**
- `src/consensus/guard_pruner.rs` — 1,837 LOC (watch — approaching limit)
- `crates/riir-chaind/src/mcp.rs` — 1,633 LOC (daemon, correctly placed)
- `src/karc_commit.rs` — 1,463 LOC (was larger pre-extraction)

### Action taken

- **`issues/001_cold_batch_store_dry.md`** — DONE. Commit `93a91de`. Generic `ColdBatchStore<B>` + `ColdBatch`/`ColdReceipt` traits. Net -166 LOC. 310/310 tests pass.

---

## riir-neuron-db — 27K LOC, single leaf crate

### Executive summary

**At natural endgame. Excellent cohesion and feature-flag discipline.** The `merkle_root` and `can_freeze` lessons are fully respected. Lean proof parity is complete (28 theorems, 17 paired Rust spec-match tests).

### Findings

**NeuronShard constructor audit: CLEAN.** All 4 constructors (`new`, `new_unchecked`, `new_spectral`, `from_bytes`) initialize all 7 fields consistently, including the feature-gated `compressed_root`. The `merkle_root` lesson is fully internalized.

**Lean proof parity: COMPLETE.**
- `.proofs/NeuronDbProof/Shard/Layout.lean` — 16 theorems (axiom-free)
- `.proofs/NeuronDbProof/Consolidation/FreezeGate.lean` — 8 theorems (standard foundation)
- `.proofs/NeuronDbProof/Merkle/Soundness.lean` — 4 theorems (axiom-free under injectivity hypothesis)
- Paired Rust spec-match tests: 17 tests across 3 files

**One file over 2048 limit (FIXED):** `consolidation.rs` was 2,666 LOC but ~72% was inline test blocks (~1,911 LOC tests, ~755 LOC library code). Split into `consolidation/mod.rs` (786 LOC) + 6 test files. Commit `4bea4f0`. Test parity: 233/233 (default), 478/478 (all-features).

**Minor doc drift:** `.proofs/README.md` says "15 Lean theorems" but `Layout.lean` alone has 16; actual total is ~28. Cosmetic, doesn't affect correctness.

### Action taken

- **`consolidation.rs` test split** — DONE. Commit `4bea4f0`. File now 786 LOC (under limit). Perfect test parity.

---

## Comparison to katgpt-rs Proposal 003

| Dimension | katgpt-rs (pre-Proposal-003) | Sibling repos (now) |
|---|---|---|
| Cross-crate extraction debt | HIGH (~18K LOC trapped at root) | LOW (substrate already in leaves) |
| Root leaf isolation | Poor (root hosted both substrate + integration) | Good (riir-engine is clean root leaf) |
| File-hygiene (2048 rule) | Moderate | Moderate (riir-ai), Good (riir-chain, riir-neuron-db) |
| DRY violations | Low | Low (1 fixed in riir-chain, 1 intentional fork in riir-ai) |
| Spin-off cleanliness | N/A (no spin-offs) | Exemplary (neuron-db split is textbook) |

**Conclusion:** The sibling repos do NOT need a Proposal-003-scale consolidation. They are architecturally more mature than katgpt-rs was. The remaining debt is surgical (file splits, one DRY extraction, one domain split) and has been either addressed (riir-chain, riir-neuron-db) or documented for future work (riir-ai).

---

## Commits landed this session

| Repo | Commit | Description |
|---|---|---|
| riir-neuron-db | `4bea4f0` | Split `consolidation.rs` test blocks into sub-directory (2666→786 LOC lib) |
| riir-chain | `93a91de` | DRY-extract `cold_store` into generic `ColdBatchStore<B>` (-166 LOC net) |
