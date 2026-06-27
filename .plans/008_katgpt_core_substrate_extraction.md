# Plan 008: katgpt-core Substrate Extraction (Phase 1+2 of Issue 007)

> **Origin:** [Issue 007](../issues/007_katgpt_rs_cargo_publish_substrate_reorg.md)
> **Status:** Active — Phase 1 step 1 ✅ done (pre-existing); steps 2-7 queued
> **Branch:** `develop`
> **Created:** 2026-06-27
> **Cross-repo:** katgpt-rs (primary moves), riir-ai/riir-engine (Phase 2 dedup consumer)

---

## TL;DR

Issue 007 was written against a snapshot that has since drifted. This plan
captures the **corrected** scope after a full audit of the current tree.

**Three findings that change the issue's scope:**

1. **Phase 5 (publish `katgpt-rs`) is dead.** Both `Cargo.toml:9`
   (`publish = false  # dev/examples aggregator — never published`) and
   `release-plz.toml:9-12` (`release = false, publish = false`) lock the root
   crate private permanently, with the rationale: "Only katgpt-core ships to
   crates.io." This decision was made AFTER Issue 007 was filed and overrides
   its Phase 5.

2. **Phase 1 step 1 (move `types`) is already done.** `katgpt-core/src/types/`
   has 14 files (`config.rs`, `enums.rs`, `rng.rs`, `math.rs`, …). Root
   `src/types.rs` is already a thin `pub use katgpt_core::types::*;` re-export
   shim plus a handful of root-only items (`QuantizedKVCache`,
   `AsymmetricKVConfig`, `top_p_coreset`, `OutlierGuardConfig`).

3. **Phase 2B's premise is inverted.** Issue 007 says "cgsp in core, cce in
   root — move cgsp UP". Reality: `cgsp` is already in `katgpt-core/src/cgsp/`
   AND re-exported from root `src/cgsp.rs` (verbatim `pub use katgpt_core::cgsp`).
   `cce` is in root `src/cce/`. The issue's "tier inconsistency" is real but
   the proposed direction ("move cgsp up") contradicts the cargo-publish
   decision in finding #1 — the only publishable crate is core, so substrate
   goes DOWN, not up. `cce` is correctly in root (cognitive layer, not
   substrate); `cgsp` is correctly in core. **No tier move needed.**

**What IS still real and valuable** (the heart of Issue 007): the
**cross-repo DRY violation**. `riir-engine/src/` has its own divergent
`crate::hla`, `crate::transformer`, `crate::types`, `crate::tokenizer`,
`crate::dd_tree`, `crate::spec_types`, `crate::mcts`, `crate::sampling`,
`crate::delta_mem`, `crate::simd` — all confirmed via grep using `crate::`
prefix (not `katgpt_core::`). These are PUBLIC inference mechanics per the
refined strategy doc, stranded in a private fork.

---

## Verdicts on Issue 007 Open Questions

| Q | Issue wording | Verdict |
|---|---|---|
| 1 | Phase 1 scope: full chain or subset? | **Full chain.** Anything left behind stays duplicated. Order enforced by deps: transformer → {weights, hla, dd_tree, spec_types} → {mcts, sampling, delta_mem}. |
| 2 | tokenizer to core or root? | **Defer; audit-first.** Risk 4 stands: SentencePiece-sys is a C++ build dep that disqualifies from leaf-clean core. Move only after transformer/hla land; verify no SentencePiece dep before moving. |
| 3 | mcts/dd_tree generic-vs-game split aggressiveness | **Generic core trait + concrete game impls in riir-engine.** Mirror the existing `traits.rs` pattern (already half-done for spec_types). Move `TreeNode`/`DDTreeBranchCache`/`SpeculativeContext` types to join their traits in core; leave game-coupled impls in riir-engine. |
| 4 | Go order: 1+2 first or push to publish? | **Phase 1+2 only. Phase 3-5 deferred indefinitely.** Phase 5 rescinded (finding #1). Phase 3 (root subdir reorg) is cosmetic — not worth the churn while 100+ features still flatten at root. Phase 4 (`plotters` optional) is independent and can be done as a standalone fix if `cargo check --no-default-features` ever fails; not blocking. |

---

## Task list

### Phase 1 — Substrate extraction to `katgpt-core`

- [x] **Step 1 — `types` → core.** DONE pre-this-plan. `katgpt-core/src/types/` 14 files; root is re-export shim.
- [ ] **Step 2 — `transformer` + `weights` → core.** ⚠️ `transformer.rs` is **8398 lines** (4× the 2048-line ceiling). MUST be split during move:
  - [ ] 2a. Map `transformer.rs` internal sections (gemma2 forward, llama forward, kv cache, forward context, etc.)
  - [ ] 2b. Split into `katgpt-core/src/transformer/{mod,context,kv_cache,gemma2,llama,prefill,...}.rs` — each <2048 lines
  - [ ] 2c. Move `weights.rs` (472 lines) to `katgpt-core/src/weights.rs` — depends on `TransformerWeights` from 2b
  - [ ] 2d. Root `src/transformer.rs` + `src/weights.rs` → thin `pub use katgpt_core::{transformer, weights};` re-exports
  - [ ] 2e. Audit feature gates — transformer has many `#[cfg(feature = "...")]` blocks; verify each still resolves
  - [ ] 2f. `cargo check` + `cargo test -p katgpt-core --lib` + `cargo test --lib` all green
  - [ ] 2g. Commit: `feat(core): Plan 008 step 2 — move transformer+weights substrate to katgpt-core`
- [ ] **Step 3 — `tokenizer` → core.** DEFERRED per Q2 verdict. Audit SentencePiece-sys dep first; if present, leave in root.
- [ ] **Step 4 — `hla` → core.** 2248 lines total (`forward.rs` 569 + `kernel.rs` 1019 + `types.rs` 606 + `mod.rs` 54). Depends on step 2.
  - [ ] 4a. Move `hla/{mod,types,kernel,forward}.rs` → `katgpt-core/src/hla/`
  - [ ] 4b. Port riir-engine's `*_role_aware` variants behind a new core feature `hla_role_aware` (consolidation, not blind copy — per Issue 007 §"Cross-repo consumer cleanup")
  - [ ] 4c. Root `src/hla/` → thin re-export
  - [ ] 4d. GOAT gate: bit-identical forward output vs pre-move (sigmoid, never softmax)
  - [ ] 4e. Commit: `feat(core): Plan 008 step 4 — move HLA substrate to katgpt-core + port role-aware variants`
- [ ] **Step 5 — `dd_tree` + `spec_types` → core.** Traits already in `core/traits.rs`; move dependent types (`TreeNode`, `DDTreeBranchCache`, `SpeculativeContext`, `DraftResult`, `NoPruner`, `ScreeningPruner` dep types) to join them.
- [ ] **Step 6 — `mcts`, `sampling`, `delta_mem` → core.** Leaf inference mechanics. `mcts` parameterize over a core `Game` trait (Q3 verdict); leave game-specific impls in riir-engine.
- [ ] **Step 7 — riir-engine `simd/wasm32.rs` → consume `katgpt_core::simd`.** Already shipped in core under `wasm32_simd128_*` kernels. Diff for riir-engine-only improvements, port if any, then delete reimplementation.

### Phase 2 — riir-engine dedup (the DRY payoff)

After each Phase 1 step lands, riir-engine deletes its copy and imports from
`katgpt_core` the same way `analytic_lattice` / `arg_runtime` already do.

- [ ] 2.1 riir-engine `src/hla/` → `use katgpt_core::hla::{...}` (after step 4)
- [ ] 2.2 riir-engine `src/transformer/` → `use katgpt_core::transformer::{...}` (after step 2)
- [ ] 2.3 riir-engine `src/types.rs` → `use katgpt_core::types::{...}` (already partially done via `spec_types.rs:11`)
- [ ] 2.4 riir-engine `src/tokenizer.rs` → consume core (after step 3, if it moves)
- [ ] 2.5 riir-engine `src/dd_tree.rs` + `spec_types.rs` → consume core (after step 5)
- [ ] 2.6 riir-engine `src/mcts.rs`, `sampling.rs`, `delta_mem/` → consume core (after step 6)
- [ ] 2.7 riir-engine `src/simd/` → consume core (after step 7)
- [ ] 2.8 Bit-identical verification: `forward_hla`/`forward_gemma2`/`dd_tree` tests pass unchanged in both repos

### Phase 3-5 — DEFERRED

- [ ] **Phase 3** (root subdir reorg into `primitives/`/`inference/`/`games/`/`backends/`) — cosmetic, not worth churn while 100+ features flatten at root. Revisit if/when root module count becomes unnavigable.
- [ ] **Phase 4** (`plotters` optional, `cargo check --no-default-features` clean on root) — independent quick win; do as standalone if it becomes blocking.
- [ ] ~~**Phase 5** (publish `katgpt-rs` to crates.io)~~ — **RESCINDED.** Conflicts with the post-issue decision in `Cargo.toml:9` + `release-plz.toml:9-12` to keep root private permanently. Only `katgpt-core` ships.

---

## Risk register (carried from Issue 007, updated)

1. **`transformer.rs` is 8398 lines.** Moving without splitting violates the 2048-line ceiling. **Mitigation:** step 2 mandates split-then-move. This makes step 2 the single biggest commit of the plan — allocate a full focused session, not a tag-end task.
2. **riir-engine HLA diverged** (`*_role_aware` variants + `role_transport` wiring). **Mitigation:** port kernel variants into core behind `hla_role_aware` feature; keep `role_transport.rs` as private composition in riir-engine (Category C).
3. **Version churn.** Core moves to next minor (new modules); root version is meaningless (`publish = false`). **Mitigation:** only core version matters; release-plz handles it.
4. **`tokenizer` may pull SentencePiece C++ build dep.** **Mitigation:** Q2 verdict — audit-first, defer to step 3.
5. **`dd_tree`/`spec_types` reconciliation.** riir-engine's copy may have game-coupled additions. **Mitigation:** Q3 verdict — generic types to core, game impls stay in riir-engine.
6. **`mcts.rs` imports `crate::game_state::GameState`** (Category C game IP). **Mitigation:** Q3 verdict — parameterize over core `Game` trait in core; keep game-specific impl in riir-engine.

---

## Acceptance

Mirrors Issue 007 §Acceptance, updated:

- [ ] Phase 1 step 2: `transformer`+`weights` live in `katgpt-core`, split into <2048-line files, re-exported from root. `cargo test -p katgpt-core --lib` + `cargo test --lib` green on arm64. (x86_64 already cleared per Issue 006.)
- [ ] Phase 1 step 4: `hla` lives in `katgpt-core` with role-aware variants behind feature flag. Bit-identical forward output vs pre-move.
- [ ] Phase 1 steps 5-7: each substrate module lives in core, re-exported from root.
- [ ] Phase 2: riir-engine has zero Category A duplicates; all consume `katgpt_core::`. Bit-identical tests in both repos.
- [ ] Each phase commit includes GOAT/bench evidence per AGENTS.md "dont defer benchmark task".

---

## Out of scope (explicit)

- Publishing `katgpt-rs` root crate (Phase 5 — rescinded).
- Moving cognitive/reasoning primitives (`cce`, `clr`, `compaction`, `claim_rubric`, etc.) out of root. They are correctly tiered: root = cognitive basics + composition, core = pure substrate, riir-* = GOAT tuning. Per Issue 007 §"Cognitive/reasoning is a NEW MOAT".
- Moving Category C game IP (`arena/`, `bom_arena/`, `cce_runtime/`, etc.) — stays private per the 003 strategy.
