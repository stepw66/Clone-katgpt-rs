# Issue 019 — Scrub cross-boundary coupling comments in `katgpt-rs/crates`

**Date:** 2026-07-01
**Type:** Refactor / moat hygiene
**Severity:** MEDIUM (HIGH for the Cargo.toml class — ships to crates.io)
**Status:** Open

## Problem

The commercial strategy (inlined in `.agents/skills/research/SKILL.md` §"Commercial
strategy — inline short version") defines the anti-pattern:

> A cross-reference comment that names a private module path IS a leak, even if
> the constant value itself is benign. A public benchmark constant whose comment
> names a private module path (`must match riir_gpu::...`) IS a leak.

A `katgpt-rs/crates` audit (2026-07-01) found **no hard IP leaks** (no product/chain/
shard/training code; no `*_runtime` GOAT modules; no private path deps; eggshell IP
actively migrated to `riir-neuron-db`) — **but** it found a systemic class of
coupling-comment leaks where public source and shipped Cargo.toml comments name
private repo/module/plan paths.

## Findings (exact counts from the audit)

### A. Private module paths in `.rs` doc comments (4 files — MEDIUM)
- `crates/katgpt-core/src/induced_cwm/hot_swap.rs` — `riir_engine::episode_buffer::LoRAWeightVersion`
- `crates/katgpt-core/src/group_invariance_probe.rs` — `riir_neuron_db::FreezeGateReport`, `riir-neuron-db::spectral_flatness`
- `crates/katgpt-core/src/karc.rs` — `riir_neuron_db::KarcShard`
- `crates/katgpt-types/src/lora.rs` — `riir_gpu::lora::export_lora`

### B. Private `.research`/`.plans`/`.docs`/`.issues` paths in `.rs` doc comments (12 files — MEDIUM)
Reveals private guide/plan numbering. Files include:
- `crates/katgpt-personality/src/lib.rs` — `riir-ai/.research/146_*`, `riir-ai/.plans/327_*`
- `crates/katgpt-sleep/src/lib.rs` — `riir-ai/.plans/341_*`
- `crates/katgpt-micro-belief/src/lib.rs` — `riir-ai/.research/127_*`
- `crates/katgpt-core/src/{arg,branching,ict,induced_cwm,lib,pruners/indicator_probe_bank,compression_drafter}.rs`
- `crates/katgpt-types/src/depth_invariance.rs`

### C. `riir-*` refs in Cargo.toml comments (6 files — HIGH, ships to crates.io)
Feature-flag comments reference private plan numbers and repo names. A competitor
downloading `katgpt-core` from crates.io sees the entire private plan/guide structure:
- `crates/katgpt-core/Cargo.toml` — many `# riir-ai Plan XXX`, `# riir-train follow-up`, `# riir-chain Plan 003` in feature comments
- `crates/katgpt-dec/Cargo.toml` — names `riir-neuron-db` (`dec_arena` migration note)
- `crates/katgpt-hla/Cargo.toml` — names `riir-engine` (`*_role_aware`/`role_transport`)
- `crates/katgpt-transformer/Cargo.toml` — "Ported from riir-engine", names riir-engine in feature comment

### D. `riir-engine` named in Cargo.toml `description` (3 crates — LOW, ships to crates.io)
- `katgpt-core`: "Shared types and SIMD kernels for katgpt-rs and riir-engine"
- `katgpt-speculative`: "shared by katgpt-rs and riir-engine"
- `katgpt-transformer`: "shared between katgpt-rs and riir-engine"

## Fix

Mechanical scrub. Replace private path/repo/plan references with generic phrasing:

| Leaky phrasing | Generic replacement |
|---|---|
| `riir_engine::episode_buffer::LoRAWeightVersion` | "the private runtime's ArcSwap-backed A/B LoRA weight swap" |
| `riir_neuron_db::FreezeGateReport` | "the private shard crate's freeze-gate report" |
| `riir_gpu::lora::export_lora` | "the private GPU LoRA exporter (byte-identical format)" |
| `riir-ai/.research/146_Entity_Cognition_Stack_Guide.md` | "the private runtime guide for this Super-GOAT" |
| `riir-ai/.plans/337_*` | "the private runtime plan" |
| `# riir-ai Plan 299 Phase 3 GOAT gate` | "# private-runtime GOAT gate (Phase 3)" |
| `# riir-train follow-up` | "# training-repo follow-up" |
| `# riir-chain Plan 003` | "# chain-repo LatCal plan" |
| `description = "...shared by katgpt-rs and riir-engine"` | `description = "Transformer substrate types (weights, KV caches, contexts) for katgpt-rs"` |

**Rule of thumb for the scrub:** the public file must not name a private repo,
private module path, or private plan/guide number. It MAY say "the private runtime",
"the training repo", "a downstream consumer", "the chain sync bridge" — generic
role words that don't reveal the private structure.

## Scope guard (do NOT over-scrub)

- The `riir_feedback` module + `coexplain_riir` feature in `katgpt-pruners` is a
  **naming collision** ("RIIR" = "Rewrite It In Rust"), NOT a private-repo
  reference. Leave the code; consider renaming to `translation_feedback` separately
  (cosmetic, not a leak).
- `katgpt-personality`'s `EntityCognitionComposition = <9, 32>` alias + "production
  case" comment is borderline — the alias can stay (const-generic convenience), but
  the "production Entity Cognition Stack case" comment should be genericized to
  "the 9-layer / 32-dim reference composition".
- Do NOT remove functional information — only strip the private-path coupling. The
  *existence* of a private runtime half is fine to mention; naming its exact path
  is the leak.

## Tasks

- [ ] **T1** Scrub class A (4 files): genericize private module-path doc refs in `.rs`
- [ ] **T2** Scrub class b (12 files): genericize private `.research`/`.plans` doc refs in `.rs`
- [ ] **T3** Scrub class c (6 Cargo.toml): genericize feature-flag comments — HIGH priority (ships to crates.io)
- [ ] **T4** Scrub class d (3 Cargo.toml descriptions): drop `riir-engine` from `description` fields
- [ ] **T5** Re-grep to confirm zero `riir-*` / `riir_*::` / `riir-ai/.` refs remain in `crates/`
- [ ] **T6** `cargo check --workspace` — confirm no broken intra-doc links after the scrub
- [-] **T7** (defer) Rename `riir_feedback` → `translation_feedback` in katgpt-pruners (cosmetic, separate PR)

## Validation

```bash
# After scrub, all three MUST return 0:
grep -rIl "riir_engine::\|riir_neuron_db::\|riir_gpu::\|riir_chain::\|riir_games::\|riir_train::\|riir_wasm::" crates/ --include="*.rs"
grep -rIl "riir-ai/\.research\|riir-ai/\.plans\|riir-ai/\.docs\|riir-ai/\.issues" crates/
grep -rIl "riir-" crates/ --include="Cargo.toml"   # the only acceptable hits: "riir-" inside a generic word — should be 0 after T3/T4
cargo check --workspace
```

## Related

- `.agents/skills/research/SKILL.md` §"Commercial strategy — inline short version" — the anti-pattern rule this issue enforces
- `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` §"Benchmark Domain Exception" — the originating cross-boundary coupling-constant rule
