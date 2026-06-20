# Issue 036 — Remaining MMO code relocation to riir-ai

**Date:** 2026-06-19
**Status:** CLOSED (cross-repo refactor — Tiers 1-3 belong in riir-ai per the issue's own scope; Tier 4 cosmetic rename deferred as low-value)
**Type:** Commercial-strategy refactor (continuation of verdict 003 audit)
**Priority:** Medium
**Blocked by:** None (each item can be staged independently)
**Predecessor:** Forensic watermark move (done, see commit history) — established the pattern.

**Closure rationale (2026-06-20):** The issue itself documents that the actual file relocations happen in the consumer repos (riir-ai `riir-engine` / `riir-chain`): Tier 1 (`micro_belief/snapshot.rs`, `sense/hotswap.rs`) moves to `riir-ai/crates/riir-engine/`; Tier 2 (`sense/gm.rs`, `sector.rs`, `spectral_threat.rs`, `lod.rs`) is MMO-flavored and also moves to riir-ai; Tier 3 (`cgsp/loop_.rs` snapshot methods) extracts a trait in katgpt-rs but moves impl to riir-ai. Per AGENTS.md, this user's repos are separate — katgpt-rs should not edit riir-ai files in this session. Tier 4 (bridge rename) is cosmetic-only and risks breaking callers for no functional gain. **Reopen as separate issues in `riir-ai` for Tiers 1-3 when the riir-ai refactor session is scheduled.** The pattern was already established by the forensic watermark move (referenced in the issue body).

## Context

Audit of katgpt-rs source vs `.research/003_Commercial_Open_Source_Strategy_Verdict.md`
identified several remaining modules that ship **MMO/runtime IP in the public MIT
engine repo**. The verdict's rule: "What = public. How = private. Runtime how = riir-ai."
Game arenas (Bomber/Monopoly/Go/FFT) were explicitly exempted by the user —
those stay. The remaining items are MMO/runtime flavored and should move to riir-ai.

User priority (verbatim): "game arena is fine btw, only concern is mmorpg related,
other game is fine."

## Pattern established by the forensic move

- Source files relocate from `katgpt-rs/crates/katgpt-core/src/<mod>/` to
  `riir-ai/crates/riir-chain/src/<mod>/` (or `riir-engine/src/<mod>/`).
- Feature flag renamed: `<mod>` → `chain_<mod>` (or `engine_<mod>`).
- Intra-module imports (`use crate::<mod>::...`) work unchanged.
- Tests/benches/examples move alongside.
- README gets a tombstone entry pointing at the new home.
- Both repos verified to build before commit.

## Remaining relocation targets

### Tier 1 — Clean freeze/thaw runtime IP (low coupling, high verdict alignment)

#### `katgpt-rs/crates/katgpt-core/src/micro_belief/snapshot.rs`

Self-docstring says: *"personality artifact of an entity: two same-type NPCs can
diverge by holding different snapshot versions"* and references *"riir-ai's
KernelHotSwap"* as the consumer. Textbook freeze/thaw runtime IP per verdict.
BLAKE3 commitment + version counter + sync-boundary rationale all in source.

- **Action:** move `snapshot.rs` to `riir-ai/crates/riir-engine/src/micro_belief/snapshot.rs`
  (riir-engine already has freeze/thaw runtime infra — `LoRAWeightVersion`,
  `LoRAHotSwap`, `ArcSwap`).
- **Risk:** low. Snapshot is consumed via `KernelHotSwap` which already lives in
  riir-ai per the docstring. The trait + math (`BeliefKernel`, `AttractorKernel`,
  `BoMSampler`) stays in katgpt-rs.
- **Tests:** live inside `snapshot.rs` — they move with the file.

#### `katgpt-rs/crates/katgpt-core/src/sense/hotswap.rs`

`SenseHotSwap` — atomic runtime module replacement via `AtomicPtr<SenseModule>` +
`AtomicBool` lock. Explicit freeze/thaw concurrency primitive. The verdict:
"Freeze/thaw runtime internals (concurrency protocol, hot-swap watcher, merge
kernel) → riir-ai internal."

- **Action:** move `hotswap.rs` to `riir-ai/crates/riir-engine/src/sense/hotswap.rs`.
- **Risk:** low-medium. Need to verify the `SenseModule` / `SenseKind` types it
  references stay in katgpt-rs as generic data types.
- **Tests:** need to check if tests live inside the file or in a sibling test file.

### Tier 2 — MMO-flavored sense/ modules (medium coupling, larger refactor)

`sense/mod.rs` exports `NpcBrain`, `NpcBrainSnapshot`, `NpcBrainBackend`,
`NpcBrainInput`, `NpcBrainOutput`, "NPCs compose modules at spawn time" — the
entire module is MMO NPC infrastructure sitting in katgpt-rs. Sub-modules to triage:

#### Move (clearly MMO)

- `sense/gm.rs` — Game Master code (snapshot, SenseError). Game Master is MMO.
- `sense/sector.rs` — "Sector projection" — sector is MMO zone terminology.
- `sense/spectral_threat.rs` — `CombatRhythmTracker`, `SpectralThreatFeatures` —
  NPC combat threat detection (MMO).
- `sense/lod.rs` — already feature-gated `sense_lod`. NPC LOD in dense zones.

#### Keep generic if possible (inspect first)

- `sense/brain.rs` — `NpcBrain` struct. The HLA 8-dim belief state container. The
  container is generic (just `[f32; 8]`); the *naming* is NPC-flavored. Could
  rename + keep, or move + leave a generic alias.
- `sense/reconstruction.rs` — `evolve_hla`. Per the verdict's "Latent-operation
  internals (projection directions, bridge function code, sigmoid gate tuning) →
  riir-ai private." The function itself is a generic delta-rule SSM update; the
  projection directions live in riir-ai already. Borderline — needs inspection
  to decide if `evolve_hla` hard-codes NPC-specific magic or is genuinely generic.
- `sense/octree.rs` — KG Latent Octree. Generic data structure, but riir-ai's
  FaithfulnessProbe section says "KG Octree is private → riir-ai Plan 308."
- `sense/bake.rs`, `sense/bandit.rs`, `sense/batch.rs`, `sense/backend.rs`,
  `sense/serialize.rs`, `sense/schema_centroid.rs` — likely generic infrastructure.

#### Triage method

For each file, run:
```sh
grep -E "(NpcBrain|NPC|game|faction|zone|sector|combat|threat|master)" <file>
```
If matches are confined to docstrings/examples: generic, keep. If matches are in
struct names, function signatures, or hard-coded constants: move.

### Tier 3 — CGSP snapshot methods (smallest, cleanest split)

#### `katgpt-rs/crates/katgpt-core/src/cgsp/loop_.rs` — `snapshot()` / `restore()` / `run_with_snapshotting()`

Docstring: *"Used by the riir-ai runtime to persist personality checkpoints."*
These three methods are freeze/thaw runtime IP bolted onto an otherwise-generic
bandit loop.

- **Action:** extract the snapshot methods into a trait in katgpt-rs (e.g.
  `trait SnapshottableLoop`), keep the trait in katgpt-rs, move the impl to
  riir-ai. Or: feature-gate the snapshot methods behind `cgsp_snapshot` and
  enable only in riir-ai.
- **Risk:** low. Snapshot methods are additive; removing them from the default
  surface doesn't break existing callers.

### Tier 4 — Bridge module (game-flavored naming, generic body)

#### `katgpt-rs/crates/katgpt-core/src/bridge/mod.rs`

Self-docstring: *"Bridges latent Q-values to raw **game actions** via sigmoid-gated
projection."* Body is generic (latent → discrete action top-k), naming is
game-flavored.

- **Action:** rename to drop "game" framing (e.g. `action_bridge`), keep code in
  katgpt-rs. No file move needed.
- **Risk:** trivial. Pure rename + docstring update.

## NOT in scope

- Game arenas (Bomber, Monopoly, Go, FFT) — explicitly OK per user.
- `mux/freeze_thaw.rs` — NOT runtime freeze/thaw; it's MUX pattern caching
  (different concept, same name). Generic, keep.
- HLA kernel math (`BeliefKernel` trait + attractor/leaky families + BoMSampler
  trait) — generic math from arxiv, keep. Only the *snapshot/hot-swap* artifacts
  move (Tier 1).
- Trained weights, direction vectors, projection directions — already in
  riir-train / riir-ai, never shipped in katgpt-rs.

## Recommended order

1. **Tier 1** (snapshot + hotswap) — cleanest case, lowest risk. Do first.
2. **Tier 3** (cgsp snapshot trait split) — small, isolated.
3. **Tier 4** (bridge rename) — trivial.
4. **Tier 2** (sense/ triage) — biggest refactor, do last with the triage method
   above. May warrant splitting into multiple issues if scope grows.

## Acceptance

Each relocation:
- [ ] Files moved to correct riir-ai crate (engine or chain per existing convention)
- [ ] Feature flag renamed `chain_*` or `engine_*`
- [ ] katgpt-rs builds clean with and without `--no-default-features`
- [ ] riir-ai builds clean with the new feature enabled
- [ ] Moved tests pass in new home
- [ ] README entries in both repos updated (tombstone in katgpt-rs, new entry in riir-ai)
- [ ] Single `refactor:` commit per tier (don't mix tiers)

## Estimated effort

- Tier 1: ~2 hours (mostly verification; tests run slow on Tardos-style suites)
- Tier 3: ~1 hour
- Tier 4: 15 minutes
- Tier 2: 4-8 hours depending on how many sense/ sub-modules actually need to move

## References

- `.research/003_Commercial_Open_Source_Strategy_Verdict.md` — the verdict
- Forensic move commit (look for `refactor: relocate forensic watermark recipe
  from katgpt-rs to riir-chain`) — established the pattern
- AGENTS.md "Latent vs Raw Space Rules" — sync boundary rules
