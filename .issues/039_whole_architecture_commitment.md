# Issue 039 — Whole-Architecture Commitment (single tamper-evident root for an NPC's full cognitive architecture)

**Status:** ✅ **RESOLVED** (2026-07-04). Shipped as `CognitiveArchitectureRoot` in `crates/katgpt-core/src/engram/architecture_root.rs`, DEFAULT-ON feature `cognitive_architecture_root`. All GOAT gates PASS — see `.benchmarks/039_architecture_root_goat.md`. All six tasks T1–T6 ticked below.

**Filed:** 2026-07-04
**Priority:** P2 (high product payoff: chain-verifiable "NPC brain" — enables quorum-attested personality freeze/thaw, anti-cheat on cognitive state, and on-chain avatar import/export)
**Origin:** Evaluation of Gemini's "Continuous Neuro-Symbolic DAG" proposal against the codebase (2026-07-04). The mechanism atoms all ship (PTG, EngramTable, NeuronShard, latent_functor direction sets); the missing piece is the unifying commitment root.
**Blocks:** On-chain NPC personality portability, quorum-attested cognitive-state checkpoints. **Blocked by:** Nothing — every component already exists.
**Type:** New primitive (modelless, BLAKE3-only, ~80 LOC estimate).

---

## Problem

Every individual atom of an NPC's cognitive architecture already has its own tamper-evident commitment:

| Atom | Commitment | Where |
|---|---|---|
| Primitive Transition Graph (symbolic execution trace) | `serialize_postcard` + BLAKE3 (`closure/mod.rs:78`) | `katgpt-rs/crates/katgpt-core/src/closure/mod.rs` |
| EngramTable (latent anchor set) | `EngramTableId([u8; 32])` — Merkle root | `katgpt-rs/crates/katgpt-core/src/engram/commitment.rs` |
| NeuronShard (weight blob + HLA moments) | `merkle_root` field, BLAKE3 | `riir-neuron-db/src/shard.rs` |
| Functor direction set | Per-table Merkle root via `EngramTableId` | (reuses engram commitment) |
| SvopKgTriple four-slot anchors | Engram-row BLAKE3 | `riir-games/src/quest_grammar/svop_kg.rs` |

**But no single primitive commits "this NPC's full cognitive architecture"** — the (PTG, EngramTable, ShardSet, FunctorSignatureSet) tuple — as ONE tamper-evident root. The grep for `ArchitectureRoot | NpcBrainCommitment | WholeBrainCommitment | cognitive_architecture_root` returns ZERO hits across all 6 repos.

### Why this matters

1. **Anti-cheat on cognitive state.** Today an attacker can tamper with the PTG while leaving the shard set bit-identical (or vice versa) and no single hash detects the cross-component inconsistency. A root hash over the tuple closes this hole.
2. **Quorum-attested personality freeze/thaw.** The freeze/thaw runtime (Issue 348 T2, proven in Lean) currently operates on individual shards. A whole-architecture root lets a chain quorum attest "this exact brain state" atomically — enabling on-chain avatar portability (export NPC → import on another server → quorum-verify the brain matches).
3. **Deterministic replay across components.** Replay currently needs to verify each component's commitment separately and trust that they were consistent at the captured tick. A root hash makes "consistent at tick T" a single checkable claim.

## Scope

A new primitive (proposed name: `CognitiveArchitectureRoot` or `NpcBrainId`) that:

1. Is a `#[derive(Copy, Clone, PartialEq, Eq, Hash)]` newtype around `[u8; 32]` (BLAKE3 output), same shape as `EngramTableId`.
2. Is computed as `blake3(concat(ptg_root, engram_table_root, shard_set_merkle_root, functor_signature_root, tick, npc_id))` — a single level of hashing over the existing roots, NOT a re-walk of the underlying data.
3. Provides `from_parts(...) -> Self` and `verify(&self, ...) -> bool` (recompute and compare, mirroring `EngramTableId::from_table`).
4. Is zero-allocation (stack-only, no `Vec`, no `Box`).
5. Ships behind an opt-in feature flag (`cognitive_architecture_root`); GOAT-gated before any promotion to default.

### Non-Goals

- ❌ Replacing any existing per-component commitment. The root is an *additional* layer over existing roots; per-component hashes stay as the audit trail.
- ❌ Network sync wiring. That's a `riir-chain` consumer concern (a `SyncBlock` field). This issue ships only the hashing primitive.
- ❌ Encrypting the architecture. Commitment ≠ encryption; this is tamper-detection, not confidentiality (which is the turso/libSQL encryption layer's job).
- ❌ Lean proof. The hash is a composition of existing hashes; the Lean proofs for the underlying commitments already cover the math. A spec-match Rust test (mirroring `engram/commitment.rs` test style) is sufficient.

## Proposed direction (not committed)

### File location

`katgpt-rs/crates/katgpt-core/src/engram/architecture_root.rs` — sibling to `commitment.rs`, same module. Reuses `blake3` crate already in the workspace.

### Sketch

```rust
/// Tamper-evident root over an NPC's full cognitive architecture.
///
/// Computed as a single BLAKE3 hash over the existing per-component roots:
///   blake3(ptg_root || engram_table_root || shard_set_root || functor_sig_root || tick_le_bytes || npc_id_le_bytes)
///
/// Verification re-derives the same concatenation and compares. A mismatch
/// indicates tampering in ANY component (or in the binding tuple itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CognitiveArchitectureRoot(pub [u8; 32]);

impl CognitiveArchitectureRoot {
    /// Compute from component roots. Zero-alloc.
    #[inline]
    pub fn from_parts(
        ptg_root: &[u8; 32],
        engram_table_id: &EngramTableId,
        shard_set_root: &[u8; 32],
        functor_sig_root: &[u8; 32],
        tick: u32,
        npc_id: u64,
    ) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(ptg_root);
        h.update(&engram_table_id.0);
        h.update(shard_set_root);
        h.update(functor_sig_root);
        h.update(&tick.to_le_bytes());
        h.update(&npc_id.to_le_bytes());
        let mut out = [0u8; 32];
        h.finalize_xof().fill(&mut out);
        CognitiveArchitectureRoot(out)
    }

    /// Bit-identical re-derivation. `true` = architecture is intact.
    #[inline]
    pub fn verify(&self, /* same args as from_parts */) -> bool {
        *self == Self::from_parts(/* ... */)
    }
}
```

### GOAT gate

Per AGENTS.md feature-flag discipline:

- **G1 (correctness):** Spec-match test — compute a root, mutate any one input by a single bit, verify the root changes (avalanche). Mutate the root by one bit, verify `verify()` returns false.
- **G2 (perf):** `criterion` bench. Target: < 500 ns (one BLAKE3 pass over 6 * 32 bytes + 12 bytes header ≈ 200 bytes; BLAKE3 does ~1 GB/s so ~200 ns expected). 0-alloc mandatory.
- **G3 (no-regression):** Default-on path is unchanged (feature is opt-in); no existing test should regress.
- **G4 (alloc-free):** `assert_eq!(std::mem::size_of::<CognitiveArchitectureRoot>(), 32)` and verify no `Vec`/`Box` in the public API.
- **G5/G6 (modelless):** No training dependency. BLAKE3 is deterministic. ✅ trivially.

If G1–G6 pass → promote to default-on (modelless gain: anti-cheat + chain verifiability).

## Tasks

- [x] **T1** Confirm the four component-root accessors all exist and return `[u8; 32]`:
  - PTG: `closure::commitment(ptg: &PrimitiveTransitionGraph) -> [u8; 32]` — **already exists** at `closure/mod.rs:84-90`. No new helper needed.
  - EngramTable: `EngramTableId::from_table(...).0` — **exists** at `engram/commitment.rs:62-64`.
  - ShardSet: `NeuronShard::merkle_root() -> [u8; 32]` — **exists** in `riir-neuron-db/src/shard.rs` (chain-set).
  - FunctorSignatureSet: **grep returns ZERO hits** for `FunctorSignature|functor_signature|FunctorSig` in katgpt-rs. **Decision:** take a raw `[u8; 32]` parameter named `functor_sig_root` — the primitive doesn't care about the source type, only the bytes. Caller supplies whatever commitment they have (an `EngramTableId` if signatures live in an engram table, `[0u8; 32]` if absent). Avoids forcing a `FunctorSignatureSet` type to exist before the primitive can ship. See `.benchmarks/039_architecture_root_goat.md` §"Design note — FunctorSignatureSet".
- [x] **T2** Add `crates/katgpt-core/src/engram/architecture_root.rs` behind feature `cognitive_architecture_root`. Implement `CognitiveArchitectureRoot` per the sketch above.
- [x] **T3** Add spec-match tests: avalanche (single-bit input mutation changes ≥ N output bits), `verify()` round-trip, `size_of == 32`. **13 unit tests** covering every input field's bit-flip sensitivity + determinism + verify round-trip + free-function/method agreement + absent-component sentinel convention.
- [x] **T4** Add `criterion` bench in `benches/bench_039_architecture_root_goat.rs`. Record in `.benchmarks/039_architecture_root_goat.md`.
- [x] **T5** Run GOAT gate. **G1–G6 all PASS** → promoted `cognitive_architecture_root` to `default` in `crates/katgpt-core/Cargo.toml`. See `.benchmarks/039_architecture_root_goat.md` for the full gate table.
- [x] **T6** If promoted: add a doc section in `engram/mod.rs` explaining the layering (per-component roots → architecture root → chain SyncBlock field). **Done** — see the ASCII diagram in `engram/mod.rs` above the `mod architecture_root;` declaration.

## Cross-References

- **Component commitments:** `engram/commitment.rs` (`EngramTableId`), `closure/mod.rs:78` (PTG BLAKE3), `riir-neuron-db` (`NeuronShard::merkle_root`).
- **Freeze/thaw Lean proof:** `riir-ai/.proofs/RiirAiProof/Runtime/FreezeThaw.lean` (Issue 348 T2). The architecture root is the natural extension target — currently freeze/thaw operates per-shard; whole-architecture freeze is the next step (out of scope for this issue, but this primitive enables it).
- **Chain consumer (deferred):** `riir-chain/.research/007_Engram_LatCal_Commitment_Bridge.md` — once this primitive ships, the chain can add a `SyncBlock` field carrying `CognitiveArchitectureRoot` for quorum-attested personality checkpoints. File a `riir-chain/.issues/*` when this ships.
- **Origin evaluation:** Gemini "Continuous Neuro-Symbolic DAG" proposal review (2026-07-04) — every atom ships, the composition root does not.

## TL;DR

Every individual atom of an NPC's cognitive architecture (PTG, EngramTable, NeuronShard, functor signatures) already has its own BLAKE3 commitment, but no single primitive commits the *tuple* as ONE tamper-evident root. This issue tracks adding `CognitiveArchitectureRoot([u8; 32])` — a single BLAKE3 pass over the existing component roots plus (tick, npc_id). ~80 LOC, modelless, zero-alloc, behind feature flag `cognitive_architecture_root`. High product payoff: chain-verifiable NPC brains, quorum-attested personality freeze/thaw, anti-cheat on cognitive state. GOAT gate is trivial (BLAKE3 avalanche + perf bench). Promote to default-on if G1–G6 pass.
