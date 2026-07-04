# Issue 039 — Cognitive Architecture Root GOAT Gate

**Date:** 2026-07-04
**Primitive:** `CognitiveArchitectureRoot([u8; 32])` — whole-architecture BLAKE3 commitment.
**Location:** `crates/katgpt-core/src/engram/architecture_root.rs`
**Bench:** `crates/katgpt-core/benches/bench_039_architecture_root_goat.rs`
**Feature:** `cognitive_architecture_root = ["engram"]` → **promoted to DEFAULT-ON**.

## GOAT gate summary

| Gate | Target | Measured | Result |
|---|---|---|---|
| G1 correctness | determinism + verify round-trip + single-bit-flip breaks verify on every input | 13/13 unit tests + bench re-run all pass | ✅ PASS |
| G1 avalanche | ≥ 96 differing output bits on 1-bit input flip (BLAKE3 expected ~128) | min 120/256, avg 126/256 across 6 single-bit mutations | ✅ PASS |
| G2 perf | `from_parts` and `verify` each < 500 ns (BLAKE3 over ~204 bytes, expected ~200 ns) | `from_parts` **208 ns**, `verify` **208 ns** (2.4× headroom) | ✅ PASS |
| G2-alloc | 0 hot-path allocations across 1000 calls | `from_parts × 1000` = 0, `verify × 1000` = 0, construct × 1 = 0 | ✅ PASS |
| G3 no-regression | default / `--all-features` / `--no-default-features` all clean | all clean; 1029 lib tests pass with default features | ✅ PASS |
| G4 alloc-free | `size_of::<CognitiveArchitectureRoot>() == 32` | 32 bytes (no padding, no indirection) | ✅ PASS |
| G5/G6 modelless | no training dependency, deterministic | pure BLAKE3 composition, no learned params | ✅ PASS (trivially) |

**All gates PASS → promoted `cognitive_architecture_root` to `default`.** The gain is pure modelless (anti-cheat on cognitive state + quorum-attested personality freeze/thaw + on-chain NPC avatar portability) — no training dependency, no softmax, no behavior change to existing primitives.

## Composition, not re-walk

The primitive hashes over the **existing per-component roots** (each 32 bytes), not the underlying data:

```
blake3(ptg_root || engram_table_id || shard_set_root || functor_sig_root || tick_le || npc_id_le)
```

Total input ≈ 6 × 32 + 4 + 8 = **204 bytes**. BLAKE3 does ~1 GB/s on modern CPUs → ~200 ns expected, **208 ns measured** (matches prediction). Re-walking the underlying data would be O(N) in shard/node count and miss the point.

## T1 audit — component root accessors

| Atom | Accessor | Location |
|---|---|---|
| Primitive Transition Graph | `closure::commitment(&PrimitiveTransitionGraph) -> [u8; 32]` | `crates/katgpt-core/src/closure/mod.rs:84-90` |
| EngramTable | `EngramTableId::from_table(&dyn EngramTable).0` | `crates/katgpt-core/src/engram/commitment.rs:62-64` |
| NeuronShard set | `NeuronShard::merkle_root() -> [u8; 32]` (chain-set) | `riir-neuron-db/src/shard.rs` |
| FunctorSignatureSet | **No separate type exists** — caller supplies whatever commitment they have (an `EngramTableId` if signatures live in an engram table, `[0u8; 32]` if absent) | n/a — see design note below |

### Design note — `FunctorSignatureSet`

A grep for `FunctorSignature|functor_signature|FunctorSig` returns ZERO hits across `katgpt-rs/**/*.rs`. Per Issue 039 T1.4: "likely reuses `EngramTableId` if signatures live in an engram table."

**Decision:** take a raw `[u8; 32]` parameter named `functor_sig_root`. The primitive doesn't care about the source type — only the bytes. This is the cleaner abstraction:

- If the signatures live in an EngramTable, the caller passes `&engram_table_id.0`.
- If they live elsewhere, the caller passes whatever commitment they have.
- If there is no signature set, the caller passes `&[0u8; 32]` (the well-defined "absent" sentinel, matching the `padding leaf = zero hash` Merkle convention from `build_merkle_root`).

This avoids forcing a `FunctorSignatureSet` type to exist before the primitive can ship.

## Avalanche evidence

BLAKE3 avalanche target is ~50% of output bits flip on a single-bit input change (128/256 for a 32-byte output). Measured across 6 single-bit input mutations (one per input field: `ptg_root`, `engram_table_id`, `shard_set_root`, `functor_sig_root`, `tick`, `npc_id`):

- **min Hamming distance:** 120/256 bits
- **avg Hamming distance:** 126/256 bits (BLAKE3 expected ~128)

Floor: ≥ 96 bits (37.5% — a regression guard, not a quality gate). BLAKE3 reliably gives ~128 ± 12; the floor catches catastrophic collisions. The per-mutation breakdown is not reported by the bench (only min/avg) — see `g1_avalanche()` in the bench source for the mutation list.

## Wire order (canonical, do not change)

The `update()` calls happen in this exact order. Changing the order breaks cross-version compatibility:

```rust
h.update(ptg_root);              // 32 bytes
h.update(&engram_table_id.0);    // 32 bytes
h.update(shard_set_root);        // 32 bytes
h.update(functor_sig_root);      // 32 bytes
h.update(&tick.to_le_bytes());   //  4 bytes
h.update(&npc_id.to_le_bytes()); //  8 bytes
                                 // ─────────
                                 // 204 bytes total
```

`tick` and `npc_id` are encoded little-endian for host-endianness invariance.

## Public API

```rust
pub struct CognitiveArchitectureRoot(pub [u8; 32]);

impl CognitiveArchitectureRoot {
    pub fn from_parts(
        ptg_root: &[u8; 32],
        engram_table_id: &EngramTableId,
        shard_set_root: &[u8; 32],
        functor_sig_root: &[u8; 32],
        tick: u32,
        npc_id: u64,
    ) -> Self;

    pub fn verify(
        &self,
        /* same args as from_parts */
    ) -> bool;
}

pub fn verify_parts(
    root: &[u8; 32],
    /* same args as from_parts */
) -> bool;
```

`verify_parts` is a free-function variant for callers that already have a `[u8; 32]` (e.g. deserialized from a chain block) and don't want to construct the newtype. Bit-identical to the method form.

## Files

- `crates/katgpt-core/src/engram/architecture_root.rs` (431 LOC) — primitive + 13 unit tests.
- `crates/katgpt-core/src/engram/mod.rs` — module registration + layering doc.
- `crates/katgpt-core/benches/bench_039_architecture_root_goat.rs` (306 LOC) — GOAT gate.
- `crates/katgpt-core/Cargo.toml` — feature flag + default-on promotion + bench registration.

## Reproduce

```bash
# GOAT bench
cargo bench -p katgpt-core --bench bench_039_architecture_root_goat -- --nocapture

# Unit tests (default features now include the primitive)
cargo test -p katgpt-core --lib architecture_root

# Regression
cargo check --all-features
cargo check --no-default-features
cargo test -p katgpt-core --lib  # 1029 passed, 0 failed
```

## Cross-references

- **Issue:** `katgpt-rs/.issues/039_whole_architecture_commitment.md`
- **Component commitments:** `engram/commitment.rs` (`EngramTableId`), `closure/mod.rs:84-90` (PTG BLAKE3), `riir-neuron-db/src/shard.rs` (`NeuronShard::merkle_root`).
- **Freeze/thaw Lean proof (sibling):** `riir-ai/.proofs/RiirAiProof/Runtime/FreezeThaw.lean` (Issue 348 T2). The architecture root is the natural extension target — currently freeze/thaw operates per-shard; whole-architecture freeze is the next step (out of scope here, but this primitive enables it).
- **Chain consumer (deferred):** `riir-chain/.research/007_Engram_LatCal_Commitment_Bridge.md` — once this primitive ships, the chain can add a `SyncBlock` field carrying `CognitiveArchitectureRoot` for quorum-attested personality checkpoints. File a `riir-chain/.issues/*` when a chain consumer lands.

## TL;DR

`CognitiveArchitectureRoot([u8; 32])` — a single BLAKE3 pass over the existing per-component roots (PTG + EngramTable + ShardSet + FunctorSig) plus the (tick, npc_id) binding pair. ~200 bytes hashed, ~208 ns measured. Zero-alloc, stack-only newtype mirroring `EngramTableId`. All GOAT gates PASS, promoted to DEFAULT-ON. Pure modelless (no training, no softmax). Enables anti-cheat on cognitive state, quorum-attested personality freeze/thaw, and on-chain NPC avatar portability.
