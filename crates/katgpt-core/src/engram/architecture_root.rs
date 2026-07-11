//! Whole-architecture commitment — single tamper-evident root over an NPC's
//! full cognitive architecture.
//!
//! Issue 039. Every atom of an NPC's cognitive architecture (PTG, EngramTable,
//! NeuronShard, functor signature set) already has its own BLAKE3 commitment.
//! This module adds the unifying layer: a single BLAKE3 hash over the existing
//! per-component roots plus the (tick, npc_id) binding pair. The primitive
//! enables:
//!
//! - **Anti-cheat on cognitive state.** Tampering with any one component while
//!   leaving others bit-identical is detected at the root.
//! - **Quorum-attested personality freeze/thaw.** The freeze/thaw runtime
//!   (Issue 348 T2, proven in Lean) currently operates on individual shards;
//!   a whole-architecture root lets a chain quorum attest "this exact brain
//!   state" atomically.
//! - **Deterministic replay enrichment.** "Consistent at tick T" becomes a
//!   single checkable claim instead of N per-component checks.
//!
//! # Composition, not re-walk
//!
//! [`CognitiveArchitectureRoot`] hashes over the existing per-component roots
//! (each 32 bytes), NOT over the underlying data. A single BLAKE3 pass over
//! ~200 bytes is ~200 ns; re-walking the underlying data would be O(N) in
//! shard / node count and miss the point.
//!
//! # Hot-path contract
//!
//! [`CognitiveArchitectureRoot::from_parts`] and [`verify_parts`] are
//! **zero-allocation**: stack-only `[u8; 32]` newtype, single BLAKE3 hasher
//! reused across `update()` calls, no `Vec` / `Box` / `String` anywhere.
//!
//! # Bridge pattern (AGENTS.md)
//!
//! - Component roots (`[u8; 32]`) → raw, already-committed audit artifacts.
//! - Architecture root → raw, syncable, syncs through the chain as a single
//!   32-byte field on `SyncBlock` (deferred chain consumer concern).
//! - The primitive performs no latent-space operation; it is a composition
//!   of raw hashes.
//!
//! # Why this lives next to `commitment.rs`
//!
//! [`EngramTableId`] (sibling module) is the model for a 32-byte
//! content-addressed identity with a `from_*` constructor and a `verify`
//! method. [`CognitiveArchitectureRoot`] mirrors that shape at the
//! architecture layer. Reuses the `blake3` crate already in the workspace.
//!
//! # Feature gate
//!
//! Entire module is `#[cfg(feature = "cognitive_architecture_root")]`. The
//! feature is DEFAULT-ON (Issue 039 T5, 2026-07-04): all GOAT gates pass and
//! the primitive is pure modelless (BLAKE3 composition). The gate remains so
//! callers can opt OUT if they want to strip the API surface; turning it off
//! is zero-cost (every public API vanishes from the build).

use super::EngramTableId;

/// Tamper-evident root over an NPC's full cognitive architecture.
///
/// Computed as a single BLAKE3 hash over the existing per-component roots plus
/// the (tick, npc_id) binding pair:
///
/// ```text
/// blake3(ptg_root
///      || engram_table_id
///      || shard_set_root
///      || functor_sig_root
///      || tick.to_le_bytes()
///      || npc_id.to_le_bytes())
/// ```
///
/// Verification re-derives the same hash from a freshly-captured set of
/// component roots and compares. A mismatch indicates tampering in ANY
/// component (or in the binding tuple itself) — there is no way to flip a
/// bit in any input without flipping on average half the output bits.
///
/// # Layout
///
/// `#[derive(Copy, Clone, PartialEq, Eq, Hash)]` newtype around `[u8; 32]`.
/// `size_of::<Self>() == 32`, no padding, no indirection. Same shape as
/// [`EngramTableId`], so the two can be stored in the same arrays / maps.
///
/// # Determinism
///
/// Same inputs (component roots + tick + npc_id) → same architecture root,
/// always, regardless of host endianness (we hash canonical little-endian
/// bytes for `tick` and `npc_id`) or run order. This is the contract that
/// makes the root usable as a chain quorum-attested checkpoint.
///
/// # Empty-component convention
///
/// For NPCs that lack one of the components (e.g. an early-game NPC with no
/// shard set yet), the caller SHOULD pass `[0u8; 32]` as the absent root.
/// This is the same "padding leaf = zero hash" Merkle convention used by
/// [`super::build_merkle_root`]; the architecture root treats it as a
/// well-defined "absent" sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CognitiveArchitectureRoot(pub [u8; 32]);

impl CognitiveArchitectureRoot {
    /// Compute the architecture root from per-component roots + binding pair.
    ///
    /// # Arguments
    ///
    /// - `ptg_root` — BLAKE3 of the [`super::super::closure::PrimitiveTransitionGraph`]
    ///   (via [`super::super::closure::commitment`]).
    /// - `engram_table_id` — content-addressed identity of the EngramTable.
    /// - `shard_set_root` — Merkle root over the NPC's `NeuronShard` set
    ///   (computed by the chain layer via batch commit; `[0u8; 32]` if the
    ///   NPC has no shards yet).
    /// - `functor_sig_root` — BLAKE3 over the functor signature set. If
    ///   signatures live in an EngramTable, this is that table's
    ///   [`EngramTableId`]; otherwise the caller supplies whatever
    ///   commitment they have, or `[0u8; 32]` for "no signature set".
    /// - `tick` — the deterministic tick at which this snapshot was captured.
    ///   Binds the architecture to a replayable moment.
    /// - `npc_id` — stable NPC identifier. Disambiguates two NPCs that happen
    ///   to have identical architectures at the same tick (legitimate clones,
    ///   template instances, etc.) without conflating them.
    ///
    /// # Allocation
    ///
    /// Zero. Single `blake3::Hasher` on the stack, six `update()` calls,
    /// `finalize_xof().fill(&mut out)` writes directly into the returned
    /// `[u8; 32]`. No `Vec`, no `Box`, no `String`.
    #[inline]
    pub fn from_parts(
        ptg_root: &[u8; 32],
        engram_table_id: &EngramTableId,
        shard_set_root: &[u8; 32],
        functor_sig_root: &[u8; 32],
        tick: u32,
        npc_id: u64,
    ) -> Self {
        let mut out = [0u8; 32];
        // SAFETY: write_root_into writes exactly 32 bytes via fill().
        Self::write_root_into(
            ptg_root,
            engram_table_id,
            shard_set_root,
            functor_sig_root,
            tick,
            npc_id,
            &mut out,
        );
        CognitiveArchitectureRoot(out)
    }

    /// Re-derive the architecture root from a fresh capture and compare.
    ///
    /// Returns `true` iff recomputing [`from_parts`](Self::from_parts) from
    /// the supplied inputs yields a bit-identical root. A `false` result
    /// indicates tampering or corruption in any component, or in the binding
    /// pair.
    ///
    /// Equivalent to `*self == Self::from_parts(...)`, provided for symmetry
    /// with [`EngramTableId::verify`](super::EngramTableId::verify).
    #[inline]
    pub fn verify(
        &self,
        ptg_root: &[u8; 32],
        engram_table_id: &EngramTableId,
        shard_set_root: &[u8; 32],
        functor_sig_root: &[u8; 32],
        tick: u32,
        npc_id: u64,
    ) -> bool {
        *self
            == Self::from_parts(
                ptg_root,
                engram_table_id,
                shard_set_root,
                functor_sig_root,
                tick,
                npc_id,
            )
    }

    /// Shared hash-write path for [`from_parts`](Self::from_parts) and
    /// [`verify`](Self::verify). Inline-able so the optimizer sees a single
    /// BLAKE3 pass.
    ///
    /// Writes exactly 32 bytes into `out`. Order of `update()` calls is the
    /// canonical wire order (ptg || engram || shard || functor || tick ||
    /// npc_id); changing the order breaks cross-version compatibility.
    #[inline]
    fn write_root_into(
        ptg_root: &[u8; 32],
        engram_table_id: &EngramTableId,
        shard_set_root: &[u8; 32],
        functor_sig_root: &[u8; 32],
        tick: u32,
        npc_id: u64,
        out: &mut [u8; 32],
    ) {
        let mut h = blake3::Hasher::new();
        h.update(ptg_root);
        h.update(&engram_table_id.0);
        h.update(shard_set_root);
        h.update(functor_sig_root);
        h.update(&tick.to_le_bytes());
        h.update(&npc_id.to_le_bytes());
        h.finalize_xof().fill(out);
    }
}

/// Free-function variant of [`CognitiveArchitectureRoot::verify`] for callers
/// that already have a `[u8; 32]` root (e.g. deserialized from a chain block)
/// and don't want to construct the newtype. Bit-identical to the method form.
#[inline]
pub fn verify_parts(
    root: &[u8; 32],
    ptg_root: &[u8; 32],
    engram_table_id: &EngramTableId,
    shard_set_root: &[u8; 32],
    functor_sig_root: &[u8; 32],
    tick: u32,
    npc_id: u64,
) -> bool {
    let mut recomputed = [0u8; 32];
    CognitiveArchitectureRoot::write_root_into(
        ptg_root,
        engram_table_id,
        shard_set_root,
        functor_sig_root,
        tick,
        npc_id,
        &mut recomputed,
    );
    *root == recomputed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic EngramTableId for tests (avoids depending on the
    /// full engram builder machinery).
    fn eid(b: u8) -> EngramTableId {
        EngramTableId([b; 32])
    }

    /// Count the number of differing bits between two 32-byte arrays. BLAKE3
    /// avalanche target: ≥ 50% of bits flip on a single-bit input mutation.
    fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
        let mut d = 0;
        for i in 0..32 {
            d += (a[i] ^ b[i]).count_ones();
        }
        d
    }

    // ─── G1 (correctness): spec-match ───────────────────────────────────────

    #[test]
    fn size_of_is_32() {
        // G4 (alloc-free): struct is exactly 32 bytes, no padding.
        assert_eq!(std::mem::size_of::<CognitiveArchitectureRoot>(), 32);
    }

    #[test]
    fn from_parts_is_deterministic() {
        // Same inputs → same root, every time.
        let root_a = CognitiveArchitectureRoot::from_parts(
            &[0xAA; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        let root_b = CognitiveArchitectureRoot::from_parts(
            &[0xAA; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        assert_eq!(
            root_a, root_b,
            "identical inputs must produce identical roots"
        );
    }

    #[test]
    fn verify_round_trip_passes_on_identical_inputs() {
        // The canonical happy path: verify() returns true when nothing changed.
        let root = CognitiveArchitectureRoot::from_parts(
            &[0x11; 32],
            &eid(2),
            &[0x22; 32],
            &[0x33; 32],
            100,
            7,
        );
        assert!(
            root.verify(&[0x11; 32], &eid(2), &[0x22; 32], &[0x33; 32], 100, 7),
            "verify must succeed on identical inputs"
        );
    }

    #[test]
    fn verify_free_function_matches_method() {
        // verify_parts must agree with the method form bit-identically.
        let root = CognitiveArchitectureRoot::from_parts(
            &[0x11; 32],
            &eid(2),
            &[0x22; 32],
            &[0x33; 32],
            100,
            7,
        );
        let method = root.verify(&[0x11; 32], &eid(2), &[0x22; 32], &[0x33; 32], 100, 7);
        let free_fn = verify_parts(
            &root.0,
            &[0x11; 32],
            &eid(2),
            &[0x22; 32],
            &[0x33; 32],
            100,
            7,
        );
        assert_eq!(method, free_fn, "method and free-fn forms must agree");
        assert!(method);
    }

    // ─── G1: avalanche (single-bit input mutation changes ≥ N output bits) ──

    #[test]
    fn ptg_root_single_bit_flip_breaks_verify() {
        let ptg = [0x55; 32];
        let root =
            CognitiveArchitectureRoot::from_parts(&ptg, &eid(1), &[0xBB; 32], &[0xCC; 32], 42, 99);
        let mut tampered = ptg;
        tampered[0] ^= 1; // flip the LSB of byte 0
        assert!(
            !root.verify(&tampered, &eid(1), &[0xBB; 32], &[0xCC; 32], 42, 99),
            "single-bit PTG mutation must break verify"
        );
    }

    #[test]
    fn engram_table_id_single_bit_flip_breaks_verify() {
        let root = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        let tampered = EngramTableId({
            let mut b = [1u8; 32];
            b[31] ^= 0x80; // flip the MSB of the last byte
            b
        });
        assert!(
            !root.verify(&[0x55; 32], &tampered, &[0xBB; 32], &[0xCC; 32], 42, 99),
            "single-bit engram mutation must break verify"
        );
    }

    #[test]
    fn shard_set_root_single_bit_flip_breaks_verify() {
        let shard = [0xBB; 32];
        let root = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &shard,
            &[0xCC; 32],
            42,
            99,
        );
        let mut tampered = shard;
        tampered[10] ^= 0x40;
        assert!(
            !root.verify(&[0x55; 32], &eid(1), &tampered, &[0xCC; 32], 42, 99),
            "single-bit shard mutation must break verify"
        );
    }

    #[test]
    fn functor_sig_root_single_bit_flip_breaks_verify() {
        let sig = [0xCC; 32];
        let root =
            CognitiveArchitectureRoot::from_parts(&[0x55; 32], &eid(1), &[0xBB; 32], &sig, 42, 99);
        let mut tampered = sig;
        tampered[5] ^= 0x01;
        assert!(
            !root.verify(&[0x55; 32], &eid(1), &[0xBB; 32], &tampered, 42, 99),
            "single-bit functor-sig mutation must break verify"
        );
    }

    #[test]
    fn tick_change_breaks_verify() {
        let root = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        assert!(
            !root.verify(&[0x55; 32], &eid(1), &[0xBB; 32], &[0xCC; 32], 43, 99),
            "tick change must break verify"
        );
    }

    #[test]
    fn npc_id_change_breaks_verify() {
        let root = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        assert!(
            !root.verify(&[0x55; 32], &eid(1), &[0xBB; 32], &[0xCC; 32], 42, 100),
            "npc_id change must break verify"
        );
    }

    #[test]
    fn avalanche_ptg_root_single_bit_flips_many_output_bits() {
        // BLAKE3 avalanche target: ≥ 50% of output bits flip on a 1-bit input
        // change. 32 bytes = 256 bits; ≥ 50% = ≥ 128 bits. Allow some slack
        // for run-to-run variance: require ≥ 96 bits (37.5%) — well above the
        // "no avalanche" floor of 1 bit. In practice BLAKE3 reliably gives
        // ~128 ± 12 bits; the floor guards against catastrophic regression
        // to a near-collision.
        let ptg = [0x55; 32];
        let root_a =
            CognitiveArchitectureRoot::from_parts(&ptg, &eid(1), &[0xBB; 32], &[0xCC; 32], 42, 99);
        let mut tampered = ptg;
        tampered[0] ^= 1;
        let root_b = CognitiveArchitectureRoot::from_parts(
            &tampered,
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        let dist = hamming_distance(&root_a.0, &root_b.0);
        assert!(
            dist >= 96,
            "avalanche: expected ≥ 96 differing bits, got {dist}/256 (BLAKE3 should be ~128)"
        );
    }

    #[test]
    fn distinct_npcs_have_distinct_roots_even_with_identical_components() {
        // Two NPCs with the same architecture at the same tick must still
        // differ by npc_id — this is the binding-pair contract that prevents
        // cross-NPC conflation.
        let r1 = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            1,
        );
        let r2 = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            2,
        );
        assert_ne!(r1, r2, "distinct npc_ids must produce distinct roots");
    }

    #[test]
    fn zero_root_absent_component_convention_is_stable() {
        // The "absent component = [0u8; 32]" sentinel convention is
        // well-defined: an NPC with no shard set today must verify against
        // the same zero-root tomorrow.
        let root_a = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0u8; 32],
            &[0xCC; 32],
            42,
            99,
        );
        let root_b = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0u8; 32],
            &[0xCC; 32],
            42,
            99,
        );
        assert_eq!(root_a, root_b);
        // And it differs from an NPC that DOES have a shard set.
        let root_with_shard = CognitiveArchitectureRoot::from_parts(
            &[0x55; 32],
            &eid(1),
            &[0xBB; 32],
            &[0xCC; 32],
            42,
            99,
        );
        assert_ne!(root_a, root_with_shard);
    }
}
