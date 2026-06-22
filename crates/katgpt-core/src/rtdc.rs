//! RTDC — Resolution-Tiered Deterministic Commitment.
//!
//! Wraps the existing `MerkleOctree` (Plan 253) with per-depth roots aligned
//! to SLoD (Plan 235) σ-boundaries. Each depth corresponds to an abstraction
//! level, so a curator at distance d* can verify inclusion at the abstraction
//! level it actually operates at, with proof size shrinking at coarser depths.
//!
//! ## Capability
//!
//! **Trust-minimized semantic zoom.** A light client cryptographically
//! verifies its fog-of-war view is a faithful sub-summation of the
//! chain-committed full KG, with O(log n) BLAKE3 proof at the abstraction
//! level it operates at — no full-KG download, no trust in the serving node.
//!
//! ## Phase 1 (this module)
//!
//! - 3 deterministic roots derived from the octree: depth-0 (global),
//!   depth-1 (regional, 8 internal nodes), depth-2 (fine, 64 leaves).
//! - `DepthTieredMerkleOctree::build` consumes a `MerkleOctree` and
//!   `&[ScaleBoundary]` (≥2 boundaries required).
//! - `prove_at_depth(d=2)` is fully sound (full Merkle path, 14 siblings).
//! - `prove_at_depth(d=1)` reconstructs the parent internal node hash from
//!   7 leaf-level siblings; soundness that this internal is in `roots[1]`
//!   requires the Phase 2 `subtree_inclusion` proof.
//! - `prove_at_depth(d=0)` returns a trivial proof; full soundness also
//!   requires Phase 2.
//! - `DeterministicLeafEncode` trait — the encoding contract for
//!   cross-platform bit-identical leaf hashing.
//!
//! ## What Phase 1 does NOT deliver
//!
//! - Cross-depth consistency proof (`subtree_inclusion`) — tracked in
//!   `riir-chain/issues/002_rtdc_subtree_inclusion_research.md`.
//! - LatCal-backed `DeterministicLeafEncode` impl — lives in riir-chain
//!   (Plan 003, gated on this module's trait existing).
//! - Chain quorum over 3 roots — riir-chain Plan 003.
//!
//! Plan 302 Phase 1 (T1.1–T1.6). Open primitive, public MIT.

use crate::merkle::{
    HASH_SIZE, MERKLE_OCTREE_BRANCHING, MERKLE_OCTREE_INTERNAL, MERKLE_OCTREE_LEAVES, MerkleOctree,
    MerkleProof,
};
use crate::slod::ScaleBoundary;

/// BLAKE3 domain-separation tag for the regional root (`roots[1]`).
/// Ensures the regional root is distinct from the fine root (`roots[2]`),
/// which the existing `MerkleOctree` derives from the same 8 internal
/// node hashes (without a tag).
const RTDC_REGIONAL_TAG: &[u8] = b"rtdc_regional_v1";

/// BLAKE3 domain-separation tag for the global root (`roots[0]`).
const RTDC_GLOBAL_TAG: &[u8] = b"rtdc_global_v1";

// ─── Depth-tiered roots ────────────────────────────────────────────────

/// One Merkle root per depth tier. Depth-3 octree ⇒ 3 roots.
///
/// Each root is a distinct commitment derived from the same `MerkleOctree`:
///
/// | Index | Name       | Commits to                                                |
/// |-------|------------|-----------------------------------------------------------|
/// | `0`   | global     | `BLAKE3(roots[2])` — single hash of the fine root.        |
/// | `1`   | regional   | `BLAKE3(h₁ ‖ h₂ ‖ … ‖ h₈)` — flat hash of 8 internal nodes.|
/// | `2`   | fine       | `inner.root()` — the full Merkle root (transitively       |
/// |       |            | commits to all 64 leaves).                                |
///
/// The three roots are distinct by construction (different input domains),
/// so a change to any leaf propagates: leaves → `roots[2]` → `roots[0]`,
/// and the internal re-hash propagates to `roots[1]`.
///
/// **Soundness caveat (Phase 1):** each root is independently correct, but
/// cross-depth consistency (`roots[d]` faithfully aggregates `roots[d+1]`)
/// requires the Phase 2 `subtree_inclusion` proof — see
/// `riir-chain/issues/002_rtdc_subtree_inclusion_research.md`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthTieredRoots {
    /// `[roots[0], roots[1], roots[2]]` as described in the type docs.
    pub roots: [[u8; HASH_SIZE]; 3],
}

impl DepthTieredRoots {
    /// All-zero placeholder. Useful for `MaybeUninit` patterns and tests.
    pub const EMPTY: Self = Self {
        roots: [[0u8; HASH_SIZE]; 3],
    };
}

// ─── Depth selector ────────────────────────────────────────────────────

/// Selects which depth to verify at, given a continuous σ.
///
/// Built from SLoD `ScaleBoundary` set (must have ≥2 boundaries for 3 tiers).
#[derive(Clone, Debug)]
pub struct DepthSelector {
    /// σ thresholds from SLoD `boundary_scan`, ascending.
    ///
    /// - `sigma_thresholds[0]` = boundary between depth-2 (fine) and depth-1 (regional).
    /// - `sigma_thresholds[1]` = boundary between depth-1 (regional) and depth-0 (global).
    pub sigma_thresholds: [f32; 2],
}

impl DepthSelector {
    /// Map a query σ to a depth index.
    ///
    /// - `σ <= sigma_thresholds[0]`        → depth 2 (full detail)
    /// - `sigma_thresholds[0] < σ <= t₁`   → depth 1 (regional)
    /// - `σ > sigma_thresholds[1]`          → depth 0 (global)
    ///
    /// Branchless: count how many thresholds `σ` is at-or-below. Each
    /// `<=` adds 1, so the result is the depth index directly. (Earlier
    /// draft counted `>` crossings and returned the count — that gave the
    /// inverse mapping and was wrong.)
    #[inline(always)]
    pub fn select(&self, sigma: f32) -> usize {
        let at_or_below_t0 = (sigma <= self.sigma_thresholds[0]) as usize;
        let at_or_below_t1 = (sigma <= self.sigma_thresholds[1]) as usize;
        at_or_below_t0 + at_or_below_t1
    }

    /// Construct from SLoD boundaries. Returns `None` if `< 2` boundaries
    /// (3-tier RTDC needs 2 thresholds).
    ///
    /// SLoD's `boundary_scan` returns boundaries sorted ascending by σ — we
    /// trust that invariant here and do not re-sort.
    pub fn from_boundaries(boundaries: &[ScaleBoundary]) -> Option<Self> {
        if boundaries.len() < 2 {
            return None;
        }
        Some(Self {
            sigma_thresholds: [boundaries[0].sigma, boundaries[1].sigma],
        })
    }
}

// ─── Depth-tiered octree ───────────────────────────────────────────────

/// Depth-tiered Merkle octree. Wraps `MerkleOctree` with per-depth roots
/// and a SLoD-derived `DepthSelector`.
#[derive(Clone, Debug)]
pub struct DepthTieredMerkleOctree {
    inner: MerkleOctree,
    roots: DepthTieredRoots,
    selector: DepthSelector,
}

impl DepthTieredMerkleOctree {
    /// Build from a constructed `MerkleOctree` and SLoD boundaries.
    ///
    /// Derives 3 independent roots from the octree (see [`DepthTieredRoots`]
    /// for the exact derivation). Returns
    /// [`RtdcError::InsufficientBoundaries`] if `< 2` boundaries are supplied.
    ///
    /// **Phase 1 cost:** 9 BLAKE3 updates (8 for `roots[1]`, 1 for
    /// `roots[0]`) on top of the existing `MerkleOctree::build_from_leaves`.
    /// GOAT G1 target: ≤ 3× the bare-octree build cost.
    pub fn build(inner: MerkleOctree, boundaries: &[ScaleBoundary]) -> Result<Self, RtdcError> {
        let selector = DepthSelector::from_boundaries(boundaries).ok_or(
            RtdcError::InsufficientBoundaries {
                n: boundaries.len(),
                need: 2,
            },
        )?;

        // roots[2]: full Merkle root — commits transitively to all 64 leaves.
        let depth_2_root = *inner.root();

        // roots[1]: flat hash of 8 internal node hashes. Commits to regional
        // summaries but NOT to individual leaves — a verifier at depth-1
        // cannot reconstruct any specific leaf from this commitment alone.
        //
        // Domain separator: the existing `MerkleOctree` computes its root as
        // `BLAKE3(h₁ ‖ h₂ ‖ … ‖ h₈)` (no prefix), so without a tag this would
        // equal `roots[2]`. The `rtdc_regional_v1` tag follows BIP-340
        // tagged-hash practice and ensures all three roots are distinct.
        let mut h1 = blake3::Hasher::new();
        h1.update(RTDC_REGIONAL_TAG);
        for i in 0..MERKLE_OCTREE_INTERNAL {
            h1.update(&inner.hashes[1 + i]);
        }
        let depth_1_root = *h1.finalize().as_bytes();

        // roots[0]: global — single hash representing the whole KG. We hash
        // roots[2] under a domain separator so any leaf change propagates
        // here too, while remaining distinct from roots[1] and roots[2].
        // Phase 2 may replace this with the Fréchet centroid of all leaves
        // (see Research 280 §3.2) — wire-compatible as long as the encoding
        // is documented.
        let mut h0 = blake3::Hasher::new();
        h0.update(RTDC_GLOBAL_TAG);
        h0.update(&depth_2_root);
        let depth_0_root = *h0.finalize().as_bytes();

        Ok(Self {
            inner,
            roots: DepthTieredRoots {
                roots: [depth_0_root, depth_1_root, depth_2_root],
            },
            selector,
        })
    }

    /// The 3 depth-tiered roots (the sync payload).
    #[inline]
    pub fn roots(&self) -> &DepthTieredRoots {
        &self.roots
    }

    /// The SLoD-derived depth selector.
    #[inline]
    pub fn selector(&self) -> &DepthSelector {
        &self.selector
    }

    /// Convenience: select the appropriate depth for a query σ.
    #[inline]
    pub fn depth_for_sigma(&self, sigma: f32) -> usize {
        self.selector.select(sigma)
    }

    /// Reference to the inner `MerkleOctree`. Useful for forensics, tests,
    /// and Phase 2 cross-depth proof construction.
    #[inline]
    pub fn inner(&self) -> &MerkleOctree {
        &self.inner
    }

    /// Generate an inclusion proof for `leaf_index` at `depth`.
    ///
    /// Sibling count by depth:
    ///
    /// | Depth | Siblings | Levels traversed | Soundness |
    /// |-------|----------|------------------|-----------|
    /// | 0     | 0        | none             | Phase 2 (subtree_inclusion) |
    /// | 1     | 7        | leaf → internal  | Phase 2 (subtree_inclusion) |
    /// | 2     | 14       | leaf → internal → root | **Full (this Phase 1)** |
    ///
    /// At depth-2 the proof is the existing `MerkleProof` and is fully
    /// cryptographically sound. At depth-1 and depth-0 the proof establishes
    /// the leaf's position in the octree but does NOT prove the resulting
    /// internal/root hash is part of the published `roots[d]` — that
    /// requires the Phase 2 cross-depth consistency proof (see
    /// `riir-chain/issues/002_rtdc_subtree_inclusion_research.md`).
    ///
    /// Returns `None` if `depth > 2` or `leaf_index >= 64`.
    pub fn prove_at_depth(&self, leaf_index: u8, depth: usize) -> Option<RtdcProof> {
        if depth > 2 || leaf_index as usize >= MERKLE_OCTREE_LEAVES {
            return None;
        }

        let full = MerkleProof::generate(&self.inner, leaf_index)?;
        let n_siblings = siblings_for_depth(depth);

        Some(RtdcProof {
            leaf_index,
            depth,
            leaf_hash: full.leaf_hash,
            siblings: full.siblings[..n_siblings].to_vec(),
            expected_root: self.roots.roots[depth],
        })
    }

    /// Verify a `prove_at_depth` proof against a published `DepthTieredRoots`.
    ///
    /// Returns `true` only if the proof's `expected_root` matches
    /// `roots[proof.depth]` AND the leaf hash reconstructs to that root via
    /// the supplied siblings (where applicable).
    ///
    /// **Phase 1 soundness caveat:** at depth-0 and depth-1, this function
    /// returns `true` on any well-formed proof whose `expected_root` matches
    /// `roots[d]`. It does NOT establish that `roots[d]` is a faithful
    /// aggregation of `roots[d+1]`. Cross-depth soundness is Phase 2.
    pub fn verify_at_depth(proof: &RtdcProof, roots: &DepthTieredRoots) -> bool {
        if proof.depth > 2 {
            return false;
        }
        if proof.expected_root != roots.roots[proof.depth] {
            return false;
        }

        let expected_n_siblings = siblings_for_depth(proof.depth);
        if proof.siblings.len() != expected_n_siblings {
            return false;
        }

        match proof.depth {
            // Phase 1: depth-0 carries only the leaf hash; the global root
            // is a single hash of roots[2]. We accept on faith that the
            // curator's roots[0] commits to this leaf — Phase 2's
            // subtree_inclusion proof replaces this `true` with a real check.
            0 => true,
            // Phase 1: depth-1 reconstructs the parent internal node hash
            // from 7 leaf-level siblings. We check that hash is non-zero
            // (well-formed). Soundness that this internal is part of
            // roots[1] requires Phase 2.
            1 => {
                let position = (proof.leaf_index as usize) % MERKLE_OCTREE_BRANCHING;
                let internal_hash = recompute_parent(&proof.leaf_hash, &proof.siblings, position);
                internal_hash != [0u8; HASH_SIZE]
            }
            // Full Merkle proof — fully sound. Reuses MerkleProof::verify
            // for parity with the existing tested code path.
            2 => {
                // siblings_for_depth(2) = 2 * (MERKLE_OCTREE_BRANCHING - 1) = 14.
                // `MerkleProof::siblings` is a fixed `[[u8; 32]; 14]` array —
                // we already validated `proof.siblings.len() == 14` above via
                // `siblings_for_depth(proof.depth)`.
                const FULL_SIBLING_COUNT: usize = 2 * (MERKLE_OCTREE_BRANCHING - 1);
                let mut siblings = [[0u8; HASH_SIZE]; FULL_SIBLING_COUNT];
                siblings.copy_from_slice(&proof.siblings);
                let mp = MerkleProof {
                    leaf_index: proof.leaf_index,
                    leaf_hash: proof.leaf_hash,
                    siblings,
                };
                mp.verify(&roots.roots[2])
            }
            _ => false,
        }
    }
}

/// Number of sibling hashes a depth-d proof carries.
#[inline]
fn siblings_for_depth(depth: usize) -> usize {
    match depth {
        0 => 0,
        1 => MERKLE_OCTREE_BRANCHING - 1, // 7 leaf-level siblings
        2 => 2 * (MERKLE_OCTREE_BRANCHING - 1), // 14 (full Merkle path)
        _ => 0,
    }
}

/// Recompute a parent hash from one child + (k-1) siblings.
///
/// `position` is the child's slot among the parent's 8 children
/// (`0..MERKLE_OCTREE_BRANCHING`). Mirrors the inner loop of
/// `MerkleProof::verify` but on a borrowed slice.
#[inline]
fn recompute_parent(
    child: &[u8; HASH_SIZE],
    siblings: &[[u8; HASH_SIZE]],
    position: usize,
) -> [u8; HASH_SIZE] {
    debug_assert_eq!(siblings.len(), MERKLE_OCTREE_BRANCHING - 1);
    debug_assert!(position < MERKLE_OCTREE_BRANCHING);
    let mut hasher = blake3::Hasher::new();
    let mut sib_idx = 0;
    for c in 0..MERKLE_OCTREE_BRANCHING {
        if c == position {
            hasher.update(child);
        } else {
            hasher.update(&siblings[sib_idx]);
            sib_idx += 1;
        }
    }
    *hasher.finalize().as_bytes()
}

// ─── Proof types ───────────────────────────────────────────────────────

/// Inclusion proof at a specific depth.
///
/// See [`DepthTieredMerkleOctree::prove_at_depth`] for the sibling-count
/// contract per depth.
#[derive(Clone, Debug)]
pub struct RtdcProof {
    /// Leaf index in `[0..64)`.
    pub leaf_index: u8,
    /// Depth this proof was generated for (`0..=2`).
    pub depth: usize,
    /// BLAKE3 hash of the leaf being proven.
    pub leaf_hash: [u8; HASH_SIZE],
    /// Sibling hashes along the truncated Merkle path. Length depends on
    /// `depth` — see `siblings_for_depth`.
    pub siblings: Vec<[u8; HASH_SIZE]>,
    /// Precomputed `roots[depth]` from the issuing curator. The verifier
    /// checks this matches its locally-trusted `DepthTieredRoots.roots[depth]`.
    pub expected_root: [u8; HASH_SIZE],
}

/// Cross-depth sub-summation proof. **Phase 2 deliverable.**
///
/// Proves that `roots[d]` is a faithful aggregation of `roots[d+1]`. Empty
/// stub in Phase 1 — protocol TBD, tracked in
/// `riir-chain/issues/002_rtdc_subtree_inclusion_research.md`.
#[derive(Clone, Debug, Default)]
pub struct SubtreeProof {
    pub shallow_depth: usize,
    pub deep_depth: usize,
    /// Phase 2 — encoding TBD (Pedersen commitment / FFT batch / sampling).
    pub proof_bytes: Vec<u8>,
}

// ─── Deterministic encoding contract ───────────────────────────────────

/// Deterministic leaf encoder contract.
///
/// Replaces raw `f32` embedding bytes (which are NOT bit-identical across
/// ARM64 / x86_64 / wasm32 due to NaN payloads and subnormals) with a
/// platform-independent byte encoding. Implementations MUST guarantee
/// bit-identical output for logically equal input across all targets.
///
/// **The LatCal-backed impl lives in `riir-chain`** (Plan 003) — katgpt-rs
/// sees only the `[u8; N]` contract. This keeps the encoding IP private
/// while letting the open primitive verify any conforming encoder.
pub trait DeterministicLeafEncode {
    /// Encode `self` into `out`. Callers MUST size `out` to at least
    /// [`Self::encode_len`] bytes.
    ///
    /// Contract:
    /// - Pure function of `self` (no I/O, no global state, no allocation).
    /// - Output length is fixed for a given `Self` type.
    /// - Bit-identical across ARM64 / x86_64 / wasm32 for logically equal input.
    /// - Suitable for direct BLAKE3 hashing (caller hashes `out[..encode_len()]`).
    fn encode_deterministic(&self, out: &mut [u8]);

    /// Fixed output length. Callers use this to size `out`.
    fn encode_len() -> usize;
}

// ─── Errors ────────────────────────────────────────────────────────────

/// Errors returned by RTDC operations.
///
/// Hand-rolled `Display` + `Error` impls (matches the existing convention
/// in `engram/tokenizer.rs::SurjectiveMapLoadError` — katgpt-core does not
/// depend on `thiserror`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RtdcError {
    /// `DepthTieredMerkleOctree::build` was called with fewer than 2 SLoD
    /// boundaries (3-tier RTDC needs 2 thresholds).
    InsufficientBoundaries { n: usize, need: usize },
    /// A depth argument was outside `0..=2`.
    DepthOutOfRange { depth: usize },
    /// A leaf index was outside `0..64`.
    LeafOutOfRange { index: u8 },
}

impl std::fmt::Display for RtdcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientBoundaries { n, need } => {
                write!(f, "need {need} SLoD boundaries for 3-tier RTDC, got {n}")
            }
            Self::DepthOutOfRange { depth } => {
                write!(f, "depth {depth} out of range (0..=2)")
            }
            Self::LeafOutOfRange { index } => {
                write!(f, "leaf index {index} out of range (0..64)")
            }
        }
    }
}

impl std::error::Error for RtdcError {}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::{HASH_SIZE, MERKLE_OCTREE_LEAVES, MerkleOctree};

    fn dummy_boundaries() -> Vec<ScaleBoundary> {
        vec![
            ScaleBoundary {
                sigma: 0.5,
                score: 1.0,
                k_star: 8,
            },
            ScaleBoundary {
                sigma: 2.0,
                score: 0.5,
                k_star: 2,
            },
        ]
    }

    fn populated_octree() -> MerkleOctree {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }
        MerkleOctree::build_from_leaves(&leaf_hashes)
    }

    // ── DepthSelector ──

    #[test]
    fn selector_branchless_at_boundaries() {
        let s = DepthSelector {
            sigma_thresholds: [0.5, 2.0],
        };
        // Below t0 → depth 2
        assert_eq!(s.select(0.0), 2);
        assert_eq!(s.select(0.5), 2); // boundary inclusive (≤)
        // Between t0 and t1 → depth 1
        assert_eq!(s.select(0.5_f32 + 1e-6), 1);
        assert_eq!(s.select(1.0), 1);
        assert_eq!(s.select(2.0), 1); // boundary inclusive
        // Above t1 → depth 0
        assert_eq!(s.select(2.0_f32 + 1e-6), 0);
        assert_eq!(s.select(f32::INFINITY), 0);
    }

    #[test]
    fn selector_from_boundaries_needs_two() {
        let one = vec![ScaleBoundary {
            sigma: 1.0,
            score: 0.0,
            k_star: 0,
        }];
        assert!(DepthSelector::from_boundaries(&one).is_none());

        let s = DepthSelector::from_boundaries(&dummy_boundaries()).unwrap();
        assert_eq!(s.sigma_thresholds, [0.5, 2.0]);
    }

    // ── DepthTieredMerkleOctree::build ──

    #[test]
    fn build_rejects_insufficient_boundaries() {
        let octree = populated_octree();
        let err = DepthTieredMerkleOctree::build(octree, &[]).unwrap_err();
        match err {
            RtdcError::InsufficientBoundaries { n, need } => {
                assert_eq!(n, 0);
                assert_eq!(need, 2);
            }
            other => panic!("expected InsufficientBoundaries, got {other:?}"),
        }
    }

    #[test]
    fn build_derives_three_distinct_roots() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let roots = tree.roots();

        // roots[2] == inner.root()
        assert_eq!(roots.roots[2], *tree.inner().root());
        // All three distinct
        assert_ne!(roots.roots[0], roots.roots[1]);
        assert_ne!(roots.roots[1], roots.roots[2]);
        assert_ne!(roots.roots[0], roots.roots[2]);
        // None zero
        for r in &roots.roots {
            assert_ne!(*r, [0u8; HASH_SIZE]);
        }
    }

    #[test]
    fn build_roots_propagate_leaf_changes() {
        let octree_a = populated_octree();

        // Flip one leaf hash and rebuild.
        let mut leaf_hashes_b = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes_b[i] = *blake3::hash(&buf).as_bytes();
        }
        leaf_hashes_b[0][0] ^= 0xFF;
        let octree_b = MerkleOctree::build_from_leaves(&leaf_hashes_b);

        let tree_a = DepthTieredMerkleOctree::build(octree_a, &dummy_boundaries()).unwrap();
        let tree_b = DepthTieredMerkleOctree::build(octree_b, &dummy_boundaries()).unwrap();

        // All three roots must differ when any leaf changes.
        for d in 0..3 {
            assert_ne!(
                tree_a.roots().roots[d],
                tree_b.roots().roots[d],
                "roots[{d}] must change when leaf 0 changes"
            );
        }
    }

    // ── prove_at_depth / verify_at_depth ──

    #[test]
    fn prove_depth_2_is_sound_for_all_leaves() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let roots = *tree.roots();

        for i in 0..MERKLE_OCTREE_LEAVES {
            let proof = tree
                .prove_at_depth(i as u8, 2)
                .unwrap_or_else(|| panic!("prove_at_depth({i}, 2) returned None"));
            assert_eq!(proof.depth, 2);
            assert_eq!(proof.siblings.len(), 14);
            assert!(
                DepthTieredMerkleOctree::verify_at_depth(&proof, &roots),
                "depth-2 proof for leaf {i} must verify"
            );
        }
    }

    #[test]
    fn prove_depth_2_rejects_tampered_leaf_hash() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let roots = *tree.roots();

        let mut proof = tree.prove_at_depth(7, 2).unwrap();
        assert!(DepthTieredMerkleOctree::verify_at_depth(&proof, &roots));

        // Flip a bit in the leaf hash — must fail at depth 2.
        proof.leaf_hash[0] ^= 0xFF;
        assert!(
            !DepthTieredMerkleOctree::verify_at_depth(&proof, &roots),
            "tampered leaf hash must fail depth-2 verification"
        );
    }

    #[test]
    fn prove_depth_2_rejects_wrong_root() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let mut roots = *tree.roots();
        let proof = tree.prove_at_depth(0, 2).unwrap();

        // Tamper with roots[2] — depth-2 must fail.
        roots.roots[2][0] ^= 0xFF;
        assert!(!DepthTieredMerkleOctree::verify_at_depth(&proof, &roots));
    }

    #[test]
    fn prove_depth_1_well_formed_but_phase1_unsound() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let roots = *tree.roots();

        // Phase 1 contract: depth-1 returns 7 siblings and verifies when
        // expected_root matches roots[1]. Documented unsoundness: it would
        // also verify for a tampered roots[1] that still differs from roots[2].
        let proof = tree.prove_at_depth(13, 1).unwrap();
        assert_eq!(proof.depth, 1);
        assert_eq!(proof.siblings.len(), 7);
        assert!(DepthTieredMerkleOctree::verify_at_depth(&proof, &roots));
    }

    #[test]
    fn prove_depth_0_trivial() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let roots = *tree.roots();

        let proof = tree.prove_at_depth(42, 0).unwrap();
        assert_eq!(proof.depth, 0);
        assert!(proof.siblings.is_empty());
        // Phase 1: returns true when expected_root matches roots[0].
        assert!(DepthTieredMerkleOctree::verify_at_depth(&proof, &roots));
    }

    #[test]
    fn prove_rejects_out_of_range_inputs() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();

        assert!(tree.prove_at_depth(0, 3).is_none());
        assert!(tree.prove_at_depth(64, 2).is_none());
        assert!(tree.prove_at_depth(255, 0).is_none());
    }

    #[test]
    fn verify_rejects_depth_mismatch() {
        let octree = populated_octree();
        let tree = DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).unwrap();
        let roots = *tree.roots();

        // depth=2 proof but swap expected_root to roots[1] — must fail.
        let mut proof = tree.prove_at_depth(5, 2).unwrap();
        proof.expected_root = roots.roots[1];
        assert!(!DepthTieredMerkleOctree::verify_at_depth(&proof, &roots));
    }

    // ── Sibling-count contract ──

    #[test]
    fn siblings_for_depth_table() {
        assert_eq!(siblings_for_depth(0), 0);
        assert_eq!(siblings_for_depth(1), 7);
        assert_eq!(siblings_for_depth(2), 14);
        assert_eq!(siblings_for_depth(3), 0); // invalid → 0
    }

    // ── DeterministicLeafEncode trait shape ──

    #[test]
    fn deterministic_leaf_encode_trait_compiles() {
        // Smoke test: a trivial impl produces stable bytes.
        struct Foo;
        impl DeterministicLeafEncode for Foo {
            fn encode_deterministic(&self, out: &mut [u8]) {
                out[0] = 0x42;
            }
            fn encode_len() -> usize {
                1
            }
        }
        let mut buf = [0u8; 1];
        Foo.encode_deterministic(&mut buf);
        assert_eq!(buf, [0x42]);
        assert_eq!(Foo::encode_len(), 1);
    }
}
