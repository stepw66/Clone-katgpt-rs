//! Decoupled type definitions for the chunked content store (Plan 272 T1.4).
//!
//! Per AGENTS.md ("Use `types.rs` for decoupled structs/impls"): all POD types
//! and pure helper methods live here. Behaviour that requires a backing store
//! lives in [`crate::content_store::r#trait`] / [`crate::content_store::in_memory`].

use bytemuck::{Pod, Zeroable};

/// A blob's content-addressed identity = Merkle root over its chunk hashes.
///
/// 32 bytes (BLAKE3 output). Two blobs with byte-identical content always share
/// a `BlobId`. Two blobs that share *some* chunks (e.g. two sword variants with
/// a common texture) share chunk hashes but differ in `BlobId`.
///
/// `#[repr(transparent)]` so it can be `bytemuck::cast_ref` / reinterpreted as
/// `&[u8; 32]` with zero overhead, and so it round-trips through `transmute`
/// for FFI / sync wire formats.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct BlobId(pub [u8; 32]);

impl BlobId {
    /// The zero BlobId — corresponds to the Merkle root of an empty leaf set,
    /// i.e. `blake3::hash(b"").into()`. Used as the canonical id of an
    /// empty blob and as the zero-padding leaf in
    /// [`crate::content_store::merkle::build_binary_merkle_root`].
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        // Cache the empty-hash at compile time? blake3 doesn't expose a const
        // fn, so compute once via a helper. For hot paths, callers should pass
        // `&BlobId` rather than reconstructing.
        Self(*blake3::hash(b"").as_bytes())
    }

    /// View as a 32-byte slice (FFI / hashing).
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl From<[u8; 32]> for BlobId {
    #[inline]
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl From<BlobId> for [u8; 32] {
    #[inline]
    fn from(id: BlobId) -> Self {
        id.0
    }
}

/// A byte range within a blob — used by [`super::r#trait::ChunkFetcher`] for
/// partial / sparse hydration (e.g. fetch only LOD-0 of a multi-LOD asset).
///
/// `length: u32` caps a single range at 4 GiB, which exceeds any single chunk
/// (max 64 KiB by default) and any realistic LOD section.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkRange {
    /// Byte offset from the start of the blob.
    pub offset: u64,
    /// Number of bytes to read.
    pub length: u32,
}

impl ChunkRange {
    /// Construct a new range. Asserts `length` fits in the blob's logical extent
    /// when used by a fetcher (caller responsibility).
    #[inline]
    #[must_use]
    pub const fn new(offset: u64, length: u32) -> Self {
        Self { offset, length }
    }

    /// One-past-the-end byte offset (`offset + length`), saturating at `u64::MAX`.
    #[inline]
    #[must_use]
    pub const fn end(&self) -> u64 {
        // `as u64` cast in const fn (stable); `u64::from(u32)` is not yet
        // stable as a const trait fn (issue rust-lang/rust#143874).
        self.offset.saturating_add(self.length as u64)
    }
}

/// A Merkle inclusion proof for a single leaf (chunk hash) in a binary tree.
///
/// Constructed by [`crate::content_store::merkle::build_binary_merkle_proof`]
/// and verified by
/// [`crate::content_store::merkle::verify_binary_merkle_proof`] (pure BLAKE3,
/// no store access — light-client friendly, G4).
///
/// `siblings` is ordered from the leaf's immediate sibling up to the root's
/// child level. Combined with `leaf_index`, this determines left/right
/// ordering at each level via the leaf's bit at that depth.
#[derive(Clone, Debug)]
pub struct MerkleProof {
    /// Position of the leaf in the bottom level of the (padded) tree.
    pub leaf_index: usize,
    /// Sibling hashes from leaf level → root level. `len() == tree_depth`.
    pub siblings: Vec<[u8; 32]>,
    /// The expected root. Verifiers compare their recomputed root against this.
    pub expected_root: [u8; 32],
}

/// Aggregate statistics for a chunked content store (Plan 272 T1.4 / Research 262 §2.1).
///
/// `dedup_ratio = total_bytes_logical / total_bytes_stored`. A ratio of 1.0
/// means no dedup; 5.0 means 5× storage savings (GOAT gate G1 target).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StoreStats {
    /// Unique chunks currently in the chunk index.
    pub n_chunks_stored: u64,
    /// Blobs indexed (blob_id → metadata). May exceed `n_chunks_stored`'s
    /// implied unique count when blobs share chunks.
    pub n_blobs_indexed: u64,
    /// Sum of bytes across all unique stored chunks (the dedup'd footprint).
    pub total_bytes_stored: u64,
    /// Sum of logical bytes across all indexed blobs (what callers handed to `put`).
    pub total_bytes_logical: u64,
    /// `total_bytes_logical / total_bytes_stored`. Recompute via
    /// [`StoreStats::compute_dedup_ratio`] after mutating the counters.
    pub dedup_ratio: f32,
}

impl StoreStats {
    /// Recompute `dedup_ratio` from `total_bytes_logical / total_bytes_stored`.
    ///
    /// Returns `1.0` when `total_bytes_stored == 0` (empty store: trivially no
    /// dedup, no division by zero).
    #[inline]
    pub fn compute_dedup_ratio(&mut self) {
        self.dedup_ratio = if self.total_bytes_stored == 0 {
            1.0
        } else {
            // u64 → f32 cast is exact for any realistic byte count (< 2^53);
            // the division is the only lossy step, which is acceptable for a
            // summary statistic. Use `as f64` for the division to retain
            // precision before f32 narrowing.
            (self.total_bytes_logical as f64 / self.total_bytes_stored as f64) as f32
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_id_transparent_layout() {
        // repr(transparent) — BlobId and [u8; 32] must have identical layout.
        assert_eq!(
            std::mem::size_of::<BlobId>(),
            std::mem::size_of::<[u8; 32]>()
        );
        assert_eq!(
            std::mem::align_of::<BlobId>(),
            std::mem::align_of::<[u8; 32]>()
        );
    }

    #[test]
    fn test_blob_id_roundtrip() {
        let bytes = [7u8; 32];
        let id = BlobId::from(bytes);
        assert_eq!(id.as_bytes(), &bytes);
        let back: [u8; 32] = id.into();
        assert_eq!(back, bytes);
    }

    #[test]
    fn test_blob_id_zero_is_empty_hash() {
        let id = BlobId::zero();
        let expected: [u8; 32] = blake3::hash(b"").into();
        assert_eq!(id.0, expected);
    }

    #[test]
    fn test_chunk_range_end_saturates() {
        let r = ChunkRange::new(u64::MAX, 1);
        assert_eq!(r.end(), u64::MAX);
        let r = ChunkRange::new(100, 50);
        assert_eq!(r.end(), 150);
    }

    #[test]
    fn test_store_stats_dedup_ratio() {
        let mut s = StoreStats::default();
        s.compute_dedup_ratio();
        assert!((s.dedup_ratio - 1.0).abs() < 1e-6, "empty store: ratio 1.0");

        s.total_bytes_stored = 100;
        s.total_bytes_logical = 500;
        s.compute_dedup_ratio();
        assert!((s.dedup_ratio - 5.0).abs() < 1e-6, "5x dedup");
    }
}
