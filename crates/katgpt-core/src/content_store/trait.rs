//! Traits for the chunked content store (Plan 272 T1.5).
//!
//! File named `trait.rs` (reserved keyword) — referenced from
//! [`super::mod`] as `r#trait`. Three traits, mirroring Research 262 §2.1:
//!
//! - [`ChunkingStrategy`] — pure: borrowed slice → borrowed slices, zero-alloc.
//! - [`ChunkFetcher`] — fetcher for sparse hydration (LOD-aware reads).
//! - [`ChunkedContentStore`] — the main put/get/prove/verify contract.

use super::types::{BlobId, ChunkRange, MerkleProof, StoreStats};

/// Chunking strategy — converts raw bytes into borrowed slices.
///
/// Implementations: [`super::chunker::FixedSizeChunker`] (Phase 1),
/// `FastCdcChunker` (Phase 2). Content-defined chunking (CDC) gives stable
/// boundaries across similar blobs (insertions only change local chunks),
/// enabling cross-blob dedup. Fixed-size chunking is simpler and faster for
/// known-shape blobs (`SenseModule` pod, `LatentPatch`, small WASM modules).
///
/// **Zero-copy contract:** the returned slices borrow from `bytes`; no
/// allocation happens in `chunk()`. Callers that need owned chunks (e.g. to
/// insert into a store) call `.to_vec()` per slice.
pub trait ChunkingStrategy: Send + Sync {
    /// Split `bytes` into non-overlapping chunks. Final chunk may be shorter.
    ///
    /// For empty input, returns an empty `Vec` (zero chunks), matching the
    /// empty-blob contract in Plan 272 T1.10 `test_empty_blob`.
    fn chunk<'a>(&self, bytes: &'a [u8]) -> Vec<&'a [u8]>;
}

/// Strategy for fetching chunks not present in the local store.
///
/// Used by `ChunkedContentStore::get` when hydrating a blob whose chunks may
/// live on a remote backend (filesystem, S3, IPFS, riir-chain Cold tier, a Lore
/// server — the deploy decides). Implementations land in Phase 3
/// (`fetcher.rs`): `InMemoryChunkFetcher`, `FsChunkFetcher`,
/// `NetChunkFetcher`, `TieredChunkFetcher`.
///
/// `fetch_range` is for partial / sparse hydration: the caller may know only
/// the range they need (e.g. LOD-0 of a multi-LOD asset) and want to skip
/// downloading the rest.
pub trait ChunkFetcher: Send + Sync {
    /// Fetch a single chunk by its BLAKE3 hash. Returns the chunk bytes, or
    /// `None` if not available from this fetcher (caller may try a fallback).
    fn fetch(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>>;

    /// Fetch a byte range within a blob — for sparse / partial hydration.
    ///
    /// Default returns `None` (range fetch is opt-in: not every backend
    /// supports it efficiently). Implementations that do (e.g. HTTP Range
    /// requests, mmap'd FS) override.
    fn fetch_range(&self, _blob_id: &BlobId, _range: ChunkRange) -> Option<Vec<u8>> {
        None
    }
}

/// A content-addressed chunk store with Merkle dedup (Research 262 §2.1).
///
/// Generic over the chunk-fetching strategy (filesystem, network, in-memory).
/// No game semantics, no chain, no consensus — pure data structure.
///
/// Inspired by Epic Games Lore's chunked storage model, distilled to the
/// modelless primitive: chunk → BLAKE3 → dedup → Merkle root → inclusion proof.
pub trait ChunkedContentStore: Send + Sync {
    /// Put a blob into the store. Returns the content-addressed [`BlobId`]
    /// (Merkle root over the blob's chunk hashes).
    ///
    /// **Idempotent:** putting the same bytes always returns the same `BlobId`.
    /// **Dedup:** chunks already in the store are not re-stored.
    /// **Empty input:** returns `BlobId::zero()` (Merkle root of zero leaves).
    fn put(&self, bytes: &[u8]) -> BlobId;

    /// Get a blob by its `BlobId`. Returns `None` if the blob is not indexed
    /// locally. Hydrates by concatenating chunks in chunk-index order.
    ///
    /// Note: this is the *materializing* path — it allocates a `Vec` of the
    /// full logical byte count. For sparse reads, use [`Self::get_chunk`] or
    /// the (Phase 3) `ChunkFetcher::fetch_range` path.
    fn get(&self, blob_id: &BlobId) -> Option<Vec<u8>>;

    /// Get a single chunk by its BLAKE3 hash — **zero-allocation hot path**.
    ///
    /// Returns a borrowed slice into the store's chunk buffer. Per AGENTS.md
    /// ("zero-allocation hot path"), the implementation MUST NOT allocate in
    /// the body (no `Vec`, `String`, `Box`, `format!`, or `to_vec()`). For
    /// `InMemoryChunkedStore`, the borrow is sound because chunks are
    /// append-only and stored as leaked `&'static [u8]` (reclaimed on Drop).
    ///
    /// Returns `None` if the chunk is not in the local store.
    fn get_chunk(&self, chunk_hash: &[u8; 32]) -> Option<&[u8]>;

    /// Prove that a chunk (by leaf index in the blob's chunk array) is part
    /// of `blob_id`. Returns `None` if the blob is unknown or the index is
    /// out of range.
    ///
    /// O(log n) siblings, constructed via
    /// [`super::merkle::build_binary_merkle_proof`].
    fn prove_chunk(&self, blob_id: &BlobId, leaf_index: usize) -> Option<MerkleProof>;

    /// **Associated function** (G4 light-client gate): verify a Merkle
    /// inclusion proof against a known leaf hash.
    ///
    /// Pure BLAKE3 — **no `&self` access**. A light client (browser, curator,
    /// anti-cheat) can call this with only the proof and the chunk it already
    /// has, without downloading the blob or holding a store reference.
    ///
    /// Delegates to [`super::merkle::verify_binary_merkle_proof`].
    fn verify_proof(proof: &MerkleProof, leaf_hash: &[u8; 32]) -> bool;

    /// Verify that `bytes` hash to `blob_id` — pure computation, no storage.
    ///
    /// Computes the chunk-Merkle root of `bytes` (chunk → BLAKE3 → binary
    /// Merkle tree → root) and compares it to `blob_id`. This is the
    /// verification-only counterpart to [`Self::put`]: it does NOT store
    /// chunks, does NOT insert blob metadata, and does NOT bump stats.
    ///
    /// Use this instead of `put(bytes) == *blob_id` when you only need to
    /// verify (e.g. defense-in-depth checks on store-hit paths where the
    /// bytes were already retrieved via [`Self::get`]). The default
    /// implementation falls back to `put` + comparison; stores that can hash
    /// without storing should override for a cheaper path.
    fn verify_blob(&self, blob_id: &BlobId, bytes: &[u8]) -> bool {
        &self.put(bytes) == blob_id
    }

    /// Aggregate stats: chunks stored, blobs indexed, dedup ratio, total bytes.
    fn stats(&self) -> StoreStats;
}
