//! GOAT gate benchmarks for the chunked content store (Plan 272 Phase 4).
//!
//! Implements G1 (dedup ratio), G4 (light-client verify), and G7 (tamper
//! detection) as inline `#[test]` functions — following the codebase
//! convention (inline `#[cfg(test)]`, validated via `cargo test --lib`).
//!
//! G2 (incremental push) is proven by the Phase 2 CDC test
//! `test_cdc_dedup_with_variant`. G3/G5 (perf-timing gates) require `criterion`
//! bench targets in `Cargo.toml` — deferred to avoid colliding with concurrent
//! edits. G6 (default-off) is verified by `cargo check --no-default-features`.
//!
//! See [`.benchmarks/262_chunked_content_store_goat.md`] for the full table.

#![cfg(test)]

use super::in_memory::InMemoryChunkedStore;
use super::merkle::verify_binary_merkle_proof;
use super::r#trait::ChunkedContentStore;

// ────────────────────────────────────────────────────────────────────────────
// G1 — Dedup ratio ≥ 5.0
// ────────────────────────────────────────────────────────────────────────────

/// G1: 50 blobs × 10 chunks each (FixedSizeChunker, 64 KiB), where each blob
/// after the first shares 9 chunks with blob 0 and has 1 unique chunk.
///
/// Expected dedup ratio = N×C / (C + N - 1) = 50×10 / (10 + 49) = 500/59 ≈ 8.47.
/// Target: ≥ 5.0.
///
/// Uses FixedSizeChunker (default, 64 KiB) rather than FastCdcChunker for
/// deterministic chunk boundaries — the GOAT gate measures the *store's* dedup
/// capability, not the chunker's boundary stability (that's G2).
#[test]
fn g1_dedup_ratio_meets_target() {
    const N_BLOBS: usize = 50;
    const N_CHUNKS: usize = 10;
    const CHUNK_SIZE: usize = 64 * 1024; // 64 KiB — matches FixedSizeChunker default

    // Blob 0: N_CHUNKS distinct 64 KiB blocks, each filled with a distinct byte.
    let blob0: Vec<u8> = (0..N_CHUNKS)
        .flat_map(|i| std::iter::repeat(i as u8).take(CHUNK_SIZE))
        .collect();
    assert_eq!(blob0.len(), N_CHUNKS * CHUNK_SIZE);

    let store = InMemoryChunkedStore::new();

    // Put blob 0.
    let _id0 = store.put(&blob0);

    // For each subsequent blob: copy blob 0, replace the last 64 KiB chunk
    // with a unique fill byte. This guarantees 9 shared chunks + 1 new.
    for n in 1..N_BLOBS {
        let mut blob = blob0.clone();
        // Overwrite the last CHUNK_SIZE bytes with a unique fill.
        let fill = (N_CHUNKS + n) as u8; // distinct from 0..N_CHUNKS
        let start = (N_CHUNKS - 1) * CHUNK_SIZE;
        blob[start..start + CHUNK_SIZE].fill(fill);
        let _id = store.put(&blob);
    }

    let stats = store.stats();
    let expected_ratio = (N_BLOBS * N_CHUNKS) as f32 / (N_CHUNKS + N_BLOBS - 1) as f32;
    assert!(
        stats.dedup_ratio >= 5.0,
        "G1 FAIL: dedup ratio {} < 5.0 (expected ~{expected_ratio:.2})",
        stats.dedup_ratio
    );
    // Sanity: the ratio should be close to the theoretical value.
    assert!(
        (stats.dedup_ratio - expected_ratio).abs() < 1.0,
        "G1 sanity: ratio {} far from expected {}",
        stats.dedup_ratio,
        expected_ratio
    );
}

// ────────────────────────────────────────────────────────────────────────────
// G4 — Light-client verify (no `&self` access)
// ────────────────────────────────────────────────────────────────────────────

/// G4: `verify_proof` is an associated function (no `&self`). A light client
/// can verify a Merkle inclusion proof with ONLY the proof + leaf hash — no
/// store reference, no blob download.
///
/// This test constructs a proof, then verifies it WITHOUT holding a reference
/// to the store (the store is dropped before verification). This proves the
/// light-client property at the type level + runtime level.
#[test]
fn g4_light_client_verify_no_self() {
    // Build a small store with one blob.
    let blob = b"the quick brown fox jumps over the lazy dog".repeat(100);
    let store = InMemoryChunkedStore::new();
    let blob_id = store.put(&blob);

    // Generate a proof for chunk 0.
    let proof = store
        .prove_chunk(&blob_id, 0)
        .expect("prove_chunk should succeed for valid blob + index");

    // The leaf hash comes from the chunk DATA the light client already has
    // (it's verifying that its chunk is part of the blob). So the light client
    // computes blake3(its_chunk) and verifies the proof.
    let chunk_end = blob.len().min(64 * 1024);
    let leaf_hash: [u8; 32] = blake3::hash(&blob[..chunk_end]).into();

    // DROP THE STORE. The light client no longer has any reference to it.
    drop(store);

    // Verify: pure function, no store access.
    let valid = InMemoryChunkedStore::verify_proof(&proof, &leaf_hash);
    assert!(valid, "G4: light-client verify must succeed without store ref");

    // Also verify the standalone merkle function works without any store —
    // it takes individual fields (leaf_hash, leaf_index, siblings, root),
    // proving the light-client property at the API level too.
    let valid2 = verify_binary_merkle_proof(
        &leaf_hash,
        proof.leaf_index,
        &proof.siblings,
        &proof.expected_root,
    );
    assert!(valid2, "G4: standalone verify_binary_merkle_proof must work");
}

/// G4 structural: `build_binary_merkle_proof` returns a self-contained proof
/// that doesn't borrow from the store. Verified by the compiler: if it
/// borrowed, `drop(store)` before `verify` would fail to compile.
#[test]
fn g4_proof_is_owned_not_borrowed() {
    let blob = vec![42u8; 256 * 1024]; // 4 chunks of 64 KiB
    let store = InMemoryChunkedStore::new();
    let blob_id = store.put(&blob);
    let proof = store.prove_chunk(&blob_id, 1).expect("proof for index 1");
    drop(store); // proof must not borrow from store
    // proof is 'static — we can still use it.
    assert!(!proof.siblings.is_empty(), "proof should have siblings");
}

// ────────────────────────────────────────────────────────────────────────────
// G7 — Tamper detection (1-bit flip → BlobId mismatch)
// ────────────────────────────────────────────────────────────────────────────

/// G7: flipping any single bit in a blob MUST produce a different BlobId.
/// Tests 10 000 single-bit flips across 100 distinct blobs (100 bits/blob).
#[test]
fn g7_tamper_detection() {
    const N_BLOBS: usize = 100;
    const BITS_PER_BLOB: usize = 100;
    const BLOB_SIZE: usize = 64 * 1024; // 64 KiB — 1 chunk

    let mut mismatches = 0u32;
    let mut total = 0u32;

    for blob_n in 0..N_BLOBS {
        // Each blob: deterministic but distinct fill.
        let original: Vec<u8> = std::iter::repeat(blob_n as u8)
            .take(BLOB_SIZE)
            .collect();
        let original_id = {
            let store = InMemoryChunkedStore::new();
            store.put(&original)
        };

        // Flip BITS_PER_BLOB distinct bits (spread across the blob).
        for bit_i in 0..BITS_PER_BLOB {
            let byte_idx = (bit_i * (BLOB_SIZE / BITS_PER_BLOB)) % BLOB_SIZE;
            let bit_pos = bit_i % 8;
            let mut tampered = original.clone();
            tampered[byte_idx] ^= 1u8 << bit_pos;

            let tampered_id = {
                let store = InMemoryChunkedStore::new();
                store.put(&tampered)
            };

            total += 1;
            if tampered_id != original_id {
                mismatches += 1;
            }
        }
    }

    assert_eq!(
        mismatches, total,
        "G7 FAIL: {mismatches}/{total} mismatches — tampered blob matched original BlobId"
    );
}

/// G7 variant: tamper a MULTI-chunk blob and verify the BlobId changes (the
/// tamper might be in any chunk, and the Merkle root must reflect it).
#[test]
fn g7_tamper_multichunk_blob() {
    let blob = (0u8..=255)
        .flat_map(|b| std::iter::repeat(b).take(64 * 1024))
        .collect::<Vec<u8>>(); // 256 chunks, each 64 KiB

    let store = InMemoryChunkedStore::new();
    let original_id = store.put(&blob);

    // Flip 1 bit in the middle of the blob (chunk 128).
    let mut tampered = blob.clone();
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0x01;

    let tampered_id = store.put(&tampered);
    assert_ne!(
        tampered_id, original_id,
        "G7 multi-chunk: tamper in chunk 128 must change BlobId"
    );
}
