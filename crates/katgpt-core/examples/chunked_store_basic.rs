//! Example: ChunkedContentStore basic usage (Plan 272 T1.11).
//!
//! Demonstrates:
//! - Constructing two synthetic blobs sharing 50% of chunks (sword_base +
//!   sword_variant with a mutated handle).
//! - Putting both into [`InMemoryChunkedStore`] with `FixedSizeChunker`
//!   (`chunk_size = 32` so dedup is visible at small scale).
//! - Printing `BlobId`s, `StoreStats::dedup_ratio`, and an inclusion proof
//!   for chunk 0 of `sword_variant`.
//! - Light-client verification: `verify_proof` succeeds without store access
//!   (called as an associated function — G4 gate).
//!
//! Run with:
//! ```sh
//! cargo run --example chunked_store_basic --features chunked_content_store --release
//! ```

use katgpt_core::{BlobId, ChunkedContentStore, FixedSizeChunker, InMemoryChunkedStore};

/// Hex-encode the first 8 bytes of a BlobId for compact printing.
fn short_hex(id: &BlobId) -> String {
    let bytes = &id.as_bytes()[..8];
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("=== ChunkedContentStore Basic Example (Plan 272) ===\n");

    // Small chunk size so dedup is visible at small blob scale.
    let chunk_size = 32;
    let store = InMemoryChunkedStore::with_chunker(FixedSizeChunker::new(chunk_size));

    // --- Synthetic blob construction --------------------------------------
    //
    // Simulate two sword assets that share geometry but differ in the handle.
    // 64 bytes each = 2 chunks of 32 bytes per blob.
    //
    //   sword_base    = [blade_chunk (32B)] [handle_v1_chunk (32B)]
    //   sword_variant = [blade_chunk (32B)] [handle_v2_chunk (32B)]
    //
    // The blade chunk is shared → 1 dedup hit, 3 unique chunks total.
    let mut blade = [0u8; 32];
    for (i, b) in blade.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7);
    }
    let mut handle_v1 = [0u8; 32];
    for (i, b) in handle_v1.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(11);
    }
    let mut handle_v2 = [0u8; 32];
    for (i, b) in handle_v2.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(13);
    }

    let mut sword_base = Vec::with_capacity(64);
    sword_base.extend_from_slice(&blade);
    sword_base.extend_from_slice(&handle_v1);

    let mut sword_variant = Vec::with_capacity(64);
    sword_variant.extend_from_slice(&blade);
    sword_variant.extend_from_slice(&handle_v2);

    println!(
        "Blob layout: chunk_size = {} bytes; each sword = 2 chunks = 64 bytes",
        chunk_size
    );
    println!(
        "sword_base    = [blade | handle_v1]   ({} bytes)",
        sword_base.len()
    );
    println!(
        "sword_variant = [blade | handle_v2]   ({} bytes)\n",
        sword_variant.len()
    );

    // --- Put both blobs ----------------------------------------------------
    let id_base = store.put(&sword_base);
    let id_variant = store.put(&sword_variant);

    println!("Put sword_base    → BlobId {}", short_hex(&id_base));
    println!(
        "Put sword_variant → BlobId {}{}\n",
        short_hex(&id_variant),
        if id_base == id_variant {
            "  [IDENTICAL — unexpected]"
        } else {
            "  [different — expected, handles differ]"
        }
    );

    // --- Stats -------------------------------------------------------------
    let stats = store.stats();
    println!("StoreStats:");
    println!("  n_chunks_stored     = {}", stats.n_chunks_stored);
    println!("  n_blobs_indexed     = {}", stats.n_blobs_indexed);
    println!("  total_bytes_stored  = {}", stats.total_bytes_stored);
    println!("  total_bytes_logical = {}", stats.total_bytes_logical);
    println!(
        "  dedup_ratio         = {:.3}  (logical / stored; 1.0 = no dedup)\n",
        stats.dedup_ratio
    );

    // Expected: 3 unique chunks (blade, handle_v1, handle_v2) for 4 logical
    // chunk slots (2 per blob × 2 blobs). dedup_ratio = 128 / 96 ≈ 1.333.
    assert_eq!(
        stats.n_chunks_stored, 3,
        "expected 3 unique chunks (1 shared + 2 unique handles)"
    );
    assert_eq!(stats.n_blobs_indexed, 2);
    assert!(
        (stats.dedup_ratio - (128.0 / 96.0)).abs() < 1e-3,
        "expected dedup_ratio ≈ 1.333"
    );

    // --- Inclusion proof for chunk 0 (the shared blade) ------------------
    println!("--- Inclusion proof: chunk 0 of sword_variant (the blade) ---");
    let proof = store
        .prove_chunk(&id_variant, 0)
        .expect("leaf 0 must be provable");
    println!(
        "Proof: leaf_index = {}, siblings.len() = {}, expected_root = {}",
        proof.leaf_index,
        proof.siblings.len(),
        short_hex(&BlobId(proof.expected_root))
    );

    // The leaf hash for chunk 0 is BLAKE3(blade_chunk).
    let leaf_0_hash: [u8; 32] = blake3::hash(&blade).into();

    // --- Light-client verify (G4) -----------------------------------------
    //
    // verify_proof is an ASSOCIATED function — no `&self`. A light client
    // (browser, curator, anti-cheat) can verify the chunk is part of the
    // blob with ONLY the proof + the chunk it already has. No store access.
    let verified = InMemoryChunkedStore::verify_proof(&proof, &leaf_0_hash);
    println!(
        "Light-client verify_proof(blade_hash, proof) = {}  [G4: no &self]\n",
        verified
    );
    assert!(verified, "verify_proof must succeed for the correct leaf");

    // --- Negative control: wrong leaf hash must fail ----------------------
    let wrong_leaf_hash: [u8; 32] = blake3::hash(&handle_v2).into();
    let rejected = InMemoryChunkedStore::verify_proof(&proof, &wrong_leaf_hash);
    println!(
        "Negative control: verify_proof(handle_v2_hash, proof_for_leaf_0) = {}  [must be false]",
        rejected
    );
    assert!(!rejected, "wrong leaf hash must NOT verify");

    println!("\n=== All assertions passed. Plan 272 Phase 1 reference impl is operational. ===");
}
