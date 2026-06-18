//! Fixed-size chunker (Plan 272 T1.6).
//!
//! Splits input bytes into non-overlapping fixed-size slices. Final slice may
//! be shorter than `chunk_size`. Zero-alloc: returns borrowed `&[u8]` views
//! into the input.
//!
//! `FastCdcChunker` (Rabin-Fingerprint CDC) lands in Phase 2 — required for G1
//! (5× dedup on similar large blobs) and G2 (≤5% incremental push on 1-byte
//! change). Fixed-size chunking is the simpler reference impl that proves G3
//! (proof cost), G4 (light-client), G6 (default-off regression), G7 (tamper).

use super::r#trait::ChunkingStrategy;

/// Default chunk size — 64 KiB (matches Plan 272 T1.6 + Research 262 §2.3).
///
/// Large enough that per-chunk overhead is amortized; small enough that
/// incremental change at byte N only rewrites the tail from N onwards
/// (for fixed-size chunking; CDC is better but Phase 2).
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Fixed-size chunker — splits bytes into non-overlapping slices of
/// `chunk_size` bytes (final slice may be shorter).
///
/// Default `chunk_size = `[`DEFAULT_CHUNK_SIZE`] (64 KiB).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FixedSizeChunker {
    /// Bytes per chunk. Must be > 0. Asserted in [`Self::new`].
    pub chunk_size: usize,
}

impl FixedSizeChunker {
    /// Construct with a custom chunk size. Panics if `chunk_size == 0`
    /// (zero-size chunking is undefined — would loop forever).
    #[inline]
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "FixedSizeChunker: chunk_size must be > 0");
        Self { chunk_size }
    }
}

impl Default for FixedSizeChunker {
    #[inline]
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }
}

impl ChunkingStrategy for FixedSizeChunker {
    #[inline]
    fn chunk<'a>(&self, bytes: &'a [u8]) -> Vec<&'a [u8]> {
        // Empty input → zero chunks (matches Plan 272 T1.10 `test_empty_blob`:
        // a zero-length blob has 0 chunks and its BlobId is BLAKE3(empty)).
        if bytes.is_empty() {
            return Vec::new();
        }
        // Pre-size the output Vec to avoid reallocation.
        let n = bytes.len().div_ceil(self.chunk_size);
        let mut out = Vec::with_capacity(n);
        let mut start = 0usize;
        while start < bytes.len() {
            // early-return pattern over `if start + chunk_size > len` (AGENTS.md:
            // prefer match over if; here a while loop with a single saturating
            // split_at is cleanest).
            let end = (start + self.chunk_size).min(bytes.len());
            out.push(&bytes[start..end]);
            start = end;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_chunk_size() {
        assert_eq!(DEFAULT_CHUNK_SIZE, 64 * 1024);
        let c = FixedSizeChunker::default();
        assert_eq!(c.chunk_size, DEFAULT_CHUNK_SIZE);
    }

    #[test]
    fn test_empty_input_returns_zero_chunks() {
        let c = FixedSizeChunker::default();
        let chunks = c.chunk(b"");
        assert!(chunks.is_empty(), "empty input must yield zero chunks");
    }

    #[test]
    fn test_single_byte_input() {
        let c = FixedSizeChunker::new(1024);
        let chunks = c.chunk(&[42u8]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], &[42u8]);
    }

    #[test]
    fn test_exact_multiple() {
        let c = FixedSizeChunker::new(4);
        let bytes = b"abcdefghijkl"; // 12 bytes, exactly 3 chunks of 4
        let chunks = c.chunk(bytes);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"abcd");
        assert_eq!(chunks[1], b"efgh");
        assert_eq!(chunks[2], b"ijkl");
        // Roundtrip: concatenation must equal input.
        let recon: Vec<u8> = chunks.concat();
        assert_eq!(recon.as_slice(), bytes);
    }

    #[test]
    fn test_partial_last_chunk() {
        let c = FixedSizeChunker::new(4);
        let bytes = b"abcdefghij"; // 10 bytes: 2 full + 1 partial of 2
        let chunks = c.chunk(bytes);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[1].len(), 4);
        assert_eq!(chunks[2].len(), 2);
        assert_eq!(chunks[2], b"ij");
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn test_zero_chunk_size_panics() {
        let _ = FixedSizeChunker::new(0);
    }

    #[test]
    fn test_with_capacity_no_reallocation() {
        // Indirect check: a chunker over 10 * chunk_size input should yield
        // exactly 10 chunks with no over-allocation in the Vec.
        let cs = 8;
        let c = FixedSizeChunker::new(cs);
        let bytes: Vec<u8> = (0..(10 * cs)).map(|i| i as u8).collect();
        let chunks = c.chunk(&bytes);
        assert_eq!(chunks.len(), 10);
        assert!(chunks.iter().all(|c| c.len() == cs));
        // capacity may be >= len but the pre-sizing should make it equal.
        assert_eq!(chunks.capacity(), 10);
    }
}
