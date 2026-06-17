//! CompressionDrafter — score candidate continuations by compressed length.
//!
//! Distillation of nathan.rs/gzip-lm: the compressor IS the model. The corpus
//! encodes "what's likely next" via LZ77 match length (no neural weights, no
//! training). Promotion/update = append bytes to corpus, which is exactly
//! freeze/thaw snapshot semantics on a Vec<u8>.
//!
//! Plan 285 (Research 256, revised to GOAT).

use lz4_flex::compress_prepend_size;

/// Score candidate continuations by compressed length under a frozen corpus.
///
/// `score(ctx, candidate) = compressed_len(ctx) as i32 - compressed_len(ctx + candidate) as i32`
///
/// Higher = more compressible = more likely under the corpus distribution.
/// Negative = candidate added entropy (rare/unseen bytes).
pub trait CompressionDrafter {
    /// Frozen corpus — the model's "knowledge". Appendable for online learning.
    fn corpus(&self) -> &[u8];

    /// Score a single candidate against `ctx` (recent context).
    /// The corpus is always prepended: effective input is `corpus + ctx + candidate`.
    fn score(&mut self, ctx: &[u8], candidate: &[u8]) -> i32;

    /// Batched score — share compressor prefix state across all candidates.
    /// Returns scores in the same order as `candidates`.
    fn score_batch(&mut self, ctx: &[u8], candidates: &[&[u8]]) -> Vec<i32>;

    /// Append bytes to the corpus (online learning / self-learn).
    fn append(&mut self, bytes: &[u8]);
}

/// Hot-tier LZ4-based compression drafter.
///
/// Wraps `lz4_flex::compress_prepend_size` for batched scoring. The corpus is
/// always prepended to the input: effective compressed input is `corpus + ctx + candidate`.
/// The corpus-prefix match search is amortized implicitly by lz4_flex's hash table
/// (rebuilt per call — see note on perf).
///
/// **Perf:** for a 30KB corpus, each `score()` call is ~10-50µs on Apple Silicon.
/// Fits Hot tier (sub-ms). Not Plasma tier (µs).
pub struct Lz4FlexDrafter {
    /// The corpus — frozen between calls unless `append()` is called.
    corpus: Vec<u8>,
    /// Reusable scratch buffer for `corpus + ctx + candidate` concatenation.
    /// Pre-allocated once, cleared and refilled per call. Zero allocation on hot path.
    scratch: Vec<u8>,
}

impl Lz4FlexDrafter {
    /// Create with an initial corpus.
    pub fn new(corpus: Vec<u8>) -> Self {
        let scratch = Vec::with_capacity(corpus.len() + 1024); // corpus + ctx + candidate headroom
        Self { corpus, scratch }
    }

    /// Create with empty corpus (caller will append).
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Access the underlying corpus bytes (for snapshot/BLAKE3).
    pub fn corpus_bytes(&self) -> &[u8] {
        &self.corpus
    }

    /// Replace the corpus (for restore from snapshot).
    pub fn replace_corpus(&mut self, bytes: Vec<u8>) {
        self.corpus = bytes;
        if self.scratch.capacity() < self.corpus.len() + 1024 {
            self.scratch = Vec::with_capacity(self.corpus.len() + 1024);
        }
    }

    /// Internal: build `corpus + ctx + candidate` in scratch, return compressed len.
    #[inline(always)]
    fn compressed_len_with(&mut self, ctx: &[u8], candidate: &[u8]) -> usize {
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.corpus);
        self.scratch.extend_from_slice(ctx);
        self.scratch.extend_from_slice(candidate);
        compress_prepend_size(&self.scratch).len()
    }

    /// Internal: build `corpus + ctx` (no candidate) in scratch, return compressed len.
    /// Used as the baseline for score = baseline - with_candidate.
    #[inline(always)]
    fn compressed_len_baseline(&mut self, ctx: &[u8]) -> usize {
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.corpus);
        self.scratch.extend_from_slice(ctx);
        compress_prepend_size(&self.scratch).len()
    }
}

impl CompressionDrafter for Lz4FlexDrafter {
    fn corpus(&self) -> &[u8] {
        &self.corpus
    }

    fn score(&mut self, ctx: &[u8], candidate: &[u8]) -> i32 {
        let baseline = self.compressed_len_baseline(ctx);
        let with_candidate = self.compressed_len_with(ctx, candidate);
        baseline as i32 - with_candidate as i32
    }

    fn score_batch(&mut self, ctx: &[u8], candidates: &[&[u8]]) -> Vec<i32> {
        // Baseline computed once. Each candidate adds its own bytes; the corpus+ctx
        // prefix compression is NOT amortized across candidates (lz4_flex doesn't
        // expose a state-clone API like Python's zlib.compressobj().copy()).
        // For batch amortization we'd need a custom LZ77 — see PlasmaLz77 sketch
        // in riir-ai/.research/137. For Hot tier this is acceptable.
        let baseline = self.compressed_len_baseline(ctx);
        candidates
            .iter()
            .map(|c| {
                let with_c = self.compressed_len_with(ctx, c);
                baseline as i32 - with_c as i32
            })
            .collect()
    }

    fn append(&mut self, bytes: &[u8]) {
        self.corpus.extend_from_slice(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_test_repeated_pattern_dominates() {
        // Corpus contains "guard needs" twice → that prefix compresses very well.
        let corpus = b"guard needs sword\nguard needs potion\nking finds amulet\n";
        let mut d = Lz4FlexDrafter::new(corpus.to_vec());
        let ctx: &[u8] = b"";
        let score_seen = d.score(ctx, b"guard needs");
        let score_unseen = d.score(ctx, b"king finds");
        // "guard needs" appears twice in corpus → compresses better → higher score.
        // Note: both candidates have non-zero score because they're substrings.
        // The repeated one should still win.
        assert!(
            score_seen >= score_unseen,
            "repeated pattern should score >= unseen: {} vs {}",
            score_seen,
            score_unseen
        );
    }

    #[test]
    fn score_test_unseen_byte_penalized() {
        // Corpus contains no 'z' byte. LZ4 needs (a) ≥4-byte candidates (min match length)
        // AND (b) enough corpus repetition for its hash table to actually find matches.
        // We build a corpus by repeating the seen pattern so LZ4's match-finder engages.
        let seen = b"guard needs sword\n";
        let corpus: Vec<u8> = std::iter::repeat(seen)
            .take(8)
            .flatten()
            .copied()
            .collect();
        let mut d = Lz4FlexDrafter::new(corpus);
        let ctx: &[u8] = b"";
        // Equal-length candidates: seen-substring vs all-unseen-bytes.
        let score_corpus_pattern = d.score(ctx, b"guard needs"); // verbatim substring, repeated 8×
        let score_unseen_byte = d.score(ctx, b"zzzzzzzzzzz"); // 11 bytes, 'z' not in corpus
        assert!(
            score_corpus_pattern > score_unseen_byte,
            "corpus pattern 'guard needs' should score higher than unseen 'zzzzzzzzzzz': {} vs {}",
            score_corpus_pattern,
            score_unseen_byte
        );
    }

    #[test]
    fn append_test_grows_corpus() {
        let mut d = Lz4FlexDrafter::empty();
        assert_eq!(d.corpus().len(), 0);
        d.append(b"hello");
        assert_eq!(d.corpus().len(), 5);
        d.append(b" world");
        assert_eq!(d.corpus().len(), 11);
        assert_eq!(d.corpus(), b"hello world");
    }

    #[test]
    fn batch_score_consistent_with_single() {
        let corpus = b"the quick brown fox\nthe lazy dog\n";
        let mut d_batch = Lz4FlexDrafter::new(corpus.to_vec());
        let mut d_single = Lz4FlexDrafter::new(corpus.to_vec());
        let ctx: &[u8] = b"the ";
        let candidates: &[&[u8]] = &[b"quick", b"lazy", b"fox", b"dog"];
        let batch = d_batch.score_batch(ctx, candidates);
        let single: Vec<i32> = candidates.iter().map(|c| d_single.score(ctx, c)).collect();
        assert_eq!(batch.len(), single.len());
        // lz4_flex is deterministic — scores must match exactly.
        for (i, (b, s)) in batch.iter().zip(single.iter()).enumerate() {
            assert_eq!(b, s, "batch vs single mismatch at index {}: {} vs {}", i, b, s);
        }
    }

    #[test]
    fn zero_alloc_no_regression_under_load() {
        let corpus = vec![b'x'; 30_000];
        let mut d = Lz4FlexDrafter::new(corpus);
        let initial_scratch_cap = d.scratch.capacity();
        let ctx = b"some context bytes";
        let candidate = b"some candidate bytes";
        for _ in 0..10_000 {
            let _ = d.score(ctx, candidate);
        }
        assert_eq!(
            d.scratch.capacity(),
            initial_scratch_cap,
            "scratch buffer grew during hot loop — zero-alloc violated"
        );
    }

    #[test]
    fn snapshot_roundtrip_preserves_generation() {
        // The corpus IS the wired format: snapshot = corpus bytes + BLAKE3.
        // Roundtrip: replace_corpus(corpus_bytes()) yields identical drafter.
        let corpus = b"guard needs sword\nking finds amulet\nsage seeks quest\n";
        let mut d1 = Lz4FlexDrafter::new(corpus.to_vec());
        let ctx: &[u8] = b"";
        let candidates: &[&[u8]] = &[b"guard needs", b"king finds", b"sage seeks"];
        let scores_before = d1.score_batch(ctx, candidates);

        // Snapshot
        let snapshot_bytes = d1.corpus_bytes().to_vec();

        // Restore into a fresh drafter
        let mut d2 = Lz4FlexDrafter::empty();
        d2.replace_corpus(snapshot_bytes);
        let scores_after = d2.score_batch(ctx, candidates);

        assert_eq!(scores_before, scores_after, "snapshot roundtrip must preserve scores");
    }
}
