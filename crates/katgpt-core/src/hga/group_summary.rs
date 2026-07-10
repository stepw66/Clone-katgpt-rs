//! `GroupSummaryCache` — sub-chunk group middle routing tier.
//!
//! Plan 397 T1.5 + T1.6, Research 379 §1.2(a).
//!
//! Stores per-chunk per-group summary vectors and provides group-level scoring
//! for stage-2 routing (within chunks selected by stage-1 chunk entmax).

use super::summary::{MixedRopeSummarizer, dot_score};

/// Group-level summary cache — `[n_chunks * n_groups_per_chunk * D]` flattened.
///
/// Const generics:
/// - `D` — head dimension.
///
/// Runtime parameters:
/// - `chunk_size` (C) and `group_size` (GS) are runtime (must divide).
/// - `n_groups_per_chunk = C / GS`.
///
/// The cache is append-only during decode. Each `append_chunk` computes
/// `C/GS` group summaries using the `MixedRopeSummarizer` and appends them.
pub struct GroupSummaryCache {
    head_dim: usize,
    chunk_size: usize,
    group_size: usize,
    n_groups_per_chunk: usize,
    /// Flattened `[n_chunks * n_groups_per_chunk * D]` summary store.
    summaries: Vec<f32>,
    /// Number of chunks stored so far.
    n_chunks: usize,
    /// The summarizer (owns the RoPE freq + high/low mask).
    summarizer: MixedRopeSummarizer,
}

/// A scored group — output of `score_groups`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupScore {
    pub chunk_idx: usize,
    pub group_idx: usize,
    pub score: f32,
}

impl GroupSummaryCache {
    /// Create a new group summary cache.
    ///
    /// - `head_dim` — D.
    /// - `chunk_size` — C (must divide evenly by group_size).
    /// - `group_size` — GS.
    /// - `summarizer` — the mixed-RoPE summarizer (owns RoPE freqs).
    pub fn new(
        head_dim: usize,
        chunk_size: usize,
        group_size: usize,
        summarizer: MixedRopeSummarizer,
    ) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        assert!(
            chunk_size.is_multiple_of(group_size),
            "chunk_size ({chunk_size}) must be divisible by group_size ({group_size})"
        );
        let n_groups_per_chunk = chunk_size / group_size;
        Self {
            head_dim,
            chunk_size,
            group_size,
            n_groups_per_chunk,
            summaries: Vec::new(),
            n_chunks: 0,
            summarizer,
        }
    }

    /// Number of chunks stored.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        self.n_chunks
    }

    /// Number of groups per chunk.
    #[inline]
    pub fn n_groups_per_chunk(&self) -> usize {
        self.n_groups_per_chunk
    }

    /// Total number of group summaries stored.
    pub fn n_groups(&self) -> usize {
        self.n_chunks * self.n_groups_per_chunk
    }

    /// Reference to the summarizer.
    pub fn summarizer(&self) -> &MixedRopeSummarizer {
        &self.summarizer
    }

    /// Append a chunk's worth of group summaries.
    ///
    /// - `keys_flat` — `[C * D]` flattened key vectors for the chunk.
    /// - `positions` — `[C]` token positions.
    ///
    /// Computes `n_groups_per_chunk` summaries and appends to the store.
    pub fn append_chunk(&mut self, keys_flat: &[f32], positions: &[usize]) {
        debug_assert_eq!(keys_flat.len(), self.chunk_size * self.head_dim);
        debug_assert_eq!(positions.len(), self.chunk_size);

        // Reserve capacity for the new summaries.
        let needed = self.n_groups_per_chunk * self.head_dim;
        self.summaries.reserve(needed);

        for g in 0..self.n_groups_per_chunk {
            let group_start = g * self.group_size;
            let summary =
                self.summarizer
                    .summarize(keys_flat, positions, group_start, self.group_size);
            debug_assert_eq!(summary.len(), self.head_dim);
            self.summaries.extend_from_slice(&summary);
        }
        self.n_chunks += 1;
    }

    /// Get the summary for a specific `(chunk, group)` pair: `[D]`.
    pub fn summary(&self, chunk_idx: usize, group_idx: usize) -> &[f32] {
        let offset = (chunk_idx * self.n_groups_per_chunk + group_idx) * self.head_dim;
        &self.summaries[offset..offset + self.head_dim]
    }

    /// Score groups within selected chunks against a query.
    ///
    /// - `query` — `[D]` query vector.
    /// - `selected_chunks` — chunks to score (output of stage-1 chunk routing).
    ///
    /// Returns `Vec<GroupScore>` sorted by score descending. The caller takes
    /// the top `k_g` to form the stage-2 group selection.
    ///
    /// Uses dot-product scoring (NOT entmax). Entmax is used only at the
    /// chunk-selection level in `forward_hga`; group selection is dot-product
    /// + top-K (deterministic, no normalization needed for ranking).
    pub fn score_groups(&self, query: &[f32], selected_chunks: &[usize]) -> Vec<GroupScore> {
        let mut scores = Vec::with_capacity(selected_chunks.len() * self.n_groups_per_chunk);
        for &chunk_idx in selected_chunks {
            if chunk_idx >= self.n_chunks {
                continue;
            }
            for g in 0..self.n_groups_per_chunk {
                let summary = self.summary(chunk_idx, g);
                let score = dot_score(query, summary);
                scores.push(GroupScore {
                    chunk_idx,
                    group_idx: g,
                    score,
                });
            }
        }
        // Sort by score descending.
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores
    }

    /// Select the top-K groups across selected chunks.
    ///
    /// Returns a `GroupSelection` suitable for `TieredKvStore::fetch_working_set`.
    pub fn select_top_k_groups(
        &self,
        query: &[f32],
        selected_chunks: &[usize],
        k_g: usize,
    ) -> crate::tiered_kv::GroupSelection {
        let scores = self.score_groups(query, selected_chunks);
        let top_k: Vec<&GroupScore> = scores.iter().take(k_g).collect();

        // Group by chunk to build contiguous ranges.
        let mut by_chunk: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for s in top_k {
            by_chunk.entry(s.chunk_idx).or_default().push(s.group_idx);
        }

        let mut selection = crate::tiered_kv::GroupSelection::empty();
        for (chunk_idx, mut groups) in by_chunk {
            groups.sort_unstable();
            // Coalesce contiguous groups into ranges.
            let mut range_start = groups[0];
            let mut range_end = groups[0];
            for &g in &groups[1..] {
                if g == range_end + 1 {
                    range_end = g;
                } else {
                    selection.add_range(chunk_idx, range_start, range_end - range_start + 1);
                    range_start = g;
                    range_end = g;
                }
            }
            selection.add_range(chunk_idx, range_start, range_end - range_start + 1);
        }
        selection
    }
}
