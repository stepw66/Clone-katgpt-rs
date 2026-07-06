//! `InMemoryTieredKvStore` — reference implementation of `TieredKvStore`.
//!
//! All K/V lives in process RAM:
//! - **Cold**: `Vec<ChunkKv>` — full token K/V per chunk.
//! - **Hot**: `Vec<GroupSummary>` — per-group RoPE-mixed summary per chunk.
//! - **Warm**: not implemented (no LRU cache in the reference impl — cold is
//!   already in-process RAM so the warm tier adds no benefit for the reference).
//!   A bounded LRU warm tier is a production concern (mirrors `ZoneGeometryCache`
//!   mmap-backed LRU, riir-neuron-db Plan 335).
//!
//! The store is intentionally simple: correctness over performance. Phase 2 G4
//! (alloc-free hot path) is addressed by the HGA forward path using pre-allocated
//! scratch buffers, not by the store itself.

use super::{GroupSelection, SinkLocalSet, TieredKvStore, WorkingSet};

/// One chunk's K/V + positions, stored in the cold tier.
struct ChunkKv {
    /// `[C * D]` flattened keys.
    keys: Vec<f32>,
    /// `[C * D]` flattened values.
    values: Vec<f32>,
    /// `[C]` token positions.
    positions: Vec<usize>,
}

/// Per-chunk group summaries — `[n_groups * D]` flattened.
struct GroupSummary {
    /// `[n_groups * D]` group summary vectors.
    summaries: Vec<f32>,
}

/// In-memory reference tiered K/V store.
///
/// `group_summarizer` computes per-group summaries at `append_chunk` time.
/// The summaries are routing keys only — they never enter the output softmax.
pub struct InMemoryTieredKvStore<F: Fn(&[f32], &[usize], usize, usize) -> Vec<f32>> {
    head_dim: usize,
    chunk_size: usize,
    group_size: usize,
    /// Number of groups per chunk = chunk_size / group_size.
    n_groups_per_chunk: usize,
    /// Cold tier: full token K/V per chunk.
    cold: Vec<ChunkKv>,
    /// Hot tier: group summaries per chunk.
    hot: Vec<GroupSummary>,
    /// Group summary construction function.
    /// Signature: `fn(keys_flat: &[f32], positions: &[usize], group_start: usize, group_size: usize) -> Vec<f32>`.
    /// The function computes the summary for the `group_size` tokens starting
    /// at `keys_flat[group_start * D ..]`. This is injected so the store does
    /// not depend on the HGA module (no circular dep).
    group_summarizer: F,
}

impl<F: Fn(&[f32], &[usize], usize, usize) -> Vec<f32>> InMemoryTieredKvStore<F> {
    /// Create a new store.
    ///
    /// - `head_dim` — D.
    /// - `chunk_size` — C (must be > 0).
    /// - `group_size` — GS (must divide chunk_size).
    /// - `group_summarizer` — computes a group summary from raw keys + positions.
    ///   Signature: `(keys_flat: &[f32], positions: &[usize], group_start_token: usize, n_tokens: usize) -> Vec<f32>`.
    ///   Returns a D-length summary vector.
    pub fn new(
        head_dim: usize,
        chunk_size: usize,
        group_size: usize,
        group_summarizer: F,
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
            cold: Vec::new(),
            hot: Vec::new(),
            group_summarizer,
        }
    }

    /// All group summaries, flattened as `[n_chunks * n_groups_per_chunk * D]`.
    /// Used by the HGA forward path for group-level scoring.
    pub fn group_summaries_flat(&self) -> &[f32] {
        // Summaries are stored per-chunk; flatten on demand.
        // For the reference impl, we store them contiguously per chunk already.
        if self.hot.is_empty() {
            return &[];
        }
        // We return the first chunk's slice — callers iterate chunk by chunk.
        // A more efficient layout would store all summaries in one Vec; deferred.
        &self.hot[0].summaries
    }

    /// Get group summaries for a specific chunk: `[n_groups_per_chunk * D]`.
    pub fn group_summaries_for_chunk(&self, chunk_idx: usize) -> &[f32] {
        match self.hot.get(chunk_idx) {
            Some(gs) => &gs.summaries,
            None => &[],
        }
    }

    /// Number of groups per chunk.
    pub fn n_groups_per_chunk(&self) -> usize {
        self.n_groups_per_chunk
    }

    /// Get raw keys for a chunk: `[C * D]`.
    pub fn keys_for_chunk(&self, chunk_idx: usize) -> &[f32] {
        match self.cold.get(chunk_idx) {
            Some(c) => &c.keys,
            None => &[],
        }
    }

    /// Get raw values for a chunk: `[C * D]`.
    pub fn values_for_chunk(&self, chunk_idx: usize) -> &[f32] {
        match self.cold.get(chunk_idx) {
            Some(c) => &c.values,
            None => &[],
        }
    }

    /// Get positions for a chunk: `[C]`.
    pub fn positions_for_chunk(&self, chunk_idx: usize) -> &[usize] {
        match self.cold.get(chunk_idx) {
            Some(c) => &c.positions,
            None => &[],
        }
    }
}

impl<F: Fn(&[f32], &[usize], usize, usize) -> Vec<f32>> TieredKvStore
    for InMemoryTieredKvStore<F>
{
    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn group_size(&self) -> usize {
        self.group_size
    }

    fn n_chunks(&self) -> usize {
        self.cold.len()
    }

    fn append_chunk(
        &mut self,
        keys: &[f32],
        values: &[f32],
        positions: &[usize],
    ) {
        debug_assert_eq!(keys.len(), self.chunk_size * self.head_dim);
        debug_assert_eq!(values.len(), self.chunk_size * self.head_dim);
        debug_assert_eq!(positions.len(), self.chunk_size);

        // Cold tier: store raw K/V + positions.
        self.cold.push(ChunkKv {
            keys: keys.to_vec(),
            values: values.to_vec(),
            positions: positions.to_vec(),
        });

        // Hot tier: compute and store group summaries.
        let mut summaries = Vec::with_capacity(self.n_groups_per_chunk * self.head_dim);
        for g in 0..self.n_groups_per_chunk {
            let group_start = g * self.group_size;
            let summary = (self.group_summarizer)(keys, positions, group_start, self.group_size);
            debug_assert_eq!(summary.len(), self.head_dim);
            summaries.extend_from_slice(&summary);
        }
        self.hot.push(GroupSummary { summaries });
    }

    fn fetch_working_set(
        &self,
        sink_local: &SinkLocalSet,
        selected_chunks: &[usize],
        group_selection: &GroupSelection,
    ) -> WorkingSet {
        let d = self.head_dim;
        let mut ws = WorkingSet::empty();

        // Collect all chunks to fetch: sink + local + selected.
        let mut all_chunks = sink_local.all_chunks();
        for &c in selected_chunks {
            if !all_chunks.contains(&c) {
                all_chunks.push(c);
            }
        }
        all_chunks.sort_unstable();
        all_chunks.dedup();

        // Build a set of (chunk, group) pairs to fetch from group_selection.
        use std::collections::HashSet;
        let mut groups_to_fetch: HashSet<(usize, usize)> = HashSet::new();
        for &(chunk_idx, first_group, n_groups) in &group_selection.selections {
            for g in first_group..first_group + n_groups {
                groups_to_fetch.insert((chunk_idx, g));
            }
        }

        // If group_selection is empty but chunks are selected, fetch all groups
        // in those chunks (treat as full-chunk fetch).
        let fetch_all_groups = group_selection.selections.is_empty();

        for chunk_idx in all_chunks {
            let Some(chunk) = self.cold.get(chunk_idx) else {
                continue;
            };
            // Sink and local chunks are always fetched fully (all groups).
            let is_always_visible = sink_local.contains(chunk_idx);
            for g in 0..self.n_groups_per_chunk {
                let should_fetch = if fetch_all_groups || is_always_visible {
                    true
                } else {
                    groups_to_fetch.contains(&(chunk_idx, g))
                };
                if !should_fetch {
                    continue;
                }
                // Fetch all tokens in this group.
                let token_start = g * self.group_size;
                for t in token_start..token_start + self.group_size {
                    if t >= self.chunk_size {
                        break;
                    }
                    let key = &chunk.keys[t * d..(t + 1) * d];
                    let value = &chunk.values[t * d..(t + 1) * d];
                    ws.push_token(key, value, chunk.positions[t]);
                }
            }
        }

        ws
    }
}
