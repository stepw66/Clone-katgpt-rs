//! Iterative chunk-based KV cache compaction.
//!
//! Processes KV cache in fixed-size chunks with a lookahead buffer,
//! enabling streaming compaction for very long sequences.

use half::f16;

/// A chunk of KV cache data for iterative processing.
#[derive(Debug, Clone)]
pub struct KVChunk {
    /// Key data for this chunk — flat f16, shape `[chunk_size * num_heads * head_dim]`.
    pub keys: Vec<f16>,
    /// Value data for this chunk — flat f16, shape `[chunk_size * num_heads * head_dim]`.
    pub values: Vec<f16>,
    /// Starting position of this chunk.
    pub start_pos: usize,
    /// Number of tokens in this chunk.
    pub len: usize,
}

impl KVChunk {
    /// Create a new empty chunk.
    pub fn new(start_pos: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            start_pos,
            len: 0,
        }
    }

    /// Returns true if the chunk has no tokens.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Iterative chunk-based KV cache compactor.
///
/// Processes the KV cache in fixed-size chunks, maintaining a lookahead buffer
/// for context-aware compaction decisions. This enables memory-bounded
/// compaction for arbitrarily long sequences.
#[derive(Debug, Clone)]
pub struct IterativeChunkCompactor {
    /// Number of tokens per processing chunk.
    pub chunk_size: usize,
    /// Number of lookahead tokens for context awareness.
    pub lookahead_buffer: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
}

impl IterativeChunkCompactor {
    /// Create a new iterative compactor.
    pub fn new(
        chunk_size: usize,
        lookahead_buffer: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            chunk_size,
            lookahead_buffer,
            num_heads,
            head_dim,
        }
    }

    /// Split a full KV cache into chunks for iterative processing.
    ///
    /// # Arguments
    /// * `keys` - Flat f16 key buffer
    /// * `values` - Flat f16 value buffer
    /// * `start_pos` - Starting position
    ///
    /// # Returns
    /// Iterator-friendly Vec of KVChunk.
    pub fn split_into_chunks(
        &self,
        keys: &[f16],
        values: &[f16],
        start_pos: usize,
    ) -> Vec<KVChunk> {
        let tokens_per_element = self.num_heads * self.head_dim;
        let total_tokens = match tokens_per_element {
            0 => return Vec::new(),
            t => keys.len() / t,
        };

        let mut chunks = Vec::new();
        let mut pos = start_pos;
        let mut offset = 0;

        while offset < total_tokens {
            let chunk_len = self.chunk_size.min(total_tokens - offset);
            let elem_start = offset * tokens_per_element;
            let elem_end = (offset + chunk_len) * tokens_per_element;

            chunks.push(KVChunk {
                keys: keys[elem_start..elem_end].to_vec(),
                values: values[elem_start..elem_end].to_vec(),
                start_pos: pos,
                len: chunk_len,
            });

            offset += chunk_len;
            pos += chunk_len;
        }

        chunks
    }

    /// Compact a single chunk with lookahead context.
    ///
    /// # Arguments
    /// * `chunk` - Current chunk to compact
    /// * `lookahead` - Optional lookahead chunk for context
    /// * `budget` - Target number of compact tokens
    ///
    /// # Returns
    /// Compacted chunk.
    pub fn compact_chunk(
        &self,
        chunk: &KVChunk,
        _lookahead: Option<&KVChunk>,
        budget: usize,
    ) -> KVChunk {
        // TODO: Implement per-chunk compaction using perceiver + query bank.
        // For now, return truncated chunk within budget.
        let tokens_per_element = self.num_heads * self.head_dim;
        let actual_budget = budget.min(chunk.len);
        let elem_end = actual_budget * tokens_per_element;

        KVChunk {
            keys: chunk.keys[..elem_end].to_vec(),
            values: chunk.values[..elem_end].to_vec(),
            start_pos: chunk.start_pos,
            len: actual_budget,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_chunk_new() {
        let chunk = KVChunk::new(10);
        assert!(chunk.is_empty());
        assert_eq!(chunk.start_pos, 10);
    }

    #[test]
    fn test_split_into_chunks() {
        let compactor = IterativeChunkCompactor::new(4, 2, 2, 4);
        // 8 tokens × 2 heads × 4 dim = 64 elements
        let keys = vec![f16::from_f32(1.0); 64];
        let values = vec![f16::from_f32(2.0); 64];
        let chunks = compactor.split_into_chunks(&keys, &values, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len, 4);
        assert_eq!(chunks[1].len, 4);
    }
}
