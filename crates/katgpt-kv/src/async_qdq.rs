//! Async Q/DQ Overlap — double-buffered KV dequantize for GPU pipeline (Plan 227 Phase 6).
//!
//! Overlaps CPU dequantization of KV cache chunk N+1 with GPU processing of chunk N.
//! Requires `inference_router` feature (GPU backend).
//!
//! # Architecture
//!
//! ```text
//! Time →
//! CPU: [dequantize chunk 0] [dequantize chunk 1] [dequantize chunk 2] ...
//! GPU:                       [attention chunk 0]  [attention chunk 1]  ...
//! ```
//!
//! Double buffering: while GPU processes the current dequantized chunk,
//! CPU prepares the next chunk in a shadow buffer.

/// Double-buffered dequantize context.
///
/// Maintains two buffers: one "active" (being consumed by GPU) and one "shadow"
/// (being filled by CPU dequantize). Swaps on each chunk boundary.
pub struct DoubleBuffer {
    /// Active buffer (currently being consumed).
    pub active: Vec<f32>,
    /// Shadow buffer (being filled by next dequantize).
    pub shadow: Vec<f32>,
    /// Buffer size in f32 elements.
    pub buffer_size: usize,
    /// Whether the shadow buffer has valid data ready to swap.
    pub shadow_ready: bool,
}

impl DoubleBuffer {
    /// Create a new double buffer with the given capacity per buffer.
    pub fn new(buffer_size: usize) -> Self {
        Self {
            active: vec![0.0; buffer_size],
            shadow: vec![0.0; buffer_size],
            buffer_size,
            shadow_ready: false,
        }
    }

    /// Swap active and shadow buffers.
    /// Returns true if shadow was ready and swap occurred.
    pub fn swap(&mut self) -> bool {
        if !self.shadow_ready {
            return false;
        }
        std::mem::swap(&mut self.active, &mut self.shadow);
        self.shadow_ready = false;
        true
    }

    /// Get a mutable reference to the shadow buffer for filling.
    pub fn shadow_mut(&mut self) -> &mut [f32] {
        &mut self.shadow
    }

    /// Get a reference to the active buffer for consumption.
    pub fn active(&self) -> &[f32] {
        &self.active
    }

    /// Mark shadow as ready after filling.
    pub fn mark_shadow_ready(&mut self) {
        self.shadow_ready = true;
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.buffer_size * 2 * std::mem::size_of::<f32>()
    }
}

/// Chunked KV dequantize scheduler.
///
/// Breaks the KV cache into chunks and schedules double-buffered dequantize
/// operations that overlap with GPU attention computation.
pub struct AsyncQdqScheduler {
    /// Double buffer for key cache chunks.
    pub key_buffer: DoubleBuffer,
    /// Double buffer for value cache chunks.
    pub value_buffer: DoubleBuffer,
    /// Chunk size in tokens.
    pub chunk_size: usize,
    /// KV dimension.
    pub kv_dim: usize,
    /// Current chunk index being processed by GPU.
    pub current_chunk: usize,
    /// Total chunks.
    pub total_chunks: usize,
}

impl AsyncQdqScheduler {
    /// Create a new async Q/DQ scheduler.
    ///
    /// # Arguments
    /// * `kv_dim` - KV cache dimension
    /// * `seq_len` - Total sequence length
    /// * `chunk_size` - Tokens per chunk (default: 128)
    pub fn new(kv_dim: usize, seq_len: usize, chunk_size: usize) -> Self {
        let total_chunks = seq_len.div_ceil(chunk_size);
        let chunk_buffer_size = chunk_size * kv_dim;

        Self {
            key_buffer: DoubleBuffer::new(chunk_buffer_size),
            value_buffer: DoubleBuffer::new(chunk_buffer_size),
            chunk_size,
            kv_dim,
            current_chunk: 0,
            total_chunks,
        }
    }

    /// Dequantize the next chunk into the shadow buffer.
    /// Call this while GPU is processing the current chunk.
    ///
    /// Returns the chunk index that was dequantized, or None if all done.
    pub fn prefetch_next<F>(&mut self, dequant_fn: F) -> Option<usize>
    where
        F: FnOnce(usize, &mut [f32]),
    {
        let next_chunk = self.current_chunk + 1;
        if next_chunk >= self.total_chunks {
            return None;
        }

        // Fill shadow buffers
        dequant_fn(next_chunk, self.key_buffer.shadow_mut());
        self.key_buffer.mark_shadow_ready();

        Some(next_chunk)
    }

    /// Advance to the next chunk: swap buffers.
    /// Returns true if advance was successful (shadow was ready).
    pub fn advance(&mut self) -> bool {
        let swapped = self.key_buffer.swap();
        if swapped {
            self.current_chunk += 1;
        }
        self.value_buffer.swap();
        swapped
    }

    /// Reset scheduler for a new sequence.
    pub fn reset(&mut self) {
        self.current_chunk = 0;
        self.key_buffer.shadow_ready = false;
        self.value_buffer.shadow_ready = false;
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.key_buffer.memory_bytes() + self.value_buffer.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_buffer_swap() {
        let mut db = DoubleBuffer::new(4);

        // Fill shadow
        db.shadow_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        db.mark_shadow_ready();

        // Swap
        assert!(db.swap());
        assert_eq!(db.active(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(!db.shadow_ready);
    }

    #[test]
    fn test_double_buffer_no_swap_when_not_ready() {
        let mut db = DoubleBuffer::new(4);
        assert!(!db.swap());
    }

    #[test]
    fn test_double_buffer_double_swap() {
        let mut db = DoubleBuffer::new(4);

        // First fill
        db.shadow_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        db.mark_shadow_ready();
        assert!(db.swap());
        assert_eq!(db.active(), &[1.0, 2.0, 3.0, 4.0]);

        // Second fill
        db.shadow_mut().copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        db.mark_shadow_ready();
        assert!(db.swap());
        assert_eq!(db.active(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_scheduler_prefetch() {
        let mut scheduler = AsyncQdqScheduler::new(64, 256, 128);

        // Prefetch chunk 1 while processing chunk 0
        let result = scheduler.prefetch_next(|chunk_idx, buf| {
            assert_eq!(chunk_idx, 1);
            for v in buf.iter_mut() {
                *v = chunk_idx as f32;
            }
        });

        assert_eq!(result, Some(1));
        assert!(scheduler.key_buffer.shadow_ready);
    }

    #[test]
    fn test_scheduler_no_prefetch_past_end() {
        let mut scheduler = AsyncQdqScheduler::new(64, 128, 128);
        scheduler.current_chunk = 0;

        // Only 1 chunk total, so next chunk doesn't exist
        let result = scheduler.prefetch_next(|_, _| {});
        assert_eq!(result, None);
    }

    #[test]
    fn test_scheduler_advance() {
        let mut scheduler = AsyncQdqScheduler::new(64, 256, 128);

        // Fill shadow and advance
        scheduler
            .key_buffer
            .shadow_mut()
            .copy_from_slice(&[42.0; 64 * 128]);
        scheduler.key_buffer.mark_shadow_ready();

        assert!(scheduler.advance());
        assert_eq!(scheduler.current_chunk, 1);
    }

    #[test]
    fn test_scheduler_reset() {
        let mut scheduler = AsyncQdqScheduler::new(64, 256, 128);
        scheduler.current_chunk = 5;
        scheduler.key_buffer.mark_shadow_ready();

        scheduler.reset();
        assert_eq!(scheduler.current_chunk, 0);
        assert!(!scheduler.key_buffer.shadow_ready);
    }

    #[test]
    fn test_memory_usage() {
        let scheduler = AsyncQdqScheduler::new(128, 512, 128);
        let expected = 2 * 128 * 128 * std::mem::size_of::<f32>() * 2; // key + value, 2 buffers each
        assert_eq!(scheduler.memory_bytes(), expected);
    }
}
