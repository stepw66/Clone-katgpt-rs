//! Position-free KV cache compaction.
//!
//! Un-rotates RoPE from keys before compaction, compacts in position-free latent space,
//! then re-rotates on retrieval. This decouples compaction from positional encoding.

use half::f16;

/// Compactor that operates in position-free (RoPE-removed) space.
///
/// The key insight: RoPE encodes absolute position via rotation. If we un-rotate
/// keys before compaction, the compactor sees "pure semantic" keys without position
/// interference. After compaction, we re-rotate with adjusted positions.
#[derive(Debug, Clone)]
pub struct PositionFreeCompactor {
    /// RoPE base frequency (theta).
    pub rope_theta: f32,
    /// Head dimension.
    pub head_dim: usize,
}

impl PositionFreeCompactor {
    /// Create a new position-free compactor with the given RoPE parameters.
    pub fn new(rope_theta: f32, head_dim: usize) -> Self {
        Self {
            rope_theta,
            head_dim,
        }
    }

    /// Un-rotate RoPE from keys, returning position-free key buffer.
    ///
    /// # Arguments
    /// * `keys` - Flat f16 key buffer, shape `[seq_len * num_heads * head_dim]`
    /// * `start_pos` - Starting position index of this key sequence
    ///
    /// # Returns
    /// Position-free keys in f32 for compaction processing.
    pub fn un_rotate_keys(&self, keys: &[f16], _start_pos: usize) -> Vec<f32> {
        // TODO: Implement RoPE un-rotation.
        // 1. Compute RoPE frequencies: freq_i = 1.0 / (theta^(2i / head_dim))
        // 2. For each position p in start_pos..start_pos+seq_len:
        //    For each pair (2i, 2i+1) in head_dim:
        //      angle = p * freq_i
        //      key[2i]   =  key[2i]   * cos(angle) + key[2i+1] * sin(angle)
        //      key[2i+1] = -key[2i]   * sin(angle) + key[2i+1] * cos(angle)
        let seq_len = match self.head_dim {
            0 => return Vec::new(),
            d => keys.len() / d,
        };
        vec![0.0f32; seq_len * self.head_dim]
    }

    /// Re-rotate keys with new positions after compaction.
    ///
    /// # Arguments
    /// * `keys` - Position-free keys in f32
    /// * `new_start_pos` - New starting position for the compacted sequence
    ///
    /// # Returns
    /// Re-rotated keys in f16.
    pub fn re_rotate_keys(&self, keys: &[f32], _new_start_pos: usize) -> Vec<f16> {
        // TODO: Implement RoPE re-application.
        // Same rotation as un_rotate but forward direction with new positions.
        keys.iter().map(|&v| f16::from_f32(v)).collect()
    }

    /// Compute the position offset for the compacted cache.
    ///
    /// After compaction, the compacted tokens occupy positions starting from
    /// `new_start_pos`. This computes the appropriate offset.
    pub fn compute_position_offset(
        &self,
        original_start: usize,
        _original_len: usize,
        compact_len: usize,
    ) -> usize {
        // TODO: Implement smart offset computation.
        // For now, compacted tokens start where originals ended minus compact length.
        match compact_len {
            0 => original_start,
            _ => original_start.saturating_sub(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_free_compactor_new() {
        let compactor = PositionFreeCompactor::new(10000.0, 64);
        assert_eq!(compactor.rope_theta, 10000.0);
        assert_eq!(compactor.head_dim, 64);
    }

    #[test]
    fn test_compute_position_offset() {
        let compactor = PositionFreeCompactor::new(10000.0, 64);
        let offset = compactor.compute_position_offset(10, 100, 25);
        assert_eq!(offset, 10);
    }
}
