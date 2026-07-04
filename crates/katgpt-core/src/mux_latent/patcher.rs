//! Patch protocol for latent-to-latent context updates.
//!
//! Overwrites weights in a single LatentSegment, recomputes BLAKE3,
//! and returns the updated context. No decompress/recompress round-trip.

use crate::mux_latent::context::{CompressedContext, LatentSegment};
use crate::mux_latent::wire::{LatentPatch, PatchReceipt, PatchRejection};

/// Dirty tracking: which segment_ids changed since last flush.
#[derive(Debug, Clone, Default)]
pub struct DirtyTracker {
    dirty: std::collections::HashSet<u32>,
}

impl DirtyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark(&mut self, segment_id: u32) {
        self.dirty.insert(segment_id);
    }

    pub fn is_dirty(&self, segment_id: u32) -> bool {
        self.dirty.contains(&segment_id)
    }

    pub fn dirty_ids(&self) -> Vec<u32> {
        self.dirty.iter().copied().collect()
    }

    pub fn clear(&mut self) {
        self.dirty.clear();
    }

    pub fn len(&self) -> usize {
        self.dirty.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dirty.is_empty()
    }
}

/// Patches a CompressedContext in-place with latent weight updates.
pub struct LatentPatcher;

impl LatentPatcher {
    /// Patch a single segment in the context.
    /// Returns Ok(()) if the patch was applied, Err with rejection reason otherwise.
    /// The context is modified in-place if accepted.
    pub fn patch(
        context: &mut CompressedContext,
        patch: &LatentPatch,
    ) -> Result<(), PatchRejection> {
        if !patch.weights_finite() {
            return Err(PatchRejection::NonFiniteWeights {
                segment_id: patch.segment_id,
            });
        }

        let expected = LatentPatch::compute_commitment(patch.segment_id, &patch.weights);
        if expected != patch.commitment {
            return Err(PatchRejection::CommitmentMismatch {
                segment_id: patch.segment_id,
            });
        }

        let segment = context.segments.iter_mut().find(|s| {
            matches!(
                s,
                LatentSegment::Compressed { segment_id, .. } if *segment_id == patch.segment_id
            )
        });

        match segment {
            Some(LatentSegment::Compressed { weights, .. }) => {
                // Overwrite weights, truncating or padding to match span size
                let patch_weights: Vec<f32> =
                    patch.weights.iter().take(weights.len()).copied().collect();
                if patch_weights.len() < weights.len() {
                    let mut padded = patch_weights;
                    padded.extend_from_slice(&weights[padded.len()..]);
                    *weights = padded;
                } else {
                    *weights = patch_weights;
                }
                Ok(())
            }
            Some(LatentSegment::Raw { .. }) => Err(PatchRejection::SegmentNotCompressible {
                segment_id: patch.segment_id,
            }),
            None => Err(PatchRejection::OutOfRange {
                segment_id: patch.segment_id,
            }),
        }
    }

    /// Batch patch multiple segments using SIMD-friendly chunking.
    pub fn patch_batch(
        context: &mut CompressedContext,
        batch: &crate::mux_latent::wire::LatentPatchBatch,
    ) -> PatchReceipt {
        let mut committed = Vec::new();
        let mut rejected = Vec::new();

        for chunk in batch.patches.chunks(4) {
            for patch in chunk {
                match Self::patch(context, patch) {
                    Ok(()) => committed.push(patch.segment_id),
                    Err(r) => rejected.push(r),
                }
            }
        }

        PatchReceipt {
            committed,
            rejected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux_latent::config::{CompressionRatio, MuxLatentConfig};
    use crate::mux_latent::encoder::MuxLatentEncoder;

    fn make_context_x8() -> CompressedContext {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);
        // 32 tokens at X8 = 4 segments
        let tokens: Vec<u32> = (0..32).collect();
        encoder.encode(&tokens)
    }

    #[test]
    fn test_single_patch() {
        let mut ctx = make_context_x8();
        assert_eq!(ctx.latent_slot_count, 4);

        // Get original weights for segment 1
        let original_weights = match &ctx.segments[1] {
            LatentSegment::Compressed { weights, .. } => weights.clone(),
            _ => panic!("expected compressed segment"),
        };

        // Create patch with different weights
        let new_weights = [0.5f32; 8];
        let patch = LatentPatch::new(1, new_weights);
        let result = LatentPatcher::patch(&mut ctx, &patch);
        assert!(result.is_ok());

        // Verify weights changed
        match &ctx.segments[1] {
            LatentSegment::Compressed { weights, .. } => {
                assert_ne!(*weights, original_weights);
                assert_eq!(weights[0], 0.5);
            }
            _ => panic!("expected compressed segment"),
        }
    }

    #[test]
    fn test_patch_out_of_range() {
        let mut ctx = make_context_x8();
        let patch = LatentPatch::new(999, [0.1f32; 8]);
        let result = LatentPatcher::patch(&mut ctx, &patch);
        assert!(matches!(
            result,
            Err(PatchRejection::OutOfRange { segment_id: 999 })
        ));
    }

    #[test]
    fn test_patch_raw_segment() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: true,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);
        // Small input that produces raw segments (< 3 tokens per span)
        let tokens: Vec<u32> = (0..2).collect();
        let mut ctx = encoder.encode(&tokens);

        // Segment 0 is raw, not compressed — patching segment_id 0 should fail
        // (segment_id 0 only exists if there are compressed segments)
        // Use a segment_id that won't match any compressed segment
        let patch = LatentPatch::new(0, [0.1f32; 8]);
        let result = LatentPatcher::patch(&mut ctx, &patch);
        // If segment 0 is raw, we get SegmentNotCompressible
        // If segment 0 doesn't exist at all, we get OutOfRange
        match result {
            Err(PatchRejection::SegmentNotCompressible { .. })
            | Err(PatchRejection::OutOfRange { .. }) => {}
            other => panic!("expected rejection, got {:?}", other),
        }
    }

    #[test]
    fn test_batch_patch_3_segments() {
        let mut ctx = make_context_x8();
        assert_eq!(ctx.latent_slot_count, 4);

        let patches: Vec<LatentPatch> = (0..3)
            .map(|id| LatentPatch::new(id, [id as f32 * 0.1; 8]))
            .collect();
        let batch =
            crate::mux_latent::wire::LatentPatchBatch::new(patches, 4, CompressionRatio::X8, 0);
        let receipt = LatentPatcher::patch_batch(&mut ctx, &batch);

        assert_eq!(receipt.committed.len(), 3);
        assert!(receipt.rejected.is_empty());
        assert!(receipt.committed.contains(&0));
        assert!(receipt.committed.contains(&1));
        assert!(receipt.committed.contains(&2));
    }

    #[test]
    fn test_dirty_tracker() {
        let mut tracker = DirtyTracker::new();
        assert!(tracker.is_empty());

        tracker.mark(1);
        tracker.mark(5);
        tracker.mark(42);
        assert_eq!(tracker.len(), 3);
        assert!(tracker.is_dirty(1));
        assert!(tracker.is_dirty(5));
        assert!(tracker.is_dirty(42));
        assert!(!tracker.is_dirty(2));

        let mut ids = tracker.dirty_ids();
        ids.sort();
        assert_eq!(ids, vec![1, 5, 42]);

        tracker.clear();
        assert!(tracker.is_empty());
        assert!(!tracker.is_dirty(1));
    }
}
