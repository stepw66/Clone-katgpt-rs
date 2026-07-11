//! MUX-Latent Wire Patch — latent-to-latent patching without decompress/recompress.
//!
//! A LatentPatch overwrites weights in a single LatentSegment, recomputes BLAKE3,
//! and sends the patch over the wire. The receiver injects directly into KV via
//! DomainLatent. No raw-token round-trip.

use crate::mux_latent::config::CompressionRatio;

/// A single latent patch — the wire-level unit.
/// Size: 4 + 32 + 32 = 68 bytes (fits in 1 cache line with padding).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LatentPatch {
    /// Which segment to patch (maps to octree morton code).
    pub segment_id: u32,
    /// New superposition weights (fixed-size for SIMD).
    /// X8 = 8 × f32 = 32 bytes.
    pub weights: [f32; 8],
    /// BLAKE3 commitment over weights.
    pub commitment: [u8; 32],
}

impl LatentPatch {
    /// Create a new patch with BLAKE3 commitment computed from weights.
    pub fn new(segment_id: u32, weights: [f32; 8]) -> Self {
        let commitment = Self::compute_commitment(segment_id, &weights);
        Self {
            segment_id,
            weights,
            commitment,
        }
    }

    /// Compute BLAKE3 commitment over segment_id + weights.
    pub fn compute_commitment(segment_id: u32, weights: &[f32; 8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&segment_id.to_le_bytes());
        for w in weights {
            hasher.update(&w.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    /// Verify the BLAKE3 commitment matches.
    pub fn verify_commitment(&self) -> bool {
        let expected = Self::compute_commitment(self.segment_id, &self.weights);
        expected == self.commitment
    }

    /// Validate all weights are finite (no NaN/Inf).
    pub fn weights_finite(&self) -> bool {
        self.weights.iter().all(|w| w.is_finite())
    }
}

/// Batch of patches — SIMD-friendly.
#[derive(Debug, Clone)]
pub struct LatentPatchBatch {
    pub patches: Vec<LatentPatch>,
    pub total_segments: u32,
    pub compression_ratio: CompressionRatio,
    pub tick: u64,
    /// Effective modulo for this batch (1 = full validation, 2+ = adaptive).
    /// Plan 244: set by AdaptiveModConfig::resolve() on game layer.
    /// Chain layer: always 1.
    pub validation_mod: usize,
}

/// Receipt for a batch of patches — committed and/or rejected segments.
#[derive(Debug, Clone)]
pub struct PatchReceipt {
    /// Segment IDs that were successfully committed.
    pub committed: Vec<u32>,
    /// Rejections with reasons.
    pub rejected: Vec<PatchRejection>,
}

impl PatchReceipt {
    /// Whether all patches were committed (no rejections).
    pub fn all_committed(&self) -> bool {
        self.rejected.is_empty()
    }
}

/// Reason a patch was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum PatchRejection {
    /// BLAKE3 commitment doesn't match (tampered or corrupt).
    CommitmentMismatch { segment_id: u32 },
    /// Segment contains NaN/Inf weights.
    NonFiniteWeights { segment_id: u32 },
    /// Segment is raw (uncompressed) — cannot patch latent weights.
    SegmentNotCompressible { segment_id: u32 },
    /// Segment ID doesn't exist in the context.
    OutOfRange { segment_id: u32 },
}

impl LatentPatchBatch {
    /// Create a new batch with default validation_mod=1.
    pub fn new(
        patches: Vec<LatentPatch>,
        total_segments: u32,
        compression_ratio: CompressionRatio,
        tick: u64,
    ) -> Self {
        Self {
            patches,
            total_segments,
            compression_ratio,
            tick,
            validation_mod: 1,
        }
    }

    /// Verify all patches' BLAKE3 commitments in batch using SIMD-friendly chunking.
    /// Processes patches in chunks of 4 for cache-friendly access.
    pub fn verify_all_commitments(&self) -> Result<PatchReceipt, PatchReceipt> {
        let mut committed = Vec::with_capacity(self.patches.len());
        let mut rejected = Vec::new();

        for chunk in self.patches.chunks(4) {
            for patch in chunk {
                let expected = LatentPatch::compute_commitment(patch.segment_id, &patch.weights);
                let commitment_ok = expected == patch.commitment;
                let finite_ok = patch.weights_finite();

                if commitment_ok && finite_ok {
                    committed.push(patch.segment_id);
                } else if !finite_ok {
                    rejected.push(PatchRejection::NonFiniteWeights {
                        segment_id: patch.segment_id,
                    });
                } else {
                    rejected.push(PatchRejection::CommitmentMismatch {
                        segment_id: patch.segment_id,
                    });
                }
            }
        }

        let receipt = PatchReceipt {
            committed,
            rejected,
        };
        if receipt.all_committed() {
            Ok(receipt)
        } else {
            Err(receipt)
        }
    }

    /// Chain-layer guard: assert validation_mod == 1.
    /// Panics if used with adaptive modulo (chain forbidden).
    pub fn assert_chain_safe(&self) {
        assert_eq!(
            self.validation_mod, 1,
            "Chain-bound patches MUST use validation_mod=1, got {}",
            self.validation_mod
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_patch(segment_id: u32) -> LatentPatch {
        let weights = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        LatentPatch::new(segment_id, weights)
    }

    #[test]
    fn test_patch_commitment_roundtrip() {
        let patch = make_valid_patch(42);
        assert!(patch.verify_commitment());
        assert!(patch.weights_finite());
    }

    #[test]
    fn test_patch_tamper_detection() {
        let mut patch = make_valid_patch(7);
        patch.commitment[0] ^= 0xFF;
        assert!(!patch.verify_commitment());
    }

    #[test]
    fn test_patch_weights_nan_rejection() {
        let weights = [f32::NAN, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut patch = LatentPatch::new(1, weights);
        // Recompute commitment so it matches the NaN weights
        patch.commitment = LatentPatch::compute_commitment(patch.segment_id, &patch.weights);
        assert!(!patch.weights_finite());
        // Even with valid commitment, NaN should cause batch rejection
        let batch = LatentPatchBatch::new(vec![patch], 1, CompressionRatio::X8, 0);
        let result = batch.verify_all_commitments();
        assert!(result.is_err());
        let receipt = result.unwrap_err();
        assert_eq!(receipt.rejected.len(), 1);
        assert!(matches!(
            receipt.rejected[0],
            PatchRejection::NonFiniteWeights { segment_id: 1 }
        ));
    }

    #[test]
    fn test_batch_verify_all_pass() {
        let patches: Vec<LatentPatch> = (0..12).map(make_valid_patch).collect();
        let batch = LatentPatchBatch::new(patches, 12, CompressionRatio::X8, 100);
        let result = batch.verify_all_commitments();
        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert_eq!(receipt.committed.len(), 12);
        assert!(receipt.rejected.is_empty());
    }

    #[test]
    fn test_batch_verify_some_fail() {
        let mut patches: Vec<LatentPatch> = (0..6).map(make_valid_patch).collect();
        // Tamper patches 2 and 4
        patches[2].commitment[0] ^= 0xFF;
        patches[4].commitment[15] ^= 0xAA;

        let batch = LatentPatchBatch::new(patches, 6, CompressionRatio::X8, 200);
        let result = batch.verify_all_commitments();
        assert!(result.is_err());
        let receipt = result.unwrap_err();
        assert_eq!(receipt.committed.len(), 4);
        assert_eq!(receipt.rejected.len(), 2);

        let rejected_ids: Vec<u32> = receipt
            .rejected
            .iter()
            .map(|r| match r {
                PatchRejection::CommitmentMismatch { segment_id } => *segment_id,
                _ => unreachable!(),
            })
            .collect();
        assert!(rejected_ids.contains(&2));
        assert!(rejected_ids.contains(&4));
    }

    #[test]
    fn test_chain_safe_guard() {
        let batch = LatentPatchBatch::new(vec![], 0, CompressionRatio::X8, 0);
        batch.assert_chain_safe(); // validation_mod=1, should pass

        let mut batch_adaptive = batch.clone();
        batch_adaptive.validation_mod = 3;
        let result = std::panic::catch_unwind(|| batch_adaptive.assert_chain_safe());
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_size() {
        assert_eq!(
            std::mem::size_of::<LatentPatch>(),
            68,
            "LatentPatch must be exactly 68 bytes: 4 (u32) + 32 ([f32; 8]) + 32 ([u8; 32])"
        );
    }
}
